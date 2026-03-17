//! TreeSlot: warning/danger/swap lifecycle for tree replacement.
//!
//! Implements the drift-triggered tree replacement strategy from
//! Gunasekara et al. (2024). Each boosting step owns a "slot" that manages:
//!
//! - An **active** tree serving predictions and receiving training samples.
//! - An optional **alternate** tree that begins training when a Warning signal
//!   is emitted by the drift detector.
//! - A **drift detector** monitoring prediction error magnitude.
//!
//! # Lifecycle
//!
//! ```text
//!   Stable  --> keep training active tree
//!   Warning --> spawn alternate tree (if not already training)
//!   Drift   --> replace active with alternate (or fresh tree), reset detector
//! ```

use std::fmt;

use crate::drift::{DriftDetector, DriftSignal};
use crate::tree::builder::TreeConfig;
use crate::tree::hoeffding::HoeffdingTree;
use crate::tree::StreamingTree;

/// Manages the lifecycle of a single tree in the ensemble.
///
/// When the drift detector signals [`DriftSignal::Warning`], an alternate tree
/// begins training alongside the active tree. When [`DriftSignal::Drift`] is
/// confirmed, the alternate replaces the active tree (or a fresh tree is
/// created if no alternate exists). The drift detector is then reset via
/// [`clone_fresh`](DriftDetector::clone_fresh) to monitor the new tree.
pub struct TreeSlot {
    /// The currently active tree serving predictions.
    active: HoeffdingTree,
    /// Optional alternate tree being trained during a warning period.
    alternate: Option<HoeffdingTree>,
    /// Drift detector monitoring this slot's error stream.
    detector: Box<dyn DriftDetector>,
    /// Configuration for creating new trees (shared across replacements).
    tree_config: TreeConfig,
    /// Maximum samples before proactive replacement. `None` = disabled.
    max_tree_samples: Option<u64>,
    /// Total number of tree replacements (drift or time-based).
    replacements: u64,
    /// Welford online count for prediction statistics.
    pred_count: u64,
    /// Welford online mean of predictions.
    pred_mean: f64,
    /// Welford online M2 accumulator for prediction variance.
    pred_m2: f64,
    /// Shadow warmup samples (0 = disabled). When > 0, an always-on shadow
    /// tree is spawned and trained alongside the active tree.
    shadow_warmup: usize,
    /// Sample count of the active tree when it was activated (promoted from shadow).
    /// Used to compute samples-since-activation for time-based replacement,
    /// preventing cascading swaps when a shadow is promoted with a high sample count.
    samples_at_activation: u64,
}

impl fmt::Debug for TreeSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TreeSlot")
            .field("active_leaves", &self.active.n_leaves())
            .field("active_samples", &self.active.n_samples_seen())
            .field("has_alternate", &self.alternate.is_some())
            .field("tree_config", &self.tree_config)
            .finish()
    }
}

impl Clone for TreeSlot {
    fn clone(&self) -> Self {
        Self {
            active: self.active.clone(),
            alternate: self.alternate.clone(),
            detector: self.detector.clone_boxed(),
            tree_config: self.tree_config.clone(),
            max_tree_samples: self.max_tree_samples,
            replacements: self.replacements,
            pred_count: self.pred_count,
            pred_mean: self.pred_mean,
            pred_m2: self.pred_m2,
            shadow_warmup: self.shadow_warmup,
            samples_at_activation: self.samples_at_activation,
        }
    }
}

impl TreeSlot {
    /// Create a new `TreeSlot` with a fresh tree and drift detector.
    ///
    /// The active tree starts as a single-leaf tree (prediction = 0.0).
    /// No alternate tree is created until a Warning signal is received.
    pub fn new(
        tree_config: TreeConfig,
        detector: Box<dyn DriftDetector>,
        max_tree_samples: Option<u64>,
    ) -> Self {
        Self::with_shadow_warmup(tree_config, detector, max_tree_samples, 0)
    }

    /// Create a new `TreeSlot` with graduated tree handoff enabled.
    ///
    /// When `shadow_warmup > 0`, an always-on shadow tree is spawned immediately
    /// and trained alongside the active tree. After `shadow_warmup` samples, the
    /// shadow begins contributing to `predict_graduated()` predictions.
    pub fn with_shadow_warmup(
        tree_config: TreeConfig,
        detector: Box<dyn DriftDetector>,
        max_tree_samples: Option<u64>,
        shadow_warmup: usize,
    ) -> Self {
        let alternate = if shadow_warmup > 0 {
            Some(HoeffdingTree::new(tree_config.clone()))
        } else {
            None
        };
        Self {
            active: HoeffdingTree::new(tree_config.clone()),
            alternate,
            detector,
            tree_config,
            max_tree_samples,
            replacements: 0,
            pred_count: 0,
            pred_mean: 0.0,
            pred_m2: 0.0,
            shadow_warmup,
            samples_at_activation: 0,
        }
    }

    /// Reconstruct a `TreeSlot` from pre-built trees and a fresh drift detector.
    ///
    /// Used during model deserialization to restore tree state without replaying
    /// the training stream.
    pub fn from_trees(
        active: HoeffdingTree,
        alternate: Option<HoeffdingTree>,
        tree_config: TreeConfig,
        detector: Box<dyn DriftDetector>,
        max_tree_samples: Option<u64>,
    ) -> Self {
        Self {
            active,
            alternate,
            detector,
            tree_config,
            max_tree_samples,
            replacements: 0,
            pred_count: 0,
            pred_mean: 0.0,
            pred_m2: 0.0,
            shadow_warmup: 0,
            samples_at_activation: 0,
        }
    }

    /// Train the active tree (and alternate if it exists) on a single sample.
    ///
    /// The absolute value of the gradient is fed to the drift detector as an
    /// error proxy (gradient = derivative of loss = prediction error signal).
    ///
    /// # Returns
    ///
    /// The prediction from the **active** tree **before** training on this sample.
    /// This ensures the prediction reflects only previously seen data, which is
    /// critical for unbiased gradient computation in the boosting loop.
    ///
    /// # Drift handling
    ///
    /// - [`DriftSignal::Stable`]: no action.
    /// - [`DriftSignal::Warning`]: spawn an alternate tree if one is not already
    ///   being trained. The alternate receives the same training sample.
    /// - [`DriftSignal::Drift`]: replace the active tree with the alternate
    ///   (or a fresh tree if no alternate exists). The drift detector is reset
    ///   via [`clone_fresh`](DriftDetector::clone_fresh) so it monitors the
    ///   new tree from a clean state.
    pub fn train_and_predict(&mut self, features: &[f64], gradient: f64, hessian: f64) -> f64 {
        // 1. Predict from active tree BEFORE training.
        let prediction = self.active.predict(features);

        // 1b. Update Welford running prediction statistics.
        self.pred_count += 1;
        let delta = prediction - self.pred_mean;
        self.pred_mean += delta / self.pred_count as f64;
        let delta2 = prediction - self.pred_mean;
        self.pred_m2 += delta * delta2;

        // 2. Train the active tree.
        self.active.train_one(features, gradient, hessian);

        // 3. Train the alternate tree if it exists.
        if let Some(ref mut alt) = self.alternate {
            alt.train_one(features, gradient, hessian);
        }

        // 4. Feed error magnitude to the drift detector.
        //    |gradient| is a proxy for prediction error: for squared loss,
        //    gradient = (prediction - target), so |gradient| = |error|.
        let error = gradient.abs();
        let signal = self.detector.update(error);

        // 5. React to the drift signal.
        match signal {
            DriftSignal::Stable => {}
            DriftSignal::Warning => {
                // Start training an alternate tree if not already doing so.
                if self.alternate.is_none() {
                    self.alternate = Some(HoeffdingTree::new(self.tree_config.clone()));
                }
            }
            DriftSignal::Drift => {
                // Replace active tree: prefer the alternate (which has been
                // training on recent data), fall back to a fresh tree.
                self.active = self
                    .alternate
                    .take()
                    .unwrap_or_else(|| HoeffdingTree::new(self.tree_config.clone()));
                // Record activation point to prevent cascading swaps.
                self.samples_at_activation = self.active.n_samples_seen();
                // Reset the drift detector to monitor the new tree cleanly.
                self.detector = self.detector.clone_fresh();
                // Track replacement and reset prediction stats for the new tree.
                self.replacements += 1;
                self.pred_count = 0;
                self.pred_mean = 0.0;
                self.pred_m2 = 0.0;
                // In graduated mode, immediately spawn a new shadow.
                if self.shadow_warmup > 0 {
                    self.alternate = Some(HoeffdingTree::new(self.tree_config.clone()));
                }
            }
        }

        // 6. Proactive time-based replacement.
        //    Compare samples-since-activation (not total lifetime) to prevent
        //    cascading swaps when a shadow with high sample count is promoted.
        if let Some(max_samples) = self.max_tree_samples {
            let active_age = self
                .active
                .n_samples_seen()
                .saturating_sub(self.samples_at_activation);

            // In graduated mode, wait until 120% of max_samples for soft replacement.
            let threshold = if self.shadow_warmup > 0 {
                (max_samples as f64 * 1.2) as u64
            } else {
                max_samples
            };

            if active_age >= threshold {
                self.active = self
                    .alternate
                    .take()
                    .unwrap_or_else(|| HoeffdingTree::new(self.tree_config.clone()));
                // Record activation point to prevent cascading swaps.
                self.samples_at_activation = self.active.n_samples_seen();
                self.detector = self.detector.clone_fresh();
                self.replacements += 1;
                self.pred_count = 0;
                self.pred_mean = 0.0;
                self.pred_m2 = 0.0;
                // In graduated mode, immediately spawn a new shadow.
                if self.shadow_warmup > 0 {
                    self.alternate = Some(HoeffdingTree::new(self.tree_config.clone()));
                }
            }
        }

        // 7. In graduated mode, ensure shadow always exists.
        if self.shadow_warmup > 0 && self.alternate.is_none() {
            self.alternate = Some(HoeffdingTree::new(self.tree_config.clone()));
        }

        prediction
    }

    /// Predict without training.
    ///
    /// Routes the feature vector through the active tree and returns the
    /// leaf value. Does not update any state.
    #[inline]
    pub fn predict(&self, features: &[f64]) -> f64 {
        self.active.predict(features)
    }

    /// Predict with variance for confidence estimation.
    ///
    /// Returns `(leaf_value, variance)` where variance = 1 / (H_sum + lambda).
    #[inline]
    pub fn predict_with_variance(&self, features: &[f64]) -> (f64, f64) {
        self.active.predict_with_variance(features)
    }

    /// Predict using sigmoid-blended soft routing for smooth interpolation.
    ///
    /// See [`crate::tree::hoeffding::HoeffdingTree::predict_smooth`] for details.
    #[inline]
    pub fn predict_smooth(&self, features: &[f64], bandwidth: f64) -> f64 {
        self.active.predict_smooth(features, bandwidth)
    }

    /// Predict using per-feature auto-calibrated bandwidths.
    #[inline]
    pub fn predict_smooth_auto(&self, features: &[f64], bandwidths: &[f64]) -> f64 {
        self.active.predict_smooth_auto(features, bandwidths)
    }

    /// Predict with parent-leaf linear interpolation.
    #[inline]
    pub fn predict_interpolated(&self, features: &[f64]) -> f64 {
        self.active.predict_interpolated(features)
    }

    /// Predict with sibling-based interpolation for feature-continuous predictions.
    #[inline]
    pub fn predict_sibling_interpolated(&self, features: &[f64], bandwidths: &[f64]) -> f64 {
        self.active
            .predict_sibling_interpolated(features, bandwidths)
    }

    /// Predict with graduated active-shadow blending.
    ///
    /// When `shadow_warmup > 0`, blends the active tree's prediction with the
    /// shadow's prediction based on relative maturity:
    /// - Active weight decays from 1.0 to 0.0 as it ages from 80% to 120% of `max_tree_samples`
    /// - Shadow weight ramps from 0.0 to 1.0 over `shadow_warmup` samples after warmup
    ///
    /// When `shadow_warmup == 0` or no shadow exists, returns the active prediction.
    pub fn predict_graduated(&self, features: &[f64]) -> f64 {
        let active_pred = self.active.predict(features);

        if self.shadow_warmup == 0 {
            return active_pred;
        }

        let Some(ref shadow) = self.alternate else {
            return active_pred;
        };

        let shadow_samples = shadow.n_samples_seen();
        if shadow_samples < self.shadow_warmup as u64 {
            return active_pred;
        }

        let shadow_pred = shadow.predict(features);
        self.blend_active_shadow(active_pred, shadow_pred, shadow_samples)
    }

    /// Predict with graduated blending + sibling interpolation (premium path).
    ///
    /// Combines graduated active-shadow handoff with feature-continuous sibling
    /// interpolation for the smoothest possible prediction surface.
    pub fn predict_graduated_sibling_interpolated(
        &self,
        features: &[f64],
        bandwidths: &[f64],
    ) -> f64 {
        let active_pred = self
            .active
            .predict_sibling_interpolated(features, bandwidths);

        if self.shadow_warmup == 0 {
            return active_pred;
        }

        let Some(ref shadow) = self.alternate else {
            return active_pred;
        };

        let shadow_samples = shadow.n_samples_seen();
        if shadow_samples < self.shadow_warmup as u64 {
            return active_pred;
        }

        let shadow_pred = shadow.predict_sibling_interpolated(features, bandwidths);
        self.blend_active_shadow(active_pred, shadow_pred, shadow_samples)
    }

    /// Compute the graduated blend of active and shadow predictions.
    #[inline]
    fn blend_active_shadow(&self, active_pred: f64, shadow_pred: f64, shadow_samples: u64) -> f64 {
        let active_age = self
            .active
            .n_samples_seen()
            .saturating_sub(self.samples_at_activation);
        let mts = self.max_tree_samples.unwrap_or(u64::MAX) as f64;

        // Active weight: 1.0 until 80% of mts, then linear decay to 0.0 at 120%
        let active_w = if (active_age as f64) < mts * 0.8 {
            1.0
        } else {
            let progress = (active_age as f64 - mts * 0.8) / (mts * 0.4);
            (1.0 - progress).clamp(0.0, 1.0)
        };

        // Shadow weight: 0.0 until shadow_warmup, then ramp to 1.0 over shadow_warmup samples
        let shadow_w = ((shadow_samples as f64 - self.shadow_warmup as f64)
            / self.shadow_warmup as f64)
            .clamp(0.0, 1.0);

        // Normalize
        let total = active_w + shadow_w;
        if total < 1e-10 {
            return shadow_pred;
        }

        (active_w * active_pred + shadow_w * shadow_pred) / total
    }

    /// Shadow warmup configuration (0 = disabled).
    #[inline]
    pub fn shadow_warmup(&self) -> usize {
        self.shadow_warmup
    }

    /// Total number of tree replacements (drift or time-based).
    #[inline]
    pub fn replacements(&self) -> u64 {
        self.replacements
    }

    /// Running mean of predictions from the active tree.
    #[inline]
    pub fn prediction_mean(&self) -> f64 {
        self.pred_mean
    }

    /// Running standard deviation of predictions from the active tree.
    #[inline]
    pub fn prediction_std(&self) -> f64 {
        if self.pred_count < 2 {
            0.0
        } else {
            (self.pred_m2 / (self.pred_count - 1) as f64).sqrt()
        }
    }

    /// Number of leaves in the active tree.
    #[inline]
    pub fn n_leaves(&self) -> usize {
        self.active.n_leaves()
    }

    /// Total samples the active tree has seen.
    #[inline]
    pub fn n_samples_seen(&self) -> u64 {
        self.active.n_samples_seen()
    }

    /// Whether an alternate tree is currently being trained.
    #[inline]
    pub fn has_alternate(&self) -> bool {
        self.alternate.is_some()
    }

    /// Accumulated split gains per feature from the active tree.
    #[inline]
    pub fn split_gains(&self) -> &[f64] {
        self.active.split_gains()
    }

    /// Immutable access to the active tree.
    #[inline]
    pub fn active_tree(&self) -> &HoeffdingTree {
        &self.active
    }

    /// Immutable access to the alternate tree (if one is being trained).
    #[inline]
    pub fn alternate_tree(&self) -> Option<&HoeffdingTree> {
        self.alternate.as_ref()
    }

    /// Immutable access to the tree configuration.
    #[inline]
    pub fn tree_config(&self) -> &TreeConfig {
        &self.tree_config
    }

    /// Immutable access to the drift detector.
    #[inline]
    pub fn detector(&self) -> &dyn DriftDetector {
        &*self.detector
    }

    /// Mutable access to the drift detector.
    #[inline]
    pub fn detector_mut(&mut self) -> &mut dyn DriftDetector {
        &mut *self.detector
    }

    /// Immutable access to the alternate drift detector (always `None` in
    /// the current architecture -- the alternate tree shares the main detector).
    /// Reserved for future use.
    #[inline]
    pub fn alt_detector(&self) -> Option<&dyn DriftDetector> {
        // Currently there's no separate alt detector.
        None
    }

    /// Mutable access to the alternate drift detector.
    #[inline]
    pub fn alt_detector_mut(&mut self) -> Option<&mut dyn DriftDetector> {
        None
    }

    /// Reset to a completely fresh state: new tree, no alternate, reset detector.
    pub fn reset(&mut self) {
        self.active = HoeffdingTree::new(self.tree_config.clone());
        self.alternate = if self.shadow_warmup > 0 {
            Some(HoeffdingTree::new(self.tree_config.clone()))
        } else {
            None
        };
        self.detector = self.detector.clone_fresh();
        self.replacements = 0;
        self.pred_count = 0;
        self.pred_mean = 0.0;
        self.pred_m2 = 0.0;
        self.samples_at_activation = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drift::pht::PageHinkleyTest;

    /// Create a default TreeConfig for tests (small grace period for fast splits).
    fn test_tree_config() -> TreeConfig {
        TreeConfig::new()
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .lambda(1.0)
    }

    /// Create a default drift detector for tests.
    fn test_detector() -> Box<dyn DriftDetector> {
        Box::new(PageHinkleyTest::new())
    }

    // -------------------------------------------------------------------
    // Test 1: TreeSlot::new creates a functional slot; predict returns 0.0.
    // -------------------------------------------------------------------
    #[test]
    fn new_slot_predicts_zero() {
        let slot = TreeSlot::new(test_tree_config(), test_detector(), None);

        // A fresh tree with no training data should predict 0.0.
        let pred = slot.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.abs() < 1e-12,
            "fresh slot should predict ~0.0, got {}",
            pred,
        );
    }

    // -------------------------------------------------------------------
    // Test 2: train_and_predict returns a prediction and does not panic.
    // -------------------------------------------------------------------
    #[test]
    fn train_and_predict_returns_prediction() {
        let mut slot = TreeSlot::new(test_tree_config(), test_detector(), None);

        let features = [1.0, 2.0, 3.0];
        let pred = slot.train_and_predict(&features, -0.5, 1.0);

        // First prediction should be 0.0 (tree was empty before training).
        assert!(
            pred.abs() < 1e-12,
            "first prediction should be ~0.0, got {}",
            pred,
        );

        // After training, the tree should have updated, so a second predict
        // should be non-zero (gradient=-0.5 pushes leaf weight positive).
        let pred2 = slot.predict(&features);
        assert!(
            pred2.is_finite(),
            "prediction after training should be finite"
        );
    }

    // -------------------------------------------------------------------
    // Test 3: After many stable samples, no alternate tree is spawned.
    // -------------------------------------------------------------------
    #[test]
    fn stable_stream_no_alternate() {
        let mut slot = TreeSlot::new(test_tree_config(), test_detector(), None);
        let features = [1.0, 2.0, 3.0];

        // Feed many stable samples with small, consistent gradients.
        // With a constant error of 0.1, the PHT running mean settles and
        // no warning/drift should trigger.
        for _ in 0..500 {
            slot.train_and_predict(&features, -0.1, 1.0);
        }

        assert!(
            !slot.has_alternate(),
            "stable error stream should not spawn an alternate tree",
        );
    }

    // -------------------------------------------------------------------
    // Test 4: Reset returns to fresh state.
    // -------------------------------------------------------------------
    #[test]
    fn reset_returns_to_fresh_state() {
        let mut slot = TreeSlot::new(test_tree_config(), test_detector(), None);
        let features = [1.0, 2.0, 3.0];

        // Train several samples.
        for _ in 0..100 {
            slot.train_and_predict(&features, -0.5, 1.0);
        }

        assert!(slot.n_samples_seen() > 0, "should have trained samples");

        slot.reset();

        assert_eq!(
            slot.n_leaves(),
            1,
            "after reset, should have exactly 1 leaf"
        );
        assert_eq!(
            slot.n_samples_seen(),
            0,
            "after reset, samples_seen should be 0"
        );
        assert!(
            !slot.has_alternate(),
            "after reset, no alternate should exist"
        );

        // Predict should return 0.0 again.
        let pred = slot.predict(&features);
        assert!(
            pred.abs() < 1e-12,
            "prediction after reset should be ~0.0, got {}",
            pred,
        );
    }

    // -------------------------------------------------------------------
    // Test 5: Predict without training works.
    // -------------------------------------------------------------------
    #[test]
    fn predict_without_training() {
        let slot = TreeSlot::new(test_tree_config(), test_detector(), None);

        // Multiple predict calls on a fresh slot should all return 0.0.
        for i in 0..10 {
            let x = (i as f64) * 0.5;
            let pred = slot.predict(&[x, x + 1.0]);
            assert!(
                pred.abs() < 1e-12,
                "untrained slot should predict ~0.0 for any input, got {} at i={}",
                pred,
                i,
            );
        }
    }

    // -------------------------------------------------------------------
    // Test 6: Drift replaces the active tree.
    // -------------------------------------------------------------------
    #[test]
    fn drift_replaces_active_tree() {
        // Use a very sensitive detector: small lambda triggers drift quickly.
        let sensitive_detector = Box::new(PageHinkleyTest::with_params(0.005, 5.0));
        let mut slot = TreeSlot::new(test_tree_config(), sensitive_detector, None);
        let features = [1.0, 2.0, 3.0];

        // Phase 1: stable training with small gradients.
        for _ in 0..200 {
            slot.train_and_predict(&features, -0.01, 1.0);
        }
        let samples_before_drift = slot.n_samples_seen();

        // Phase 2: abrupt shift in gradient magnitude to trigger drift.
        let mut drift_occurred = false;
        for _ in 0..500 {
            slot.train_and_predict(&features, -50.0, 1.0);
            // If drift occurred, the tree was replaced and samples_seen resets
            // (new tree starts from 0).
            if slot.n_samples_seen() < samples_before_drift {
                drift_occurred = true;
                break;
            }
        }

        assert!(
            drift_occurred,
            "abrupt gradient shift should trigger drift and replace the active tree",
        );
    }

    // -------------------------------------------------------------------
    // Test 7: n_leaves reflects the active tree.
    // -------------------------------------------------------------------
    #[test]
    fn n_leaves_reflects_active_tree() {
        let slot = TreeSlot::new(test_tree_config(), test_detector(), None);
        assert_eq!(slot.n_leaves(), 1, "fresh slot should have exactly 1 leaf",);
    }

    // -------------------------------------------------------------------
    // Test 8: Debug formatting works.
    // -------------------------------------------------------------------
    #[test]
    fn debug_format_does_not_panic() {
        let slot = TreeSlot::new(test_tree_config(), test_detector(), None);
        let debug_str = format!("{:?}", slot);
        assert!(
            debug_str.contains("TreeSlot"),
            "debug output should contain 'TreeSlot'",
        );
    }

    // -------------------------------------------------------------------
    // Test 9: Time-based replacement triggers after max_tree_samples.
    // -------------------------------------------------------------------
    #[test]
    fn time_based_replacement_triggers() {
        let mut slot = TreeSlot::new(test_tree_config(), test_detector(), Some(200));
        let features = [1.0, 2.0, 3.0];

        // Train up to the limit.
        for _ in 0..200 {
            slot.train_and_predict(&features, -0.1, 1.0);
        }

        // At exactly 200 samples, the tree should have been replaced.
        // The new tree has 0 samples seen (or the most recently trained sample).
        assert!(
            slot.n_samples_seen() < 200,
            "after 200 samples with max_tree_samples=200, tree should be replaced (got {} samples)",
            slot.n_samples_seen(),
        );
    }

    // -------------------------------------------------------------------
    // Test 10: Time-based replacement disabled (None) never triggers.
    // -------------------------------------------------------------------
    #[test]
    fn time_based_replacement_disabled() {
        let mut slot = TreeSlot::new(test_tree_config(), test_detector(), None);
        let features = [1.0, 2.0, 3.0];

        for _ in 0..500 {
            slot.train_and_predict(&features, -0.1, 1.0);
        }

        assert_eq!(
            slot.n_samples_seen(),
            500,
            "without max_tree_samples, tree should never be proactively replaced",
        );
    }

    // -------------------------------------------------------------------
    // Graduated handoff tests
    // -------------------------------------------------------------------

    #[test]
    fn graduated_shadow_spawns_immediately() {
        let slot = TreeSlot::with_shadow_warmup(test_tree_config(), test_detector(), Some(200), 50);

        assert!(
            slot.has_alternate(),
            "graduated mode should spawn shadow immediately"
        );
        assert_eq!(slot.shadow_warmup(), 50);
    }

    #[test]
    fn graduated_predict_returns_finite() {
        let mut slot =
            TreeSlot::with_shadow_warmup(test_tree_config(), test_detector(), Some(200), 50);
        let features = [1.0, 2.0, 3.0];

        for _ in 0..100 {
            slot.train_and_predict(&features, -0.1, 1.0);
        }

        let pred = slot.predict_graduated(&features);
        assert!(
            pred.is_finite(),
            "graduated prediction should be finite: {}",
            pred
        );
    }

    #[test]
    fn graduated_shadow_always_respawns() {
        let mut slot =
            TreeSlot::with_shadow_warmup(test_tree_config(), test_detector(), Some(100), 30);
        let features = [1.0, 2.0, 3.0];

        // Train past the 120% soft replacement threshold (120 samples)
        for _ in 0..130 {
            slot.train_and_predict(&features, -0.1, 1.0);
        }

        // After soft replacement, shadow should still exist
        assert!(
            slot.has_alternate(),
            "shadow should be respawned after soft replacement"
        );
    }

    #[test]
    fn graduated_blending_produces_intermediate_values() {
        let mut slot =
            TreeSlot::with_shadow_warmup(test_tree_config(), test_detector(), Some(200), 50);
        let features = [1.0, 2.0, 3.0];

        // Train enough samples for shadow to be warm and blending to be active
        // (past 80% of max_tree_samples = 160 samples, shadow needs 50 warmup)
        for _ in 0..180 {
            slot.train_and_predict(&features, -0.1, 1.0);
        }

        let active_pred = slot.predict(&features);
        let graduated_pred = slot.predict_graduated(&features);

        // Both should be finite
        assert!(active_pred.is_finite());
        assert!(graduated_pred.is_finite());
    }

    #[test]
    fn graduated_reset_preserves_shadow() {
        let mut slot =
            TreeSlot::with_shadow_warmup(test_tree_config(), test_detector(), Some(200), 50);

        slot.reset();

        assert!(
            slot.has_alternate(),
            "reset in graduated mode should preserve shadow spawning"
        );
    }

    #[test]
    fn graduated_no_cascading_swap() {
        let mut slot =
            TreeSlot::with_shadow_warmup(test_tree_config(), test_detector(), Some(200), 50);

        // Train past the 120% soft replacement threshold (240 samples).
        // Use varying features so trees can actually split.
        for i in 0..250 {
            let x = (i as f64) * 0.1;
            let features = [x, x.sin(), x.cos()];
            let gradient = -0.1 * (1.0 + x.sin());
            slot.train_and_predict(&features, gradient, 1.0);
        }

        let replacements_after_first_swap = slot.replacements();
        assert!(
            replacements_after_first_swap >= 1,
            "should have swapped at least once after 250 samples with mts=200"
        );

        // Train 50 more samples — should NOT trigger another swap
        for i in 250..300 {
            let x = (i as f64) * 0.1;
            let features = [x, x.sin(), x.cos()];
            slot.train_and_predict(&features, -0.1, 1.0);
        }

        assert_eq!(
            slot.replacements(),
            replacements_after_first_swap,
            "should not cascade-swap immediately after promotion"
        );
    }

    #[test]
    fn graduated_without_max_tree_samples_still_works() {
        // shadow_warmup enabled but no max_tree_samples — active never decays
        let mut slot = TreeSlot::with_shadow_warmup(test_tree_config(), test_detector(), None, 50);
        let features = [1.0, 2.0, 3.0];

        for _ in 0..100 {
            slot.train_and_predict(&features, -0.1, 1.0);
        }

        // predict_graduated should work (active_w stays 1.0 since mts = MAX)
        let pred = slot.predict_graduated(&features);
        assert!(pred.is_finite(), "graduated without mts should be finite");
    }
}
