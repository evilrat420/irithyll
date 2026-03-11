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
use crate::tree::hoeffding::HoeffdingTree;
use crate::tree::builder::TreeConfig;
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
        Self {
            active: HoeffdingTree::new(tree_config.clone()),
            alternate: None,
            detector,
            tree_config,
            max_tree_samples,
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
    pub fn train_and_predict(
        &mut self,
        features: &[f64],
        gradient: f64,
        hessian: f64,
    ) -> f64 {
        // 1. Predict from active tree BEFORE training.
        let prediction = self.active.predict(features);

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
                self.active = self.alternate.take().unwrap_or_else(|| {
                    HoeffdingTree::new(self.tree_config.clone())
                });
                // Always clear the alternate slot after replacement.
                self.alternate = None;
                // Reset the drift detector to monitor the new tree cleanly.
                self.detector = self.detector.clone_fresh();
            }
        }

        // 6. Proactive time-based replacement.
        //    If the active tree has processed too many samples, replace it
        //    regardless of drift signal. This prevents stale structure from
        //    accumulating in slowly-drifting streams.
        if let Some(max_samples) = self.max_tree_samples {
            if self.active.n_samples_seen() >= max_samples {
                self.active = self.alternate.take().unwrap_or_else(|| {
                    HoeffdingTree::new(self.tree_config.clone())
                });
                self.alternate = None;
                self.detector = self.detector.clone_fresh();
            }
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

    /// Reset to a completely fresh state: new tree, no alternate, reset detector.
    pub fn reset(&mut self) {
        self.active = HoeffdingTree::new(self.tree_config.clone());
        self.alternate = None;
        self.detector = self.detector.clone_fresh();
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
        assert!(pred2.is_finite(), "prediction after training should be finite");
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

        assert_eq!(slot.n_leaves(), 1, "after reset, should have exactly 1 leaf");
        assert_eq!(slot.n_samples_seen(), 0, "after reset, samples_seen should be 0");
        assert!(!slot.has_alternate(), "after reset, no alternate should exist");

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
        assert_eq!(
            slot.n_leaves(),
            1,
            "fresh slot should have exactly 1 leaf",
        );
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
}
