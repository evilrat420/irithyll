//! Single boosting step: owns one tree + drift detector + optional alternate.
//!
//! [`BoostingStep`] is a thin wrapper around [`TreeSlot`]
//! that adds SGBT variant logic. The three variants from Gunasekara et al. (2024) are:
//!
//! - **Standard** (`train_count = 1`): each sample trains the tree exactly once.
//! - **Skip** (`train_count = 0`): the sample is skipped (only prediction returned).
//! - **Multiple Iterations** (`train_count > 1`): the sample trains the tree
//!   multiple times, weighted by the hessian.
//!
//! The variant logic is computed externally (by `SGBTVariant::train_count()` in the
//! ensemble orchestrator) and passed in as `train_count`. This keeps `BoostingStep`
//! focused on execution rather than policy.

use std::fmt;

use crate::drift::DriftDetector;
use crate::ensemble::replacement::TreeSlot;
use crate::tree::builder::TreeConfig;

/// A single step in the SGBT boosting sequence.
///
/// Owns a [`TreeSlot`] and applies variant-aware training repetition. The number
/// of training iterations per sample is determined by the caller (the ensemble
/// orchestrator computes `train_count` from the configured SGBT variant).
///
/// # Prediction semantics
///
/// Both [`train_and_predict`](BoostingStep::train_and_predict) and
/// [`predict`](BoostingStep::predict) return the active tree's prediction
/// **before** any training on the current sample. This ensures unbiased
/// gradient computation in the boosting loop.
#[derive(Clone)]
pub struct BoostingStep {
    /// The tree slot managing the active tree, alternate, and drift detector.
    slot: TreeSlot,
}

impl fmt::Debug for BoostingStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoostingStep")
            .field("slot", &self.slot)
            .finish()
    }
}

impl BoostingStep {
    /// Create a new boosting step with a fresh tree and drift detector.
    pub fn new(tree_config: TreeConfig, detector: Box<dyn DriftDetector>) -> Self {
        Self {
            slot: TreeSlot::new(tree_config, detector, None),
        }
    }

    /// Create a new boosting step with optional time-based tree replacement.
    pub fn new_with_max_samples(
        tree_config: TreeConfig,
        detector: Box<dyn DriftDetector>,
        max_tree_samples: Option<u64>,
    ) -> Self {
        Self {
            slot: TreeSlot::new(tree_config, detector, max_tree_samples),
        }
    }

    /// Create a new boosting step with graduated tree handoff.
    pub fn new_with_graduated(
        tree_config: TreeConfig,
        detector: Box<dyn DriftDetector>,
        max_tree_samples: Option<u64>,
        shadow_warmup: usize,
    ) -> Self {
        Self {
            slot: TreeSlot::with_shadow_warmup(
                tree_config,
                detector,
                max_tree_samples,
                shadow_warmup,
            ),
        }
    }

    /// Reconstruct a boosting step from a pre-built tree slot.
    ///
    /// Used during model deserialization.
    pub fn from_slot(slot: TreeSlot) -> Self {
        Self { slot }
    }

    /// Train on a single sample with variant-aware repetition.
    ///
    /// # Arguments
    ///
    /// * `features` - Input feature vector.
    /// * `gradient` - Negative gradient of the loss at this sample.
    /// * `hessian` - Second derivative (curvature) of the loss at this sample.
    /// * `train_count` - Number of training iterations for this sample:
    ///   - `0`: skip training entirely (SK variant or stochastic skip).
    ///   - `1`: standard single-pass training.
    ///   - `>1`: multiple iterations (MI variant).
    ///
    /// # Returns
    ///
    /// The prediction from the active tree **before** training.
    pub fn train_and_predict(
        &mut self,
        features: &[f64],
        gradient: f64,
        hessian: f64,
        train_count: usize,
    ) -> f64 {
        if train_count == 0 {
            // Skip variant: no training, just predict.
            return self.slot.predict(features);
        }

        // First iteration: train and get the pre-training prediction.
        let pred = self.slot.train_and_predict(features, gradient, hessian);

        // Additional iterations for MI variant.
        // Each subsequent call still feeds the same gradient/hessian to the
        // tree and drift detector, effectively weighting this sample more heavily.
        for _ in 1..train_count {
            self.slot.train_and_predict(features, gradient, hessian);
        }

        pred
    }

    /// Predict without training.
    ///
    /// Routes the feature vector through the active tree and returns the
    /// leaf value. Does not update any state.
    #[inline]
    pub fn predict(&self, features: &[f64]) -> f64 {
        self.slot.predict(features)
    }

    /// Predict with variance for confidence estimation.
    ///
    /// Returns `(leaf_value, variance)` where variance = 1 / (H_sum + lambda).
    #[inline]
    pub fn predict_with_variance(&self, features: &[f64]) -> (f64, f64) {
        self.slot.predict_with_variance(features)
    }

    /// Predict using sigmoid-blended soft routing for smooth interpolation.
    ///
    /// See [`crate::tree::hoeffding::HoeffdingTree::predict_smooth`] for details.
    #[inline]
    pub fn predict_smooth(&self, features: &[f64], bandwidth: f64) -> f64 {
        self.slot.predict_smooth(features, bandwidth)
    }

    /// Predict using per-feature auto-calibrated bandwidths.
    #[inline]
    pub fn predict_smooth_auto(&self, features: &[f64], bandwidths: &[f64]) -> f64 {
        self.slot.predict_smooth_auto(features, bandwidths)
    }

    /// Predict with parent-leaf linear interpolation.
    #[inline]
    pub fn predict_interpolated(&self, features: &[f64]) -> f64 {
        self.slot.predict_interpolated(features)
    }

    /// Predict with sibling-based interpolation for feature-continuous predictions.
    #[inline]
    pub fn predict_sibling_interpolated(&self, features: &[f64], bandwidths: &[f64]) -> f64 {
        self.slot.predict_sibling_interpolated(features, bandwidths)
    }

    /// Predict using per-node auto-bandwidth soft routing.
    #[inline]
    pub fn predict_soft_routed(&self, features: &[f64]) -> f64 {
        self.slot.predict_soft_routed(features)
    }

    /// Predict with graduated active-shadow blending.
    #[inline]
    pub fn predict_graduated(&self, features: &[f64]) -> f64 {
        self.slot.predict_graduated(features)
    }

    /// Predict with graduated blending + sibling interpolation.
    #[inline]
    pub fn predict_graduated_sibling_interpolated(
        &self,
        features: &[f64],
        bandwidths: &[f64],
    ) -> f64 {
        self.slot
            .predict_graduated_sibling_interpolated(features, bandwidths)
    }

    /// Number of leaves in the active tree.
    #[inline]
    pub fn n_leaves(&self) -> usize {
        self.slot.n_leaves()
    }

    /// Total samples the active tree has seen.
    #[inline]
    pub fn n_samples_seen(&self) -> u64 {
        self.slot.n_samples_seen()
    }

    /// Whether the slot has an alternate tree being trained.
    #[inline]
    pub fn has_alternate(&self) -> bool {
        self.slot.has_alternate()
    }

    /// Reset to a completely fresh state: new tree, no alternate, reset detector.
    pub fn reset(&mut self) {
        self.slot.reset();
    }

    /// Immutable access to the underlying [`TreeSlot`].
    #[inline]
    pub fn slot(&self) -> &TreeSlot {
        &self.slot
    }

    /// Mutable access to the underlying [`TreeSlot`].
    #[inline]
    pub fn slot_mut(&mut self) -> &mut TreeSlot {
        &mut self.slot
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drift::pht::PageHinkleyTest;

    /// Create a default TreeConfig for tests.
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
    // Test 1: train_count=0 skips training (just predicts).
    // -------------------------------------------------------------------
    #[test]
    fn train_count_zero_skips_training() {
        let mut step = BoostingStep::new(test_tree_config(), test_detector());
        let features = [1.0, 2.0, 3.0];

        // Train with count=0 should not actually train.
        let pred = step.train_and_predict(&features, -0.5, 1.0, 0);
        assert!(
            pred.abs() < 1e-12,
            "train_count=0 should return fresh prediction (~0.0), got {}",
            pred,
        );

        // Verify no samples were actually trained.
        assert_eq!(
            step.n_samples_seen(),
            0,
            "train_count=0 should not increment samples_seen",
        );
    }

    // -------------------------------------------------------------------
    // Test 2: train_count=1 trains once.
    // -------------------------------------------------------------------
    #[test]
    fn train_count_one_trains_once() {
        let mut step = BoostingStep::new(test_tree_config(), test_detector());
        let features = [1.0, 2.0, 3.0];

        let pred = step.train_and_predict(&features, -0.5, 1.0, 1);
        assert!(
            pred.abs() < 1e-12,
            "first prediction should be ~0.0, got {}",
            pred,
        );

        // After one training call, the tree should have seen 1 sample.
        assert_eq!(
            step.n_samples_seen(),
            1,
            "train_count=1 should train exactly once",
        );

        // Second call should return non-zero (tree has been trained).
        let pred2 = step.predict(&features);
        assert!(
            pred2.is_finite(),
            "prediction after training should be finite",
        );
    }

    // -------------------------------------------------------------------
    // Test 3: train_count=3 trains multiple times.
    // -------------------------------------------------------------------
    #[test]
    fn train_count_three_trains_multiple_times() {
        let mut step = BoostingStep::new(test_tree_config(), test_detector());
        let features = [1.0, 2.0, 3.0];

        let pred = step.train_and_predict(&features, -0.5, 1.0, 3);
        assert!(
            pred.abs() < 1e-12,
            "first prediction should be ~0.0, got {}",
            pred,
        );

        // After train_count=3, the tree should have seen 3 samples.
        assert_eq!(
            step.n_samples_seen(),
            3,
            "train_count=3 should train exactly 3 times",
        );
    }

    // -------------------------------------------------------------------
    // Test 4: Reset works.
    // -------------------------------------------------------------------
    #[test]
    fn reset_clears_state() {
        let mut step = BoostingStep::new(test_tree_config(), test_detector());
        let features = [1.0, 2.0, 3.0];

        // Train several samples.
        for _ in 0..50 {
            step.train_and_predict(&features, -0.5, 1.0, 1);
        }

        assert!(step.n_samples_seen() > 0, "should have trained samples");

        step.reset();

        assert_eq!(step.n_leaves(), 1, "after reset, should have 1 leaf");
        assert_eq!(
            step.n_samples_seen(),
            0,
            "after reset, samples_seen should be 0"
        );
        assert!(
            !step.has_alternate(),
            "after reset, no alternate should exist"
        );

        let pred = step.predict(&features);
        assert!(
            pred.abs() < 1e-12,
            "prediction after reset should be ~0.0, got {}",
            pred,
        );
    }

    // -------------------------------------------------------------------
    // Test 5: Predict-only (no training) works on fresh step.
    // -------------------------------------------------------------------
    #[test]
    fn predict_only_on_fresh_step() {
        let step = BoostingStep::new(test_tree_config(), test_detector());

        for i in 0..10 {
            let x = (i as f64) * 0.5;
            let pred = step.predict(&[x, x + 1.0, x + 2.0]);
            assert!(
                pred.abs() < 1e-12,
                "untrained step should predict ~0.0, got {} at i={}",
                pred,
                i,
            );
        }
    }

    // -------------------------------------------------------------------
    // Test 6: Multiple calls with different train_counts produce expected
    //         cumulative sample counts.
    // -------------------------------------------------------------------
    #[test]
    fn mixed_train_counts_accumulate_correctly() {
        let mut step = BoostingStep::new(test_tree_config(), test_detector());
        let features = [1.0, 2.0, 3.0];

        // count=2 -> 2 samples
        step.train_and_predict(&features, -0.1, 1.0, 2);
        assert_eq!(step.n_samples_seen(), 2);

        // count=0 -> still 2 samples (skipped)
        step.train_and_predict(&features, -0.1, 1.0, 0);
        assert_eq!(step.n_samples_seen(), 2);

        // count=1 -> 3 samples
        step.train_and_predict(&features, -0.1, 1.0, 1);
        assert_eq!(step.n_samples_seen(), 3);

        // count=5 -> 8 samples
        step.train_and_predict(&features, -0.1, 1.0, 5);
        assert_eq!(step.n_samples_seen(), 8);
    }

    // -------------------------------------------------------------------
    // Test 7: n_leaves and has_alternate passthrough to slot.
    // -------------------------------------------------------------------
    #[test]
    fn accessors_match_slot() {
        let step = BoostingStep::new(test_tree_config(), test_detector());

        assert_eq!(step.n_leaves(), step.slot().n_leaves());
        assert_eq!(step.has_alternate(), step.slot().has_alternate());
        assert_eq!(step.n_samples_seen(), step.slot().n_samples_seen());
    }

    // -------------------------------------------------------------------
    // Test 8: Debug formatting works.
    // -------------------------------------------------------------------
    #[test]
    fn debug_format_does_not_panic() {
        let step = BoostingStep::new(test_tree_config(), test_detector());
        let debug_str = format!("{:?}", step);
        assert!(
            debug_str.contains("BoostingStep"),
            "debug output should contain 'BoostingStep'",
        );
    }
}
