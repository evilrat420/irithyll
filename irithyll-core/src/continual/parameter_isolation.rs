//! Drift-triggered parameter isolation.
//!
//! When drift is detected, computes an importance mask from accumulated
//! gradient magnitudes and freezes the most important parameters (setting
//! their gradients to zero). The remaining parameters are free to adapt.
//!
//! This is a streaming version of PackNet (Mallya & Lazebnik, 2018):
//! instead of task boundaries, irithyll's drift detectors trigger mask updates.
//!
//! # How it works
//!
//! 1. Importance is tracked as an EWMA of `|gradient|` for each parameter.
//! 2. When a `DriftSignal::Drift` arrives, the top `freeze_fraction` of
//!    parameters (by importance) are frozen.
//! 3. Frozen parameters have their gradients zeroed in every subsequent
//!    `pre_update` call, effectively isolating them from further learning.
//! 4. Multiple drift events accumulate: once frozen, a parameter stays frozen
//!    until explicitly unfrozen or the strategy is reset.

use alloc::vec::Vec;

use super::ContinualStrategy;
use crate::drift::DriftSignal;
use crate::math;

/// Drift-triggered parameter isolation.
///
/// When drift is detected, computes an importance mask from the accumulated
/// gradient magnitudes and freezes the most important parameters (setting their
/// gradients to zero). The remaining parameters are free to adapt.
///
/// This is a streaming version of PackNet (Mallya & Lazebnik, 2018):
/// instead of task boundaries, irithyll's drift detectors trigger mask updates.
///
/// # Example
///
/// ```
/// use irithyll_core::continual::{DriftMask, ContinualStrategy};
/// use irithyll_core::drift::DriftSignal;
///
/// let mut mask = DriftMask::with_defaults(10);
///
/// // Train for a while, accumulating importance
/// for _ in 0..100 {
///     let params = vec![0.0; 10];
///     let mut grads = vec![0.1; 10];
///     mask.pre_update(&params, &mut grads);
/// }
///
/// // Drift detected: freeze top 30% of params
/// mask.on_drift(&[0.0; 10], DriftSignal::Drift);
/// assert_eq!(mask.n_frozen(), 3); // 30% of 10
/// ```
pub struct DriftMask {
    /// Accumulated importance per parameter (EWMA of |grad|).
    importance: Vec<f64>,
    /// Per-parameter frozen mask: `true` = frozen (gradient zeroed).
    frozen: Vec<bool>,
    /// Fraction of params to freeze on each drift event (default: 0.3).
    freeze_fraction: f64,
    /// EWMA decay for importance tracking (default: 0.99).
    importance_alpha: f64,
    /// Count of currently frozen parameters.
    n_frozen: usize,
}

impl DriftMask {
    /// Create a new `DriftMask` with explicit hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `n_params` -- number of parameters to manage
    /// * `freeze_fraction` -- fraction of *currently unfrozen* params to freeze on drift
    /// * `importance_alpha` -- EWMA decay for importance; closer to 1.0 = longer memory
    ///
    /// # Panics
    ///
    /// Panics if `freeze_fraction` is not in `[0.0, 1.0]` or `importance_alpha`
    /// is not in `[0.0, 1.0]`.
    pub fn new(n_params: usize, freeze_fraction: f64, importance_alpha: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&freeze_fraction),
            "freeze_fraction must be in [0.0, 1.0], got {freeze_fraction}"
        );
        assert!(
            (0.0..=1.0).contains(&importance_alpha),
            "importance_alpha must be in [0.0, 1.0], got {importance_alpha}"
        );
        Self {
            importance: alloc::vec![0.0; n_params],
            frozen: alloc::vec![false; n_params],
            freeze_fraction,
            importance_alpha,
            n_frozen: 0,
        }
    }

    /// Create a new `DriftMask` with default hyperparameters.
    ///
    /// Defaults: `freeze_fraction = 0.3`, `importance_alpha = 0.99`.
    pub fn with_defaults(n_params: usize) -> Self {
        Self::new(n_params, 0.3, 0.99)
    }

    /// Check if a specific parameter is frozen.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_params()`.
    #[inline]
    pub fn is_frozen(&self, idx: usize) -> bool {
        self.frozen[idx]
    }

    /// Number of currently frozen parameters.
    #[inline]
    pub fn n_frozen(&self) -> usize {
        self.n_frozen
    }

    /// Fraction of parameters currently frozen.
    #[inline]
    pub fn frozen_fraction(&self) -> f64 {
        if self.frozen.is_empty() {
            return 0.0;
        }
        self.n_frozen as f64 / self.frozen.len() as f64
    }

    /// Current importance values.
    #[inline]
    pub fn importance(&self) -> &[f64] {
        &self.importance
    }

    /// Unfreeze all parameters (fresh start for isolation).
    pub fn unfreeze_all(&mut self) {
        for f in &mut self.frozen {
            *f = false;
        }
        self.n_frozen = 0;
    }

    /// Compute the freeze mask from current importance values.
    ///
    /// Freezes the top `freeze_fraction` of currently-unfrozen parameters
    /// by importance. Already-frozen parameters remain frozen.
    fn apply_freeze(&mut self) {
        let n = self.importance.len();
        if n == 0 {
            return;
        }

        // Collect importance values of unfrozen parameters
        let mut unfrozen_importance: Vec<(usize, f64)> = Vec::new();
        for i in 0..n {
            if !self.frozen[i] {
                unfrozen_importance.push((i, self.importance[i]));
            }
        }

        if unfrozen_importance.is_empty() {
            return;
        }

        let n_unfrozen = unfrozen_importance.len();
        // Number to freeze from the currently-unfrozen set
        let n_to_freeze = math::round(self.freeze_fraction * n_unfrozen as f64) as usize;
        if n_to_freeze == 0 {
            return;
        }

        // Sort by importance descending to find top-k
        // Manual sort: simple insertion sort (small param counts in streaming models)
        for i in 1..unfrozen_importance.len() {
            let mut j = i;
            while j > 0 && unfrozen_importance[j].1 > unfrozen_importance[j - 1].1 {
                unfrozen_importance.swap(j, j - 1);
                j -= 1;
            }
        }

        // Freeze the top n_to_freeze
        for &(idx, _) in unfrozen_importance.iter().take(n_to_freeze) {
            self.frozen[idx] = true;
        }

        // Recount
        self.n_frozen = self.frozen.iter().filter(|&&f| f).count();
    }
}

impl ContinualStrategy for DriftMask {
    fn pre_update(&mut self, _params: &[f64], gradients: &mut [f64]) {
        let n = self.importance.len();
        debug_assert_eq!(gradients.len(), n);

        let alpha = self.importance_alpha;
        let one_minus_alpha = 1.0 - alpha;

        for ((imp, grad), &is_frozen) in self
            .importance
            .iter_mut()
            .zip(gradients.iter_mut())
            .zip(self.frozen.iter())
        {
            // Update importance EWMA with |grad|
            *imp = alpha * *imp + one_minus_alpha * math::abs(*grad);

            // Zero gradients for frozen parameters
            if is_frozen {
                *grad = 0.0;
            }
        }
    }

    fn post_update(&mut self, _params: &[f64]) {
        // No-op
    }

    fn on_drift(&mut self, _params: &[f64], signal: DriftSignal) {
        match signal {
            DriftSignal::Drift => {
                self.apply_freeze();
            }
            DriftSignal::Warning | DriftSignal::Stable => {
                // No action
            }
        }
    }

    #[inline]
    fn n_params(&self) -> usize {
        self.importance.len()
    }

    fn reset(&mut self) {
        for v in &mut self.importance {
            *v = 0.0;
        }
        for f in &mut self.frozen {
            *f = false;
        }
        self.n_frozen = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initially_nothing_frozen() {
        let mask = DriftMask::with_defaults(10);
        assert_eq!(mask.n_frozen(), 0);
        for i in 0..10 {
            assert!(
                !mask.is_frozen(i),
                "param {i} should not be frozen initially"
            );
        }
        assert!((mask.frozen_fraction() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn drift_freezes_top_fraction() {
        let mut mask = DriftMask::new(10, 0.3, 0.0);
        // With alpha=0.0, importance = |grad| directly (no memory of old)

        let params = [0.0; 10];
        // Give different importance to each param
        let mut grads = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        mask.pre_update(&params, &mut grads);

        // Importance is now [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        // Top 30% of 10 = 3 params: indices 7, 8, 9 (importance 8, 9, 10)
        mask.on_drift(&params, DriftSignal::Drift);

        assert_eq!(mask.n_frozen(), 3, "should freeze 30% = 3 params");
        assert!(
            mask.is_frozen(9),
            "param 9 (importance 10) should be frozen"
        );
        assert!(mask.is_frozen(8), "param 8 (importance 9) should be frozen");
        assert!(mask.is_frozen(7), "param 7 (importance 8) should be frozen");
        assert!(!mask.is_frozen(6), "param 6 should remain unfrozen");
        assert!(!mask.is_frozen(0), "param 0 should remain unfrozen");
    }

    #[test]
    fn frozen_params_have_zero_gradient() {
        let mut mask = DriftMask::new(4, 0.5, 0.0);

        let params = [0.0; 4];
        // Set importance: [1, 2, 3, 4]
        let mut grads = [1.0, 2.0, 3.0, 4.0];
        mask.pre_update(&params, &mut grads);

        // Freeze top 50% = 2 params (indices 2 and 3)
        mask.on_drift(&params, DriftSignal::Drift);
        assert!(mask.is_frozen(2));
        assert!(mask.is_frozen(3));

        // Now call pre_update again with new gradients
        let mut new_grads = [0.5, 0.5, 0.5, 0.5];
        mask.pre_update(&params, &mut new_grads);

        assert!(
            new_grads[2].abs() < 1e-12,
            "frozen param 2 gradient should be zero, got {}",
            new_grads[2]
        );
        assert!(
            new_grads[3].abs() < 1e-12,
            "frozen param 3 gradient should be zero, got {}",
            new_grads[3]
        );
    }

    #[test]
    fn unfrozen_params_pass_gradient_through() {
        let mut mask = DriftMask::new(4, 0.5, 0.0);

        let params = [0.0; 4];
        let mut grads = [1.0, 2.0, 3.0, 4.0];
        mask.pre_update(&params, &mut grads);

        // Freeze top 50% (indices 2, 3)
        mask.on_drift(&params, DriftSignal::Drift);

        // Check unfrozen params still pass gradients through
        let mut new_grads = [0.7, 0.8, 0.9, 1.0];
        mask.pre_update(&params, &mut new_grads);

        // Unfrozen params (0, 1) should have non-zero gradients
        assert!(
            new_grads[0].abs() > 1e-12,
            "unfrozen param 0 should have non-zero gradient"
        );
        assert!(
            new_grads[1].abs() > 1e-12,
            "unfrozen param 1 should have non-zero gradient"
        );
        // They should still be approximately 0.7 and 0.8 (not zeroed)
        assert!(
            (new_grads[0] - 0.7).abs() < 1e-12,
            "unfrozen param 0 gradient should pass through: got {}",
            new_grads[0]
        );
        assert!(
            (new_grads[1] - 0.8).abs() < 1e-12,
            "unfrozen param 1 gradient should pass through: got {}",
            new_grads[1]
        );
    }

    #[test]
    fn importance_tracks_gradient_magnitude() {
        let mut mask = DriftMask::new(3, 0.3, 0.5);

        let params = [0.0; 3];

        // First update: importance = 0.5 * 0 + 0.5 * |grad|
        let mut grads = [2.0, -4.0, 6.0];
        mask.pre_update(&params, &mut grads);

        let expected = [1.0, 2.0, 3.0]; // 0.5 * |grad|
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (mask.importance()[i] - exp).abs() < 1e-12,
                "importance[{i}] = {}, expected {}",
                mask.importance()[i],
                exp
            );
        }

        // Second update
        let mut grads2 = [0.0, 0.0, 0.0];
        mask.pre_update(&params, &mut grads2);
        // importance = 0.5 * prev + 0.5 * 0 = prev / 2
        let expected2 = [0.5, 1.0, 1.5];
        for (i, &exp) in expected2.iter().enumerate() {
            assert!(
                (mask.importance()[i] - exp).abs() < 1e-12,
                "importance[{i}] after 2nd = {}, expected {}",
                mask.importance()[i],
                exp
            );
        }
    }

    #[test]
    fn unfreeze_all_resets_mask() {
        let mut mask = DriftMask::new(5, 0.4, 0.0);

        let params = [0.0; 5];
        let mut grads = [1.0, 2.0, 3.0, 4.0, 5.0];
        mask.pre_update(&params, &mut grads);

        mask.on_drift(&params, DriftSignal::Drift);
        assert!(mask.n_frozen() > 0, "should have frozen some params");

        mask.unfreeze_all();
        assert_eq!(mask.n_frozen(), 0, "all params should be unfrozen");
        for i in 0..5 {
            assert!(!mask.is_frozen(i), "param {i} should be unfrozen");
        }
    }

    #[test]
    fn multiple_drifts_accumulate_frozen() {
        let mut mask = DriftMask::new(10, 0.3, 0.0);

        let params = [0.0; 10];
        let mut grads = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        mask.pre_update(&params, &mut grads);

        // First drift: freeze top 30% of 10 unfrozen = 3 params (indices 7, 8, 9)
        mask.on_drift(&params, DriftSignal::Drift);
        let frozen_after_first = mask.n_frozen();
        assert_eq!(frozen_after_first, 3);

        // Update importance for unfrozen params with new grads
        let mut grads2 = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.0, 0.0, 0.0];
        mask.pre_update(&params, &mut grads2);

        // Second drift: freeze top 30% of 7 unfrozen = round(0.3 * 7) = 2 params
        mask.on_drift(&params, DriftSignal::Drift);
        let frozen_after_second = mask.n_frozen();
        assert!(
            frozen_after_second > frozen_after_first,
            "second drift should freeze more: first={}, second={}",
            frozen_after_first,
            frozen_after_second
        );
        // Previously frozen params should still be frozen
        assert!(mask.is_frozen(9), "param 9 should still be frozen");
        assert!(mask.is_frozen(8), "param 8 should still be frozen");
        assert!(mask.is_frozen(7), "param 7 should still be frozen");
    }

    #[test]
    fn reset_clears_everything() {
        let mut mask = DriftMask::new(5, 0.4, 0.0);

        let params = [0.0; 5];
        let mut grads = [1.0, 2.0, 3.0, 4.0, 5.0];
        mask.pre_update(&params, &mut grads);
        mask.on_drift(&params, DriftSignal::Drift);

        assert!(mask.n_frozen() > 0);
        assert!(mask.importance().iter().any(|&v| v > 0.0));

        mask.reset();

        assert_eq!(
            mask.n_frozen(),
            0,
            "frozen count should be zero after reset"
        );
        assert!(
            mask.importance().iter().all(|&v| v == 0.0),
            "importance should be zeroed after reset"
        );
        for i in 0..5 {
            assert!(
                !mask.is_frozen(i),
                "param {i} should be unfrozen after reset"
            );
        }
    }

    #[test]
    fn warning_and_stable_do_not_freeze() {
        let mut mask = DriftMask::new(5, 0.5, 0.0);

        let params = [0.0; 5];
        let mut grads = [1.0, 2.0, 3.0, 4.0, 5.0];
        mask.pre_update(&params, &mut grads);

        mask.on_drift(&params, DriftSignal::Warning);
        assert_eq!(mask.n_frozen(), 0, "Warning should not freeze anything");

        mask.on_drift(&params, DriftSignal::Stable);
        assert_eq!(mask.n_frozen(), 0, "Stable should not freeze anything");
    }

    #[test]
    fn empty_mask_operations() {
        let mut mask = DriftMask::with_defaults(0);
        assert_eq!(mask.n_frozen(), 0);
        assert!((mask.frozen_fraction() - 0.0).abs() < 1e-12);

        let params: [f64; 0] = [];
        let mut grads: [f64; 0] = [];
        mask.pre_update(&params, &mut grads);
        mask.on_drift(&params, DriftSignal::Drift);
        mask.reset();
        assert_eq!(mask.n_params(), 0);
    }
}
