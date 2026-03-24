//! Streaming Elastic Weight Consolidation.
//!
//! Maintains a diagonal Fisher Information Matrix as an exponential moving
//! average of squared gradients. When drift is detected, the anchor point
//! (parameter snapshot) shifts, allowing adaptation to the new regime while
//! preserving knowledge of the old.
//!
//! # The Math
//!
//! The EWC penalty added to the loss is:
//! ```text
//! L_ewc = (ewc_lambda / 2) * sum_i F_i * (theta_i - theta_i*)^2
//! ```
//! where F_i is the diagonal Fisher, theta_i* is the anchor, theta_i is current.
//!
//! The gradient modification is:
//! ```text
//! grad_i = grad_i + ewc_lambda * F_i * (theta_i - theta_i*)
//! ```
//!
//! The Fisher is maintained as EWMA:
//! ```text
//! F_i = fisher_alpha * F_{i-1} + (1 - fisher_alpha) * grad_i^2
//! ```
//!
//! # References
//!
//! - Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks",
//!   PNAS 114(13), 2017.

use alloc::vec::Vec;

use super::ContinualStrategy;
use crate::drift::DriftSignal;

/// Streaming EWC: prevents catastrophic forgetting via Fisher regularization.
///
/// Maintains a diagonal Fisher Information Matrix as an exponential moving
/// average of squared gradients. When drift is detected, the anchor point
/// (parameter snapshot) shifts, allowing adaptation to the new regime while
/// preserving knowledge of the old.
///
/// # Example
///
/// ```
/// use irithyll_core::continual::{StreamingEWC, ContinualStrategy};
/// use irithyll_core::drift::DriftSignal;
///
/// let mut ewc = StreamingEWC::with_defaults(4);
/// let params = vec![1.0, 2.0, 3.0, 4.0];
/// ewc.set_anchor(&params);
///
/// let mut grads = vec![0.1, -0.2, 0.3, -0.4];
/// ewc.pre_update(&params, &mut grads);
/// // Gradients are modified by EWC penalty toward anchor
/// ```
pub struct StreamingEWC {
    /// Diagonal Fisher information (EWMA of grad^2).
    fisher_diag: Vec<f64>,
    /// Parameter snapshot from last drift/init.
    anchor_params: Vec<f64>,
    /// EWMA decay for Fisher (default: 0.99).
    fisher_alpha: f64,
    /// Regularization strength (default: 1.0).
    ewc_lambda: f64,
    /// Total updates seen.
    n_updates: u64,
    /// Whether anchor has been set.
    initialized: bool,
}

impl StreamingEWC {
    /// Create a new `StreamingEWC` with explicit hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `n_params` -- number of parameters to protect
    /// * `ewc_lambda` -- regularization strength; higher = stronger memory
    /// * `fisher_alpha` -- EWMA decay for Fisher; closer to 1.0 = longer memory
    ///
    /// # Panics
    ///
    /// Panics if `fisher_alpha` is not in `[0.0, 1.0]`.
    pub fn new(n_params: usize, ewc_lambda: f64, fisher_alpha: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&fisher_alpha),
            "fisher_alpha must be in [0.0, 1.0], got {fisher_alpha}"
        );
        Self {
            fisher_diag: alloc::vec![0.0; n_params],
            anchor_params: alloc::vec![0.0; n_params],
            fisher_alpha,
            ewc_lambda,
            n_updates: 0,
            initialized: false,
        }
    }

    /// Create a new `StreamingEWC` with default hyperparameters.
    ///
    /// Defaults: `ewc_lambda = 1.0`, `fisher_alpha = 0.99`.
    pub fn with_defaults(n_params: usize) -> Self {
        Self::new(n_params, 1.0, 0.99)
    }

    /// Current diagonal Fisher information.
    #[inline]
    pub fn fisher(&self) -> &[f64] {
        &self.fisher_diag
    }

    /// Current anchor parameters.
    #[inline]
    pub fn anchor(&self) -> &[f64] {
        &self.anchor_params
    }

    /// Current EWC regularization strength.
    #[inline]
    pub fn ewc_lambda(&self) -> f64 {
        self.ewc_lambda
    }

    /// Number of updates processed.
    #[inline]
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Whether the anchor has been initialized.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Manually set the anchor to a parameter snapshot.
    ///
    /// # Panics
    ///
    /// Panics if `params.len() != n_params()`.
    pub fn set_anchor(&mut self, params: &[f64]) {
        assert_eq!(
            params.len(),
            self.fisher_diag.len(),
            "set_anchor: expected {} params, got {}",
            self.fisher_diag.len(),
            params.len()
        );
        self.anchor_params.copy_from_slice(params);
        self.initialized = true;
    }

    /// Compute the EWC penalty value for given parameters.
    ///
    /// Returns `(ewc_lambda / 2) * sum_i F_i * (theta_i - anchor_i)^2`.
    ///
    /// Returns `0.0` if the anchor has not been set.
    pub fn penalty(&self, params: &[f64]) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        let mut total = 0.0;
        for ((&f, &a), &p) in self
            .fisher_diag
            .iter()
            .zip(self.anchor_params.iter())
            .zip(params.iter())
        {
            let diff = p - a;
            total += f * diff * diff;
        }
        0.5 * self.ewc_lambda * total
    }
}

impl ContinualStrategy for StreamingEWC {
    fn pre_update(&mut self, params: &[f64], gradients: &mut [f64]) {
        let n = self.fisher_diag.len();
        debug_assert_eq!(params.len(), n);
        debug_assert_eq!(gradients.len(), n);

        let alpha = self.fisher_alpha;
        let one_minus_alpha = 1.0 - alpha;

        for i in 0..n {
            // Update Fisher EWMA with squared gradient
            self.fisher_diag[i] =
                alpha * self.fisher_diag[i] + one_minus_alpha * gradients[i] * gradients[i];

            // Add EWC gradient penalty if anchor is set
            if self.initialized {
                let diff = params[i] - self.anchor_params[i];
                gradients[i] += self.ewc_lambda * self.fisher_diag[i] * diff;
            }
        }
    }

    fn post_update(&mut self, _params: &[f64]) {
        self.n_updates += 1;
    }

    fn on_drift(&mut self, params: &[f64], signal: DriftSignal) {
        match signal {
            DriftSignal::Drift => {
                // Confirmed drift: snapshot current params as new anchor
                self.set_anchor(params);
            }
            DriftSignal::Warning | DriftSignal::Stable => {
                // No action on Warning or Stable
            }
        }
    }

    #[inline]
    fn n_params(&self) -> usize {
        self.fisher_diag.len()
    }

    fn reset(&mut self) {
        for v in &mut self.fisher_diag {
            *v = 0.0;
        }
        for v in &mut self.anchor_params {
            *v = 0.0;
        }
        self.n_updates = 0;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewc_gradient_penalty_pushes_toward_anchor() {
        let mut ewc = StreamingEWC::new(3, 2.0, 0.5);
        let anchor = [0.0, 0.0, 0.0];
        ewc.set_anchor(&anchor);

        // Seed Fisher with a first call so it's nonzero
        let params = [1.0, -1.0, 0.5];
        let mut grads = [0.1, 0.1, 0.1];
        ewc.pre_update(&params, &mut grads);

        // After one step, Fisher has accumulated some value.
        // The penalty gradient is ewc_lambda * F_i * (param_i - anchor_i).
        // For param=1.0, anchor=0.0 => penalty pushes grad positive (increasing loss
        // for moving away from anchor), so grad should be > original 0.1.
        // For param=-1.0, anchor=0.0 => penalty pushes grad negative.
        // Verify the direction:
        // grad[0] was 0.1, param[0]-anchor[0] = 1.0 > 0 => penalty added is positive => grad[0] > 0.1
        assert!(
            grads[0] > 0.1,
            "gradient should be pushed away from anchor direction: got {}",
            grads[0]
        );
        // grad[1] was 0.1, param[1]-anchor[1] = -1.0 < 0 => penalty is negative => grad[1] < 0.1
        assert!(
            grads[1] < 0.1,
            "gradient should be pushed toward anchor: got {}",
            grads[1]
        );
    }

    #[test]
    fn fisher_accumulates_squared_gradients() {
        let mut ewc = StreamingEWC::new(2, 0.0, 0.5);
        // lambda=0 so no penalty contamination

        let params = [0.0, 0.0];
        let mut grads = [2.0, 3.0];
        ewc.pre_update(&params, &mut grads);

        // After one update: F_i = 0.5 * 0.0 + 0.5 * grad_i^2
        let expected_f0 = 0.5 * 0.0 + 0.5 * 4.0; // 2.0
        let expected_f1 = 0.5 * 0.0 + 0.5 * 9.0; // 4.5
        assert!(
            (ewc.fisher()[0] - expected_f0).abs() < 1e-12,
            "fisher[0] = {}, expected {}",
            ewc.fisher()[0],
            expected_f0
        );
        assert!(
            (ewc.fisher()[1] - expected_f1).abs() < 1e-12,
            "fisher[1] = {}, expected {}",
            ewc.fisher()[1],
            expected_f1
        );

        // Second update with different grads
        let mut grads2 = [1.0, 1.0];
        ewc.pre_update(&params, &mut grads2);
        // F_i = 0.5 * prev + 0.5 * grad_i^2
        let expected_f0_2 = 0.5 * expected_f0 + 0.5 * 1.0; // 1.5
        let expected_f1_2 = 0.5 * expected_f1 + 0.5 * 1.0; // 2.75
        assert!(
            (ewc.fisher()[0] - expected_f0_2).abs() < 1e-12,
            "fisher[0] after 2nd = {}, expected {}",
            ewc.fisher()[0],
            expected_f0_2
        );
        assert!(
            (ewc.fisher()[1] - expected_f1_2).abs() < 1e-12,
            "fisher[1] after 2nd = {}, expected {}",
            ewc.fisher()[1],
            expected_f1_2
        );
    }

    #[test]
    fn drift_signal_updates_anchor() {
        let mut ewc = StreamingEWC::with_defaults(3);
        let initial = [1.0, 2.0, 3.0];
        ewc.set_anchor(&initial);
        assert_eq!(ewc.anchor(), &[1.0, 2.0, 3.0]);

        let new_params = [4.0, 5.0, 6.0];
        ewc.on_drift(&new_params, DriftSignal::Drift);
        assert_eq!(
            ewc.anchor(),
            &[4.0, 5.0, 6.0],
            "anchor should be updated on Drift signal"
        );
    }

    #[test]
    fn warning_signal_no_anchor_change() {
        let mut ewc = StreamingEWC::with_defaults(2);
        let anchor = [1.0, 2.0];
        ewc.set_anchor(&anchor);

        let new_params = [10.0, 20.0];
        ewc.on_drift(&new_params, DriftSignal::Warning);
        assert_eq!(
            ewc.anchor(),
            &[1.0, 2.0],
            "anchor should not change on Warning"
        );
    }

    #[test]
    fn stable_signal_no_effect() {
        let mut ewc = StreamingEWC::with_defaults(2);
        let anchor = [1.0, 2.0];
        ewc.set_anchor(&anchor);

        let new_params = [10.0, 20.0];
        ewc.on_drift(&new_params, DriftSignal::Stable);
        assert_eq!(
            ewc.anchor(),
            &[1.0, 2.0],
            "anchor should not change on Stable"
        );
    }

    #[test]
    fn penalty_increases_with_distance_from_anchor() {
        let mut ewc = StreamingEWC::new(2, 1.0, 0.5);
        let anchor = [0.0, 0.0];
        ewc.set_anchor(&anchor);

        // Seed Fisher with known values
        let params = [0.0, 0.0];
        let mut grads = [1.0, 1.0];
        ewc.pre_update(&params, &mut grads);
        // Fisher is now [0.5, 0.5]

        let close = [0.1, 0.1];
        let far = [1.0, 1.0];
        let penalty_close = ewc.penalty(&close);
        let penalty_far = ewc.penalty(&far);

        assert!(
            penalty_far > penalty_close,
            "penalty should increase with distance: close={}, far={}",
            penalty_close,
            penalty_far
        );
        assert!(
            penalty_close > 0.0,
            "penalty should be positive for non-zero distance"
        );
    }

    #[test]
    fn reset_clears_all_state() {
        let mut ewc = StreamingEWC::with_defaults(3);
        let params = [1.0, 2.0, 3.0];
        ewc.set_anchor(&params);

        let mut grads = [0.5, 0.5, 0.5];
        ewc.pre_update(&params, &mut grads);
        ewc.post_update(&params);

        assert!(ewc.is_initialized());
        assert!(ewc.n_updates() > 0);
        assert!(ewc.fisher().iter().any(|&f| f > 0.0));

        ewc.reset();

        assert!(!ewc.is_initialized());
        assert_eq!(ewc.n_updates(), 0);
        assert!(
            ewc.fisher().iter().all(|&f| f == 0.0),
            "Fisher should be zeroed after reset"
        );
        assert!(
            ewc.anchor().iter().all(|&a| a == 0.0),
            "anchor should be zeroed after reset"
        );
    }

    #[test]
    fn zero_lambda_means_no_penalty() {
        let mut ewc = StreamingEWC::new(3, 0.0, 0.99);
        let anchor = [0.0, 0.0, 0.0];
        ewc.set_anchor(&anchor);

        // Seed Fisher
        let params = [0.0, 0.0, 0.0];
        let mut grads_seed = [1.0, 1.0, 1.0];
        ewc.pre_update(&params, &mut grads_seed);

        // Now with params far from anchor
        let params_far = [10.0, 10.0, 10.0];
        let original_grads = [0.5, -0.3, 0.7];
        let mut grads = original_grads;
        ewc.pre_update(&params_far, &mut grads);

        // With lambda=0, the EWC penalty term is zero, so gradients should
        // only differ due to Fisher update (which doesn't affect them with lambda=0)
        for i in 0..3 {
            assert!(
                (grads[i] - original_grads[i]).abs() < 1e-12,
                "gradient[{i}] should be unchanged with lambda=0: got {}, expected {}",
                grads[i],
                original_grads[i]
            );
        }

        assert!(
            ewc.penalty(&params_far).abs() < 1e-12,
            "penalty should be zero with lambda=0"
        );
    }

    #[test]
    fn uninitialized_ewc_has_no_penalty() {
        let ewc = StreamingEWC::with_defaults(3);
        let params = [10.0, 20.0, 30.0];
        assert!(
            ewc.penalty(&params).abs() < 1e-12,
            "penalty should be zero before anchor is set"
        );
    }

    #[test]
    fn post_update_increments_counter() {
        let mut ewc = StreamingEWC::with_defaults(2);
        assert_eq!(ewc.n_updates(), 0);

        ewc.post_update(&[1.0, 2.0]);
        assert_eq!(ewc.n_updates(), 1);

        ewc.post_update(&[1.0, 2.0]);
        assert_eq!(ewc.n_updates(), 2);
    }
}
