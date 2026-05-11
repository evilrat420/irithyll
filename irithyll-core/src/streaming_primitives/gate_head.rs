//! Learned gate projection head: `g = sigmoid(W·x + b)`.
//!
//! Shared building block for models that compute a data-dependent gate from the
//! current input. Used by Mamba V3 (selection gate), GatedDeltaNet (β_t),
//! sLSTM (auxiliary gates), and Titans (α_t / η_t / θ_t data-dependence).
//!
//! The gate is trained online via stochastic gradient descent on a squared
//! gate-target loss. The parameter update for a single sample:
//!
//! ```text
//! g          = sigmoid(W·x + b)
//! err        = g − target_gate
//! grad_scale = err * g * (1 − g)   // BCE-style gradient through sigmoid
//! W_i        += −lr * grad_scale * x_i
//! b          += −lr * grad_scale
//! ```

#[cfg(feature = "alloc")]
use crate::math;

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Open-interval clamp for sigmoid output.
///
/// Sigmoid in f64 saturates at exactly 0.0 / 1.0 once `|logit| ≳ 36`, which
/// breaks any downstream consumer that takes `log(g)` or `log(1-g)`. We clamp
/// to `[GATE_EPS, 1.0 - GATE_EPS]` to honor the documented `(0, 1)` contract.
/// `1e-15` sits well above f64 subnormals and well below normal-operation
/// gate noise floor.
#[cfg(feature = "alloc")]
const GATE_EPS: f64 = 1e-15;

/// Single-output learned gate: `g = sigmoid(W·x + b)`, online SGD updates.
///
/// # no_std
///
/// Requires the `alloc` feature. All allocations happen in [`new`](Self::new).
#[cfg(feature = "alloc")]
pub struct GateHead {
    /// Weight vector W, length = d_in.
    weights: Vec<f64>,
    /// Scalar bias term b.
    bias: f64,
}

#[cfg(feature = "alloc")]
impl GateHead {
    /// Create a new gate head for `d_in`-dimensional inputs.
    ///
    /// Weights are initialized to zero (cold-start: gate = 0.5 everywhere,
    /// i.e. sigmoid(0) = 0.5). The gate learns its useful values through
    /// [`update`](Self::update) calls.
    pub fn new(d_in: usize) -> Self {
        Self {
            weights: vec![0.0; d_in],
            bias: 0.0,
        }
    }

    /// Compute the gate value `g = sigmoid(W·x + b)` from input `x`.
    ///
    /// Returns a value strictly in `(0, 1)`. Output is clamped to
    /// `[GATE_EPS, 1.0 - GATE_EPS]` to honor the open-interval contract
    /// under extreme inputs — raw f64 sigmoid saturates exactly at 0.0
    /// and 1.0 once `|logit| ≳ 36`, which would produce `-∞` in any
    /// downstream `log(g)` / `log(1-g)` consumer.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `x.len() != self.weights.len()`.
    #[inline]
    pub fn forward(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(
            x.len(),
            self.weights.len(),
            "GateHead: input len {} != weight len {}",
            x.len(),
            self.weights.len()
        );
        let mut logit = self.bias;
        for (&w, &xi) in self.weights.iter().zip(x.iter()) {
            logit += w * xi;
        }
        math::sigmoid(logit).clamp(GATE_EPS, 1.0 - GATE_EPS)
    }

    /// Update weights toward `target_gate ∈ (0,1)` with learning rate `lr`.
    ///
    /// Applies one step of SGD on the squared loss `(g − target_gate)²`:
    ///
    /// ```text
    /// g          = sigmoid(W·x + b)
    /// err        = g − target_gate
    /// grad_scale = err * g * (1 − g)
    /// W_i       -= lr * grad_scale * x_i
    /// b         -= lr * grad_scale
    /// ```
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `x.len() != self.weights.len()`.
    pub fn update(&mut self, x: &[f64], target_gate: f64, lr: f64) {
        debug_assert_eq!(x.len(), self.weights.len());
        let g = self.forward(x);
        let err = g - target_gate;
        let grad_scale = err * g * (1.0 - g);
        let scaled_lr = lr * grad_scale;
        for (w, &xi) in self.weights.iter_mut().zip(x.iter()) {
            *w -= scaled_lr * xi;
        }
        self.bias -= scaled_lr;
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    /// Gate output is in (0, 1) on zero-init weights (logit = 0 → 0.5).
    #[test]
    fn forward_returns_value_in_unit_interval() {
        let head = GateHead::new(4);
        // Exact value at all-zero weights: sigmoid(0) = 0.5.
        let g = head.forward(&[1.0, -2.0, 3.0, -4.0]);
        assert!(g > 0.0 && g < 1.0, "gate output {g} must be in (0, 1)");
        assert!((g - 0.5).abs() < 1e-12, "zero-init gate should equal 0.5");
    }

    /// After repeated updates toward target, gate moves closer.
    #[test]
    fn update_reduces_gate_target_distance() {
        let x = [1.0, 0.5, -0.5, 0.0];
        let target = 0.9;
        let lr = 0.5;
        let mut head = GateHead::new(4);

        let initial_dist = (head.forward(&x) - target).abs();
        for _ in 0..200 {
            head.update(&x, target, lr);
        }
        let final_dist = (head.forward(&x) - target).abs();

        assert!(
            final_dist < initial_dist,
            "gate should converge toward target: initial_dist={initial_dist}, final_dist={final_dist}"
        );
    }

    /// Output always in (0,1) regardless of extreme input magnitudes.
    #[test]
    fn forward_bounded_on_extreme_inputs() {
        let mut head = GateHead::new(3);
        // Push weights to large values via updates.
        for _ in 0..1000 {
            head.update(&[100.0, -100.0, 50.0], 1.0, 0.1);
        }
        let g = head.forward(&[1000.0, -1000.0, 500.0]);
        assert!(g > 0.0 && g < 1.0, "gate {g} must stay in (0, 1)");
    }

    /// Update with target=0.5 on zero-init inputs should leave weights unchanged.
    #[test]
    fn update_to_current_gate_value_is_zero_gradient() {
        // sigmoid(0) = 0.5; target = 0.5 → err = 0 → grad = 0 → no weight change.
        let mut head = GateHead::new(2);
        head.update(&[1.0, 2.0], 0.5, 1.0);
        // Weights should remain at zero (err = 0 ⇒ no update).
        for &w in &head.weights {
            assert!(
                w.abs() < 1e-15,
                "weight should not change when gate == target"
            );
        }
    }
}
