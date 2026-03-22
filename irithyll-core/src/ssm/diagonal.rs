//! Non-selective diagonal state space model with fixed parameters.
//!
//! [`DiagonalSSM`] implements a simple diagonal SSM where the A, B, C matrices
//! and step size delta are fixed at construction time. This provides a baseline
//! temporal feature extractor without the input-dependent selectivity of Mamba.
//!
//! The model processes scalar inputs and produces scalar outputs:
//!
//! ```text
//! h_t = A_bar .* h_{t-1} + B_bar * x_t    (state update, element-wise)
//! y_t = C^T * h_t + D * x_t                 (output)
//! ```
//!
//! where `A_bar` and `B_bar` are computed once via ZOH discretization of the
//! continuous-time parameters.

use alloc::vec;
use alloc::vec::Vec;

use crate::math;
use crate::ssm::discretize::zoh_discretize;
use crate::ssm::init::mamba_init;
use crate::ssm::projection::dot;
use crate::ssm::SSMLayer;

/// Non-selective diagonal SSM with fixed A, B, C, and step size.
///
/// This is the simplest SSM variant: all parameters are determined at
/// construction time and do not adapt to input content. The hidden state
/// is an N-dimensional vector that evolves through element-wise recurrence.
///
/// # When to Use
///
/// Use `DiagonalSSM` when:
/// - You want a simple temporal smoothing/memory layer
/// - Input-dependent selectivity is not needed
/// - You need a fast baseline to compare against [`SelectiveSSM`](super::SelectiveSSM)
///
/// # Example
///
/// ```
/// use irithyll_core::ssm::diagonal::DiagonalSSM;
/// use irithyll_core::ssm::SSMLayer;
///
/// let mut ssm = DiagonalSSM::new(8, 0.1);
/// let output = ssm.forward(&[1.0]);
/// assert_eq!(output.len(), 1);
/// ```
pub struct DiagonalSSM {
    /// Log-space A parameters (N). Actual A_n = -exp(log_a_n).
    log_a: Vec<f64>,
    /// B vector (N) -- input-to-state projection.
    b: Vec<f64>,
    /// C vector (N) -- state-to-output projection.
    c: Vec<f64>,
    /// Fixed discretization step size.
    delta: f64,
    /// Skip connection weight (D in the SSM equations).
    d_skip: f64,
    /// Hidden state vector (N).
    h: Vec<f64>,
    /// Pre-computed discretized A_bar values (N). Cached for efficiency.
    a_bar: Vec<f64>,
    /// Pre-computed discretized B_bar factor values (N). Cached for efficiency.
    b_bar_factor: Vec<f64>,
}

impl DiagonalSSM {
    /// Create a new non-selective diagonal SSM.
    ///
    /// Initializes A via the Mamba strategy (`A_n = -(n+1)`), sets B and C
    /// to ones, D (skip connection) to zero, and pre-computes the discretized
    /// matrices using ZOH.
    ///
    /// # Arguments
    ///
    /// * `n_state` -- number of hidden state dimensions (N)
    /// * `delta` -- fixed discretization step size (positive)
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll_core::ssm::diagonal::DiagonalSSM;
    ///
    /// let ssm = DiagonalSSM::new(16, 0.01);
    /// ```
    pub fn new(n_state: usize, delta: f64) -> Self {
        let log_a = mamba_init(n_state);
        let b = vec![1.0; n_state];
        let c = vec![1.0; n_state];
        let h = vec![0.0; n_state];

        // Pre-compute discretized A_bar and B_bar_factor
        let mut a_bar = Vec::with_capacity(n_state);
        let mut b_bar_factor = Vec::with_capacity(n_state);
        for la in &log_a {
            let a_n = -math::exp(*la);
            let (ab, bbf) = zoh_discretize(a_n, delta);
            a_bar.push(ab);
            b_bar_factor.push(bbf);
        }

        Self {
            log_a,
            b,
            c,
            delta,
            d_skip: 0.0,
            h,
            a_bar,
            b_bar_factor,
        }
    }

    /// Create a diagonal SSM with custom B, C vectors and skip connection.
    ///
    /// # Arguments
    ///
    /// * `n_state` -- number of hidden state dimensions
    /// * `delta` -- fixed discretization step size
    /// * `b` -- input projection vector (length n_state)
    /// * `c` -- output projection vector (length n_state)
    /// * `d_skip` -- skip connection weight
    ///
    /// # Panics
    ///
    /// Debug-asserts that `b.len() == n_state` and `c.len() == n_state`.
    pub fn with_params(n_state: usize, delta: f64, b: Vec<f64>, c: Vec<f64>, d_skip: f64) -> Self {
        debug_assert_eq!(b.len(), n_state);
        debug_assert_eq!(c.len(), n_state);
        let log_a = mamba_init(n_state);
        let h = vec![0.0; n_state];

        let mut a_bar = Vec::with_capacity(n_state);
        let mut b_bar_factor = Vec::with_capacity(n_state);
        for la in &log_a {
            let a_n = -math::exp(*la);
            let (ab, bbf) = zoh_discretize(a_n, delta);
            a_bar.push(ab);
            b_bar_factor.push(bbf);
        }

        Self {
            log_a,
            b,
            c,
            delta,
            d_skip,
            h,
            a_bar,
            b_bar_factor,
        }
    }

    /// Process a single scalar input and return the scalar output.
    ///
    /// Updates the hidden state via:
    /// ```text
    /// h_n = a_bar_n * h_n + b_bar_factor_n * b_n * x
    /// y = C^T * h + D * x
    /// ```
    #[inline]
    pub fn forward_scalar(&mut self, x: f64) -> f64 {
        let n_state = self.h.len();
        for n in 0..n_state {
            self.h[n] = self.a_bar[n] * self.h[n] + self.b_bar_factor[n] * self.b[n] * x;
        }
        dot(&self.c, &self.h) + self.d_skip * x
    }

    /// Get the number of state dimensions.
    #[inline]
    pub fn n_state(&self) -> usize {
        self.h.len()
    }

    /// Get the log-A parameters (for inspection/serialization).
    #[inline]
    pub fn log_a(&self) -> &[f64] {
        &self.log_a
    }

    /// Get the fixed step size.
    #[inline]
    pub fn delta(&self) -> f64 {
        self.delta
    }
}

impl SSMLayer for DiagonalSSM {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        // DiagonalSSM processes scalar input -- take first element or 0
        let x = if input.is_empty() { 0.0 } else { input[0] };
        vec![self.forward_scalar(x)]
    }

    fn state(&self) -> &[f64] {
        &self.h
    }

    fn output_dim(&self) -> usize {
        1
    }

    fn reset(&mut self) {
        for h in self.h.iter_mut() {
            *h = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_zero_state() {
        let ssm = DiagonalSSM::new(8, 0.1);
        assert_eq!(ssm.n_state(), 8);
        for &h in ssm.state() {
            assert!(math::abs(h) < 1e-15, "initial state should be zero");
        }
    }

    #[test]
    fn forward_scalar_produces_finite_output() {
        let mut ssm = DiagonalSSM::new(4, 0.1);
        let y = ssm.forward_scalar(1.0);
        assert!(y.is_finite(), "output should be finite, got {}", y);
    }

    #[test]
    fn forward_updates_state() {
        let mut ssm = DiagonalSSM::new(4, 0.1);
        let _ = ssm.forward_scalar(1.0);
        let state_norm: f64 = ssm.state().iter().map(|h| h * h).sum();
        assert!(
            state_norm > 0.0,
            "state should be non-zero after processing input"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut ssm = DiagonalSSM::new(4, 0.1);
        let _ = ssm.forward_scalar(1.0);
        ssm.reset();
        for &h in ssm.state() {
            assert!(math::abs(h) < 1e-15, "state should be zero after reset");
        }
    }

    #[test]
    fn state_decays_without_input() {
        let mut ssm = DiagonalSSM::new(4, 0.1);
        // Inject some state
        let _ = ssm.forward_scalar(10.0);
        let energy_after_input: f64 = ssm.state().iter().map(|h| h * h).sum();

        // Process many zero inputs -- state should decay
        for _ in 0..100 {
            let _ = ssm.forward_scalar(0.0);
        }
        let energy_after_decay: f64 = ssm.state().iter().map(|h| h * h).sum();
        assert!(
            energy_after_decay < energy_after_input * 0.01,
            "state energy should decay: initial={}, after={}",
            energy_after_input,
            energy_after_decay
        );
    }

    #[test]
    fn ssm_layer_trait_works() {
        let mut ssm = DiagonalSSM::new(4, 0.1);
        let out = ssm.forward(&[1.0]);
        assert_eq!(out.len(), 1, "output_dim should be 1");
        assert_eq!(ssm.output_dim(), 1);
    }

    #[test]
    fn constant_input_converges() {
        // With constant input, output should converge to a steady state
        let mut ssm = DiagonalSSM::new(4, 0.1);
        let mut prev_y = 0.0;
        let mut settled = false;
        for i in 0..500 {
            let y = ssm.forward_scalar(1.0);
            if i > 10 && math::abs(y - prev_y) < 1e-10 {
                settled = true;
                break;
            }
            prev_y = y;
        }
        assert!(settled, "output should converge for constant input");
    }

    #[test]
    fn skip_connection_passes_through() {
        let b = vec![0.0; 4]; // Zero B means no state update from input
        let c = vec![0.0; 4]; // Zero C means no state contribution to output
        let mut ssm = DiagonalSSM::with_params(4, 0.1, b, c, 1.0);
        let y = ssm.forward_scalar(5.0);
        assert!(
            math::abs(y - 5.0) < 1e-12,
            "with zero B/C and d_skip=1, output should equal input: got {}",
            y
        );
    }

    #[test]
    fn empty_input_treated_as_zero() {
        let mut ssm = DiagonalSSM::new(4, 0.1);
        let out = ssm.forward(&[]);
        assert_eq!(out.len(), 1);
        assert!(
            math::abs(out[0]) < 1e-15,
            "empty input should be treated as zero"
        );
    }

    #[test]
    fn different_delta_changes_dynamics() {
        let mut ssm_fast = DiagonalSSM::new(4, 1.0);
        let mut ssm_slow = DiagonalSSM::new(4, 0.001);

        // Same input sequence
        let _ = ssm_fast.forward_scalar(1.0);
        let y_fast = ssm_fast.forward_scalar(0.0);

        let _ = ssm_slow.forward_scalar(1.0);
        let y_slow = ssm_slow.forward_scalar(0.0);

        // The faster step size should produce different decay behavior
        assert!(
            math::abs(y_fast - y_slow) > 1e-6,
            "different delta should produce different dynamics: fast={}, slow={}",
            y_fast,
            y_slow
        );
    }
}
