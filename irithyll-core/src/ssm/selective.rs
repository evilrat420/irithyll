//! Mamba-style selective state space model with input-dependent projections.
//!
//! [`SelectiveSSM`] implements the core selective mechanism from Mamba
//! (Gu & Dao, 2023), where the discretization step size Delta, input matrix B,
//! and output matrix C are all functions of the current input. This enables
//! content-aware filtering: the model can learn to selectively remember or
//! forget information based on what it sees.
//!
//! # Architecture
//!
//! For each input timestep `x_t` (a d_in-dimensional vector):
//!
//! ```text
//! Delta_t = softplus(W_delta * x_t + b_delta)    // scalar step size
//! B_t     = W_B * x_t                             // N-dim input projection
//! C_t     = W_C * x_t                             // N-dim output projection
//! A       = -exp(log_A)                           // fixed, always negative diagonal
//!
//! For each input channel d in 0..d_in:
//!   For each state dim n in 0..N:
//!     A_bar = exp(Delta_t * A_n)
//!     B_bar = (A_bar - 1) / A_n * B_t[n]
//!     h[d,n] = A_bar * h[d,n] + B_bar * x_t[d]
//!   output[d] = C_t^T * h[d,:] + D[d] * x_t[d]
//! ```
//!
//! Each input channel maintains its own N-dimensional state vector, allowing
//! the model to track per-channel temporal patterns independently.

use alloc::vec;
use alloc::vec::Vec;

use crate::math;
use crate::ssm::init::mamba_init;
use crate::ssm::projection::{dot, mat_vec, softplus, Xorshift64};
use crate::ssm::SSMLayer;

/// Mamba-style selective state space model.
///
/// The selective mechanism computes input-dependent B, C, and Delta at each
/// timestep, enabling the model to dynamically control what information is
/// stored in and retrieved from the hidden state.
///
/// # Dimensions
///
/// - `d_in` -- input/output dimension (number of channels)
/// - `n_state` -- hidden state dimension per channel (N)
/// - Total hidden state size: `d_in * n_state`
///
/// # Weight Shapes
///
/// | Weight | Shape | Purpose |
/// |--------|-------|---------|
/// | `w_delta` | d_in | Projects input to scalar step size |
/// | `w_b` | N x d_in | Projects input to state-input coupling |
/// | `w_c` | N x d_in | Projects input to state-output coupling |
/// | `d_skip` | d_in | Skip connection weights |
/// | `log_a` | N | Fixed state transition (always negative after exp) |
///
/// # Example
///
/// ```
/// use irithyll_core::ssm::selective::SelectiveSSM;
/// use irithyll_core::ssm::SSMLayer;
///
/// let mut ssm = SelectiveSSM::new(4, 8, 42);
/// let output = ssm.forward(&[1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(output.len(), 4);
/// ```
pub struct SelectiveSSM {
    /// Log-space A parameters (N). Actual A_n = -exp(log_a[n]).
    log_a: Vec<f64>,
    /// Delta projection weights (d_in). Maps input to scalar step size.
    w_delta: Vec<f64>,
    /// Delta projection bias.
    b_delta: f64,
    /// B projection weights (N x d_in, row-major). Maps input to B_t.
    w_b: Vec<f64>,
    /// C projection weights (N x d_in, row-major). Maps input to C_t.
    w_c: Vec<f64>,
    /// Skip connection weights (d_in).
    d_skip: Vec<f64>,
    /// Hidden state (d_in * N, laid out as [channel_0_state, channel_1_state, ...]).
    h: Vec<f64>,
    /// Number of state dimensions per channel.
    n_state: usize,
    /// Input/output dimension.
    d_in: usize,
}

impl SelectiveSSM {
    /// Create a new selective SSM with random weight initialization.
    ///
    /// Weights are initialized from a small normal distribution (scale 0.1)
    /// using the provided seed for reproducibility. A is initialized via the
    /// Mamba strategy (A_n = -(n+1)). Skip connections (D) are initialized
    /// to 1.0 to enable input passthrough by default.
    ///
    /// # Arguments
    ///
    /// * `d_in` -- input/output dimension (number of channels)
    /// * `n_state` -- hidden state dimension per channel (N)
    /// * `seed` -- random seed for weight initialization
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll_core::ssm::selective::SelectiveSSM;
    ///
    /// let ssm = SelectiveSSM::new(4, 16, 42);
    /// ```
    pub fn new(d_in: usize, n_state: usize, seed: u64) -> Self {
        let log_a = mamba_init(n_state);
        let mut rng = Xorshift64(seed);
        let scale = 0.1;

        // Initialize projection weights from small normal distribution
        let w_delta: Vec<f64> = (0..d_in).map(|_| rng.next_normal() * scale).collect();
        let b_delta = 0.0;
        let w_b: Vec<f64> = (0..n_state * d_in)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let w_c: Vec<f64> = (0..n_state * d_in)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let d_skip = vec![1.0; d_in];
        let h = vec![0.0; d_in * n_state];

        Self {
            log_a,
            w_delta,
            b_delta,
            w_b,
            w_c,
            d_skip,
            h,
            n_state,
            d_in,
        }
    }

    /// Get the input/output dimension.
    #[inline]
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Get the number of state dimensions per channel.
    #[inline]
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// Compute the selective SSM forward pass for one timestep.
    ///
    /// This is the core Mamba recurrence: compute input-dependent Delta, B, C,
    /// then update each channel's state and produce the output.
    fn selective_forward(&mut self, input: &[f64]) -> Vec<f64> {
        let d_in = self.d_in;
        let n_state = self.n_state;

        // 1. Compute delta = softplus(dot(w_delta, input) + b_delta)
        let delta_raw = dot(&self.w_delta, input) + self.b_delta;
        let delta = softplus(delta_raw);

        // 2. Compute B_t = W_B * input (shape: N)
        let mut b_t = vec![0.0; n_state];
        mat_vec(&self.w_b, input, n_state, d_in, &mut b_t);

        // 3. Compute C_t = W_C * input (shape: N)
        let mut c_t = vec![0.0; n_state];
        mat_vec(&self.w_c, input, n_state, d_in, &mut c_t);

        // 4. For each input channel, update state and compute output
        let mut output = vec![0.0; d_in];

        for d in 0..d_in {
            let h_offset = d * n_state;
            let mut y = 0.0;

            for n in 0..n_state {
                let a_n = -math::exp(self.log_a[n]); // negative real diagonal

                // ZOH discretization: A_bar = exp(delta * A_n)
                let a_bar = math::exp(delta * a_n);

                // B_bar = (A_bar - 1) / A_n * B_t[n]
                // Handle A_n ~ 0 (shouldn't happen with mamba_init, but be safe)
                let b_bar = if math::abs(a_n) < 1e-12 {
                    delta * b_t[n]
                } else {
                    (a_bar - 1.0) / a_n * b_t[n]
                };

                // State update: h[d,n] = A_bar * h[d,n] + B_bar * input[d]
                self.h[h_offset + n] = a_bar * self.h[h_offset + n] + b_bar * input[d];

                // Accumulate output: y += C_t[n] * h[d,n]
                y += c_t[n] * self.h[h_offset + n];
            }

            // Add skip connection
            output[d] = y + self.d_skip[d] * input[d];
        }

        output
    }
}

impl SSMLayer for SelectiveSSM {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            input.len(),
            self.d_in,
            "input length {} must match d_in {}",
            input.len(),
            self.d_in
        );
        self.selective_forward(input)
    }

    fn state(&self) -> &[f64] {
        &self.h
    }

    fn output_dim(&self) -> usize {
        self.d_in
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
    fn new_creates_correct_dimensions() {
        let ssm = SelectiveSSM::new(4, 8, 42);
        assert_eq!(ssm.d_in(), 4);
        assert_eq!(ssm.n_state(), 8);
        assert_eq!(ssm.state().len(), 4 * 8);
        assert_eq!(ssm.output_dim(), 4);
    }

    #[test]
    fn initial_state_is_zero() {
        let ssm = SelectiveSSM::new(3, 16, 42);
        for &h in ssm.state() {
            assert!(math::abs(h) < 1e-15, "initial state should be zero");
        }
    }

    #[test]
    fn forward_produces_correct_output_dim() {
        let mut ssm = SelectiveSSM::new(5, 8, 42);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = ssm.forward(&input);
        assert_eq!(output.len(), 5, "output dim should match d_in");
    }

    #[test]
    fn forward_produces_finite_output() {
        let mut ssm = SelectiveSSM::new(3, 8, 42);
        let input = vec![1.0, -1.0, 0.5];
        let output = ssm.forward(&input);
        for (i, &y) in output.iter().enumerate() {
            assert!(y.is_finite(), "output[{}] should be finite, got {}", i, y);
        }
    }

    #[test]
    fn forward_updates_state() {
        let mut ssm = SelectiveSSM::new(3, 8, 42);
        let input = vec![1.0, 2.0, 3.0];
        let _ = ssm.forward(&input);
        let state_norm: f64 = ssm.state().iter().map(|h| h * h).sum();
        assert!(
            state_norm > 0.0,
            "state should be non-zero after processing non-zero input"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut ssm = SelectiveSSM::new(3, 8, 42);
        let _ = ssm.forward(&[1.0, 2.0, 3.0]);
        ssm.reset();
        for &h in ssm.state() {
            assert!(math::abs(h) < 1e-15, "state should be zero after reset");
        }
    }

    #[test]
    fn state_decays_without_input() {
        let mut ssm = SelectiveSSM::new(2, 4, 42);
        // Inject state
        let _ = ssm.forward(&[10.0, 10.0]);
        let energy_after: f64 = ssm.state().iter().map(|h| h * h).sum();

        // Feed zeros for many steps
        for _ in 0..200 {
            let _ = ssm.forward(&[0.0, 0.0]);
        }
        let energy_decayed: f64 = ssm.state().iter().map(|h| h * h).sum();
        assert!(
            energy_decayed < energy_after * 0.01,
            "state should decay with zero input: initial={}, after={}",
            energy_after,
            energy_decayed
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut ssm1 = SelectiveSSM::new(3, 8, 42);
        let mut ssm2 = SelectiveSSM::new(3, 8, 42);
        let input = vec![1.0, 2.0, 3.0];
        let out1 = ssm1.forward(&input);
        let out2 = ssm2.forward(&input);
        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                math::abs(a - b) < 1e-15,
                "output[{}] should be identical for same seed: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn different_seeds_produce_different_outputs() {
        let mut ssm1 = SelectiveSSM::new(3, 8, 42);
        let mut ssm2 = SelectiveSSM::new(3, 8, 99);
        let input = vec![1.0, 2.0, 3.0];
        let out1 = ssm1.forward(&input);
        let out2 = ssm2.forward(&input);
        let diff: f64 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            diff > 1e-20,
            "different seeds should generally produce different outputs"
        );
    }

    #[test]
    fn single_channel_works() {
        let mut ssm = SelectiveSSM::new(1, 4, 42);
        let output = ssm.forward(&[3.0]);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }

    #[test]
    fn single_state_dim_works() {
        let mut ssm = SelectiveSSM::new(3, 1, 42);
        let output = ssm.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(output.len(), 3);
        for &y in &output {
            assert!(y.is_finite());
        }
    }

    #[test]
    fn sequential_outputs_differ() {
        let mut ssm = SelectiveSSM::new(2, 4, 42);
        let out1 = ssm.forward(&[1.0, 0.0]);
        let out2 = ssm.forward(&[1.0, 0.0]);
        // Second call has non-zero state from first call, so outputs should differ
        let diff: f64 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            diff > 1e-20,
            "sequential calls with same input should differ due to state: out1={:?}, out2={:?}",
            out1,
            out2
        );
    }

    #[test]
    fn large_input_no_overflow() {
        let mut ssm = SelectiveSSM::new(2, 4, 42);
        let input = vec![1000.0, -1000.0];
        let output = ssm.forward(&input);
        for (i, &y) in output.iter().enumerate() {
            assert!(
                y.is_finite(),
                "output[{}] should be finite for large inputs, got {}",
                i,
                y
            );
        }
    }

    #[test]
    fn zero_input_zero_state_gives_zero_output() {
        let mut ssm = SelectiveSSM::new(3, 8, 42);
        let output = ssm.forward(&[0.0, 0.0, 0.0]);
        for (i, &y) in output.iter().enumerate() {
            assert!(
                math::abs(y) < 1e-15,
                "zero input with zero state should give zero output[{}], got {}",
                i,
                y
            );
        }
    }
}
