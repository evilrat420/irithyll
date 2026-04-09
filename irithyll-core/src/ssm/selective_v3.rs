//! Mamba-3 selective state space model with MIMO groups and complex states.
//!
//! [`SelectiveSSMv3`] implements the Mamba-3 SSM from Gu & Dao (ICLR 2026),
//! which extends the original Mamba selective mechanism with:
//!
//! - **MIMO groups** -- channels are partitioned into groups that share state,
//!   enabling cross-channel information flow within each group while keeping
//!   computation tractable.
//! - **Complex hidden state** -- state elements are complex-valued (re/im pairs),
//!   enabling the model to capture oscillatory dynamics and periodic patterns.
//! - **Trapezoidal discretization** -- the bilinear (Tustin) transform extended
//!   to complex A, which preserves stability and has better frequency-domain
//!   properties than ZOH for oscillatory systems.
//!
//! # Architecture
//!
//! For each input timestep `x_t` (a d_in-dimensional vector):
//!
//! ```text
//! Delta_t = softplus(W_delta * x_t + b_delta)    // scalar step size
//! B_t     = W_B * x_t                             // N-dim input projection
//! C_t     = W_C * x_t                             // N-dim output projection
//! A       = -exp(log_A_re) + j * A_im             // complex diagonal, stable
//!
//! For each group g in 0..n_groups:
//!   x_group = mean(x_t[channels in group g])
//!   For each state dim n in 0..N:
//!     (A_bar, B_factor) = trapezoidal(A[n], Delta_t)     // complex
//!     h[g,n] = A_bar * h[g,n] + B_factor * B_t[n] * x_group   // complex
//!   y_group = sum_n( C_t[n] * Re(h[g,n]) )
//!   output[d] = y_group + D[d] * x_t[d]  for d in group g
//! ```

use alloc::vec;
use alloc::vec::Vec;

use crate::math;
use crate::ssm::discretize::trapezoidal_complex;
use crate::ssm::init::s4d_inv_complex;
use crate::ssm::projection::{dot, mat_vec, softplus, Xorshift64};
use crate::ssm::SSMLayer;

/// Mamba-3 selective state space model with MIMO groups and complex state.
///
/// Extends [`SelectiveSSM`](crate::ssm::SelectiveSSM) (Mamba-1) with grouped
/// channels and complex-valued hidden states for richer temporal modeling.
///
/// # Dimensions
///
/// - `d_in` -- input/output dimension (number of channels)
/// - `n_state` -- complex hidden state dimension per group (N)
/// - `n_groups` -- number of channel groups (must divide d_in evenly)
/// - Total hidden state size: `2 * n_groups * n_state` (re/im interleaved)
///
/// # Weight Shapes
///
/// | Weight | Shape | Purpose |
/// |--------|-------|---------|
/// | `log_a_complex` | 2*N | Complex A params (log\|re\|, im interleaved) |
/// | `w_delta` | d_in | Projects input to scalar step size |
/// | `w_b` | (G*N) x d_in | Per-group input projection (G groups, N state dims) |
/// | `w_c` | (G*N) x d_in | Per-group output projection (G groups, N state dims) |
/// | `d_skip` | d_in | Skip connection weights |
///
/// # Example
///
/// ```
/// use irithyll_core::ssm::selective_v3::SelectiveSSMv3;
/// use irithyll_core::ssm::SSMLayer;
///
/// let mut ssm = SelectiveSSMv3::new(4, 8, 2, 42);
/// let output = ssm.forward(&[1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(output.len(), 4);
/// ```
pub struct SelectiveSSMv3 {
    /// Complex log-A parameters (2*n_state: [log|re|, im, log|re|, im, ...]).
    /// Actual A_n = -exp(log_a_complex[2*n]) + j * log_a_complex[2*n+1].
    log_a_complex: Vec<f64>,
    /// Delta projection weights (d_in). Maps input to scalar step size.
    w_delta: Vec<f64>,
    /// Delta projection bias.
    b_delta: f64,
    /// Per-group B projection weights (n_groups * n_state x d_in, row-major).
    /// Group g's slice: rows [g*n_state .. (g+1)*n_state].
    w_b: Vec<f64>,
    /// Per-group C projection weights (n_groups * n_state x d_in, row-major).
    /// Group g's slice: rows [g*n_state .. (g+1)*n_state].
    w_c: Vec<f64>,
    /// Skip connection weights (d_in).
    d_skip: Vec<f64>,
    /// Complex hidden state (2 * n_groups * n_state: [re, im, ...] per group).
    h: Vec<f64>,
    /// Number of complex state dimensions per group.
    n_state: usize,
    /// Input/output dimension.
    d_in: usize,
    /// Number of channel groups.
    n_groups: usize,
}

impl SelectiveSSMv3 {
    /// Create a new Mamba-3 selective SSM with random weight initialization.
    ///
    /// Weights are initialized from a small normal distribution (scale 0.1)
    /// using the provided seed for reproducibility. Complex A is initialized
    /// via `s4d_inv_complex` which gives stable eigenvalues with negative real
    /// parts and oscillatory imaginary parts. Skip connections (D) are
    /// initialized to 1.0 to enable input passthrough by default.
    ///
    /// # Arguments
    ///
    /// * `d_in` -- input/output dimension (number of channels)
    /// * `n_state` -- complex hidden state dimension per group (N)
    /// * `n_groups` -- number of channel groups (must divide d_in evenly)
    /// * `seed` -- random seed for weight initialization
    ///
    /// # Panics
    ///
    /// Panics if `d_in` is not evenly divisible by `n_groups`.
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll_core::ssm::selective_v3::SelectiveSSMv3;
    ///
    /// let ssm = SelectiveSSMv3::new(6, 8, 3, 42);
    /// ```
    pub fn new(d_in: usize, n_state: usize, n_groups: usize, seed: u64) -> Self {
        assert!(
            d_in % n_groups == 0,
            "d_in ({}) must be evenly divisible by n_groups ({})",
            d_in,
            n_groups
        );

        let log_a_complex = s4d_inv_complex(n_state);
        let mut rng = Xorshift64(seed);
        let scale = 0.1;

        // Initialize projection weights from small normal distribution
        let w_delta: Vec<f64> = (0..d_in).map(|_| rng.next_normal() * scale).collect();
        let b_delta = 0.0;
        let w_b: Vec<f64> = (0..n_groups * n_state * d_in)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let w_c: Vec<f64> = (0..n_groups * n_state * d_in)
            .map(|_| rng.next_normal() * scale)
            .collect();
        let d_skip = vec![1.0; d_in];
        let h = vec![0.0; 2 * n_groups * n_state];

        Self {
            log_a_complex,
            w_delta,
            b_delta,
            w_b,
            w_c,
            d_skip,
            h,
            n_state,
            d_in,
            n_groups,
        }
    }

    /// Get the input/output dimension.
    #[inline]
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Get the number of complex state dimensions per group.
    #[inline]
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// Get the number of channel groups.
    #[inline]
    pub fn n_groups(&self) -> usize {
        self.n_groups
    }

    /// Compute the Mamba-3 forward pass for one timestep.
    ///
    /// This is the core MIMO recurrence: compute input-dependent Delta, then
    /// for each group compute per-group B_g and C_g projections, average the
    /// group's input channels, update complex state via trapezoidal
    /// discretization, and broadcast the output to all channels in the group.
    fn mimo_forward(&mut self, input: &[f64]) -> Vec<f64> {
        let d_in = self.d_in;
        let n_state = self.n_state;
        let n_groups = self.n_groups;
        let cpg = d_in / n_groups; // channels per group

        // 1. Compute delta = softplus(dot(w_delta, input) + b_delta)
        let delta_raw = dot(&self.w_delta, input) + self.b_delta;
        let delta = softplus(delta_raw);

        // 2. For each group, compute per-group B/C, update state, produce output
        let mut output = vec![0.0; d_in];

        for g in 0..n_groups {
            // Per-group B_t = W_B_g * input (shape: N)
            // W_B_g is rows [g*n_state .. (g+1)*n_state] of w_b
            let wb_offset = g * n_state * d_in;
            let mut b_t_g = vec![0.0; n_state];
            mat_vec(
                &self.w_b[wb_offset..wb_offset + n_state * d_in],
                input,
                n_state,
                d_in,
                &mut b_t_g,
            );

            // Per-group C_t = W_C_g * input (shape: N)
            let wc_offset = g * n_state * d_in;
            let mut c_t_g = vec![0.0; n_state];
            mat_vec(
                &self.w_c[wc_offset..wc_offset + n_state * d_in],
                input,
                n_state,
                d_in,
                &mut c_t_g,
            );

            // Average input across channels in this group
            let group_start = g * cpg;
            let mut x_group = 0.0;
            for d in 0..cpg {
                x_group += input[group_start + d];
            }
            x_group /= cpg as f64;

            let mut y_group = 0.0;

            for n in 0..n_state {
                // Recover complex A: A = -exp(log|re|) + j * im
                let a_re = -math::exp(self.log_a_complex[2 * n]);
                let a_im = self.log_a_complex[2 * n + 1];

                // Trapezoidal discretization for complex A
                let (a_bar_re, a_bar_im, b_fac_re, b_fac_im) =
                    trapezoidal_complex(a_re, a_im, delta);

                // B_t_g[n] and x_group are both real, so the input contribution is:
                // b_bar_input = b_factor * B_t_g[n] * x_group (complex * real * real)
                let bx = b_t_g[n] * x_group;
                let b_input_re = b_fac_re * bx;
                let b_input_im = b_fac_im * bx;

                // State index: h[(g * n_state + n) * 2] = re, +1 = im
                let h_idx = (g * n_state + n) * 2;
                let h_re_old = self.h[h_idx];
                let h_im_old = self.h[h_idx + 1];

                // Complex state update: h = A_bar * h + b_bar_input
                let h_re = a_bar_re * h_re_old - a_bar_im * h_im_old + b_input_re;
                let h_im = a_bar_re * h_im_old + a_bar_im * h_re_old + b_input_im;

                self.h[h_idx] = h_re;
                self.h[h_idx + 1] = h_im;

                // Output: C_t_g[n] is real, h is complex, so contribute Re(C_t_g[n] * h)
                // = C_t_g[n] * h_re (since C is real, only real part of h matters)
                y_group += c_t_g[n] * h_re;
            }

            // Broadcast y_group to all channels in the group, adding skip connection
            for d in 0..cpg {
                let idx = group_start + d;
                output[idx] = y_group + self.d_skip[idx] * input[idx];
            }
        }

        output
    }
}

impl SSMLayer for SelectiveSSMv3 {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            input.len(),
            self.d_in,
            "input length {} must match d_in {}",
            input.len(),
            self.d_in
        );
        self.mimo_forward(input)
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
    fn selective_v3_output_dimension() {
        let mut ssm = SelectiveSSMv3::new(6, 8, 2, 42);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = ssm.forward(&input);
        assert_eq!(
            output.len(),
            6,
            "output dim should match d_in, got {}",
            output.len()
        );
    }

    #[test]
    fn selective_v3_complex_state_bounded() {
        let mut ssm = SelectiveSSMv3::new(4, 8, 2, 42);
        let input = vec![1.0, -0.5, 0.3, -0.8];
        for step in 0..1000 {
            let output = ssm.forward(&input);
            for (i, &y) in output.iter().enumerate() {
                assert!(
                    y.is_finite(),
                    "output[{}] is not finite at step {}: {}",
                    i,
                    step,
                    y
                );
            }
        }
        // Verify state has no NaN/Inf
        for (i, &h) in ssm.state().iter().enumerate() {
            assert!(
                h.is_finite(),
                "state[{}] is not finite after 1000 steps: {}",
                i,
                h
            );
        }
        // Verify state norm is bounded (not exploding)
        let state_norm: f64 = ssm.state().iter().map(|h| h * h).sum();
        assert!(
            state_norm < 1e12,
            "state norm should be bounded, got {}",
            state_norm
        );
    }

    #[test]
    fn selective_v3_trapezoidal_stability() {
        // Verify that complex eigenvalues after trapezoidal discretization
        // stay inside the unit disk (|A_bar| < 1 for stable continuous A)
        let log_a = s4d_inv_complex(16);
        let delta = 0.5; // moderate step size
        for n in 0..16 {
            let a_re = -math::exp(log_a[2 * n]);
            let a_im = log_a[2 * n + 1];
            let (a_bar_re, a_bar_im, _, _) = trapezoidal_complex(a_re, a_im, delta);
            let mag_sq = a_bar_re * a_bar_re + a_bar_im * a_bar_im;
            assert!(
                mag_sq < 1.0,
                "eigenvalue {} has |A_bar|^2 = {} >= 1 (a_re={}, a_im={}, delta={})",
                n,
                mag_sq,
                a_re,
                a_im,
                delta
            );
        }
    }

    #[test]
    fn selective_v3_mimo_groups() {
        // n_groups=1 (all channels share one state) vs n_groups=d_in (each channel own state)
        // Both should produce valid but different outputs
        let d_in = 4;
        let n_state = 4;
        let seed = 42;

        let mut ssm_one = SelectiveSSMv3::new(d_in, n_state, 1, seed);
        let mut ssm_max = SelectiveSSMv3::new(d_in, n_state, d_in, seed);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out_one = ssm_one.forward(&input);
        let out_max = ssm_max.forward(&input);

        // Both should have correct dimensions
        assert_eq!(out_one.len(), d_in);
        assert_eq!(out_max.len(), d_in);

        // Both should be finite
        for &y in &out_one {
            assert!(y.is_finite(), "n_groups=1 output should be finite");
        }
        for &y in &out_max {
            assert!(y.is_finite(), "n_groups=d_in output should be finite");
        }

        // They should differ because group averaging differs
        let diff: f64 = out_one
            .iter()
            .zip(out_max.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            diff > 1e-20,
            "different n_groups should produce different outputs: diff={}",
            diff
        );
    }

    #[test]
    fn selective_v3_reset_clears_state() {
        let mut ssm = SelectiveSSMv3::new(4, 8, 2, 42);
        let _ = ssm.forward(&[1.0, 2.0, 3.0, 4.0]);

        // State should be non-zero after processing input
        let energy: f64 = ssm.state().iter().map(|h| h * h).sum();
        assert!(energy > 0.0, "state should be non-zero after forward pass");

        ssm.reset();
        for (i, &h) in ssm.state().iter().enumerate() {
            assert!(
                math::abs(h) < 1e-15,
                "state[{}] should be zero after reset, got {}",
                i,
                h
            );
        }
    }

    #[test]
    fn selective_v3_initial_state_zero() {
        let ssm = SelectiveSSMv3::new(4, 8, 2, 42);
        assert_eq!(
            ssm.state().len(),
            2 * 2 * 8,
            "state size = 2 * n_groups * n_state"
        );
        for &h in ssm.state() {
            assert!(math::abs(h) < 1e-15, "initial state should be zero");
        }
    }

    #[test]
    fn selective_v3_deterministic_same_seed() {
        let mut ssm1 = SelectiveSSMv3::new(4, 8, 2, 42);
        let mut ssm2 = SelectiveSSMv3::new(4, 8, 2, 42);
        let input = vec![1.0, -1.0, 0.5, -0.5];
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
    fn selective_v3_zero_input_zero_output() {
        let mut ssm = SelectiveSSMv3::new(4, 8, 2, 42);
        let output = ssm.forward(&[0.0, 0.0, 0.0, 0.0]);
        for (i, &y) in output.iter().enumerate() {
            assert!(
                math::abs(y) < 1e-15,
                "zero input with zero state should give zero output[{}], got {}",
                i,
                y
            );
        }
    }

    #[test]
    fn selective_v3_single_group() {
        // d_in == n_groups: each channel is its own group
        let mut ssm = SelectiveSSMv3::new(3, 4, 3, 42);
        let output = ssm.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(output.len(), 3);
        for &y in &output {
            assert!(y.is_finite());
        }
    }

    #[test]
    fn selective_v3_sequential_outputs_differ() {
        let mut ssm = SelectiveSSMv3::new(4, 8, 2, 42);
        let input = vec![1.0, 0.0, -1.0, 0.5];
        let out1 = ssm.forward(&input);
        let out2 = ssm.forward(&input);
        let diff: f64 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            diff > 1e-20,
            "sequential calls should differ due to state: out1={:?}, out2={:?}",
            out1,
            out2
        );
    }

    #[test]
    fn selective_v3_accessors() {
        let ssm = SelectiveSSMv3::new(6, 4, 3, 42);
        assert_eq!(ssm.d_in(), 6);
        assert_eq!(ssm.n_state(), 4);
        assert_eq!(ssm.n_groups(), 3);
        assert_eq!(ssm.output_dim(), 6);
    }
}
