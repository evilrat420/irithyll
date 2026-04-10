//! Block-Diagonal Linear Recurrent Unit (BD-LRU) selective state space model.
//!
//! [`SelectiveSSMBD`] implements a block-diagonal SSM variant inspired by
//! Dubinin et al. (2026), where input channels are partitioned into blocks and
//! each block has a dense A matrix enabling cross-channel state mixing within
//! the block. This sits between the fully diagonal Mamba-1 (no cross-channel
//! mixing) and a full dense SSM (quadratic cost).
//!
//! # Architecture
//!
//! For each input timestep `x_t` (a d_in-dimensional vector):
//!
//! ```text
//! Delta_t = softplus(W_delta * x_t + b_delta)    // scalar step size
//! B_t     = W_B * x_t                             // N-dim input projection
//! C_t     = W_C * x_t                             // N-dim output projection
//!
//! For each block k (channels k*m .. (k+1)*m, m = block_size):
//!   x_block = x_t[k*m .. (k+1)*m]
//!   For each state dim n in 0..N:
//!     // Euler discretization with row-L1-normalized dense A:
//!     // A_disc[i,j] = delta * A[i,j]  for i != j
//!     // A_disc[i,i] = 1 + delta * A[i,i]
//!     h_block[n] = A_disc * h_block[n] + delta * B_t[n] * x_block
//!
//!   // Output: weighted sum over state dims
//!   y_block = sum_n C_t[n] * h_block[n]
//!
//! output[d] = y_block[d_within_block] + D[d] * x_t[d]
//! ```
//!
//! # Block-Diagonal vs Diagonal
//!
//! The key differentiator from [`SelectiveSSM`](crate::ssm::SelectiveSSM) is
//! that channels within a block can influence each other's state evolution
//! through the off-diagonal entries of the block's A matrix. With `block_size=1`,
//! this reduces to a per-channel diagonal recurrence (equivalent to Mamba-1).
//! Larger block sizes enable richer cross-channel dynamics at O(m^2) cost per
//! block instead of O(d_in^2) for a full dense A.
//!
//! # Stability
//!
//! Each block's A matrix is row-wise L1-normalized so that the sum of absolute
//! values in each row is at most 1.0. Combined with Euler discretization
//! (`I + Delta * A`), this ensures the discretized transition matrix has
//! bounded spectral radius for small Delta, preventing state explosion.

use alloc::vec;
use alloc::vec::Vec;

use crate::math;
use crate::ssm::init::s4d_inv_real;
use crate::ssm::projection::{dot, mat_vec, softplus, Xorshift64};
use crate::ssm::SSMLayer;

/// Block-Diagonal Linear Recurrent Unit selective state space model.
///
/// Partitions `d_in` channels into `n_blocks = d_in / block_size` blocks, each
/// with a dense `block_size x block_size` A matrix for within-block
/// cross-channel state mixing. B, C, and Delta projections are shared across
/// blocks (same structure as Mamba-1).
///
/// # Dimensions
///
/// - `d_in` -- input/output dimension (number of channels)
/// - `n_state` -- hidden state dimension per block-channel (N)
/// - `block_size` -- number of channels per block (m)
/// - `n_blocks` -- number of blocks (d_in / block_size)
/// - Total hidden state size: `n_blocks * n_state * block_size`
///
/// # Weight Shapes
///
/// | Weight | Shape | Purpose |
/// |--------|-------|---------|
/// | `a_matrices` | n_blocks * m * m | Dense A per block (row-major, L1-normalized) |
/// | `w_b` | N x d_in | Projects input to state-input coupling |
/// | `w_c` | N x d_in | Projects input to state-output coupling |
/// | `w_delta` | d_in | Projects input to scalar step size |
/// | `d_skip` | d_in | Skip connection weights |
///
/// # Example
///
/// ```
/// use irithyll_core::ssm::selective_bd::SelectiveSSMBD;
/// use irithyll_core::ssm::SSMLayer;
///
/// let mut ssm = SelectiveSSMBD::new(4, 8, 2, 42);
/// let output = ssm.forward(&[1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(output.len(), 4);
/// ```
pub struct SelectiveSSMBD {
    /// Per-block A matrices: n_blocks * block_size * block_size, row-major per block.
    /// Each block's m x m matrix is contiguous, L1-row-normalized for stability.
    a_matrices: Vec<f64>,
    /// B projection weights (n_state x d_in, row-major). Maps input to B_t.
    w_b: Vec<f64>,
    /// C projection weights (n_state x d_in, row-major). Maps input to C_t.
    w_c: Vec<f64>,
    /// Delta projection weights (d_in). Maps input to scalar step size.
    w_delta: Vec<f64>,
    /// Delta projection bias.
    b_delta: f64,
    /// Skip connection weights (d_in).
    d_skip: Vec<f64>,
    /// Hidden state: n_blocks * n_state * block_size.
    /// Layout: h[block * n_state * block_size + state_dim * block_size + channel_within_block]
    h: Vec<f64>,
    /// Input/output dimension.
    d_in: usize,
    /// Number of state dimensions per block-channel.
    n_state: usize,
    /// Number of channels per block.
    block_size: usize,
    /// Number of blocks (d_in / block_size).
    n_blocks: usize,
}

/// Normalize each row of an m x m matrix in-place so that the L1 norm
/// (sum of absolute values) of each row is at most 1.0.
///
/// Rows with L1 norm already <= 1.0 are left unchanged.
fn normalize_row_l1(a: &mut [f64], m: usize) {
    for row in 0..m {
        let start = row * m;
        let row_sum: f64 = a[start..start + m].iter().map(|x| math::abs(*x)).sum();
        if row_sum > 1.0 {
            for j in 0..m {
                a[start + j] /= row_sum;
            }
        }
    }
}

impl SelectiveSSMBD {
    /// Create a new block-diagonal selective SSM with random weight initialization.
    ///
    /// A matrices are initialized with S4D-Inv diagonal values and small random
    /// off-diagonal entries (scale 0.02), then row-wise L1-normalized. Projection
    /// weights are initialized from a small normal distribution (scale 0.1).
    /// Skip connections (D) are initialized to 1.0 for input passthrough.
    ///
    /// # Arguments
    ///
    /// * `d_in` -- input/output dimension (must be divisible by `block_size`)
    /// * `n_state` -- hidden state dimension per block-channel (N)
    /// * `block_size` -- number of channels per block (m)
    /// * `seed` -- random seed for weight initialization
    ///
    /// # Panics
    ///
    /// Panics if `d_in` is not evenly divisible by `block_size`.
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll_core::ssm::selective_bd::SelectiveSSMBD;
    ///
    /// let ssm = SelectiveSSMBD::new(6, 8, 2, 42);
    /// ```
    pub fn new(d_in: usize, n_state: usize, block_size: usize, seed: u64) -> Self {
        assert!(
            d_in % block_size == 0,
            "d_in ({}) must be evenly divisible by block_size ({})",
            d_in,
            block_size
        );

        let n_blocks = d_in / block_size;
        let m = block_size;
        let mut rng = Xorshift64(seed);
        let scale = 0.1;
        let off_diag_scale = 0.02;

        // Initialize A matrices: S4D-Inv diagonal + small random off-diagonal
        let log_a = s4d_inv_real(m);
        let mut a_matrices = vec![0.0; n_blocks * m * m];

        for blk in 0..n_blocks {
            let base = blk * m * m;
            // Fill with small random off-diagonal values
            for i in 0..m {
                for j in 0..m {
                    if i == j {
                        // Diagonal: negative S4D-Inv values
                        // A_i = -(0.5 + i/m), use directly (not log-space here)
                        a_matrices[base + i * m + j] = -math::exp(log_a[i]);
                    } else {
                        // Off-diagonal: small random normal
                        a_matrices[base + i * m + j] = rng.next_normal() * off_diag_scale;
                    }
                }
            }
            // Apply row-wise L1 normalization for stability
            normalize_row_l1(&mut a_matrices[base..base + m * m], m);
        }

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
        let h = vec![0.0; n_blocks * n_state * block_size];

        Self {
            a_matrices,
            w_b,
            w_c,
            w_delta,
            b_delta,
            d_skip,
            h,
            d_in,
            n_state,
            block_size,
            n_blocks,
        }
    }

    /// Get the input/output dimension.
    #[inline]
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Get the number of state dimensions per block-channel.
    #[inline]
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// Get the number of channels per block.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get the number of blocks.
    #[inline]
    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Compute the block-diagonal SSM forward pass for one timestep.
    ///
    /// This is the core BD-LRU recurrence: compute input-dependent Delta, B, C,
    /// then for each block apply the dense A state update with Euler
    /// discretization and accumulate the output.
    fn bd_forward(&mut self, input: &[f64]) -> Vec<f64> {
        let d_in = self.d_in;
        let n_state = self.n_state;
        let m = self.block_size;
        let n_blocks = self.n_blocks;

        // 1. Compute delta = softplus(dot(w_delta, input) + b_delta)
        let delta_raw = dot(&self.w_delta, input) + self.b_delta;
        let delta = softplus(delta_raw);

        // 2. Compute B_t = W_B * input (shape: n_state)
        let mut b_t = vec![0.0; n_state];
        mat_vec(&self.w_b, input, n_state, d_in, &mut b_t);

        // 3. Compute C_t = W_C * input (shape: n_state)
        let mut c_t = vec![0.0; n_state];
        mat_vec(&self.w_c, input, n_state, d_in, &mut c_t);

        // 4. For each block: apply dense A state update
        let mut output = vec![0.0; d_in];

        for blk in 0..n_blocks {
            let a_base = blk * m * m;
            let x_start = blk * m;
            let h_block_base = blk * n_state * m;

            for (n, &b_n) in b_t.iter().enumerate().take(n_state) {
                let h_offset = h_block_base + n * m;

                // Apply block state update with Euler discretization:
                // h_new[i] = sum_j(A_disc[i,j] * h_old[j]) + delta * B_t[n] * x_block[i]
                // where A_disc[i,j] = delta * A[i,j] for i != j
                //       A_disc[i,i] = 1 + delta * A[i,i]
                //
                // We compute h_new into a temp buffer to avoid reading stale values.
                let db = delta * b_n;

                // Temporary buffer for new state (avoid allocation for small blocks
                // by using a stack array would be nice, but we need Vec for generality)
                let mut h_new = vec![0.0; m];

                for i in 0..m {
                    let a_row = a_base + i * m;
                    let mut sum = 0.0;
                    for j in 0..m {
                        let a_disc = if i == j {
                            1.0 + delta * self.a_matrices[a_row + j]
                        } else {
                            delta * self.a_matrices[a_row + j]
                        };
                        sum += a_disc * self.h[h_offset + j];
                    }
                    // Input injection: delta * B_t[n] * x_block[i]
                    h_new[i] = sum + db * input[x_start + i];
                }

                // Write back new state
                self.h[h_offset..h_offset + m].copy_from_slice(&h_new);
            }

            // 5. Output accumulation: y_block[i] = sum_n C_t[n] * h[block, n, i]
            for (n, &c_n) in c_t.iter().enumerate().take(n_state) {
                let h_offset = h_block_base + n * m;
                for i in 0..m {
                    output[x_start + i] += c_n * self.h[h_offset + i];
                }
            }
        }

        // 6. Add skip connection: output[d] += D[d] * input[d]
        for (out_d, (&skip, &x_d)) in output.iter_mut().zip(self.d_skip.iter().zip(input.iter())) {
            *out_d += skip * x_d;
        }

        output
    }
}

impl SSMLayer for SelectiveSSMBD {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            input.len(),
            self.d_in,
            "input length {} must match d_in {}",
            input.len(),
            self.d_in
        );
        self.bd_forward(input)
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
    fn bd_new_correct_dimensions() {
        let ssm = SelectiveSSMBD::new(6, 8, 2, 42);
        assert_eq!(ssm.d_in(), 6);
        assert_eq!(ssm.n_state(), 8);
        assert_eq!(ssm.block_size(), 2);
        assert_eq!(ssm.n_blocks(), 3);
        assert_eq!(
            ssm.state().len(),
            3 * 8 * 2,
            "state size = n_blocks * n_state * block_size"
        );
        assert_eq!(ssm.output_dim(), 6);
    }

    #[test]
    fn bd_initial_state_zero() {
        let ssm = SelectiveSSMBD::new(4, 8, 2, 42);
        for &h in ssm.state() {
            assert!(math::abs(h) < 1e-15, "initial state should be zero");
        }
    }

    #[test]
    fn bd_forward_correct_output_dim() {
        let mut ssm = SelectiveSSMBD::new(6, 8, 3, 42);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = ssm.forward(&input);
        assert_eq!(output.len(), 6, "output dim should match d_in");
    }

    #[test]
    fn bd_forward_finite_output() {
        let mut ssm = SelectiveSSMBD::new(4, 8, 2, 42);
        let input = vec![1.0, -1.0, 0.5, -0.5];
        let output = ssm.forward(&input);
        for (i, &y) in output.iter().enumerate() {
            assert!(y.is_finite(), "output[{}] should be finite, got {}", i, y);
        }
    }

    #[test]
    fn bd_forward_updates_state() {
        let mut ssm = SelectiveSSMBD::new(4, 8, 2, 42);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let _ = ssm.forward(&input);
        let state_norm: f64 = ssm.state().iter().map(|h| h * h).sum();
        assert!(
            state_norm > 0.0,
            "state should be non-zero after processing non-zero input"
        );
    }

    #[test]
    fn bd_reset_clears_state() {
        let mut ssm = SelectiveSSMBD::new(4, 8, 2, 42);
        let _ = ssm.forward(&[1.0, 2.0, 3.0, 4.0]);
        ssm.reset();
        for &h in ssm.state() {
            assert!(math::abs(h) < 1e-15, "state should be zero after reset");
        }
    }

    #[test]
    fn bd_deterministic_same_seed() {
        let mut ssm1 = SelectiveSSMBD::new(4, 8, 2, 42);
        let mut ssm2 = SelectiveSSMBD::new(4, 8, 2, 42);
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
    fn bd_different_seeds_differ() {
        let mut ssm1 = SelectiveSSMBD::new(4, 8, 2, 42);
        let mut ssm2 = SelectiveSSMBD::new(4, 8, 2, 99);
        let input = vec![1.0, 2.0, 3.0, 4.0];
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
    fn bd_zero_input_zero_state_zero_output() {
        let mut ssm = SelectiveSSMBD::new(4, 8, 2, 42);
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
    fn bd_cross_channel_mixing() {
        // With block_size > 1, off-diagonal A entries cause cross-channel mixing
        // within each block. With block_size=1, there are no off-diagonal entries,
        // so channels evolve independently. Verify the two produce different results.
        let d_in = 4;
        let n_state = 4;
        let seed = 42;

        let mut ssm_blk1 = SelectiveSSMBD::new(d_in, n_state, 1, seed);
        let mut ssm_blk2 = SelectiveSSMBD::new(d_in, n_state, 2, seed);

        let input = vec![1.0, 2.0, 3.0, 4.0];

        // Run a few steps to accumulate state differences from cross-channel mixing
        for _ in 0..5 {
            let _ = ssm_blk1.forward(&input);
            let _ = ssm_blk2.forward(&input);
        }

        let out1 = ssm_blk1.forward(&input);
        let out2 = ssm_blk2.forward(&input);

        // Both should be valid
        for &y in &out1 {
            assert!(y.is_finite(), "block_size=1 output should be finite");
        }
        for &y in &out2 {
            assert!(y.is_finite(), "block_size=2 output should be finite");
        }

        // They should differ because block_size=2 has cross-channel mixing
        let diff: f64 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            diff > 1e-20,
            "block_size=1 vs block_size=2 should produce different outputs due to cross-channel mixing: diff={}",
            diff
        );
    }

    #[test]
    fn bd_state_bounded_under_constant_input() {
        let mut ssm = SelectiveSSMBD::new(4, 8, 2, 42);
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
            "state norm should be bounded after 1000 constant-input steps, got {}",
            state_norm
        );
    }

    #[test]
    fn bd_block_sizes_produce_different_outputs() {
        // block_size=2 vs block_size=4 should produce different outputs
        // because the A block structure differs
        let d_in = 8;
        let n_state = 4;
        let seed = 42;

        let mut ssm_bs2 = SelectiveSSMBD::new(d_in, n_state, 2, seed);
        let mut ssm_bs4 = SelectiveSSMBD::new(d_in, n_state, 4, seed);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Run a few steps
        for _ in 0..5 {
            let _ = ssm_bs2.forward(&input);
            let _ = ssm_bs4.forward(&input);
        }

        let out_bs2 = ssm_bs2.forward(&input);
        let out_bs4 = ssm_bs4.forward(&input);

        assert_eq!(out_bs2.len(), d_in);
        assert_eq!(out_bs4.len(), d_in);

        for &y in &out_bs2 {
            assert!(y.is_finite(), "block_size=2 output should be finite");
        }
        for &y in &out_bs4 {
            assert!(y.is_finite(), "block_size=4 output should be finite");
        }

        let diff: f64 = out_bs2
            .iter()
            .zip(out_bs4.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        assert!(
            diff > 1e-20,
            "block_size=2 vs block_size=4 should produce different outputs: diff={}",
            diff
        );
    }
}
