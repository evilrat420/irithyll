//! MinGRU cell -- the simplest possible gated recurrence.
//!
//! The minGRU (Feng et al., 2024) removes the recurrent connection from
//! the candidate computation, yielding a drastically simplified update:
//!
//! ```text
//! z_t     = sigmoid(W_z * [x_t, h_{t-1}] + b_z)     // update gate
//! h_tilde = W_h * x_t + b_h                           // candidate (NO recurrence)
//! h_t     = (1 - z_t) * h_{t-1} + z_t * h_tilde      // interpolate
//! ```
//!
//! Weights are lazily initialized on the first call to [`MinGRUCell::forward`]
//! when the input dimension becomes known. Xavier normal initialization is used.

use alloc::vec;
use alloc::vec::Vec;

use crate::math;
use crate::rng::standard_normal;

/// MinGRU cell with lazy initialization and stateful hidden state.
///
/// # Example
///
/// ```
/// use irithyll_core::mgrade::MinGRUCell;
///
/// let mut cell = MinGRUCell::new(8, 42);
/// let input = [0.1, -0.2, 0.3, 0.4];
/// let h = cell.forward(&input);
/// assert_eq!(h.len(), 8);
/// ```
pub struct MinGRUCell {
    /// Gate weight matrix: [d_hidden x (d_input + d_hidden)] row-major.
    w_z: Vec<f64>,
    /// Gate bias: [d_hidden].
    b_z: Vec<f64>,
    /// Candidate weight matrix: [d_hidden x d_input] row-major (NOT recurrent).
    w_h: Vec<f64>,
    /// Candidate bias: [d_hidden].
    b_h: Vec<f64>,
    /// Hidden state: [d_hidden].
    h: Vec<f64>,
    d_hidden: usize,
    d_input: usize,
    seed: u64,
}

impl MinGRUCell {
    /// Create a new MinGRU cell with the given hidden dimension.
    ///
    /// Weights are not allocated until the first call to [`forward`](MinGRUCell::forward),
    /// when the input dimension is inferred from the input slice length.
    ///
    /// # Arguments
    ///
    /// * `d_hidden` -- number of hidden units
    /// * `seed` -- RNG seed for deterministic weight initialization
    pub fn new(d_hidden: usize, seed: u64) -> Self {
        Self {
            w_z: Vec::new(),
            b_z: Vec::new(),
            w_h: Vec::new(),
            b_h: Vec::new(),
            h: vec![0.0; d_hidden],
            d_hidden,
            d_input: 0,
            seed,
        }
    }

    /// Lazily initialize weight matrices when the input dimension is first known.
    ///
    /// Uses Xavier initialization: `standard_normal * sqrt(2 / (fan_in + fan_out))`.
    fn ensure_initialized(&mut self, d_input: usize) {
        if self.d_input != 0 {
            return;
        }
        self.d_input = d_input;
        let d_total = d_input + self.d_hidden;

        // Gate weights: fan_in = d_total, fan_out = d_hidden
        let scale_z = math::sqrt(2.0 / (d_total + self.d_hidden) as f64);
        let n_gate = self.d_hidden * d_total;
        self.w_z = (0..n_gate)
            .map(|_| standard_normal(&mut self.seed) * scale_z)
            .collect();
        self.b_z = vec![0.0; self.d_hidden];

        // Candidate weights: fan_in = d_input, fan_out = d_hidden
        let scale_h = math::sqrt(2.0 / (d_input + self.d_hidden) as f64);
        let n_cand = self.d_hidden * d_input;
        self.w_h = (0..n_cand)
            .map(|_| standard_normal(&mut self.seed) * scale_h)
            .collect();
        self.b_h = vec![0.0; self.d_hidden];
    }

    /// Process one input timestep, updating internal state and returning a
    /// reference to the new hidden state.
    ///
    /// On the first call, weights are lazily initialized from `x.len()`.
    ///
    /// # Arguments
    ///
    /// * `x` -- input feature vector of length `d_input`
    ///
    /// # Returns
    ///
    /// Reference to the hidden state `h` (length `d_hidden`).
    pub fn forward(&mut self, x: &[f64]) -> &[f64] {
        self.ensure_initialized(x.len());
        let d_h = self.d_hidden;
        let d_in = self.d_input;
        let d_total = d_in + d_h;

        // 1. Build xh = [x, h_{t-1}]
        let mut xh = vec![0.0; d_total];
        xh[..d_in].copy_from_slice(x);
        xh[d_in..].copy_from_slice(&self.h);

        // 2. Compute gate: z = sigmoid(W_z * xh + b_z)
        let mut z = vec![0.0; d_h];
        crate::simd::simd_mat_vec(&self.w_z, &xh, d_h, d_total, &mut z);
        for (zi, bi) in z.iter_mut().zip(self.b_z.iter()) {
            *zi = math::sigmoid(*zi + bi);
        }

        // 3. Compute candidate: h_tilde = W_h * x + b_h (NO recurrence)
        let mut h_tilde = vec![0.0; d_h];
        crate::simd::simd_mat_vec(&self.w_h, x, d_h, d_in, &mut h_tilde);
        for (hi, bi) in h_tilde.iter_mut().zip(self.b_h.iter()) {
            *hi += bi;
        }

        // 4. Interpolate: h_t = (1 - z) * h_{t-1} + z * h_tilde
        for ((hj, zj), htj) in self.h.iter_mut().zip(z.iter()).zip(h_tilde.iter()) {
            *hj = (1.0 - zj) * *hj + zj * htj;
        }

        &self.h
    }

    /// Compute what the hidden state would be after processing `x`, without
    /// mutating any internal state.
    ///
    /// # Panics
    ///
    /// Panics if called before the cell has been initialized (i.e., before any
    /// call to [`forward`](MinGRUCell::forward)).
    pub fn forward_predict(&self, x: &[f64]) -> Vec<f64> {
        assert!(
            self.d_input != 0,
            "forward_predict called before initialization; call forward() first"
        );
        let d_h = self.d_hidden;
        let d_in = self.d_input;
        let d_total = d_in + d_h;

        // 1. Build xh = [x, h_{t-1}]
        let mut xh = vec![0.0; d_total];
        xh[..d_in].copy_from_slice(x);
        xh[d_in..].copy_from_slice(&self.h);

        // 2. Gate
        let mut z = vec![0.0; d_h];
        crate::simd::simd_mat_vec(&self.w_z, &xh, d_h, d_total, &mut z);
        for (zi, bi) in z.iter_mut().zip(self.b_z.iter()) {
            *zi = math::sigmoid(*zi + bi);
        }

        // 3. Candidate
        let mut h_tilde = vec![0.0; d_h];
        crate::simd::simd_mat_vec(&self.w_h, x, d_h, d_in, &mut h_tilde);
        for (hi, bi) in h_tilde.iter_mut().zip(self.b_h.iter()) {
            *hi += bi;
        }

        // 4. Interpolate
        let h_out: Vec<f64> = self
            .h
            .iter()
            .zip(z.iter())
            .zip(h_tilde.iter())
            .map(|((hj, zj), htj)| (1.0 - zj) * hj + zj * htj)
            .collect();

        h_out
    }

    /// Reference to the current hidden state vector.
    #[inline]
    pub fn state(&self) -> &[f64] {
        &self.h
    }

    /// Reset the hidden state to zeros, preserving learned weights.
    pub fn reset(&mut self) {
        self.h.fill(0.0);
    }

    /// Number of hidden units in this cell.
    #[inline]
    pub fn d_hidden(&self) -> usize {
        self.d_hidden
    }

    /// Output dimension (equal to `d_hidden`).
    #[inline]
    pub fn output_dim(&self) -> usize {
        self.d_hidden
    }

    /// Whether the cell has been initialized (i.e., first forward has been called).
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.d_input != 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_gru_cell_new() {
        let cell = MinGRUCell::new(16, 42);
        assert_eq!(cell.d_hidden(), 16, "d_hidden should match constructor arg");
        assert_eq!(cell.output_dim(), 16, "output_dim should equal d_hidden");
        assert!(
            !cell.is_initialized(),
            "cell should not be initialized before first forward"
        );
        assert_eq!(
            cell.state().len(),
            16,
            "hidden state should be pre-allocated to d_hidden"
        );
    }

    #[test]
    fn min_gru_cell_forward_initializes() {
        let mut cell = MinGRUCell::new(8, 42);
        assert!(!cell.is_initialized(), "should start uninitialized");

        let x = [0.1, -0.2, 0.3, 0.4];
        let h_len = cell.forward(&x).len();

        assert!(
            cell.is_initialized(),
            "should be initialized after first forward"
        );
        assert_eq!(h_len, 8, "output length should be d_hidden");
        assert_eq!(
            cell.d_input, 4,
            "d_input should be inferred from input length"
        );
        // Verify weight matrices were allocated
        assert_eq!(
            cell.w_z.len(),
            8 * (4 + 8),
            "w_z should have d_hidden * d_total elements"
        );
        assert_eq!(
            cell.w_h.len(),
            8 * 4,
            "w_h should have d_hidden * d_input elements (no recurrence)"
        );
    }

    #[test]
    fn min_gru_cell_forward_finite() {
        let mut cell = MinGRUCell::new(8, 123);
        let x = [1.0, -0.5, 0.3, 2.0, -1.0];
        let h = cell.forward(&x);

        for (i, &val) in h.iter().enumerate() {
            assert!(
                val.is_finite(),
                "h[{}] = {} should be finite after forward",
                i,
                val
            );
        }
    }

    #[test]
    fn min_gru_cell_forward_predict_no_state_change() {
        let mut cell = MinGRUCell::new(4, 99);
        let x = [0.5, -0.3, 0.8];

        // Run one forward to initialize
        cell.forward(&x);

        // Snapshot state before forward_predict
        let h_before = cell.h.clone();

        let x2 = [0.1, 0.2, -0.4];
        let _h_predict = cell.forward_predict(&x2);

        assert_eq!(
            cell.h, h_before,
            "hidden state should not change after forward_predict"
        );
    }

    #[test]
    fn min_gru_cell_reset() {
        let mut cell = MinGRUCell::new(4, 77);
        let x = [1.0, -1.0];

        // Run a few steps to build up state
        for _ in 0..5 {
            cell.forward(&x);
        }

        // Snapshot weights before reset
        let w_z_before = cell.w_z.clone();
        let w_h_before = cell.w_h.clone();

        cell.reset();

        // State should be zeroed
        assert!(
            cell.h.iter().all(|&v| v == 0.0),
            "h should be all zeros after reset"
        );

        // Weights should be preserved
        assert_eq!(
            cell.w_z, w_z_before,
            "w_z weights should be preserved after reset"
        );
        assert_eq!(
            cell.w_h, w_h_before,
            "w_h weights should be preserved after reset"
        );
    }

    #[test]
    fn min_gru_cell_sequence_evolves_state() {
        let mut cell = MinGRUCell::new(4, 42);
        let x = [0.5, -0.3, 0.8];

        let h1 = cell.forward(&x).to_vec();
        let h2 = cell.forward(&x).to_vec();
        let h3 = cell.forward(&x).to_vec();

        // After multiple steps with the same input, hidden state should differ
        // because of the recurrent gate connection.
        assert_ne!(
            h1, h2,
            "hidden state should evolve between step 1 and step 2"
        );
        assert_ne!(
            h2, h3,
            "hidden state should evolve between step 2 and step 3"
        );
    }

    #[test]
    fn min_gru_cell_candidate_has_no_recurrence() {
        // Verify that w_h dimensions are [d_hidden x d_input], NOT [d_hidden x d_total].
        // This is the key difference from standard GRU.
        let mut cell = MinGRUCell::new(8, 42);
        let x = [1.0, 2.0, 3.0];
        cell.forward(&x);

        assert_eq!(
            cell.w_h.len(),
            8 * 3,
            "candidate weights should be d_hidden * d_input (no recurrence), not d_hidden * d_total"
        );
        assert_eq!(
            cell.w_z.len(),
            8 * (3 + 8),
            "gate weights should include recurrent connection: d_hidden * (d_input + d_hidden)"
        );
    }

    #[test]
    fn min_gru_cell_forward_predict_matches_forward() {
        let mut cell = MinGRUCell::new(4, 42);
        let x1 = [0.5, -0.3, 0.8];
        cell.forward(&x1);

        let x2 = [0.1, 0.2, -0.4];
        let h_predict = cell.forward_predict(&x2);
        let h_forward = cell.forward(&x2).to_vec();

        for (i, (p, f)) in h_predict.iter().zip(h_forward.iter()).enumerate() {
            assert!(
                (p - f).abs() < 1e-12,
                "forward_predict[{i}]={p} should match forward[{i}]={f}"
            );
        }
    }

    #[test]
    fn min_gru_cell_hidden_bounded() {
        // The sigmoid gate ensures interpolation stays bounded.
        let mut cell = MinGRUCell::new(16, 55);
        let x_large: Vec<f64> = (0..10).map(|i| (i as f64 - 5.0) * 10.0).collect();

        for _ in 0..100 {
            let h = cell.forward(&x_large);
            for (i, &val) in h.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "h[{}] = {} should be finite even with large inputs",
                    i,
                    val
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "forward_predict called before initialization")]
    fn min_gru_cell_forward_predict_panics_before_init() {
        let cell = MinGRUCell::new(4, 42);
        let _ = cell.forward_predict(&[1.0, 2.0]);
    }
}
