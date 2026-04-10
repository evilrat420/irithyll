//! sLSTM cell implementation with exponential gating and log-domain stabilization.
//!
//! The sLSTM cell processes one timestep at a time, maintaining hidden state `h`,
//! cell state `c`, normalizer state `n`, and log-domain stabilizer `m`. Weights
//! are lazily initialized on the first call to [`SLSTMCell::forward`] when the
//! input dimension becomes known.
//!
//! # Per-step equations
//!
//! For each hidden unit j:
//!
//! ```text
//! xh = [x_t, h_{t-1}]                       // concatenated input
//!
//! pre_f_j = dot(w_f[j], xh) + b_f[j]        // forget gate pre-activation
//! pre_i_j = dot(w_i[j], xh) + b_i[j]        // input gate pre-activation
//! pre_o_j = dot(w_o[j], xh) + b_o[j]        // output gate pre-activation
//! pre_z_j = dot(w_z[j], xh) + b_z[j]        // cell candidate pre-activation
//!
//! m_new_j = max(pre_f_j + m_j, pre_i_j)     // log-domain stabilizer
//! f'_j    = exp(pre_f_j + m_j - m_new_j)    // stabilized forget gate
//! i'_j    = exp(pre_i_j - m_new_j)           // stabilized input gate
//! o_j     = sigmoid(pre_o_j)                 // output gate (standard)
//! z_j     = tanh(pre_z_j)                    // cell candidate
//!
//! c_j     = f'_j * c_j + i'_j * z_j         // cell state update
//! n_j     = f'_j * n_j + i'_j               // normalizer state update
//! m_j     = m_new_j                          // stabilizer update
//!
//! h_j     = o_j * (c_j / max(|n_j|, 1.0))   // normalized output
//! ```

use alloc::vec;
use alloc::vec::Vec;

use crate::math;
use crate::rng::standard_normal;

/// Maximum absolute value for gate pre-activations before `exp()`.
///
/// `exp(20) ~ 4.85e8` which is safe; `exp(700)` would overflow to infinity.
const PRE_GATE_CLAMP: f64 = 20.0;

/// sLSTM cell with exponential gating, log-domain stabilization, and normalizer state.
///
/// Weights are lazily allocated on the first [`forward`](SLSTMCell::forward) call
/// when the input dimension becomes known. Xavier initialization is used for gate
/// weight matrices, forget gate biases start at 1.0 (standard LSTM practice for
/// strong initial memory retention), and the normalizer state starts at 1.0 to
/// avoid division by zero.
///
/// # Example
///
/// ```
/// use irithyll_core::lstm::SLSTMCell;
///
/// let mut cell = SLSTMCell::new(8, 42);
/// let input = [0.1, -0.2, 0.3, 0.4];
/// let h = cell.forward(&input);
/// assert_eq!(h.len(), 8);
/// ```
pub struct SLSTMCell {
    // Gate weight matrices: each [d_hidden x d_total] row-major,
    // where d_total = d_input + d_hidden.
    w_f: Vec<f64>,
    w_i: Vec<f64>,
    w_o: Vec<f64>,
    w_z: Vec<f64>,

    // Gate biases: [d_hidden] each.
    b_f: Vec<f64>,
    b_i: Vec<f64>,
    b_o: Vec<f64>,
    b_z: Vec<f64>,

    // Recurrent state vectors: [d_hidden] each.
    h: Vec<f64>,
    c: Vec<f64>,
    n: Vec<f64>,
    m: Vec<f64>,

    d_input: usize,
    d_hidden: usize,
    initialized: bool,
    rng_state: u64,
}

impl SLSTMCell {
    /// Create a new sLSTM cell with the given hidden dimension.
    ///
    /// Weights are not allocated until the first call to [`forward`](SLSTMCell::forward),
    /// when the input dimension is inferred from the input slice length.
    ///
    /// # Arguments
    ///
    /// * `d_hidden` -- number of hidden units
    /// * `seed` -- RNG seed for deterministic weight initialization
    pub fn new(d_hidden: usize, seed: u64) -> Self {
        Self {
            w_f: Vec::new(),
            w_i: Vec::new(),
            w_o: Vec::new(),
            w_z: Vec::new(),
            b_f: Vec::new(),
            b_i: Vec::new(),
            b_o: Vec::new(),
            b_z: Vec::new(),
            h: vec![0.0; d_hidden],
            c: vec![0.0; d_hidden],
            n: vec![1.0; d_hidden],
            m: vec![0.0; d_hidden],
            d_input: 0,
            d_hidden,
            initialized: false,
            rng_state: seed,
        }
    }

    /// Lazily initialize weight matrices when the input dimension is first known.
    ///
    /// Uses Xavier initialization: `standard_normal * sqrt(2 / (fan_in + fan_out))`.
    /// Forget gate bias is set to 1.0; all other biases start at 0.0.
    fn ensure_initialized(&mut self, d_input: usize) {
        if self.initialized {
            return;
        }
        self.d_input = d_input;
        let d_total = d_input + self.d_hidden;
        let scale = math::sqrt(2.0 / (d_input + self.d_hidden) as f64);
        let n_weights = self.d_hidden * d_total;

        self.w_f = (0..n_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();
        self.w_i = (0..n_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();
        self.w_o = (0..n_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();
        self.w_z = (0..n_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();

        // Forget gate bias = 1.0 (strong initial memory retention)
        self.b_f = vec![1.0; self.d_hidden];
        self.b_i = vec![0.0; self.d_hidden];
        self.b_o = vec![0.0; self.d_hidden];
        self.b_z = vec![0.0; self.d_hidden];

        self.initialized = true;
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
        let d_total = self.d_input + d_h;

        for j in 0..d_h {
            let row_offset = j * d_total;

            // Compute pre-activations: dot(w[j], [x, h]) + b[j]
            let mut pre_f = self.b_f[j];
            let mut pre_i = self.b_i[j];
            let mut pre_o = self.b_o[j];
            let mut pre_z = self.b_z[j];

            for (k, &xk) in x.iter().enumerate() {
                pre_f += self.w_f[row_offset + k] * xk;
                pre_i += self.w_i[row_offset + k] * xk;
                pre_o += self.w_o[row_offset + k] * xk;
                pre_z += self.w_z[row_offset + k] * xk;
            }
            let h_offset = row_offset + self.d_input;
            for k in 0..d_h {
                let hk = self.h[k];
                pre_f += self.w_f[h_offset + k] * hk;
                pre_i += self.w_i[h_offset + k] * hk;
                pre_o += self.w_o[h_offset + k] * hk;
                pre_z += self.w_z[h_offset + k] * hk;
            }

            // Clamp pre-activations for forget and input gates to prevent exp overflow
            pre_f = clamp(pre_f, -PRE_GATE_CLAMP, PRE_GATE_CLAMP);
            pre_i = clamp(pre_i, -PRE_GATE_CLAMP, PRE_GATE_CLAMP);

            // Log-domain stabilizer
            let m_old = self.m[j];
            let log_f = pre_f + m_old;
            let m_new = if log_f > pre_i { log_f } else { pre_i };

            // Stabilized exponential gates
            let f_prime = math::exp(log_f - m_new);
            let i_prime = math::exp(pre_i - m_new);

            // Standard gates
            let o = math::sigmoid(pre_o);
            let z = math::tanh(pre_z);

            // State updates
            self.c[j] = f_prime * self.c[j] + i_prime * z;
            self.n[j] = f_prime * self.n[j] + i_prime;
            self.m[j] = m_new;

            // Normalized output
            let denom = if math::abs(self.n[j]) > 1.0 {
                math::abs(self.n[j])
            } else {
                1.0
            };
            self.h[j] = o * (self.c[j] / denom);
        }

        &self.h
    }

    /// Compute what the hidden state would be after processing `x`, without
    /// mutating any internal state.
    ///
    /// This is useful for prediction/inference where the model state should
    /// remain unchanged (e.g., during look-ahead evaluation).
    ///
    /// # Arguments
    ///
    /// * `x` -- input feature vector of length `d_input`
    ///
    /// # Returns
    ///
    /// The computed hidden state as a new `Vec<f64>`.
    ///
    /// # Panics
    ///
    /// Panics if called before the cell has been initialized (i.e., before any
    /// call to [`forward`](SLSTMCell::forward)).
    pub fn forward_predict(&self, x: &[f64]) -> Vec<f64> {
        assert!(
            self.initialized,
            "forward_predict called before initialization; call forward() first"
        );
        let d_h = self.d_hidden;
        let d_total = self.d_input + d_h;

        let mut h_out = vec![0.0; d_h];
        let mut c_tmp = self.c.clone();
        let mut n_tmp = self.n.clone();
        let mut m_tmp = self.m.clone();

        for j in 0..d_h {
            let row_offset = j * d_total;

            let mut pre_f = self.b_f[j];
            let mut pre_i = self.b_i[j];
            let mut pre_o = self.b_o[j];
            let mut pre_z = self.b_z[j];

            for (k, &xk) in x.iter().enumerate() {
                pre_f += self.w_f[row_offset + k] * xk;
                pre_i += self.w_i[row_offset + k] * xk;
                pre_o += self.w_o[row_offset + k] * xk;
                pre_z += self.w_z[row_offset + k] * xk;
            }
            let h_offset = row_offset + self.d_input;
            for k in 0..d_h {
                let hk = self.h[k];
                pre_f += self.w_f[h_offset + k] * hk;
                pre_i += self.w_i[h_offset + k] * hk;
                pre_o += self.w_o[h_offset + k] * hk;
                pre_z += self.w_z[h_offset + k] * hk;
            }

            pre_f = clamp(pre_f, -PRE_GATE_CLAMP, PRE_GATE_CLAMP);
            pre_i = clamp(pre_i, -PRE_GATE_CLAMP, PRE_GATE_CLAMP);

            let m_old = m_tmp[j];
            let log_f = pre_f + m_old;
            let m_new = if log_f > pre_i { log_f } else { pre_i };

            let f_prime = math::exp(log_f - m_new);
            let i_prime = math::exp(pre_i - m_new);
            let o = math::sigmoid(pre_o);
            let z = math::tanh(pre_z);

            c_tmp[j] = f_prime * c_tmp[j] + i_prime * z;
            n_tmp[j] = f_prime * n_tmp[j] + i_prime;
            m_tmp[j] = m_new;

            let denom = if math::abs(n_tmp[j]) > 1.0 {
                math::abs(n_tmp[j])
            } else {
                1.0
            };
            h_out[j] = o * (c_tmp[j] / denom);
        }

        h_out
    }

    /// Reset all recurrent state to initial values, preserving learned weights.
    ///
    /// After reset: `h` and `c` are zeroed, `n` is set to 1.0, and `m` is set
    /// to 0.0.
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
        self.n.fill(1.0);
        self.m.fill(0.0);
    }

    /// Reference to the current hidden state vector.
    #[inline]
    pub fn hidden_state(&self) -> &[f64] {
        &self.h
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
}

/// Clamp a value to `[lo, hi]`.
#[inline]
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slstm_cell_new() {
        let cell = SLSTMCell::new(16, 42);
        assert_eq!(cell.d_hidden(), 16, "d_hidden should match constructor arg");
        assert_eq!(cell.output_dim(), 16, "output_dim should equal d_hidden");
        assert!(
            !cell.initialized,
            "cell should not be initialized before first forward"
        );
        assert_eq!(
            cell.hidden_state().len(),
            16,
            "hidden state should be pre-allocated to d_hidden"
        );
    }

    #[test]
    fn slstm_cell_forward_initializes() {
        let mut cell = SLSTMCell::new(8, 42);
        assert!(!cell.initialized, "should start uninitialized");

        let x = [0.1, -0.2, 0.3, 0.4];
        let h_len = cell.forward(&x).len();

        assert!(
            cell.initialized,
            "should be initialized after first forward"
        );
        assert_eq!(h_len, 8, "output length should be d_hidden");
        assert_eq!(
            cell.d_input, 4,
            "d_input should be inferred from input length"
        );
        // Verify weight matrices were allocated
        assert_eq!(
            cell.w_f.len(),
            8 * (4 + 8),
            "w_f should have d_hidden * d_total elements"
        );
    }

    #[test]
    fn slstm_cell_forward_finite() {
        let mut cell = SLSTMCell::new(8, 123);
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
    fn slstm_cell_forward_predict_no_state_change() {
        let mut cell = SLSTMCell::new(4, 99);
        let x = [0.5, -0.3, 0.8];

        // Run one forward to initialize
        cell.forward(&x);

        // Snapshot state before forward_predict
        let h_before = cell.h.clone();
        let c_before = cell.c.clone();
        let n_before = cell.n.clone();
        let m_before = cell.m.clone();

        let x2 = [0.1, 0.2, -0.4];
        let _h_predict = cell.forward_predict(&x2);

        assert_eq!(
            cell.h, h_before,
            "hidden state should not change after forward_predict"
        );
        assert_eq!(
            cell.c, c_before,
            "cell state should not change after forward_predict"
        );
        assert_eq!(
            cell.n, n_before,
            "normalizer state should not change after forward_predict"
        );
        assert_eq!(
            cell.m, m_before,
            "stabilizer state should not change after forward_predict"
        );
    }

    #[test]
    fn slstm_cell_reset() {
        let mut cell = SLSTMCell::new(4, 77);
        let x = [1.0, -1.0];

        // Run a few steps to build up state
        for _ in 0..5 {
            cell.forward(&x);
        }

        // Snapshot weights before reset
        let w_f_before = cell.w_f.clone();
        let w_i_before = cell.w_i.clone();

        cell.reset();

        // State should be zeroed/reset
        assert!(
            cell.h.iter().all(|&v| v == 0.0),
            "h should be all zeros after reset"
        );
        assert!(
            cell.c.iter().all(|&v| v == 0.0),
            "c should be all zeros after reset"
        );
        assert!(
            cell.n.iter().all(|&v| v == 1.0),
            "n should be all 1.0 after reset"
        );
        assert!(
            cell.m.iter().all(|&v| v == 0.0),
            "m should be all zeros after reset"
        );

        // Weights should be preserved
        assert_eq!(
            cell.w_f, w_f_before,
            "w_f weights should be preserved after reset"
        );
        assert_eq!(
            cell.w_i, w_i_before,
            "w_i weights should be preserved after reset"
        );
    }

    #[test]
    fn slstm_cell_exponential_gating_range() {
        let mut cell = SLSTMCell::new(16, 55);

        // Feed large-magnitude inputs that would cause exp overflow without clamping
        let x_large: Vec<f64> = (0..10).map(|i| (i as f64 - 5.0) * 10.0).collect();

        for _ in 0..50 {
            let h = cell.forward(&x_large);
            for (i, &val) in h.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "h[{}] = {} should be finite even with large inputs",
                    i,
                    val
                );
                assert!(
                    !val.is_nan(),
                    "h[{}] should not be NaN even with large inputs",
                    i,
                );
            }
        }
    }

    #[test]
    fn slstm_cell_sequence_evolves_state() {
        let mut cell = SLSTMCell::new(4, 42);
        let x = [0.5, -0.3, 0.8];

        let h1 = cell.forward(&x).to_vec();
        let h2 = cell.forward(&x).to_vec();
        let h3 = cell.forward(&x).to_vec();

        // After multiple steps with the same input, hidden state should differ
        // between steps (the recurrent connection + exponential gating causes
        // state evolution).
        assert_ne!(
            h1, h2,
            "hidden state should evolve between step 1 and step 2"
        );
        assert_ne!(
            h2, h3,
            "hidden state should evolve between step 2 and step 3"
        );
    }
}
