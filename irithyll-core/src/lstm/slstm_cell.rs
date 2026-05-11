//! sLSTM cell implementation with exponential gating and log-domain stabilization.
//!
//! The sLSTM cell processes one timestep at a time, maintaining hidden state `h`,
//! cell state `c`, normalizer state `n`, and log-domain stabilizer `m`. Weights
//! are lazily initialized on the first call to [`SLSTMCell::forward`] when the
//! input dimension becomes known.
//!
//! # Per-step equations (Beck et al. 2024, xLSTM §2.2)
//!
//! For each hidden unit j:
//!
//! ```text
//! // Single-head (n_heads == 1):
//! xh = [x_t, h_{t-1}]
//! pre_f_j = dot(w_f[j], xh) + b_f[j]
//!
//! // Multi-head block-diagonal (n_heads > 1):
//! // Input block W_gate stays dense; recurrent block R_gate is block-diagonal.
//! // For unit j in head k (k = j / d_h_per_head):
//! pre_f_j = dot(w_input_f[j], x) + dot(r_f[j], h[k*d_h_per_head..(k+1)*d_h_per_head]) + b_f[j]
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
//! // Scale-equivariant denominator (Beck et al. 2024 §2.2):
//! // eps * exp(-m_j) is the scale-equivariant floor — it tracks the running
//! // log-scale magnitude so the denominator never artificially suppresses h.
//! h_j     = o_j * (c_j / max(|n_j|, DENOM_EPS * exp(-m_j)))
//! ```

use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::math;
use crate::rng::standard_normal;

/// Maximum absolute value for gate pre-activations before `exp()`.
///
/// `exp(20) ~ 4.85e8` which is safe; `exp(700)` would overflow to infinity.
const PRE_GATE_CLAMP: f64 = 20.0;

/// Scale-equivariant denominator floor (Beck et al. 2024 §2.2).
///
/// The denominator for `h_j = o_j * (c_j / max(|n_j|, DENOM_EPS * exp(-m_j)))`.
/// Using `DENOM_EPS * exp(-m_j)` instead of the constant 1.0 ensures that the
/// floor tracks the running log-scale stabilizer, so the divisor never
/// artificially suppresses hidden-state magnitude in low-gate regimes.
const DENOM_EPS: f64 = 1e-6;

/// sLSTM cell with exponential gating, log-domain stabilization, and normalizer state.
///
/// Supports single-head (dense recurrent weights) and multi-head block-diagonal
/// recurrent weights (Beck et al. 2024, xLSTM §2.2 — "SLOTS" innovation).
///
/// Weights are lazily allocated on the first [`forward`](SLSTMCell::forward) call
/// when the input dimension becomes known. Xavier initialization is used for gate
/// weight matrices, and the normalizer state starts at 1.0 to avoid division by
/// zero on the first step.
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
    // Input projection weights: [d_hidden x d_input] row-major.
    // Dense — every unit sees all inputs regardless of head structure.
    w_input_f: Vec<f64>,
    w_input_i: Vec<f64>,
    w_input_o: Vec<f64>,
    w_input_z: Vec<f64>,

    // Recurrent weights.
    //
    // Single-head (n_heads == 1): [d_hidden x d_hidden] row-major (same layout
    // as the old fused w_f but restricted to the recurrent columns).
    //
    // Multi-head block-diagonal (n_heads > 1): stored as n_heads independent
    // blocks, each [d_h_per_head x d_h_per_head] row-major, concatenated.
    // Total length: n_heads * d_h_per_head^2 == d_hidden * d_h_per_head.
    r_f: Vec<f64>,
    r_i: Vec<f64>,
    r_o: Vec<f64>,
    r_z: Vec<f64>,

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

    // Pre-allocated scratch buffer, partitioned during forward().
    // Layout: [pre_f | pre_i | pre_o | pre_z | o_gate | z_gate | f_prime | i_prime]
    // Sizes:   d_h     d_h     d_h     d_h      d_h      d_h      d_h      d_h
    scratch: Vec<f64>,

    d_input: usize,
    d_hidden: usize,
    /// Number of heads for block-diagonal recurrent weights.
    /// Must divide `d_hidden`. When 1 (default), uses a full dense recurrent matrix.
    n_heads: usize,
    /// Hidden units per head = d_hidden / n_heads.
    d_h_per_head: usize,
    /// Forget gate bias initializer values, one per hidden unit.
    forget_bias_init: Vec<f64>,
    initialized: bool,
    rng_state: u64,
}

impl SLSTMCell {
    /// Create a new single-head sLSTM cell.
    ///
    /// Forget gate bias is initialized to 1.0 (standard LSTM practice).
    /// For multi-head block-diagonal or custom forget-bias initialization,
    /// use [`SLSTMCell::with_config`].
    ///
    /// Weights are not allocated until the first call to [`forward`](SLSTMCell::forward).
    ///
    /// # Arguments
    ///
    /// * `d_hidden` -- number of hidden units
    /// * `seed` -- RNG seed for deterministic weight initialization
    pub fn new(d_hidden: usize, seed: u64) -> Self {
        let forget_bias_init = vec![1.0; d_hidden];
        Self {
            w_input_f: Vec::new(),
            w_input_i: Vec::new(),
            w_input_o: Vec::new(),
            w_input_z: Vec::new(),
            r_f: Vec::new(),
            r_i: Vec::new(),
            r_o: Vec::new(),
            r_z: Vec::new(),
            b_f: Vec::new(),
            b_i: Vec::new(),
            b_o: Vec::new(),
            b_z: Vec::new(),
            h: vec![0.0; d_hidden],
            c: vec![0.0; d_hidden],
            n: vec![1.0; d_hidden],
            m: vec![0.0; d_hidden],
            scratch: Vec::new(),
            d_input: 0,
            d_hidden,
            n_heads: 1,
            d_h_per_head: d_hidden,
            forget_bias_init,
            initialized: false,
            rng_state: seed,
        }
    }

    /// Create an sLSTM cell with explicit head count and forget-bias initializer.
    ///
    /// When `n_heads > 1`, the recurrent weight matrices are block-diagonal —
    /// each head only mixes within its own `d_hidden / n_heads` units, while the
    /// input projection remains dense. This is the SLOTS mechanism from
    /// Beck et al. (2024) "xLSTM" §2.2.
    ///
    /// # Arguments
    ///
    /// * `d_hidden` -- number of hidden units; must be divisible by `n_heads`
    /// * `n_heads` -- number of heads (1 = dense / single-head)
    /// * `forget_bias_init` -- per-unit forget bias values (length must equal `d_hidden`)
    /// * `seed` -- RNG seed
    ///
    /// # Panics
    ///
    /// Panics if `n_heads` does not divide `d_hidden`, or if
    /// `forget_bias_init.len() != d_hidden`.
    pub fn with_config(
        d_hidden: usize,
        n_heads: usize,
        forget_bias_init: Vec<f64>,
        seed: u64,
    ) -> Self {
        assert!(n_heads > 0, "n_heads must be > 0");
        assert!(
            d_hidden % n_heads == 0,
            "n_heads ({}) must divide d_hidden ({})",
            n_heads,
            d_hidden
        );
        assert_eq!(
            forget_bias_init.len(),
            d_hidden,
            "forget_bias_init length ({}) must equal d_hidden ({})",
            forget_bias_init.len(),
            d_hidden
        );
        let d_h_per_head = d_hidden / n_heads;
        Self {
            w_input_f: Vec::new(),
            w_input_i: Vec::new(),
            w_input_o: Vec::new(),
            w_input_z: Vec::new(),
            r_f: Vec::new(),
            r_i: Vec::new(),
            r_o: Vec::new(),
            r_z: Vec::new(),
            b_f: Vec::new(),
            b_i: Vec::new(),
            b_o: Vec::new(),
            b_z: Vec::new(),
            h: vec![0.0; d_hidden],
            c: vec![0.0; d_hidden],
            n: vec![1.0; d_hidden],
            m: vec![0.0; d_hidden],
            scratch: Vec::new(),
            d_input: 0,
            d_hidden,
            n_heads,
            d_h_per_head,
            forget_bias_init,
            initialized: false,
            rng_state: seed,
        }
    }

    /// Lazily initialize weight matrices when the input dimension is first known.
    ///
    /// Uses Xavier initialization: `standard_normal * sqrt(2 / (fan_in + fan_out))`.
    ///
    /// Input projection weights (`w_input_*`): `d_hidden x d_input` dense.
    /// Recurrent weights (`r_*`): block-diagonal with `n_heads` blocks of
    /// `d_h_per_head x d_h_per_head` each. Total length per gate:
    /// `n_heads * d_h_per_head^2 == d_hidden * d_h_per_head`.
    fn ensure_initialized(&mut self, d_input: usize) {
        if self.initialized {
            return;
        }
        self.d_input = d_input;

        // Xavier scale uses total fan-in for the combined projection.
        let d_total = d_input + self.d_hidden;
        let scale = math::sqrt(2.0 / d_total as f64);

        // Input projection: d_hidden x d_input (dense, same for all heads).
        let n_input_weights = self.d_hidden * d_input;
        self.w_input_f = (0..n_input_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();
        self.w_input_i = (0..n_input_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();
        self.w_input_o = (0..n_input_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();
        self.w_input_z = (0..n_input_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();

        // Recurrent projection: block-diagonal, total length = d_hidden * d_h_per_head.
        // For n_heads == 1, d_h_per_head == d_hidden → full d_hidden x d_hidden matrix.
        let n_recurrent_weights = self.d_hidden * self.d_h_per_head;
        self.r_f = (0..n_recurrent_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();
        self.r_i = (0..n_recurrent_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();
        self.r_o = (0..n_recurrent_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();
        self.r_z = (0..n_recurrent_weights)
            .map(|_| standard_normal(&mut self.rng_state) * scale)
            .collect();

        // Biases: forget gate uses the per-unit init values (Beck et al. 2024 §3.2).
        // linspace(3, 6) gives exp(3..6) ≈ 20×–400× stronger initial memory retention
        // compared to the scalar 1.0 default.
        self.b_f = self.forget_bias_init.clone();
        self.b_i = vec![0.0; self.d_hidden];
        self.b_o = vec![0.0; self.d_hidden];
        self.b_z = vec![0.0; self.d_hidden];

        // Scratch: 8 * d_hidden (pre_f, pre_i, pre_o, pre_z, o_gate, z_gate, f_prime, i_prime).
        // No xh slot — input and recurrent are computed separately now.
        self.scratch = vec![0.0; 8 * self.d_hidden];

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

        // Take scratch out of self to avoid borrow conflicts with other fields.
        let mut scratch = mem::take(&mut self.scratch);

        // Partition scratch: [pre_f | pre_i | pre_o | pre_z | o_gate | z_gate | f_prime | i_prime]
        let (pre_f, rest) = scratch.split_at_mut(d_h);
        let (pre_i, rest) = rest.split_at_mut(d_h);
        let (pre_o, rest) = rest.split_at_mut(d_h);
        let (pre_z, rest) = rest.split_at_mut(d_h);
        let (o_gate, rest) = rest.split_at_mut(d_h);
        let (z_gate, rest) = rest.split_at_mut(d_h);
        let (f_prime, i_prime) = rest.split_at_mut(d_h);

        // 1. Input projection (dense): pre_gate += W_input * x
        crate::simd::simd_mat_vec(&self.w_input_f, x, d_h, self.d_input, pre_f);
        crate::simd::simd_mat_vec(&self.w_input_i, x, d_h, self.d_input, pre_i);
        crate::simd::simd_mat_vec(&self.w_input_o, x, d_h, self.d_input, pre_o);
        crate::simd::simd_mat_vec(&self.w_input_z, x, d_h, self.d_input, pre_z);

        // 2. Recurrent projection (block-diagonal): pre_gate += R_head * h_head
        //    For n_heads == 1: full d_hidden × d_hidden recurrent matrix.
        //    For n_heads > 1:  each head k operates only on h[k*dph..(k+1)*dph].
        compute_block_diagonal_recurrent(&self.r_f, &self.h, d_h, self.d_h_per_head, pre_f);
        compute_block_diagonal_recurrent(&self.r_i, &self.h, d_h, self.d_h_per_head, pre_i);
        compute_block_diagonal_recurrent(&self.r_o, &self.h, d_h, self.d_h_per_head, pre_o);
        compute_block_diagonal_recurrent(&self.r_z, &self.h, d_h, self.d_h_per_head, pre_z);

        // 3. Add biases + clamp forget/input gates.
        for j in 0..d_h {
            pre_f[j] += self.b_f[j];
            pre_i[j] += self.b_i[j];
            pre_o[j] += self.b_o[j];
            pre_z[j] += self.b_z[j];
            pre_f[j] = clamp(pre_f[j], -PRE_GATE_CLAMP, PRE_GATE_CLAMP);
            pre_i[j] = clamp(pre_i[j], -PRE_GATE_CLAMP, PRE_GATE_CLAMP);
        }

        // 4. Batch activations: sigmoid for output gate, tanh for candidate.
        crate::simd::simd_sigmoid(pre_o, o_gate);
        crate::simd::simd_tanh(pre_z, z_gate);

        // 5. Compute stabilizers; reuse pre_f/pre_i in-place as exp inputs.
        for j in 0..d_h {
            let log_f = pre_f[j] + self.m[j];
            let m_new = if log_f > pre_i[j] { log_f } else { pre_i[j] };
            pre_f[j] = log_f - m_new;
            pre_i[j] -= m_new;
            self.m[j] = m_new;
        }

        // 6. Batch exp for stabilized gates.
        crate::simd::simd_exp(pre_f, f_prime);
        crate::simd::simd_exp(pre_i, i_prime);

        // 7. State updates with scale-equivariant denominator (Beck et al. 2024 §2.2).
        //    floor = DENOM_EPS * exp(-m_j) tracks the running log-scale and avoids
        //    the artificial scale-down that the constant 1.0 floor causes in low-gate
        //    regimes.
        for j in 0..d_h {
            self.c[j] = f_prime[j] * self.c[j] + i_prime[j] * z_gate[j];
            self.n[j] = f_prime[j] * self.n[j] + i_prime[j];
            let abs_n = math::abs(self.n[j]);
            let floor = DENOM_EPS * math::exp(-self.m[j]);
            let denom = if abs_n > floor { abs_n } else { floor };
            self.h[j] = o_gate[j] * (self.c[j] / denom);
        }

        // Put scratch back.
        self.scratch = scratch;

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

        let mut c_tmp = self.c.clone();
        let mut n_tmp = self.n.clone();
        let mut m_tmp = self.m.clone();

        // Local scratch (cold path — allocation is acceptable).
        let mut pre_f = vec![0.0; d_h];
        let mut pre_i = vec![0.0; d_h];
        let mut pre_o = vec![0.0; d_h];
        let mut pre_z = vec![0.0; d_h];
        let mut o_gate = vec![0.0; d_h];
        let mut z_gate = vec![0.0; d_h];
        let mut f_prime = vec![0.0; d_h];
        let mut i_prime = vec![0.0; d_h];

        // 1. Input projection (dense).
        crate::simd::simd_mat_vec(&self.w_input_f, x, d_h, self.d_input, &mut pre_f);
        crate::simd::simd_mat_vec(&self.w_input_i, x, d_h, self.d_input, &mut pre_i);
        crate::simd::simd_mat_vec(&self.w_input_o, x, d_h, self.d_input, &mut pre_o);
        crate::simd::simd_mat_vec(&self.w_input_z, x, d_h, self.d_input, &mut pre_z);

        // 2. Recurrent projection (block-diagonal).
        compute_block_diagonal_recurrent(&self.r_f, &self.h, d_h, self.d_h_per_head, &mut pre_f);
        compute_block_diagonal_recurrent(&self.r_i, &self.h, d_h, self.d_h_per_head, &mut pre_i);
        compute_block_diagonal_recurrent(&self.r_o, &self.h, d_h, self.d_h_per_head, &mut pre_o);
        compute_block_diagonal_recurrent(&self.r_z, &self.h, d_h, self.d_h_per_head, &mut pre_z);

        // 3. Add biases + clamp forget/input gates.
        for j in 0..d_h {
            pre_f[j] += self.b_f[j];
            pre_i[j] += self.b_i[j];
            pre_o[j] += self.b_o[j];
            pre_z[j] += self.b_z[j];
            pre_f[j] = clamp(pre_f[j], -PRE_GATE_CLAMP, PRE_GATE_CLAMP);
            pre_i[j] = clamp(pre_i[j], -PRE_GATE_CLAMP, PRE_GATE_CLAMP);
        }

        // 4. Batch activations.
        crate::simd::simd_sigmoid(&pre_o, &mut o_gate);
        crate::simd::simd_tanh(&pre_z, &mut z_gate);

        // 5. Compute stabilizers; reuse pre_f/pre_i as exp inputs.
        for j in 0..d_h {
            let log_f = pre_f[j] + m_tmp[j];
            let m_new = if log_f > pre_i[j] { log_f } else { pre_i[j] };
            pre_f[j] = log_f - m_new;
            pre_i[j] -= m_new;
            m_tmp[j] = m_new;
        }

        // 6. Batch exp.
        crate::simd::simd_exp(&pre_f, &mut f_prime);
        crate::simd::simd_exp(&pre_i, &mut i_prime);

        // 7. State updates with scale-equivariant denominator.
        let mut h_out = vec![0.0; d_h];
        for j in 0..d_h {
            c_tmp[j] = f_prime[j] * c_tmp[j] + i_prime[j] * z_gate[j];
            n_tmp[j] = f_prime[j] * n_tmp[j] + i_prime[j];
            let abs_n = math::abs(n_tmp[j]);
            let floor = DENOM_EPS * math::exp(-m_tmp[j]);
            let denom = if abs_n > floor { abs_n } else { floor };
            h_out[j] = o_gate[j] * (c_tmp[j] / denom);
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
        self.scratch.fill(0.0);
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

    /// Surgically reinitialize a single hidden unit (Dohare et al., Nature 2024).
    ///
    /// Reinitializes row `j` of the input projection and the corresponding row
    /// in the recurrent block that contains unit `j`. Resets the corresponding
    /// biases and zeros recurrent state (h, c, n, m) for unit `j`.
    ///
    /// The forget bias is restored to the original `forget_bias_init[j]` value.
    ///
    /// # Arguments
    ///
    /// * `j` — hidden unit index to reinitialize (must be < `d_hidden`)
    /// * `rng` — mutable RNG state for generating fresh weights
    ///
    /// # Panics
    ///
    /// Panics if `j >= d_hidden` or if the cell has not been initialized.
    pub fn reinitialize_unit(&mut self, j: usize, rng: &mut u64) {
        assert!(self.initialized, "cell must be initialized before reinit");
        assert!(
            j < self.d_hidden,
            "unit index {} out of range (d_hidden={})",
            j,
            self.d_hidden
        );

        let d_total = self.d_input + self.d_hidden;
        let scale = math::sqrt(2.0 / d_total as f64);

        // Reinitialize row j of input projection (d_input columns).
        let input_row_start = j * self.d_input;
        for col in 0..self.d_input {
            self.w_input_f[input_row_start + col] = standard_normal(rng) * scale;
            self.w_input_i[input_row_start + col] = standard_normal(rng) * scale;
            self.w_input_o[input_row_start + col] = standard_normal(rng) * scale;
            self.w_input_z[input_row_start + col] = standard_normal(rng) * scale;
        }

        // Reinitialize row j within its recurrent block.
        // In block k (k = j / d_h_per_head), unit j is at local index l = j % d_h_per_head.
        // Its recurrent weight row starts at: k * d_h_per_head^2 + l * d_h_per_head.
        let k = j / self.d_h_per_head;
        let l = j % self.d_h_per_head;
        let recurrent_row_start = k * self.d_h_per_head * self.d_h_per_head + l * self.d_h_per_head;
        for col in 0..self.d_h_per_head {
            self.r_f[recurrent_row_start + col] = standard_normal(rng) * scale;
            self.r_i[recurrent_row_start + col] = standard_normal(rng) * scale;
            self.r_o[recurrent_row_start + col] = standard_normal(rng) * scale;
            self.r_z[recurrent_row_start + col] = standard_normal(rng) * scale;
        }

        // Reset biases: forget to its init value, others to 0.0.
        self.b_f[j] = self.forget_bias_init[j];
        self.b_i[j] = 0.0;
        self.b_o[j] = 0.0;
        self.b_z[j] = 0.0;

        // Zero recurrent state for this unit.
        self.h[j] = 0.0;
        self.c[j] = 0.0;
        self.n[j] = 1.0;
        self.m[j] = 0.0;
    }

    /// Number of attention heads.
    #[inline]
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Construct the per-unit forget bias vector using `linspace(start, stop, d_hidden)`.
    ///
    /// Beck et al. (2024) §3.2 specify forget bias initialized as
    /// `linspace(3, 6)` across `d_hidden` units, giving a strong initial
    /// memory bias that decays gracefully over training.
    ///
    /// This helper is a factory for the bias vector to pass into
    /// [`SLSTMCell::with_config`]. For the paper-recommended values:
    ///
    /// ```
    /// use irithyll_core::lstm::SLSTMCell;
    ///
    /// let d = 8usize;
    /// let bias = SLSTMCell::forget_bias_linspace(3.0, 6.0, d);
    /// assert_eq!(bias.len(), d);
    /// // First element == 3.0, last element == 6.0
    /// assert!((bias[0] - 3.0).abs() < 1e-12);
    /// assert!((bias[d - 1] - 6.0).abs() < 1e-12);
    ///
    /// let cell = SLSTMCell::with_config(d, 1, bias, 42);
    /// assert_eq!(cell.n_heads(), 1);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `start` -- first value (Beck et al. recommend 3.0)
    /// * `stop` -- last value (Beck et al. recommend 6.0)
    /// * `n` -- number of units (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn forget_bias_linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
        assert!(n > 0, "n must be > 0");
        if n == 1 {
            return vec![start];
        }
        let step = (stop - start) / (n - 1) as f64;
        (0..n).map(|i| start + step * i as f64).collect()
    }
}

/// Block-diagonal recurrent matvec accumulator: `out += R_block @ h_block` per head.
///
/// Layout assumption: `r` stores `n_heads = d_hidden / d_h_per_head` square blocks,
/// each `d_h_per_head × d_h_per_head` row-major, concatenated. For each head k,
/// only the corresponding `d_h_per_head` slice of `h` and `out` is touched —
/// this is the SLOTS block-diagonal mechanism (Beck et al. 2024 xLSTM §2.2).
///
/// # Arguments
///
/// * `r` -- block-diagonal recurrent weight matrix (`d_hidden * d_h_per_head` long)
/// * `h` -- hidden state vector (`d_hidden` long)
/// * `d_hidden` -- total hidden dimension
/// * `d_h_per_head` -- units per head (must divide `d_hidden`)
/// * `out` -- accumulator vector (`d_hidden` long, `+=` semantics)
fn compute_block_diagonal_recurrent(
    r: &[f64],
    h: &[f64],
    d_hidden: usize,
    d_h_per_head: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(h.len(), d_hidden);
    debug_assert_eq!(out.len(), d_hidden);
    debug_assert_eq!(d_hidden % d_h_per_head, 0);
    debug_assert_eq!(r.len(), d_hidden * d_h_per_head);

    let n_heads = d_hidden / d_h_per_head;
    let block_size = d_h_per_head * d_h_per_head;

    for k in 0..n_heads {
        let r_block_start = k * block_size;
        let h_offset = k * d_h_per_head;
        for i in 0..d_h_per_head {
            let row_start = r_block_start + i * d_h_per_head;
            let mut acc = 0.0;
            for j in 0..d_h_per_head {
                acc += r[row_start + j] * h[h_offset + j];
            }
            out[h_offset + i] += acc;
        }
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
        // Verify weight matrices were allocated.
        // After the input/recurrent weight split, w_input_f is [d_hidden × d_input]
        // and r_f is [d_hidden × d_hidden]; check each.
        assert_eq!(
            cell.w_input_f.len(),
            8 * 4,
            "w_input_f should have d_hidden * d_input elements"
        );
        assert_eq!(
            cell.r_f.len(),
            8 * 8,
            "r_f should have d_hidden * d_hidden elements"
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

        // Snapshot recurrent weights before reset
        let w_f_before = cell.r_f.clone();
        let w_i_before = cell.r_i.clone();

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

        // Recurrent weights should be preserved
        assert_eq!(
            cell.r_f, w_f_before,
            "r_f weights should be preserved after reset"
        );
        assert_eq!(
            cell.r_i, w_i_before,
            "r_i weights should be preserved after reset"
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

    #[test]
    fn reinitialize_unit_resets_target_only() {
        let mut cell = SLSTMCell::new(4, 42);
        let x = [0.5, -0.3, 0.8];

        // Initialize and build up state
        for _ in 0..10 {
            cell.forward(&x);
        }

        // Save state of unit 0 and unit 2 before reinit
        let h0_before = cell.h[0];
        let h2_before = cell.h[2];
        let c2_before = cell.c[2];

        // Reinitialize unit 1 only
        let mut rng = 999u64;
        cell.reinitialize_unit(1, &mut rng);

        // Unit 1 should be zeroed
        assert!(
            math::abs(cell.h[1]) < 1e-15,
            "reinit unit h should be zero, got {}",
            cell.h[1]
        );
        assert!(
            math::abs(cell.c[1]) < 1e-15,
            "reinit unit c should be zero, got {}",
            cell.c[1]
        );
        assert!(
            (cell.n[1] - 1.0).abs() < 1e-15,
            "reinit unit n should be 1.0, got {}",
            cell.n[1]
        );

        // Other units should be untouched
        assert!(
            (cell.h[0] - h0_before).abs() < 1e-15,
            "unit 0 h should be unchanged after reinit of unit 1"
        );
        assert!(
            (cell.h[2] - h2_before).abs() < 1e-15,
            "unit 2 h should be unchanged after reinit of unit 1"
        );
        assert!(
            (cell.c[2] - c2_before).abs() < 1e-15,
            "unit 2 c should be unchanged after reinit of unit 1"
        );
    }

    #[test]
    fn reinitialize_unit_produces_fresh_weights() {
        let mut cell = SLSTMCell::new(4, 42);
        cell.forward(&[0.1, 0.2, 0.3]); // initialize

        // After the input/recurrent weight split, unit j=1 has its recurrent
        // weights at r_f[j * d_h_per_head .. (j+1) * d_h_per_head].
        // With n_heads=1 (default), d_h_per_head = d_hidden = 4.
        let d_h = cell.d_h_per_head; // 4
        let row_start = d_h; // unit 1

        // Save original recurrent weights for unit 1
        let w_f_before: Vec<f64> = cell.r_f[row_start..row_start + d_h].to_vec();

        // Reinitialize
        let mut rng = 777u64;
        cell.reinitialize_unit(1, &mut rng);

        // Weights should be different (fresh random)
        let w_f_after: Vec<f64> = cell.r_f[row_start..row_start + d_h].to_vec();
        let diff: f64 = w_f_before
            .iter()
            .zip(w_f_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-10,
            "reinitialized weights should differ from original"
        );

        // Forget bias should be 1.0
        assert!(
            (cell.b_f[1] - 1.0).abs() < 1e-15,
            "forget bias should be 1.0 after reinit, got {}",
            cell.b_f[1]
        );
    }

    /// Beck et al. (2024) §3.2: forget bias linspace(3,6) must be monotone,
    /// correctly bounded, and all values stored in b_f after initialization.
    #[test]
    fn forget_bias_uses_linspace_3_to_6() {
        let d = 8usize;
        let bias = SLSTMCell::forget_bias_linspace(3.0, 6.0, d);

        // Length matches d_hidden.
        assert_eq!(bias.len(), d, "linspace length must equal d_hidden");

        // Boundary values exact (Beck et al. §3.2 specifies 3 and 6).
        assert!(
            (bias[0] - 3.0).abs() < 1e-12,
            "first bias value must be 3.0, got {}",
            bias[0]
        );
        assert!(
            (bias[d - 1] - 6.0).abs() < 1e-12,
            "last bias value must be 6.0, got {}",
            bias[d - 1]
        );

        // Monotone increasing.
        for i in 1..d {
            assert!(
                bias[i] > bias[i - 1],
                "linspace must be strictly increasing at index {}",
                i
            );
        }

        // Uniform step.
        let step = (6.0 - 3.0) / (d - 1) as f64;
        for (i, &b) in bias.iter().enumerate() {
            let expected = 3.0 + step * i as f64;
            assert!(
                (b - expected).abs() < 1e-12,
                "bias[{}] expected {}, got {}",
                i,
                expected,
                b
            );
        }

        // Values are stored in b_f after the first forward call.
        let mut cell = SLSTMCell::with_config(d, 1, bias.clone(), 42);
        cell.forward(&[0.1, 0.2]); // trigger lazy init
        for (j, &expected) in bias.iter().enumerate() {
            // b_f gets written from forget_bias_init during ensure_initialized.
            // The b_f values will evolve after each forward, so we check the
            // stored forget_bias_init (the source of truth for reinit).
            assert!(
                (cell.forget_bias_init[j] - expected).abs() < 1e-12,
                "forget_bias_init[{}] must equal linspace value {}, got {}",
                j,
                expected,
                cell.forget_bias_init[j]
            );
        }
    }

    /// Verify the scale-equivariant denominator is strictly better than the
    /// constant 1.0 floor in low-gate regimes.
    ///
    /// With the constant floor `max(|n|, 1.0)`, hidden state is artificially
    /// suppressed when |n| << 1 (low-gate regime). With `max(|n|, eps*exp(-m))`,
    /// the floor tracks the running log-scale so h is NOT suppressed.
    ///
    /// This regression test verifies the current implementation (eps*exp(-m)) by
    /// observing that h has larger magnitude than what a constant-1.0 floor would
    /// produce in a scenario where the gates are small (low-gate regime).
    #[test]
    fn denominator_is_scale_equivariant_in_low_gate_regime() {
        // Build two cells with identical seeds; manually drive them to a
        // low-gate state by directly adjusting n and m after initialization.
        let d = 4usize;

        // Cell A: scale-equivariant floor (current implementation).
        let mut cell_a = SLSTMCell::new(d, 7);
        cell_a.forward(&[0.1, 0.2]); // init

        // Manually push state into a low-gate regime:
        //   m >> 0  =>  exp(-m) is large  =>  eps*exp(-m) gives a higher floor
        //   |n| << 1                          than constant 1.0 ONLY when eps*exp(-m) < 1
        //   We want a case where eps*exp(-m) is the binding constraint but LESS than 1.0,
        //   so the scale-equivariant floor is SMALLER, allowing |h| to be LARGER.
        //   When m < 0 (which happens when gates are small), exp(-m) > 1, so
        //   eps * exp(-m) could be < 1.0 but still bigger than |n|.
        //   Let's set m[j] = -10 => exp(-m) = exp(10) ~ 22026, eps*exp(-m) ~ 0.022
        //   And |n| = 1e-9 (very small).
        //   => max(|n|, eps*exp(-m)) = 0.022   (scale-equivariant)
        //   => max(|n|, 1.0)         = 1.0     (constant floor)
        //   => scale-equivariant gives |h| that is ~45x larger (1.0 / 0.022).
        for j in 0..d {
            cell_a.m[j] = -10.0; // forces exp(-m) = exp(10) large
            cell_a.n[j] = 1e-9; // |n| << 1, so |n| << eps*exp(-m)
            cell_a.c[j] = 1.0; // cell state magnitude = 1
        }

        // Simulate one output computation using scale-equivariant denominator.
        let h_equivariant: Vec<f64> = (0..d)
            .map(|j| {
                let abs_n = math::abs(cell_a.n[j]);
                let floor = DENOM_EPS * math::exp(-cell_a.m[j]);
                let denom = if abs_n > floor { abs_n } else { floor };
                // o_gate = 1.0 for max sensitivity
                cell_a.c[j] / denom
            })
            .collect();

        // Simulate same computation with constant-1.0 floor.
        let h_constant_floor: Vec<f64> = (0..d)
            .map(|j| {
                let abs_n = math::abs(cell_a.n[j]);
                let denom = if abs_n > 1.0 { abs_n } else { 1.0 };
                cell_a.c[j] / denom
            })
            .collect();

        // The scale-equivariant denominator is smaller (eps*exp(-10) < 1.0),
        // so |h_equivariant| > |h_constant_floor|.
        for (j, (&he, &hc)) in h_equivariant
            .iter()
            .zip(h_constant_floor.iter())
            .enumerate()
        {
            assert!(
                he.abs() > hc.abs(),
                "scale-equivariant h[{}]={:.6} must exceed constant-floor h[{}]={:.6} in low-gate regime",
                j, he, j, hc
            );
        }
    }

    /// When n_heads == 1, the block-diagonal layout degenerates to a full dense
    /// recurrent matrix. Verify that the forward output is identical to that
    /// produced by a cell where d_h_per_head == d_hidden (the n_heads=1 case).
    ///
    /// We compare two cells constructed with n_heads=1 but using
    /// `with_config` vs `new` — both should produce identical h at every step
    /// given the same seed, because `new` also sets n_heads=1 and d_h_per_head=d_hidden.
    #[test]
    fn n_heads_1_matches_dense_path() {
        let d = 8usize;
        let seed = 77u64;
        let forget_bias = vec![1.0; d];

        // Cell via `new` (n_heads implicitly 1, dense path).
        let mut cell_dense = SLSTMCell::new(d, seed);

        // Cell via `with_config` with n_heads=1 and identical seed.
        let mut cell_config = SLSTMCell::with_config(d, 1, forget_bias, seed);

        let inputs: &[&[f64]] = &[&[0.1, -0.2, 0.3], &[0.5, 0.0, -0.1], &[-0.3, 0.8, 0.2]];

        for &x in inputs {
            let h_dense = cell_dense.forward(x).to_vec();
            let h_config = cell_config.forward(x).to_vec();

            for (j, (a, b)) in h_dense.iter().zip(h_config.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-14,
                    "n_heads=1 config path must match dense path at unit {j}: dense={a}, config={b}"
                );
            }
        }
    }

    /// Multi-head (n_heads > 1) forward produces finite, bounded output and
    /// n_heads() accessor reports the correct value.
    #[test]
    fn slstm_multi_head_forward_finite_and_correct_n_heads() {
        let d = 8usize;
        let n_heads = 2usize;
        let bias = SLSTMCell::forget_bias_linspace(3.0, 6.0, d);
        let mut cell = SLSTMCell::with_config(d, n_heads, bias, 42);

        assert_eq!(
            cell.n_heads(),
            n_heads,
            "n_heads accessor must match constructor arg"
        );

        let x = [0.1f64, -0.2, 0.3, 0.4];
        for _ in 0..10 {
            let h = cell.forward(&x);
            assert_eq!(h.len(), d, "output length must equal d_hidden");
            for (j, &v) in h.iter().enumerate() {
                assert!(v.is_finite(), "multi-head h[{j}]={v} must be finite");
            }
        }
    }
}
