//! Core TTT (Test-Time Training) layer with prediction-directed fast weight updates.
//!
//! The hidden state is a weight matrix W ("fast weights") updated per step
//! by gradient descent. The gradient source depends on whether prediction
//! feedback is available:
//!
//! - **With prediction feedback** (after warmup): the fast weight update is
//!   directed by the prediction error projected onto the query space. This
//!   makes W_fast directly minimize prediction loss rather than reconstruction.
//! - **Without prediction feedback** (during warmup): falls back to the
//!   self-supervised reconstruction loss `L = ||W * k_t - (v_t - k_t)||^2`.
//!
//! The gradient is a rank-1 outer product: `dW = residual * k_t^T`.
//!
//! Based on Sun et al. (2024) "Learning to (Learn at Test Time)" ICML,
//! with Titans extensions (Behrouz et al., 2025): weight decay (forgetting)
//! and optional momentum for temporal coherence.

// ---------------------------------------------------------------------------
// RNG helpers (xorshift64, same as the rest of irithyll)
// ---------------------------------------------------------------------------

/// Advance xorshift64 state and return raw u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Uniform f64 in [0, 1) from xorshift64.
#[inline]
fn xorshift64_f64(state: &mut u64) -> f64 {
    // Use 53-bit mantissa for full f64 precision.
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Standard normal via Box-Muller transform (returns one sample).
fn standard_normal(state: &mut u64) -> f64 {
    loop {
        let u1 = xorshift64_f64(state);
        let u2 = xorshift64_f64(state);
        if u1 > 0.0 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            return r * theta.cos();
        }
    }
}

// ---------------------------------------------------------------------------
// TTTLayer
// ---------------------------------------------------------------------------

/// TTT-Linear layer with GELU nonlinearity, LayerNorm, and mini-batch support.
///
/// The hidden state is a weight matrix `W_fast` ∈ R^{d_state × d_state}
/// updated by gradient descent on a self-supervised reconstruction loss.
/// Projection matrices `W_K`, `W_V`, `W_Q` are lazily initialized on the
/// first `forward()` call when the input dimension becomes known.
///
/// Per-step computation:
///
/// 1. Project: `k = W_K · x`, `v = W_V · x`, `q = W_Q · x`
/// 2. Inner forward: `z = W_fast · k`
/// 3. Gradient: compute residual, then `grad = residual · k^T`
///    - With prediction feedback: `residual = -pred_err * q / d` (prediction-directed)
///    - Without (warmup): `residual = z - (v - k)` (reconstruction)
/// 4. Update: apply or accumulate gradient (see `batch_mode`)
/// 5. Output: `q + LN(GELU(q + W_fast · q))` (nonlinear with residual)
///
/// # Mini-batch mode
///
/// When `batch_mode` is true, gradients are accumulated instead of applied
/// immediately. Call [`flush_batch`] to apply the averaged gradient as a
/// single update. Sun et al. (2024) prove b=16 gives 1.7 perplexity
/// improvement over b=1 (online).
///
/// # References
///
/// - Sun et al. (2024) "Learning to (Learn at Test Time)" ICML
/// - Liu et al. (2026) "TTT with 1 step + frozen projections = RLS"
/// - Zhang et al. (2025, LaCT) "SwiGLU nonlinear fast weights >> linear"
/// - Behrouz et al. (2025) "Titans: Learning to Memorize at Test Time"
pub(crate) struct TTTLayer {
    // Projection matrices (fixed, initialized randomly)
    w_k: Vec<f64>, // [d_state × d_model] row-major (projects to keys)
    w_v: Vec<f64>, // [d_state × d_model] row-major (projects to values)
    w_q: Vec<f64>, // [d_state × d_model] row-major (projects to queries)

    // Fast weights (the hidden state, updated per step)
    w_fast: Vec<f64>, // [d_state × d_state] row-major

    // Momentum buffer (Titans extension)
    momentum_buf: Vec<f64>, // [d_state × d_state], only used if use_momentum

    // Mini-batch gradient accumulation
    /// Accumulated gradient for mini-batch updates. Only used when `batch_mode` is true.
    accumulated_grad: Vec<f64>, // [d_state × d_state]
    /// Number of gradients accumulated in current batch.
    n_accumulated: usize,
    /// Whether to accumulate gradients instead of applying immediately.
    pub(crate) batch_mode: bool,

    // Dimensions
    d_model: usize, // input dimension (set on first forward)
    d_state: usize, // fast weight dimension

    // Hyperparameters
    eta: f64,            // inner learning rate
    alpha: f64,          // weight decay (forgetting factor, 0 = no decay)
    use_momentum: bool,  // whether momentum is enabled
    momentum_decay: f64, // momentum coefficient (typically 0.9)

    /// External prediction error signal for prediction-directed fast weight updates.
    ///
    /// When non-zero, replaces the reconstruction residual with a prediction-
    /// directed residual: `-pred_err * q[i] / d`. This makes the fast weight
    /// update directly minimize prediction loss instead of reconstruction loss.
    /// Falls back to reconstruction during warmup (before first prediction).
    pub prediction_feedback: f64,

    initialized: bool,
    rng_state: u64,
}

impl TTTLayer {
    /// Create a new TTT layer.
    ///
    /// Projection matrices are lazily initialized on first `forward()` call
    /// when the input dimension becomes known.
    ///
    /// # Arguments
    ///
    /// * `d_state` — fast weight dimension (output dimension)
    /// * `eta` — inner learning rate for fast-weight gradient steps
    /// * `alpha` — weight decay / forgetting factor (0.0 = no decay)
    /// * `use_momentum` — enable Titans-style momentum
    /// * `momentum_decay` — momentum coefficient (typically 0.9)
    /// * `seed` — RNG seed (0 is mapped to 1 for xorshift64 safety)
    pub fn new(
        d_state: usize,
        eta: f64,
        alpha: f64,
        use_momentum: bool,
        momentum_decay: f64,
        seed: u64,
    ) -> Self {
        Self {
            w_k: Vec::new(),
            w_v: Vec::new(),
            w_q: Vec::new(),
            w_fast: vec![0.0; d_state * d_state],
            momentum_buf: if use_momentum {
                vec![0.0; d_state * d_state]
            } else {
                Vec::new()
            },
            accumulated_grad: vec![0.0; d_state * d_state],
            n_accumulated: 0,
            batch_mode: false,
            d_model: 0,
            d_state,
            eta,
            alpha,
            use_momentum,
            momentum_decay,
            prediction_feedback: 0.0,
            initialized: false,
            rng_state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Forward pass: project input, update fast weights, produce output.
    ///
    /// Returns output features of dimension `d_state`.
    ///
    /// Per-step computation:
    /// 1. Project: `k = W_K · x`, `v = W_V · x`, `q = W_Q · x`
    /// 2. Inner forward: `z = W_fast · k`
    /// 3. Gradient: compute residual, then `grad = residual · k^T`
    ///    - With prediction feedback: `residual = -pred_err * q / d` (prediction-directed)
    ///    - Without (warmup): `residual = z - (v - k)` (reconstruction)
    /// 4. Update: apply or accumulate gradient (batch_mode controls behavior)
    /// 5. Output: `q + LN(GELU(q + W_fast · q))` (nonlinear with residual)
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&mut self, features: &[f64]) -> Vec<f64> {
        self.ensure_init(features.len());

        let d = self.d_state;

        // Normalize input to prevent large projections (L2 norm, floor at 1.0)
        let input_norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
        let normalized: Vec<f64> = features.iter().map(|x| x / input_norm).collect();

        // 1. Project: k, v, q (each d_state-dimensional)
        let mut k = mat_vec_mul(&self.w_k, &normalized, d);
        let v = mat_vec_mul(&self.w_v, &normalized, d);
        let mut q = mat_vec_mul(&self.w_q, &normalized, d);

        // L2 normalize keys and queries (Titans recommendation for training stability)
        let k_norm = k.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
        for ki in k.iter_mut() {
            *ki /= k_norm;
        }
        let q_norm = q.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
        for qi in q.iter_mut() {
            *qi /= q_norm;
        }

        // 2. Inner forward: z = W_fast * k
        let z = fast_mat_vec(&self.w_fast, &k, d);

        // 3. Compute residual for fast weight update.
        //    Gradient: dW = residual * k^T (rank-1 outer product).
        let mut residual = vec![0.0; d];
        if self.prediction_feedback.abs() > 1e-15 {
            // Prediction-directed: use prediction error projected onto query space.
            // d(pred_err^2)/dW_fast ∝ -pred_err * q (for the output dimension).
            // This makes W_fast directly minimize prediction loss.
            let pred_err = self.prediction_feedback;
            for i in 0..d {
                residual[i] = -pred_err * q[i] / (d as f64);
            }
        } else {
            // Fallback to reconstruction during warmup (before first prediction).
            // L = ||z - (v - k)||^2, dL/dW = 2 * residual * k^T
            // (the 2 is absorbed into eta).
            for i in 0..d {
                residual[i] = z[i] - (v[i] - k[i]);
            }
        }

        // 4. Update fast weights (with per-element gradient clipping)
        //    In batch_mode: accumulate clipped gradients for later flush.
        //    Otherwise: apply immediately (legacy online behavior).
        if self.batch_mode {
            // Accumulate gradient for mini-batch update
            for i in 0..d {
                for j in 0..d {
                    let idx = i * d + j;
                    let grad = residual[i] * k[j];
                    let clipped_grad = grad.clamp(-1.0, 1.0);
                    self.accumulated_grad[idx] += clipped_grad;
                }
            }
            self.n_accumulated += 1;
        } else if self.use_momentum {
            // Titans: S = momentum_decay * S - eta * (residual * k^T)
            //         W = (1 - alpha) * W + S
            for i in 0..d {
                for j in 0..d {
                    let idx = i * d + j;
                    let grad = residual[i] * k[j];
                    let clipped_grad = grad.clamp(-1.0, 1.0);
                    self.momentum_buf[idx] =
                        self.momentum_decay * self.momentum_buf[idx] - self.eta * clipped_grad;
                    self.w_fast[idx] =
                        (1.0 - self.alpha) * self.w_fast[idx] + self.momentum_buf[idx];
                }
            }
        } else {
            // Standard TTT: W = (1 - alpha) * W - eta * (residual * k^T)
            for i in 0..d {
                for j in 0..d {
                    let idx = i * d + j;
                    let grad = residual[i] * k[j];
                    let clipped_grad = grad.clamp(-1.0, 1.0);
                    self.w_fast[idx] =
                        (1.0 - self.alpha) * self.w_fast[idx] - self.eta * clipped_grad;
                }
            }
        }

        // Fast weight magnitude guard: rescale if norm exceeds threshold
        let w_max = self.w_fast.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if w_max > 1e4 || w_max.is_nan() {
            let scale = 1e3 / w_max.max(1e-15);
            for w in &mut self.w_fast {
                *w *= scale;
            }
        }

        // 5. Output: q + LN(GELU(q + W_fast * q))
        //    GELU nonlinearity replaces tanh — Zhang et al. (2025, LaCT) proved
        //    nonlinear fast weights >> linear. GELU is smoother than ReLU and matches
        //    the transformer FFN convention. LayerNorm stabilizes the activated output,
        //    and the residual connection preserves the query signal through the layer.
        let wq = fast_mat_vec(&self.w_fast, &q, d);

        // Apply GELU activation to (q + W_fast * q)
        let mut activated = vec![0.0; d];
        for i in 0..d {
            activated[i] = gelu(q[i] + wq[i]);
        }

        // Layer normalization (zero mean, unit variance)
        let mean = activated.iter().sum::<f64>() / d as f64;
        let var = activated
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / d as f64;
        let std_inv = 1.0 / (var + 1e-8).sqrt();
        for a in activated.iter_mut() {
            *a = (*a - mean) * std_inv;
        }

        // Residual connection: output = q + LN(GELU(q + W_fast * q))
        let mut output = vec![0.0; d];
        for i in 0..d {
            output[i] = q[i] + activated[i];
        }

        output
    }

    /// Output dimension (= `d_state`).
    pub fn output_dim(&self) -> usize {
        self.d_state
    }

    /// Set the inner learning rate for fast weight updates.
    ///
    /// Used by [`StreamingTTT`](super::StreamingTTT) to dynamically modulate
    /// eta based on prediction uncertainty.
    #[inline]
    pub fn set_eta(&mut self, eta: f64) {
        self.eta = eta;
    }

    /// Reset only the fast weights (preserves projections and initialization).
    ///
    /// Used by drift detection: when prediction error spikes, clear the fast
    /// weight matrix to allow clean adaptation to the new regime.
    pub fn reset_fast_weights(&mut self) {
        self.w_fast.fill(0.0);
        if self.use_momentum {
            self.momentum_buf.fill(0.0);
        }
        self.accumulated_grad.fill(0.0);
        self.n_accumulated = 0;
        self.prediction_feedback = 0.0;
    }

    /// Full reset including projections (returns to uninitialized state).
    pub fn reset_full(&mut self) {
        self.w_fast.fill(0.0);
        if self.use_momentum {
            self.momentum_buf.fill(0.0);
        }
        self.accumulated_grad.fill(0.0);
        self.n_accumulated = 0;
        self.w_k.clear();
        self.w_v.clear();
        self.w_q.clear();
        self.d_model = 0;
        self.prediction_feedback = 0.0;
        self.initialized = false;
    }

    /// Flush the accumulated mini-batch gradient and apply it as a single update.
    ///
    /// Averages the accumulated rank-1 gradients over the batch and applies one
    /// update to W_fast. Sun et al. (2024) prove that batch averaging (b=16)
    /// gives better-conditioned updates than applying 16 individual rank-1 steps.
    ///
    /// No-op if no gradients have been accumulated.
    pub(crate) fn flush_batch(&mut self) {
        if self.n_accumulated == 0 {
            return;
        }
        let d = self.d_state;
        let n = self.n_accumulated as f64;

        // Apply averaged gradient
        if self.use_momentum {
            for i in 0..d {
                for j in 0..d {
                    let idx = i * d + j;
                    let avg_grad = self.accumulated_grad[idx] / n;
                    self.momentum_buf[idx] =
                        self.momentum_decay * self.momentum_buf[idx] - self.eta * avg_grad;
                    self.w_fast[idx] =
                        (1.0 - self.alpha) * self.w_fast[idx] + self.momentum_buf[idx];
                }
            }
        } else {
            for i in 0..d {
                for j in 0..d {
                    let idx = i * d + j;
                    let avg_grad = self.accumulated_grad[idx] / n;
                    self.w_fast[idx] = (1.0 - self.alpha) * self.w_fast[idx] - self.eta * avg_grad;
                }
            }
        }

        // Reset accumulator
        self.accumulated_grad.fill(0.0);
        self.n_accumulated = 0;
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl TTTLayer {
    /// Lazily initialize projection matrices when input dimension is known.
    fn ensure_init(&mut self, d_model: usize) {
        if self.initialized {
            return;
        }
        self.d_model = d_model;
        let d = self.d_state;

        if d == d_model {
            // Identity: no random subspace noise. Fast weights learn in input space.
            self.w_k = (0..d * d)
                .map(|idx| if idx / d == idx % d { 1.0 } else { 0.0 })
                .collect();
            self.w_v = self.w_k.clone();
            self.w_q = self.w_k.clone();
        } else {
            // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
            let scale = (2.0 / (d_model + d) as f64).sqrt();

            self.w_k = random_matrix(&mut self.rng_state, d, d_model, scale);
            self.w_v = random_matrix(&mut self.rng_state, d, d_model, scale);
            self.w_q = random_matrix(&mut self.rng_state, d, d_model, scale);
        }

        self.initialized = true;
    }

    /// Compute gradients for W_Q, W_K, W_V given current state.
    ///
    /// Returns `(grad_W_Q, grad_W_K, grad_W_V)` as flat `Vec<f64>` in
    /// row-major `[d_state x d_model]` layout.
    ///
    /// - **W_Q gradient** — through prediction loss: `output = (I + W_fast) @ q`,
    ///   `q = W_Q @ x_norm`, chain rule gives `d_loss/d_W_Q[i,j] = d_loss/d_q[i] * x_norm[j]`.
    /// - **W_K / W_V gradient** — through reconstruction loss:
    ///   `||W_fast @ k - (v - k)||^2`, differentiated w.r.t. k and v.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn compute_projection_gradients(
        &self,
        features: &[f64],
        pred_error: f64,
        readout_weights: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let d = self.d_state;

        // Normalize input (same as forward)
        let input_norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
        let x_norm: Vec<f64> = features.iter().map(|x| x / input_norm).collect();
        let n_input = x_norm.len();

        // Project (same as forward)
        let mut k = mat_vec_mul(&self.w_k, &x_norm, d);
        let v = mat_vec_mul(&self.w_v, &x_norm, d);
        let mut q = mat_vec_mul(&self.w_q, &x_norm, d);

        // L2 normalize keys and queries (same as forward)
        let k_norm = k.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
        for ki in k.iter_mut() {
            *ki /= k_norm;
        }
        let q_norm = q.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
        for qi in q.iter_mut() {
            *qi /= q_norm;
        }

        // --- W_Q gradient (prediction-directed) ---
        // output = (I + W_fast) @ q
        // loss = (target - w_r^T @ output)^2
        // d_loss/d_q[i] = -2 * error * sum_k(w_r[k] * (delta_{ki} + W_fast[k,i]))
        // d_loss/d_W_Q[i,j] = d_loss/d_q[i] * x_norm[j]
        let mut d_loss_d_q = vec![0.0; d];
        for i in 0..d {
            let mut sum = 0.0;
            for k_idx in 0..d {
                let w_fast_ki = self.w_fast[k_idx * d + i];
                let identity = if k_idx == i { 1.0 } else { 0.0 };
                sum += readout_weights.get(k_idx).copied().unwrap_or(0.0) * (identity + w_fast_ki);
            }
            d_loss_d_q[i] = -2.0 * pred_error * sum;
        }

        let mut grad_wq = vec![0.0; d * n_input];
        for i in 0..d {
            for j in 0..n_input {
                grad_wq[i * n_input + j] = d_loss_d_q[i] * x_norm[j];
            }
        }

        // --- W_K gradient (reconstruction-directed) ---
        // residual = W_fast @ k - (v - k) = W_fast @ k - v + k
        // recon_loss = ||residual||^2
        // d_recon/d_k[i] = 2 * sum_j(residual[j] * (W_fast[j,i] + delta_{ji}))
        let z = fast_mat_vec(&self.w_fast, &k, d);
        let mut residual = vec![0.0; d];
        for i in 0..d {
            residual[i] = z[i] - v[i] + k[i];
        }

        let mut d_recon_d_k = vec![0.0; d];
        for i in 0..d {
            let mut sum = 0.0;
            for j in 0..d {
                let w_fast_ji = self.w_fast[j * d + i];
                let identity = if j == i { 1.0 } else { 0.0 };
                sum += residual[j] * (w_fast_ji + identity);
            }
            d_recon_d_k[i] = 2.0 * sum;
        }

        let mut grad_wk = vec![0.0; d * n_input];
        for i in 0..d {
            for j in 0..n_input {
                grad_wk[i * n_input + j] = d_recon_d_k[i] * x_norm[j];
            }
        }

        // --- W_V gradient (reconstruction-directed) ---
        // d_recon/d_v[i] = -2 * residual[i]
        let mut grad_wv = vec![0.0; d * n_input];
        for i in 0..d {
            for j in 0..n_input {
                grad_wv[i * n_input + j] = -2.0 * residual[i] * x_norm[j];
            }
        }

        (grad_wq, grad_wk, grad_wv)
    }

    /// Apply gradient updates to projection matrices W_Q, W_K, W_V.
    ///
    /// Each gradient must have the same layout as its corresponding matrix:
    /// `[d_state x d_model]` in row-major order.
    pub(crate) fn update_projections(
        &mut self,
        grad_wq: &[f64],
        grad_wk: &[f64],
        grad_wv: &[f64],
        lr: f64,
    ) {
        for (w, g) in self.w_q.iter_mut().zip(grad_wq.iter()) {
            *w -= lr * g;
        }
        for (w, g) in self.w_k.iter_mut().zip(grad_wk.iter()) {
            *w -= lr * g;
        }
        for (w, g) in self.w_v.iter_mut().zip(grad_wv.iter()) {
            *w -= lr * g;
        }
    }

    /// Force initialization of projections without running a full forward pass.
    ///
    /// This is used by `pretrain_projections()` to ensure the projection
    /// matrices exist before computing gradients.
    pub(crate) fn ensure_initialized(&mut self, d_model: usize) {
        self.ensure_init(d_model);
    }

    /// Inject externally provided projection matrices and mark as initialized.
    ///
    /// Each matrix is stored directly — no validation of element values.
    /// The caller is responsible for ensuring the lengths are correct
    /// (`d_state * d_model` elements each in row-major order).
    pub(crate) fn set_projections(&mut self, w_k: Vec<f64>, w_v: Vec<f64>, w_q: Vec<f64>) {
        self.w_k = w_k;
        self.w_v = w_v;
        self.w_q = w_q;
        self.initialized = true;
    }
}

// ---------------------------------------------------------------------------
// Free-function helpers (avoids borrow conflicts in forward)
// ---------------------------------------------------------------------------

/// GELU activation function (Gaussian Error Linear Unit).
///
/// Approximation from Hendrycks & Gimpel (2016):
///   GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// Zhang et al. (2025, LaCT) proved nonlinear fast weights (SwiGLU/GELU)
/// strictly dominate linear fast weights in expressiveness.
#[inline]
fn gelu(x: f64) -> f64 {
    let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Generate a random matrix [rows x cols] with elements ~ N(0, scale).
fn random_matrix(rng: &mut u64, rows: usize, cols: usize, scale: f64) -> Vec<f64> {
    let n = rows * cols;
    let mut mat = Vec::with_capacity(n);
    for _ in 0..n {
        mat.push(standard_normal(rng) * scale);
    }
    mat
}

/// Row-major matrix-vector multiply: result = W * x where W is [rows x cols].
fn mat_vec_mul(w: &[f64], x: &[f64], rows: usize) -> Vec<f64> {
    let cols = x.len();
    let mut result = vec![0.0; rows];
    for (i, out) in result.iter_mut().enumerate() {
        let row_start = i * cols;
        let mut sum = 0.0;
        for j in 0..cols {
            sum += w[row_start + j] * x[j];
        }
        *out = sum;
    }
    result
}

/// Fast weights matrix-vector: result = W * x where W is [d x d].
fn fast_mat_vec(w: &[f64], x: &[f64], d: usize) -> Vec<f64> {
    let mut result = vec![0.0; d];
    for (i, out) in result.iter_mut().enumerate() {
        let row_start = i * d;
        let mut sum = 0.0;
        for j in 0..d {
            sum += w[row_start + j] * x[j];
        }
        *out = sum;
    }
    result
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_uninit() {
        let layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, 42);
        assert!(!layer.initialized, "should be uninitialized after new()");
        assert_eq!(layer.d_state, 8, "d_state should be 8");
        assert!(
            layer.w_fast.iter().all(|&v| v == 0.0),
            "w_fast should be all zeros initially"
        );
    }

    #[test]
    fn forward_initializes_projections() {
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, 42);
        assert!(!layer.initialized, "should start uninitialized");
        assert!(
            layer.w_k.is_empty(),
            "w_k should be empty before first forward"
        );

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let _ = layer.forward(&input);

        assert!(
            layer.initialized,
            "should be initialized after first forward"
        );
        assert_eq!(layer.d_model, 5, "d_model should be set to input length");
        assert_eq!(
            layer.w_k.len(),
            8 * 5,
            "w_k should be [d_state x d_model] = [8 x 5]"
        );
        assert_eq!(
            layer.w_v.len(),
            8 * 5,
            "w_v should be [d_state x d_model] = [8 x 5]"
        );
        assert_eq!(
            layer.w_q.len(),
            8 * 5,
            "w_q should be [d_state x d_model] = [8 x 5]"
        );
    }

    #[test]
    fn forward_output_dimension() {
        let mut layer = TTTLayer::new(16, 0.01, 0.0, false, 0.0, 42);
        let input = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&input);
        assert_eq!(
            output.len(),
            16,
            "output dimension should equal d_state=16, got {}",
            output.len()
        );
    }

    #[test]
    fn forward_output_finite() {
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, 42);
        let input = vec![0.5, -0.3, 1.2, 0.0, -1.0];
        let output = layer.forward(&input);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] = {} is not finite", i, v);
        }
    }

    #[test]
    fn fast_weights_update() {
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, 42);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let _ = layer.forward(&input);

        let changed = layer.w_fast.iter().any(|&v| v != 0.0);
        assert!(
            changed,
            "w_fast should no longer be all zeros after forward pass"
        );
    }

    #[test]
    fn reset_zeros_fast_weights() {
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, 42);
        let input = vec![1.0, 2.0, 3.0];
        let _ = layer.forward(&input);

        // Verify weights are non-zero after forward
        assert!(
            layer.w_fast.iter().any(|&v| v != 0.0),
            "w_fast should be non-zero after forward"
        );

        layer.reset_full();

        assert!(
            layer.w_fast.iter().all(|&v| v == 0.0),
            "w_fast should be all zeros after reset"
        );
    }

    #[test]
    fn reset_full_clears_projections() {
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, 42);
        let input = vec![1.0, 2.0, 3.0];
        let _ = layer.forward(&input);

        layer.reset_full();

        // Projections should be cleared
        assert!(
            !layer.initialized,
            "initialized should be false after reset_full"
        );
        assert!(
            layer.w_k.is_empty(),
            "w_k should be cleared after reset_full"
        );

        // Should still work — projections will be re-initialized on next forward
        let output = layer.forward(&input);
        assert_eq!(output.len(), 8, "forward should still work after reset");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "output[{}] = {} is not finite after reset",
                i,
                v
            );
        }
    }

    #[test]
    fn reset_full_clears_everything() {
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, 42);
        let input = vec![1.0, 2.0, 3.0];
        let _ = layer.forward(&input);

        layer.reset_full();

        assert!(
            !layer.initialized,
            "initialized should be false after reset_full"
        );
        assert!(
            layer.w_k.is_empty(),
            "w_k should be cleared after reset_full"
        );
        assert!(
            layer.w_v.is_empty(),
            "w_v should be cleared after reset_full"
        );
        assert!(
            layer.w_q.is_empty(),
            "w_q should be cleared after reset_full"
        );
        assert_eq!(layer.d_model, 0, "d_model should be 0 after reset_full");
        assert!(
            layer.w_fast.iter().all(|&v| v == 0.0),
            "w_fast should be all zeros after reset_full"
        );
    }

    #[test]
    fn momentum_changes_behavior() {
        let input = vec![1.0, -0.5, 0.3, 2.0];

        // Without momentum
        let mut layer_no_mom = TTTLayer::new(8, 0.01, 0.0, false, 0.0, 42);
        let _ = layer_no_mom.forward(&input);
        let _ = layer_no_mom.forward(&input);
        let out_no_mom = layer_no_mom.forward(&input);

        // With momentum
        let mut layer_mom = TTTLayer::new(8, 0.01, 0.0, true, 0.9, 42);
        let _ = layer_mom.forward(&input);
        let _ = layer_mom.forward(&input);
        let out_mom = layer_mom.forward(&input);

        let diff: f64 = out_no_mom
            .iter()
            .zip(out_mom.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-10,
            "momentum should produce different output after multiple steps, total diff = {}",
            diff
        );
    }

    #[test]
    fn deterministic_with_seed() {
        let input = vec![0.5, -1.0, 2.0, 0.3];

        let mut layer_a = TTTLayer::new(8, 0.01, 0.0, false, 0.0, 12345);
        let out_a1 = layer_a.forward(&input);
        let out_a2 = layer_a.forward(&input);

        let mut layer_b = TTTLayer::new(8, 0.01, 0.0, false, 0.0, 12345);
        let out_b1 = layer_b.forward(&input);
        let out_b2 = layer_b.forward(&input);

        for i in 0..8 {
            assert!(
                (out_a1[i] - out_b1[i]).abs() < 1e-15,
                "step 1 output[{}] differs: {} vs {}",
                i,
                out_a1[i],
                out_b1[i]
            );
            assert!(
                (out_a2[i] - out_b2[i]).abs() < 1e-15,
                "step 2 output[{}] differs: {} vs {}",
                i,
                out_a2[i],
                out_b2[i]
            );
        }
    }

    #[test]
    fn convergence_on_pattern() {
        // Feed a repeating pattern and verify the reconstruction error decreases.
        // d_state != d_model so Xavier init is used (identity init makes v-k=0).
        let mut layer = TTTLayer::new(4, 0.05, 0.0, false, 0.0, 42);
        let pattern = vec![1.0, 0.0, 0.5, -0.5, 0.3];

        // Collect reconstruction errors over time.
        // Error = ||W_fast * k - (v - k)||^2 after each step.
        let mut errors: Vec<f64> = Vec::new();

        for _ in 0..50 {
            // Compute k, v before forward to measure pre-update error
            // We use forward which updates weights, then measure the next step's error.
            let _ = layer.forward(&pattern);

            // After forward, compute the reconstruction error for the same input
            // (since we feed the same pattern, this measures how well W learned it).
            let k = mat_vec_mul(&layer.w_k, &pattern, layer.d_state);
            let v = mat_vec_mul(&layer.w_v, &pattern, layer.d_state);
            let z = fast_mat_vec(&layer.w_fast, &k, layer.d_state);

            let err: f64 = (0..layer.d_state)
                .map(|i| {
                    let r = z[i] - (v[i] - k[i]);
                    r * r
                })
                .sum();
            errors.push(err);
        }

        // The error in the later half should be significantly lower than the first half
        let first_half_avg: f64 = errors[..25].iter().sum::<f64>() / 25.0;
        let second_half_avg: f64 = errors[25..].iter().sum::<f64>() / 25.0;

        assert!(
            second_half_avg < first_half_avg,
            "reconstruction error should decrease over time: first_half_avg={}, second_half_avg={}",
            first_half_avg,
            second_half_avg
        );
    }
}
