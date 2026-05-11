//! Core TTT (Test-Time Training) layer with prediction-directed fast weight updates.
//!
//! The hidden state is a weight matrix W ("fast weights") updated per step
//! by gradient descent. The gradient source depends on whether prediction
//! feedback is available:
//!
//! - **With prediction feedback** (after warmup): the fast weight update is
//!   directed by the prediction error projected onto the query space:
//!   `residual = -pred_err * q` (q is L2-normalized, no /d scaling needed).
//!   This makes W_fast directly minimize prediction loss.
//! - **Without prediction feedback** (during warmup): falls back to the
//!   self-supervised reconstruction loss `L = ||W * k_t - (v_t - k_t)||^2`.
//!
//! The gradient is a rank-1 outer product: `dW = residual * k_t^T`.
//!
//! Based on Sun et al. (2024) "Learning to (Learn at Test Time)" ICML,
//! with Titans extensions (Behrouz et al., 2025): weight decay (forgetting)
//! and optional momentum for temporal coherence.

use irithyll_core::rng::standard_normal;

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
///    - With prediction feedback: `residual = -pred_err * q` (prediction-directed)
///    - Without (warmup): `residual = z - (v - k)` (reconstruction)
/// 4. Update: apply gradient immediately (single-sample streaming)
/// 5. Output: `q + LN(GELU(q + W_fast · k))` (nonlinear with residual)
///
/// # Mini-batch mode
///
/// When `batch_mode` is true (batch_size > 1), gradients are accumulated
/// instead of applied immediately. Call [`flush_batch`] to apply the averaged
/// gradient as a single update. Default is single-sample streaming (batch_size=1)
/// for maximum responsiveness to regime changes.
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
    nesterov: bool,      // Nesterov-accelerated momentum (Titans)
    alpha_warmup: usize, // steps to linearly ramp alpha from 0 (0 = disabled)
    step_count: u64,     // forward() call counter for alpha warmup schedule
    /// The alpha actually applied at the last forward() call (post-warmup ramp).
    /// Exposed via `effective_alpha()` so diagnostics reflect runtime state, not
    /// the static config value.
    effective_alpha: f64,

    /// Optional 2-layer MLP fast-weights (Titans §3.1, Behrouz et al. 2025).
    ///
    /// When `Some`, memory uses W2·σ(W1·k) instead of the linear W_fast·k.
    /// Strictly more expressive than the single-layer degenerate case, which
    /// Liu et al. (2026) proved equals online linear regression (RLS).
    mlp_w1: Option<Vec<f64>>, // [hidden_dim × d_state] row-major; None for Linear
    mlp_w2: Option<Vec<f64>>, // [d_state × hidden_dim] row-major; None for Linear
    mlp_v1: Option<Vec<f64>>, // momentum buffer for W1 (same shape as mlp_w1)
    mlp_v2: Option<Vec<f64>>, // momentum buffer for W2 (same shape as mlp_w2)
    mlp_hidden_dim: usize,    // 0 when Linear

    /// External prediction error signal for prediction-directed fast weight updates.
    ///
    /// When non-zero, replaces the reconstruction residual with a prediction-
    /// directed residual: `-pred_err * q[i]`. The query vector q is already
    /// L2-normalized, so no /d scaling is needed. This makes the fast weight
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
    /// * `nesterov` — use Nesterov-accelerated momentum (only when momentum enabled)
    /// * `alpha_warmup` — steps to linearly ramp alpha from 0 (0 = disabled)
    /// * `mlp_hidden_dim` — 0 for Linear memory; > 0 for 2-layer MLP memory (Titans §3.1)
    /// * `seed` — RNG seed (0 is mapped to 1 for xorshift64 safety)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_state: usize,
        eta: f64,
        alpha: f64,
        use_momentum: bool,
        momentum_decay: f64,
        nesterov: bool,
        alpha_warmup: usize,
        mlp_hidden_dim: usize,
        seed: u64,
    ) -> Self {
        let (mlp_w1, mlp_w2, mlp_v1, mlp_v2) = if mlp_hidden_dim > 0 {
            let mut rng_init = if seed == 0 { 1 } else { seed };
            // Xavier init: scale = sqrt(2 / (fan_in + fan_out))
            let scale_w1 = (2.0 / (d_state + mlp_hidden_dim) as f64).sqrt();
            let scale_w2 = (2.0 / (mlp_hidden_dim + d_state) as f64).sqrt();
            let w1 = random_matrix(&mut rng_init, mlp_hidden_dim, d_state, scale_w1);
            let w2 = random_matrix(&mut rng_init, d_state, mlp_hidden_dim, scale_w2);
            let v1 = vec![0.0; mlp_hidden_dim * d_state];
            let v2 = vec![0.0; d_state * mlp_hidden_dim];
            (Some(w1), Some(w2), Some(v1), Some(v2))
        } else {
            (None, None, None, None)
        };

        Self {
            w_k: Vec::new(),
            w_v: Vec::new(),
            w_q: Vec::new(),
            w_fast: if mlp_hidden_dim == 0 {
                vec![0.0; d_state * d_state]
            } else {
                Vec::new() // not used in MLP mode
            },
            momentum_buf: if use_momentum && mlp_hidden_dim == 0 {
                vec![0.0; d_state * d_state]
            } else {
                Vec::new()
            },
            accumulated_grad: if mlp_hidden_dim == 0 {
                vec![0.0; d_state * d_state]
            } else {
                Vec::new() // not used in MLP mode
            },
            n_accumulated: 0,
            batch_mode: false,
            d_model: 0,
            d_state,
            eta,
            alpha,
            use_momentum,
            momentum_decay,
            nesterov,
            alpha_warmup,
            step_count: 0,
            effective_alpha: alpha,
            mlp_w1,
            mlp_w2,
            mlp_v1,
            mlp_v2,
            mlp_hidden_dim,
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
    ///    - With prediction feedback: `residual = -pred_err * q` (prediction-directed)
    ///    - Without (warmup): `residual = z - (v - k)` (reconstruction)
    /// 4. Update: apply gradient immediately or accumulate (batch_mode)
    /// 5. Output: `q + LN(GELU(q + W_fast · k))` (nonlinear with residual)
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&mut self, features: &[f64]) -> Vec<f64> {
        self.ensure_init(features.len());

        let d = self.d_state;

        // Increment step counter (used for alpha warmup schedule).
        self.step_count += 1;

        // Compute effective alpha with optional warmup ramp.
        // When alpha_warmup > 0, alpha linearly ramps from 0 over that many steps.
        let effective_alpha = if self.alpha_warmup > 0 {
            self.alpha * (self.step_count as f64 / self.alpha_warmup as f64).min(1.0)
        } else {
            self.alpha
        };
        // Track post-warmup effective alpha for diagnostics (exposes runtime state
        // rather than the static config value, which may differ during warmup).
        self.effective_alpha = effective_alpha;

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

        // 2. Inner forward: z = M(k), where M is the memory module.
        //    Linear path: z = W_fast * k  (degenerate case = online linear regression,
        //    Liu et al. 2026 Thm 5.1).
        //    MLP path: z = W2 · σ(W1 · k)  (Titans §3.1, strictly more expressive;
        //    Behrouz et al. 2025 §5.5 shows L_M >= 2 escapes the RLS degeneracy).
        let (z, mlp_h) = if self.mlp_hidden_dim > 0 {
            // MLP forward: h = σ(W1 · k),  z = W2 · h  (σ = GELU)
            let h_dim = self.mlp_hidden_dim;
            let w1 = self.mlp_w1.as_ref().unwrap();
            let w2 = self.mlp_w2.as_ref().unwrap();
            let h_raw = mat_vec_mul_sq(w1, &k, h_dim, d);
            let h: Vec<f64> = h_raw.iter().map(|&x| gelu(x)).collect();
            let z_mlp = mat_vec_mul_sq(w2, &h, d, h_dim);
            (z_mlp, Some(h))
        } else if self.nesterov && self.use_momentum {
            // Linear + Nesterov look-ahead: compute z at W_fast + momentum-adjusted position.
            // W_lookahead[i] = (1 - effective_alpha) * W_fast[i] + momentum_decay * S[i]
            let mut z_nesterov = vec![0.0; d];
            for i in 0..d {
                let mut sum = 0.0;
                for j in 0..d {
                    let idx = i * d + j;
                    let w_look = (1.0 - effective_alpha) * self.w_fast[idx]
                        + self.momentum_decay * self.momentum_buf[idx];
                    sum += w_look * k[j];
                }
                z_nesterov[i] = sum;
            }
            (z_nesterov, None)
        } else {
            (fast_mat_vec(&self.w_fast, &k, d), None)
        };

        // 3. Compute residual for fast weight update.
        //    For both Linear and MLP paths the output-space residual is the same;
        //    the difference is in how we backpropagate through the memory module.
        let mut residual = vec![0.0; d];
        if self.prediction_feedback.abs() > 1e-15 {
            // Prediction-directed: use prediction error projected onto query space.
            // d(pred_err^2)/dW_fast ∝ -pred_err * q (for the output dimension).
            // This makes W_fast directly minimize prediction loss.
            // No /d division: q is already L2-normalized (unit norm), so the
            // gradient magnitude is proportional to the prediction error.
            let pred_err = self.prediction_feedback;
            for i in 0..d {
                residual[i] = -pred_err * q[i];
            }
        } else {
            // Fallback to reconstruction during warmup (before first prediction).
            // L = ||z - (v - k)||^2, dL/dW = 2 * residual * k^T
            // (the 2 is absorbed into eta).
            for i in 0..d {
                residual[i] = z[i] - (v[i] - k[i]);
            }
        }

        // 4. Update memory weights (with per-element gradient clipping).
        if self.mlp_hidden_dim > 0 {
            // MLP path: backprop through W2 → σ → W1 (chain rule).
            //
            // Forward:  h = GELU(W1 k),   z = W2 h
            // Loss:     L = 0.5 * ||residual||^2   (residual = z - target)
            // dL/dW2:   outer(residual, h)          [d × h_dim]
            // dL/dh:    W2^T · residual             [h_dim]
            // dL/dh_pre: dL/dh ⊙ gelu'(W1 k)       [h_dim]  (elementwise)
            // dL/dW1:   outer(dL/dh_pre, k)         [h_dim × d_state]
            //
            // Titans Eq. 13-14 weight-decay update applied independently to W1, W2:
            //   W_{t+1} = (1 - alpha) * W_t - eta * grad_t
            // (with optional momentum on each, sharing the same decay scalars).
            let h = mlp_h.unwrap(); // guaranteed Some when mlp_hidden_dim > 0
            let h_dim = self.mlp_hidden_dim;
            let w1 = self.mlp_w1.as_ref().unwrap();
            let w2 = self.mlp_w2.as_mut().unwrap();

            // dL/dW2 = outer(residual, h) → shape [d × h_dim]
            // Apply Titans update: W2 = (1-alpha)*W2 - eta*grad_W2 (clipped)
            for i in 0..d {
                for j in 0..h_dim {
                    let idx = i * h_dim + j;
                    let g = (residual[i] * h[j]).clamp(-1.0, 1.0);
                    if self.use_momentum {
                        let v2 = self.mlp_v2.as_mut().unwrap();
                        v2[idx] = self.momentum_decay * v2[idx] - self.eta * g;
                        w2[idx] = (1.0 - effective_alpha) * w2[idx] + v2[idx];
                    } else {
                        w2[idx] = (1.0 - effective_alpha) * w2[idx] - self.eta * g;
                    }
                }
            }

            // dL/dh = W2^T · residual  (before the GELU'(pre) application)
            // Note: w2 is now post-update; use a snapshot from before would
            // be more accurate, but single-step SGD for online learning treats
            // the pre-update weights as the gradient context — same as linear.
            let w2_snap = self.mlp_w2.as_ref().unwrap();
            let mut d_h = vec![0.0f64; h_dim];
            for j in 0..h_dim {
                let mut s = 0.0;
                for i in 0..d {
                    s += w2_snap[i * h_dim + j] * residual[i];
                }
                d_h[j] = s;
            }

            // dL/dh_pre = dL/dh ⊙ GELU'(W1 · k)
            // GELU'(x) ≈ 0.5*(1+tanh(c*(x+a*x^3))) + 0.5*x*tanh'(...)
            // We use the analytic approximation of the derivative.
            let w1_ref = w1; // borrow before mutable borrow of mlp_w1
            let h_pre_raw = mat_vec_mul_sq(w1_ref, &k, h_dim, d);
            let d_h_pre: Vec<f64> = h_pre_raw
                .iter()
                .zip(d_h.iter())
                .map(|(&x, &dh)| dh * gelu_grad(x))
                .collect();

            // dL/dW1 = outer(d_h_pre, k) → shape [h_dim × d_state]
            let w1_mut = self.mlp_w1.as_mut().unwrap();
            for i in 0..h_dim {
                for j in 0..d {
                    let idx = i * d + j;
                    let g = (d_h_pre[i] * k[j]).clamp(-1.0, 1.0);
                    if self.use_momentum {
                        let v1 = self.mlp_v1.as_mut().unwrap();
                        v1[idx] = self.momentum_decay * v1[idx] - self.eta * g;
                        w1_mut[idx] = (1.0 - effective_alpha) * w1_mut[idx] + v1[idx];
                    } else {
                        w1_mut[idx] = (1.0 - effective_alpha) * w1_mut[idx] - self.eta * g;
                    }
                }
            }
        } else if self.batch_mode {
            // Linear path, batch-accumulation
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
            // Linear path, Titans momentum:
            // S = momentum_decay * S - eta * (residual * k^T)
            // W = (1 - alpha) * W + S
            // For Nesterov: gradient was already computed at look-ahead position,
            // so the standard momentum update formula applies.
            for i in 0..d {
                for j in 0..d {
                    let idx = i * d + j;
                    let grad = residual[i] * k[j];
                    let clipped_grad = grad.clamp(-1.0, 1.0);
                    self.momentum_buf[idx] =
                        self.momentum_decay * self.momentum_buf[idx] - self.eta * clipped_grad;
                    self.w_fast[idx] =
                        (1.0 - effective_alpha) * self.w_fast[idx] + self.momentum_buf[idx];
                }
            }
        } else {
            // Linear path, standard TTT: W = (1 - alpha) * W - eta * (residual * k^T)
            for i in 0..d {
                for j in 0..d {
                    let idx = i * d + j;
                    let grad = residual[i] * k[j];
                    let clipped_grad = grad.clamp(-1.0, 1.0);
                    self.w_fast[idx] =
                        (1.0 - effective_alpha) * self.w_fast[idx] - self.eta * clipped_grad;
                }
            }
        }

        // Fast weight magnitude guard: rescale if norm exceeds threshold (linear path only).
        if self.mlp_hidden_dim == 0 {
            let w_max = self.w_fast.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            if w_max > 1e4 || w_max.is_nan() {
                let scale = 1e3 / w_max.max(1e-15);
                for w in &mut self.w_fast {
                    *w *= scale;
                }
            }
        }

        // 5. Readout: q + LN(GELU(q + M(k)))
        //    M(k) = W2·σ(W1·k) (MLP) or W_fast·k (Linear).
        //    The re-read of M(k) uses the *updated* weights — consistent with
        //    how the linear path reads w_fast after the update.
        let wk = if self.mlp_hidden_dim > 0 {
            let h_dim = self.mlp_hidden_dim;
            let w1 = self.mlp_w1.as_ref().unwrap();
            let w2 = self.mlp_w2.as_ref().unwrap();
            let h_raw = mat_vec_mul_sq(w1, &k, h_dim, d);
            let h: Vec<f64> = h_raw.iter().map(|&x| gelu(x)).collect();
            mat_vec_mul_sq(w2, &h, d, h_dim)
        } else {
            fast_mat_vec(&self.w_fast, &k, d)
        };

        // Apply GELU activation to (q + M(k))
        let mut activated = vec![0.0; d];
        for i in 0..d {
            activated[i] = gelu(q[i] + wk[i]);
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

        // Residual connection: output = q + LN(GELU(q + M(k)))
        let mut output = vec![0.0; d];
        for i in 0..d {
            output[i] = q[i] + activated[i];
        }

        output
    }

    /// Prediction-only forward pass: project input and compute output WITHOUT
    /// updating fast weights, momentum, or batch accumulators.
    ///
    /// This is the read-only counterpart of [`forward`]. It runs:
    /// 1. L2 normalize input
    /// 2. Project: `k = W_K · x`, `q = W_Q · x` (no v needed)
    /// 3. L2 normalize k and q
    /// 4. Compute `W_fast · k` using current (frozen) fast weights
    /// 5. Output: `q + LN(GELU(q + W_fast · k))`
    ///
    /// Takes `&self` (not `&mut self`), making it truly side-effect-free.
    /// Returns `d_state`-dimensional output, or zeros if not yet initialized.
    pub fn forward_predict(&self, features: &[f64]) -> Vec<f64> {
        if !self.initialized {
            return vec![0.0; self.d_state];
        }

        let d = self.d_state;

        // Normalize input (same as forward)
        let input_norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
        let normalized: Vec<f64> = features.iter().map(|x| x / input_norm).collect();

        // Project: k and q (no v needed — we skip gradient computation)
        let mut k = mat_vec_mul(&self.w_k, &normalized, d);
        let mut q = mat_vec_mul(&self.w_q, &normalized, d);

        // L2 normalize keys and queries (same as forward)
        let k_norm = k.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
        for ki in k.iter_mut() {
            *ki /= k_norm;
        }
        let q_norm = q.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
        for qi in q.iter_mut() {
            *qi /= q_norm;
        }

        // M(k): Linear = W_fast·k, MLP = W2·σ(W1·k) (read-only)
        let wk = if self.mlp_hidden_dim > 0 {
            let h_dim = self.mlp_hidden_dim;
            let w1 = self.mlp_w1.as_ref().unwrap();
            let w2 = self.mlp_w2.as_ref().unwrap();
            let h_raw = mat_vec_mul_sq(w1, &k, h_dim, d);
            let h: Vec<f64> = h_raw.iter().map(|&x| gelu(x)).collect();
            mat_vec_mul_sq(w2, &h, d, h_dim)
        } else {
            fast_mat_vec(&self.w_fast, &k, d)
        };

        // GELU activation on (q + M(k))
        let mut activated = vec![0.0; d];
        for i in 0..d {
            activated[i] = gelu(q[i] + wk[i]);
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

        // Residual connection: output = q + LN(GELU(q + M(k)))
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

    /// Read-only access to the fast weight matrix (row-major, `[d_state x d_state]`).
    #[allow(dead_code)]
    #[inline]
    pub fn fast_weights(&self) -> &[f64] {
        &self.w_fast
    }

    /// Set the inner learning rate for fast weight updates.
    ///
    /// Used by [`StreamingTTT`](super::StreamingTTT) to dynamically modulate
    /// eta based on prediction uncertainty.
    #[inline]
    pub fn set_eta(&mut self, eta: f64) {
        self.eta = eta;
    }

    /// Set the base weight decay (alpha).
    ///
    /// Updates the stored `alpha` field so that future forward passes compute
    /// `effective_alpha` from the new base. Does not reset the warmup schedule.
    #[inline]
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Return the alpha actually applied at the last `forward()` call.
    ///
    /// During alpha warmup this is less than `config.alpha`; once the warmup
    /// completes it equals `config.alpha`. AutoTuner and diagnostics should
    /// read this value, not the static config field.
    #[inline]
    pub fn effective_alpha(&self) -> f64 {
        self.effective_alpha
    }

    /// Reset only the fast weights (preserves projections and initialization).
    ///
    /// Used by drift detection: when prediction error spikes, clear the fast
    /// weight matrix to allow clean adaptation to the new regime.
    pub fn reset_fast_weights(&mut self) {
        if self.mlp_hidden_dim > 0 {
            // MLP path: zero both W1 and W2 (and momentum buffers).
            if let Some(w) = &mut self.mlp_w1 {
                w.fill(0.0);
            }
            if let Some(w) = &mut self.mlp_w2 {
                w.fill(0.0);
            }
            if let Some(v) = &mut self.mlp_v1 {
                v.fill(0.0);
            }
            if let Some(v) = &mut self.mlp_v2 {
                v.fill(0.0);
            }
        } else {
            self.w_fast.fill(0.0);
            if self.use_momentum {
                self.momentum_buf.fill(0.0);
            }
            self.accumulated_grad.fill(0.0);
            self.n_accumulated = 0;
        }
        self.prediction_feedback = 0.0;
        self.step_count = 0;
    }

    /// Surgically reinitialize a single unit (row) of the fast weight matrix.
    ///
    /// When unit `j` dies (low utility), reinitialize row `j` of W_fast with
    /// Xavier-scaled random values, and zero row `j` of the momentum buffer
    /// and accumulated gradient. This preserves all other units' learned
    /// fast weight representations.
    ///
    /// # Arguments
    ///
    /// * `j` — unit index (row of W_fast) to reinitialize (must be < `d_state`)
    /// * `rng` — mutable RNG state for generating fresh weights
    ///
    /// # Panics
    ///
    /// Panics if `j >= d_state`.
    pub fn reinitialize_unit(&mut self, j: usize, rng: &mut u64) {
        assert!(
            j < self.d_state,
            "unit index {} out of range (d_state={})",
            j,
            self.d_state
        );

        let scale = (2.0 / (self.d_state + self.d_state) as f64).sqrt();
        let row_start = j * self.d_state;

        // Reinit row j of W_fast with Xavier-scaled random values.
        for col in 0..self.d_state {
            self.w_fast[row_start + col] = standard_normal(rng) * scale;
        }

        // Zero row j of momentum buffer (if momentum is enabled).
        if self.use_momentum {
            for col in 0..self.d_state {
                self.momentum_buf[row_start + col] = 0.0;
            }
        }

        // Zero row j of accumulated gradient.
        for col in 0..self.d_state {
            self.accumulated_grad[row_start + col] = 0.0;
        }
    }

    /// Full reset including projections (returns to uninitialized state).
    pub fn reset_full(&mut self) {
        self.reset_fast_weights();
        self.w_fast.fill(0.0);
        self.w_k.clear();
        self.w_v.clear();
        self.w_q.clear();
        self.d_model = 0;
        self.initialized = false;
    }

    /// Flush the accumulated mini-batch gradient and apply it as a single update.
    ///
    /// Averages the accumulated rank-1 gradients over the batch and applies one
    /// update to W_fast. Only used when `batch_size > 1` (non-default). With the
    /// default `batch_size = 1`, gradients are applied immediately per sample.
    ///
    /// No-op if no gradients have been accumulated.
    pub(crate) fn flush_batch(&mut self) {
        if self.n_accumulated == 0 {
            return;
        }
        let d = self.d_state;
        let n = self.n_accumulated as f64;

        // Compute effective alpha with optional warmup ramp (consistent with forward()).
        let effective_alpha = if self.alpha_warmup > 0 {
            self.alpha * (self.step_count as f64 / self.alpha_warmup as f64).min(1.0)
        } else {
            self.alpha
        };

        // Apply averaged gradient
        if self.use_momentum {
            for i in 0..d {
                for j in 0..d {
                    let idx = i * d + j;
                    let avg_grad = self.accumulated_grad[idx] / n;
                    self.momentum_buf[idx] =
                        self.momentum_decay * self.momentum_buf[idx] - self.eta * avg_grad;
                    self.w_fast[idx] =
                        (1.0 - effective_alpha) * self.w_fast[idx] + self.momentum_buf[idx];
                }
            }
        } else {
            for i in 0..d {
                for j in 0..d {
                    let idx = i * d + j;
                    let avg_grad = self.accumulated_grad[idx] / n;
                    self.w_fast[idx] =
                        (1.0 - effective_alpha) * self.w_fast[idx] - self.eta * avg_grad;
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
        // Linear path: output = (I + W_fast) @ q
        //   d_loss/d_q[i] = -2 * error * sum_k(w_r[k] * (delta_{ki} + W_fast[k,i]))
        // MLP path: W_fast is empty; use identity approximation (I only).
        //   This is an upper-bound approximation — the MLP contribution to the
        //   Jacobian is expensive to compute exactly in the projection gradient pass.
        //   Identity-only gives a correct-sign gradient at low cost.
        let mut d_loss_d_q = vec![0.0; d];
        if self.mlp_hidden_dim == 0 {
            // Linear path: W_fast is available.
            for i in 0..d {
                let mut sum = 0.0;
                for k_idx in 0..d {
                    let w_fast_ki = self.w_fast[k_idx * d + i];
                    let identity = if k_idx == i { 1.0 } else { 0.0 };
                    sum +=
                        readout_weights.get(k_idx).copied().unwrap_or(0.0) * (identity + w_fast_ki);
                }
                d_loss_d_q[i] = -2.0 * pred_error * sum;
            }
        } else {
            // MLP path: approximate Jacobian with identity (I), no W contribution.
            for i in 0..d {
                let identity_contrib = readout_weights.get(i).copied().unwrap_or(0.0);
                d_loss_d_q[i] = -2.0 * pred_error * identity_contrib;
            }
        }

        let mut grad_wq = vec![0.0; d * n_input];
        for i in 0..d {
            for j in 0..n_input {
                grad_wq[i * n_input + j] = d_loss_d_q[i] * x_norm[j];
            }
        }

        // --- W_K gradient (reconstruction-directed) ---
        // Linear path: residual = W_fast @ k - (v - k)
        //   d_recon/d_k[i] = 2 * sum_j(residual[j] * (W_fast[j,i] + delta_{ji}))
        // MLP path: residual = M(k) - (v - k) where M(k) = W2·GELU(W1·k).
        //   d_recon/d_k approximated with identity Jacobian (lower cost).
        let z = if self.mlp_hidden_dim == 0 {
            fast_mat_vec(&self.w_fast, &k, d)
        } else {
            let h_dim = self.mlp_hidden_dim;
            let w1 = self.mlp_w1.as_ref().unwrap();
            let w2 = self.mlp_w2.as_ref().unwrap();
            let h_raw = mat_vec_mul_sq(w1, &k, h_dim, d);
            let h: Vec<f64> = h_raw.iter().map(|&x| gelu(x)).collect();
            mat_vec_mul_sq(w2, &h, d, h_dim)
        };
        let mut residual = vec![0.0; d];
        for i in 0..d {
            residual[i] = z[i] - v[i] + k[i];
        }

        let mut d_recon_d_k = vec![0.0; d];
        if self.mlp_hidden_dim == 0 {
            // Linear path: exact Jacobian.
            for i in 0..d {
                let mut sum = 0.0;
                for j in 0..d {
                    let w_fast_ji = self.w_fast[j * d + i];
                    let identity = if j == i { 1.0 } else { 0.0 };
                    sum += residual[j] * (w_fast_ji + identity);
                }
                d_recon_d_k[i] = 2.0 * sum;
            }
        } else {
            // MLP path: approximate Jacobian with identity (skip M Jacobian).
            for i in 0..d {
                d_recon_d_k[i] = 2.0 * residual[i];
            }
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

/// Analytic GELU gradient: d/dx [0.5·x·(1+tanh(c·(x+a·x³)))].
///
/// Derived from the chain rule on the tanh approximation used in `gelu()`.
/// Used for MLP fast-weight backward pass (Titans §3.1, Behrouz et al. 2025).
#[inline]
fn gelu_grad(x: f64) -> f64 {
    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
    let a = 0.044715_f64;
    let inner = c * (x + a * x * x * x);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * c * (1.0 + 3.0 * a * x * x)
}

/// Row-major matrix-vector multiply with explicit dimensions: result = W * x
/// where W is [rows × cols] stored row-major.
///
/// The `_cols` argument is accepted for call-site clarity (matches x.len())
/// but derived from `x` internally — no separate validation is needed because
/// out-of-bounds access would panic deterministically.
#[inline]
fn mat_vec_mul_sq(w: &[f64], x: &[f64], rows: usize, _cols: usize) -> Vec<f64> {
    mat_vec_mul(w, x, rows)
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
        let layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, false, 0, 0, 42);
        assert!(!layer.initialized, "should be uninitialized after new()");
        assert_eq!(layer.d_state, 8, "d_state should be 8");
        assert!(
            layer.w_fast.iter().all(|&v| v == 0.0),
            "w_fast should be all zeros initially"
        );
    }

    #[test]
    fn forward_initializes_projections() {
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, false, 0, 0, 42);
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
        let mut layer = TTTLayer::new(16, 0.01, 0.0, false, 0.0, false, 0, 0, 42);
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
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, false, 0, 0, 42);
        let input = vec![0.5, -0.3, 1.2, 0.0, -1.0];
        let output = layer.forward(&input);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] = {} is not finite", i, v);
        }
    }

    #[test]
    fn fast_weights_update() {
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, false, 0, 0, 42);
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
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, false, 0, 0, 42);
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
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, false, 0, 0, 42);
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
        let mut layer = TTTLayer::new(8, 0.01, 0.0, false, 0.0, false, 0, 0, 42);
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
        let mut layer_no_mom = TTTLayer::new(8, 0.01, 0.0, false, 0.0, false, 0, 0, 42);
        let _ = layer_no_mom.forward(&input);
        let _ = layer_no_mom.forward(&input);
        let out_no_mom = layer_no_mom.forward(&input);

        // With momentum
        let mut layer_mom = TTTLayer::new(8, 0.01, 0.0, true, 0.9, false, 0, 0, 42);
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

        let mut layer_a = TTTLayer::new(8, 0.01, 0.0, false, 0.0, false, 0, 0, 12345);
        let out_a1 = layer_a.forward(&input);
        let out_a2 = layer_a.forward(&input);

        let mut layer_b = TTTLayer::new(8, 0.01, 0.0, false, 0.0, false, 0, 0, 12345);
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
        let mut layer = TTTLayer::new(4, 0.05, 0.0, false, 0.0, false, 0, 0, 42);
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
