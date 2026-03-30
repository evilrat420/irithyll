//! Core TTT (Test-Time Training) layer with dual-objective fast weight updates.
//!
//! The hidden state is a weight matrix W ("fast weights") updated per step
//! by gradient descent on a self-supervised reconstruction loss:
//! `L = ||W * k_t - (v_t - k_t)||^2`
//!
//! The gradient is a rank-1 outer product: `dW = residual * k_t^T`.
//!
//! The reconstruction residual is modulated by an external prediction error
//! signal (`prediction_feedback`): when prediction error is high, fast weights
//! adapt more aggressively; when prediction is accurate, reconstruction drives
//! slow refinement. The inner objective remains reconstruction (preserving
//! TTT's identity), but adaptation magnitude is coupled to prediction quality.
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

/// TTT-Linear layer with optional Titans extensions.
///
/// The hidden state is a weight matrix `W_fast` ∈ R^{d_state × d_state}
/// updated by gradient descent on a self-supervised reconstruction loss at
/// every forward step. Projection matrices `W_K`, `W_V`, `W_Q` are lazily
/// initialized on the first `forward()` call when the input dimension
/// becomes known.
///
/// Per-step computation:
///
/// 1. Project: `k = W_K · x`, `v = W_V · x`, `q = W_Q · x`
/// 2. Inner forward: `z = W_fast · k`
/// 3. Gradient: `residual = z - (v - k)`, `grad = residual · k^T`
///    - Dual-objective: scale residual by `prediction_feedback` magnitude
/// 4. Update: `W_fast = (1 - α) · W_fast - η · grad` (with optional momentum)
/// 5. Output: `output = q + W_fast · q` (residual connection)
///
/// # References
///
/// - Sun et al. (2024) "Learning to (Learn at Test Time)" ICML
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

    // Dimensions
    d_model: usize, // input dimension (set on first forward)
    d_state: usize, // fast weight dimension

    // Hyperparameters
    eta: f64,            // inner learning rate
    alpha: f64,          // weight decay (forgetting factor, 0 = no decay)
    use_momentum: bool,  // whether momentum is enabled
    momentum_decay: f64, // momentum coefficient (typically 0.9)

    /// External prediction error signal for dual-objective fast weight updates.
    ///
    /// When non-zero, scales the reconstruction residual by the prediction error
    /// magnitude: high prediction error → more aggressive fast weight adaptation,
    /// low prediction error → reconstruction-only refinement.
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
    /// 3. Gradient: `residual = z - (v - k)`, `grad = residual · k^T`
    ///    - Dual-objective: scale residual by `prediction_feedback` magnitude
    /// 4. Update: `W_fast = (1 - α) · W_fast - η · grad` (with optional momentum)
    /// 5. Output: `output = q + W_fast · q` (residual connection)
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&mut self, features: &[f64]) -> Vec<f64> {
        self.ensure_init(features.len());

        let d = self.d_state;

        // Normalize input to prevent large projections (L2 norm, floor at 1.0)
        let input_norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
        let normalized: Vec<f64> = features.iter().map(|x| x / input_norm).collect();

        // 1. Project: k, v, q (each d_state-dimensional)
        let k = mat_vec_mul(&self.w_k, &normalized, d);
        let v = mat_vec_mul(&self.w_v, &normalized, d);
        let q = mat_vec_mul(&self.w_q, &normalized, d);

        // 2. Inner forward: z = W_fast * k
        let z = fast_mat_vec(&self.w_fast, &k, d);

        // 3. Gradient of L = ||z - (v - k)||^2
        //    dL/dW = 2 * residual * k^T  (rank-1 outer product)
        //    We absorb the 2 into eta for simplicity.
        let mut residual = vec![0.0; d];
        for i in 0..d {
            residual[i] = z[i] - (v[i] - k[i]);
        }

        // Dual-objective: modulate reconstruction residual by prediction error.
        // When prediction error is high, scale up fast weight adaptation.
        // When prediction is accurate, reconstruction drives slow refinement.
        if self.prediction_feedback.abs() > 1e-15 {
            let pred_scale = (self.prediction_feedback.abs()).clamp(0.1, 3.0);
            for r in &mut residual {
                *r *= pred_scale;
            }
        }

        // 4. Update fast weights (with gradient clipping to prevent explosion)
        if self.use_momentum {
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

        // Emergency: scale down fast weights if they explode
        let w_max = self.w_fast.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if w_max > 1e4 || w_max.is_nan() {
            let scale = 1e3 / w_max.max(1e-15);
            for w in &mut self.w_fast {
                *w *= scale;
            }
        }

        // 5. Output: q + W_fast * q (residual connection)
        let mut wq = fast_mat_vec(&self.w_fast, &q, d);
        for val in wq.iter_mut() {
            *val = val.clamp(-10.0, 10.0);
        }
        let mut output = vec![0.0; d];
        for i in 0..d {
            output[i] = q[i] + wq[i];
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

    /// Full reset including projections (returns to uninitialized state).
    pub fn reset_full(&mut self) {
        self.w_fast.fill(0.0);
        if self.use_momentum {
            self.momentum_buf.fill(0.0);
        }
        self.w_k.clear();
        self.w_v.clear();
        self.w_q.clear();
        self.d_model = 0;
        self.prediction_feedback = 0.0;
        self.initialized = false;
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

        // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        let scale = (2.0 / (d_model + d) as f64).sqrt();

        self.w_k = random_matrix(&mut self.rng_state, d, d_model, scale);
        self.w_v = random_matrix(&mut self.rng_state, d, d_model, scale);
        self.w_q = random_matrix(&mut self.rng_state, d, d_model, scale);

        self.initialized = true;
    }
}

// ---------------------------------------------------------------------------
// Free-function helpers (avoids borrow conflicts in forward)
// ---------------------------------------------------------------------------

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
        // Feed a repeating 2D pattern and verify the reconstruction error decreases.
        // Pattern: x = [1, 0] repeatedly. The fast weights should learn to
        // reconstruct (v - k) from k, reducing the loss over time.
        let mut layer = TTTLayer::new(4, 0.05, 0.0, false, 0.0, 42);
        let pattern = vec![1.0, 0.0, 0.5, -0.5];

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
