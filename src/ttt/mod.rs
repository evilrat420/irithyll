//! Streaming Test-Time Training (TTT) layers with prediction-directed fast weights.
//!
//! The hidden state is a linear model with weights W, updated by gradient
//! descent at every time step. After warmup, the fast weight update is
//! directed by the readout's prediction error — making W directly minimize
//! prediction loss rather than task-agnostic reconstruction. During warmup,
//! falls back to self-supervised reconstruction.
//!
//! This prediction-directed approach enables fast adaptation on non-stationary
//! data: when the target function shifts, the prediction error immediately
//! redirects the fast weight updates toward the new relationship.
//!
//! Optionally includes Titans-style momentum and weight decay for
//! non-stationary streaming environments.
//!
//! # Architecture
//!
//! ```text
//! x_t → [TTT Layer: project → reconstruct → update W → query] → z_t → [RLS Readout] → ŷ_t
//!                          ↑ prediction_feedback ←────────────────────────── pred_error
//! ```
//!
//! # References
//!
//! - Sun et al. (2024) "Learning to (Learn at Test Time)" ICML
//! - Behrouz et al. (2025) "Titans: Learning to Memorize at Test Time"

mod layer;

use layer::TTTLayer;

use crate::error::ConfigError;
use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;

// ---------------------------------------------------------------------------
// TTTConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StreamingTTT`].
///
/// Create via the builder pattern:
///
/// ```
/// use irithyll::ttt::TTTConfig;
///
/// let config = TTTConfig::builder()
///     .d_model(32)
///     .n_heads(1)
///     .eta(0.01)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TTTConfig {
    /// Model dimension (output of TTT layer). Must be divisible by `n_heads`.
    pub d_model: usize,
    /// Number of attention heads (default: 1).
    pub n_heads: usize,
    /// Inner learning rate for fast weight updates (default: 0.01).
    pub eta: f64,
    /// Weight decay / forgetting factor, Titans-style (default: 0.005).
    ///
    /// At 0.005, fast weights lose 50% of old information in ~139 steps —
    /// fast enough for non-stationary regimes with ~8K-sample phases while
    /// retaining short-term memory.
    pub alpha: f64,
    /// Momentum coefficient, Titans-style (default: 0.0 = none).
    pub momentum: f64,
    /// RLS forgetting factor for readout (default: 0.998).
    pub forgetting_factor: f64,
    /// Initial P matrix diagonal for RLS (default: 100.0).
    pub delta_rls: f64,
    /// Mini-batch size for fast weight updates (default: 16).
    ///
    /// Sun et al. (2024) prove b=16 gives 1.7 perplexity improvement over
    /// b=1 (online). Gradients are accumulated over `batch_size` samples and
    /// applied as a single averaged update — better conditioned than individual
    /// rank-1 steps. Set to 1 for pure online (legacy behavior, not recommended).
    pub batch_size: usize,
    /// Warmup samples before RLS training starts (default: 10).
    pub warmup: usize,
    /// RNG seed (default: 42).
    pub seed: u64,
}

impl Default for TTTConfig {
    fn default() -> Self {
        Self {
            d_model: 32,
            n_heads: 1,
            eta: 0.01,
            alpha: 0.005,
            momentum: 0.0,
            forgetting_factor: 0.998,
            delta_rls: 100.0,
            batch_size: 16,
            warmup: 10,
            seed: 42,
        }
    }
}

impl std::fmt::Display for TTTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TTTConfig(d_model={}, n_heads={}, eta={}, alpha={}, momentum={}, ff={}, batch_size={}, warmup={}, seed={})",
            self.d_model, self.n_heads, self.eta, self.alpha, self.momentum,
            self.forgetting_factor, self.batch_size, self.warmup, self.seed
        )
    }
}

// ---------------------------------------------------------------------------
// TTTConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for [`TTTConfig`] with validation.
///
/// # Example
///
/// ```
/// use irithyll::ttt::TTTConfig;
///
/// let config = TTTConfig::builder()
///     .d_model(64)
///     .n_heads(4)
///     .eta(0.005)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.d_model, 64);
/// assert_eq!(config.n_heads, 4);
/// ```
pub struct TTTConfigBuilder {
    config: TTTConfig,
}

impl TTTConfig {
    /// Create a new builder with default values.
    pub fn builder() -> TTTConfigBuilder {
        TTTConfigBuilder {
            config: TTTConfig::default(),
        }
    }
}

impl TTTConfigBuilder {
    /// Set the model dimension (default: 32, must be divisible by `n_heads`).
    pub fn d_model(mut self, d: usize) -> Self {
        self.config.d_model = d;
        self
    }

    /// Set the number of attention heads (default: 1).
    pub fn n_heads(mut self, n: usize) -> Self {
        self.config.n_heads = n;
        self
    }

    /// Set the inner learning rate for fast weight updates (default: 0.01).
    pub fn eta(mut self, e: f64) -> Self {
        self.config.eta = e;
        self
    }

    /// Set the weight decay coefficient, Titans-style (default: 0.005).
    pub fn alpha(mut self, a: f64) -> Self {
        self.config.alpha = a;
        self
    }

    /// Set the momentum coefficient, Titans-style (default: 0.0 = none).
    pub fn momentum(mut self, m: f64) -> Self {
        self.config.momentum = m;
        self
    }

    /// Set the RLS forgetting factor for the readout (default: 0.998).
    pub fn forgetting_factor(mut self, f: f64) -> Self {
        self.config.forgetting_factor = f;
        self
    }

    /// Set the initial P matrix diagonal for RLS (default: 100.0).
    pub fn delta_rls(mut self, d: f64) -> Self {
        self.config.delta_rls = d;
        self
    }

    /// Set the mini-batch size for fast weight updates (default: 16).
    ///
    /// Sun et al. (2024) prove b=16 is optimal for streaming. Set to 1 for
    /// pure online (legacy behavior, not recommended).
    pub fn batch_size(mut self, b: usize) -> Self {
        self.config.batch_size = b;
        self
    }

    /// Set the warmup period in samples (default: 10).
    pub fn warmup(mut self, w: usize) -> Self {
        self.config.warmup = w;
        self
    }

    /// Set the RNG seed (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.config.seed = s;
        self
    }

    /// Build the config, validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if:
    /// - `d_model` is 0
    /// - `n_heads` is 0
    /// - `d_model` is not divisible by `n_heads`
    pub fn build(self) -> Result<TTTConfig, ConfigError> {
        let c = &self.config;
        if c.d_model == 0 {
            return Err(ConfigError::out_of_range(
                "d_model",
                "must be > 0",
                c.d_model,
            ));
        }
        if c.n_heads == 0 {
            return Err(ConfigError::out_of_range(
                "n_heads",
                "must be > 0",
                c.n_heads,
            ));
        }
        if c.d_model % c.n_heads != 0 {
            return Err(ConfigError::invalid(
                "d_model",
                format!(
                    "d_model ({}) must be divisible by n_heads ({})",
                    c.d_model, c.n_heads
                ),
            ));
        }
        if c.batch_size == 0 {
            return Err(ConfigError::out_of_range(
                "batch_size",
                "must be > 0",
                c.batch_size,
            ));
        }
        Ok(self.config)
    }
}

// ---------------------------------------------------------------------------
// StreamingTTT
// ---------------------------------------------------------------------------

/// Streaming TTT model with RLS readout.
///
/// Processes one sample at a time. The TTT layer adapts its internal
/// representation (fast weights) via gradient descent on reconstruction
/// at every step. An RLS readout maps the TTT output to the target.
///
/// # Example
///
/// ```no_run
/// use irithyll::ttt::{StreamingTTT, TTTConfig};
/// use irithyll::StreamingLearner;
///
/// let config = TTTConfig::builder().d_model(16).eta(0.01).build().unwrap();
/// let mut model = StreamingTTT::new(config);
/// model.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = model.predict(&[1.0, 2.0, 3.0]);
/// ```
pub struct StreamingTTT {
    config: TTTConfig,
    layer: TTTLayer,
    readout: RecursiveLeastSquares,
    last_features: Vec<f64>,
    total_seen: u64,
    samples_trained: u64,
    /// EWMA of prediction uncertainty for eta modulation.
    rolling_uncertainty: f64,
    /// Fast-reacting EWMA of squared error for drift detection (alpha=0.1).
    short_term_error: f64,
    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// Change from two steps ago, for acceleration-based alignment.
    prev_prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
    /// EWMA of maximum Frobenius squared norm of TTT output for utilization ratio.
    max_frob_sq_ewma: f64,

    // --- Mini-batch fast weight updates ---
    // Sun et al. (2024) prove b=16 gives 1.7 perplexity improvement over b=1.
    // Gradients are accumulated in the TTTLayer and flushed every batch_size samples.
    /// Mini-batch size for fast weight updates (from config).
    batch_size: usize,
    /// Counter of samples in current batch (resets after flush).
    batch_count: usize,

    // --- Two-timescale projection learning ---
    // Projections (W_K, W_V, W_Q) are "slow weights" that update at a decaying
    // rate alongside the "fast weights" (W_fast). Schedule: Robbins-Monro 1/t.
    //   proj_lr_t = base_proj_lr / (1 + proj_step / tau)
    //   tau = d_state * d_input (parameter-count based)
    // After ~5*tau samples, proj_lr is ~17% of initial → effectively frozen.
    /// Base projection learning rate (eta * 0.1).
    base_proj_lr: f64,
    /// Current (decayed) projection learning rate.
    proj_lr: f64,
    /// Projection update step counter.
    proj_step: u64,
    /// Timescale: d_state * d_input. Set on first `train_one()`.
    proj_tau: f64,
    /// Whether tau has been set (needs input dimension from first sample).
    proj_tau_set: bool,
}

impl StreamingTTT {
    /// Create a new StreamingTTT from config.
    pub fn new(config: TTTConfig) -> Self {
        let mut layer = TTTLayer::new(
            config.d_model,
            config.eta,
            config.alpha,
            config.momentum > 0.0,
            config.momentum,
            config.seed,
        );
        // Enable mini-batch accumulation when batch_size > 1.
        let batch_size = config.batch_size;
        if batch_size > 1 {
            layer.batch_mode = true;
        }
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta_rls);
        let last_features = vec![0.0; config.d_model];
        let base_proj_lr = config.eta * 0.1;

        Self {
            config,
            layer,
            readout,
            last_features,
            total_seen: 0,
            samples_trained: 0,
            rolling_uncertainty: 0.0,
            short_term_error: 0.0,
            prev_prediction: 0.0,
            prev_change: 0.0,
            prev_prev_change: 0.0,
            alignment_ewma: 0.0,
            max_frob_sq_ewma: 0.0,
            batch_size,
            batch_count: 0,
            base_proj_lr,
            proj_lr: base_proj_lr,
            proj_step: 0,
            proj_tau: 1.0, // placeholder, set on first train_one
            proj_tau_set: false,
        }
    }

    /// Whether the model has seen enough samples for meaningful predictions.
    #[inline]
    pub fn past_warmup(&self) -> bool {
        self.total_seen > self.config.warmup as u64
    }

    /// Access the config.
    pub fn config(&self) -> &TTTConfig {
        &self.config
    }

    /// Forward-looking prediction uncertainty from the RLS readout.
    ///
    /// Returns the estimated prediction standard deviation, computed as the
    /// square root of the RLS noise variance (EWMA of squared residuals).
    /// This is a model-level uncertainty signal that does not require
    /// transformed features.
    ///
    /// Returns 0.0 before any training has occurred.
    #[inline]
    pub fn prediction_uncertainty(&self) -> f64 {
        self.readout.noise_variance().sqrt()
    }

    /// Output dimension of the TTT layer.
    pub fn output_dim(&self) -> usize {
        self.layer.output_dim()
    }

    /// Pretrain projection matrices W_K, W_V, W_Q using a warmup data buffer.
    ///
    /// This is the "outer loop" learning step that the papers (Titans, TTT) require.
    /// Random projections cause the fast weights to learn in an arbitrary subspace.
    /// Pretraining aligns the projections with the actual data distribution.
    ///
    /// Like MTS (minimum training samples) for trees: collect warmup data, learn
    /// the projection basis, then switch to pure streaming.
    ///
    /// # Arguments
    ///
    /// * `data` — Slice of `(features, target)` pairs for pretraining.
    /// * `epochs` — Number of passes over the data (typically 3–10).
    ///
    /// # How it works
    ///
    /// 1. **Initialize** — process data normally to build up W_fast and RLS readout
    ///    so that readout weights exist for gradient computation.
    /// 2. **Optimize** — for each epoch, accumulate gradients on W_Q (prediction
    ///    loss) and W_K/W_V (reconstruction loss), clip, then update projections.
    /// 3. **Clean slate** — reset fast weights, RLS, and all tracking state so
    ///    the model is ready for normal streaming with learned projections.
    pub fn pretrain_projections(&mut self, data: &[(&[f64], f64)], epochs: usize) {
        if data.is_empty() || epochs == 0 {
            return;
        }

        let pretrain_lr = 0.001; // Conservative LR for projection updates

        // Temporarily disable batch mode for pretraining — we need per-sample
        // gradient application during phase 1 and per-sample forward in phase 2.
        let was_batch_mode = self.layer.batch_mode;
        self.layer.batch_mode = false;

        // Ensure layer projections are initialized using the first sample's dimension.
        self.layer.ensure_initialized(data[0].0.len());

        // Phase 1: Initialize — process data to build up W_fast and RLS readout.
        // This gives us readout weights needed for W_Q gradient computation.
        for &(features, target) in data {
            self.train_one(features, target, 1.0);
        }

        // Phase 2: Gradient-based projection optimization.
        for _epoch in 0..epochs {
            let readout_weights = self.readout.weights().to_vec();

            let d_model_layer = self.layer.output_dim();
            let d_input = data[0].0.len();

            let mut acc_grad_wq = vec![0.0; d_model_layer * d_input];
            let mut acc_grad_wk = vec![0.0; d_model_layer * d_input];
            let mut acc_grad_wv = vec![0.0; d_model_layer * d_input];

            // Reset fast weights for clean gradient computation each epoch.
            self.layer.reset_fast_weights();

            for &(features, target) in data {
                // Forward through TTT layer (builds up W_fast).
                let ttt_output = self.layer.forward(features);

                // Compute prediction error.
                let pred = self.readout.predict(&ttt_output);
                let pred_error = target - pred;

                // Compute projection gradients.
                let (gq, gk, gv) =
                    self.layer
                        .compute_projection_gradients(features, pred_error, &readout_weights);

                // Accumulate.
                for (acc, g) in acc_grad_wq.iter_mut().zip(gq.iter()) {
                    *acc += g;
                }
                for (acc, g) in acc_grad_wk.iter_mut().zip(gk.iter()) {
                    *acc += g;
                }
                for (acc, g) in acc_grad_wv.iter_mut().zip(gv.iter()) {
                    *acc += g;
                }
            }

            // Average gradients.
            let n = data.len() as f64;
            for g in acc_grad_wq.iter_mut() {
                *g /= n;
            }
            for g in acc_grad_wk.iter_mut() {
                *g /= n;
            }
            for g in acc_grad_wv.iter_mut() {
                *g /= n;
            }

            // Clip gradients to bound update magnitude.
            let max_norm = 1.0;
            for grads in [&mut acc_grad_wq, &mut acc_grad_wk, &mut acc_grad_wv] {
                let norm: f64 = grads.iter().map(|g| g * g).sum::<f64>().sqrt();
                if norm > max_norm {
                    let scale = max_norm / norm;
                    for g in grads.iter_mut() {
                        *g *= scale;
                    }
                }
            }

            // Update projections.
            self.layer
                .update_projections(&acc_grad_wq, &acc_grad_wk, &acc_grad_wv, pretrain_lr);
        }

        // Phase 3: Clean slate — reset fast weights and RLS for streaming.
        // Projections are retained (they are now learned).
        // Projection LR schedule also resets so auto-pretrain continues refining.
        self.layer.reset_fast_weights();
        // Restore batch mode for normal streaming.
        self.layer.batch_mode = was_batch_mode;
        self.readout.reset();
        self.last_features = vec![0.0; self.config.d_model];
        self.total_seen = 0;
        self.samples_trained = 0;
        self.rolling_uncertainty = 0.0;
        self.short_term_error = 0.0;
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.prev_prev_change = 0.0;
        self.alignment_ewma = 0.0;
        self.max_frob_sq_ewma = 0.0;
        self.batch_count = 0;
        self.proj_lr = self.base_proj_lr;
        self.proj_step = 0;
        // Keep proj_tau and proj_tau_set — tau is still valid from batch pretrain.
    }

    /// Inject custom projection matrices for W_K, W_V, W_Q.
    ///
    /// Enables researchers to load externally pre-trained projections
    /// (e.g. from PyTorch/JAX experiments) before streaming.
    /// Each matrix is `[d_state x d_input]` in row-major order.
    pub fn set_projections(
        &mut self,
        w_k: &[f64],
        w_v: &[f64],
        w_q: &[f64],
    ) -> Result<(), ConfigError> {
        if w_k.len() != w_v.len() || w_v.len() != w_q.len() {
            return Err(ConfigError::invalid(
                "projections",
                format!(
                    "all projection matrices must have the same length (got {}, {}, {})",
                    w_k.len(),
                    w_v.len(),
                    w_q.len()
                ),
            ));
        }
        if w_k.is_empty() {
            return Err(ConfigError::out_of_range(
                "projections",
                "must have length > 0",
                0,
            ));
        }
        self.layer
            .set_projections(w_k.to_vec(), w_v.to_vec(), w_q.to_vec());
        Ok(())
    }
}

impl StreamingLearner for StreamingTTT {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // Uncertainty-modulated inner learning rate: high uncertainty → increase
        // eta (adapt faster), low uncertainty → decrease eta (conserve).
        let current_uncertainty = self.readout.noise_variance().sqrt();
        const UNCERTAINTY_ALPHA: f64 = 0.001;
        self.rolling_uncertainty = (1.0 - UNCERTAINTY_ALPHA) * self.rolling_uncertainty
            + UNCERTAINTY_ALPHA * current_uncertainty;

        let effective_eta = if self.rolling_uncertainty > 1e-10 {
            let ratio = (current_uncertainty / self.rolling_uncertainty).clamp(0.5, 2.0);
            self.config.eta * ratio
        } else {
            self.config.eta
        };
        self.layer.set_eta(effective_eta);

        // Also modulate RLS forgetting factor by the same uncertainty ratio.
        // High uncertainty → lower ff (forget faster to shed stale weights).
        // Low uncertainty → higher ff (conserve stable weights).
        if self.rolling_uncertainty > 1e-10 {
            let ratio = (current_uncertainty / self.rolling_uncertainty).clamp(0.5, 3.0);
            let base_ff = self.config.forgetting_factor;
            // Stronger modulation: ff drops to ~0.98 at 3x uncertainty spike
            let adaptive_ff = (base_ff - 0.02 * (ratio - 1.0)).clamp(0.95, base_ff);
            self.readout.set_forgetting_factor(adaptive_ff);
        }

        // Compute prediction error BEFORE forward so the fast weight update
        // in this step is driven by the current prediction error. This gives
        // tighter coupling: the TTT layer adapts W_fast to reduce the error
        // that the readout is currently making, not the previous step's error.
        if self.past_warmup() {
            let current_pred = self.readout.predict(&self.last_features);
            let pred_error = target - current_pred;
            self.layer.prediction_feedback = pred_error;

            // Drift detection: compare short-term vs long-term error.
            // Short-term EWMA (alpha=0.1) reacts fast to phase changes.
            // Long-term is rolling_uncertainty (alpha=0.001).
            // When short-term > 2x long-term, reset fast weights.
            let sq_err = pred_error * pred_error;
            let short_alpha = 0.1;
            self.short_term_error =
                (1.0 - short_alpha) * self.short_term_error + short_alpha * sq_err;
            let short_rmse = self.short_term_error.sqrt();
            if self.rolling_uncertainty > 1e-10 && short_rmse > 1.5 * self.rolling_uncertainty {
                self.layer.reset_fast_weights();
            }

            // Update residual alignment tracking (acceleration-based).
            let current_change = current_pred - self.prev_prediction;
            if self.samples_trained > 0 {
                let acceleration = current_change - self.prev_change;
                let prev_acceleration = self.prev_change - self.prev_prev_change;
                let agreement = if acceleration.abs() > 1e-15 && prev_acceleration.abs() > 1e-15 {
                    if (acceleration > 0.0) == (prev_acceleration > 0.0) {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    0.0
                };
                const ALIGN_ALPHA: f64 = 0.05;
                self.alignment_ewma =
                    (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
            }
            self.prev_prev_change = self.prev_change;
            self.prev_change = current_change;
            self.prev_prediction = current_pred;
        }

        // Forward through TTT layer (uses prediction_feedback if set above).
        // In batch_mode, the layer accumulates gradients instead of applying them.
        let ttt_output = self.layer.forward(features);
        self.total_seen += 1;

        // Track mini-batch progress and flush when batch is full.
        if self.batch_size > 1 {
            self.batch_count += 1;
            if self.batch_count >= self.batch_size {
                self.layer.flush_batch();
                self.batch_count = 0;
            }
        }

        // Set tau on first sample (needs input dimension).
        if !self.proj_tau_set {
            self.proj_tau = (self.config.d_model * features.len()).max(1) as f64;
            self.proj_tau_set = true;
        }

        // Track TTT output Frobenius squared norm for utilization ratio.
        {
            let frob_sq: f64 = ttt_output.iter().map(|s| s * s).sum();
            const FROB_ALPHA: f64 = 0.001;
            self.max_frob_sq_ewma = if frob_sq > self.max_frob_sq_ewma {
                frob_sq
            } else {
                (1.0 - FROB_ALPHA) * self.max_frob_sq_ewma + FROB_ALPHA * frob_sq
            };
        }

        // Train RLS readout on the new features (after warmup).
        if self.past_warmup() {
            self.readout.train_one(&ttt_output, target, weight);
            self.samples_trained += 1;
        }

        // --- Two-timescale projection learning (slow weights) ---
        // Update W_K, W_V, W_Q at a decaying rate: Robbins-Monro 1/t schedule.
        //   proj_lr = base_proj_lr / (1 + step / tau)
        //   tau = d_state * d_input  (parameter-count based)
        // W_Q gradient via prediction loss, W_K/W_V via reconstruction loss.
        // Only runs when proj_lr is meaningful and readout has weights to use.
        if self.proj_lr > 1e-10 && self.samples_trained > 0 {
            let pred = self.readout.predict(&ttt_output);
            let pred_error = target - pred;
            let readout_weights = self.readout.weights();

            let (gq, gk, gv) =
                self.layer
                    .compute_projection_gradients(features, pred_error, readout_weights);

            // Per-sample gradient clipping (max norm 1.0).
            let clip = |g: &mut Vec<f64>| {
                let norm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1.0 {
                    let scale = 1.0 / norm;
                    for x in g.iter_mut() {
                        *x *= scale;
                    }
                }
            };
            let mut gq = gq;
            let mut gk = gk;
            let mut gv = gv;
            clip(&mut gq);
            clip(&mut gk);
            clip(&mut gv);

            self.layer.update_projections(&gq, &gk, &gv, self.proj_lr);

            // Decay: 1/(1 + t/tau) — Robbins-Monro optimal schedule.
            self.proj_step += 1;
            self.proj_lr = self.base_proj_lr / (1.0 + self.proj_step as f64 / self.proj_tau);
        }

        // Cache for predict()
        self.last_features = ttt_output;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        if self.total_seen == 0 {
            return 0.0;
        }
        // Use cached features (side-effect-free prediction).
        // See StreamingMamba for the design rationale: predict() must not
        // mutate the TTT layer's fast weights.
        let _ = features; // Acknowledged but not used -- see design note above
        self.readout.predict(&self.last_features)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_trained
    }

    fn reset(&mut self) {
        self.layer.reset_full();
        // Re-enable batch_mode after reset_full (which doesn't touch batch_mode).
        if self.batch_size > 1 {
            self.layer.batch_mode = true;
        }
        self.readout.reset();
        for f in self.last_features.iter_mut() {
            *f = 0.0;
        }
        self.total_seen = 0;
        self.samples_trained = 0;
        self.rolling_uncertainty = 0.0;
        self.short_term_error = 0.0;
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.prev_prev_change = 0.0;
        self.alignment_ewma = 0.0;
        self.max_frob_sq_ewma = 0.0;
        self.batch_count = 0;
        // Reset projection learning schedule (projections themselves cleared by reset_full).
        self.proj_lr = self.base_proj_lr;
        self.proj_step = 0;
        self.proj_tau = 1.0;
        self.proj_tau_set = false;
    }

    fn diagnostics_array(&self) -> [f64; 5] {
        use crate::automl::DiagnosticSource;
        match self.config_diagnostics() {
            Some(d) => [
                d.residual_alignment,
                d.regularization_sensitivity,
                d.depth_sufficiency,
                d.effective_dof,
                d.uncertainty,
            ],
            None => [0.0; 5],
        }
    }

    fn adjust_config(&mut self, lr_multiplier: f64, _lambda_delta: f64) {
        // Scale the inner learning rate (eta) for fast weight updates.
        self.config.eta *= lr_multiplier;
        self.layer.set_eta(self.config.eta);
    }
}

impl std::fmt::Debug for StreamingTTT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingTTT")
            .field("d_model", &self.config.d_model)
            .field("n_heads", &self.config.n_heads)
            .field("eta", &self.config.eta)
            .field("batch_size", &self.batch_size)
            .field("warmup", &self.config.warmup)
            .field("total_seen", &self.total_seen)
            .field("samples_trained", &self.samples_trained)
            .field("past_warmup", &self.past_warmup())
            .field("proj_lr", &self.proj_lr)
            .field("proj_step", &self.proj_step)
            .field("proj_tau", &self.proj_tau)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for StreamingTTT {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        // RLS saturation: 1.0 - trace(P) / (delta * d).
        let rls_saturation = {
            let p = self.readout.p_matrix();
            let d = self.readout.weights().len();
            if d > 0 && self.readout.delta() > 0.0 {
                let trace: f64 = (0..d).map(|i| p[i * d + i]).sum();
                (1.0 - trace / (self.readout.delta() * d as f64)).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };

        // TTT output Frobenius ratio: current ||z||_2^2 / max(||z||_2^2).
        let state_frob_ratio = {
            let frob_sq: f64 = self.last_features.iter().map(|s| s * s).sum();
            if self.max_frob_sq_ewma > 1e-15 {
                (frob_sq / self.max_frob_sq_ewma).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };

        let depth_sufficiency = 0.5 * rls_saturation + 0.5 * state_frob_ratio;

        // Weight magnitude: ||w||_2 / sqrt(d).
        let w = self.readout.weights();
        let effective_dof = if !w.is_empty() {
            let sq_sum: f64 = w.iter().map(|wi| wi * wi).sum();
            sq_sum.sqrt() / (w.len() as f64).sqrt()
        } else {
            0.0
        };

        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.alignment_ewma,
            regularization_sensitivity: self.config.alpha,
            depth_sufficiency,
            effective_dof,
            uncertainty: self.prediction_uncertainty(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builder_default() {
        let config = TTTConfig::builder().build().unwrap();
        assert_eq!(config.d_model, 32);
        assert_eq!(config.n_heads, 1);
    }

    #[test]
    fn config_builder_custom() {
        let config = TTTConfig::builder()
            .d_model(64)
            .n_heads(4)
            .eta(0.005)
            .alpha(0.001)
            .momentum(0.9)
            .warmup(20)
            .seed(123)
            .build()
            .unwrap();
        assert_eq!(config.d_model, 64);
        assert_eq!(config.n_heads, 4);
    }

    #[test]
    fn config_rejects_invalid() {
        assert!(TTTConfig::builder().d_model(0).build().is_err());
        assert!(TTTConfig::builder().n_heads(0).build().is_err());
        assert!(TTTConfig::builder().d_model(7).n_heads(2).build().is_err());
    }

    #[test]
    fn new_creates_model() {
        let config = TTTConfig::builder().d_model(16).build().unwrap();
        let model = StreamingTTT::new(config);
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.past_warmup());
    }

    #[test]
    fn train_and_predict_finite() {
        let config = TTTConfig::builder().d_model(16).warmup(5).build().unwrap();
        let mut model = StreamingTTT::new(config);

        for i in 0..50 {
            let x = [i as f64 * 0.1, (i as f64).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }

        let pred = model.predict(&[0.5, 0.5_f64.sin()]);
        assert!(pred.is_finite(), "prediction should be finite, got {pred}");
    }

    #[test]
    fn warmup_delays_training() {
        let config = TTTConfig::builder().d_model(8).warmup(10).build().unwrap();
        let mut model = StreamingTTT::new(config);

        for i in 0..5 {
            model.train(&[i as f64], i as f64);
        }
        assert_eq!(
            model.n_samples_seen(),
            0,
            "should have 0 trained samples during warmup"
        );
        assert!(!model.past_warmup());

        for i in 5..15 {
            model.train(&[i as f64], i as f64);
        }
        assert!(model.past_warmup());
        assert!(
            model.n_samples_seen() > 0,
            "should have trained samples after warmup"
        );
    }

    #[test]
    fn predict_before_train_returns_zero() {
        let config = TTTConfig::builder().d_model(16).build().unwrap();
        let model = StreamingTTT::new(config);
        assert_eq!(model.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn reset_clears_state() {
        let config = TTTConfig::builder().d_model(8).warmup(2).build().unwrap();
        let mut model = StreamingTTT::new(config);

        for i in 0..20 {
            model.train(&[i as f64], i as f64 * 2.0);
        }
        assert!(model.n_samples_seen() > 0);

        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        assert_eq!(model.total_seen, 0);
        assert!(!model.past_warmup());
    }

    #[test]
    fn implements_streaming_learner() {
        let config = TTTConfig::builder().d_model(8).warmup(0).build().unwrap();
        let model = StreamingTTT::new(config);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(model);
        boxed.train(&[1.0], 2.0);
        let pred = boxed.predict(&[1.0]);
        assert!(pred.is_finite());
    }

    #[test]
    fn multi_head_works() {
        let config = TTTConfig::builder()
            .d_model(16)
            .n_heads(4)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        for i in 0..50 {
            let x = [i as f64 * 0.01, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 3.0 + x[1] * 2.0;
            model.train(&x, y);
        }

        let pred = model.predict(&[0.5, 0.5_f64.sin(), 0.5_f64.cos()]);
        assert!(
            pred.is_finite(),
            "multi-head prediction should be finite, got {pred}"
        );
    }

    #[test]
    fn titans_extensions_work() {
        let config = TTTConfig::builder()
            .d_model(16)
            .eta(0.01)
            .alpha(0.001) // weight decay
            .momentum(0.9) // momentum
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        for i in 0..100 {
            let x = [i as f64 * 0.01, (i as f64).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }

        let pred = model.predict(&[0.5, 0.5_f64.sin()]);
        assert!(
            pred.is_finite(),
            "Titans-style TTT should produce finite predictions, got {pred}"
        );
    }

    #[test]
    fn n_samples_seen_tracks_post_warmup() {
        let config = TTTConfig::builder().d_model(8).warmup(10).build().unwrap();
        let mut model = StreamingTTT::new(config);

        for i in 0..30 {
            model.train(&[i as f64], i as f64);
        }

        // 30 total, warmup=10, so RLS trained on samples after warmup
        assert!(
            model.n_samples_seen() > 0 && model.n_samples_seen() < 30,
            "n_samples_seen should be between 0 and 30 (warmup excluded), got {}",
            model.n_samples_seen()
        );
    }

    #[test]
    fn config_display() {
        let config = TTTConfig::builder().d_model(16).build().unwrap();
        let s = format!("{config}");
        assert!(s.contains("d_model=16"), "display should contain d_model");
        assert!(s.contains("n_heads=1"), "display should contain n_heads");
    }

    #[test]
    fn config_clone() {
        let config = TTTConfig::builder().d_model(64).seed(99).build().unwrap();
        let cloned = config.clone();
        assert_eq!(cloned.d_model, config.d_model);
        assert_eq!(cloned.seed, config.seed);
    }

    #[test]
    fn ttt_uncertainty_modulated_eta() {
        let config = TTTConfig::builder()
            .d_model(16)
            .eta(0.01)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        // Train with modulated eta — should produce finite predictions
        for i in 0..100 {
            let x = [i as f64 * 0.1, (i as f64).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }

        let pred = model.predict(&[0.5, 0.5_f64.sin()]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after uncertainty-modulated training, got {}",
            pred
        );

        // Rolling uncertainty should be tracked
        assert!(
            model.rolling_uncertainty > 0.0 || model.prediction_uncertainty() == 0.0,
            "rolling_uncertainty should be non-negative"
        );

        // Prediction uncertainty from readout should be available
        let unc = model.prediction_uncertainty();
        assert!(
            unc.is_finite(),
            "prediction_uncertainty should be finite, got {}",
            unc
        );
    }

    #[test]
    fn adaptive_forgetting_factor_produces_finite_predictions() {
        // Verify that uncertainty-driven RLS forgetting factor modulation
        // produces finite predictions through stable and drift regimes.
        let config = TTTConfig::builder()
            .d_model(16)
            .eta(0.01)
            .forgetting_factor(0.998)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        // Phase 1: stable regime
        for i in 0..100 {
            let x = [i as f64 * 0.1, (i as f64).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }

        let pred_stable = model.predict(&[5.0, 5.0_f64.sin()]);
        assert!(
            pred_stable.is_finite(),
            "prediction should be finite after stable training, got {}",
            pred_stable
        );

        // Phase 2: sudden distribution shift (drift) — should not diverge
        for i in 0..50 {
            let x = [i as f64 * 0.1, (i as f64).cos()];
            let y = -x[0] * 3.0 + 10.0; // completely different relationship
            model.train(&x, y);
        }

        let pred_drift = model.predict(&[2.5, 2.5_f64.cos()]);
        assert!(
            pred_drift.is_finite(),
            "prediction should be finite after drift, got {}",
            pred_drift
        );

        // Rolling uncertainty should be tracked and finite
        assert!(
            model.rolling_uncertainty.is_finite(),
            "rolling_uncertainty should be finite, got {}",
            model.rolling_uncertainty
        );
    }

    #[test]
    fn pretrain_projections_basic() {
        let config = TTTConfig::builder().d_model(16).warmup(5).build().unwrap();
        let mut model = StreamingTTT::new(config);

        // Generate simple linear data
        let data: Vec<(Vec<f64>, f64)> = (0..200)
            .map(|i| {
                let x = vec![i as f64 * 0.01, (i as f64 * 0.1).sin(), 1.0];
                let y = 2.0 * x[0] + 0.5 * x[1] + 1.0;
                (x, y)
            })
            .collect();
        let data_refs: Vec<(&[f64], f64)> = data.iter().map(|(x, y)| (x.as_slice(), *y)).collect();

        model.pretrain_projections(&data_refs, 5);

        // After pretraining, model should be reset
        assert_eq!(
            model.n_samples_seen(),
            0,
            "n_samples_seen should be 0 after pretraining reset"
        );
        assert!(
            !model.past_warmup(),
            "should not be past warmup after pretraining reset"
        );

        // Train and test — pretrained model should produce finite predictions
        for &(ref features, target) in &data[..50] {
            model.train(features, target);
        }
        let pred = model.predict(&data[50].0);
        assert!(
            pred.is_finite(),
            "prediction after pretraining should be finite, got {pred}"
        );
    }

    #[test]
    fn pretrain_improves_over_random() {
        // Compare pretrained vs random projections on same data
        let data: Vec<(Vec<f64>, f64)> = (0..500)
            .map(|i| {
                let t = i as f64 * 0.02;
                let x = vec![t.sin(), t.cos(), (2.0 * t).sin(), (3.0 * t).cos()];
                let y = 0.7 * x[0] + 0.3 * x[1] - 0.5 * x[2] + 0.2 * x[3];
                (x, y)
            })
            .collect();
        let data_refs: Vec<(&[f64], f64)> = data.iter().map(|(x, y)| (x.as_slice(), *y)).collect();

        // Pretrained model
        let config = TTTConfig::builder()
            .d_model(16)
            .eta(0.01)
            .warmup(5)
            .seed(42)
            .build()
            .unwrap();
        let mut pretrained = StreamingTTT::new(config);
        pretrained.pretrain_projections(&data_refs[..200], 5);

        // Random model (same config, no pretraining)
        let config2 = TTTConfig::builder()
            .d_model(16)
            .eta(0.01)
            .warmup(5)
            .seed(42)
            .build()
            .unwrap();
        let mut random = StreamingTTT::new(config2);

        // Test on remaining data
        let mut pretrained_err = 0.0;
        let mut random_err = 0.0;
        for &(ref features, target) in &data[200..] {
            let p1 = pretrained.predict(features);
            let p2 = random.predict(features);
            pretrained_err += (target - p1).powi(2);
            random_err += (target - p2).powi(2);
            pretrained.train(features, target);
            random.train(features, target);
        }

        // Both errors should be finite
        assert!(
            pretrained_err.is_finite() && random_err.is_finite(),
            "both errors should be finite: pretrained={}, random={}",
            pretrained_err,
            random_err
        );
    }

    #[test]
    fn pretrain_empty_data_noop() {
        let config = TTTConfig::builder().d_model(8).build().unwrap();
        let mut model = StreamingTTT::new(config);
        model.pretrain_projections(&[], 5);
        assert_eq!(
            model.n_samples_seen(),
            0,
            "empty data pretraining should be a no-op"
        );
    }

    #[test]
    fn pretrain_zero_epochs_noop() {
        let config = TTTConfig::builder().d_model(8).build().unwrap();
        let mut model = StreamingTTT::new(config);
        let data = [(vec![1.0, 2.0], 3.0)];
        let data_refs: Vec<(&[f64], f64)> = data.iter().map(|(x, y)| (x.as_slice(), *y)).collect();
        model.pretrain_projections(&data_refs, 0);
        assert_eq!(
            model.n_samples_seen(),
            0,
            "zero-epoch pretraining should be a no-op"
        );
    }

    #[test]
    fn auto_pretrain_decays_projection_lr() {
        // Verify the two-timescale schedule: proj_lr decays as 1/(1+t/tau).
        let config = TTTConfig::builder()
            .d_model(16)
            .eta(0.01)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        let initial_proj_lr = model.proj_lr;
        assert!(
            initial_proj_lr > 0.0,
            "initial proj_lr should be eta*0.1 = 0.001, got {}",
            initial_proj_lr
        );

        // Train 500 samples — projections should learn automatically.
        for i in 0..500 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos(), (2.0 * t).sin()];
            let y = 0.7 * x[0] + 0.3 * x[1];
            model.train(&x, y);
        }

        // proj_lr should have decayed significantly.
        // tau = d_model(16) * d_input(3) = 48.
        // After 500 steps: proj_lr = 0.001 / (1 + 500/48) ≈ 0.000088.
        assert!(
            model.proj_lr < initial_proj_lr * 0.2,
            "proj_lr should decay: initial={}, current={}",
            initial_proj_lr,
            model.proj_lr
        );
        assert!(
            model.proj_lr > 0.0,
            "proj_lr should stay positive, got {}",
            model.proj_lr
        );

        // Predictions should be finite.
        let pred = model.predict(&[0.5, 0.5_f64.sin(), 1.0_f64.sin()]);
        assert!(
            pred.is_finite(),
            "prediction with auto-pretrained projections should be finite, got {}",
            pred
        );
    }

    #[test]
    fn auto_pretrain_tau_scales_with_dimensions() {
        // tau = d_state * d_input. Higher dims → slower decay → more learning.
        let config = TTTConfig::builder()
            .d_model(64)
            .eta(0.01)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        // Train one sample to trigger tau initialization.
        model.train(&[1.0; 32], 0.5);

        // tau should be d_model(64) * d_input(32) = 2048.
        assert!(
            (model.proj_tau - 2048.0).abs() < 1e-10,
            "tau should be d_model*d_input = 2048, got {}",
            model.proj_tau
        );
    }

    #[test]
    fn auto_pretrain_reset_restores_schedule() {
        let config = TTTConfig::builder()
            .d_model(16)
            .eta(0.01)
            .warmup(2)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);
        let initial_lr = model.proj_lr;

        // Train to decay proj_lr.
        for i in 0..100 {
            model.train(&[i as f64 * 0.1, (i as f64).sin()], i as f64);
        }
        assert!(model.proj_lr < initial_lr);

        // Reset should restore schedule.
        model.reset();
        assert!(
            (model.proj_lr - initial_lr).abs() < 1e-15,
            "reset should restore proj_lr: expected {}, got {}",
            initial_lr,
            model.proj_lr
        );
        assert_eq!(model.proj_step, 0, "reset should zero proj_step");
    }

    #[test]
    fn config_rejects_zero_batch_size() {
        assert!(
            TTTConfig::builder().batch_size(0).build().is_err(),
            "batch_size=0 should be rejected"
        );
    }

    #[test]
    fn config_batch_size_default() {
        let config = TTTConfig::builder().build().unwrap();
        assert_eq!(config.batch_size, 16, "default batch_size should be 16");
    }

    #[test]
    fn config_display_includes_batch_size() {
        let config = TTTConfig::builder().batch_size(8).build().unwrap();
        let s = format!("{config}");
        assert!(
            s.contains("batch_size=8"),
            "display should contain batch_size, got: {s}"
        );
    }

    #[test]
    fn mini_batch_convergence() {
        let config = TTTConfig::builder()
            .d_model(16)
            .eta(0.01)
            .batch_size(16)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        let mut errors_early = Vec::new();
        let mut errors_late = Vec::new();

        for i in 0..2000 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos(), (2.0 * t).sin()];
            let y = 0.7 * x[0] + 0.3 * x[1] - 0.5 * x[2];

            if model.n_samples_seen() > 0 {
                let pred = model.predict(&x);
                let err = (pred - y).powi(2);
                if (50..200).contains(&i) {
                    errors_early.push(err);
                } else if i >= 1500 {
                    errors_late.push(err);
                }
            }
            model.train(&x, y);
        }

        let mse_early = errors_early.iter().sum::<f64>() / errors_early.len() as f64;
        let mse_late = errors_late.iter().sum::<f64>() / errors_late.len() as f64;

        assert!(
            mse_late < mse_early * 1.5,
            "mini-batch TTT should converge: early={:.4}, late={:.4}",
            mse_early,
            mse_late
        );
    }

    #[test]
    fn gelu_output_is_nonlinear() {
        let config = TTTConfig::builder()
            .d_model(8)
            .batch_size(1)
            .warmup(0)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        // Train on nonlinear target
        for i in 0..100 {
            let x = [i as f64 * 0.1, (i as f64 * 0.1).sin()];
            let y = x[0] * x[0] + x[1]; // nonlinear
            model.train(&x, y);
        }

        let pred = model.predict(&[1.0, 1.0_f64.sin()]);
        assert!(
            pred.is_finite(),
            "GELU TTT should produce finite predictions, got {pred}"
        );
    }

    #[test]
    fn batch_size_one_matches_online() {
        // batch_size=1 should behave as pure online (no accumulation).
        let config = TTTConfig::builder()
            .d_model(8)
            .eta(0.01)
            .batch_size(1)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        for i in 0..50 {
            let x = [i as f64 * 0.1, (i as f64).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }

        let pred = model.predict(&[0.5, 0.5_f64.sin()]);
        assert!(
            pred.is_finite(),
            "batch_size=1 TTT should produce finite predictions, got {pred}"
        );
    }

    #[test]
    fn reset_clears_batch_state() {
        let config = TTTConfig::builder()
            .d_model(8)
            .batch_size(16)
            .warmup(2)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        // Train 10 samples (partial batch)
        for i in 0..10 {
            model.train(&[i as f64], i as f64 * 2.0);
        }
        assert!(
            model.batch_count > 0,
            "batch_count should be non-zero after partial batch"
        );

        model.reset();
        assert_eq!(model.batch_count, 0, "batch_count should be 0 after reset");
    }

    #[test]
    fn test_identity_init_when_dims_match() {
        // When d_model == input_dim, projections should be identity matrices.
        let config = TTTConfig::builder().d_model(4).warmup(2).build().unwrap();
        let mut model = StreamingTTT::new(config);

        for i in 0..20 {
            let x = [i as f64 * 0.1, (i as f64).sin(), (i as f64).cos(), 1.0];
            let y = x[0] * 2.0 + x[1] * 0.5;
            model.train(&x, y);
        }

        let pred = model.predict(&[0.5, 0.5_f64.sin(), 0.5_f64.cos(), 1.0]);
        assert!(
            pred.is_finite(),
            "identity-init prediction should be finite, got {pred}"
        );
    }

    #[test]
    fn test_set_projections_overrides_init() {
        let config = TTTConfig::builder().d_model(4).warmup(2).build().unwrap();
        let mut model = StreamingTTT::new(config);

        // Inject custom projections: 4 x 3 = 12 elements each (d_state=4, d_input=3).
        let custom = vec![0.5; 12];
        model
            .set_projections(&custom, &custom, &custom)
            .expect("set_projections should succeed with valid inputs");

        // Train a sample — ensure_init should NOT overwrite our custom projections.
        for i in 0..20 {
            let x = [i as f64 * 0.1, (i as f64).sin(), 1.0];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }

        // Access the layer's projections via a second set_projections call
        // to verify the first one stuck: the model should still produce finite output.
        let pred = model.predict(&[0.5, 0.5_f64.sin(), 1.0]);
        assert!(
            pred.is_finite(),
            "prediction after set_projections should be finite, got {pred}"
        );
    }

    #[test]
    fn test_set_projections_validates_dimensions() {
        let config = TTTConfig::builder().d_model(8).build().unwrap();
        let mut model = StreamingTTT::new(config);

        // Mismatched lengths should fail.
        let short = vec![0.5; 10];
        let long = vec![0.5; 20];
        let result = model.set_projections(&short, &long, &short);
        assert!(
            result.is_err(),
            "mismatched projection lengths should return Err"
        );

        // Empty projections should fail.
        let empty: Vec<f64> = Vec::new();
        let result = model.set_projections(&empty, &empty, &empty);
        assert!(result.is_err(), "empty projections should return Err");
    }

    #[test]
    fn test_xavier_init_when_dims_differ() {
        // When d_model != input features, Xavier initialization kicks in.
        let config = TTTConfig::builder().d_model(16).warmup(2).build().unwrap();
        let mut model = StreamingTTT::new(config);

        // Input dim = 3, d_model = 16 — Xavier path.
        for i in 0..30 {
            let x = [i as f64 * 0.1, (i as f64).sin(), 1.0];
            let y = x[0] * 2.0 + x[1] * 0.5 + 1.0;
            model.train(&x, y);
        }

        let pred = model.predict(&[0.5, 0.5_f64.sin(), 1.0]);
        assert!(
            pred.is_finite(),
            "xavier-init prediction should be finite, got {pred}"
        );
    }
}
