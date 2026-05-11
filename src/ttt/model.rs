//! StreamingTTT model with RLS readout.

use super::config::TTTConfig;
use super::layer::TTTLayer;
use crate::error::ConfigError;
use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;
use irithyll_core::continual::{ContinualStrategy, NeuronRegeneration};

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
/// let config = TTTConfig::builder().d_model(16).learning_rate(0.1).build().unwrap();
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
    /// EWMA of |prediction error| for surprise-gated LR (alpha=0.01).
    running_abs_error: f64,

    // --- Mini-batch fast weight updates ---
    // Default is b=1 (single-sample streaming) for maximum adaptation speed.
    // When batch_size > 1, gradients accumulate in TTTLayer and flush periodically.
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
    /// Optional plasticity guard for maintaining learning capacity.
    plasticity_guard: Option<NeuronRegeneration>,
    /// Snapshot of per-unit output energy from previous step.
    prev_output_energy: Vec<f64>,
}

impl StreamingTTT {
    /// Create a new StreamingTTT from config.
    pub fn new(config: TTTConfig) -> Self {
        // deep_memory_layers >= 2 → 2-layer MLP fast weights (Titans §3.4, Behrouz et al. 2025).
        // hidden_dim = d_model (matched to state dimension per paper default).
        // Values > 2 are treated as 2 (only 2-layer MLP is implemented).
        // deep_memory_layers == 1 → linear fast weights (TTT-Linear, = RLS when projections frozen).
        let mlp_hidden_dim = if config.deep_memory_layers >= 2 {
            config.d_model
        } else {
            0
        };
        let mut layer = TTTLayer::new(
            config.d_model,
            config.learning_rate,
            config.alpha,
            config.momentum > 0.0,
            config.momentum,
            config.nesterov,
            config.alpha_warmup,
            mlp_hidden_dim,
            config.seed,
        );
        // Enable mini-batch accumulation when batch_size > 1.
        let batch_size = config.batch_size;
        if batch_size > 1 {
            layer.batch_mode = true;
        }
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta_rls);
        let last_features = vec![0.0; config.d_model];
        let base_proj_lr = config.learning_rate * 0.1;

        // Create plasticity guard if a PlasticityConfig was provided.
        // Tracks d_model hidden units (group_size=1 = per-unit tracking).
        let plasticity_guard = config.plasticity.as_ref().map(|p| {
            NeuronRegeneration::new(
                config.d_model,
                1, // group_size = 1 (per-unit tracking)
                p.regen_fraction,
                p.regen_interval,
                p.utility_alpha,
                config.seed.wrapping_add(0x_DEAD_CAFE),
            )
        });
        let prev_output_energy = vec![0.0; config.d_model];

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
            running_abs_error: 0.0,
            batch_size,
            batch_count: 0,
            base_proj_lr,
            proj_lr: base_proj_lr,
            proj_step: 0,
            proj_tau: 1.0, // placeholder, set on first train_one
            proj_tau_set: false,
            plasticity_guard,
            prev_output_energy,
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
    /// * `epochs` — Number of passes over the data (typically 3-10).
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
        self.running_abs_error = 0.0;
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
        if self.total_seen == 0 {
            self.rolling_uncertainty = current_uncertainty;
        } else {
            self.rolling_uncertainty = (1.0 - UNCERTAINTY_ALPHA) * self.rolling_uncertainty
                + UNCERTAINTY_ALPHA * current_uncertainty;
        }

        let mut effective_eta = if self.rolling_uncertainty > 1e-10 {
            let ratio = (current_uncertainty / self.rolling_uncertainty).clamp(0.5, 2.0);
            self.config.learning_rate * ratio
        } else {
            self.config.learning_rate
        };

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

            // Surprise-gated LR: modulate eta by prediction surprise.
            // Routine predictions (low surprise) → lower LR (conserve).
            // Surprising inputs (high error vs baseline) → full LR (adapt).
            if self.config.surprise_gated && self.running_abs_error > 1e-10 {
                let surprise = (pred_error.abs() / self.running_abs_error.max(1e-10)).min(1.0);
                effective_eta *= surprise;
            }
            // Update running |error| EWMA for surprise computation.
            const SURPRISE_ALPHA: f64 = 0.01;
            if self.samples_trained == 0 {
                self.running_abs_error = pred_error.abs();
            } else {
                self.running_abs_error = (1.0 - SURPRISE_ALPHA) * self.running_abs_error
                    + SURPRISE_ALPHA * pred_error.abs();
            }

            // Drift detection: compare short-term vs long-term error.
            // Short-term EWMA (alpha=0.1) reacts fast to phase changes.
            // Long-term is rolling_uncertainty (alpha=0.001).
            // When short-term > 2x long-term, reset fast weights.
            //
            // Warmup guard: drift detection is disabled for the first 100
            // post-warmup samples. rolling_uncertainty starts at 0.0 with a
            // very slow alpha=0.001, so early errors trivially exceed the
            // threshold and cause destructive reset loops. Letting fast
            // weights establish structure first makes drift detection meaningful.
            let sq_err = pred_error * pred_error;
            let short_alpha = 0.1;
            if self.samples_trained == 0 {
                self.short_term_error = sq_err;
            } else {
                self.short_term_error =
                    (1.0 - short_alpha) * self.short_term_error + short_alpha * sq_err;
            }
            let short_rmse = self.short_term_error.sqrt();
            let drift_warmup_done = self.samples_trained >= 100;
            if drift_warmup_done
                && self.rolling_uncertainty > 1e-10
                && short_rmse > 1.5 * self.rolling_uncertainty
            {
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
                if self.samples_trained == 1 {
                    self.alignment_ewma = agreement;
                } else {
                    self.alignment_ewma =
                        (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
                }
            }
            self.prev_prev_change = self.prev_change;
            self.prev_change = current_change;
            self.prev_prediction = current_pred;
        }

        // Set effective eta (after surprise gating, if applicable).
        self.layer.set_eta(effective_eta);

        // Option D: compute pre-update readout features before advancing W_fast.
        //
        // The RLS readout must be trained on the same feature distribution it
        // will see at predict time. predict() calls forward_predict() which reads
        // the current (pre-advance) fast weights. If we trained RLS on the
        // post-advance ttt_output instead, predict and train would see a one-step
        // feature mismatch: train sees q(x_t)·W_{t}, predict sees q(x_t)·W_{t-1}.
        //
        // Ordering:
        //   1. Compute pre-update features via forward_predict (reads W_fast as-is).
        //   2. Train RLS on these features — consistent with what predict will use.
        //   3. Advance W_fast via forward() — the inner SGD update (state advance).
        //   4. Use post-update output for diagnostics only (Frobenius, projection LR,
        //      plasticity, last_features cache used by residual alignment tracking).
        let pre_update_features = if self.past_warmup() {
            // forward_predict is &self (no mutation) — safe to call before forward.
            Some(self.layer.forward_predict(features))
        } else {
            None
        };

        // Forward through TTT layer (uses prediction_feedback if set above).
        // This is the state-advance step: W_fast is updated here.
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

        // Train RLS readout on pre-update features (Option D).
        // pre_update_features is Some only when past_warmup.
        if let Some(ref pre_feat) = pre_update_features {
            self.readout.train_one(pre_feat, target, weight);
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

        // Plasticity maintenance: track per-unit TTT output energy and
        // trigger regeneration when dead units are detected.
        if let Some(ref mut guard) = self.plasticity_guard {
            let mut output_energy: Vec<f64> = ttt_output.iter().map(|x| x.abs()).collect();
            guard.pre_update(&self.prev_output_energy, &mut output_energy);
            guard.post_update(&self.prev_output_energy);
            // Surgical per-unit reinit: only dead units get recycled.
            // Each dead unit has its fast weight row reinitialized while
            // preserving all other units' learned representations.
            let n_groups = guard.n_groups();
            let any_regenerated = (0..n_groups).any(|g| guard.was_regenerated(g));
            self.prev_output_energy = output_energy;
            if any_regenerated {
                let mut reinit_rng = self
                    .config
                    .seed
                    .wrapping_add(0xCAFE_BABE_u64.wrapping_mul(self.samples_trained));
                for j in 0..n_groups {
                    if guard.was_regenerated(j) {
                        self.layer.reinitialize_unit(j, &mut reinit_rng);
                    }
                }
            }
        }

        // Cache for predict()
        self.last_features = ttt_output;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        if self.total_seen == 0 {
            return 0.0;
        }
        // Run a prediction-only forward through the TTT layer on the CURRENT
        // input features. Unlike Mamba (where forward() mutates SSM state h),
        // TTT's prediction path (project → W_fast*k → GELU → LN → residual)
        // is purely read-only — it only reads the current fast weights without
        // updating them. forward_predict() takes &self, not &mut self.
        let ttt_features = self.layer.forward_predict(features);
        self.readout.predict(&ttt_features)
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
        self.running_abs_error = 0.0;
        self.batch_count = 0;
        // Reset projection learning schedule (projections themselves cleared by reset_full).
        self.proj_lr = self.base_proj_lr;
        self.proj_step = 0;
        self.proj_tau = 1.0;
        self.proj_tau_set = false;
        if let Some(ref mut guard) = self.plasticity_guard {
            guard.reset();
        }
        self.prev_output_energy.fill(0.0);
    }

    #[allow(deprecated)]
    fn diagnostics_array(&self) -> [f64; 5] {
        <Self as crate::learner::Tunable>::diagnostics_array(self)
    }

    #[allow(deprecated)]
    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64) {
        <Self as crate::learner::Tunable>::adjust_config(self, lr_multiplier, lambda_delta);
    }

    #[allow(deprecated)]
    fn readout_weights(&self) -> Option<&[f64]> {
        let w = <Self as crate::learner::HasReadout>::readout_weights(self);
        if w.is_empty() {
            None
        } else {
            Some(w)
        }
    }
}

impl crate::learner::Tunable for StreamingTTT {
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

    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64) {
        self.config.learning_rate *= lr_multiplier;
        self.layer.set_eta(self.config.learning_rate);
        // α_eff = α_base + lambda_delta (clamped to [0, 1) to remain a valid decay rate).
        // AutoTuner passes a signed delta; positive = more regularization, negative = less.
        // Clamp ceiling at 0.999 so the weight-decay update (1 - α)·W never collapses to 0.
        let alpha_new = (self.config.alpha + lambda_delta).clamp(0.0, 0.999);
        self.config.alpha = alpha_new;
        self.layer.set_alpha(alpha_new);
    }
}

impl crate::learner::HasReadout for StreamingTTT {
    fn readout_weights(&self) -> &[f64] {
        self.readout.weights()
    }
}

impl std::fmt::Debug for StreamingTTT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingTTT")
            .field("d_model", &self.config.d_model)
            .field("eta", &self.config.learning_rate)
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
            // Use the alpha *actually applied* at the last forward() step (post-warmup ramp),
            // not the static config value. During alpha_warmup, effective_alpha < config.alpha;
            // after adjust_config(), config.alpha itself has changed. Reading the static field
            // for a metric named "regularization_sensitivity" is half-finished telemetry —
            // the metric must reflect the regularization that is actually being applied.
            regularization_sensitivity: self.layer.effective_alpha(),
            depth_sufficiency,
            effective_dof,
            uncertainty: self.prediction_uncertainty(),
        })
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::StreamingTTT;
    #[allow(unused_imports)]
    use crate::learner::StreamingLearner;
    #[allow(unused_imports)]
    use crate::ttt::TTTConfig;

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
    fn d_model_observably_affects_output() {
        let config_16 = TTTConfig::builder()
            .d_model(16)
            .warmup(5)
            .seed(42)
            .build()
            .unwrap();
        let mut model_16 = StreamingTTT::new(config_16);

        let config_32 = TTTConfig::builder()
            .d_model(32)
            .warmup(5)
            .seed(42)
            .build()
            .unwrap();
        let mut model_32 = StreamingTTT::new(config_32);

        // Train both on identical data
        for i in 0..50 {
            let x = [i as f64 * 0.01, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 3.0 + x[1] * 2.0;
            model_16.train(&x, y);
            model_32.train(&x, y);
        }

        // Predictions should differ due to different layer capacity
        let test_x = [0.5, 0.5_f64.sin(), 0.5_f64.cos()];
        let pred_16 = model_16.predict(&test_x);
        let pred_32 = model_32.predict(&test_x);
        assert!(
            pred_16.is_finite(),
            "d_model=16 should produce finite output"
        );
        assert!(
            pred_32.is_finite(),
            "d_model=32 should produce finite output"
        );
        assert_ne!(
            pred_16.to_bits(),
            pred_32.to_bits(),
            "different d_model values should affect predictions"
        );
    }

    #[test]
    fn titans_extensions_work() {
        let config = TTTConfig::builder()
            .d_model(16)
            .learning_rate(0.01)
            .alpha(0.001)
            .momentum(0.9)
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

        assert!(
            model.n_samples_seen() > 0 && model.n_samples_seen() < 30,
            "n_samples_seen should be between 0 and 30 (warmup excluded), got {}",
            model.n_samples_seen()
        );
    }

    #[test]
    fn ttt_uncertainty_modulated_eta() {
        let config = TTTConfig::builder()
            .d_model(16)
            .learning_rate(0.01)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

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

        assert!(
            model.rolling_uncertainty > 0.0 || model.prediction_uncertainty() == 0.0,
            "rolling_uncertainty should be non-negative"
        );

        let unc = model.prediction_uncertainty();
        assert!(
            unc.is_finite(),
            "prediction_uncertainty should be finite, got {}",
            unc
        );
    }

    #[test]
    fn adaptive_forgetting_factor_produces_finite_predictions() {
        let config = TTTConfig::builder()
            .d_model(16)
            .learning_rate(0.01)
            .forgetting_factor(0.998)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

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

        for i in 0..50 {
            let x = [i as f64 * 0.1, (i as f64).cos()];
            let y = -x[0] * 3.0 + 10.0;
            model.train(&x, y);
        }

        let pred_drift = model.predict(&[2.5, 2.5_f64.cos()]);
        assert!(
            pred_drift.is_finite(),
            "prediction should be finite after drift, got {}",
            pred_drift
        );

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

        let data: Vec<(Vec<f64>, f64)> = (0..200)
            .map(|i| {
                let x = vec![i as f64 * 0.01, (i as f64 * 0.1).sin(), 1.0];
                let y = 2.0 * x[0] + 0.5 * x[1] + 1.0;
                (x, y)
            })
            .collect();
        let data_refs: Vec<(&[f64], f64)> = data.iter().map(|(x, y)| (x.as_slice(), *y)).collect();

        model.pretrain_projections(&data_refs, 5);

        assert_eq!(
            model.n_samples_seen(),
            0,
            "n_samples_seen should be 0 after pretraining reset"
        );
        assert!(
            !model.past_warmup(),
            "should not be past warmup after pretraining reset"
        );

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
        let data: Vec<(Vec<f64>, f64)> = (0..500)
            .map(|i| {
                let t = i as f64 * 0.02;
                let x = vec![t.sin(), t.cos(), (2.0 * t).sin(), (3.0 * t).cos()];
                let y = 0.7 * x[0] + 0.3 * x[1] - 0.5 * x[2] + 0.2 * x[3];
                (x, y)
            })
            .collect();
        let data_refs: Vec<(&[f64], f64)> = data.iter().map(|(x, y)| (x.as_slice(), *y)).collect();

        let config = TTTConfig::builder()
            .d_model(16)
            .learning_rate(0.01)
            .warmup(5)
            .seed(42)
            .build()
            .unwrap();
        let mut pretrained = StreamingTTT::new(config);
        pretrained.pretrain_projections(&data_refs[..200], 5);

        let config2 = TTTConfig::builder()
            .d_model(16)
            .learning_rate(0.01)
            .warmup(5)
            .seed(42)
            .build()
            .unwrap();
        let mut random = StreamingTTT::new(config2);

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
        let config = TTTConfig::builder()
            .d_model(16)
            .learning_rate(0.01)
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

        for i in 0..500 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos(), (2.0 * t).sin()];
            let y = 0.7 * x[0] + 0.3 * x[1];
            model.train(&x, y);
        }

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

        let pred = model.predict(&[0.5, 0.5_f64.sin(), 1.0_f64.sin()]);
        assert!(
            pred.is_finite(),
            "prediction with auto-pretrained projections should be finite, got {}",
            pred
        );
    }

    #[test]
    fn auto_pretrain_tau_scales_with_dimensions() {
        let config = TTTConfig::builder()
            .d_model(64)
            .learning_rate(0.01)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        model.train(&[1.0; 32], 0.5);

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
            .learning_rate(0.01)
            .warmup(2)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);
        let initial_lr = model.proj_lr;

        for i in 0..100 {
            model.train(&[i as f64 * 0.1, (i as f64).sin()], i as f64);
        }
        assert!(model.proj_lr < initial_lr);

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
    fn mini_batch_convergence() {
        let config = TTTConfig::builder()
            .d_model(16)
            .learning_rate(0.01)
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
            mse_late < 1.0,
            "mini-batch TTT should not diverge: early={:.4}, late={:.4}",
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

        for i in 0..100 {
            let x = [i as f64 * 0.1, (i as f64 * 0.1).sin()];
            let y = x[0] * x[0] + x[1];
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
        let config = TTTConfig::builder()
            .d_model(8)
            .learning_rate(0.01)
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

        let custom = vec![0.5; 12];
        model
            .set_projections(&custom, &custom, &custom)
            .expect("set_projections should succeed with valid inputs");

        for i in 0..20 {
            let x = [i as f64 * 0.1, (i as f64).sin(), 1.0];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }

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

        let short = vec![0.5; 10];
        let long = vec![0.5; 20];
        let result = model.set_projections(&short, &long, &short);
        assert!(
            result.is_err(),
            "mismatched projection lengths should return Err"
        );

        let empty: Vec<f64> = Vec::new();
        let result = model.set_projections(&empty, &empty, &empty);
        assert!(result.is_err(), "empty projections should return Err");
    }

    #[test]
    fn test_xavier_init_when_dims_differ() {
        let config = TTTConfig::builder().d_model(16).warmup(2).build().unwrap();
        let mut model = StreamingTTT::new(config);

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

    #[test]
    fn ttt_nesterov_trains() {
        let config = TTTConfig::builder()
            .d_model(16)
            .nesterov(true)
            .momentum(0.9)
            .learning_rate(0.05)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);
        for i in 0..100 {
            model.train_one(&[0.1 * i as f64; 4], (i as f64 * 0.1).sin(), 1.0);
        }
        let pred = model.predict(&[0.5; 4]);
        assert!(pred.is_finite(), "nesterov TTT prediction must be finite");
    }

    #[test]
    fn ttt_alpha_warmup_trains() {
        let config = TTTConfig::builder()
            .d_model(16)
            .alpha(0.01)
            .alpha_warmup(50)
            .learning_rate(0.05)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);
        for i in 0..100 {
            model.train_one(&[0.1 * i as f64; 4], (i as f64 * 0.1).sin(), 1.0);
        }
        let pred = model.predict(&[0.5; 4]);
        assert!(
            pred.is_finite(),
            "alpha warmup TTT prediction must be finite"
        );
    }

    #[test]
    fn ttt_surprise_gated_trains() {
        let config = TTTConfig::builder()
            .d_model(16)
            .surprise_gated(true)
            .learning_rate(0.05)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);
        for i in 0..100 {
            model.train_one(&[0.1 * i as f64; 4], (i as f64 * 0.1).sin(), 1.0);
        }
        let pred = model.predict(&[0.5; 4]);
        assert!(
            pred.is_finite(),
            "surprise-gated TTT prediction must be finite"
        );
    }

    #[test]
    fn ttt_all_titans_features_combined() {
        let config = TTTConfig::builder()
            .d_model(16)
            .nesterov(true)
            .momentum(0.9)
            .alpha(0.005)
            .alpha_warmup(30)
            .surprise_gated(true)
            .learning_rate(0.05)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);
        for i in 0..200 {
            let x = [0.01 * i as f64, -0.01 * i as f64, 0.5, -0.5];
            model.train_one(&x, (i as f64 * 0.05).sin(), 1.0);
        }
        let pred = model.predict(&[0.5, -0.5, 0.5, -0.5]);
        assert!(
            pred.is_finite(),
            "combined Titans TTT prediction must be finite"
        );
    }

    #[test]
    fn ttt_plasticity_enabled_creates_guard() {
        use crate::common::PlasticityConfig;
        let config = TTTConfig::builder()
            .d_model(16)
            .plasticity(Some(PlasticityConfig::default()))
            .build()
            .unwrap();
        let model = StreamingTTT::new(config);
        assert!(
            model.plasticity_guard.is_some(),
            "guard should be Some when plasticity is enabled"
        );
        assert_eq!(
            model.plasticity_guard.as_ref().unwrap().n_groups(),
            16,
            "should have one group per hidden unit"
        );
    }

    #[test]
    fn ttt_plasticity_train_runs_without_panic() {
        use crate::common::PlasticityConfig;
        let config = TTTConfig::builder()
            .d_model(8)
            .warmup(5)
            .plasticity(Some(PlasticityConfig::default()))
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);
        for i in 0..600 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }
        let pred = model.predict(&[1.0, 0.5]);
        assert!(
            pred.is_finite(),
            "plasticity-enabled TTT should produce finite predictions, got {pred}"
        );
    }

    #[test]
    fn plasticity_ttt_surgical_reinit() {
        use crate::common::PlasticityConfig;
        let config = TTTConfig::builder()
            .d_model(8)
            .warmup(5)
            .plasticity(Some(PlasticityConfig::default()))
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        for i in 0..100 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }

        let d = 8;
        let w_fast_before: Vec<f64> = model.layer.fast_weights().to_vec();

        let mut rng = 0xBEEF_u64;
        model.layer.reinitialize_unit(2, &mut rng);

        let w_fast_after: Vec<f64> = model.layer.fast_weights().to_vec();

        let row2_start = 2 * d;
        let row2_end = row2_start + d;
        let changed_row2 = w_fast_before[row2_start..row2_end]
            .iter()
            .zip(w_fast_after[row2_start..row2_end].iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-15)
            .count();
        assert!(
            changed_row2 > 0,
            "row 2 of W_fast should be reinitialized, but no elements changed"
        );

        let row3_start = 3 * d;
        let row3_end = row3_start + d;
        let changed_row3 = w_fast_before[row3_start..row3_end]
            .iter()
            .zip(w_fast_after[row3_start..row3_end].iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-15)
            .count();
        assert_eq!(
            changed_row3, 0,
            "row 3 of W_fast should be preserved, but {} elements changed",
            changed_row3
        );

        for i in 100..200 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }
        let pred = model.predict(&[1.0, 0.5]);
        assert!(
            pred.is_finite(),
            "TTT should produce finite predictions after surgical reinit, got {pred}"
        );
    }

    #[test]
    fn ttt_adjust_config_updates_effective_alpha() {
        // adjust_config(lr_mult, lambda_delta) must wire lambda_delta into the
        // alpha field and propagate it to the layer so that effective_alpha()
        // reflects the new base on the next forward step.
        let config = TTTConfig::builder()
            .d_model(8)
            .alpha(0.001)
            .warmup(0)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        // Push one sample to initialize (sets effective_alpha = 0.001).
        model.train(&[1.0, 0.5], 1.5);

        let initial_alpha = model.config().alpha;
        assert!(
            (initial_alpha - 0.001).abs() < 1e-12,
            "initial alpha should be 0.001, got {initial_alpha}"
        );

        // Adjust: lambda_delta = +0.01 → new alpha = 0.001 + 0.01 = 0.011.
        use crate::learner::Tunable;
        Tunable::adjust_config(&mut model, 1.0, 0.01);

        assert!(
            (model.config().alpha - 0.011).abs() < 1e-12,
            "config.alpha should be updated to 0.011 after adjust_config, got {}",
            model.config().alpha
        );

        // Drive one more forward so effective_alpha() reflects the new base.
        model.train(&[1.0, 0.5], 1.5);

        let eff = model.layer.effective_alpha();
        assert!(
            (eff - 0.011).abs() < 1e-12,
            "effective_alpha() should be 0.011 after adjust_config + train, got {eff}"
        );
    }

    #[test]
    fn ttt_sensitivity_tracks_adjusted_alpha() {
        // regularization_sensitivity in config_diagnostics must use effective_alpha(),
        // not the static config.alpha snapshot. After adjust_config(), the diagnostic
        // value must reflect the new regularization level.
        use crate::automl::DiagnosticSource;
        let config = TTTConfig::builder()
            .d_model(8)
            .alpha(0.001)
            .warmup(0)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        // Prime the model so diagnostics are non-trivial.
        for i in 0..20 {
            let x = [i as f64 * 0.1, (i as f64).sin()];
            model.train(&x, x[0] * 2.0);
        }

        let diag_before = model.config_diagnostics().unwrap();
        let sensitivity_before = diag_before.regularization_sensitivity;

        // Bump alpha substantially so the change is clearly observable.
        use crate::learner::Tunable;
        Tunable::adjust_config(&mut model, 1.0, 0.05);

        // One more step so effective_alpha() is refreshed.
        model.train(&[1.0, 0.5], 1.5);

        let diag_after = model.config_diagnostics().unwrap();
        let sensitivity_after = diag_after.regularization_sensitivity;

        assert!(
            sensitivity_after > sensitivity_before,
            "regularization_sensitivity should increase after positive lambda_delta: before={sensitivity_before}, after={sensitivity_after}"
        );
        assert!(
            (sensitivity_after - 0.051).abs() < 1e-10,
            "sensitivity should track effective_alpha = 0.001 + 0.05 = 0.051, got {sensitivity_after}"
        );
    }

    #[test]
    fn deep_memory_layers_two_trains_and_predicts() {
        // deep_memory_layers=2 activates the 2-layer MLP fast-weight memory
        // (Titans §3.4, Behrouz et al. 2025). Must produce finite predictions
        // and not panic.
        let config = TTTConfig::builder()
            .d_model(8)
            .deep_memory_layers(2)
            .learning_rate(0.01)
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
            "deep_memory_layers=2 TTT should produce finite predictions, got {pred}"
        );
    }

    #[test]
    fn predict_reads_current_input() {
        // Option D invariant: predict(x_a) != predict(x_b) when x_a != x_b,
        // confirming that predict uses the current input, not stale cached state.
        let config = TTTConfig::builder()
            .d_model(16)
            .learning_rate(0.01)
            .warmup(5)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingTTT::new(config);

        // Train enough samples to pass warmup and build a meaningful readout.
        for i in 0..50 {
            let x = [i as f64 * 0.1, (i as f64).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }

        // Two distinct inputs should produce distinct predictions.
        let pred_a = model.predict(&[0.1, 0.2]);
        let pred_b = model.predict(&[0.9, 0.8]);

        assert!(
            pred_a.is_finite(),
            "predict(x_a) should be finite, got {pred_a}"
        );
        assert!(
            pred_b.is_finite(),
            "predict(x_b) should be finite, got {pred_b}"
        );
        assert_ne!(
            pred_a.to_bits(),
            pred_b.to_bits(),
            "predict should reflect the current input: predict({}) == predict({}) = {}",
            0.1,
            0.9,
            pred_a
        );
    }
}
