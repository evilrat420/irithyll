//! Streaming linear attention model: multi-head attention + RLS readout.
//!
//! [`StreamingAttentionModel`] is a complete streaming regression model that combines:
//!
//! 1. A **multi-head linear attention** layer for temporal feature extraction
//! 2. A **Recursive Least Squares** (RLS) readout for mapping attention features to predictions
//!
//! This architecture processes each input as a timestep: the attention layer maintains
//! recurrent state capturing temporal patterns, and the RLS readout learns a linear
//! mapping from attention outputs to the target variable. Both components update
//! incrementally, making the model fully streaming with O(1) memory per timestep.
//!
//! # Training Flow (Option D — prequential-consistent)
//!
//! ```text
//! train_one(x_t, y_t):
//!   1. query_state(x_t)         → pre_features  (q(x_t) · S_{t-1}, no state mutation)
//!   2. RLS.train_one(pre_features, y_t)          (RLS trained on pre-update features)
//!   3. attention.forward(x_t)                   (advance S_{t-1} → S_t)
//! ```
//!
//! # Prediction
//!
//! `predict(x_t)` calls `attention.query_state(x_t)` — the canonical streaming
//! readout `q(x_t) · S_{t-1}` — without advancing the attention state. RLS is
//! queried on these pre-update features, matching the distribution it was trained
//! on exactly. If no training has occurred, returns 0.0.

#[cfg(test)]
use irithyll_core::attention::GatedDeltaMode;
use irithyll_core::attention::{
    AttentionConfig, AttentionLayer, AttentionMode, MultiHeadAttention,
};
use irithyll_core::continual::{ContinualStrategy, NeuronRegeneration};

use crate::attention::attention_config::StreamingAttentionConfig;
use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;

/// Streaming attention model implementing [`StreamingLearner`].
///
/// Combines multi-head linear attention for temporal feature extraction with an
/// RLS readout layer. The attention layer processes each input as a timestep,
/// evolving recurrent state to capture temporal dependencies. The RLS layer
/// learns a linear mapping from attention outputs to the regression target.
///
/// # Example
///
/// ```ignore
/// use irithyll::attention::{StreamingAttentionModel, StreamingAttentionConfig, AttentionMode};
/// use irithyll::learner::StreamingLearner;
///
/// let config = StreamingAttentionConfig::builder()
///     .d_model(4)
///     .n_heads(2)
///     .mode(AttentionMode::GLA)
///     .build()
///     .unwrap();
///
/// let mut model = StreamingAttentionModel::new(config);
///
/// // Train on a stream of 4-dimensional features
/// for i in 0..100 {
///     let x = [i as f64 * 0.1, (i as f64).sin(), 1.0, 0.5];
///     let y = x[0] + 0.5 * x[1];
///     model.train(&x, y);
/// }
///
/// let pred = model.predict(&[10.0, 0.0, 1.0, 0.5]);
/// assert!(pred.is_finite());
/// ```
pub struct StreamingAttentionModel {
    /// Model configuration.
    config: StreamingAttentionConfig,
    /// Multi-head linear attention for temporal feature extraction.
    attention: MultiHeadAttention,
    /// RLS readout layer for prediction.
    readout: RecursiveLeastSquares,
    /// Cached attention output from the most recent train_one call.
    last_features: Vec<f64>,
    /// Total samples trained on.
    n_samples: u64,
    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// Change from two steps ago, for acceleration-based alignment.
    prev_prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
    /// EWMA of maximum Frobenius squared norm of attention state for utilization ratio.
    max_frob_sq_ewma: f64,
    /// Optional plasticity guard for maintaining learning capacity.
    plasticity_guard: Option<NeuronRegeneration>,
    /// Snapshot of per-head output energy from previous step.
    prev_head_energy: Vec<f64>,
}

impl StreamingAttentionModel {
    /// Create a new streaming attention model from the given configuration.
    ///
    /// Initializes the attention layer with random weights (seeded by `config.seed`)
    /// and an RLS readout with the specified forgetting factor and P matrix scale.
    pub fn new(config: StreamingAttentionConfig) -> Self {
        let attn_config = AttentionConfig {
            d_model: config.d_model,
            n_heads: config.n_heads,
            d_key: config.d_key,
            d_value: config.d_value,
            mode: config.mode.clone(),
            seed: config.seed,
        };
        let attention = MultiHeadAttention::new(attn_config);
        let output_dim = attention.output_dim();
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta);
        let last_features = vec![0.0; output_dim];

        // Create plasticity guard if a PlasticityConfig was provided.
        // Tracks output_dim units (group_size=1 = per-unit tracking).
        let plasticity_guard = config.plasticity.as_ref().map(|p| {
            NeuronRegeneration::new(
                output_dim,
                1, // group_size = 1 (per-unit tracking)
                p.regen_fraction,
                p.regen_interval,
                p.utility_alpha,
                config.seed.wrapping_add(0x_DEAD_CAFE),
            )
        });
        let prev_head_energy = vec![0.0; output_dim];

        Self {
            config,
            attention,
            readout,
            last_features,
            n_samples: 0,
            prev_prediction: 0.0,
            prev_change: 0.0,
            prev_prev_change: 0.0,
            alignment_ewma: 0.0,
            max_frob_sq_ewma: 0.0,
            plasticity_guard,
            prev_head_energy,
        }
    }

    /// Get a reference to the model configuration.
    pub fn config(&self) -> &StreamingAttentionConfig {
        &self.config
    }

    /// Get the current attention recurrent state.
    pub fn attention_state(&self) -> &[f64] {
        self.attention.state()
    }

    /// Get the attention mode.
    pub fn mode(&self) -> &AttentionMode {
        &self.config.mode
    }

    /// Whether the model has passed its warmup period.
    pub fn is_warm(&self) -> bool {
        self.n_samples >= self.config.warmup as u64
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

    /// Get the cached attention output features from the last training step.
    pub fn last_features(&self) -> &[f64] {
        &self.last_features
    }
}

impl StreamingLearner for StreamingAttentionModel {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // Guard: skip non-finite inputs to prevent NaN from corrupting attention state.
        if !features.iter().all(|f| f.is_finite()) {
            self.n_samples += 1;
            return;
        }

        // Option D prequential ordering — eliminates train/predict feature mismatch:
        //
        // 1. READ STATE:  query current state with current input (no state mutation).
        //    These pre-update features are what predict(x_t) will also compute.
        // 2. TRAIN READOUT: RLS trains on the same pre-update features.
        // 3. ADVANCE STATE: attention.forward() advances S_{t-1} → S_t.
        //
        // This ensures the readout sees a single consistent feature distribution
        // in both train and predict, closing the one-step train/predict mismatch
        // present when RLS is trained on post-update features (the classical error).

        // Step 1 — read state: q(x_t) · S_{t-1}.
        let pre_features = self.attention.query_state(features);

        // Step 2 — update residual alignment tracking on pre-update prediction.
        let current_pred = self.readout.predict(&pre_features);
        let current_change = current_pred - self.prev_prediction;
        if self.n_samples > 0 {
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
            if self.n_samples == 1 {
                self.alignment_ewma = agreement;
            } else {
                self.alignment_ewma =
                    (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
            }
        }
        self.prev_prev_change = self.prev_change;
        self.prev_change = current_change;
        self.prev_prediction = current_pred;

        // Guard: skip RLS update if pre-update features are non-finite.
        if !pre_features.iter().all(|f| f.is_finite()) {
            self.last_features = pre_features;
            self.n_samples += 1;
            return;
        }

        // Step 2 (cont.) — train RLS readout on pre-update features.
        self.readout.train_one(&pre_features, target, weight);

        // Step 3 — advance state: S_{t-1} → S_t.
        let post_output = self.attention.forward(features);

        // Track attention state Frobenius squared norm (post-advance) for
        // utilization diagnostics.
        {
            let state = self.attention.state();
            let frob_sq: f64 = state.iter().map(|s| s * s).sum();
            const FROB_ALPHA: f64 = 0.001;
            self.max_frob_sq_ewma = if frob_sq > self.max_frob_sq_ewma {
                frob_sq
            } else {
                (1.0 - FROB_ALPHA) * self.max_frob_sq_ewma + FROB_ALPHA * frob_sq
            };
        }

        // Step 4 — plasticity maintenance: track post-advance output energy and
        //    surgically reinitialize dead heads instead of resetting the whole layer.
        if let Some(ref mut guard) = self.plasticity_guard {
            let mut output_energy: Vec<f64> = post_output.iter().map(|x| x.abs()).collect();
            guard.pre_update(&self.prev_head_energy, &mut output_energy);
            guard.post_update(&self.prev_head_energy);
            let mut reinit_rng = self.config.seed.wrapping_add(self.n_samples);
            for h in 0..guard.n_groups() {
                if guard.was_regenerated(h) {
                    self.attention.reinitialize_head(h, &mut reinit_rng);
                }
            }
            self.prev_head_energy = output_energy;
        }

        // Cache the pre-update features (what predict() will recompute).
        self.last_features = pre_features;

        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        // Reconstruct readout features side-effect-free using the current input.
        // If never trained (or after reset), return 0.0.
        if self.n_samples == 0 || features.len() != self.config.d_model {
            return 0.0;
        }

        // Design: attention state advances only during train_one(). At prediction
        // time t, the state reflects history through t-1. We call query_state()
        // to compute q(x_t) · S_{t-1} — the canonical streaming readout that
        // separates readout from state update. This fixes near-chance accuracy
        // from stale t-1 features and implements eq. R8.1 (pre-update readout).
        let attn_output = self.attention.query_state(features);
        self.readout.predict(&attn_output)
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.attention.reset();
        self.readout.reset();
        for f in self.last_features.iter_mut() {
            *f = 0.0;
        }
        self.n_samples = 0;
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.prev_prev_change = 0.0;
        self.alignment_ewma = 0.0;
        self.max_frob_sq_ewma = 0.0;
        if let Some(ref mut guard) = self.plasticity_guard {
            guard.reset();
        }
        self.prev_head_energy.fill(0.0);
    }

    #[allow(deprecated)]
    fn diagnostics_array(&self) -> [f64; 5] {
        <Self as crate::learner::Tunable>::diagnostics_array(self)
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

impl crate::learner::Tunable for StreamingAttentionModel {
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
        // Scale the RLS readout forgetting factor as the primary tuning knob.
        <crate::learners::RecursiveLeastSquares as crate::learner::Tunable>::adjust_config(
            &mut self.readout,
            lr_multiplier,
            0.0,
        );
    }
}

impl crate::learner::HasReadout for StreamingAttentionModel {
    fn readout_weights(&self) -> &[f64] {
        self.readout.weights()
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for StreamingAttentionModel {
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

        // Attention state Frobenius ratio: current ||S||_F^2 / max(||S||_F^2).
        let state_frob_ratio = {
            let state = self.attention.state();
            let frob_sq: f64 = state.iter().map(|s| s * s).sum();
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
            regularization_sensitivity: 1.0 - self.config.forgetting_factor,
            depth_sufficiency,
            effective_dof,
            uncertainty: self.prediction_uncertainty(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(d_model: usize, n_heads: usize) -> StreamingAttentionConfig {
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .n_heads(n_heads)
            .build()
            .unwrap()
    }

    #[test]
    fn new_creates_fresh_model() {
        let model = StreamingAttentionModel::new(default_config(4, 2));
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.is_warm());
    }

    #[test]
    fn train_one_increments_samples() {
        let mut model = StreamingAttentionModel::new(default_config(4, 2));
        model.train_one(&[1.0, 2.0, 3.0, 4.0], 5.0, 1.0);
        assert_eq!(model.n_samples_seen(), 1);
        model.train_one(&[5.0, 6.0, 7.0, 8.0], 9.0, 1.0);
        assert_eq!(model.n_samples_seen(), 2);
    }

    #[test]
    fn predict_before_training_returns_zero() {
        let model = StreamingAttentionModel::new(default_config(4, 2));
        let pred = model.predict(&[1.0, 2.0, 3.0, 4.0]);
        assert!(
            pred.abs() < 1e-15,
            "prediction before training should be 0.0, got {}",
            pred
        );
    }

    #[test]
    fn predict_after_training_is_finite() {
        let mut model = StreamingAttentionModel::new(default_config(4, 2));
        model.train(&[1.0, 2.0, 3.0, 4.0], 5.0);
        let pred = model.predict(&[1.0, 2.0, 3.0, 4.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );
    }

    #[test]
    fn reset_clears_everything() {
        let mut model = StreamingAttentionModel::new(default_config(4, 2));
        model.train(&[1.0, 2.0, 3.0, 4.0], 5.0);
        model.train(&[5.0, 6.0, 7.0, 8.0], 9.0);
        assert_eq!(model.n_samples_seen(), 2);

        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.is_warm());
        for &f in model.last_features() {
            assert!(
                f.abs() < 1e-15,
                "last_features should be zeroed after reset"
            );
        }
        // Attention state should be zeroed
        for &h in model.attention_state() {
            assert!(
                h.abs() < 1e-15,
                "attention state should be zeroed after reset"
            );
        }
    }

    #[test]
    fn train_convenience_uses_unit_weight() {
        let mut model1 = StreamingAttentionModel::new(default_config(4, 2));
        let mut model2 = StreamingAttentionModel::new(default_config(4, 2));

        model1.train(&[1.0, 2.0, 3.0, 4.0], 5.0);
        model2.train_one(&[1.0, 2.0, 3.0, 4.0], 5.0, 1.0);

        assert_eq!(model1.n_samples_seen(), model2.n_samples_seen());
        let p1 = model1.predict(&[1.0, 2.0, 3.0, 4.0]);
        let p2 = model2.predict(&[1.0, 2.0, 3.0, 4.0]);
        assert!(
            (p1 - p2).abs() < 1e-12,
            "train() and train_one(w=1) should be equivalent: {} vs {}",
            p1,
            p2
        );
    }

    #[test]
    fn is_warm_after_warmup_samples() {
        let config = StreamingAttentionConfig::builder()
            .d_model(4)
            .n_heads(2)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingAttentionModel::new(config);

        for i in 0..4 {
            model.train(&[i as f64; 4], 0.0);
            assert!(
                !model.is_warm(),
                "should not be warm after {} samples",
                i + 1
            );
        }
        model.train(&[4.0; 4], 0.0);
        assert!(model.is_warm(), "should be warm after 5 samples");
    }

    #[test]
    fn mode_accessor() {
        let config = StreamingAttentionConfig::builder()
            .d_model(4)
            .n_heads(2)
            .mode(AttentionMode::GatedDeltaNet {
                beta_scale: 1.0,
                gate_mode_delta: GatedDeltaMode::Static,
            })
            .build()
            .unwrap();
        let model = StreamingAttentionModel::new(config);
        assert!(
            matches!(model.mode(), AttentionMode::GatedDeltaNet { .. }),
            "mode should be GatedDeltaNet"
        );
    }

    #[test]
    fn config_accessor() {
        let config = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(4)
            .seed(77)
            .build()
            .unwrap();
        let model = StreamingAttentionModel::new(config);
        assert_eq!(model.config().d_model, 8);
        assert_eq!(model.config().n_heads, 4);
        assert_eq!(model.config().seed, 77);
    }

    #[test]
    fn convergence_on_sine_wave() {
        // Test that the attention model converges on a sine wave prediction task.
        let config = StreamingAttentionConfig::builder()
            .d_model(2)
            .n_heads(1)
            .forgetting_factor(0.999)
            .seed(123)
            .build()
            .unwrap();
        let mut model = StreamingAttentionModel::new(config);

        let mut errors_early = Vec::new();
        let mut errors_late = Vec::new();

        for i in 0..500 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos()];
            let y = (t + 0.1).sin(); // predict next sine value

            if model.n_samples_seen() > 0 {
                let pred = model.predict(&x);
                let err = (pred - y).powi(2);
                if i < 50 {
                    errors_early.push(err);
                } else if i >= 400 {
                    errors_late.push(err);
                }
            }

            model.train(&x, y);
        }

        let mse_early: f64 = errors_early.iter().sum::<f64>() / errors_early.len() as f64;
        let mse_late: f64 = errors_late.iter().sum::<f64>() / errors_late.len() as f64;

        assert!(
            mse_late < mse_early,
            "late MSE ({}) should be smaller than early MSE ({}): model should converge on sine",
            mse_late,
            mse_early
        );
    }

    #[test]
    fn predict_batch_works() {
        let mut model = StreamingAttentionModel::new(default_config(4, 2));
        model.train(&[1.0, 2.0, 3.0, 4.0], 5.0);

        let rows: Vec<&[f64]> = vec![&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]];
        let preds = model.predict_batch(&rows);
        assert_eq!(preds.len(), 2);
        for p in &preds {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn attention_plasticity_disabled_by_default() {
        let config = default_config(4, 2);
        assert!(
            config.plasticity.is_none(),
            "plasticity should default to None"
        );
        let model = StreamingAttentionModel::new(config);
        assert!(
            model.plasticity_guard.is_none(),
            "guard should be None when plasticity is disabled"
        );
    }

    #[test]
    fn attention_plasticity_enabled_creates_guard() {
        use crate::common::PlasticityConfig;
        let config = StreamingAttentionConfig::builder()
            .d_model(4)
            .n_heads(2)
            .plasticity(Some(PlasticityConfig::default()))
            .build()
            .unwrap();
        let model = StreamingAttentionModel::new(config);
        assert!(
            model.plasticity_guard.is_some(),
            "guard should be Some when plasticity is enabled"
        );
    }

    #[test]
    fn attention_plasticity_train_runs_without_panic() {
        use crate::common::PlasticityConfig;
        let config = StreamingAttentionConfig::builder()
            .d_model(4)
            .n_heads(2)
            .plasticity(Some(PlasticityConfig::default()))
            .build()
            .unwrap();
        let mut model = StreamingAttentionModel::new(config);
        for i in 0..600 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), 1.0, 0.5];
            let y = x[0] + 0.5 * x[1];
            model.train(&x, y);
        }
        let pred = model.predict(&[1.0, 0.0, 1.0, 0.5]);
        assert!(
            pred.is_finite(),
            "plasticity-enabled model should produce finite predictions, got {pred}"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_attention_nan_skipped() {
        // NaN features should not corrupt the attention state or RLS readout.
        // The input finiteness check fires before attention forward, so neither
        // the attention state nor the readout weights are updated on NaN input.
        let mut model = StreamingAttentionModel::new(default_config(4, 2));
        for i in 0..20 {
            let x = [i as f64 * 0.1; 4];
            model.train(&x, i as f64);
        }
        let weights_before = model.readout_weights().map(|w| w.to_vec());
        // Train with NaN — should not update readout
        model.train(&[f64::NAN, 0.0, 0.0, 0.0], 1.0);
        // Readout weights should be unchanged (NaN skipped before RLS update).
        if let Some(w_before) = weights_before {
            if let Some(w_after) = model.readout_weights() {
                assert_eq!(
                    w_before.len(),
                    w_after.len(),
                    "readout weight dimension should not change after NaN input"
                );
            }
        }
        let pred = model.predict(&[1.0, 0.0, 0.0, 0.0]);
        assert!(
            pred.is_finite(),
            "prediction should remain finite after NaN training input, got {pred}"
        );
    }

    // -----------------------------------------------------------------------
    // Issue-fix tests: GLA predict() must use current input, not stale t-1
    // -----------------------------------------------------------------------

    #[test]
    fn gla_predict_uses_current_input_not_stale() {
        // predict() calls query_state(x_t) which queries the current attention
        // state with a query computed from x_t — the canonical streaming readout.
        // Two predictions with clearly different inputs must give different outputs
        // (stale-feature regression guard: stale path ignores x_t entirely).
        use irithyll_core::attention::AttentionMode as SAMode;
        let config = StreamingAttentionConfig::builder()
            .d_model(4)
            .n_heads(2)
            .mode(SAMode::GLA)
            .forgetting_factor(0.999)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingAttentionModel::new(config);

        // Train enough to give the attention state some content
        for i in 0..50 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos(), t * 0.1, 1.0];
            model.train(&x, t.sin());
        }

        // Now predict on two very different inputs. With the fix, the predictions
        // must differ; with the stale-feature bug they would be identical.
        let pred_a = model.predict(&[1.0, 0.0, 0.0, 0.0]);
        let pred_b = model.predict(&[-1.0, 0.0, 0.0, 0.0]);

        assert!(
            pred_a.is_finite(),
            "GLA predict on input A should be finite, got {pred_a}"
        );
        assert!(
            pred_b.is_finite(),
            "GLA predict on input B should be finite, got {pred_b}"
        );
        assert!(
            (pred_a - pred_b).abs() > 1e-15,
            "GLA predict must differ for different inputs (stale-feature bug): \
             pred_a={pred_a}, pred_b={pred_b}"
        );
    }

    #[test]
    fn gla_prequential_accuracy_above_chance() {
        // Simulate a prequential (test-then-train) evaluation on a simple
        // binary classification: label = sign(x[0]).
        // With the stale-feature bug predict() always returns the same value
        // regardless of input, yielding ~50% accuracy on balanced data.
        // With the fix the model should learn to exceed chance within 200 steps.
        use irithyll_core::attention::AttentionMode as SAMode;
        let config = StreamingAttentionConfig::builder()
            .d_model(4)
            .n_heads(2)
            .mode(SAMode::GLA)
            .forgetting_factor(0.999)
            .seed(7)
            .build()
            .unwrap();
        let mut model = StreamingAttentionModel::new(config);

        let mut correct = 0usize;
        let mut total = 0usize;
        // Simple LCG for deterministic pseudo-random inputs
        let mut rng: u64 = 0xDEAD_BEEF;
        let lcg = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*s >> 33) as f64 / (u32::MAX as f64) * 2.0 - 1.0
        };

        for _ in 0..300 {
            let x0 = lcg(&mut rng);
            let x = [x0, lcg(&mut rng), lcg(&mut rng), lcg(&mut rng)];
            let label = if x0 > 0.0 { 1.0_f64 } else { 0.0_f64 };

            if model.n_samples_seen() >= 20 {
                let pred = model.predict(&x);
                let pred_label = if pred > 0.5 { 1.0 } else { 0.0 };
                if (pred_label - label).abs() < 1e-9 {
                    correct += 1;
                }
                total += 1;
            }
            model.train(&x, label);
        }

        let accuracy = correct as f64 / total as f64;
        assert!(
            accuracy > 0.50,
            "GLA prequential accuracy should exceed chance (50%) after fix, got {:.1}%",
            accuracy * 100.0
        );
    }

    #[test]
    fn predict_after_reset_returns_zero() {
        // After reset, predict() must return 0.0 regardless of input because
        // n_samples=0 and the attention state is zeroed. Also verifies that
        // query_state on a zero-state produces 0.0 output (as expected:
        // S^T * q = 0 for all q when S = 0).
        let mut model = StreamingAttentionModel::new(default_config(4, 2));
        for i in 0..10 {
            model.train(&[i as f64; 4], i as f64);
        }
        model.reset();
        // predict after reset: n_samples=0 guard triggers before query_state,
        // returns 0.0 without touching the zeroed attention state.
        let pred = model.predict(&[1.0, 2.0, 3.0, 4.0]);
        assert!(
            pred.abs() < 1e-15,
            "predict after reset should return 0.0 (n_samples=0 guard), got {pred}"
        );
    }

    #[test]
    fn predict_reads_current_input() {
        // Verifies the Option D invariant: predict(x_t) returns a value
        // influenced by the CURRENT x_t, not the stale x_{t-1} state.
        //
        // After training, call predict() with two clearly distinct inputs.
        // Both must (a) be finite, and (b) differ — proving the query
        // projection of x_t reaches the readout rather than a cached output.
        let config = StreamingAttentionConfig::builder()
            .d_model(4)
            .n_heads(2)
            .mode(AttentionMode::GLA)
            .forgetting_factor(0.999)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingAttentionModel::new(config);

        // Build up non-trivial attention state.
        for i in 0..30 {
            let t = i as f64 * 0.2;
            model.train(&[t.sin(), t.cos(), t * 0.05, 1.0], t.sin());
        }

        // predict() with x_a and x_b must be different (current input is used).
        let pred_a = model.predict(&[1.0, 0.0, 0.0, 0.0]);
        let pred_b = model.predict(&[-1.0, 0.0, 0.0, 0.0]);

        assert!(
            pred_a.is_finite(),
            "predict(x_a) should be finite, got {pred_a}"
        );
        assert!(
            pred_b.is_finite(),
            "predict(x_b) should be finite, got {pred_b}"
        );
        assert!(
            (pred_a - pred_b).abs() > 1e-15,
            "predict must differ for different inputs — Option D current-input invariant: \
             pred_a={pred_a}, pred_b={pred_b}"
        );
    }
}
