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
//! # Training Flow
//!
//! ```text
//! features ──→ attention.forward() ──→ attn_output ──→ RLS.train_one(attn_output, target)
//!                                                      RLS.predict(attn_output)  ←── prediction
//! ```
//!
//! # Prediction
//!
//! `predict()` uses the cached attention output from the most recent `train_one()` call.
//! This avoids a side-effect (advancing attention state) during prediction, maintaining
//! the contract that `predict()` is read-only. If no training has occurred, returns 0.0.

use irithyll_core::attention::{
    AttentionConfig, AttentionLayer, AttentionMode, MultiHeadAttention,
};

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
        // 1. Forward through attention to get temporal features
        let attn_output = self.attention.forward(features);

        // Track attention state Frobenius squared norm for utilization ratio.
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

        // 2. Update residual alignment tracking (acceleration-based).
        let current_pred = self.readout.predict(&attn_output);
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
            self.alignment_ewma =
                (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
        }
        self.prev_prev_change = self.prev_change;
        self.prev_change = current_change;
        self.prev_prediction = current_pred;

        // 3. Train RLS readout on attention output
        self.readout.train_one(&attn_output, target, weight);

        // 4. Cache the attention output for predict()
        self.last_features = attn_output;

        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        // Use cached features from last train_one to avoid attention state mutation.
        // If never trained (or after reset), return 0.0.
        if self.n_samples == 0 {
            return 0.0;
        }

        // During warmup, predictions may be unreliable but we still return them
        // so the caller can decide how to handle them.
        //
        // Note: We use the LAST attention output, not the current features.
        // This is because predict() must be side-effect-free (no attention state change).
        // In a streaming context, the typical pattern is:
        //   predict(x_t) -> p_t  (using state from x_{t-1})
        //   train(x_t, y_t)      (advances state to include x_t)
        //
        // If the caller wants to predict on new features without training,
        // they should use AttentionPreprocessor + separate RLS.
        let _ = features; // Acknowledged but not used -- see design note above
        self.readout.predict(&self.last_features)
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
            .mode(AttentionMode::GatedDeltaNet)
            .build()
            .unwrap();
        let model = StreamingAttentionModel::new(config);
        assert!(
            matches!(model.mode(), AttentionMode::GatedDeltaNet),
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
}
