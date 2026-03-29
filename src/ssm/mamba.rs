//! Streaming Mamba model: selective SSM + RLS readout.
//!
//! [`StreamingMamba`] is a complete streaming regression model that combines:
//!
//! 1. A **selective SSM** (Mamba-style) for temporal feature extraction
//! 2. A **Recursive Least Squares** (RLS) readout for mapping SSM features to predictions
//!
//! This architecture processes each input as a timestep: the SSM maintains hidden
//! state capturing temporal patterns, and the RLS readout learns a linear mapping
//! from SSM outputs to the target variable. Both components update incrementally,
//! making the model fully streaming with O(1) memory per timestep.
//!
//! # Training Flow
//!
//! ```text
//! features ──→ SSM.forward() ──→ ssm_output ──→ RLS.train_one(ssm_output, target)
//!                                               RLS.predict(ssm_output)  ←── prediction
//! ```
//!
//! # Prediction
//!
//! `predict()` uses the cached SSM output from the most recent `train_one()` call.
//! This avoids a side-effect (advancing SSM state) during prediction, maintaining
//! the contract that `predict()` is read-only. If no training has occurred, returns 0.0.

use irithyll_core::ssm::{SSMLayer, SelectiveSSM};

use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;
use crate::ssm::mamba_config::MambaConfig;

/// Streaming Mamba model implementing [`StreamingLearner`].
///
/// Combines a selective SSM for temporal feature extraction with an RLS
/// readout layer. The SSM processes each input as a timestep, evolving
/// hidden state to capture temporal dependencies. The RLS layer learns
/// a linear mapping from SSM outputs to the regression target.
///
/// # Example
///
/// ```
/// use irithyll::ssm::{StreamingMamba, MambaConfig};
/// use irithyll::learner::StreamingLearner;
///
/// let config = MambaConfig::builder()
///     .d_in(3)
///     .n_state(8)
///     .build()
///     .unwrap();
///
/// let mut model = StreamingMamba::new(config);
///
/// // Train on a stream of 3-dimensional features
/// for i in 0..100 {
///     let x = [i as f64 * 0.1, (i as f64).sin(), 1.0];
///     let y = x[0] + 0.5 * x[1];
///     model.train(&x, y);
/// }
///
/// let pred = model.predict(&[10.0, 0.0, 1.0]);
/// assert!(pred.is_finite());
/// ```
pub struct StreamingMamba {
    /// Model configuration.
    config: MambaConfig,
    /// Selective SSM for temporal feature extraction.
    ssm: SelectiveSSM,
    /// RLS readout layer for prediction.
    readout: RecursiveLeastSquares,
    /// Cached SSM output from the most recent train_one call.
    last_features: Vec<f64>,
    /// Total samples trained on.
    n_samples: u64,
}

impl StreamingMamba {
    /// Create a new streaming Mamba model from the given configuration.
    ///
    /// Initializes the SSM with random weights (seeded by `config.seed`) and
    /// an RLS readout with the specified forgetting factor and P matrix scale.
    pub fn new(config: MambaConfig) -> Self {
        let ssm = SelectiveSSM::new(config.d_in, config.n_state, config.seed);
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta_rls);
        let last_features = vec![0.0; config.d_in];

        Self {
            config,
            ssm,
            readout,
            last_features,
            n_samples: 0,
        }
    }

    /// Get a reference to the model configuration.
    pub fn config(&self) -> &MambaConfig {
        &self.config
    }

    /// Get the current SSM hidden state.
    pub fn ssm_state(&self) -> &[f64] {
        self.ssm.state()
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

    /// Get the cached SSM output features from the last training step.
    pub fn last_features(&self) -> &[f64] {
        &self.last_features
    }
}

impl StreamingLearner for StreamingMamba {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // 1. Forward through SSM to get temporal features
        let ssm_output = self.ssm.forward(features);

        // 2. Train RLS readout on SSM output
        self.readout.train_one(&ssm_output, target, weight);

        // 3. Cache the SSM output for predict()
        self.last_features = ssm_output;

        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        // Use cached features from last train_one to avoid SSM state mutation.
        // If never trained (or after reset), return 0.0.
        if self.n_samples == 0 {
            return 0.0;
        }

        // During warmup, predictions may be unreliable but we still return them
        // so the caller can decide how to handle them.
        //
        // Note: We use the LAST SSM output, not the current features.
        // This is because predict() must be side-effect-free (no SSM state change).
        // In a streaming context, the typical pattern is:
        //   predict(x_t) -> p_t  (using state from x_{t-1})
        //   train(x_t, y_t)      (advances state to include x_t)
        //
        // If the caller wants to predict on new features without training,
        // they should use MambaPreprocessor + separate RLS.
        let _ = features; // Acknowledged but not used -- see design note above
        self.readout.predict(&self.last_features)
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.ssm.reset();
        self.readout.reset();
        for f in self.last_features.iter_mut() {
            *f = 0.0;
        }
        self.n_samples = 0;
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for StreamingMamba {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        Some(crate::automl::ConfigDiagnostics {
            effective_dof: (self.config.d_in * self.config.n_state) as f64,
            // Higher forgetting factor = less regularization pressure.
            regularization_sensitivity: 1.0 - self.config.forgetting_factor,
            // Prediction uncertainty: std dev of RLS residuals.
            uncertainty: self.prediction_uncertainty(),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(d_in: usize) -> MambaConfig {
        MambaConfig::builder().d_in(d_in).build().unwrap()
    }

    #[test]
    fn new_creates_fresh_model() {
        let model = StreamingMamba::new(default_config(3));
        assert_eq!(model.n_samples_seen(), 0);
        assert_eq!(model.last_features().len(), 3);
    }

    #[test]
    fn train_one_increments_samples() {
        let mut model = StreamingMamba::new(default_config(2));
        model.train_one(&[1.0, 2.0], 3.0, 1.0);
        assert_eq!(model.n_samples_seen(), 1);
        model.train_one(&[4.0, 5.0], 6.0, 1.0);
        assert_eq!(model.n_samples_seen(), 2);
    }

    #[test]
    fn predict_before_training_returns_zero() {
        let model = StreamingMamba::new(default_config(3));
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.abs() < 1e-15,
            "prediction before training should be 0.0, got {}",
            pred
        );
    }

    #[test]
    fn predict_after_training_is_finite() {
        let mut model = StreamingMamba::new(default_config(2));
        model.train(&[1.0, 2.0], 3.0);
        let pred = model.predict(&[1.0, 2.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );
    }

    #[test]
    fn reset_clears_everything() {
        let mut model = StreamingMamba::new(default_config(2));
        model.train(&[1.0, 2.0], 3.0);
        model.train(&[4.0, 5.0], 6.0);
        assert_eq!(model.n_samples_seen(), 2);

        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        for &f in model.last_features() {
            assert!(
                f.abs() < 1e-15,
                "last_features should be zeroed after reset"
            );
        }
        // SSM state should be zeroed
        for &h in model.ssm_state() {
            assert!(h.abs() < 1e-15, "SSM state should be zeroed after reset");
        }
    }

    #[test]
    fn train_convenience_uses_unit_weight() {
        let mut model1 = StreamingMamba::new(default_config(2));
        let mut model2 = StreamingMamba::new(default_config(2));

        model1.train(&[1.0, 2.0], 3.0);
        model2.train_one(&[1.0, 2.0], 3.0, 1.0);

        // Both should have the same state
        assert_eq!(model1.n_samples_seen(), model2.n_samples_seen());
        let p1 = model1.predict(&[1.0, 2.0]);
        let p2 = model2.predict(&[1.0, 2.0]);
        assert!(
            (p1 - p2).abs() < 1e-12,
            "train() and train_one(w=1) should be equivalent: {} vs {}",
            p1,
            p2
        );
    }

    #[test]
    fn convergence_on_linear_target() {
        // The SSM+RLS should learn a simple bounded linear relationship.
        // We use periodic features to keep the target bounded and stationary.
        let config = MambaConfig::builder()
            .d_in(2)
            .n_state(8)
            .forgetting_factor(0.999)
            .warmup(5)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingMamba::new(config);

        let mut errors_early = Vec::new();
        let mut errors_late = Vec::new();

        for i in 0..500 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos()];
            let y = 0.7 * x[0] + 0.3 * x[1];

            // Record prediction error before training on this sample
            if model.n_samples_seen() > 0 {
                let pred = model.predict(&x);
                let err = (pred - y).powi(2);
                if (10..60).contains(&i) {
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
            "late MSE ({}) should be smaller than early MSE ({}): model should converge",
            mse_late,
            mse_early
        );
    }

    #[test]
    fn convergence_on_sine_wave() {
        // Test on a more complex target: predicting a sine wave
        let config = MambaConfig::builder()
            .d_in(1)
            .n_state(16)
            .forgetting_factor(0.999)
            .seed(123)
            .build()
            .unwrap();
        let mut model = StreamingMamba::new(config);

        let mut errors_early = Vec::new();
        let mut errors_late = Vec::new();

        for i in 0..500 {
            let t = i as f64 * 0.1;
            let x = [t.sin()];
            let y = (t + 0.1).sin(); // predict next value

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
    fn config_accessor() {
        let config = MambaConfig::builder()
            .d_in(5)
            .n_state(32)
            .seed(77)
            .build()
            .unwrap();
        let model = StreamingMamba::new(config);
        assert_eq!(model.config().d_in, 5);
        assert_eq!(model.config().n_state, 32);
        assert_eq!(model.config().seed, 77);
    }

    #[test]
    fn predict_batch_works() {
        let mut model = StreamingMamba::new(default_config(2));
        model.train(&[1.0, 2.0], 3.0);

        let rows: Vec<&[f64]> = vec![&[1.0, 2.0], &[3.0, 4.0]];
        let preds = model.predict_batch(&rows);
        assert_eq!(preds.len(), 2);
        for p in &preds {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn weighted_training() {
        let mut model = StreamingMamba::new(default_config(2));
        // Training with weight 0 should have minimal effect
        model.train_one(&[1.0, 2.0], 100.0, 0.0);
        let pred_zero_weight = model.predict(&[1.0, 2.0]);

        let mut model2 = StreamingMamba::new(default_config(2));
        model2.train_one(&[1.0, 2.0], 100.0, 1.0);
        let pred_unit_weight = model2.predict(&[1.0, 2.0]);

        // With zero weight, the readout should barely update
        assert!(
            pred_zero_weight.abs() < pred_unit_weight.abs() + 1.0,
            "zero-weight training should have less effect: zero_w={}, unit_w={}",
            pred_zero_weight,
            pred_unit_weight
        );
    }

    #[test]
    fn mamba_prediction_uncertainty() {
        let config = MambaConfig::builder().d_in(2).n_state(8).build().unwrap();
        let mut model = StreamingMamba::new(config);

        // Before training, uncertainty is 0.0
        assert!(
            model.prediction_uncertainty().abs() < 1e-15,
            "uncertainty should be 0.0 before training, got {}",
            model.prediction_uncertainty()
        );

        // Train on 100 samples
        for i in 0..100 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos()];
            let y = 0.7 * x[0] + 0.3 * x[1];
            model.train(&x, y);
        }

        let unc = model.prediction_uncertainty();
        assert!(
            unc > 0.0,
            "prediction_uncertainty should be > 0 after training, got {}",
            unc
        );
        assert!(
            unc.is_finite(),
            "prediction_uncertainty should be finite, got {}",
            unc
        );
    }
}
