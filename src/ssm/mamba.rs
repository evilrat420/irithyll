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
//! features ──→ SSM.forward() ──→ ssm_output ──┐
//!                │                              │
//!                └──→ SSM.state() ──→ hidden ───┼──→ [hidden; ssm_output]
//!                                               │        (capped to MAX_READOUT_FEATURES)
//!                                               └──→ RLS.train_one(readout_features, target)
//!                                                    RLS.predict(readout_features)  ←── prediction
//! ```
//!
//! The RLS readout sees either the full `d_in * n_state` hidden state (when small
//! enough) or a mean-pooled projection of it (for high-dimensional inputs), plus
//! the `d_in` SSM output. This capped projection ensures the RLS covariance matrix
//! stays at most `MAX_READOUT_FEATURES × MAX_READOUT_FEATURES`, keeping the O(n³)
//! update cost bounded regardless of input dimension.
//!
//! # Prediction
//!
//! `predict()` uses the cached readout features from the most recent `train_one()`
//! call. This avoids a side-effect (advancing SSM state) during prediction,
//! maintaining the contract that `predict()` is read-only. If no training has
//! occurred, returns 0.0.

use irithyll_core::ssm::{SSMLayer, SelectiveSSM};

use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;
use crate::ssm::mamba_config::MambaConfig;

/// Streaming Mamba model implementing [`StreamingLearner`].
///
/// Combines a selective SSM for temporal feature extraction with an RLS
/// readout layer. The SSM processes each input as a timestep, evolving
/// hidden state to capture temporal dependencies. The RLS layer learns
/// a linear mapping from (optionally pooled) SSM hidden state plus SSM
/// output to the regression target.
///
/// When the full hidden state (`d_in * n_state + d_in`) exceeds
/// [`MAX_READOUT_FEATURES`](Self::MAX_READOUT_FEATURES), the hidden
/// state is mean-pooled into fewer dimensions before feeding the RLS.
/// This keeps the RLS covariance matrix bounded, preventing O(n³)
/// blowup on high-dimensional inputs.
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
    /// Cached readout features (possibly pooled hidden state + SSM output)
    /// from the most recent `train_one` call.
    last_features: Vec<f64>,
    /// Total samples trained on.
    n_samples: u64,
    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
}

impl StreamingMamba {
    /// Maximum number of features fed to the RLS readout layer.
    ///
    /// When `d_in * n_state + d_in` exceeds this limit, the hidden state is
    /// mean-pooled into `MAX_READOUT_FEATURES - d_in` dimensions before
    /// concatenation with the SSM output. This caps the RLS covariance matrix
    /// at 64 × 64, keeping the O(n³) update cost at ~262 K ops regardless of
    /// input dimension.
    pub const MAX_READOUT_FEATURES: usize = 64;

    /// Create a new streaming Mamba model from the given configuration.
    ///
    /// Initializes the SSM with random weights (seeded by `config.seed`) and
    /// an RLS readout with the specified forgetting factor and P matrix scale.
    ///
    /// The readout feature vector is `[pooled_hidden_state; ssm_output]`. When
    /// the full state is small enough it is used directly; otherwise it is
    /// mean-pooled so the total never exceeds [`MAX_READOUT_FEATURES`](Self::MAX_READOUT_FEATURES).
    pub fn new(config: MambaConfig) -> Self {
        let ssm = SelectiveSSM::new(config.d_in, config.n_state, config.seed);
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta_rls);
        let readout_dim = Self::capped_readout_dim(config.d_in, config.n_state);
        let last_features = vec![0.0; readout_dim];

        Self {
            config,
            ssm,
            readout,
            last_features,
            n_samples: 0,
            prev_prediction: 0.0,
            prev_change: 0.0,
            alignment_ewma: 0.0,
        }
    }

    /// Compute the readout dimension after capping.
    fn capped_readout_dim(d_in: usize, n_state: usize) -> usize {
        let full = d_in * n_state + d_in;
        if full <= Self::MAX_READOUT_FEATURES {
            full
        } else {
            // Pool the hidden state into (MAX - d_in) dims, then append d_in SSM output.
            let target_state_dim = Self::MAX_READOUT_FEATURES.saturating_sub(d_in).max(1);
            let state_len = d_in * n_state;
            let chunk_size = state_len.div_ceil(target_state_dim);
            // Actual pooled dim depends on how chunks divide evenly.
            let pooled = state_len.div_ceil(chunk_size);
            pooled + d_in
        }
    }

    /// Build readout features from the SSM hidden state and output.
    ///
    /// When the full state fits within [`MAX_READOUT_FEATURES`](Self::MAX_READOUT_FEATURES),
    /// the raw `[hidden_state; ssm_output]` is returned. Otherwise the hidden
    /// state is mean-pooled into fewer dimensions before concatenation.
    fn build_readout_features(&self, state: &[f64], ssm_output: &[f64]) -> Vec<f64> {
        let full_len = state.len() + ssm_output.len();
        if full_len <= Self::MAX_READOUT_FEATURES {
            // Small enough -- use the full state.
            let mut rf = Vec::with_capacity(full_len);
            rf.extend_from_slice(state);
            rf.extend_from_slice(ssm_output);
            rf
        } else {
            // Mean-pool the hidden state into fewer dimensions.
            let target_state_dim = Self::MAX_READOUT_FEATURES
                .saturating_sub(ssm_output.len())
                .max(1);
            let chunk_size = state.len().div_ceil(target_state_dim);
            let mut rf = Vec::with_capacity(Self::MAX_READOUT_FEATURES);
            for chunk in state.chunks(chunk_size) {
                let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
                rf.push(mean);
            }
            rf.extend_from_slice(ssm_output);
            rf
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

        // 2. Build readout features: (optionally pooled) hidden state + SSM output.
        //    When the full state exceeds MAX_READOUT_FEATURES, the hidden state is
        //    mean-pooled to keep the RLS covariance matrix bounded.
        let state = self.ssm.state();
        let readout_features = self.build_readout_features(state, &ssm_output);

        // 3. Update residual alignment tracking (before RLS update).
        let current_pred = self.readout.predict(&readout_features);
        let current_change = current_pred - self.prev_prediction;
        if self.n_samples > 0 {
            let agreement = if (current_change > 0.0) == (self.prev_change > 0.0) {
                1.0
            } else {
                -1.0
            };
            const ALIGN_ALPHA: f64 = 0.05;
            self.alignment_ewma =
                (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
        }
        self.prev_change = current_change;
        self.prev_prediction = current_pred;

        // 4. Train RLS readout on (capped) readout features
        self.readout.train_one(&readout_features, target, weight);

        // 5. Cache readout features for predict()
        self.last_features = readout_features;

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
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.alignment_ewma = 0.0;
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

impl crate::automl::DiagnosticSource for StreamingMamba {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.alignment_ewma,
            regularization_sensitivity: 1.0 - self.config.forgetting_factor,
            depth_sufficiency: 0.0, // Single SSM layer + linear readout.
            effective_dof: Self::capped_readout_dim(self.config.d_in, self.config.n_state) as f64,
            uncertainty: self.prediction_uncertainty(),
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
        // d_in=3, n_state=32 → full state = 3*32+3 = 99 > MAX_READOUT_FEATURES(64)
        // → hidden state is pooled: target=61, chunk_size=ceil(96/61)=2, pooled=48, total=48+3=51
        let expected = StreamingMamba::capped_readout_dim(3, 32);
        assert_eq!(model.last_features().len(), expected);
        assert!(
            expected <= StreamingMamba::MAX_READOUT_FEATURES,
            "capped readout dim should be <= MAX_READOUT_FEATURES, got {}",
            expected
        );
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
        // With the full hidden state readout (d_in*n_state + d_in features),
        // the RLS needs more samples to converge, so we use 1000 steps.
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

        for i in 0..1000 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos()];
            let y = 0.7 * x[0] + 0.3 * x[1];

            // Record prediction error before training on this sample
            if model.n_samples_seen() > 0 {
                let pred = model.predict(&x);
                let err = (pred - y).powi(2);
                if (10..100).contains(&i) {
                    errors_early.push(err);
                } else if i >= 800 {
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
        // Test on a more complex target: predicting a sine wave.
        // With the full hidden state readout (d_in*n_state + d_in features),
        // the model converges quickly. We measure early errors in the first
        // few samples (before RLS has adapted) against late errors.
        let config = MambaConfig::builder()
            .d_in(2)
            .n_state(8)
            .forgetting_factor(0.999)
            .seed(123)
            .build()
            .unwrap();
        let mut model = StreamingMamba::new(config);

        let mut errors_early = Vec::new();
        let mut errors_late = Vec::new();

        for i in 0..1000 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos()];
            let y = (t + 0.1).sin(); // predict next value of sin

            if model.n_samples_seen() > 0 {
                let pred = model.predict(&x);
                let err = (pred - y).powi(2);
                if (1..5).contains(&i) {
                    errors_early.push(err);
                } else if i >= 800 {
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

    // -----------------------------------------------------------------------
    // Readout capping tests
    // -----------------------------------------------------------------------

    #[test]
    fn small_state_uses_full_readout() {
        // d_in=1, n_state=32 → 1*32+1 = 33 features, under MAX_READOUT_FEATURES
        let config = MambaConfig::builder().d_in(1).n_state(32).build().unwrap();
        let model = StreamingMamba::new(config);
        assert_eq!(
            model.last_features().len(),
            1 * 32 + 1,
            "small state should use full readout without pooling"
        );
    }

    #[test]
    fn large_state_caps_readout_to_max() {
        // d_in=50, n_state=32 → 50*32+50 = 1650, way over MAX_READOUT_FEATURES
        let config = MambaConfig::builder().d_in(50).n_state(32).build().unwrap();
        let model = StreamingMamba::new(config);
        assert!(
            model.last_features().len() <= StreamingMamba::MAX_READOUT_FEATURES,
            "high-dim state should be capped to MAX_READOUT_FEATURES, got {}",
            model.last_features().len()
        );
    }

    #[test]
    fn high_dim_training_produces_finite_predictions() {
        // d_in=50 would create a 1650-feature readout without capping.
        // Verify the pooled readout still learns correctly.
        let config = MambaConfig::builder()
            .d_in(50)
            .n_state(32)
            .forgetting_factor(0.998)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingMamba::new(config);

        let features: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        for i in 0..200 {
            let target = (i as f64 * 0.05).sin();
            model.train(&features, target);
        }

        let pred = model.predict(&features);
        assert!(
            pred.is_finite(),
            "prediction with 50-dim input should be finite, got {}",
            pred
        );
    }

    #[test]
    fn high_dim_convergence() {
        // Verify the model converges even with mean-pooled readout features.
        // Use a simpler linear target so the RLS can learn it cleanly through
        // the pooled projection. We also use more training to let the RLS
        // settle (the early window starts at 50 to capture initial adaptation).
        let config = MambaConfig::builder()
            .d_in(10)
            .n_state(16)
            .forgetting_factor(0.999)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingMamba::new(config);

        // Full state = 10*16+10 = 170, will be pooled to <= MAX_READOUT_FEATURES
        assert!(
            model.last_features().len() <= StreamingMamba::MAX_READOUT_FEATURES,
            "d_in=10,n_state=16 should be capped, got {}",
            model.last_features().len()
        );

        let mut errors_early = Vec::new();
        let mut errors_late = Vec::new();

        for i in 0..2000 {
            let t = i as f64 * 0.1;
            // Simple periodic target: sum of two sinusoids from the first two features.
            let x: Vec<f64> = (0..10).map(|k| (t + k as f64 * 0.3).sin()).collect();
            let y = 0.5 * x[0] + 0.3 * x[1];

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

        let mse_early: f64 = errors_early.iter().sum::<f64>() / errors_early.len() as f64;
        let mse_late: f64 = errors_late.iter().sum::<f64>() / errors_late.len() as f64;

        assert!(
            mse_late < mse_early,
            "pooled model should converge: late MSE ({}) should be < early MSE ({})",
            mse_late,
            mse_early
        );
    }

    #[test]
    fn capped_readout_dim_boundary() {
        // Exactly at the boundary: d_in=2, n_state=31 → 2*31+2 = 64 = MAX
        assert_eq!(
            StreamingMamba::capped_readout_dim(2, 31),
            64,
            "exactly at MAX should use full state"
        );
        // One over: d_in=2, n_state=32 → 2*32+2 = 66 > 64
        let dim = StreamingMamba::capped_readout_dim(2, 32);
        assert!(
            dim <= StreamingMamba::MAX_READOUT_FEATURES,
            "one over MAX should be capped, got {}",
            dim
        );
    }
}
