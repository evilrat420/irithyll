//! Streaming Test-Time Training (TTT) layers.
//!
//! The hidden state is a linear model with weights W, updated by gradient
//! descent on self-supervised reconstruction at every time step. This makes
//! the model's representation continuously adapt to the input distribution.
//!
//! Optionally includes Titans-style momentum and weight decay for
//! non-stationary streaming environments.
//!
//! # Architecture
//!
//! ```text
//! x_t → [TTT Layer: project → reconstruct → update W → query] → z_t → [RLS Readout] → ŷ_t
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
    /// Weight decay / forgetting factor, Titans-style (default: 0.0001).
    pub alpha: f64,
    /// Momentum coefficient, Titans-style (default: 0.0 = none).
    pub momentum: f64,
    /// RLS forgetting factor for readout (default: 0.998).
    pub forgetting_factor: f64,
    /// Initial P matrix diagonal for RLS (default: 100.0).
    pub delta_rls: f64,
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
            alpha: 0.0001,
            momentum: 0.0,
            forgetting_factor: 0.998,
            delta_rls: 100.0,
            warmup: 10,
            seed: 42,
        }
    }
}

impl std::fmt::Display for TTTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TTTConfig(d_model={}, n_heads={}, eta={}, alpha={}, momentum={}, ff={}, warmup={}, seed={})",
            self.d_model, self.n_heads, self.eta, self.alpha, self.momentum,
            self.forgetting_factor, self.warmup, self.seed
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

    /// Set the weight decay coefficient, Titans-style (default: 0.0001).
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
    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
}

impl StreamingTTT {
    /// Create a new StreamingTTT from config.
    pub fn new(config: TTTConfig) -> Self {
        let layer = TTTLayer::new(
            config.d_model,
            config.eta,
            config.alpha,
            config.momentum > 0.0,
            config.momentum,
            config.seed,
        );
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta_rls);
        let last_features = vec![0.0; config.d_model];

        Self {
            config,
            layer,
            readout,
            last_features,
            total_seen: 0,
            samples_trained: 0,
            rolling_uncertainty: 0.0,
            prev_prediction: 0.0,
            prev_change: 0.0,
            alignment_ewma: 0.0,
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
            let ratio = (current_uncertainty / self.rolling_uncertainty).clamp(0.5, 2.0);
            let base_ff = self.config.forgetting_factor;
            let adaptive_ff = base_ff * (1.0 - 0.001 * (ratio - 1.0));
            self.readout.set_forgetting_factor(adaptive_ff);
        }

        // Always forward through TTT layer (even during warmup, to warm up projections)
        let ttt_output = self.layer.forward(features);
        self.total_seen += 1;

        // Only train RLS after warmup
        if self.past_warmup() {
            // Update residual alignment tracking (before RLS update).
            let current_pred = self.readout.predict(&ttt_output);
            let current_change = current_pred - self.prev_prediction;
            if self.samples_trained > 0 {
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

            self.readout.train_one(&ttt_output, target, weight);
            self.samples_trained += 1;
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
        self.readout.reset();
        for f in self.last_features.iter_mut() {
            *f = 0.0;
        }
        self.total_seen = 0;
        self.samples_trained = 0;
        self.rolling_uncertainty = 0.0;
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
            .field("warmup", &self.config.warmup)
            .field("total_seen", &self.total_seen)
            .field("samples_trained", &self.samples_trained)
            .field("past_warmup", &self.past_warmup())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for StreamingTTT {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.alignment_ewma,
            // Weight decay alpha = regularization strength.
            regularization_sensitivity: self.config.alpha,
            depth_sufficiency: 0.0, // Single TTT layer + linear readout.
            // Layer dimension as effective DOF.
            effective_dof: self.config.d_model as f64,
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
}
