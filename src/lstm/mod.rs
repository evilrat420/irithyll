//! Streaming sLSTM (stabilized LSTM) with exponential gating.
//!
//! sLSTM (Beck et al., 2024 -- xLSTM) replaces sigmoid gates with exponential
//! gates and adds log-domain stabilization for numerically stable long-range
//! memory. The output gate remains sigmoid. A normalizer state tracks
//! cumulative gate products to prevent unbounded cell growth.
//!
//! # Architecture
//!
//! ```text
//! x_t -> [sLSTM Cell: exp gates -> log stabilizer -> cell update] -> h_t -> [RLS Readout] -> y_hat_t
//! ```
//!
//! # References
//!
//! - Beck et al. (2024) "xLSTM: Extended Long Short-Term Memory" NeurIPS

use crate::error::ConfigError;
use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;

// ---------------------------------------------------------------------------
// SLSTMConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StreamingsLSTM`].
///
/// Create via the builder pattern:
///
/// ```
/// use irithyll::lstm::SLSTMConfig;
///
/// let config = SLSTMConfig::builder()
///     .d_model(32)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SLSTMConfig {
    /// Hidden state dimension (default: 32).
    pub d_model: usize,
    /// RLS forgetting factor for readout (default: 0.998).
    pub forgetting_factor: f64,
    /// Initial P matrix diagonal for RLS (default: 100.0).
    pub delta_rls: f64,
    /// Warmup samples before RLS training starts (default: 10).
    pub warmup: usize,
    /// RNG seed (default: 42).
    pub seed: u64,
}

impl Default for SLSTMConfig {
    fn default() -> Self {
        Self {
            d_model: 32,
            forgetting_factor: 0.998,
            delta_rls: 100.0,
            warmup: 10,
            seed: 42,
        }
    }
}

impl std::fmt::Display for SLSTMConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SLSTMConfig(d_model={}, ff={}, delta_rls={}, warmup={}, seed={})",
            self.d_model, self.forgetting_factor, self.delta_rls, self.warmup, self.seed
        )
    }
}

// ---------------------------------------------------------------------------
// SLSTMConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for [`SLSTMConfig`] with validation.
///
/// # Example
///
/// ```
/// use irithyll::lstm::SLSTMConfig;
///
/// let config = SLSTMConfig::builder()
///     .d_model(16)
///     .forgetting_factor(0.995)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.d_model, 16);
/// ```
pub struct SLSTMConfigBuilder {
    config: SLSTMConfig,
}

impl SLSTMConfig {
    /// Create a new builder with default values.
    pub fn builder() -> SLSTMConfigBuilder {
        SLSTMConfigBuilder {
            config: SLSTMConfig::default(),
        }
    }
}

impl SLSTMConfigBuilder {
    /// Set the hidden state dimension (default: 32).
    pub fn d_model(mut self, d: usize) -> Self {
        self.config.d_model = d;
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
    /// Returns [`ConfigError`] if `d_model` is 0.
    pub fn build(self) -> Result<SLSTMConfig, ConfigError> {
        let c = &self.config;
        if c.d_model == 0 {
            return Err(ConfigError::out_of_range(
                "d_model",
                "must be > 0",
                c.d_model,
            ));
        }
        Ok(self.config)
    }
}

// ---------------------------------------------------------------------------
// StreamingsLSTM
// ---------------------------------------------------------------------------

/// Streaming sLSTM model with RLS readout.
///
/// Processes one sample at a time. The sLSTM cell uses exponential gating
/// with log-domain stabilization for numerically stable long-range memory.
/// An RLS readout maps the cell hidden state to predictions.
///
/// # Example
///
/// ```no_run
/// use irithyll::lstm::{StreamingsLSTM, SLSTMConfig};
/// use irithyll::StreamingLearner;
///
/// let config = SLSTMConfig::builder().d_model(16).build().unwrap();
/// let mut model = StreamingsLSTM::new(config);
/// model.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = model.predict(&[1.0, 2.0, 3.0]);
/// ```
pub struct StreamingsLSTM {
    config: SLSTMConfig,
    cell: irithyll_core::lstm::SLSTMCell,
    readout: RecursiveLeastSquares,
    last_features: Vec<f64>,
    total_seen: u64,
    samples_trained: u64,
    /// EWMA of prediction uncertainty for forgetting factor modulation.
    rolling_uncertainty: f64,
    /// Fast-reacting EWMA of squared error for drift detection (alpha=0.1).
    short_term_error: f64,
    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// EWMA of maximum Frobenius squared norm of cell output for utilization ratio.
    max_frob_sq_ewma: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// Change from two steps ago, for acceleration-based alignment.
    prev_prev_change: f64,
}

impl StreamingsLSTM {
    /// Create a new StreamingsLSTM from config.
    pub fn new(config: SLSTMConfig) -> Self {
        let cell = irithyll_core::lstm::SLSTMCell::new(config.d_model, config.seed);
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta_rls);
        let last_features = vec![0.0; config.d_model];

        Self {
            config,
            cell,
            readout,
            last_features,
            total_seen: 0,
            samples_trained: 0,
            rolling_uncertainty: 0.0,
            short_term_error: 0.0,
            prev_prediction: 0.0,
            max_frob_sq_ewma: 0.0,
            alignment_ewma: 0.0,
            prev_change: 0.0,
            prev_prev_change: 0.0,
        }
    }

    /// Whether the model has seen enough samples for meaningful predictions.
    #[inline]
    pub fn past_warmup(&self) -> bool {
        self.total_seen > self.config.warmup as u64
    }

    /// Access the config.
    pub fn config(&self) -> &SLSTMConfig {
        &self.config
    }

    /// Forward-looking prediction uncertainty from the RLS readout.
    ///
    /// Returns the estimated prediction standard deviation, computed as the
    /// square root of the RLS noise variance (EWMA of squared residuals).
    ///
    /// Returns 0.0 before any training has occurred.
    #[inline]
    pub fn prediction_uncertainty(&self) -> f64 {
        self.readout.noise_variance().sqrt()
    }
}

impl StreamingLearner for StreamingsLSTM {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // 1. Uncertainty-modulated RLS forgetting factor
        let current_uncertainty = self.readout.noise_variance().sqrt();
        const UNCERTAINTY_ALPHA: f64 = 0.001;
        self.rolling_uncertainty = (1.0 - UNCERTAINTY_ALPHA) * self.rolling_uncertainty
            + UNCERTAINTY_ALPHA * current_uncertainty;

        if self.rolling_uncertainty > 1e-10 {
            let ratio = (current_uncertainty / self.rolling_uncertainty).clamp(0.5, 3.0);
            let base_ff = self.config.forgetting_factor;
            let adaptive_ff = (base_ff - 0.02 * (ratio - 1.0)).clamp(0.95, base_ff);
            self.readout.set_forgetting_factor(adaptive_ff);
        }

        // 2. Residual alignment tracking (only after warmup)
        if self.past_warmup() {
            let current_pred = self.readout.predict(&self.last_features);
            let pred_error = target - current_pred;

            // Short-term error tracking for drift
            let sq_err = pred_error * pred_error;
            self.short_term_error = 0.9 * self.short_term_error + 0.1 * sq_err;
            let short_rmse = self.short_term_error.sqrt();
            if self.samples_trained >= 100
                && self.rolling_uncertainty > 1e-10
                && short_rmse > 1.5 * self.rolling_uncertainty
            {
                self.cell.reset();
            }

            // Alignment tracking
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
                self.alignment_ewma = 0.95 * self.alignment_ewma + 0.05 * agreement;
            }
            self.prev_prev_change = self.prev_change;
            self.prev_change = current_change;
            self.prev_prediction = current_pred;
        }

        // 3. Forward through sLSTM cell (updates state)
        // Clone immediately to release the borrow on self.cell.
        let cell_output = self.cell.forward(features).to_vec();
        self.total_seen += 1;

        // 4. Track output utilization
        let frob_sq: f64 = cell_output.iter().map(|s| s * s).sum();
        const FROB_ALPHA: f64 = 0.001;
        self.max_frob_sq_ewma = if frob_sq > self.max_frob_sq_ewma {
            frob_sq
        } else {
            (1.0 - FROB_ALPHA) * self.max_frob_sq_ewma + FROB_ALPHA * frob_sq
        };

        // 5. Train RLS readout (after warmup)
        if self.past_warmup() {
            self.readout.train_one(&cell_output, target, weight);
            self.samples_trained += 1;
        }

        // 6. Cache for predict()
        self.last_features = cell_output;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        if self.total_seen == 0 {
            return 0.0;
        }
        let cell_features = self.cell.forward_predict(features);
        self.readout.predict(&cell_features)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_trained
    }

    fn reset(&mut self) {
        self.cell.reset();
        self.readout.reset();
        self.last_features.iter_mut().for_each(|f| *f = 0.0);
        self.total_seen = 0;
        self.samples_trained = 0;
        self.rolling_uncertainty = 0.0;
        self.short_term_error = 0.0;
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

    fn readout_weights(&self) -> Option<&[f64]> {
        self.readout.readout_weights()
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

impl std::fmt::Debug for StreamingsLSTM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingsLSTM")
            .field("d_model", &self.config.d_model)
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

impl crate::automl::DiagnosticSource for StreamingsLSTM {
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

        // sLSTM output Frobenius ratio: current ||h||_2^2 / max(||h||_2^2).
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
            regularization_sensitivity: 0.0,
            depth_sufficiency,
            effective_dof,
            uncertainty: self.readout.noise_variance().sqrt(),
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
    fn slstm_config_builder_default() {
        let config = SLSTMConfig::builder().build().unwrap();
        assert_eq!(config.d_model, 32);
        assert_eq!(config.warmup, 10);
    }

    #[test]
    fn slstm_config_rejects_zero_d_model() {
        assert!(SLSTMConfig::builder().d_model(0).build().is_err());
    }

    #[test]
    fn slstm_new_creates_model() {
        let config = SLSTMConfig::builder().d_model(16).build().unwrap();
        let model = StreamingsLSTM::new(config);
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.past_warmup());
    }

    #[test]
    fn slstm_train_and_predict_finite() {
        let config = SLSTMConfig::builder()
            .d_model(16)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingsLSTM::new(config);
        for i in 0..50 {
            let x = [i as f64 * 0.1, (i as f64).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }
        let pred = model.predict(&[1.0, 0.5]);
        assert!(pred.is_finite(), "prediction must be finite, got {pred}");
        assert_eq!(model.n_samples_seen(), 45); // 50 - 5 warmup
    }

    #[test]
    fn slstm_reset_clears_state() {
        let config = SLSTMConfig::builder().d_model(8).warmup(3).build().unwrap();
        let mut model = StreamingsLSTM::new(config);
        for i in 0..20 {
            model.train(&[i as f64], i as f64 * 2.0);
        }
        assert!(model.n_samples_seen() > 0);
        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.past_warmup());
    }

    #[test]
    fn slstm_predict_before_train_returns_zero() {
        let config = SLSTMConfig::builder().d_model(8).build().unwrap();
        let model = StreamingsLSTM::new(config);
        assert_eq!(model.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn slstm_diagnostics_array_finite() {
        let config = SLSTMConfig::builder().d_model(8).warmup(3).build().unwrap();
        let mut model = StreamingsLSTM::new(config);
        for i in 0..30 {
            model.train(&[i as f64 * 0.1], i as f64);
        }
        let diag = model.diagnostics_array();
        for (idx, val) in diag.iter().enumerate() {
            assert!(
                val.is_finite(),
                "diagnostics[{idx}] must be finite, got {val}"
            );
        }
    }

    #[test]
    fn slstm_readout_weights_available_after_training() {
        let config = SLSTMConfig::builder().d_model(8).warmup(3).build().unwrap();
        let mut model = StreamingsLSTM::new(config);
        assert!(model.readout_weights().is_none());
        for i in 0..20 {
            model.train(&[i as f64], i as f64);
        }
        assert!(model.readout_weights().is_some());
    }

    #[test]
    fn slstm_streaming_learner_boxable() {
        let config = SLSTMConfig::builder().d_model(8).build().unwrap();
        let model = StreamingsLSTM::new(config);
        let _boxed: Box<dyn StreamingLearner> = Box::new(model);
    }
}
