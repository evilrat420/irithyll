//! Streaming mGRADE (Minimal Recurrent Gating with Delay Convolutions).
//!
//! mGRADE (arXiv July 2025) combines a minGRU cell -- the simplest possible
//! gated recurrence -- with a learnable delay convolution that captures fast
//! temporal patterns. An RLS readout maps the combined representation to
//! predictions.
//!
//! # Architecture
//!
//! ```text
//! x_t -> [DelayConv1D] -> delayed_features -> [MinGRU] -> h_t -> [h_t; delay_out] -> [RLS Readout] -> y_hat_t
//! ```
//!
//! The readout sees `d_hidden + d_in` features (minGRU hidden state +
//! delay conv output), giving it access to both the recurrent summary and
//! the raw delayed temporal features.
//!
//! # References
//!
//! - mGRADE (arXiv July 2025) -- minimal recurrent gating with delay convolutions
//! - Feng et al. (2024) "Were RNNs All We Needed?" -- minGRU

use crate::error::ConfigError;
use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;

// ---------------------------------------------------------------------------
// MGradeConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StreamingMGrade`].
///
/// Create via the builder pattern:
///
/// ```
/// use irithyll::mgrade::MGradeConfig;
///
/// let config = MGradeConfig::builder()
///     .d_in(3)
///     .d_hidden(32)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MGradeConfig {
    /// Input feature dimension (required).
    pub d_in: usize,
    /// MinGRU hidden state dimension (default: 32).
    pub d_hidden: usize,
    /// Delay convolution kernel size (default: 4).
    pub kernel_size: usize,
    /// RLS forgetting factor for readout (default: 0.998).
    pub forgetting_factor: f64,
    /// Initial P matrix diagonal for RLS (default: 100.0).
    pub delta_rls: f64,
    /// Warmup samples before RLS training starts (default: 10).
    pub warmup: usize,
    /// RNG seed (default: 42).
    pub seed: u64,
}

impl Default for MGradeConfig {
    fn default() -> Self {
        Self {
            d_in: 0,
            d_hidden: 32,
            kernel_size: 4,
            forgetting_factor: 0.998,
            delta_rls: 100.0,
            warmup: 10,
            seed: 42,
        }
    }
}

impl std::fmt::Display for MGradeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MGradeConfig(d_in={}, d_hidden={}, kernel_size={}, ff={}, warmup={}, seed={})",
            self.d_in,
            self.d_hidden,
            self.kernel_size,
            self.forgetting_factor,
            self.warmup,
            self.seed
        )
    }
}

// ---------------------------------------------------------------------------
// MGradeConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for [`MGradeConfig`] with validation.
///
/// # Example
///
/// ```
/// use irithyll::mgrade::MGradeConfig;
///
/// let config = MGradeConfig::builder()
///     .d_in(5)
///     .d_hidden(16)
///     .kernel_size(4)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.d_hidden, 16);
/// ```
pub struct MGradeConfigBuilder {
    config: MGradeConfig,
}

impl MGradeConfig {
    /// Create a new builder with default values.
    pub fn builder() -> MGradeConfigBuilder {
        MGradeConfigBuilder {
            config: MGradeConfig::default(),
        }
    }
}

impl MGradeConfigBuilder {
    /// Set the input feature dimension (required).
    pub fn d_in(mut self, d: usize) -> Self {
        self.config.d_in = d;
        self
    }

    /// Set the MinGRU hidden state dimension (default: 32).
    pub fn d_hidden(mut self, d: usize) -> Self {
        self.config.d_hidden = d;
        self
    }

    /// Set the delay convolution kernel size (default: 4).
    pub fn kernel_size(mut self, k: usize) -> Self {
        self.config.kernel_size = k;
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
    /// - `d_in` is 0
    /// - `d_hidden` is 0
    /// - `kernel_size` is less than 2
    /// - `forgetting_factor` is not in (0, 1]
    /// - `delta_rls` is not > 0
    pub fn build(self) -> Result<MGradeConfig, ConfigError> {
        let c = &self.config;
        if c.d_in == 0 {
            return Err(ConfigError::out_of_range("d_in", "must be > 0", c.d_in));
        }
        if c.d_hidden == 0 {
            return Err(ConfigError::out_of_range(
                "d_hidden",
                "must be > 0",
                c.d_hidden,
            ));
        }
        if c.kernel_size < 2 {
            return Err(ConfigError::out_of_range(
                "kernel_size",
                "must be >= 2",
                c.kernel_size,
            ));
        }
        if c.forgetting_factor <= 0.0 || c.forgetting_factor > 1.0 {
            return Err(ConfigError::out_of_range(
                "forgetting_factor",
                "must be in (0, 1]",
                c.forgetting_factor,
            ));
        }
        if c.delta_rls <= 0.0 {
            return Err(ConfigError::out_of_range(
                "delta_rls",
                "must be > 0",
                c.delta_rls,
            ));
        }
        Ok(self.config)
    }
}

// ---------------------------------------------------------------------------
// StreamingMGrade
// ---------------------------------------------------------------------------

/// Streaming mGRADE model with RLS readout.
///
/// Processes one sample at a time. A delay convolution captures fast temporal
/// patterns, a minGRU cell provides recurrent gating, and an RLS readout maps
/// the combined representation to predictions.
///
/// # Example
///
/// ```no_run
/// use irithyll::mgrade::{StreamingMGrade, MGradeConfig};
/// use irithyll::StreamingLearner;
///
/// let config = MGradeConfig::builder().d_in(3).d_hidden(16).build().unwrap();
/// let mut model = StreamingMGrade::new(config);
/// model.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = model.predict(&[1.0, 2.0, 3.0]);
/// ```
pub struct StreamingMGrade {
    config: MGradeConfig,
    delay_conv: irithyll_core::mgrade::DelayConv1D,
    min_gru: irithyll_core::mgrade::MinGRUCell,
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
    /// Welford online mean for input normalization (per feature).
    input_mean: Vec<f64>,
    /// Welford online variance accumulator for input normalization (per feature).
    input_var: Vec<f64>,
    /// Count of samples seen for Welford normalization.
    input_count: u64,
}

impl StreamingMGrade {
    /// Create a new StreamingMGrade from config.
    pub fn new(config: MGradeConfig) -> Self {
        let delay_conv =
            irithyll_core::mgrade::DelayConv1D::new(config.d_in, config.kernel_size, config.seed);
        let min_gru = irithyll_core::mgrade::MinGRUCell::new(config.d_hidden, config.seed);
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta_rls);
        // Readout sees d_hidden + d_in features
        let readout_dim = config.d_hidden + config.d_in;
        let last_features = vec![0.0; readout_dim];

        Self {
            config,
            delay_conv,
            min_gru,
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
            input_mean: Vec::new(),
            input_var: Vec::new(),
            input_count: 0,
        }
    }

    /// Normalize a feature vector via Welford online mean/std, updating stats.
    ///
    /// Returns the normalized features clamped to [-5, 5].
    fn normalize_input(&mut self, features: &[f64]) -> Vec<f64> {
        let d = features.len();
        if self.input_mean.len() != d {
            self.input_mean = vec![0.0; d];
            self.input_var = vec![0.0; d];
        }
        self.input_count += 1;
        let n = self.input_count as f64;
        let mut out = vec![0.0; d];
        for i in 0..d {
            let x = features[i];
            let delta = x - self.input_mean[i];
            self.input_mean[i] += delta / n;
            let delta2 = x - self.input_mean[i];
            self.input_var[i] += delta * delta2;
            let std = if n > 1.0 {
                (self.input_var[i] / (n - 1.0)).sqrt()
            } else {
                1.0
            };
            let std = if std < 1e-8 { 1.0 } else { std };
            out[i] = ((x - self.input_mean[i]) / std).clamp(-5.0, 5.0);
        }
        out
    }

    /// Whether the model has seen enough samples for meaningful predictions.
    #[inline]
    pub fn past_warmup(&self) -> bool {
        self.total_seen > self.config.warmup as u64
    }

    /// Access the config.
    pub fn config(&self) -> &MGradeConfig {
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

    /// Build readout features from delay conv output and minGRU hidden state.
    ///
    /// Layout: [hidden_state; delay_output]
    fn build_readout_features(hidden: &[f64], delay_out: &[f64], out: &mut Vec<f64>) {
        let total = hidden.len() + delay_out.len();
        out.resize(total, 0.0);
        out[..hidden.len()].copy_from_slice(hidden);
        out[hidden.len()..].copy_from_slice(delay_out);
    }
}

impl StreamingLearner for StreamingMGrade {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // 1. Uncertainty-modulated RLS forgetting factor
        let current_uncertainty = self.readout.noise_variance().sqrt();
        const UNCERTAINTY_ALPHA: f64 = 0.001;
        if self.total_seen == 0 {
            self.rolling_uncertainty = current_uncertainty;
        } else {
            self.rolling_uncertainty = (1.0 - UNCERTAINTY_ALPHA) * self.rolling_uncertainty
                + UNCERTAINTY_ALPHA * current_uncertainty;
        }

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
            if self.samples_trained == 0 {
                self.short_term_error = sq_err;
            } else {
                self.short_term_error = 0.9 * self.short_term_error + 0.1 * sq_err;
            }
            // Note: resetting min_gru + delay_conv on drift detection is intentionally omitted.
            // Resetting the recurrent state mid-stream destroys the feature distribution
            // that the RLS readout was trained on, causing prediction explosions that
            // cascade into further resets. Input normalization stabilizes feature scale
            // at the source instead.
            let _short_rmse = self.short_term_error.sqrt();

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
                if self.samples_trained == 1 {
                    self.alignment_ewma = agreement;
                } else {
                    self.alignment_ewma = 0.95 * self.alignment_ewma + 0.05 * agreement;
                }
            }
            self.prev_prev_change = self.prev_change;
            self.prev_change = current_change;
            self.prev_prediction = current_pred;
        }

        // 3. Guard: skip non-finite inputs to prevent NaN from entering recurrent state.
        if !features.iter().all(|f| f.is_finite()) {
            return;
        }

        // 4a. Normalize input via Welford online mean/std before feeding to the pipeline.
        //     Raw feature scale can vary wildly (e.g. Friedman: 0-100, Lorenz: -20 to +20).
        //     Without normalization, the delay conv receives large values and produces
        //     unbounded linear combinations that cause RLS weight explosion.
        let normalized = self.normalize_input(features);

        // Option D step 1: compute readout features from PRE-update recurrent state.
        // delay_conv.forward_predict does not mutate buffer; min_gru.forward_predict does not
        // mutate hidden state. Together they compute what the cell output would be, using the
        // current recurrent state, without advancing it. This is the same feature path used
        // by predict(), making train and predict use identical feature distributions.
        //
        // delay_conv.forward_predict is always safe (reads from buffer, no init requirement).
        // min_gru.forward_predict requires the cell to be initialized (total_seen > 0).
        let pre_readout_features: Option<Vec<f64>> = if self.total_seen > 0 {
            let pre_delay_raw = self.delay_conv.forward_predict(&normalized);
            let pre_delay: Vec<f64> = pre_delay_raw.iter().map(|&v| v.tanh()).collect();
            let pre_cell = self.min_gru.forward_predict(&pre_delay);
            let mut feats = vec![0.0; self.config.d_hidden + self.config.d_in];
            Self::build_readout_features(&pre_cell, &pre_delay, &mut feats);
            Some(feats)
        } else {
            None
        };

        // Option D step 2: advance the pipeline state (delay conv then minGRU).
        let delay_output_raw = self.delay_conv.forward(&normalized);

        // Clamp delay conv output via tanh to ensure it is bounded in [-1, 1].
        //     The delay conv is a linear combination of buffered inputs, so without
        //     squashing it is fully unbounded. All readout features must be bounded
        //     so RLS weights stay stable — this mirrors Mamba's gated output approach.
        let delay_output: Vec<f64> = delay_output_raw.iter().map(|&v| v.tanh()).collect();

        let cell_output = self.min_gru.forward(&delay_output).to_vec();
        self.total_seen += 1;

        // Option D step 3: train RLS on pre-update features (before caching new state).
        // past_warmup() uses total_seen which was just incremented, preserving the same
        // warmup boundary as the original implementation.
        if self.past_warmup() {
            if let Some(ref feats) = pre_readout_features {
                if feats.iter().all(|f| f.is_finite()) {
                    self.readout.train_one(feats, target, weight);
                    self.samples_trained += 1;
                }
            }
        }

        // Build post-update readout features for diagnostics (Frobenius ratio).
        let mut readout_features = std::mem::take(&mut self.last_features);
        Self::build_readout_features(&cell_output, &delay_output, &mut readout_features);

        // Track output utilization (post-update features for diagnostic ratio).
        let frob_sq: f64 = readout_features.iter().map(|s| s * s).sum();
        const FROB_ALPHA: f64 = 0.001;
        self.max_frob_sq_ewma = if frob_sq > self.max_frob_sq_ewma {
            frob_sq
        } else {
            (1.0 - FROB_ALPHA) * self.max_frob_sq_ewma + FROB_ALPHA * frob_sq
        };

        // Cache post-update features for alignment diagnostics and last_features.
        self.last_features = readout_features;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        if self.total_seen == 0 {
            return 0.0;
        }
        // Apply same Welford normalization as train_one (read-only: use frozen stats).
        let d = features.len();
        let mut normalized = vec![0.0; d];
        if self.input_count > 0 && self.input_mean.len() == d {
            let n = self.input_count as f64;
            for i in 0..d {
                let std = if n > 1.0 {
                    (self.input_var[i] / (n - 1.0)).sqrt()
                } else {
                    1.0
                };
                let std = if std < 1e-8 { 1.0 } else { std };
                normalized[i] = ((features[i] - self.input_mean[i]) / std).clamp(-5.0, 5.0);
            }
        } else {
            normalized.copy_from_slice(features);
        }
        let delay_output_raw = self.delay_conv.forward_predict(&normalized);
        // Apply tanh to delay conv output (matches train_one).
        let delay_output: Vec<f64> = delay_output_raw.iter().map(|&v| v.tanh()).collect();
        let cell_output = self.min_gru.forward_predict(&delay_output);
        let mut readout_features = vec![0.0; self.config.d_hidden + self.config.d_in];
        Self::build_readout_features(&cell_output, &delay_output, &mut readout_features);
        self.readout.predict(&readout_features)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_trained
    }

    fn reset(&mut self) {
        self.delay_conv.reset();
        self.min_gru.reset();
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
        self.input_mean.clear();
        self.input_var.clear();
        self.input_count = 0;
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

    #[allow(deprecated)]
    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64) {
        <Self as crate::learner::Tunable>::adjust_config(self, lr_multiplier, lambda_delta);
    }
}

impl crate::learner::Tunable for StreamingMGrade {
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
        self.config.forgetting_factor =
            (self.config.forgetting_factor * lr_multiplier).clamp(0.9, 1.0);
    }
}

impl crate::learner::HasReadout for StreamingMGrade {
    fn readout_weights(&self) -> &[f64] {
        self.readout.weights()
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

impl std::fmt::Debug for StreamingMGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingMGrade")
            .field("d_in", &self.config.d_in)
            .field("d_hidden", &self.config.d_hidden)
            .field("kernel_size", &self.config.kernel_size)
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

impl crate::automl::DiagnosticSource for StreamingMGrade {
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

        // Output Frobenius ratio: current ||features||_2^2 / max(||features||_2^2).
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
    fn mgrade_config_builder_default() {
        let config = MGradeConfig::builder().d_in(3).build().unwrap();
        assert_eq!(config.d_hidden, 32);
        assert_eq!(config.kernel_size, 4);
        assert_eq!(config.warmup, 10);
    }

    #[test]
    fn mgrade_config_rejects_zero_d_in() {
        assert!(MGradeConfig::builder().build().is_err());
    }

    #[test]
    fn mgrade_config_rejects_zero_d_hidden() {
        assert!(MGradeConfig::builder().d_in(3).d_hidden(0).build().is_err());
    }

    #[test]
    fn mgrade_config_rejects_kernel_size_one() {
        assert!(MGradeConfig::builder()
            .d_in(3)
            .kernel_size(1)
            .build()
            .is_err());
    }

    #[test]
    fn mgrade_new_creates_model() {
        let config = MGradeConfig::builder()
            .d_in(3)
            .d_hidden(16)
            .build()
            .unwrap();
        let model = StreamingMGrade::new(config);
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.past_warmup());
    }

    #[test]
    fn mgrade_train_and_predict_finite() {
        let config = MGradeConfig::builder()
            .d_in(2)
            .d_hidden(16)
            .warmup(5)
            .build()
            .unwrap();
        let mut model = StreamingMGrade::new(config);
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
    fn mgrade_reset_clears_state() {
        let config = MGradeConfig::builder()
            .d_in(2)
            .d_hidden(8)
            .warmup(3)
            .build()
            .unwrap();
        let mut model = StreamingMGrade::new(config);
        for i in 0..20 {
            model.train(&[i as f64, (i as f64) * 0.5], i as f64 * 2.0);
        }
        assert!(model.n_samples_seen() > 0);
        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.past_warmup());
    }

    #[test]
    fn mgrade_predict_before_train_returns_zero() {
        let config = MGradeConfig::builder().d_in(2).d_hidden(8).build().unwrap();
        let model = StreamingMGrade::new(config);
        assert_eq!(model.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    #[allow(deprecated)]
    fn mgrade_diagnostics_array_finite() {
        let config = MGradeConfig::builder()
            .d_in(1)
            .d_hidden(8)
            .warmup(3)
            .build()
            .unwrap();
        let mut model = StreamingMGrade::new(config);
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
    #[allow(deprecated)]
    fn mgrade_readout_weights_available_after_training() {
        let config = MGradeConfig::builder()
            .d_in(2)
            .d_hidden(8)
            .warmup(3)
            .build()
            .unwrap();
        let mut model = StreamingMGrade::new(config);
        assert!(model.readout_weights().is_none());
        for i in 0..20 {
            model.train(&[i as f64, (i as f64) * 0.5], i as f64);
        }
        assert!(model.readout_weights().is_some());
    }

    #[test]
    fn mgrade_streaming_learner_boxable() {
        let config = MGradeConfig::builder().d_in(2).d_hidden(8).build().unwrap();
        let model = StreamingMGrade::new(config);
        let _boxed: Box<dyn StreamingLearner> = Box::new(model);
    }

    #[test]
    fn test_mgrade_nan_skipped() {
        // Train past warmup then send a NaN sample; model must remain healthy.
        let config = MGradeConfig::builder()
            .d_in(2)
            .d_hidden(8)
            .warmup(3)
            .build()
            .unwrap();
        let mut model = StreamingMGrade::new(config);
        for i in 0..20 {
            model.train(&[i as f64 * 0.1, (i as f64).sin()], i as f64);
        }
        let samples_before = model.n_samples_seen();
        model.train(&[f64::NAN, 1.0], 1.0);
        assert_eq!(
            model.n_samples_seen(),
            samples_before,
            "NaN input should not increment samples_trained: before={}, after={}",
            samples_before,
            model.n_samples_seen()
        );
        let pred = model.predict(&[1.0, 0.5]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after NaN input, got {pred}"
        );
    }

    #[test]
    #[allow(deprecated)]
    fn test_mgrade_adjust_config() {
        // adjust_config should scale forgetting_factor by lr_multiplier.
        let config = MGradeConfig::builder()
            .d_in(2)
            .d_hidden(8)
            .forgetting_factor(0.998)
            .build()
            .unwrap();
        let mut model = StreamingMGrade::new(config);
        let ff_before = model.config().forgetting_factor;
        model.adjust_config(0.99, 0.0);
        let ff_after = model.config().forgetting_factor;
        assert!(
            ff_after < ff_before,
            "forgetting_factor should decrease after adjust_config(0.99, ..): before={ff_before}, after={ff_after}"
        );
        assert!(
            ff_after >= 0.9,
            "forgetting_factor should not go below 0.9, got {ff_after}"
        );
    }

    #[test]
    fn mgrade_type_is_pascal_case() {
        // MGradeConfig and StreamingMGrade use canonical PascalCase names.
        let config = MGradeConfig::builder().d_in(2).d_hidden(8).build().unwrap();
        let _model: StreamingMGrade = StreamingMGrade::new(config);
    }

    /// Regression test: mGRADE must achieve reasonable RMSE on a sine regression task.
    ///
    /// Before the input normalization + delay conv tanh fix, the delay conv produced
    /// unbounded linear combinations of raw inputs, causing RLS readout weight explosion
    /// and RMSE ~300. After the fix, RMSE should be well under 5.0.
    #[test]
    fn test_mgrade_sine_regression_reasonable() {
        let config = MGradeConfig::builder()
            .d_in(1)
            .d_hidden(16)
            .kernel_size(4)
            .warmup(10)
            .forgetting_factor(0.998)
            .build()
            .unwrap();
        let mut model = StreamingMGrade::new(config);

        // Train on sin(x) for 500 samples.
        let n = 500usize;
        for i in 0..n {
            let x = i as f64 * 0.05;
            model.train(&[x], x.sin());
        }

        // Compute RMSE by replaying the sequence with a fresh model.
        let mut model2 = {
            let config2 = MGradeConfig::builder()
                .d_in(1)
                .d_hidden(16)
                .kernel_size(4)
                .warmup(10)
                .forgetting_factor(0.998)
                .build()
                .unwrap();
            StreamingMGrade::new(config2)
        };
        let mut sq_err_sum = 0.0;
        let mut count = 0usize;
        for i in 0..n {
            let x = i as f64 * 0.05;
            let y = x.sin();
            if model2.past_warmup() {
                let pred = model2.predict(&[x]);
                let err = pred - y;
                sq_err_sum += err * err;
                count += 1;
            }
            model2.train(&[x], y);
        }
        let rmse = if count > 0 {
            (sq_err_sum / count as f64).sqrt()
        } else {
            f64::INFINITY
        };
        assert!(
            rmse < 5.0,
            "mGRADE sine regression RMSE should be < 5.0 after fix, got {rmse:.4} (count={count})"
        );
    }

    #[test]
    fn mgrade_rejects_invalid_forgetting_factor() {
        assert!(
            MGradeConfig::builder()
                .d_in(3)
                .forgetting_factor(0.0)
                .build()
                .is_err(),
            "forgetting_factor=0 must be rejected"
        );
        assert!(
            MGradeConfig::builder()
                .d_in(3)
                .forgetting_factor(-0.5)
                .build()
                .is_err(),
            "negative forgetting_factor must be rejected"
        );
        assert!(
            MGradeConfig::builder()
                .d_in(3)
                .forgetting_factor(1.01)
                .build()
                .is_err(),
            "forgetting_factor>1 must be rejected"
        );
    }

    #[test]
    fn mgrade_rejects_invalid_delta_rls() {
        assert!(
            MGradeConfig::builder()
                .d_in(3)
                .delta_rls(0.0)
                .build()
                .is_err(),
            "delta_rls=0 must be rejected"
        );
        assert!(
            MGradeConfig::builder()
                .d_in(3)
                .delta_rls(-1.0)
                .build()
                .is_err(),
            "delta_rls<0 must be rejected"
        );
    }

    #[test]
    fn mgrade_accepts_forgetting_factor_one() {
        assert!(
            MGradeConfig::builder()
                .d_in(3)
                .forgetting_factor(1.0)
                .build()
                .is_ok(),
            "forgetting_factor=1.0 (no forgetting) should be valid"
        );
    }

    /// Option D correctness: predict(x_t) must strongly correlate with x_t, not x_{t-1}.
    ///
    /// Train on y_t = x_t[0] * 2.0 for many steps. Then verify that predict(x_a) and
    /// predict(x_b) differ meaningfully when x_a and x_b differ in the current input.
    /// A label-leak model would fail to distinguish inputs that differ only in the
    /// current timestep because predict would see stale (prior-step) features.
    #[test]
    fn mgrade_predict_reads_current_input() {
        let config = MGradeConfig::builder()
            .d_in(2)
            .d_hidden(16)
            .kernel_size(4)
            .warmup(5)
            .forgetting_factor(0.999)
            .build()
            .unwrap();
        let mut model = StreamingMGrade::new(config);

        // Train on y_t = x_t[0] * 2.0 for 200 samples.
        for i in 0..200 {
            let x0 = (i as f64) * 0.05;
            model.train(&[x0, 0.0], x0 * 2.0);
        }

        // predict(x_a) and predict(x_b) should differ for x_a != x_b.
        let pred_a = model.predict(&[1.0, 0.0]);
        let pred_b = model.predict(&[5.0, 0.0]);

        assert!(
            pred_a.is_finite() && pred_b.is_finite(),
            "both predictions must be finite: pred_a={pred_a}, pred_b={pred_b}"
        );
        assert!(
            (pred_a - pred_b).abs() > 0.1,
            "predict must respond to current input: pred_a={pred_a} (x=1.0), pred_b={pred_b} (x=5.0), diff={}",
            (pred_a - pred_b).abs()
        );
    }
}
