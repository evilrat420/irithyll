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
use irithyll_core::continual::{ContinualStrategy, NeuronRegeneration};

// ---------------------------------------------------------------------------
// SLSTMConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StreamingLSTM`].
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
    /// Enable plasticity maintenance via neuron regeneration (default: false).
    ///
    /// When enabled, tracks per-hidden-unit state energy and periodically
    /// reinitializes dead units to maintain learning capacity over long
    /// streams (Dohare et al., Nature 2024).
    pub plasticity: bool,
}

impl Default for SLSTMConfig {
    fn default() -> Self {
        Self {
            d_model: 32,
            forgetting_factor: 0.998,
            delta_rls: 100.0,
            warmup: 10,
            seed: 42,
            plasticity: false,
        }
    }
}

impl std::fmt::Display for SLSTMConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SLSTMConfig(d_model={}, ff={}, delta_rls={}, warmup={}, seed={}, plasticity={})",
            self.d_model,
            self.forgetting_factor,
            self.delta_rls,
            self.warmup,
            self.seed,
            self.plasticity
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

    /// Enable or disable plasticity maintenance (default: false).
    ///
    /// When enabled, tracks per-hidden-unit state energy and periodically
    /// reinitializes dead units to maintain learning capacity over long
    /// streams (Dohare et al., Nature 2024).
    pub fn plasticity(mut self, p: bool) -> Self {
        self.config.plasticity = p;
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
// StreamingLSTM
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
/// use irithyll::lstm::{StreamingLSTM, SLSTMConfig};
/// use irithyll::StreamingLearner;
///
/// let config = SLSTMConfig::builder().d_model(16).build().unwrap();
/// let mut model = StreamingLSTM::new(config);
/// model.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = model.predict(&[1.0, 2.0, 3.0]);
/// ```
pub struct StreamingLSTM {
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
    /// Optional plasticity guard for maintaining learning capacity.
    ///
    /// Tracks per-hidden-unit state energy and reinitializes dead units
    /// to prevent loss of plasticity over long streams.
    plasticity_guard: Option<NeuronRegeneration>,
    /// Snapshot of hidden state from previous step, used for plasticity
    /// utility computation (delta |h|).
    prev_h_energy: Vec<f64>,
    /// Welford online mean for input normalization (per feature).
    input_mean: Vec<f64>,
    /// Welford online variance accumulator for input normalization (per feature).
    input_var: Vec<f64>,
    /// Count of samples seen for Welford normalization.
    input_count: u64,
}

impl StreamingLSTM {
    /// Create a new StreamingLSTM from config.
    pub fn new(config: SLSTMConfig) -> Self {
        let cell = irithyll_core::lstm::SLSTMCell::new(config.d_model, config.seed);
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta_rls);
        let last_features = vec![0.0; config.d_model];

        // Create plasticity guard if enabled.
        // Tracks d_model hidden units (group_size=1), regenerating the bottom 1%
        // every 500 steps with utility EWMA alpha=0.99.
        let plasticity_guard = if config.plasticity {
            Some(NeuronRegeneration::new(
                config.d_model, // n_params = one per hidden unit
                1,              // group_size = 1 (per-unit tracking)
                0.01,           // regen_fraction = 1%
                500,            // regen_interval
                0.99,           // utility_alpha
                config.seed.wrapping_add(0x_DEAD_CAFE),
            ))
        } else {
            None
        };
        let prev_h_energy = vec![0.0; config.d_model];

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
            plasticity_guard,
            prev_h_energy,
            input_mean: Vec::new(),
            input_var: Vec::new(),
            input_count: 0,
        }
    }

    /// Normalize a feature vector via Welford online mean/std, updating stats.
    ///
    /// Returns the normalized features clamped to [-5, 5].
    /// On the first call the dimension is inferred from `features.len()`.
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

impl StreamingLearner for StreamingLSTM {
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
            // Note: cell.reset() on drift detection is intentionally omitted.
            // Resetting the cell mid-stream destroys the feature distribution that
            // the RLS readout was trained on, causing catastrophic prediction errors
            // that cascade into further resets. Input normalization stabilizes the
            // feature scale at the source instead.
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

        // 3. Guard: skip non-finite inputs to prevent NaN from entering the cell state.
        if !features.iter().all(|f| f.is_finite()) {
            return;
        }

        // 4. Normalize input via Welford online mean/std before feeding to cell.
        //    Raw feature scale can vary wildly (e.g. Friedman: 0-100, Lorenz: -20 to +20).
        //    Without normalization, the cell receives large pre-activations that make
        //    the hidden state erratic and cause RLS weight explosion.
        let normalized = self.normalize_input(features);

        // 5. Forward through sLSTM cell (updates state).
        //    Clone immediately to release the borrow on self.cell.
        let mut cell_output = self.cell.forward(&normalized).to_vec();
        self.total_seen += 1;

        // 6. Clamp cell output to [-3, 3] as a safety net.
        //    The sLSTM cell is theoretically bounded (o * c/n where o=sigmoid, c/n ~ EMA of tanh),
        //    but under extreme inputs the normalizer state can briefly allow larger values.
        //    Clamping ensures the RLS readout always sees a stable, bounded feature space.
        for v in &mut cell_output {
            *v = v.clamp(-3.0, 3.0);
        }

        // 7. Track output utilization
        let frob_sq: f64 = cell_output.iter().map(|s| s * s).sum();
        const FROB_ALPHA: f64 = 0.001;
        self.max_frob_sq_ewma = if frob_sq > self.max_frob_sq_ewma {
            frob_sq
        } else {
            (1.0 - FROB_ALPHA) * self.max_frob_sq_ewma + FROB_ALPHA * frob_sq
        };

        // 8. Train RLS readout (after warmup)
        if self.past_warmup() {
            if !cell_output.iter().all(|f| f.is_finite()) {
                return;
            }
            self.readout.train_one(&cell_output, target, weight);
            self.samples_trained += 1;
        }

        // 6. Plasticity maintenance (Dohare et al., Nature 2024):
        //    Track per-unit hidden state activation as utility signal.
        //    When a unit dies (utility drops to bottom fraction), surgically
        //    reinitialize its weight columns while preserving all other units.
        if let Some(ref mut guard) = self.plasticity_guard {
            let mut h_energy: Vec<f64> = self.cell.hidden_state().iter().map(|x| x.abs()).collect();
            guard.pre_update(&self.prev_h_energy, &mut h_energy);
            guard.post_update(&self.prev_h_energy);
            // Surgical per-unit reinit: only dead units get recycled.
            // Uses reinitialize_unit() which reinits weight rows in W_f/W_i/W_o/W_z,
            // resets biases (forget=1.0), and zeros h/c/n/m for just that unit.
            let mut reinit_rng = self
                .config
                .seed
                .wrapping_add(0xCAFE_BABE_u64.wrapping_mul(self.total_seen));
            for j in 0..guard.n_groups() {
                if guard.was_regenerated(j) {
                    self.cell.reinitialize_unit(j, &mut reinit_rng);
                }
            }
            self.prev_h_energy = self.cell.hidden_state().iter().map(|x| x.abs()).collect();
        }

        // 7. Cache for predict()
        self.last_features = cell_output;
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
        let mut cell_features = self.cell.forward_predict(&normalized);
        for v in &mut cell_features {
            *v = v.clamp(-3.0, 3.0);
        }
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
        if let Some(ref mut guard) = self.plasticity_guard {
            guard.reset();
        }
        self.prev_h_energy.fill(0.0);
        self.input_mean.clear();
        self.input_var.clear();
        self.input_count = 0;
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

impl std::fmt::Debug for StreamingLSTM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingLSTM")
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

impl crate::automl::DiagnosticSource for StreamingLSTM {
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
// Backward-compatible type alias
// ---------------------------------------------------------------------------

/// Deprecated name. Use [`StreamingLSTM`] instead.
pub type StreamingsLSTM = StreamingLSTM;

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

    #[test]
    fn slstm_plasticity_disabled_by_default() {
        let config = SLSTMConfig::builder().d_model(8).build().unwrap();
        assert!(!config.plasticity, "plasticity should default to false");
        let model = StreamingsLSTM::new(config);
        assert!(
            model.plasticity_guard.is_none(),
            "guard should be None when plasticity is disabled"
        );
    }

    #[test]
    fn slstm_plasticity_enabled_creates_guard() {
        let config = SLSTMConfig::builder()
            .d_model(16)
            .plasticity(true)
            .build()
            .unwrap();
        assert!(config.plasticity, "plasticity should be true when set");
        let model = StreamingsLSTM::new(config);
        assert!(
            model.plasticity_guard.is_some(),
            "guard should be Some when plasticity is enabled"
        );
        let guard = model.plasticity_guard.as_ref().unwrap();
        assert_eq!(
            guard.n_groups(),
            16,
            "should have one group per hidden unit"
        );
    }

    #[test]
    fn slstm_plasticity_train_runs_without_panic() {
        let config = SLSTMConfig::builder()
            .d_model(8)
            .warmup(3)
            .plasticity(true)
            .build()
            .unwrap();
        let mut model = StreamingsLSTM::new(config);
        for i in 0..600 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin()];
            let y = x[0] * 2.0 + 1.0;
            model.train(&x, y);
        }
        let pred = model.predict(&[1.0, 0.5]);
        assert!(
            pred.is_finite(),
            "plasticity-enabled model should produce finite predictions, got {pred}"
        );
    }

    #[test]
    fn slstm_plasticity_reset_clears_guard() {
        let config = SLSTMConfig::builder()
            .d_model(8)
            .warmup(3)
            .plasticity(true)
            .build()
            .unwrap();
        let mut model = StreamingsLSTM::new(config);
        for i in 0..20 {
            model.train(&[i as f64], i as f64);
        }
        model.reset();
        let guard = model.plasticity_guard.as_ref().unwrap();
        assert_eq!(
            guard.n_updates(),
            0,
            "plasticity guard should be reset after model reset"
        );
        assert!(
            model.prev_h_energy.iter().all(|&e| e == 0.0),
            "prev_h_energy should be zeroed after reset"
        );
    }

    #[test]
    fn test_lstm_nan_input_skipped() {
        // Train with valid samples past warmup, then send a NaN sample.
        // Model should not panic and should remain healthy (weights finite).
        let config = SLSTMConfig::builder().d_model(8).warmup(3).build().unwrap();
        let mut model = StreamingLSTM::new(config);
        for i in 0..20 {
            model.train(&[i as f64 * 0.1], i as f64);
        }
        let samples_before = model.n_samples_seen();
        // Feed NaN — should be silently skipped
        model.train(&[f64::NAN], 1.0);
        // Sample count should not change (NaN skipped at readout step)
        // Note: total_seen increments before the finiteness check, but samples_trained does not.
        assert_eq!(
            model.n_samples_seen(),
            samples_before,
            "NaN sample should not increment samples_trained: before={}, after={}",
            samples_before,
            model.n_samples_seen()
        );
        // Prediction should still be finite after NaN was fed
        let pred = model.predict(&[1.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after NaN input, got {pred}"
        );
    }

    #[test]
    fn test_streaming_lstm_alias() {
        // StreamingLSTM (new name) and StreamingsLSTM (alias) refer to the same type.
        let config = SLSTMConfig::builder().d_model(8).build().unwrap();
        let model: StreamingLSTM = StreamingLSTM::new(config.clone());
        let _alias: StreamingsLSTM = StreamingsLSTM::new(config);
        assert_eq!(
            model.config().d_model,
            8,
            "StreamingLSTM should have correct d_model"
        );
    }

    /// Regression test: sLSTM must achieve reasonable RMSE on a sine regression task.
    ///
    /// Before the input normalization + cell output clamp fix, the RLS readout
    /// would receive an inconsistent feature distribution (cell resets mid-stream)
    /// and produce RMSE ~180. After the fix, RMSE should be well under 5.0.
    #[test]
    fn test_slstm_sine_regression_reasonable() {
        let config = SLSTMConfig::builder()
            .d_model(16)
            .warmup(10)
            .forgetting_factor(0.998)
            .build()
            .unwrap();
        let mut model = StreamingLSTM::new(config);

        // Train on sin(x) for 500 samples.
        let n = 500usize;
        for i in 0..n {
            let x = i as f64 * 0.05;
            model.train(&[x], x.sin());
        }

        // Compute RMSE on training samples (streaming: we evaluate as we trained).
        // Re-run predictions over the same sequence to measure fit quality.
        let mut model2 = {
            let config2 = SLSTMConfig::builder()
                .d_model(16)
                .warmup(10)
                .forgetting_factor(0.998)
                .build()
                .unwrap();
            StreamingLSTM::new(config2)
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
            "sLSTM sine regression RMSE should be < 5.0 after fix, got {rmse:.4} (count={count})"
        );
    }
}
