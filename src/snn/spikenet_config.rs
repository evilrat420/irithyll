//! Configuration and builder for [`SpikeNet`](super::SpikeNet).
//!
//! All parameters are specified in natural f64 units. The builder validates
//! ranges before construction and the config is converted to Q1.14 fixed-point
//! internally when creating the underlying `SpikeNetFixed`.

use crate::error::ConfigError;

/// Synaptic learning rule for [`SpikeNet`](super::SpikeNet).
///
/// Selects which weight-update algorithm is applied during `train_one`.
///
/// # Variants
///
/// - [`Stdp`](LearningRule::Stdp): spike-timing-dependent plasticity (e-prop
///   three-factor rule, Bellec et al. 2020). This is the default and the only
///   fully-implemented rule; it is always active regardless of this setting.
///
/// - [`PpProp`](LearningRule::PpProp): predictive-propagation (PP-prop).
///   PP-prop generalises e-prop by routing the learning signal through a
///   predictive model of presynaptic activity, enabling faster credit
///   assignment across more timesteps (Kaiser et al., 2022, NeurIPS). When
///   `SpikeNet` is set to `PpProp` it falls back to e-prop (the `Stdp` path)
///   because `SpikeNetFixed` currently implements the e-prop kernel only.
///   The variant is provided as a forward-compatible API placeholder: code
///   that selects `PpProp` will compile and run without panic; it will behave
///   identically to `Stdp` until a PP-prop kernel is wired into the fixed-point
///   core. A `tracing` warning is emitted at construction time to make this
///   visible.
///
///   Reference: Kaiser et al., "PP-prop: Predictive-Propagation for Online
///   Learning in Spiking Neural Networks", NeurIPS 2022.
///   <https://proceedings.neurips.cc/paper_files/paper/2022/hash/c3b46c4e36ee2db1cb02a4b3bb5d7a04-Abstract-Conference.html>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum LearningRule {
    /// e-prop three-factor rule (Bellec et al. 2020). Default.
    #[default]
    Stdp,
    /// Predictive-propagation (Kaiser et al. 2022, NeurIPS).
    ///
    /// Falls back to e-prop until a PP-prop kernel lands in `SpikeNetFixed`.
    /// Emits a `tracing::warn!` at construction time.
    PpProp,
}

/// Configuration for a streaming SpikeNet.
///
/// All parameters use natural f64 units. The underlying fixed-point network
/// will quantize these to Q1.14 during construction.
///
/// # Defaults
///
/// | Parameter | Default | Description |
/// |-----------|---------|-------------|
/// | `n_hidden` | 64 | Number of hidden LIF neurons |
/// | `n_outputs` | 1 | Number of readout neurons |
/// | `alpha` | 0.95 | Membrane decay (0 = no memory, 1 = no decay) |
/// | `kappa` | 0.99 | Eligibility trace decay |
/// | `kappa_out` | 0.90 | Readout membrane decay |
/// | `learning_rate` | 0.001 | Learning rate |
/// | `v_thr` | 0.50 | Firing threshold |
/// | `gamma` | 0.30 | Surrogate gradient dampening |
/// | `spike_threshold` | 0.05 | Delta encoding threshold |
/// | `seed` | 42 | PRNG seed for weight initialization |
/// | `weight_init_range` | 0.10 | Weights init in `[-range, range]` |
/// | `learning_rule` | `Stdp` | Synaptic learning rule |
#[derive(Debug, Clone)]
pub struct SpikeNetConfig {
    /// Number of hidden LIF neurons.
    pub n_hidden: usize,
    /// Number of output readout neurons.
    pub n_outputs: usize,
    /// Membrane decay factor (0.0 to 1.0). Higher = longer memory.
    pub alpha: f64,
    /// Eligibility trace decay factor (0.0 to 1.0). Higher = longer credit.
    pub kappa: f64,
    /// Readout membrane decay factor (0.0 to 1.0).
    pub kappa_out: f64,
    /// Learning rate. Positive, typically 0.0001 to 0.1.
    pub learning_rate: f64,
    /// Firing threshold. Positive, typically 0.1 to 1.0.
    pub v_thr: f64,
    /// Surrogate gradient dampening factor (0.0 to 1.0).
    pub gamma: f64,
    /// Delta encoding threshold. Positive, controls spike sensitivity.
    pub spike_threshold: f64,
    /// PRNG seed for reproducible weight initialization.
    pub seed: u64,
    /// Weight initialization range: weights sampled from `[-range, range]`.
    pub weight_init_range: f64,
    /// Enable astrocyte-gated modulation of input weights.
    pub astrocyte: bool,
    /// Astrocyte EWMA time constant (higher = slower adaptation). Default: 1000.
    pub astrocyte_tau: f64,
    /// Synaptic learning rule. Default: [`LearningRule::Stdp`] (e-prop).
    ///
    /// [`LearningRule::PpProp`] is a forward-compatible placeholder that falls
    /// back to e-prop with a warning until a PP-prop kernel lands in the core.
    pub learning_rule: LearningRule,
}

impl Default for SpikeNetConfig {
    fn default() -> Self {
        Self {
            n_hidden: 64,
            n_outputs: 1,
            alpha: 0.95,
            kappa: 0.99,
            kappa_out: 0.90,
            learning_rate: 0.001,
            v_thr: 0.50,
            gamma: 0.30,
            spike_threshold: 0.05,
            seed: 42,
            weight_init_range: 0.10,
            astrocyte: false,
            astrocyte_tau: 1000.0,
            learning_rule: LearningRule::Stdp,
        }
    }
}

impl SpikeNetConfig {
    /// Create a builder for `SpikeNetConfig` with default parameters.
    pub fn builder() -> SpikeNetConfigBuilder {
        SpikeNetConfigBuilder::new()
    }

    /// Validate the configuration, returning an error if any parameter is invalid.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.n_hidden == 0 {
            return Err(ConfigError::out_of_range(
                "n_hidden",
                "must be > 0",
                self.n_hidden,
            ));
        }
        if self.n_outputs == 0 {
            return Err(ConfigError::out_of_range(
                "n_outputs",
                "must be > 0",
                self.n_outputs,
            ));
        }
        if self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(ConfigError::out_of_range(
                "alpha",
                "must be in [0.0, 1.0]",
                self.alpha,
            ));
        }
        if self.kappa < 0.0 || self.kappa > 1.0 {
            return Err(ConfigError::out_of_range(
                "kappa",
                "must be in [0.0, 1.0]",
                self.kappa,
            ));
        }
        if self.kappa_out < 0.0 || self.kappa_out > 1.0 {
            return Err(ConfigError::out_of_range(
                "kappa_out",
                "must be in [0.0, 1.0]",
                self.kappa_out,
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(ConfigError::out_of_range(
                "learning_rate",
                "must be > 0.0",
                self.learning_rate,
            ));
        }
        if self.learning_rate > 1.0 {
            return Err(ConfigError::out_of_range(
                "learning_rate",
                "must be <= 1.0",
                self.learning_rate,
            ));
        }
        if self.v_thr <= 0.0 {
            return Err(ConfigError::out_of_range(
                "v_thr",
                "must be > 0.0",
                self.v_thr,
            ));
        }
        if self.gamma < 0.0 || self.gamma > 1.0 {
            return Err(ConfigError::out_of_range(
                "gamma",
                "must be in [0.0, 1.0]",
                self.gamma,
            ));
        }
        if self.spike_threshold <= 0.0 {
            return Err(ConfigError::out_of_range(
                "spike_threshold",
                "must be > 0.0",
                self.spike_threshold,
            ));
        }
        if self.weight_init_range <= 0.0 {
            return Err(ConfigError::out_of_range(
                "weight_init_range",
                "must be > 0.0",
                self.weight_init_range,
            ));
        }
        if self.weight_init_range > 1.9 {
            return Err(ConfigError::out_of_range(
                "weight_init_range",
                "must be <= 1.9 (Q1.14 limit)",
                self.weight_init_range,
            ));
        }
        if self.astrocyte_tau <= 0.0 {
            return Err(ConfigError::out_of_range(
                "astrocyte_tau",
                "must be > 0.0",
                self.astrocyte_tau,
            ));
        }
        Ok(())
    }
}

/// Builder for [`SpikeNetConfig`] with fluent API and validation.
///
/// # Example
///
/// ```
/// use irithyll::snn::SpikeNetConfig;
///
/// let config = SpikeNetConfig::builder()
///     .n_hidden(128)
///     .learning_rate(0.005)
///     .alpha(0.9)
///     .seed(12345)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.n_hidden, 128);
/// assert_eq!(config.seed, 12345);
/// ```
pub struct SpikeNetConfigBuilder {
    config: SpikeNetConfig,
}

impl SpikeNetConfigBuilder {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self {
            config: SpikeNetConfig::default(),
        }
    }

    /// Set the number of hidden LIF neurons.
    pub fn n_hidden(mut self, n: usize) -> Self {
        self.config.n_hidden = n;
        self
    }

    /// Set the number of output readout neurons.
    pub fn n_outputs(mut self, n: usize) -> Self {
        self.config.n_outputs = n;
        self
    }

    /// Set the membrane decay factor (0.0 to 1.0).
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set the eligibility trace decay factor (0.0 to 1.0).
    pub fn kappa(mut self, kappa: f64) -> Self {
        self.config.kappa = kappa;
        self
    }

    /// Set the readout membrane decay factor (0.0 to 1.0).
    pub fn kappa_out(mut self, kappa_out: f64) -> Self {
        self.config.kappa_out = kappa_out;
        self
    }

    /// Set the learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set the firing threshold.
    pub fn v_thr(mut self, v_thr: f64) -> Self {
        self.config.v_thr = v_thr;
        self
    }

    /// Set the surrogate gradient dampening factor.
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma;
        self
    }

    /// Set the delta encoding threshold for spike generation.
    pub fn spike_threshold(mut self, threshold: f64) -> Self {
        self.config.spike_threshold = threshold;
        self
    }

    /// Set the PRNG seed for weight initialization.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Set the weight initialization range.
    pub fn weight_init_range(mut self, range: f64) -> Self {
        self.config.weight_init_range = range;
        self
    }

    /// Enable or disable astrocyte-gated modulation of input weights.
    pub fn astrocyte(mut self, enabled: bool) -> Self {
        self.config.astrocyte = enabled;
        self
    }

    /// Set the astrocyte EWMA time constant (higher = slower adaptation).
    pub fn astrocyte_tau(mut self, tau: f64) -> Self {
        self.config.astrocyte_tau = tau;
        self
    }

    /// Set the synaptic learning rule.
    ///
    /// [`LearningRule::PpProp`] is a forward-compatible placeholder that falls
    /// back to e-prop (`Stdp`) with a warning at construction time.
    pub fn learning_rule(mut self, rule: LearningRule) -> Self {
        self.config.learning_rule = rule;
        self
    }

    /// Validate and build the configuration.
    ///
    /// Returns `Err(ConfigError)` if any parameter is out of range.
    pub fn build(self) -> Result<SpikeNetConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for SpikeNetConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pp_prop_variant_constructible() {
        // LearningRule::PpProp must be constructible and round-trip via config.
        // It falls back to Stdp at the fixed-point kernel level, but the variant
        // must exist, be default-distinguishable from Stdp, and be settable via
        // the builder without error.
        let config = SpikeNetConfig::builder()
            .n_hidden(16)
            .learning_rate(0.01)
            .learning_rule(LearningRule::PpProp)
            .build()
            .unwrap();

        assert_eq!(
            config.learning_rule,
            LearningRule::PpProp,
            "builder must store PpProp, got {:?}",
            config.learning_rule
        );

        // Stdp is still the default.
        let default_config = SpikeNetConfig::default();
        assert_eq!(
            default_config.learning_rule,
            LearningRule::Stdp,
            "default learning_rule must be Stdp, got {:?}",
            default_config.learning_rule
        );

        // Both variants must be distinct.
        assert_ne!(
            LearningRule::Stdp,
            LearningRule::PpProp,
            "Stdp and PpProp must be distinct variants"
        );
    }

    #[test]
    fn default_config_is_valid() {
        let config = SpikeNetConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn builder_produces_valid_config() {
        let config = SpikeNetConfig::builder()
            .n_hidden(32)
            .n_outputs(2)
            .learning_rate(0.005)
            .alpha(0.9)
            .build()
            .unwrap();

        assert_eq!(config.n_hidden, 32);
        assert_eq!(config.n_outputs, 2);
        assert!((config.learning_rate - 0.005).abs() < 1e-10);
        assert!((config.alpha - 0.9).abs() < 1e-10);
    }

    #[test]
    fn zero_hidden_rejected() {
        let result = SpikeNetConfig::builder().n_hidden(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn zero_outputs_rejected() {
        let result = SpikeNetConfig::builder().n_outputs(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn negative_eta_rejected() {
        let result = SpikeNetConfig::builder().learning_rate(-0.01).build();
        assert!(result.is_err());
    }

    #[test]
    fn alpha_out_of_range_rejected() {
        assert!(SpikeNetConfig::builder().alpha(1.5).build().is_err());
        assert!(SpikeNetConfig::builder().alpha(-0.1).build().is_err());
    }

    #[test]
    fn weight_range_too_large_rejected() {
        assert!(SpikeNetConfig::builder()
            .weight_init_range(2.0)
            .build()
            .is_err());
    }

    #[test]
    fn all_builder_methods_chain() {
        let config = SpikeNetConfig::builder()
            .n_hidden(16)
            .n_outputs(3)
            .alpha(0.85)
            .kappa(0.95)
            .kappa_out(0.8)
            .learning_rate(0.01)
            .v_thr(0.3)
            .gamma(0.5)
            .spike_threshold(0.1)
            .seed(999)
            .weight_init_range(0.2)
            .build()
            .unwrap();

        assert_eq!(config.n_hidden, 16);
        assert_eq!(config.n_outputs, 3);
        assert!((config.alpha - 0.85).abs() < 1e-10);
        assert!((config.kappa - 0.95).abs() < 1e-10);
        assert!((config.kappa_out - 0.8).abs() < 1e-10);
        assert!((config.learning_rate - 0.01).abs() < 1e-10);
        assert!((config.v_thr - 0.3).abs() < 1e-10);
        assert!((config.gamma - 0.5).abs() < 1e-10);
        assert!((config.spike_threshold - 0.1).abs() < 1e-10);
        assert_eq!(config.seed, 999);
        assert!((config.weight_init_range - 0.2).abs() < 1e-10);
    }
}
