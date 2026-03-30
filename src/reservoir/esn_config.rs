//! Configuration and builder for Echo State Networks.
//!
//! [`ESNConfig`] holds all hyperparameters for the ESN model. Use the
//! builder pattern via [`ESNConfig::builder`] for ergonomic construction
//! with validation.

use crate::error::{ConfigError, IrithyllError, Result};

/// Configuration for [`EchoStateNetwork`](super::EchoStateNetwork).
///
/// # Hyperparameters
///
/// - `n_reservoir` -- number of reservoir neurons (default: 100).
/// - `spectral_radius` -- controls the magnitude of cycle weights (default: 0.9).
/// - `leak_rate` -- leaky integration rate in (0, 1] (default: 0.3).
/// - `input_scaling` -- input weight magnitude (default: 1.0).
/// - `bias_scaling` -- bias magnitude (default: 0.0, no bias).
/// - `forgetting_factor` -- RLS lambda for the readout (default: 0.998).
/// - `delta` -- RLS initial P matrix scale (default: 100.0).
/// - `seed` -- PRNG seed for reservoir initialization (default: 42).
/// - `warmup` -- number of samples to drive the reservoir before training (default: 50).
/// - `passthrough_input` -- whether to concatenate input features to reservoir state
///   for the readout (default: true).
///
/// # Examples
///
/// ```
/// use irithyll::reservoir::ESNConfig;
///
/// let config = ESNConfig::builder()
///     .n_reservoir(200)
///     .spectral_radius(0.95)
///     .leak_rate(0.5)
///     .seed(123)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.n_reservoir, 200);
/// ```
#[derive(Debug, Clone)]
pub struct ESNConfig {
    /// Number of reservoir neurons (default: 100).
    pub n_reservoir: usize,
    /// Spectral radius of the cycle weight matrix (default: 0.9).
    pub spectral_radius: f64,
    /// Leaky integration rate in (0, 1] (default: 0.3).
    pub leak_rate: f64,
    /// Input weight scaling (default: 1.0).
    pub input_scaling: f64,
    /// Bias scaling (default: 0.0).
    pub bias_scaling: f64,
    /// RLS forgetting factor lambda in (0, 1] (default: 0.998).
    pub forgetting_factor: f64,
    /// RLS initial P matrix scale (default: 100.0).
    pub delta: f64,
    /// PRNG seed for reproducibility (default: 42).
    pub seed: u64,
    /// Number of warmup samples before RLS training begins (default: 50).
    pub warmup: usize,
    /// Whether to concatenate raw input to reservoir state for readout (default: true).
    pub passthrough_input: bool,
}

impl Default for ESNConfig {
    fn default() -> Self {
        Self {
            n_reservoir: 100,
            spectral_radius: 0.9,
            leak_rate: 0.3,
            input_scaling: 1.0,
            bias_scaling: 0.0,
            forgetting_factor: 0.998,
            delta: 100.0,
            seed: 42,
            warmup: 50,
            passthrough_input: true,
        }
    }
}

impl ESNConfig {
    /// Start building an `ESNConfig` with default values.
    pub fn builder() -> ESNConfigBuilder {
        ESNConfigBuilder::new()
    }
}

/// Builder for [`ESNConfig`] with validation on [`build()`](Self::build).
#[derive(Debug, Clone)]
pub struct ESNConfigBuilder {
    config: ESNConfig,
}

impl ESNConfigBuilder {
    /// Create a new builder with default hyperparameters.
    pub fn new() -> Self {
        Self {
            config: ESNConfig::default(),
        }
    }

    /// Set the number of reservoir neurons.
    pub fn n_reservoir(mut self, n: usize) -> Self {
        self.config.n_reservoir = n;
        self
    }

    /// Set the spectral radius.
    pub fn spectral_radius(mut self, sr: f64) -> Self {
        self.config.spectral_radius = sr;
        self
    }

    /// Set the leak rate.
    pub fn leak_rate(mut self, lr: f64) -> Self {
        self.config.leak_rate = lr;
        self
    }

    /// Set the input weight scaling.
    pub fn input_scaling(mut self, is: f64) -> Self {
        self.config.input_scaling = is;
        self
    }

    /// Set the bias scaling.
    pub fn bias_scaling(mut self, bs: f64) -> Self {
        self.config.bias_scaling = bs;
        self
    }

    /// Set the RLS forgetting factor.
    pub fn forgetting_factor(mut self, ff: f64) -> Self {
        self.config.forgetting_factor = ff;
        self
    }

    /// Set the RLS initial P matrix scale.
    pub fn delta(mut self, delta: f64) -> Self {
        self.config.delta = delta;
        self
    }

    /// Set the PRNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Set the warmup period.
    pub fn warmup(mut self, warmup: usize) -> Self {
        self.config.warmup = warmup;
        self
    }

    /// Set whether to pass raw input through to the readout layer.
    pub fn passthrough_input(mut self, pt: bool) -> Self {
        self.config.passthrough_input = pt;
        self
    }

    /// Validate and build the configuration.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if any parameter is out of range.
    pub fn build(self) -> Result<ESNConfig> {
        let c = &self.config;

        if c.n_reservoir < 1 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "n_reservoir",
                "must be >= 1",
                c.n_reservoir,
            )));
        }
        if c.spectral_radius <= 0.0 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "spectral_radius",
                "must be > 0",
                c.spectral_radius,
            )));
        }
        if c.leak_rate <= 0.0 || c.leak_rate > 1.0 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "leak_rate",
                "must be in (0, 1]",
                c.leak_rate,
            )));
        }
        if c.input_scaling < 0.0 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "input_scaling",
                "must be >= 0",
                c.input_scaling,
            )));
        }
        if c.bias_scaling < 0.0 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "bias_scaling",
                "must be >= 0",
                c.bias_scaling,
            )));
        }
        if c.forgetting_factor <= 0.0 || c.forgetting_factor > 1.0 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "forgetting_factor",
                "must be in (0, 1]",
                c.forgetting_factor,
            )));
        }
        if c.delta <= 0.0 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "delta",
                "must be > 0",
                c.delta,
            )));
        }

        Ok(self.config)
    }
}

impl Default for ESNConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_builds() {
        let config = ESNConfig::builder().build().unwrap();
        assert_eq!(config.n_reservoir, 100);
        assert!((config.spectral_radius - 0.9).abs() < 1e-12);
        assert!((config.leak_rate - 0.3).abs() < 1e-12);
        assert!((config.input_scaling - 1.0).abs() < 1e-12);
        assert!((config.bias_scaling).abs() < 1e-12);
        assert!((config.forgetting_factor - 0.998).abs() < 1e-12);
        assert!((config.delta - 100.0).abs() < 1e-12);
        assert_eq!(config.seed, 42);
        assert_eq!(config.warmup, 50);
        assert!(config.passthrough_input);
    }

    #[test]
    fn custom_config_builds() {
        let config = ESNConfig::builder()
            .n_reservoir(200)
            .spectral_radius(0.95)
            .leak_rate(0.5)
            .input_scaling(0.5)
            .bias_scaling(0.1)
            .forgetting_factor(0.99)
            .delta(50.0)
            .seed(123)
            .warmup(100)
            .passthrough_input(false)
            .build()
            .unwrap();

        assert_eq!(config.n_reservoir, 200);
        assert!((config.spectral_radius - 0.95).abs() < 1e-12);
        assert!((config.leak_rate - 0.5).abs() < 1e-12);
        assert!((config.input_scaling - 0.5).abs() < 1e-12);
        assert!((config.bias_scaling - 0.1).abs() < 1e-12);
        assert!((config.forgetting_factor - 0.99).abs() < 1e-12);
        assert!((config.delta - 50.0).abs() < 1e-12);
        assert_eq!(config.seed, 123);
        assert_eq!(config.warmup, 100);
        assert!(!config.passthrough_input);
    }

    #[test]
    fn zero_reservoir_fails() {
        let result = ESNConfig::builder().n_reservoir(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn negative_spectral_radius_fails() {
        let result = ESNConfig::builder().spectral_radius(-0.1).build();
        assert!(result.is_err());
    }

    #[test]
    fn zero_spectral_radius_fails() {
        let result = ESNConfig::builder().spectral_radius(0.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn leak_rate_zero_fails() {
        let result = ESNConfig::builder().leak_rate(0.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn leak_rate_above_one_fails() {
        let result = ESNConfig::builder().leak_rate(1.01).build();
        assert!(result.is_err());
    }

    #[test]
    fn negative_input_scaling_fails() {
        let result = ESNConfig::builder().input_scaling(-0.1).build();
        assert!(result.is_err());
    }

    #[test]
    fn forgetting_factor_zero_fails() {
        let result = ESNConfig::builder().forgetting_factor(0.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn delta_zero_fails() {
        let result = ESNConfig::builder().delta(0.0).build();
        assert!(result.is_err());
    }
}
