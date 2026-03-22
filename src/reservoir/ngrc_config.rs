//! Configuration and builder for Next Generation Reservoir Computing (NG-RC).
//!
//! [`NGRCConfig`] holds all hyperparameters for the NG-RC model. Use the
//! builder pattern via [`NGRCConfig::builder`] for ergonomic construction
//! with validation.

use crate::error::{ConfigError, IrithyllError, Result};

/// Configuration for [`NextGenRC`](super::NextGenRC).
///
/// # Hyperparameters
///
/// - `k` -- number of time delays. Controls how far back the model looks.
/// - `s` -- skip between delays. `s = 1` means consecutive observations,
///   `s = 2` means every other observation, etc.
/// - `degree` -- polynomial degree for nonlinear features (must be >= 2).
/// - `forgetting_factor` -- RLS lambda for the readout layer.
/// - `delta` -- initial P matrix scale for RLS.
/// - `include_bias` -- whether to prepend a constant 1.0 to the feature vector.
///
/// The model requires `k * s` observations before it becomes "warm" and can
/// produce predictions.
///
/// # Examples
///
/// ```
/// use irithyll::reservoir::NGRCConfig;
///
/// let config = NGRCConfig::builder()
///     .k(3)
///     .s(2)
///     .degree(2)
///     .forgetting_factor(0.999)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.k, 3);
/// assert_eq!(config.s, 2);
/// ```
#[derive(Debug, Clone)]
pub struct NGRCConfig {
    /// Number of delays (default: 2).
    pub k: usize,
    /// Skip between delays (default: 1).
    pub s: usize,
    /// Polynomial degree for nonlinear features (default: 2).
    pub degree: usize,
    /// RLS forgetting factor lambda in (0, 1] (default: 0.999).
    pub forgetting_factor: f64,
    /// RLS initial P matrix scale (default: 100.0).
    pub delta: f64,
    /// Whether to include a constant 1.0 bias feature (default: true).
    pub include_bias: bool,
}

impl Default for NGRCConfig {
    fn default() -> Self {
        Self {
            k: 2,
            s: 1,
            degree: 2,
            forgetting_factor: 0.999,
            delta: 100.0,
            include_bias: true,
        }
    }
}

impl NGRCConfig {
    /// Start building an `NGRCConfig` with default values.
    pub fn builder() -> NGRCConfigBuilder {
        NGRCConfigBuilder::new()
    }
}

/// Builder for [`NGRCConfig`] with validation on [`build()`](Self::build).
#[derive(Debug, Clone)]
pub struct NGRCConfigBuilder {
    config: NGRCConfig,
}

impl NGRCConfigBuilder {
    /// Create a new builder with default hyperparameters.
    pub fn new() -> Self {
        Self {
            config: NGRCConfig::default(),
        }
    }

    /// Set the number of delays.
    pub fn k(mut self, k: usize) -> Self {
        self.config.k = k;
        self
    }

    /// Set the skip between delays.
    pub fn s(mut self, s: usize) -> Self {
        self.config.s = s;
        self
    }

    /// Set the polynomial degree.
    pub fn degree(mut self, degree: usize) -> Self {
        self.config.degree = degree;
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

    /// Set whether to include a constant bias feature.
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.config.include_bias = include_bias;
        self
    }

    /// Validate and build the configuration.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if any parameter is out of range.
    pub fn build(self) -> Result<NGRCConfig> {
        let c = &self.config;

        if c.k < 1 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "k",
                "must be >= 1",
                c.k,
            )));
        }
        if c.s < 1 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "s",
                "must be >= 1",
                c.s,
            )));
        }
        if c.degree < 2 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "degree",
                "must be >= 2",
                c.degree,
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

impl Default for NGRCConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_builds() {
        let config = NGRCConfig::builder().build().unwrap();
        assert_eq!(config.k, 2);
        assert_eq!(config.s, 1);
        assert_eq!(config.degree, 2);
        assert!((config.forgetting_factor - 0.999).abs() < 1e-12);
        assert!((config.delta - 100.0).abs() < 1e-12);
        assert!(config.include_bias);
    }

    #[test]
    fn custom_config_builds() {
        let config = NGRCConfig::builder()
            .k(5)
            .s(3)
            .degree(3)
            .forgetting_factor(0.99)
            .delta(50.0)
            .include_bias(false)
            .build()
            .unwrap();

        assert_eq!(config.k, 5);
        assert_eq!(config.s, 3);
        assert_eq!(config.degree, 3);
        assert!((config.forgetting_factor - 0.99).abs() < 1e-12);
        assert!((config.delta - 50.0).abs() < 1e-12);
        assert!(!config.include_bias);
    }

    #[test]
    fn k_zero_fails() {
        let result = NGRCConfig::builder().k(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn s_zero_fails() {
        let result = NGRCConfig::builder().s(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn degree_one_fails() {
        let result = NGRCConfig::builder().degree(1).build();
        assert!(result.is_err());
    }

    #[test]
    fn forgetting_factor_zero_fails() {
        let result = NGRCConfig::builder().forgetting_factor(0.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn forgetting_factor_above_one_fails() {
        let result = NGRCConfig::builder().forgetting_factor(1.01).build();
        assert!(result.is_err());
    }

    #[test]
    fn delta_zero_fails() {
        let result = NGRCConfig::builder().delta(0.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn delta_negative_fails() {
        let result = NGRCConfig::builder().delta(-1.0).build();
        assert!(result.is_err());
    }
}
