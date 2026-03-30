//! Configuration and builder for [`StreamingMamba`](super::StreamingMamba).
//!
//! [`MambaConfig`] holds all hyperparameters for the streaming Mamba model.
//! Use [`MambaConfigBuilder`] (via [`MambaConfig::builder()`]) for validated
//! construction with sensible defaults.
//!
//! # Defaults
//!
//! | Parameter | Default | Description |
//! |-----------|---------|-------------|
//! | `d_in` | (required) | Input feature dimension |
//! | `n_state` | 16 | Hidden state dimension per channel |
//! | `forgetting_factor` | 0.998 | RLS exponential forgetting |
//! | `delta_rls` | 100.0 | Initial P matrix diagonal for RLS |
//! | `seed` | 42 | PRNG seed for SSM weight initialization |
//! | `warmup` | 10 | Samples before RLS predictions are trusted |

use std::fmt;

use crate::error::ConfigError;

/// Configuration for a [`StreamingMamba`](super::StreamingMamba) model.
///
/// Create via the builder pattern:
///
/// ```
/// use irithyll::ssm::MambaConfig;
///
/// let config = MambaConfig::builder()
///     .d_in(8)
///     .n_state(16)
///     .forgetting_factor(0.998)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MambaConfig {
    /// Input/output feature dimension (required, >= 1).
    pub d_in: usize,
    /// Hidden state dimension per channel (default: 16, >= 1).
    pub n_state: usize,
    /// RLS forgetting factor (default: 0.998, in (0, 1]).
    pub forgetting_factor: f64,
    /// Initial P matrix diagonal for RLS (default: 100.0, > 0).
    pub delta_rls: f64,
    /// Random seed for SSM weight initialization (default: 42).
    pub seed: u64,
    /// Number of warmup samples before predictions are trusted (default: 10, >= 0).
    pub warmup: usize,
}

impl MambaConfig {
    /// Create a new builder with default values.
    ///
    /// Only `d_in` is required; all other parameters have sensible defaults.
    pub fn builder() -> MambaConfigBuilder {
        MambaConfigBuilder::default()
    }
}

impl fmt::Display for MambaConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MambaConfig(d_in={}, n_state={}, ff={}, delta={}, seed={}, warmup={})",
            self.d_in, self.n_state, self.forgetting_factor, self.delta_rls, self.seed, self.warmup
        )
    }
}

/// Builder for [`MambaConfig`] with validation.
///
/// # Required Parameters
///
/// - `d_in` -- must be set before calling `build()`
///
/// # Example
///
/// ```
/// use irithyll::ssm::MambaConfig;
///
/// let config = MambaConfig::builder()
///     .d_in(4)
///     .n_state(32)
///     .seed(123)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.d_in, 4);
/// assert_eq!(config.n_state, 32);
/// ```
#[derive(Debug)]
pub struct MambaConfigBuilder {
    d_in: Option<usize>,
    n_state: usize,
    forgetting_factor: f64,
    delta_rls: f64,
    seed: u64,
    warmup: usize,
}

impl Default for MambaConfigBuilder {
    fn default() -> Self {
        Self {
            d_in: None,
            n_state: 16,
            forgetting_factor: 0.998,
            delta_rls: 100.0,
            seed: 42,
            warmup: 10,
        }
    }
}

impl MambaConfigBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input feature dimension (required, >= 1).
    pub fn d_in(mut self, d_in: usize) -> Self {
        self.d_in = Some(d_in);
        self
    }

    /// Set the hidden state dimension per channel (default: 16, >= 1).
    pub fn n_state(mut self, n_state: usize) -> Self {
        self.n_state = n_state;
        self
    }

    /// Set the RLS forgetting factor (default: 0.998, must be in (0, 1]).
    pub fn forgetting_factor(mut self, ff: f64) -> Self {
        self.forgetting_factor = ff;
        self
    }

    /// Set the initial P matrix diagonal for RLS (default: 100.0, must be > 0).
    pub fn delta_rls(mut self, delta: f64) -> Self {
        self.delta_rls = delta;
        self
    }

    /// Set the random seed for SSM weight initialization (default: 42).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the warmup period in samples (default: 10).
    pub fn warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    /// Build the config, validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if:
    /// - `d_in` was not set or is 0
    /// - `n_state` is 0
    /// - `forgetting_factor` is not in (0, 1]
    /// - `delta_rls` is not positive
    pub fn build(self) -> Result<MambaConfig, ConfigError> {
        let d_in = self.d_in.ok_or_else(|| {
            ConfigError::invalid("d_in", "d_in must be set (input feature dimension)")
        })?;
        if d_in < 1 {
            return Err(ConfigError::out_of_range("d_in", "must be >= 1", d_in));
        }
        if self.n_state < 1 {
            return Err(ConfigError::out_of_range(
                "n_state",
                "must be >= 1",
                self.n_state,
            ));
        }
        if self.forgetting_factor <= 0.0 || self.forgetting_factor > 1.0 {
            return Err(ConfigError::out_of_range(
                "forgetting_factor",
                "must be in (0, 1]",
                self.forgetting_factor,
            ));
        }
        if self.delta_rls <= 0.0 {
            return Err(ConfigError::out_of_range(
                "delta_rls",
                "must be > 0",
                self.delta_rls,
            ));
        }

        Ok(MambaConfig {
            d_in,
            n_state: self.n_state,
            forgetting_factor: self.forgetting_factor,
            delta_rls: self.delta_rls,
            seed: self.seed,
            warmup: self.warmup,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let config = MambaConfig::builder().d_in(4).build().unwrap();
        assert_eq!(config.d_in, 4);
        assert_eq!(config.n_state, 16);
        assert!((config.forgetting_factor - 0.998).abs() < 1e-12);
        assert!((config.delta_rls - 100.0).abs() < 1e-12);
        assert_eq!(config.seed, 42);
        assert_eq!(config.warmup, 10);
    }

    #[test]
    fn builder_custom_values() {
        let config = MambaConfig::builder()
            .d_in(8)
            .n_state(32)
            .forgetting_factor(0.99)
            .delta_rls(50.0)
            .seed(123)
            .warmup(5)
            .build()
            .unwrap();
        assert_eq!(config.d_in, 8);
        assert_eq!(config.n_state, 32);
        assert!((config.forgetting_factor - 0.99).abs() < 1e-12);
        assert!((config.delta_rls - 50.0).abs() < 1e-12);
        assert_eq!(config.seed, 123);
        assert_eq!(config.warmup, 5);
    }

    #[test]
    fn builder_missing_d_in() {
        let result = MambaConfig::builder().build();
        assert!(result.is_err(), "should fail without d_in");
    }

    #[test]
    fn builder_invalid_n_state() {
        let result = MambaConfig::builder().d_in(4).n_state(0).build();
        assert!(result.is_err(), "n_state=0 should be invalid");
    }

    #[test]
    fn builder_invalid_forgetting_factor_zero() {
        let result = MambaConfig::builder()
            .d_in(4)
            .forgetting_factor(0.0)
            .build();
        assert!(result.is_err(), "ff=0 should be invalid");
    }

    #[test]
    fn builder_invalid_forgetting_factor_negative() {
        let result = MambaConfig::builder()
            .d_in(4)
            .forgetting_factor(-0.5)
            .build();
        assert!(result.is_err(), "ff=-0.5 should be invalid");
    }

    #[test]
    fn builder_invalid_forgetting_factor_over_one() {
        let result = MambaConfig::builder()
            .d_in(4)
            .forgetting_factor(1.01)
            .build();
        assert!(result.is_err(), "ff=1.01 should be invalid");
    }

    #[test]
    fn builder_forgetting_factor_one_valid() {
        let config = MambaConfig::builder()
            .d_in(4)
            .forgetting_factor(1.0)
            .build()
            .unwrap();
        assert!((config.forgetting_factor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn builder_invalid_delta_rls() {
        let result = MambaConfig::builder().d_in(4).delta_rls(0.0).build();
        assert!(result.is_err(), "delta_rls=0 should be invalid");
        let result = MambaConfig::builder().d_in(4).delta_rls(-1.0).build();
        assert!(result.is_err(), "delta_rls=-1 should be invalid");
    }

    #[test]
    fn display_format() {
        let config = MambaConfig::builder().d_in(4).build().unwrap();
        let s = format!("{}", config);
        assert!(s.contains("d_in=4"), "display should contain d_in");
        assert!(s.contains("n_state=16"), "display should contain n_state");
    }

    #[test]
    fn config_clone() {
        let config = MambaConfig::builder().d_in(4).seed(99).build().unwrap();
        let cloned = config.clone();
        assert_eq!(cloned.d_in, config.d_in);
        assert_eq!(cloned.seed, config.seed);
    }
}
