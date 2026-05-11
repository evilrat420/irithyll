//! Shared plasticity configuration for Dohare 2024 continual backpropagation.
//!
//! [`PlasticityConfig`] holds the hyperparameters for the neuron-regeneration
//! strategy that prevents loss of plasticity in continual learning
//! (Dohare et al., 2024 — "Loss of Plasticity in Deep Continual Learning",
//! *Nature* 632, 768–774).
//!
//! When plasticity maintenance is enabled on a model, dead or low-utility
//! neurons are periodically detected and re-initialized. This prevents the
//! "loss of plasticity" failure mode where network weights saturate and new
//! information can no longer be encoded.
//!
//! # Usage
//!
//! ```
//! use irithyll::common::PlasticityConfig;
//!
//! let plasticity = PlasticityConfig::builder()
//!     .regen_fraction(0.01)
//!     .regen_interval(500)
//!     .utility_alpha(0.99)
//!     .build()
//!     .unwrap();
//! ```

use crate::error::ConfigError;

// ---------------------------------------------------------------------------
// PlasticityConfig
// ---------------------------------------------------------------------------

/// Configuration for the neuron regeneration plasticity strategy.
///
/// Controls the Dohare et al. (2024) continual backpropagation algorithm that
/// prevents loss of plasticity in streaming models. When enabled on a model
/// via `.plasticity(PlasticityConfig::default())`, the strategy periodically
/// detects dead neurons by EWMA-smoothed activation energy and re-initializes
/// the bottom `regen_fraction` fraction.
///
/// # Reference
///
/// Dohare et al. (2024) "Loss of Plasticity in Deep Continual Learning",
/// *Nature* 632, 768–774.
///
/// # Defaults
///
/// | Parameter | Default | Source |
/// |-----------|---------|--------|
/// | `regen_fraction` | 0.01 | Dohare 2024, Table 1 (Cα=0.01) |
/// | `regen_interval` | 500 | Dohare 2024, typical evaluation cadence |
/// | `utility_alpha` | 0.99 | Dohare 2024, EWMA smoothing constant |
#[derive(Debug, Clone)]
pub struct PlasticityConfig {
    /// Fraction of lowest-utility neurons replaced per cycle (default: 0.01).
    ///
    /// Corresponds to Cα in Dohare et al. 2024, Table 1. Values in [0.001, 0.05]
    /// are typical; 0.01 is the paper-recommended default for the general case.
    pub regen_fraction: f64,

    /// Number of training steps between regeneration cycles (default: 500).
    ///
    /// Lower values react faster to plasticity loss but add overhead.
    /// Dohare 2024 uses 500–1000 steps between cycles in practice.
    pub regen_interval: u64,

    /// EWMA decay for utility tracking (default: 0.99).
    ///
    /// Higher values smooth over more history (slower response).
    /// The paper uses 0.99 as the standard smoothing constant.
    /// Must be in [0, 1).
    pub utility_alpha: f64,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            regen_fraction: 0.01,
            regen_interval: 500,
            utility_alpha: 0.99,
        }
    }
}

impl std::fmt::Display for PlasticityConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PlasticityConfig(regen_fraction={}, regen_interval={}, utility_alpha={})",
            self.regen_fraction, self.regen_interval, self.utility_alpha
        )
    }
}

// ---------------------------------------------------------------------------
// PlasticityConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for [`PlasticityConfig`] with validation.
///
/// # Example
///
/// ```
/// use irithyll::common::PlasticityConfig;
///
/// let config = PlasticityConfig::builder()
///     .regen_fraction(0.02)
///     .regen_interval(1000)
///     .utility_alpha(0.995)
///     .build()
///     .unwrap();
///
/// assert!((config.regen_fraction - 0.02).abs() < 1e-12);
/// ```
pub struct PlasticityConfigBuilder {
    config: PlasticityConfig,
}

impl PlasticityConfig {
    /// Create a new builder initialized to paper-recommended defaults.
    pub fn builder() -> PlasticityConfigBuilder {
        PlasticityConfigBuilder {
            config: PlasticityConfig::default(),
        }
    }
}

impl PlasticityConfigBuilder {
    /// Set the regeneration fraction (default: 0.01).
    ///
    /// Fraction of lowest-utility neurons replaced per cycle. Must be in (0, 1].
    /// The value 0.01 corresponds to Cα in Dohare et al. (2024), Table 1.
    pub fn regen_fraction(mut self, f: f64) -> Self {
        self.config.regen_fraction = f;
        self
    }

    /// Set the regeneration interval in steps (default: 500).
    ///
    /// Number of training steps between successive regeneration cycles.
    /// Must be >= 1.
    pub fn regen_interval(mut self, n: u64) -> Self {
        self.config.regen_interval = n;
        self
    }

    /// Set the EWMA decay for utility tracking (default: 0.99).
    ///
    /// Must be in [0, 1). Higher values smooth over more history.
    pub fn utility_alpha(mut self, a: f64) -> Self {
        self.config.utility_alpha = a;
        self
    }

    /// Build the config, validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if:
    /// - `regen_fraction` is not in (0, 1]
    /// - `regen_interval` is 0
    /// - `utility_alpha` is not in [0, 1)
    pub fn build(self) -> Result<PlasticityConfig, ConfigError> {
        let c = &self.config;

        if c.regen_fraction <= 0.0 || c.regen_fraction > 1.0 {
            return Err(ConfigError::out_of_range(
                "regen_fraction",
                "must be in (0, 1]",
                c.regen_fraction,
            ));
        }
        if c.regen_interval == 0 {
            return Err(ConfigError::out_of_range(
                "regen_interval",
                "must be >= 1",
                c.regen_interval,
            ));
        }
        if c.utility_alpha < 0.0 || c.utility_alpha >= 1.0 {
            return Err(ConfigError::out_of_range(
                "utility_alpha",
                "must be in [0, 1)",
                c.utility_alpha,
            ));
        }

        Ok(self.config)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_match_paper() {
        let c = PlasticityConfig::default();
        assert!(
            (c.regen_fraction - 0.01).abs() < 1e-12,
            "default regen_fraction should be 0.01 (Dohare 2024 Cα)"
        );
        assert_eq!(
            c.regen_interval, 500,
            "default regen_interval should be 500"
        );
        assert!(
            (c.utility_alpha - 0.99).abs() < 1e-12,
            "default utility_alpha should be 0.99"
        );
    }

    #[test]
    fn builder_round_trips() {
        let c = PlasticityConfig::builder()
            .regen_fraction(0.02)
            .regen_interval(1000)
            .utility_alpha(0.995)
            .build()
            .unwrap();
        assert!((c.regen_fraction - 0.02).abs() < 1e-12);
        assert_eq!(c.regen_interval, 1000);
        assert!((c.utility_alpha - 0.995).abs() < 1e-12);
    }

    #[test]
    fn rejects_zero_regen_fraction() {
        assert!(
            PlasticityConfig::builder()
                .regen_fraction(0.0)
                .build()
                .is_err(),
            "regen_fraction=0 must be rejected"
        );
    }

    #[test]
    fn rejects_negative_regen_fraction() {
        assert!(
            PlasticityConfig::builder()
                .regen_fraction(-0.1)
                .build()
                .is_err(),
            "regen_fraction<0 must be rejected"
        );
    }

    #[test]
    fn accepts_regen_fraction_one() {
        assert!(
            PlasticityConfig::builder()
                .regen_fraction(1.0)
                .build()
                .is_ok(),
            "regen_fraction=1.0 should be valid (replace all)"
        );
    }

    #[test]
    fn rejects_zero_regen_interval() {
        assert!(
            PlasticityConfig::builder()
                .regen_interval(0)
                .build()
                .is_err(),
            "regen_interval=0 must be rejected"
        );
    }

    #[test]
    fn rejects_utility_alpha_one() {
        assert!(
            PlasticityConfig::builder()
                .utility_alpha(1.0)
                .build()
                .is_err(),
            "utility_alpha=1.0 must be rejected (denominator issue)"
        );
    }

    #[test]
    fn rejects_negative_utility_alpha() {
        assert!(
            PlasticityConfig::builder()
                .utility_alpha(-0.1)
                .build()
                .is_err(),
            "negative utility_alpha must be rejected"
        );
    }

    #[test]
    fn accepts_utility_alpha_zero() {
        assert!(
            PlasticityConfig::builder()
                .utility_alpha(0.0)
                .build()
                .is_ok(),
            "utility_alpha=0.0 should be valid (no smoothing)"
        );
    }

    #[test]
    fn display_contains_fields() {
        let c = PlasticityConfig::default();
        let s = format!("{c}");
        assert!(
            s.contains("regen_fraction="),
            "display should contain regen_fraction"
        );
        assert!(
            s.contains("regen_interval="),
            "display should contain regen_interval"
        );
        assert!(
            s.contains("utility_alpha="),
            "display should contain utility_alpha"
        );
    }

    #[test]
    fn clone_is_deep() {
        let orig = PlasticityConfig::builder()
            .regen_fraction(0.05)
            .build()
            .unwrap();
        let cloned = orig.clone();
        assert!((cloned.regen_fraction - 0.05).abs() < 1e-12);
    }
}
