//! Configuration and builder for [`StreamingAttentionModel`](super::StreamingAttentionModel).
//!
//! [`StreamingAttentionConfig`] holds all hyperparameters for the streaming
//! attention model. Use [`StreamingAttentionConfigBuilder`] (via
//! [`StreamingAttentionConfig::builder()`]) for validated construction with
//! sensible defaults.
//!
//! # Defaults
//!
//! | Parameter | Default | Description |
//! |-----------|---------|-------------|
//! | `d_model` | (required) | Model/input feature dimension |
//! | `n_heads` | 4 | Number of attention heads |
//! | `d_key` | d_model / n_heads | Key dimension per head |
//! | `d_value` | d_model / n_heads | Value dimension per head |
//! | `mode` | GLA | Attention variant |
//! | `forgetting_factor` | 0.998 | RLS exponential forgetting |
//! | `delta` | 100.0 | Initial P matrix diagonal for RLS |
//! | `seed` | 42 | PRNG seed for attention weight initialization |
//! | `warmup` | 10 | Samples before RLS predictions are trusted |

use std::fmt;

use irithyll_core::attention::AttentionMode;
use irithyll_core::error::ConfigError;

/// Configuration for a [`StreamingAttentionModel`](super::StreamingAttentionModel).
///
/// Create via the builder pattern:
///
/// ```ignore
/// use irithyll::attention::{StreamingAttentionConfig, AttentionMode};
///
/// let config = StreamingAttentionConfig::builder()
///     .d_model(8)
///     .n_heads(2)
///     .mode(AttentionMode::GLA)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct StreamingAttentionConfig {
    /// Model/input feature dimension (required, >= 1).
    pub d_model: usize,
    /// Number of attention heads (default: 4, >= 1).
    pub n_heads: usize,
    /// Key dimension per head (default: d_model / n_heads).
    pub d_key: usize,
    /// Value dimension per head (default: d_model / n_heads).
    pub d_value: usize,
    /// Attention variant (default: GLA).
    pub mode: AttentionMode,
    /// RLS forgetting factor (default: 0.998, in (0, 1]).
    pub forgetting_factor: f64,
    /// Initial P matrix diagonal for RLS (default: 100.0, > 0).
    pub delta: f64,
    /// Random seed for attention weight initialization (default: 42).
    pub seed: u64,
    /// Number of warmup samples before predictions are trusted (default: 10).
    pub warmup: usize,
}

impl StreamingAttentionConfig {
    /// Create a new builder with default values.
    ///
    /// Only `d_model` is required; all other parameters have sensible defaults.
    pub fn builder() -> StreamingAttentionConfigBuilder {
        StreamingAttentionConfigBuilder::default()
    }
}

impl fmt::Display for StreamingAttentionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StreamingAttentionConfig(d_model={}, n_heads={}, d_key={}, d_value={}, mode={:?}, \
             ff={}, delta={}, seed={}, warmup={})",
            self.d_model,
            self.n_heads,
            self.d_key,
            self.d_value,
            self.mode,
            self.forgetting_factor,
            self.delta,
            self.seed,
            self.warmup
        )
    }
}

/// Builder for [`StreamingAttentionConfig`] with validation.
///
/// # Required Parameters
///
/// - `d_model` -- must be set before calling `build()`
///
/// # Example
///
/// ```ignore
/// use irithyll::attention::{StreamingAttentionConfig, AttentionMode};
///
/// let config = StreamingAttentionConfig::builder()
///     .d_model(16)
///     .n_heads(4)
///     .mode(AttentionMode::GatedDeltaNet)
///     .forgetting_factor(0.99)
///     .build()
///     .unwrap();
///
/// assert_eq!(config.d_model, 16);
/// assert_eq!(config.n_heads, 4);
/// assert_eq!(config.d_key, 4); // 16 / 4
/// ```
#[derive(Debug)]
pub struct StreamingAttentionConfigBuilder {
    d_model: Option<usize>,
    n_heads: usize,
    d_key: Option<usize>,
    d_value: Option<usize>,
    mode: AttentionMode,
    forgetting_factor: f64,
    delta: f64,
    seed: u64,
    warmup: usize,
}

impl Default for StreamingAttentionConfigBuilder {
    fn default() -> Self {
        Self {
            d_model: None,
            n_heads: 4,
            d_key: None,
            d_value: None,
            mode: AttentionMode::GLA,
            forgetting_factor: 0.998,
            delta: 100.0,
            seed: 42,
            warmup: 10,
        }
    }
}

impl StreamingAttentionConfigBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model/input feature dimension (required, >= 1).
    pub fn d_model(mut self, d_model: usize) -> Self {
        self.d_model = Some(d_model);
        self
    }

    /// Set the number of attention heads (default: 4, >= 1).
    pub fn n_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    /// Set the key dimension per head explicitly.
    ///
    /// If not set, defaults to `d_model / n_heads`.
    pub fn d_key(mut self, d_key: usize) -> Self {
        self.d_key = Some(d_key);
        self
    }

    /// Set the value dimension per head explicitly.
    ///
    /// If not set, defaults to `d_model / n_heads`.
    pub fn d_value(mut self, d_value: usize) -> Self {
        self.d_value = Some(d_value);
        self
    }

    /// Set the attention variant (default: GLA).
    pub fn mode(mut self, mode: AttentionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the RLS forgetting factor (default: 0.998, must be in (0, 1]).
    pub fn forgetting_factor(mut self, ff: f64) -> Self {
        self.forgetting_factor = ff;
        self
    }

    /// Set the initial P matrix diagonal for RLS (default: 100.0, must be > 0).
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Set the random seed for attention weight initialization (default: 42).
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
    /// - `d_model` was not set or is 0
    /// - `n_heads` is 0
    /// - `d_model` is not divisible by `n_heads` (when d_key/d_value not explicitly set)
    /// - `forgetting_factor` is not in (0, 1]
    /// - `delta` is not positive
    pub fn build(self) -> Result<StreamingAttentionConfig, ConfigError> {
        let d_model = self.d_model.ok_or_else(|| {
            ConfigError::invalid("d_model", "d_model must be set (model feature dimension)")
        })?;
        if d_model < 1 {
            return Err(ConfigError::out_of_range(
                "d_model",
                "must be >= 1",
                d_model,
            ));
        }
        if self.n_heads < 1 {
            return Err(ConfigError::out_of_range(
                "n_heads",
                "must be >= 1",
                self.n_heads,
            ));
        }

        // Resolve d_key and d_value
        let d_key = match self.d_key {
            Some(dk) => {
                if dk < 1 {
                    return Err(ConfigError::out_of_range("d_key", "must be >= 1", dk));
                }
                dk
            }
            None => {
                if d_model % self.n_heads != 0 {
                    return Err(ConfigError::invalid(
                        "d_model",
                        format!(
                            "d_model ({}) must be divisible by n_heads ({}) when d_key is not explicitly set",
                            d_model, self.n_heads
                        ),
                    ));
                }
                d_model / self.n_heads
            }
        };

        let d_value = match self.d_value {
            Some(dv) => {
                if dv < 1 {
                    return Err(ConfigError::out_of_range("d_value", "must be >= 1", dv));
                }
                dv
            }
            None => {
                if d_model % self.n_heads != 0 {
                    return Err(ConfigError::invalid(
                        "d_model",
                        format!(
                            "d_model ({}) must be divisible by n_heads ({}) when d_value is not explicitly set",
                            d_model, self.n_heads
                        ),
                    ));
                }
                d_model / self.n_heads
            }
        };

        if self.forgetting_factor <= 0.0 || self.forgetting_factor > 1.0 {
            return Err(ConfigError::out_of_range(
                "forgetting_factor",
                "must be in (0, 1]",
                self.forgetting_factor,
            ));
        }
        if self.delta <= 0.0 {
            return Err(ConfigError::out_of_range(
                "delta",
                "must be > 0",
                self.delta,
            ));
        }

        Ok(StreamingAttentionConfig {
            d_model,
            n_heads: self.n_heads,
            d_key,
            d_value,
            mode: self.mode,
            forgetting_factor: self.forgetting_factor,
            delta: self.delta,
            seed: self.seed,
            warmup: self.warmup,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults_with_d_model() {
        let config = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(2)
            .build()
            .unwrap();
        assert_eq!(config.d_model, 8);
        assert_eq!(config.n_heads, 2);
        assert_eq!(config.d_key, 4);
        assert_eq!(config.d_value, 4);
        assert!((config.forgetting_factor - 0.998).abs() < 1e-12);
        assert!((config.delta - 100.0).abs() < 1e-12);
        assert_eq!(config.seed, 42);
        assert_eq!(config.warmup, 10);
    }

    #[test]
    fn builder_custom_values() {
        let config = StreamingAttentionConfig::builder()
            .d_model(16)
            .n_heads(4)
            .mode(AttentionMode::GatedDeltaNet)
            .forgetting_factor(0.99)
            .delta(50.0)
            .seed(123)
            .warmup(5)
            .build()
            .unwrap();
        assert_eq!(config.d_model, 16);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.d_key, 4);
        assert_eq!(config.d_value, 4);
        assert!((config.forgetting_factor - 0.99).abs() < 1e-12);
        assert!((config.delta - 50.0).abs() < 1e-12);
        assert_eq!(config.seed, 123);
        assert_eq!(config.warmup, 5);
    }

    #[test]
    fn builder_explicit_d_key_d_value() {
        let config = StreamingAttentionConfig::builder()
            .d_model(10)
            .n_heads(3)
            .d_key(5)
            .d_value(7)
            .build()
            .unwrap();
        assert_eq!(config.d_key, 5);
        assert_eq!(config.d_value, 7);
        // d_model not divisible by n_heads but explicit dims are fine
        assert_eq!(config.n_heads, 3);
    }

    #[test]
    fn builder_missing_d_model() {
        let result = StreamingAttentionConfig::builder().build();
        assert!(result.is_err(), "should fail without d_model");
    }

    #[test]
    fn builder_d_model_not_divisible_by_n_heads() {
        let result = StreamingAttentionConfig::builder()
            .d_model(10)
            .n_heads(3)
            .build();
        assert!(
            result.is_err(),
            "d_model=10 not divisible by n_heads=3 should fail"
        );
    }

    #[test]
    fn builder_invalid_n_heads_zero() {
        let result = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(0)
            .build();
        assert!(result.is_err(), "n_heads=0 should be invalid");
    }

    #[test]
    fn builder_invalid_forgetting_factor_zero() {
        let result = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(2)
            .forgetting_factor(0.0)
            .build();
        assert!(result.is_err(), "ff=0 should be invalid");
    }

    #[test]
    fn builder_invalid_forgetting_factor_over_one() {
        let result = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(2)
            .forgetting_factor(1.01)
            .build();
        assert!(result.is_err(), "ff=1.01 should be invalid");
    }

    #[test]
    fn builder_forgetting_factor_one_valid() {
        let config = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(2)
            .forgetting_factor(1.0)
            .build()
            .unwrap();
        assert!((config.forgetting_factor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn builder_invalid_delta_zero() {
        let result = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(2)
            .delta(0.0)
            .build();
        assert!(result.is_err(), "delta=0 should be invalid");
    }

    #[test]
    fn builder_invalid_delta_negative() {
        let result = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(2)
            .delta(-1.0)
            .build();
        assert!(result.is_err(), "delta=-1 should be invalid");
    }

    #[test]
    fn display_format() {
        let config = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(2)
            .build()
            .unwrap();
        let s = format!("{}", config);
        assert!(s.contains("d_model=8"), "display should contain d_model");
        assert!(s.contains("n_heads=2"), "display should contain n_heads");
    }

    #[test]
    fn config_clone() {
        let config = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(2)
            .seed(99)
            .build()
            .unwrap();
        let cloned = config.clone();
        assert_eq!(cloned.d_model, config.d_model);
        assert_eq!(cloned.seed, config.seed);
    }

    #[test]
    fn builder_invalid_d_key_zero() {
        let result = StreamingAttentionConfig::builder()
            .d_model(8)
            .n_heads(2)
            .d_key(0)
            .build();
        assert!(result.is_err(), "d_key=0 should be invalid");
    }
}
