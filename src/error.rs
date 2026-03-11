//! Error types for Irithyll.

use thiserror::Error;

// ---------------------------------------------------------------------------
// ConfigError — structured sub-enum for configuration validation
// ---------------------------------------------------------------------------

/// Structured error for configuration validation failures.
///
/// Instead of opaque strings, each variant carries the parameter name and
/// constraint so callers can programmatically inspect what went wrong.
#[derive(Debug, Clone, Error)]
pub enum ConfigError {
    /// A parameter value is outside its valid range.
    ///
    /// # Examples
    ///
    /// ```text
    /// n_steps must be > 0 (got 0)
    /// learning_rate must be in (0, 1] (got 1.5)
    /// ```
    #[error("{param} {constraint} (got {value})")]
    OutOfRange {
        /// The parameter name (e.g. `"n_steps"`, `"drift_detector.Adwin.delta"`).
        param: &'static str,
        /// The constraint that was violated (e.g. `"must be > 0"`).
        constraint: &'static str,
        /// The actual value that was provided.
        value: String,
    },

    /// A parameter is invalid for a structural reason, typically involving
    /// a relationship between two parameters.
    ///
    /// # Examples
    ///
    /// ```text
    /// split_reeval_interval must be >= grace_period (200), got 50
    /// drift_detector.Ddm.drift_level must be > warning_level (3.0), got 2.0
    /// ```
    #[error("{param} {reason}")]
    Invalid {
        /// The parameter name.
        param: &'static str,
        /// Why the value is invalid.
        reason: String,
    },
}

impl ConfigError {
    /// Convenience for the common "value out of range" case.
    pub fn out_of_range(
        param: &'static str,
        constraint: &'static str,
        value: impl ToString,
    ) -> Self {
        ConfigError::OutOfRange {
            param,
            constraint,
            value: value.to_string(),
        }
    }

    /// Convenience for the "invalid relationship" case.
    pub fn invalid(param: &'static str, reason: impl Into<String>) -> Self {
        ConfigError::Invalid {
            param,
            reason: reason.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// IrithyllError — top-level error enum
// ---------------------------------------------------------------------------

/// Top-level error type for the Irithyll crate.
#[derive(Debug, Error)]
pub enum IrithyllError {
    /// Configuration validation failed.
    #[error("invalid configuration: {0}")]
    InvalidConfig(#[from] ConfigError),

    /// Not enough data to perform the requested operation.
    #[error("insufficient data: {0}")]
    InsufficientData(String),

    /// Feature dimension mismatch between sample and model.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Model has not been trained yet.
    #[error("model not trained")]
    NotTrained,

    /// Serialization or deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Async channel closed unexpectedly.
    #[error("channel closed")]
    ChannelClosed,
}

pub type Result<T> = std::result::Result<T, IrithyllError>;
