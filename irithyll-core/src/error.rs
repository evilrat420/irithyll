//! Error types for irithyll-core.
//!
//! No `thiserror`, no mandatory `alloc` — just `core::fmt::Display` impls.
//!
//! - [`FormatError`] — packed binary validation errors (always available, `Copy`).
//! - [`ConfigError`] — configuration validation errors (requires `alloc` for `String` fields).

#[cfg(feature = "alloc")]
use alloc::string::String;

// ---------------------------------------------------------------------------
// FormatError — packed binary validation (no_std, no alloc)
// ---------------------------------------------------------------------------

/// Errors that can occur when parsing or validating a packed ensemble binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatError {
    /// Magic bytes do not match `"IRIT"` (`0x54495249` LE).
    BadMagic,
    /// Format version is not supported by this build.
    UnsupportedVersion,
    /// Input buffer is too short to contain the declared structures.
    Truncated,
    /// Input buffer pointer is not aligned to 4 bytes.
    Unaligned,
    /// A node's child index points outside the node array bounds.
    InvalidNodeIndex,
    /// A node references a feature index >= `n_features`.
    InvalidFeatureIndex,
    /// A tree entry's byte offset is not aligned to `size_of::<PackedNode>()`.
    MisalignedTreeOffset,
}

impl core::fmt::Display for FormatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FormatError::BadMagic => write!(f, "bad magic: expected \"IRIT\""),
            FormatError::UnsupportedVersion => write!(f, "unsupported format version"),
            FormatError::Truncated => write!(f, "buffer truncated"),
            FormatError::Unaligned => write!(f, "buffer not 4-byte aligned"),
            FormatError::InvalidNodeIndex => write!(f, "node child index out of bounds"),
            FormatError::InvalidFeatureIndex => write!(f, "feature index exceeds n_features"),
            FormatError::MisalignedTreeOffset => {
                write!(f, "tree offset not aligned to node size")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigError — configuration validation (requires alloc for String fields)
// ---------------------------------------------------------------------------

/// Structured error for configuration validation failures.
///
/// Instead of opaque strings, each variant carries the parameter name and
/// constraint so callers can programmatically inspect what went wrong.
///
/// Requires the `alloc` feature for `String` fields.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// A parameter value is outside its valid range.
    ///
    /// # Examples
    ///
    /// ```text
    /// n_steps must be > 0 (got 0)
    /// learning_rate must be in (0, 1] (got 1.5)
    /// ```
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
    Invalid {
        /// The parameter name.
        param: &'static str,
        /// Why the value is invalid.
        reason: String,
    },
}

#[cfg(feature = "alloc")]
impl ConfigError {
    /// Convenience for the common "value out of range" case.
    pub fn out_of_range(
        param: &'static str,
        constraint: &'static str,
        value: impl core::fmt::Display,
    ) -> Self {
        use alloc::format;
        ConfigError::OutOfRange {
            param,
            constraint,
            value: format!("{}", value),
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

#[cfg(feature = "alloc")]
impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ConfigError::OutOfRange {
                param,
                constraint,
                value,
            } => write!(f, "{} {} (got {})", param, constraint, value),
            ConfigError::Invalid { param, reason } => write!(f, "{} {}", param, reason),
        }
    }
}

// std::error::Error impls (requires std feature)
#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}

// ---------------------------------------------------------------------------
// IrithyllError — top-level error enum (requires alloc)
// ---------------------------------------------------------------------------

/// Top-level error type for the irithyll crate.
///
/// Requires the `alloc` feature for `String` fields.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub enum IrithyllError {
    /// Configuration validation failed.
    InvalidConfig(ConfigError),
    /// Not enough data to perform the requested operation.
    InsufficientData(String),
    /// Feature dimension mismatch between sample and model.
    DimensionMismatch {
        /// Expected number of features.
        expected: usize,
        /// Actual number of features received.
        got: usize,
    },
    /// Model has not been trained yet.
    NotTrained,
}

#[cfg(feature = "alloc")]
impl core::fmt::Display for IrithyllError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IrithyllError::InvalidConfig(e) => write!(f, "invalid configuration: {}", e),
            IrithyllError::InsufficientData(msg) => write!(f, "insufficient data: {}", msg),
            IrithyllError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            IrithyllError::NotTrained => write!(f, "model not trained"),
        }
    }
}

#[cfg(feature = "alloc")]
impl From<ConfigError> for IrithyllError {
    fn from(e: ConfigError) -> Self {
        IrithyllError::InvalidConfig(e)
    }
}

/// Result type using [`IrithyllError`].
#[cfg(feature = "alloc")]
pub type Result<T> = core::result::Result<T, IrithyllError>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "alloc")]
    use alloc::string::ToString;

    #[test]
    fn format_error_display() {
        assert_eq!(
            FormatError::BadMagic.to_string(),
            "bad magic: expected \"IRIT\""
        );
        assert_eq!(FormatError::Truncated.to_string(), "buffer truncated");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn config_error_out_of_range_display() {
        let e = ConfigError::out_of_range("n_steps", "must be > 0", 0);
        assert_eq!(e.to_string(), "n_steps must be > 0 (got 0)");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn config_error_invalid_display() {
        let e = ConfigError::invalid(
            "split_reeval_interval",
            "must be >= grace_period (200), got 50",
        );
        assert!(e.to_string().contains("split_reeval_interval"));
        assert!(e.to_string().contains("must be >= grace_period"));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn irithyll_error_from_config_error() {
        let ce = ConfigError::out_of_range("learning_rate", "must be in (0, 1]", 1.5);
        let ie: IrithyllError = ce.into();
        let msg = ie.to_string();
        assert!(msg.contains("invalid configuration"));
        assert!(msg.contains("learning_rate"));
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn irithyll_error_dimension_mismatch() {
        let e = IrithyllError::DimensionMismatch {
            expected: 10,
            got: 5,
        };
        assert_eq!(e.to_string(), "dimension mismatch: expected 10, got 5");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn irithyll_error_not_trained() {
        assert_eq!(IrithyllError::NotTrained.to_string(), "model not trained");
    }
}
