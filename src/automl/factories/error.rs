//! Typed error returned by [`ModelFactory::create`].
//!
//! Separates the three reasons a factory can legitimately fail so that the
//! AutoML racing layer can handle each class correctly (skip + warn) instead
//! of crashing the whole process when a bandit-sampled config is invalid.

use crate::error::IrithyllError;
use irithyll_core::error::ConfigError;

/// Error returned by [`crate::automl::ModelFactory::create`].
///
/// The AutoML bandit may sample hyperparameter combinations that are
/// structurally invalid (e.g. `n_heads > d_model`, non-divisible group sizes,
/// or out-of-range floats that slip past the config-space bounds). Rather than
/// panicking, factories return this error and the racing layer logs a warning
/// and skips the offending arm.
///
/// The enum is `#[non_exhaustive]` so that future variants (e.g. a
/// `ResourceLimit` variant once budget accounting lands) do not become
/// breaking changes for external users implementing the trait.
#[derive(Debug)]
#[non_exhaustive]
pub enum FactoryError {
    /// A config builder's validation step failed.
    ///
    /// This is the most common case: a sampled float or integer was in the
    /// `ConfigSpace` bounds but violates a model-level constraint (e.g.
    /// `forgetting_factor` must be strictly < 1.0, `n_steps` must be > 0).
    InvalidConfig(ConfigError),

    /// Two or more sampled parameters are jointly invalid.
    ///
    /// Used when the conflict cannot be expressed as a single-field
    /// `ConfigError` — for example, `n_heads > d_model` (neither field is
    /// individually out of range, but their combination is illegal).
    IncompatibleArm {
        /// Human-readable explanation of why this arm is incompatible.
        reason: String,
    },
}

impl core::fmt::Display for FactoryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FactoryError::InvalidConfig(e) => write!(f, "invalid config: {}", e),
            FactoryError::IncompatibleArm { reason } => {
                write!(f, "incompatible arm: {}", reason)
            }
        }
    }
}

impl std::error::Error for FactoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FactoryError::InvalidConfig(e) => Some(e),
            FactoryError::IncompatibleArm { .. } => None,
        }
    }
}

impl From<ConfigError> for FactoryError {
    fn from(e: ConfigError) -> Self {
        FactoryError::InvalidConfig(e)
    }
}

impl From<IrithyllError> for FactoryError {
    fn from(e: IrithyllError) -> Self {
        match e {
            IrithyllError::InvalidConfig(ce) => FactoryError::InvalidConfig(ce),
            other => FactoryError::IncompatibleArm {
                reason: other.to_string(),
            },
        }
    }
}
