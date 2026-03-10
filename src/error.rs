//! Error types for Irithyll.

use thiserror::Error;

/// Top-level error type for the Irithyll crate.
#[derive(Debug, Error)]
pub enum IrithyllError {
    /// Configuration validation failed.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

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
