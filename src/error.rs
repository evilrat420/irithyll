//! Error types for Irithyll.
//!
//! Core error types ([`ConfigError`], [`IrithyllError`]) are defined in
//! `irithyll-core` and re-exported here. This module extends `IrithyllError`
//! with std-only variants (`Serialization`, `ChannelClosed`).

// Re-export core error types.
pub use irithyll_core::error::{ConfigError, FormatError};

use thiserror::Error;

/// Top-level error type for the Irithyll crate.
///
/// Core variants (`InvalidConfig`, `InsufficientData`, `DimensionMismatch`,
/// `NotTrained`) mirror `irithyll_core::IrithyllError`. Std-only variants
/// (`Serialization`, `ChannelClosed`) are added here.
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
