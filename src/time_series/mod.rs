//! Time series models for streaming forecasting.
//!
//! This module provides online time series algorithms that process observations
//! one at a time, maintaining O(m) state where m is the seasonal period.

pub mod decomposition;
pub mod holt_winters;
pub mod snarimax;

pub use decomposition::{
    DecomposedPoint, DecompositionConfig, DecompositionConfigBuilder, StreamingDecomposition,
};
pub use holt_winters::{HoltWinters, HoltWintersConfig, HoltWintersConfigBuilder, Seasonality};
pub use snarimax::{SNARIMAXCoefficients, SNARIMAXConfig, SNARIMAX};
