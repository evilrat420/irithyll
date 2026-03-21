//! Histogram binning and accumulation for streaming tree construction.
//!
//! Core implementation lives in `irithyll-core` and is re-exported here.

pub use irithyll_core::histogram::*;

// Re-export sub-modules for path compatibility
pub use irithyll_core::histogram::bins;
pub use irithyll_core::histogram::categorical;
pub use irithyll_core::histogram::quantile;
pub use irithyll_core::histogram::uniform;

#[cfg(feature = "simd")]
pub use irithyll_core::histogram::simd;

#[cfg(feature = "kmeans-binning")]
pub use irithyll_core::histogram::kmeans;
