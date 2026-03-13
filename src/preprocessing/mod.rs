//! Streaming preprocessing utilities for feature transformation.
//!
//! These transformers process features incrementally, maintaining running
//! statistics that update with each sample — no batch recomputation needed.
//!
//! # Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`normalizer`] | Welford-based online standardization (zero-mean, unit-variance) |
//! | [`feature_selector`] | EWMA importance tracking with dynamic feature masking |
//! | [`ccipca`] | Candid Covariance-free Incremental PCA — streaming dimensionality reduction |
//!
//! # Example
//!
//! ```
//! use irithyll::preprocessing::{IncrementalNormalizer, OnlineFeatureSelector};
//!
//! let mut norm = IncrementalNormalizer::new();
//! let standardized = norm.update_and_transform(&[100.0, 0.5, -3.0]);
//!
//! let mut selector = OnlineFeatureSelector::new(3, 0.5, 0.1, 10);
//! selector.update_importances(&[0.9, 0.1, 0.8]);
//! let masked = selector.mask_features(&standardized);
//! ```

pub mod ccipca;
pub mod feature_selector;
pub mod normalizer;

pub use ccipca::CCIPCA;
pub use feature_selector::OnlineFeatureSelector;
pub use normalizer::IncrementalNormalizer;

// Re-export the StreamingPreprocessor trait (defined in pipeline module).
pub use crate::pipeline::StreamingPreprocessor;
