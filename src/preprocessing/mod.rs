//! Streaming preprocessing utilities for feature transformation.
//!
//! These transformers process features incrementally, maintaining running
//! statistics that update with each sample -- no batch recomputation needed.
//!
//! # Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`normalizer`] | Welford-based online standardization (zero-mean, unit-variance) |
//! | [`feature_selector`] | EWMA importance tracking with dynamic feature masking |
//! | [`ccipca`] | Candid Covariance-free Incremental PCA -- streaming dimensionality reduction |
//! | [`feature_hasher`] | Feature hashing (hashing trick) for fixed-size dimensionality reduction |
//! | [`min_max`] | Streaming min-max scaler for feature normalization to a target range |
//! | [`one_hot`] | Streaming one-hot encoder with online category discovery |
//! | [`target_encoder`] | Streaming target encoder with Bayesian smoothing |
//! | [`polynomial`] | Polynomial and interaction feature generation |
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
pub mod feature_hasher;
pub mod feature_selector;
pub mod min_max;
pub mod normalizer;
pub mod one_hot;
pub mod polynomial;
pub mod target_encoder;

pub use ccipca::CCIPCA;
pub use feature_hasher::FeatureHasher;
pub use feature_selector::OnlineFeatureSelector;
pub use min_max::MinMaxScaler;
pub use normalizer::IncrementalNormalizer;
pub use one_hot::OneHotEncoder;
pub use polynomial::PolynomialFeatures;
pub use target_encoder::TargetEncoder;

// Re-export the StreamingPreprocessor trait (defined in pipeline module).
pub use crate::pipeline::StreamingPreprocessor;
