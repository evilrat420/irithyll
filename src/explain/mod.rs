//! TreeSHAP explanations for streaming gradient boosted trees.
//!
//! Implements the path-dependent TreeSHAP algorithm (Lundberg et al., 2020)
//! for computing per-feature Shapley value contributions. Each feature's SHAP
//! value indicates how much it pushed the prediction away from the expected value.
//!
//! # Invariant
//!
//! For any feature vector `x` and model `f`:
//!
//! ```text
//! base_value + sum(shap_values) ≈ f.predict(x)
//! ```

pub mod streaming;
pub mod treeshap;

pub use streaming::StreamingShap;
pub use treeshap::ShapValues;
