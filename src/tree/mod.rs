//! Streaming decision trees with Hoeffding-bound split decisions.
//!
//! Core implementation lives in `irithyll-core` and is re-exported here.

pub use irithyll_core::tree::*;

// Re-export sub-modules for path compatibility
pub use irithyll_core::tree::builder;
pub use irithyll_core::tree::hoeffding;
pub use irithyll_core::tree::hoeffding_classifier;
pub use irithyll_core::tree::leaf_model;
pub use irithyll_core::tree::node;
pub use irithyll_core::tree::predict;
pub use irithyll_core::tree::split;
