//! Streaming decision trees with Hoeffding-bound split decisions.
//!
//! Trees grow incrementally: each sample updates leaf histograms, and splits
//! are triggered only when the Hoeffding bound guarantees the best split is
//! statistically superior to alternatives.

pub mod node;
pub mod split;
pub mod hoeffding;
pub mod builder;
pub mod predict;

/// A streaming decision tree that trains incrementally.
pub trait StreamingTree: Send + Sync {
    /// Train on a single gradient/hessian pair at the given feature vector.
    fn train_one(&mut self, features: &[f64], gradient: f64, hessian: f64);

    /// Predict the leaf value for a feature vector.
    fn predict(&self, features: &[f64]) -> f64;

    /// Current number of leaf nodes.
    fn n_leaves(&self) -> usize;

    /// Total samples seen since creation.
    fn n_samples_seen(&self) -> u64;

    /// Reset to initial state (single root leaf).
    fn reset(&mut self);
}
