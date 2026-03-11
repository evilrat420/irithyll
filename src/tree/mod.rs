//! Streaming decision trees with Hoeffding-bound split decisions.
//!
//! Trees grow incrementally: each sample updates leaf histograms, and splits
//! are triggered only when the Hoeffding bound guarantees the best split is
//! statistically superior to alternatives.

pub mod builder;
pub mod hoeffding;
pub mod node;
pub mod predict;
pub mod split;

#[cfg(feature = "neural-leaves")]
pub mod leaf_model;

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

    /// Accumulated split gains per feature for importance tracking.
    ///
    /// Returns an empty slice if the tree hasn't seen any features yet
    /// or the implementation doesn't track split gains.
    fn split_gains(&self) -> &[f64] {
        &[]
    }

    /// Predict the leaf value and its variance for confidence estimation.
    ///
    /// Returns `(leaf_value, variance)` where variance = 1 / (H_sum + lambda).
    /// A smaller variance indicates higher confidence in the leaf prediction.
    ///
    /// Default implementation returns infinite variance (no confidence info).
    fn predict_with_variance(&self, features: &[f64]) -> (f64, f64) {
        (self.predict(features), f64::INFINITY)
    }
}
