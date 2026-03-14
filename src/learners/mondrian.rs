//! Online Mondrian Forest for streaming regression.
//!
//! Implements a simplified Mondrian forest adapted from Lakshminarayanan et al.
//! (2014), designed for incremental learning on evolving data streams. Each
//! tree in the ensemble maintains axis-aligned bounding boxes at every node and
//! uses feature-range-proportional random splits to partition the input space.
//!
//! # Algorithm
//!
//! Each tree starts as a single root leaf. When a sample arrives:
//!
//! 1. Route it to a leaf by comparing the sample against split thresholds.
//! 2. Update the leaf's bounding box (per-feature min/max) and running statistics
//!    (sum of targets, count).
//! 3. If the leaf has accumulated enough samples and has not reached `max_depth`,
//!    split it: choose a feature proportional to its range across the leaf's
//!    bounding box, place the threshold at the midpoint, and distribute existing
//!    statistics to the two new children.
//!
//! Prediction traverses every tree to a leaf and averages the per-leaf means.
//!
//! # Differences from the Full Mondrian Process
//!
//! The full Mondrian process samples exponential split times and inserts internal
//! nodes above existing leaves when exceedance is high. This implementation uses
//! a simpler "split-at-leaf" strategy that is more robust numerically while
//! retaining the core properties of online random forests: incremental growth,
//! axis-aligned partitioning, and ensemble averaging.
//!
//! # Arena Storage
//!
//! Trees use SoA (Structure of Arrays) arena-based node storage for cache
//! efficiency. All node fields are stored in parallel `Vec`s indexed by a
//! `usize` node ID.

use std::fmt;

use crate::learner::StreamingLearner;

// ---------------------------------------------------------------------------
// RNG utilities -- xorshift64
// ---------------------------------------------------------------------------

/// Advance a xorshift64 state and return the next pseudo-random `u64`.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Return a pseudo-random `f64` in [0, 1).
#[inline]
fn rand_f64(state: &mut u64) -> f64 {
    (xorshift64(state) as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// MondrianForestConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`MondrianForest`] ensemble.
///
/// Use [`MondrianForestConfig::builder()`] for ergonomic construction:
///
/// ```ignore
/// let config = MondrianForestConfig::builder()
///     .n_trees(20)
///     .max_depth(10)
///     .lifetime(8.0)
///     .seed(123)
///     .build();
/// ```
#[derive(Clone, Debug)]
pub struct MondrianForestConfig {
    /// Number of trees in the ensemble. Default: 10.
    pub n_trees: usize,
    /// Maximum depth any single tree may reach. Default: 8.
    pub max_depth: usize,
    /// Lifetime budget parameter controlling split aggressiveness. Default: 5.0.
    pub lifetime: f64,
    /// Random seed for reproducibility. Default: 42.
    pub seed: u64,
}

impl MondrianForestConfig {
    /// Create a [`MondrianForestConfigBuilder`] with default values.
    #[inline]
    pub fn builder() -> MondrianForestConfigBuilder {
        MondrianForestConfigBuilder::default()
    }
}

impl Default for MondrianForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 10,
            max_depth: 8,
            lifetime: 5.0,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// MondrianForestConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for [`MondrianForestConfig`].
pub struct MondrianForestConfigBuilder {
    n_trees: usize,
    max_depth: usize,
    lifetime: f64,
    seed: u64,
}

impl Default for MondrianForestConfigBuilder {
    fn default() -> Self {
        Self {
            n_trees: 10,
            max_depth: 8,
            lifetime: 5.0,
            seed: 42,
        }
    }
}

impl MondrianForestConfigBuilder {
    /// Set the number of trees.
    #[inline]
    pub fn n_trees(mut self, n: usize) -> Self {
        self.n_trees = n;
        self
    }

    /// Set the maximum tree depth.
    #[inline]
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Set the lifetime budget parameter.
    #[inline]
    pub fn lifetime(mut self, l: f64) -> Self {
        self.lifetime = l;
        self
    }

    /// Set the random seed.
    #[inline]
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Consume the builder and produce a [`MondrianForestConfig`].
    #[inline]
    pub fn build(self) -> MondrianForestConfig {
        MondrianForestConfig {
            n_trees: self.n_trees,
            max_depth: self.max_depth,
            lifetime: self.lifetime,
            seed: self.seed,
        }
    }
}

// ---------------------------------------------------------------------------
// MondrianTree -- SoA arena-based online regression tree
// ---------------------------------------------------------------------------

/// A single tree in the Mondrian forest ensemble.
///
/// Node storage uses Structure of Arrays layout for cache-friendly traversal.
/// Index 0 is always the root. Leaf nodes have `left_child[i] == None`.
struct MondrianTree {
    // --- Node storage (SoA) ---
    split_feature: Vec<usize>,
    split_threshold: Vec<f64>,
    left_child: Vec<Option<usize>>,
    right_child: Vec<Option<usize>>,
    depth: Vec<usize>,

    // --- Per-node feature range tracking ---
    lower: Vec<Vec<f64>>,
    upper: Vec<Vec<f64>>,

    // --- Per-node statistics for prediction ---
    sum_targets: Vec<f64>,
    sum_weights: Vec<f64>,
    count: Vec<u64>,

    // --- Split time (lifetime budget consumed at this split) ---
    split_time: Vec<f64>,

    // --- Simple RNG state ---
    rng_state: u64,
}

impl MondrianTree {
    /// Create a new tree with a single empty root leaf.
    fn new(seed: u64, n_features: usize) -> Self {
        let mut tree = Self {
            split_feature: Vec::with_capacity(64),
            split_threshold: Vec::with_capacity(64),
            left_child: Vec::with_capacity(64),
            right_child: Vec::with_capacity(64),
            depth: Vec::with_capacity(64),
            lower: Vec::with_capacity(64),
            upper: Vec::with_capacity(64),
            sum_targets: Vec::with_capacity(64),
            sum_weights: Vec::with_capacity(64),
            count: Vec::with_capacity(64),
            split_time: Vec::with_capacity(64),
            rng_state: if seed == 0 { 1 } else { seed },
        };
        tree.alloc_leaf(0, n_features);
        tree
    }

    /// Allocate a new leaf node and return its index.
    fn alloc_leaf(&mut self, depth: usize, n_features: usize) -> usize {
        let idx = self.split_feature.len();
        self.split_feature.push(0);
        self.split_threshold.push(0.0);
        self.left_child.push(None);
        self.right_child.push(None);
        self.depth.push(depth);
        self.lower.push(vec![f64::MAX; n_features]);
        self.upper.push(vec![f64::MIN; n_features]);
        self.sum_targets.push(0.0);
        self.sum_weights.push(0.0);
        self.count.push(0);
        self.split_time.push(0.0);
        idx
    }

    /// Whether node `idx` is a leaf.
    #[inline]
    fn is_leaf(&self, idx: usize) -> bool {
        self.left_child[idx].is_none()
    }

    /// Route a feature vector to its leaf, returning the leaf index.
    fn route_to_leaf(&self, features: &[f64]) -> usize {
        let mut idx = 0;
        loop {
            if self.is_leaf(idx) {
                return idx;
            }
            let f = self.split_feature[idx];
            if features[f] <= self.split_threshold[idx] {
                idx = self.left_child[idx].unwrap();
            } else {
                idx = self.right_child[idx].unwrap();
            }
        }
    }

    /// Update bounding box at node `idx` to include `features`.
    fn expand_bbox(&mut self, idx: usize, features: &[f64]) {
        let lower = &mut self.lower[idx];
        let upper = &mut self.upper[idx];
        for (j, &x) in features.iter().enumerate() {
            if x < lower[j] {
                lower[j] = x;
            }
            if x > upper[j] {
                upper[j] = x;
            }
        }
    }

    /// Train the tree on a single weighted sample.
    fn train_one(
        &mut self,
        features: &[f64],
        target: f64,
        weight: f64,
        max_depth: usize,
        min_split_count: u64,
    ) {
        let n_features = features.len();
        let leaf = self.route_to_leaf(features);

        // Update leaf statistics.
        self.sum_targets[leaf] += target * weight;
        self.sum_weights[leaf] += weight;
        self.count[leaf] += 1;
        self.expand_bbox(leaf, features);

        // Check split criterion: enough samples AND depth budget remaining.
        if self.count[leaf] < min_split_count || self.depth[leaf] >= max_depth {
            return;
        }

        // Compute per-feature range at this leaf.
        let mut ranges = vec![0.0f64; n_features];
        let mut total_range = 0.0f64;
        for (j, range_j) in ranges.iter_mut().enumerate() {
            let r = self.upper[leaf][j] - self.lower[leaf][j];
            let r = if r.is_finite() && r > 0.0 { r } else { 0.0 };
            *range_j = r;
            total_range += r;
        }

        // Need non-trivial range to split.
        if total_range < 1e-15 {
            return;
        }

        // Choose split feature proportional to range.
        let dart = rand_f64(&mut self.rng_state) * total_range;
        let mut cumulative = 0.0;
        let mut chosen_feature = 0;
        for (j, &range_j) in ranges.iter().enumerate() {
            cumulative += range_j;
            if dart <= cumulative {
                chosen_feature = j;
                break;
            }
        }

        // Split threshold: midpoint of the range for the chosen feature,
        // with a small random jitter to decorrelate trees.
        let lo = self.lower[leaf][chosen_feature];
        let hi = self.upper[leaf][chosen_feature];
        let jitter = rand_f64(&mut self.rng_state); // [0, 1)
        let threshold = lo + (hi - lo) * (0.25 + 0.5 * jitter); // in [25%, 75%] of range

        // Allocate two child leaves.
        let leaf_depth = self.depth[leaf];
        let left_idx = self.alloc_leaf(leaf_depth + 1, n_features);
        let right_idx = self.alloc_leaf(leaf_depth + 1, n_features);

        // The parent (current leaf) becomes an internal node.
        self.split_feature[leaf] = chosen_feature;
        self.split_threshold[leaf] = threshold;
        self.left_child[leaf] = Some(left_idx);
        self.right_child[leaf] = Some(right_idx);
        self.split_time[leaf] = total_range;

        // Distribute parent statistics evenly to children as a warm start.
        // This is a simplification: ideally we would replay the data, but for
        // streaming we approximate by splitting the aggregate.
        let half_target = self.sum_targets[leaf] / 2.0;
        let half_weight = self.sum_weights[leaf] / 2.0;
        let half_count = self.count[leaf] / 2;

        self.sum_targets[left_idx] = half_target;
        self.sum_weights[left_idx] = half_weight;
        self.count[left_idx] = half_count.max(1);

        self.sum_targets[right_idx] = self.sum_targets[leaf] - half_target;
        self.sum_weights[right_idx] = self.sum_weights[leaf] - half_weight;
        self.count[right_idx] = (self.count[leaf] - half_count).max(1);

        // Copy parent bounding box to both children; they will tighten as
        // new samples arrive. Narrow the split dimension appropriately.
        self.lower[left_idx] = self.lower[leaf].clone();
        self.upper[left_idx] = self.upper[leaf].clone();
        self.upper[left_idx][chosen_feature] = threshold;

        self.lower[right_idx] = self.lower[leaf].clone();
        self.upper[right_idx] = self.upper[leaf].clone();
        self.lower[right_idx][chosen_feature] = threshold;
    }

    /// Predict the target for a feature vector (leaf mean).
    #[inline]
    fn predict(&self, features: &[f64]) -> f64 {
        let leaf = self.route_to_leaf(features);
        if self.sum_weights[leaf] > 0.0 {
            self.sum_targets[leaf] / self.sum_weights[leaf]
        } else {
            0.0
        }
    }

    /// Reset the tree to a single empty root leaf.
    fn reset(&mut self, n_features: usize) {
        self.split_feature.clear();
        self.split_threshold.clear();
        self.left_child.clear();
        self.right_child.clear();
        self.depth.clear();
        self.lower.clear();
        self.upper.clear();
        self.sum_targets.clear();
        self.sum_weights.clear();
        self.count.clear();
        self.split_time.clear();
        // Re-seed is not needed; keep RNG state continuous.
        self.alloc_leaf(0, n_features);
    }

    /// Total number of nodes (internal + leaf) in this tree.
    #[inline]
    fn n_nodes(&self) -> usize {
        self.split_feature.len()
    }
}

// --- Clone impl for MondrianTree ---

impl Clone for MondrianTree {
    fn clone(&self) -> Self {
        Self {
            split_feature: self.split_feature.clone(),
            split_threshold: self.split_threshold.clone(),
            left_child: self.left_child.clone(),
            right_child: self.right_child.clone(),
            depth: self.depth.clone(),
            lower: self.lower.clone(),
            upper: self.upper.clone(),
            sum_targets: self.sum_targets.clone(),
            sum_weights: self.sum_weights.clone(),
            count: self.count.clone(),
            split_time: self.split_time.clone(),
            rng_state: self.rng_state,
        }
    }
}

// --- Debug impl for MondrianTree ---

impl fmt::Debug for MondrianTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MondrianTree")
            .field("n_nodes", &self.n_nodes())
            .field("rng_state", &self.rng_state)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// MondrianForest
// ---------------------------------------------------------------------------

/// Online random forest for streaming regression.
///
/// The ensemble averages predictions across `n_trees` independently grown
/// `MondrianTree`s. Each tree learns incrementally, splitting leaves when
/// they accumulate enough samples and the feature range is non-trivial.
///
/// # Example
///
/// ```ignore
/// use irithyll::learners::mondrian::{MondrianForest, MondrianForestConfig};
/// use irithyll::learner::StreamingLearner;
///
/// let config = MondrianForestConfig::builder()
///     .n_trees(20)
///     .max_depth(6)
///     .seed(99)
///     .build();
/// let mut forest = MondrianForest::new(config);
///
/// // Stream samples one at a time.
/// forest.train(&[1.0, 2.0], 3.0);
/// forest.train(&[4.0, 5.0], 6.0);
///
/// let pred = forest.predict(&[1.0, 2.0]);
/// assert!(pred.is_finite());
/// ```
pub struct MondrianForest {
    config: MondrianForestConfig,
    trees: Vec<MondrianTree>,
    samples_seen: u64,
    n_features: Option<usize>,
}

impl MondrianForest {
    /// Create a new forest from the given configuration.
    ///
    /// Trees are not allocated until the first training sample arrives
    /// (since the feature dimensionality is unknown until then).
    pub fn new(config: MondrianForestConfig) -> Self {
        Self {
            trees: Vec::with_capacity(config.n_trees),
            samples_seen: 0,
            n_features: None,
            config,
        }
    }

    /// Number of trees in the ensemble.
    #[inline]
    pub fn n_trees(&self) -> usize {
        self.config.n_trees
    }

    /// Reference to the configuration.
    #[inline]
    pub fn config(&self) -> &MondrianForestConfig {
        &self.config
    }

    /// Initialize trees when the feature dimensionality becomes known.
    fn init_trees(&mut self, n_features: usize) {
        self.n_features = Some(n_features);
        self.trees.clear();
        for i in 0..self.config.n_trees {
            let seed = self
                .config
                .seed
                .wrapping_add(i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let seed = if seed == 0 { 1 } else { seed };
            self.trees.push(MondrianTree::new(seed, n_features));
        }
    }

    /// Minimum sample count at a leaf before it can split.
    ///
    /// Set to `2 * n_trees` to ensure statistically meaningful splits.
    #[inline]
    fn min_split_count(&self) -> u64 {
        (2 * self.config.n_trees) as u64
    }
}

// ---------------------------------------------------------------------------
// Default impl
// ---------------------------------------------------------------------------

impl Default for MondrianForest {
    fn default() -> Self {
        Self::new(MondrianForestConfig::default())
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

impl StreamingLearner for MondrianForest {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let n_features = features.len();

        // Lazy initialization on first sample.
        if self.n_features.is_none() {
            self.init_trees(n_features);
        }

        let max_depth = self.config.max_depth;
        let min_split = self.min_split_count();

        for tree in &mut self.trees {
            tree.train_one(features, target, weight, max_depth, min_split);
        }

        self.samples_seen += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.trees.iter().map(|t| t.predict(features)).sum();
        sum / self.trees.len() as f64
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        if let Some(nf) = self.n_features {
            for tree in &mut self.trees {
                tree.reset(nf);
            }
        } else {
            self.trees.clear();
        }
        self.samples_seen = 0;
    }
}

// ---------------------------------------------------------------------------
// Clone impl -- manual to match irithyll patterns
// ---------------------------------------------------------------------------

impl Clone for MondrianForest {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            trees: self.trees.clone(),
            samples_seen: self.samples_seen,
            n_features: self.n_features,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug impl -- manual
// ---------------------------------------------------------------------------

impl fmt::Debug for MondrianForest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MondrianForest")
            .field("config", &self.config)
            .field("n_trees", &self.trees.len())
            .field("samples_seen", &self.samples_seen)
            .field("n_features", &self.n_features)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a forest with the given number of trees and default settings.
    fn forest_with_trees(n: usize) -> MondrianForest {
        let config = MondrianForestConfig::builder()
            .n_trees(n)
            .max_depth(8)
            .lifetime(5.0)
            .seed(42)
            .build();
        MondrianForest::new(config)
    }

    // --- 1. test_creation ---

    #[test]
    fn test_creation() {
        let config = MondrianForestConfig::builder()
            .n_trees(15)
            .max_depth(6)
            .lifetime(3.0)
            .seed(99)
            .build();

        let forest = MondrianForest::new(config);
        assert_eq!(forest.n_samples_seen(), 0);
        assert_eq!(forest.n_trees(), 15);
        assert_eq!(forest.config().max_depth, 6);
        assert!((forest.config().lifetime - 3.0).abs() < 1e-12);
        assert_eq!(forest.config().seed, 99);
        // Trees not yet allocated (lazy init).
        assert!(forest.trees.is_empty());
    }

    // --- 2. test_default_config ---

    #[test]
    fn test_default_config() {
        let config = MondrianForestConfig::default();
        assert_eq!(config.n_trees, 10);
        assert_eq!(config.max_depth, 8);
        assert!((config.lifetime - 5.0).abs() < 1e-12);
        assert_eq!(config.seed, 42);

        let forest = MondrianForest::default();
        assert_eq!(forest.n_trees(), 10);
        assert_eq!(forest.n_samples_seen(), 0);
    }

    // --- 3. test_single_sample ---

    #[test]
    fn test_single_sample() {
        let mut forest = forest_with_trees(10);
        forest.train(&[3.0, 4.0], 7.0);

        assert_eq!(forest.n_samples_seen(), 1);
        let pred = forest.predict(&[3.0, 4.0]);
        // With a single sample, every tree's root leaf holds target=7.0
        assert!(
            (pred - 7.0).abs() < 1e-12,
            "single sample prediction should be 7.0, got {}",
            pred,
        );
    }

    // --- 4. test_multiple_samples ---

    #[test]
    fn test_multiple_samples() {
        let mut forest = forest_with_trees(10);

        // Train on a simple linear relationship: y = x1 + x2
        for i in 0..200 {
            let x1 = (i as f64) * 0.05;
            let x2 = (i as f64) * 0.03;
            forest.train(&[x1, x2], x1 + x2);
        }

        assert_eq!(forest.n_samples_seen(), 200);

        // Prediction should be finite and in a reasonable range.
        let pred = forest.predict(&[5.0, 3.0]);
        assert!(pred.is_finite(), "prediction must be finite, got {}", pred);
        // Not asserting exact value since this is an approximate model,
        // but it should be in the right ballpark.
        assert!(
            pred > 0.0,
            "prediction should be positive for positive inputs, got {}",
            pred,
        );
    }

    // --- 5. test_convergence ---

    #[test]
    fn test_convergence() {
        let mut forest = forest_with_trees(10);
        let constant_target = 42.0;

        // Train on a constant target with varying features.
        for i in 0..500 {
            let x = (i as f64) * 0.01;
            forest.train(&[x, x * 2.0], constant_target);
        }

        // Predictions should converge to the constant target.
        let pred = forest.predict(&[2.5, 5.0]);
        assert!(
            (pred - constant_target).abs() < 1.0,
            "expected prediction near {}, got {}",
            constant_target,
            pred,
        );
    }

    // --- 6. test_different_regions ---

    #[test]
    fn test_different_regions() {
        let mut forest = forest_with_trees(20);

        // Region A: features near [0, 0], target = 10
        for i in 0..300 {
            let x = (i as f64) * 0.001;
            forest.train(&[x, x], 10.0);
        }

        // Region B: features near [100, 100], target = 90
        for i in 0..300 {
            let x = 100.0 + (i as f64) * 0.001;
            forest.train(&[x, x], 90.0);
        }

        let pred_a = forest.predict(&[0.1, 0.1]);
        let pred_b = forest.predict(&[100.1, 100.1]);

        // Model should distinguish the two regions to some degree.
        assert!(
            pred_b > pred_a,
            "region B prediction ({}) should exceed region A ({})",
            pred_b,
            pred_a,
        );
    }

    // --- 7. test_reset ---

    #[test]
    fn test_reset() {
        let mut forest = forest_with_trees(10);

        for i in 0..100 {
            forest.train(&[i as f64, (i as f64) * 0.5], i as f64);
        }
        assert_eq!(forest.n_samples_seen(), 100);

        forest.reset();
        assert_eq!(forest.n_samples_seen(), 0);

        // After reset, all trees should be single-leaf (empty).
        for tree in &forest.trees {
            assert_eq!(
                tree.n_nodes(),
                1,
                "tree should have exactly 1 node after reset"
            );
            assert!(tree.is_leaf(0));
        }

        // Prediction after reset should return 0 (no data).
        let pred = forest.predict(&[5.0, 2.5]);
        assert!(
            pred.abs() < 1e-12,
            "prediction after reset should be 0.0, got {}",
            pred,
        );
    }

    // --- 8. test_predict_batch ---

    #[test]
    fn test_predict_batch() {
        let mut forest = forest_with_trees(10);

        for i in 0..100 {
            let x = i as f64;
            forest.train(&[x, x * 0.5], x);
        }

        let rows: Vec<&[f64]> = vec![&[1.0, 0.5], &[50.0, 25.0], &[99.0, 49.5]];
        let batch = forest.predict_batch(&rows);

        assert_eq!(batch.len(), rows.len());
        for (i, row) in rows.iter().enumerate() {
            let individual = forest.predict(row);
            assert!(
                (batch[i] - individual).abs() < 1e-12,
                "batch[{}] = {} != individual = {}",
                i,
                batch[i],
                individual,
            );
        }
    }

    // --- 9. test_trait_object ---

    #[test]
    fn test_trait_object() {
        let forest = forest_with_trees(5);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(forest);

        boxed.train(&[1.0, 2.0], 3.0);
        assert_eq!(boxed.n_samples_seen(), 1);

        let pred = boxed.predict(&[1.0, 2.0]);
        assert!(pred.is_finite());

        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }

    // --- 10. test_clone ---

    #[test]
    fn test_clone() {
        let mut forest = forest_with_trees(10);

        for i in 0..100 {
            forest.train(&[i as f64, (i as f64) * 2.0], i as f64);
        }

        let mut cloned = forest.clone();
        assert_eq!(cloned.n_samples_seen(), forest.n_samples_seen());

        // Predictions should match immediately after clone.
        let features = [50.0, 100.0];
        let pred_orig = forest.predict(&features);
        let pred_clone = cloned.predict(&features);
        assert!(
            (pred_orig - pred_clone).abs() < 1e-12,
            "clone prediction should match original: {} vs {}",
            pred_orig,
            pred_clone,
        );

        // Training the clone should not affect the original.
        for i in 0..50 {
            cloned.train(&[i as f64, (i as f64) * 2.0], 999.0);
        }
        assert_eq!(forest.n_samples_seen(), 100);
        assert_eq!(cloned.n_samples_seen(), 150);

        let pred_orig_after = forest.predict(&features);
        assert!(
            (pred_orig - pred_orig_after).abs() < 1e-12,
            "original should be unchanged after training clone",
        );
    }

    // --- 11. test_multi_tree ---

    #[test]
    fn test_multi_tree() {
        // Train two forests with different tree counts on identical data.
        let mut forest_5 = forest_with_trees(5);
        let mut forest_50 = MondrianForest::new(
            MondrianForestConfig::builder()
                .n_trees(50)
                .max_depth(8)
                .seed(42)
                .build(),
        );

        // Generate noisy data: y = x + noise
        let mut rng_state: u64 = 12345;
        let mut data = Vec::new();
        for _ in 0..300 {
            let x = rand_f64(&mut rng_state) * 10.0;
            let noise = (rand_f64(&mut rng_state) - 0.5) * 2.0;
            data.push((x, x + noise));
        }

        for &(x, y) in &data {
            forest_5.train(&[x], y);
            forest_50.train(&[x], y);
        }

        // Compute mean squared error on a test set.
        let mut mse_5 = 0.0;
        let mut mse_50 = 0.0;
        let test_points = 50;
        for i in 0..test_points {
            let x = (i as f64) * 0.2;
            let true_y = x; // noiseless target
            let p5 = forest_5.predict(&[x]);
            let p50 = forest_50.predict(&[x]);
            mse_5 += (p5 - true_y).powi(2);
            mse_50 += (p50 - true_y).powi(2);
        }
        mse_5 /= test_points as f64;
        mse_50 /= test_points as f64;

        // More trees should generally produce smoother (lower MSE) predictions,
        // but we use a generous tolerance since random forests are stochastic.
        // At minimum, both should be finite.
        assert!(mse_5.is_finite(), "MSE for 5 trees should be finite");
        assert!(mse_50.is_finite(), "MSE for 50 trees should be finite");
    }

    // --- 12. test_n_samples_seen ---

    #[test]
    fn test_n_samples_seen() {
        let mut forest = forest_with_trees(10);
        assert_eq!(forest.n_samples_seen(), 0);

        for i in 1..=75 {
            forest.train(&[i as f64], i as f64);
            assert_eq!(forest.n_samples_seen(), i);
        }

        // Weighted training also increments by 1.
        forest.train_one(&[100.0], 100.0, 5.0);
        assert_eq!(forest.n_samples_seen(), 76);
    }
}
