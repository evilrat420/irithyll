//! Hoeffding Tree Classifier for streaming multi-class classification.
//!
//! [`HoeffdingTreeClassifier`] is a streaming decision tree that grows
//! incrementally using the VFDT algorithm (Domingos & Hulten, 2000). Unlike
//! the existing [`super::hoeffding::HoeffdingTree`] (which is a gradient-based
//! regressor), this classifier maintains per-leaf class distributions and
//! splits using information gain.
//!
//! # Algorithm
//!
//! For each incoming `(features, class_label)` pair:
//!
//! 1. Route the sample from root to a leaf via threshold comparisons.
//! 2. At the leaf, update class counts and per-feature histogram bins.
//! 3. Once enough samples arrive (grace period), evaluate candidate splits
//!    using information gain (entropy reduction).
//! 4. Apply the Hoeffding bound: if the gap between the best and second-best
//!    gain exceeds `epsilon = sqrt(R^2 * ln(1/delta) / (2n))`, commit the split.
//! 5. Split the leaf into two children, partitioning class counts by the
//!    chosen threshold.
//!
//! # Key differences from the regressor
//!
//! - Leaves store class counts, not gradient/hessian sums.
//! - Split criterion: Information Gain (reduction in entropy).
//! - Prediction: majority class or class probability distribution.
//! - No loss function -- direct classification.
//!
//! # Examples
//!
//! ```
//! use irithyll::tree::hoeffding_classifier::HoeffdingClassifierConfig;
//! use irithyll::tree::hoeffding_classifier::HoeffdingTreeClassifier;
//!
//! let config = HoeffdingClassifierConfig::builder()
//!     .max_depth(6)
//!     .delta(1e-5)
//!     .grace_period(50)
//!     .n_bins(16)
//!     .build()
//!     .unwrap();
//!
//! let mut tree = HoeffdingTreeClassifier::new(config);
//!
//! // Train: class 0 when x[0] < 5, class 1 otherwise
//! for i in 0..200 {
//!     let x = (i as f64) / 20.0;
//!     let class = if x < 5.0 { 0 } else { 1 };
//!     tree.train_one(&[x], class);
//! }
//!
//! // After training, the tree can predict class labels.
//! let pred = tree.predict_class(&[2.0]);
//! assert!(pred == 0 || pred == 1);
//! ```

use crate::learner::StreamingLearner;

/// Tie-breaking threshold (tau). When `epsilon < tau`, we accept the best split
/// even if the gap between best and second-best gain is small, because the
/// Hoeffding bound is already tight enough that further samples won't help.
const TAU: f64 = 0.05;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`HoeffdingTreeClassifier`].
///
/// Use [`HoeffdingClassifierConfig::builder()`] to construct with defaults
/// and override only the parameters you need.
#[derive(Debug, Clone)]
pub struct HoeffdingClassifierConfig {
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Hoeffding bound confidence parameter (1 - delta). Lower = more splits.
    pub delta: f64,
    /// Minimum samples at a leaf before considering a split.
    pub grace_period: usize,
    /// Number of histogram bins per feature for split evaluation.
    pub n_bins: usize,
    /// Number of features (0 = lazy init from first sample).
    pub n_features: usize,
    /// Maximum number of classes (0 = auto-discover).
    pub max_classes: usize,
}

/// Builder for [`HoeffdingClassifierConfig`].
///
/// All fields have sensible defaults. Call `.build()` to finalize.
#[derive(Debug, Clone)]
pub struct HoeffdingClassifierConfigBuilder {
    max_depth: usize,
    delta: f64,
    grace_period: usize,
    n_bins: usize,
    n_features: usize,
    max_classes: usize,
}

impl HoeffdingClassifierConfig {
    /// Create a builder with default parameters.
    ///
    /// Defaults:
    /// - `max_depth`: 10
    /// - `delta`: 1e-7
    /// - `grace_period`: 200
    /// - `n_bins`: 32
    /// - `n_features`: 0 (lazy init)
    /// - `max_classes`: 0 (auto-discover)
    pub fn builder() -> HoeffdingClassifierConfigBuilder {
        HoeffdingClassifierConfigBuilder {
            max_depth: 10,
            delta: 1e-7,
            grace_period: 200,
            n_bins: 32,
            n_features: 0,
            max_classes: 0,
        }
    }
}

impl HoeffdingClassifierConfigBuilder {
    /// Set the maximum tree depth.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Set the Hoeffding bound confidence parameter.
    pub fn delta(mut self, d: f64) -> Self {
        self.delta = d;
        self
    }

    /// Set the minimum samples before split evaluation.
    pub fn grace_period(mut self, g: usize) -> Self {
        self.grace_period = g;
        self
    }

    /// Set the number of histogram bins per feature.
    pub fn n_bins(mut self, b: usize) -> Self {
        self.n_bins = b;
        self
    }

    /// Set the number of features (0 for lazy init).
    pub fn n_features(mut self, f: usize) -> Self {
        self.n_features = f;
        self
    }

    /// Set the maximum number of classes (0 for auto-discover).
    pub fn max_classes(mut self, c: usize) -> Self {
        self.max_classes = c;
        self
    }

    /// Build the configuration, validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if:
    /// - `max_depth` is 0
    /// - `delta` is not in (0, 1)
    /// - `grace_period` is 0
    /// - `n_bins` is less than 2
    pub fn build(self) -> Result<HoeffdingClassifierConfig, String> {
        if self.max_depth == 0 {
            return Err("max_depth must be >= 1".to_string());
        }
        if self.delta <= 0.0 || self.delta >= 1.0 {
            return Err("delta must be in (0, 1)".to_string());
        }
        if self.grace_period == 0 {
            return Err("grace_period must be >= 1".to_string());
        }
        if self.n_bins < 2 {
            return Err("n_bins must be >= 2".to_string());
        }
        Ok(HoeffdingClassifierConfig {
            max_depth: self.max_depth,
            delta: self.delta,
            grace_period: self.grace_period,
            n_bins: self.n_bins,
            n_features: self.n_features,
            max_classes: self.max_classes,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

/// Per-leaf statistics for split evaluation.
///
/// Tracks class distributions both globally (for prediction) and per-feature
/// per-bin (for information gain computation).
#[derive(Debug, Clone)]
struct LeafStats {
    /// Per-class counts at this leaf.
    class_counts: Vec<u64>,

    /// Per-feature, per-bin, per-class counts for split evaluation.
    /// `feature_histograms[feature][bin][class] = count`
    feature_histograms: Vec<Vec<Vec<u64>>>,

    /// Per-feature bin boundaries (uniform between observed min/max).
    bin_boundaries: Vec<Vec<f64>>,

    /// Per-feature observed min/max for boundary computation.
    feature_ranges: Vec<(f64, f64)>,

    /// Total samples at this leaf.
    n_samples: u64,
}

impl LeafStats {
    /// Create fresh leaf stats for the given number of features, bins, and classes.
    fn new(n_features: usize, n_bins: usize, n_classes: usize) -> Self {
        let feature_histograms = vec![vec![vec![0u64; n_classes]; n_bins]; n_features];
        let bin_boundaries = vec![Vec::new(); n_features];
        let feature_ranges = vec![(f64::MAX, f64::MIN); n_features];

        Self {
            class_counts: vec![0u64; n_classes],
            feature_histograms,
            bin_boundaries,
            feature_ranges,
            n_samples: 0,
        }
    }

    /// Ensure class vectors are large enough to accommodate `class_id`.
    fn ensure_class_capacity(&mut self, n_classes: usize) {
        if self.class_counts.len() < n_classes {
            self.class_counts.resize(n_classes, 0);
            for feat_bins in &mut self.feature_histograms {
                for bin_counts in feat_bins.iter_mut() {
                    bin_counts.resize(n_classes, 0);
                }
            }
        }
    }
}

/// A single node in the classifier tree arena.
///
/// Internal (split) nodes have `split_feature` and `split_threshold` set.
/// Leaf nodes have `leaf_stats` populated for ongoing training.
/// All nodes maintain `class_counts` for graceful prediction even at split
/// nodes (useful when traversal ends early due to missing data).
#[derive(Debug, Clone)]
struct ClassifierNode {
    /// Feature index for the split. `None` for leaf nodes.
    split_feature: Option<usize>,

    /// Threshold value for the split. `None` for leaf nodes.
    split_threshold: Option<f64>,

    /// Index of the left child in the arena (samples where feature < threshold).
    left: Option<usize>,

    /// Index of the right child in the arena (samples where feature >= threshold).
    right: Option<usize>,

    /// Depth of this node in the tree (root = 0).
    depth: usize,

    /// Per-class sample counts at this node (accumulated during training).
    class_counts: Vec<u64>,

    /// Total samples routed through this node.
    n_samples: u64,

    /// Leaf-specific statistics for split evaluation. `None` for split nodes.
    leaf_stats: Option<LeafStats>,
}

impl ClassifierNode {
    /// Create a new leaf node at the given depth.
    fn new_leaf(depth: usize, n_features: usize, n_bins: usize, n_classes: usize) -> Self {
        Self {
            split_feature: None,
            split_threshold: None,
            left: None,
            right: None,
            depth,
            class_counts: vec![0u64; n_classes],
            n_samples: 0,
            leaf_stats: Some(LeafStats::new(n_features, n_bins, n_classes)),
        }
    }

    /// Returns `true` if this node is a leaf (has no split).
    #[inline]
    fn is_leaf(&self) -> bool {
        self.split_feature.is_none()
    }

    /// Ensure class vectors are large enough to accommodate `n_classes`.
    fn ensure_class_capacity(&mut self, n_classes: usize) {
        if self.class_counts.len() < n_classes {
            self.class_counts.resize(n_classes, 0);
        }
        if let Some(ref mut stats) = self.leaf_stats {
            stats.ensure_class_capacity(n_classes);
        }
    }
}

// ---------------------------------------------------------------------------
// HoeffdingTreeClassifier
// ---------------------------------------------------------------------------

/// A streaming decision tree classifier based on the VFDT algorithm.
///
/// Grows incrementally by maintaining per-leaf class distributions and
/// histogram bins. Splits are committed when the Hoeffding bound guarantees
/// the best information-gain split is statistically superior to the
/// runner-up.
///
/// # Thread Safety
///
/// `HoeffdingTreeClassifier` is `Send + Sync`, making it usable in async
/// and multi-threaded pipelines.
#[derive(Debug, Clone)]
pub struct HoeffdingTreeClassifier {
    config: HoeffdingClassifierConfig,
    /// Arena-based tree storage. Index 0 is always the root.
    nodes: Vec<ClassifierNode>,
    /// Number of features (lazy-initialized from first sample if config says 0).
    n_features: usize,
    /// Number of discovered classes.
    n_classes: usize,
    /// Total samples trained on.
    n_samples: u64,
}

impl HoeffdingTreeClassifier {
    /// Create a new classifier from the given configuration.
    ///
    /// If `config.n_features > 0`, the tree is immediately initialized with a
    /// root leaf. Otherwise, initialization is deferred until the first training
    /// sample arrives.
    pub fn new(config: HoeffdingClassifierConfig) -> Self {
        let n_features = config.n_features;
        let n_classes = config.max_classes;

        let mut tree = Self {
            config,
            nodes: Vec::new(),
            n_features,
            n_classes,
            n_samples: 0,
        };

        // If features are known up front, create the root immediately.
        if n_features > 0 {
            let root =
                ClassifierNode::new_leaf(0, n_features, tree.config.n_bins, n_classes.max(2));
            tree.nodes.push(root);
            if tree.n_classes == 0 {
                tree.n_classes = 2; // sensible minimum
            }
        }

        tree
    }

    /// Train on a single observation: route to leaf, update stats, maybe split.
    ///
    /// # Arguments
    ///
    /// * `features` -- feature vector for this observation.
    /// * `class` -- the class label (0-indexed).
    pub fn train_one(&mut self, features: &[f64], class: usize) {
        // Lazy initialization on first sample.
        if self.nodes.is_empty() {
            self.n_features = features.len();
            if self.n_classes == 0 {
                self.n_classes = (class + 1).max(2);
            }
            let root =
                ClassifierNode::new_leaf(0, self.n_features, self.config.n_bins, self.n_classes);
            self.nodes.push(root);
        }

        // Auto-discover classes.
        if class >= self.n_classes {
            self.n_classes = class + 1;
            for node in &mut self.nodes {
                node.ensure_class_capacity(self.n_classes);
            }
        }

        self.n_samples += 1;

        // Route to leaf.
        let leaf_idx = self.route_to_leaf(features);

        // Update leaf stats.
        self.update_leaf(leaf_idx, features, class);

        // Evaluate split if grace period met.
        let node = &self.nodes[leaf_idx];
        let n = node.n_samples;
        let gp = self.config.grace_period as u64;
        if n >= gp && n % gp == 0 && node.depth < self.config.max_depth {
            self.try_split(leaf_idx);
        }
    }

    /// Predict the majority class for the given feature vector.
    ///
    /// Returns the class index with the highest count at the reached leaf.
    /// If no samples have been seen, returns 0.
    pub fn predict_class(&self, features: &[f64]) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let leaf_idx = self.route_to_leaf(features);
        let node = &self.nodes[leaf_idx];
        majority_class(&node.class_counts)
    }

    /// Predict the class probability distribution for the given feature vector.
    ///
    /// Returns a `Vec<f64>` of length `n_classes` where each entry is the
    /// estimated probability (fraction of samples) for that class. The
    /// probabilities sum to 1.0 (or all zeros if no samples seen).
    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        if self.nodes.is_empty() {
            return vec![0.0; self.n_classes.max(1)];
        }
        let leaf_idx = self.route_to_leaf(features);
        let node = &self.nodes[leaf_idx];
        class_probabilities(&node.class_counts)
    }

    /// Number of leaf nodes in the tree.
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Total number of nodes (leaves + splits) in the tree.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Maximum depth reached by any node in the tree.
    pub fn max_depth_seen(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Number of discovered classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Total number of training samples seen since creation or last reset.
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Reset the tree to its initial (untrained) state.
    ///
    /// Clears all nodes and counters. If `n_features` was configured up front,
    /// the root leaf is re-created; otherwise initialization is deferred again.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.n_samples = 0;
        let n_features = self.config.n_features;
        let n_classes = self.config.max_classes;
        self.n_features = n_features;
        self.n_classes = n_classes;

        if n_features > 0 {
            let root =
                ClassifierNode::new_leaf(0, n_features, self.config.n_bins, n_classes.max(2));
            self.nodes.push(root);
            if self.n_classes == 0 {
                self.n_classes = 2;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Route a feature vector from root to a leaf, returning the leaf's arena index.
    fn route_to_leaf(&self, features: &[f64]) -> usize {
        let mut idx = 0;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf() {
                return idx;
            }
            let feat = node.split_feature.unwrap();
            let thresh = node.split_threshold.unwrap();
            if feat < features.len() && features[feat] < thresh {
                idx = node.left.unwrap();
            } else {
                idx = node.right.unwrap();
            }
        }
    }

    /// Update the leaf node at `leaf_idx` with a new observation.
    fn update_leaf(&mut self, leaf_idx: usize, features: &[f64], class: usize) {
        let n_bins = self.config.n_bins;
        let node = &mut self.nodes[leaf_idx];
        node.n_samples += 1;
        node.class_counts[class] += 1;

        let stats = node.leaf_stats.as_mut().expect("leaf must have stats");
        stats.n_samples += 1;
        stats.class_counts[class] += 1;

        let half_grace = (self.config.grace_period / 2).max(1) as u64;

        for (f_idx, &val) in features.iter().enumerate().take(self.n_features) {
            // Track observed min/max for this feature.
            let (ref mut lo, ref mut hi) = stats.feature_ranges[f_idx];
            if val < *lo {
                *lo = val;
            }
            if val > *hi {
                *hi = val;
            }

            // Initialize uniform bin boundaries after enough samples.
            if stats.bin_boundaries[f_idx].is_empty() && stats.n_samples >= half_grace {
                let lo_val = stats.feature_ranges[f_idx].0;
                let hi_val = stats.feature_ranges[f_idx].1;
                if (hi_val - lo_val).abs() > 1e-15 {
                    let boundaries: Vec<f64> = (1..n_bins)
                        .map(|i| lo_val + (hi_val - lo_val) * (i as f64) / (n_bins as f64))
                        .collect();
                    stats.bin_boundaries[f_idx] = boundaries;
                }
            }

            // If boundaries are available, update the histogram.
            if !stats.bin_boundaries[f_idx].is_empty() {
                let bin = find_bin(&stats.bin_boundaries[f_idx], val);
                if bin < stats.feature_histograms[f_idx].len() {
                    stats.feature_histograms[f_idx][bin][class] += 1;
                }
            }
        }
    }

    /// Attempt to split the leaf at `leaf_idx` using information gain + Hoeffding bound.
    fn try_split(&mut self, leaf_idx: usize) {
        let n_classes = self.n_classes;

        // Compute parent entropy.
        let node = &self.nodes[leaf_idx];
        let stats = match node.leaf_stats.as_ref() {
            Some(s) => s,
            None => return,
        };
        let parent_entropy = entropy(&stats.class_counts);
        let n_total = stats.n_samples as f64;
        if n_total < 1.0 {
            return;
        }

        // Find best and second-best information gain across all features.
        let mut best_gain = f64::NEG_INFINITY;
        let mut second_best_gain = f64::NEG_INFINITY;
        let mut best_feature = 0usize;
        let mut best_bin = 0usize;

        for f_idx in 0..self.n_features {
            if stats.bin_boundaries[f_idx].is_empty() {
                continue;
            }
            let n_bins_actual = stats.feature_histograms[f_idx].len();
            // Try each bin boundary as a split point.
            for b in 0..n_bins_actual.saturating_sub(1) {
                // Accumulate left counts (bins 0..=b) and right counts (bins b+1..end).
                let mut left_counts = vec![0u64; n_classes];
                let mut right_counts = vec![0u64; n_classes];
                let mut n_left = 0u64;
                let mut n_right = 0u64;

                for bin_idx in 0..n_bins_actual {
                    let bin_counts = &stats.feature_histograms[f_idx][bin_idx];
                    for c in 0..n_classes.min(bin_counts.len()) {
                        if bin_idx <= b {
                            left_counts[c] += bin_counts[c];
                            n_left += bin_counts[c];
                        } else {
                            right_counts[c] += bin_counts[c];
                            n_right += bin_counts[c];
                        }
                    }
                }

                if n_left == 0 || n_right == 0 {
                    continue;
                }

                let n_split = (n_left + n_right) as f64;
                let left_entropy = entropy(&left_counts);
                let right_entropy = entropy(&right_counts);
                let weighted_child_entropy = (n_left as f64 / n_split) * left_entropy
                    + (n_right as f64 / n_split) * right_entropy;
                let gain = parent_entropy - weighted_child_entropy;

                if gain > best_gain {
                    second_best_gain = best_gain;
                    best_gain = gain;
                    best_feature = f_idx;
                    best_bin = b;
                } else if gain > second_best_gain {
                    second_best_gain = gain;
                }
            }
        }

        // Nothing to split on.
        if best_gain <= 0.0 {
            return;
        }

        // Compute Hoeffding bound.
        let r = if n_classes > 1 {
            (n_classes as f64).log2()
        } else {
            1.0
        };
        let epsilon = (r * r * (1.0 / self.config.delta).ln() / (2.0 * n_total)).sqrt();

        // Check if the best split is statistically significantly better.
        let delta_g = best_gain - second_best_gain.max(0.0);
        if delta_g <= epsilon && epsilon >= TAU {
            return; // Not enough evidence yet.
        }

        // Commit the split.
        let stats = self.nodes[leaf_idx].leaf_stats.as_ref().unwrap();
        let threshold = if best_bin < stats.bin_boundaries[best_feature].len() {
            stats.bin_boundaries[best_feature][best_bin]
        } else {
            // Fallback: midpoint of feature range.
            let (lo, hi) = stats.feature_ranges[best_feature];
            (lo + hi) / 2.0
        };

        let depth = self.nodes[leaf_idx].depth;
        let n_bins = self.config.n_bins;

        // Build left and right child class counts from the histogram.
        let stats = self.nodes[leaf_idx].leaf_stats.as_ref().unwrap();
        let n_bins_actual = stats.feature_histograms[best_feature].len();
        let mut left_class_counts = vec![0u64; n_classes];
        let mut right_class_counts = vec![0u64; n_classes];
        let mut n_left = 0u64;
        let mut n_right = 0u64;

        for bin_idx in 0..n_bins_actual {
            let bin_counts = &stats.feature_histograms[best_feature][bin_idx];
            for c in 0..n_classes.min(bin_counts.len()) {
                if bin_idx <= best_bin {
                    left_class_counts[c] += bin_counts[c];
                    n_left += bin_counts[c];
                } else {
                    right_class_counts[c] += bin_counts[c];
                    n_right += bin_counts[c];
                }
            }
        }

        // Create child nodes.
        let mut left_node = ClassifierNode::new_leaf(depth + 1, self.n_features, n_bins, n_classes);
        left_node.class_counts = left_class_counts;
        left_node.n_samples = n_left;

        let mut right_node =
            ClassifierNode::new_leaf(depth + 1, self.n_features, n_bins, n_classes);
        right_node.class_counts = right_class_counts;
        right_node.n_samples = n_right;

        let left_idx = self.nodes.len();
        let right_idx = left_idx + 1;
        self.nodes.push(left_node);
        self.nodes.push(right_node);

        // Convert the current leaf into a split node.
        let node = &mut self.nodes[leaf_idx];
        node.split_feature = Some(best_feature);
        node.split_threshold = Some(threshold);
        node.left = Some(left_idx);
        node.right = Some(right_idx);
        node.leaf_stats = None; // Free leaf memory.
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

impl StreamingLearner for HoeffdingTreeClassifier {
    /// Train on a single observation.
    ///
    /// The `target` is cast to `usize` for the class label. Weight is currently
    /// unused (all samples contribute equally).
    #[inline]
    fn train_one(&mut self, features: &[f64], target: f64, _weight: f64) {
        HoeffdingTreeClassifier::train_one(self, features, target as usize);
    }

    /// Predict the majority class as a floating-point value.
    #[inline]
    fn predict(&self, features: &[f64]) -> f64 {
        self.predict_class(features) as f64
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    #[inline]
    fn reset(&mut self) {
        HoeffdingTreeClassifier::reset(self);
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Find the bin index for value `x` using binary search on sorted boundaries.
///
/// Returns the index of the first boundary that is >= x, clamped to `[0, n_bins-1]`.
/// This places values below the first boundary into bin 0, values between
/// boundary[i-1] and boundary[i] into bin i, and values above the last
/// boundary into the last bin.
#[inline]
fn find_bin(boundaries: &[f64], x: f64) -> usize {
    match boundaries.binary_search_by(|b| b.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => i,
        Err(i) => i,
    }
}

/// Compute the Shannon entropy (base 2) of a class count distribution.
///
/// Returns 0.0 for empty or single-class distributions.
fn entropy(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / total_f;
            h -= p * p.log2();
        }
    }
    h
}

/// Return the index of the class with the highest count.
///
/// Ties are broken by lowest index. Returns 0 if all counts are zero.
fn majority_class(counts: &[u64]) -> usize {
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Convert class counts to a probability distribution.
///
/// Returns a vector of probabilities summing to 1.0. If total is zero,
/// returns uniform zeros.
fn class_probabilities(counts: &[u64]) -> Vec<f64> {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return vec![0.0; counts.len().max(1)];
    }
    let total_f = total as f64;
    counts.iter().map(|&c| c as f64 / total_f).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal xorshift64 PRNG for deterministic test data generation.
    fn xorshift64(state: &mut u64) -> f64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        (x as f64) / (u64::MAX as f64)
    }

    /// Shared config for most tests: small grace period for fast splitting.
    fn test_config() -> HoeffdingClassifierConfig {
        HoeffdingClassifierConfig::builder()
            .max_depth(6)
            .delta(1e-5)
            .grace_period(50)
            .n_bins(16)
            .build()
            .unwrap()
    }

    #[test]
    fn single_sample_creates_root_leaf() {
        let config = test_config();
        let mut tree = HoeffdingTreeClassifier::new(config);

        // Before any training, no nodes exist (lazy init with n_features=0).
        assert_eq!(tree.n_nodes(), 0);

        tree.train_one(&[1.0, 2.0, 3.0], 0);

        // After one sample, a single root leaf should exist.
        assert_eq!(tree.n_nodes(), 1);
        assert_eq!(tree.n_leaves(), 1);
        assert_eq!(tree.n_samples_seen(), 1);
        assert_eq!(tree.max_depth_seen(), 0);
    }

    #[test]
    fn predict_class_returns_majority() {
        let config = test_config();
        let mut tree = HoeffdingTreeClassifier::new(config);

        // Train mostly class 0.
        for _ in 0..30 {
            tree.train_one(&[1.0, 2.0], 0);
        }
        for _ in 0..5 {
            tree.train_one(&[1.0, 2.0], 1);
        }

        let predicted = tree.predict_class(&[1.0, 2.0]);
        assert_eq!(predicted, 0, "expected majority class 0, got {}", predicted);
    }

    #[test]
    fn predict_proba_sums_to_one() {
        let config = test_config();
        let mut tree = HoeffdingTreeClassifier::new(config);

        // Train with multiple classes.
        for _ in 0..20 {
            tree.train_one(&[1.0, 2.0], 0);
        }
        for _ in 0..10 {
            tree.train_one(&[1.0, 2.0], 1);
        }
        for _ in 0..5 {
            tree.train_one(&[1.0, 2.0], 2);
        }

        let proba = tree.predict_proba(&[1.0, 2.0]);
        let sum: f64 = proba.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "probabilities should sum to 1.0, got {}",
            sum
        );

        // Check that all probabilities are non-negative.
        for (i, &p) in proba.iter().enumerate() {
            assert!(p >= 0.0, "probability for class {} is negative: {}", i, p);
        }
    }

    #[test]
    fn tree_splits_on_separable_data() {
        let config = HoeffdingClassifierConfig::builder()
            .max_depth(6)
            .delta(1e-3)
            .grace_period(50)
            .n_bins(16)
            .build()
            .unwrap();
        let mut tree = HoeffdingTreeClassifier::new(config);

        // Generate clearly separable data: class 0 if x[0] < 5, class 1 otherwise.
        let mut rng_state: u64 = 42;
        for _ in 0..1000 {
            let x0 = xorshift64(&mut rng_state) * 10.0;
            let x1 = xorshift64(&mut rng_state) * 10.0; // noise feature
            let class = if x0 < 5.0 { 0 } else { 1 };
            tree.train_one(&[x0, x1], class);
        }

        // The tree should have split at least once.
        assert!(
            tree.n_nodes() > 1,
            "expected tree to split, but has only {} node(s)",
            tree.n_nodes()
        );
        assert!(
            tree.n_leaves() >= 2,
            "expected at least 2 leaves, got {}",
            tree.n_leaves()
        );

        // Verify predictions on clearly separated points.
        assert_eq!(
            tree.predict_class(&[1.0, 5.0]),
            0,
            "expected class 0 for x[0]=1.0"
        );
        assert_eq!(
            tree.predict_class(&[9.0, 5.0]),
            1,
            "expected class 1 for x[0]=9.0"
        );
    }

    #[test]
    fn max_depth_limits_growth() {
        let config = HoeffdingClassifierConfig::builder()
            .max_depth(2)
            .delta(1e-3)
            .grace_period(30)
            .n_bins(16)
            .build()
            .unwrap();
        let mut tree = HoeffdingTreeClassifier::new(config);

        // Train on separable data to encourage splitting.
        let mut rng_state: u64 = 123;
        for _ in 0..5000 {
            let x0 = xorshift64(&mut rng_state) * 10.0;
            let x1 = xorshift64(&mut rng_state) * 10.0;
            let class = if x0 < 3.0 {
                0
            } else if x0 < 6.0 {
                1
            } else {
                2
            };
            tree.train_one(&[x0, x1], class);
        }

        // Tree should respect max_depth = 2.
        assert!(
            tree.max_depth_seen() <= 2,
            "max depth should be <= 2, got {}",
            tree.max_depth_seen()
        );
    }

    #[test]
    fn streaming_learner_trait_works() {
        let config = test_config();
        let mut tree = HoeffdingTreeClassifier::new(config);

        // Train through the StreamingLearner interface.
        let learner: &mut dyn StreamingLearner = &mut tree;
        learner.train(&[1.0, 2.0], 0.0);
        learner.train(&[3.0, 4.0], 1.0);

        assert_eq!(learner.n_samples_seen(), 2);

        let pred = learner.predict(&[1.0, 2.0]);
        assert!(pred.is_finite(), "prediction should be finite");
        assert!(
            pred == 0.0 || pred == 1.0,
            "prediction should be a class label, got {}",
            pred
        );

        learner.reset();
        assert_eq!(learner.n_samples_seen(), 0);
    }

    #[test]
    fn reset_clears_state() {
        let config = test_config();
        let mut tree = HoeffdingTreeClassifier::new(config);

        // Train enough to potentially split.
        for i in 0..200 {
            let class = if i % 2 == 0 { 0 } else { 1 };
            tree.train_one(&[i as f64, (i as f64) * 0.5], class);
        }
        assert_eq!(tree.n_samples_seen(), 200);
        assert!(tree.n_nodes() >= 1);

        tree.reset();

        assert_eq!(tree.n_samples_seen(), 0);
        // After reset with n_features=0 (lazy init), nodes should be empty.
        assert_eq!(tree.n_nodes(), 0);
    }

    #[test]
    fn auto_discovers_classes() {
        let config = HoeffdingClassifierConfig::builder()
            .max_depth(4)
            .delta(1e-5)
            .grace_period(50)
            .n_bins(16)
            .max_classes(0) // auto-discover
            .build()
            .unwrap();
        let mut tree = HoeffdingTreeClassifier::new(config);

        // Start with class 0.
        tree.train_one(&[1.0], 0);
        assert!(
            tree.n_classes() >= 2,
            "should have at least 2 classes after first sample"
        );

        // Introduce class 3 (skipping 1 and 2).
        tree.train_one(&[2.0], 3);
        assert!(
            tree.n_classes() >= 4,
            "should have at least 4 classes after seeing class 3, got {}",
            tree.n_classes()
        );

        // Verify probabilities reflect all discovered classes.
        let proba = tree.predict_proba(&[1.5]);
        assert_eq!(
            proba.len(),
            tree.n_classes(),
            "proba length should match n_classes"
        );
    }

    #[test]
    fn config_builder_validates() {
        // max_depth = 0 should fail.
        let result = HoeffdingClassifierConfig::builder().max_depth(0).build();
        assert!(result.is_err(), "max_depth=0 should be rejected");

        // delta out of range should fail.
        let result = HoeffdingClassifierConfig::builder().delta(0.0).build();
        assert!(result.is_err(), "delta=0.0 should be rejected");

        let result = HoeffdingClassifierConfig::builder().delta(1.0).build();
        assert!(result.is_err(), "delta=1.0 should be rejected");

        let result = HoeffdingClassifierConfig::builder().delta(-0.5).build();
        assert!(result.is_err(), "delta=-0.5 should be rejected");

        // grace_period = 0 should fail.
        let result = HoeffdingClassifierConfig::builder().grace_period(0).build();
        assert!(result.is_err(), "grace_period=0 should be rejected");

        // n_bins < 2 should fail.
        let result = HoeffdingClassifierConfig::builder().n_bins(1).build();
        assert!(result.is_err(), "n_bins=1 should be rejected");

        // Valid config should succeed.
        let result = HoeffdingClassifierConfig::builder()
            .max_depth(5)
            .delta(0.01)
            .grace_period(100)
            .n_bins(8)
            .build();
        assert!(result.is_ok(), "valid config should build successfully");
    }
}
