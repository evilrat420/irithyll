//! Hoeffding-bound split decisions for streaming tree construction.
//!
//! [`HoeffdingTree`] is the core streaming decision tree. It grows incrementally:
//! each sample updates per-leaf histogram accumulators, and splits are committed
//! only when the Hoeffding bound guarantees the best candidate split is
//! statistically superior to the runner-up (or a tie-breaking threshold is met).
//!
//! # Algorithm
//!
//! For each incoming `(features, gradient, hessian)` triple:
//!
//! 1. Route the sample from root to a leaf via threshold comparisons.
//! 2. At the leaf, accumulate gradient/hessian into per-feature histograms.
//! 3. Once enough samples arrive (grace period), evaluate candidate splits
//!    using the XGBoost gain formula.
//! 4. Apply the Hoeffding bound: if the gap between the best and second-best
//!    gain exceeds `epsilon = sqrt(R^2 * ln(1/delta) / (2n))`, commit the split.
//! 5. When splitting, use the histogram subtraction trick to initialize one
//!    child's histograms for free.

pub mod leaf;
pub mod split_logic;

use alloc::vec;
use alloc::vec::Vec;

use crate::histogram::bins::LeafHistograms;
use crate::math;
use crate::tree::builder::TreeConfig;
use crate::tree::leaf_model::LeafModelType;
use crate::tree::node::{NodeId, TreeArena};
use crate::tree::split::{leaf_weight, XGBoostGain};
use crate::tree::StreamingTree;

use leaf::{adaptive_bound, clip_gradient, make_binners, update_output_stats, LeafState};

/// A streaming decision tree that uses Hoeffding-bound split decisions.
///
/// The tree grows incrementally: each call to [`train_one`](StreamingTree::train_one)
/// routes one sample to its leaf, updates histograms, and potentially triggers
/// a split when statistical evidence is sufficient.
///
/// # Feature subsampling
///
/// When `config.feature_subsample_rate < 1.0`, each split evaluation considers
/// only a random subset of features (selected via a deterministic xorshift64 RNG).
/// This adds diversity when the tree is used inside an ensemble.
pub struct HoeffdingTree {
    /// Arena-allocated node storage.
    pub(crate) arena: TreeArena,

    /// Root node identifier.
    pub(crate) root: NodeId,

    /// Tree configuration / hyperparameters.
    pub(crate) config: TreeConfig,

    /// Per-leaf state indexed by `NodeId.0`. Dense Vec -- NodeIds are
    /// contiguous u32 indices from TreeArena, so direct indexing is optimal.
    pub(crate) leaf_states: Vec<Option<LeafState>>,

    /// Number of features, learned from the first sample.
    pub(crate) n_features: Option<usize>,

    /// Total samples seen across all calls to `train_one`.
    pub(crate) samples_seen: u64,

    /// Split gain evaluator.
    pub(crate) split_criterion: XGBoostGain,

    /// Scratch buffer for the feature mask (avoids repeated allocation).
    pub(crate) feature_mask: Vec<usize>,

    /// Bitset scratch buffer for O(1) membership test during feature mask generation.
    /// Each bit `i` indicates whether feature `i` is already in `feature_mask`.
    pub(crate) feature_mask_bits: Vec<u64>,

    /// xorshift64 RNG state for feature subsampling.
    pub(crate) rng_state: u64,

    /// Accumulated split gains per feature for importance tracking.
    /// Indexed by feature index; grows lazily when n_features is learned.
    pub(crate) split_gains: Vec<f64>,

    /// Per-node auto-bandwidth for soft routing, indexed by `NodeId.0`.
    /// Recomputed after every structural change (split).
    pub(crate) node_bandwidths: Vec<f64>,
}

impl HoeffdingTree {
    /// Create a new `HoeffdingTree` with the given configuration.
    ///
    /// The tree starts with a single root leaf and no feature information;
    /// the number of features is inferred from the first training sample.
    pub fn new(config: TreeConfig) -> Self {
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);

        // Insert a placeholder leaf state for the root. We don't know n_features
        // yet, so give it 0 binners -- it will be properly initialized on the
        // first sample.
        let mut leaf_states = vec![None; root.0 as usize + 1];
        let root_model = match config.leaf_model_type {
            LeafModelType::ClosedForm => None,
            _ => Some(config.leaf_model_type.create(config.seed, config.delta)),
        };
        leaf_states[root.0 as usize] = Some(LeafState {
            histograms: None,
            binners: Vec::new(),
            bins_ready: false,
            grad_sum: 0.0,
            hess_sum: 0.0,
            last_reeval_count: 0,
            clip_grad_mean: 0.0,
            clip_grad_m2: 0.0,
            clip_grad_count: 0,
            output_mean: 0.0,
            output_m2: 0.0,
            output_count: 0,
            leaf_model: root_model,
        });

        let seed = config.seed;
        Self {
            arena,
            root,
            config,
            leaf_states,
            n_features: None,
            samples_seen: 0,
            split_criterion: XGBoostGain::default(),
            feature_mask: Vec::new(),
            feature_mask_bits: Vec::new(),
            rng_state: seed,
            split_gains: Vec::new(),
            node_bandwidths: Vec::new(),
        }
    }

    /// Create a leaf model for a new leaf if the config requires one.
    ///
    /// Returns `None` for `ClosedForm` (the default), which uses the existing
    /// `leaf_weight()` path with zero overhead. For `Linear` and `MLP`, returns
    /// a fresh model seeded deterministically from the config seed and node id.
    fn make_leaf_model(
        &self,
        node: NodeId,
    ) -> Option<alloc::boxed::Box<dyn crate::tree::leaf_model::LeafModel>> {
        match self.config.leaf_model_type {
            LeafModelType::ClosedForm => None,
            _ => Some(
                self.config
                    .leaf_model_type
                    .create(self.config.seed ^ (node.0 as u64), self.config.delta),
            ),
        }
    }

    /// Reconstruct a `HoeffdingTree` from a pre-built arena.
    ///
    /// Used during model deserialization. The tree is restored with node
    /// topology and leaf values intact, but histogram accumulators are empty
    /// (they will rebuild naturally from continued training).
    ///
    /// The root is assumed to be `NodeId(0)`. Leaf states are created empty
    /// for all current leaf nodes in the arena.
    pub fn from_arena(
        config: TreeConfig,
        arena: TreeArena,
        n_features: Option<usize>,
        samples_seen: u64,
        rng_state: u64,
    ) -> Self {
        let root = if arena.n_nodes() > 0 {
            NodeId(0)
        } else {
            // Empty arena -- add a root leaf (shouldn't normally happen in restore).
            let mut arena_mut = arena;
            let root = arena_mut.add_leaf(0);
            return Self {
                arena: arena_mut,
                root,
                config: config.clone(),
                leaf_states: {
                    let mut v = vec![None; root.0 as usize + 1];
                    v[root.0 as usize] = Some(LeafState::new(n_features.unwrap_or(0)));
                    v
                },
                n_features,
                samples_seen,
                split_criterion: XGBoostGain::default(),
                feature_mask: Vec::new(),
                feature_mask_bits: Vec::new(),
                rng_state,
                split_gains: vec![0.0; n_features.unwrap_or(0)],
                node_bandwidths: Vec::new(),
            };
        };

        // Build leaf states for every leaf in the arena.
        let nf = n_features.unwrap_or(0);
        let mut leaf_states: Vec<Option<LeafState>> = vec![None; arena.n_nodes()];
        for (i, slot) in leaf_states.iter_mut().enumerate() {
            if arena.is_leaf[i] {
                *slot = Some(LeafState::new(nf));
            }
        }

        Self {
            arena,
            root,
            config,
            leaf_states,
            n_features,
            samples_seen,
            split_criterion: XGBoostGain::default(),
            feature_mask: Vec::new(),
            feature_mask_bits: Vec::new(),
            rng_state,
            split_gains: vec![0.0; nf],
            node_bandwidths: Vec::new(),
        }
    }

    /// Root node identifier.
    #[inline]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Immutable access to the underlying arena.
    #[inline]
    pub fn arena(&self) -> &TreeArena {
        &self.arena
    }

    /// Immutable access to the tree configuration.
    #[inline]
    pub fn tree_config(&self) -> &TreeConfig {
        &self.config
    }

    /// Number of features (learned from the first sample, `None` before any training).
    #[inline]
    pub fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    /// Current RNG state (for deterministic checkpoint/restore).
    #[inline]
    pub fn rng_state(&self) -> u64 {
        self.rng_state
    }

    /// Read-only access to the gradient and hessian sums for a leaf node.
    ///
    /// Returns `Some((grad_sum, hess_sum))` if `node` is a leaf with an active
    /// leaf state, or `None` if the node has no state (e.g. internal node
    /// or freshly allocated).
    ///
    /// These sums enable inverse-hessian confidence estimation:
    /// `confidence = 1.0 / (hess_sum + lambda)`. High hessian means the leaf
    /// has seen consistent, informative data; low hessian means uncertainty.
    #[inline]
    pub fn leaf_grad_hess(&self, node: NodeId) -> Option<(f64, f64)> {
        self.leaf_states
            .get(node.0 as usize)
            .and_then(|o| o.as_ref())
            .map(|state| (state.grad_sum, state.hess_sum))
    }

    /// Route a feature vector from the root down to a leaf, returning the leaf's NodeId.
    pub(crate) fn route_to_leaf(&self, features: &[f64]) -> NodeId {
        let mut current = self.root;
        while !self.arena.is_leaf(current) {
            let feat_idx = self.arena.get_feature_idx(current) as usize;
            current = if let Some(mask) = self.arena.get_categorical_mask(current) {
                // Categorical split: use bitmask routing.
                // The feature value is cast to a bin index. If that bin's bit is set
                // in the mask, go left; otherwise go right.
                // For categorical features, the bin index in the histogram corresponds
                // to the sorted category position, but for bitmask routing we use
                // the original bin index directly.
                let cat_val = features[feat_idx] as u64;
                if cat_val < 64 && (mask >> cat_val) & 1 == 1 {
                    self.arena.get_left(current)
                } else {
                    self.arena.get_right(current)
                }
            } else {
                // Continuous split: standard threshold comparison.
                let threshold = self.arena.get_threshold(current);
                if features[feat_idx] <= threshold {
                    self.arena.get_left(current)
                } else {
                    self.arena.get_right(current)
                }
            };
        }
        current
    }

    /// Get the prediction value for a leaf node.
    ///
    /// Checks (in order): leaf model, live grad/hess statistics, stored leaf value.
    /// Returns `0.0` if no leaf state exists.
    #[inline]
    fn leaf_prediction(&self, leaf_id: NodeId, features: &[f64]) -> f64 {
        let (raw, leaf_bound) = if let Some(state) = self
            .leaf_states
            .get(leaf_id.0 as usize)
            .and_then(|o| o.as_ref())
        {
            // min_hessian_sum: suppress fresh leaves with insufficient samples
            if let Some(min_h) = self.config.min_hessian_sum {
                if state.hess_sum < min_h {
                    return 0.0;
                }
            }
            let val = if let Some(ref model) = state.leaf_model {
                model.predict(features)
            } else if state.hess_sum != 0.0 {
                leaf_weight(state.grad_sum, state.hess_sum, self.config.lambda)
            } else {
                self.arena.leaf_value[leaf_id.0 as usize]
            };

            // Compute per-leaf adaptive bound while state is in scope
            let bound = self
                .config
                .adaptive_leaf_bound
                .map(|k| adaptive_bound(state, k, self.config.leaf_decay_alpha));

            (val, bound)
        } else {
            (0.0, None)
        };

        // Priority: per-leaf adaptive bound > global max_leaf_output > unclamped
        if let Some(bound) = leaf_bound {
            if bound < f64::MAX {
                return raw.clamp(-bound, bound);
            }
        }
        if let Some(max) = self.config.max_leaf_output {
            raw.clamp(-max, max)
        } else {
            raw
        }
    }

    /// Predict using sigmoid-blended soft routing for smooth interpolation.
    ///
    /// Instead of hard left/right routing at each split node, uses sigmoid
    /// blending: `alpha = sigmoid((threshold - feature) / bandwidth)`. The
    /// prediction is `alpha * left_pred + (1 - alpha) * right_pred`, computed
    /// recursively from root to leaves.
    ///
    /// The result is a continuous function that varies smoothly with every
    /// feature change — no bins, no boundaries, no jumps.
    ///
    /// # Arguments
    ///
    /// * `features` - Input feature vector.
    /// * `bandwidth` - Controls transition sharpness. Smaller = sharper
    ///   (closer to hard splits), larger = smoother.
    pub fn predict_smooth(&self, features: &[f64], bandwidth: f64) -> f64 {
        self.predict_smooth_recursive(self.root, features, bandwidth)
    }

    /// Predict using per-feature auto-calibrated bandwidths.
    ///
    /// Each feature uses its own bandwidth derived from median split threshold
    /// gaps. Features with `f64::INFINITY` bandwidth fall back to hard routing.
    pub fn predict_smooth_auto(&self, features: &[f64], bandwidths: &[f64]) -> f64 {
        self.predict_smooth_auto_recursive(self.root, features, bandwidths)
    }

    /// Predict with parent-leaf linear interpolation.
    ///
    /// Routes to the leaf but blends the leaf prediction with the parent node's
    /// preserved prediction based on the leaf's hessian sum. Fresh leaves
    /// (low hess_sum) smoothly transition from parent prediction to their own:
    ///
    /// `alpha = leaf_hess / (leaf_hess + lambda)`
    /// `pred = alpha * leaf_pred + (1 - alpha) * parent_pred`
    ///
    /// This fixes static predictions from leaves that split but haven't
    /// accumulated enough samples to outperform their parent.
    pub fn predict_interpolated(&self, features: &[f64]) -> f64 {
        let mut current = self.root;
        let mut parent = None;
        while !self.arena.is_leaf(current) {
            parent = Some(current);
            let feat_idx = self.arena.get_feature_idx(current) as usize;
            current = if let Some(mask) = self.arena.get_categorical_mask(current) {
                let cat_val = features[feat_idx] as u64;
                if cat_val < 64 && (mask >> cat_val) & 1 == 1 {
                    self.arena.get_left(current)
                } else {
                    self.arena.get_right(current)
                }
            } else {
                let threshold = self.arena.get_threshold(current);
                if features[feat_idx] <= threshold {
                    self.arena.get_left(current)
                } else {
                    self.arena.get_right(current)
                }
            };
        }

        let leaf_pred = self.leaf_prediction(current, features);

        // No parent (root is leaf) → return leaf prediction directly
        let parent_id = match parent {
            Some(p) => p,
            None => return leaf_pred,
        };

        // Get parent's preserved prediction from its old leaf state
        let parent_pred = self.leaf_prediction(parent_id, features);

        // Blend: alpha = leaf_hess / (leaf_hess + lambda)
        let leaf_hess = self
            .leaf_states
            .get(current.0 as usize)
            .and_then(|o| o.as_ref())
            .map(|s| s.hess_sum)
            .unwrap_or(0.0);

        let alpha = leaf_hess / (leaf_hess + self.config.lambda);
        alpha * leaf_pred + (1.0 - alpha) * parent_pred
    }

    /// Predict with sibling-based interpolation for feature-continuous predictions.
    ///
    /// At the leaf's parent split, blends the leaf prediction with its sibling's
    /// prediction based on the feature's distance from the split threshold:
    ///
    /// Within the margin `m` around the threshold:
    /// `t = (feature - threshold + m) / (2 * m)`  (0 at left edge, 1 at right edge)
    /// `pred = (1 - t) * left_pred + t * right_pred`
    ///
    /// Outside the margin, returns the routed child's prediction directly.
    /// The margin `m` is derived from auto-bandwidths if available, otherwise
    /// defaults to `feature_range / n_bins` heuristic per feature.
    ///
    /// This makes predictions vary continuously as features move near split
    /// boundaries, eliminating step-function artifacts.
    pub fn predict_sibling_interpolated(&self, features: &[f64], bandwidths: &[f64]) -> f64 {
        self.predict_sibling_recursive(self.root, features, bandwidths)
    }

    fn predict_sibling_recursive(&self, node: NodeId, features: &[f64], bandwidths: &[f64]) -> f64 {
        if self.arena.is_leaf(node) {
            return self.leaf_prediction(node, features);
        }

        let feat_idx = self.arena.get_feature_idx(node) as usize;
        let left = self.arena.get_left(node);
        let right = self.arena.get_right(node);

        // Categorical splits: always hard routing (no interpolation)
        if let Some(mask) = self.arena.get_categorical_mask(node) {
            let cat_val = features[feat_idx] as u64;
            return if cat_val < 64 && (mask >> cat_val) & 1 == 1 {
                self.predict_sibling_recursive(left, features, bandwidths)
            } else {
                self.predict_sibling_recursive(right, features, bandwidths)
            };
        }

        let threshold = self.arena.get_threshold(node);
        let margin = bandwidths.get(feat_idx).copied().unwrap_or(f64::INFINITY);

        // No valid margin or infinite → hard routing
        if !margin.is_finite() || margin <= 0.0 {
            return if features[feat_idx] <= threshold {
                self.predict_sibling_recursive(left, features, bandwidths)
            } else {
                self.predict_sibling_recursive(right, features, bandwidths)
            };
        }

        let dist = features[feat_idx] - threshold;

        if dist < -margin {
            // Firmly in left child territory
            self.predict_sibling_recursive(left, features, bandwidths)
        } else if dist > margin {
            // Firmly in right child territory
            self.predict_sibling_recursive(right, features, bandwidths)
        } else {
            // Within the interpolation margin: linear blend
            let t = (dist + margin) / (2.0 * margin); // 0.0 at left edge, 1.0 at right edge
            let left_pred = self.predict_sibling_recursive(left, features, bandwidths);
            let right_pred = self.predict_sibling_recursive(right, features, bandwidths);
            (1.0 - t) * left_pred + t * right_pred
        }
    }

    /// Collect all split thresholds per feature from the tree arena.
    ///
    /// Returns a `Vec<Vec<f64>>` indexed by feature, containing all thresholds
    /// used in continuous splits. Categorical splits are excluded.
    pub fn collect_split_thresholds_per_feature(&self) -> Vec<Vec<f64>> {
        let n = self.n_features.unwrap_or(0);
        let mut thresholds: Vec<Vec<f64>> = vec![Vec::new(); n];

        for i in 0..self.arena.n_nodes() {
            if !self.arena.is_leaf[i] && self.arena.categorical_mask[i].is_none() {
                let feat_idx = self.arena.feature_idx[i] as usize;
                if feat_idx < n {
                    thresholds[feat_idx].push(self.arena.threshold[i]);
                }
            }
        }

        thresholds
    }

    /// Compute per-node bandwidth from nearest neighbor thresholds on the same feature.
    fn compute_node_bandwidth(&self, node: NodeId, all_thresholds: &[Vec<f64>]) -> f64 {
        let feat_idx = self.arena.get_feature_idx(node) as usize;
        let threshold = self.arena.get_threshold(node);

        let thresholds = if feat_idx < all_thresholds.len() {
            &all_thresholds[feat_idx]
        } else {
            return f64::INFINITY;
        };

        // Find nearest neighbors (thresholds are sorted)
        let below = thresholds.iter().rev().find(|&&t| t < threshold - 1e-15);
        let above = thresholds.iter().find(|&&t| t > threshold + 1e-15);

        match (below, above) {
            (Some(&b), Some(&a)) => (threshold - b).min(a - threshold),
            (Some(&b), None) => threshold - b,
            (None, Some(&a)) => a - threshold,
            (None, None) => f64::INFINITY,
        }
    }

    /// Recompute all node bandwidths. Call after structural changes.
    pub fn recompute_bandwidths(&mut self) {
        let n = self.arena.n_nodes();
        self.node_bandwidths.resize(n, f64::INFINITY);

        // Collect and sort thresholds once
        let mut all_thresholds = self.collect_split_thresholds_per_feature();
        for v in &mut all_thresholds {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        }

        for i in 0..n {
            let nid = NodeId(i as u32);
            if !self.arena.is_leaf(nid) {
                self.node_bandwidths[i] = self.compute_node_bandwidth(nid, &all_thresholds);
            } else {
                self.node_bandwidths[i] = f64::INFINITY;
            }
        }
    }

    /// Predict using per-node auto-bandwidth soft routing.
    /// Every prediction is a continuous weighted blend — no step discontinuities.
    pub fn predict_soft_routed(&self, features: &[f64]) -> f64 {
        self.predict_soft_recursive(self.root, features)
    }

    fn predict_soft_recursive(&self, node: NodeId, features: &[f64]) -> f64 {
        if self.arena.is_leaf(node) {
            return self.leaf_prediction(node, features);
        }

        let feat_idx = self.arena.get_feature_idx(node) as usize;
        let left = self.arena.get_left(node);
        let right = self.arena.get_right(node);

        // Categorical: hard routing
        if let Some(mask) = self.arena.get_categorical_mask(node) {
            let cat_val = features[feat_idx] as u64;
            return if cat_val < 64 && (mask >> cat_val) & 1 == 1 {
                self.predict_soft_recursive(left, features)
            } else {
                self.predict_soft_recursive(right, features)
            };
        }

        let threshold = self.arena.get_threshold(node);
        let margin = self
            .node_bandwidths
            .get(node.0 as usize)
            .copied()
            .unwrap_or(f64::INFINITY);

        let left_pred = self.predict_soft_recursive(left, features);
        let right_pred = self.predict_soft_recursive(right, features);

        // Non-finite or zero margin: sigmoid fallback
        if !margin.is_finite() || margin <= 0.0 {
            let dist = features[feat_idx] - threshold;
            let scale = math::abs(threshold) * 0.01 + 1e-10;
            let z = (-dist / scale).clamp(-500.0, 500.0);
            let t = 1.0 / (1.0 + math::exp(z));
            return (1.0 - t) * left_pred + t * right_pred;
        }

        // Linear soft routing: always blend
        let dist = features[feat_idx] - threshold;
        let t = ((dist + margin) / (2.0 * margin)).clamp(0.0, 1.0);
        (1.0 - t) * left_pred + t * right_pred
    }

    /// Recursive sigmoid-blended prediction traversal.
    fn predict_smooth_recursive(&self, node: NodeId, features: &[f64], bandwidth: f64) -> f64 {
        if self.arena.is_leaf(node) {
            // At a leaf, return the leaf prediction (same as regular predict)
            return self.leaf_prediction(node, features);
        }

        let feat_idx = self.arena.get_feature_idx(node) as usize;
        let left = self.arena.get_left(node);
        let right = self.arena.get_right(node);

        // Categorical splits: hard routing (sigmoid blending is meaningless for categories)
        if let Some(mask) = self.arena.get_categorical_mask(node) {
            let cat_val = features[feat_idx] as u64;
            return if cat_val < 64 && (mask >> cat_val) & 1 == 1 {
                self.predict_smooth_recursive(left, features, bandwidth)
            } else {
                self.predict_smooth_recursive(right, features, bandwidth)
            };
        }

        // Continuous split: sigmoid blending for smooth transition around the threshold
        let threshold = self.arena.get_threshold(node);
        let z = (threshold - features[feat_idx]) / bandwidth;
        let alpha = 1.0 / (1.0 + math::exp(-z));

        let left_pred = self.predict_smooth_recursive(left, features, bandwidth);
        let right_pred = self.predict_smooth_recursive(right, features, bandwidth);

        alpha * left_pred + (1.0 - alpha) * right_pred
    }

    /// Recursive per-feature-bandwidth smooth prediction traversal.
    fn predict_smooth_auto_recursive(
        &self,
        node: NodeId,
        features: &[f64],
        bandwidths: &[f64],
    ) -> f64 {
        if self.arena.is_leaf(node) {
            return self.leaf_prediction(node, features);
        }

        let feat_idx = self.arena.get_feature_idx(node) as usize;
        let left = self.arena.get_left(node);
        let right = self.arena.get_right(node);

        // Categorical splits: always hard routing
        if let Some(mask) = self.arena.get_categorical_mask(node) {
            let cat_val = features[feat_idx] as u64;
            return if cat_val < 64 && (mask >> cat_val) & 1 == 1 {
                self.predict_smooth_auto_recursive(left, features, bandwidths)
            } else {
                self.predict_smooth_auto_recursive(right, features, bandwidths)
            };
        }

        let threshold = self.arena.get_threshold(node);
        let bw = bandwidths.get(feat_idx).copied().unwrap_or(f64::INFINITY);

        // Infinite bandwidth = feature never split on across ensemble → hard routing
        if !bw.is_finite() {
            return if features[feat_idx] <= threshold {
                self.predict_smooth_auto_recursive(left, features, bandwidths)
            } else {
                self.predict_smooth_auto_recursive(right, features, bandwidths)
            };
        }

        // Sigmoid-blended soft routing with per-feature bandwidth
        let z = (threshold - features[feat_idx]) / bw;
        let alpha = 1.0 / (1.0 + math::exp(-z));

        let left_pred = self.predict_smooth_auto_recursive(left, features, bandwidths);
        let right_pred = self.predict_smooth_auto_recursive(right, features, bandwidths);

        alpha * left_pred + (1.0 - alpha) * right_pred
    }

    /// Attempt a split at the given leaf node.
    ///
    /// Returns `true` if a split was performed.
    pub(crate) fn attempt_split(&mut self, leaf_id: NodeId) -> bool {
        let depth = self.arena.get_depth(leaf_id);

        // When adaptive_depth is enabled, max_depth * 2 is the hard safety ceiling;
        // the per-split CIR test handles generalization. Otherwise, use static max_depth.
        let hard_ceiling = if self.config.adaptive_depth.is_some() {
            self.config.max_depth.saturating_mul(2)
        } else {
            self.config.max_depth
        };
        let at_max_depth = depth as usize >= hard_ceiling;

        if at_max_depth {
            // Only proceed if split re-evaluation is enabled and the interval
            // has elapsed since the last evaluation at this leaf.
            match self.config.split_reeval_interval {
                None => return false,
                Some(interval) => {
                    let state = match self
                        .leaf_states
                        .get(leaf_id.0 as usize)
                        .and_then(|o| o.as_ref())
                    {
                        Some(s) => s,
                        None => return false,
                    };
                    let sample_count = self.arena.get_sample_count(leaf_id);
                    if sample_count - state.last_reeval_count < interval as u64 {
                        return false;
                    }
                    // Fall through to evaluate potential split.
                }
            }
        }

        let n_features = match self.n_features {
            Some(n) => n,
            None => return false,
        };

        let sample_count = self.arena.get_sample_count(leaf_id);
        if sample_count < self.config.grace_period as u64 {
            return false;
        }

        // Generate the feature mask for this split evaluation.
        let (feature_mask, feature_mask_bits) = split_logic::generate_feature_mask(
            core::mem::take(&mut self.feature_mask),
            core::mem::take(&mut self.feature_mask_bits),
            &mut self.rng_state,
            self.config.feature_subsample_rate,
            n_features,
        );
        self.feature_mask = feature_mask;
        self.feature_mask_bits = feature_mask_bits;

        // Materialize pending lazy decay before reading histogram data.
        if self.config.leaf_decay_alpha.is_some() {
            if let Some(state) = self
                .leaf_states
                .get_mut(leaf_id.0 as usize)
                .and_then(|o| o.as_mut())
            {
                if let Some(ref mut histograms) = state.histograms {
                    histograms.materialize_decay();
                }
            }
        }

        // Evaluate splits for each feature in the mask.
        let state = match self
            .leaf_states
            .get(leaf_id.0 as usize)
            .and_then(|o| o.as_ref())
        {
            Some(s) => s,
            None => return false,
        };

        let histograms = match &state.histograms {
            Some(h) => h,
            None => return false,
        };

        let ctx = split_logic::private::SplitContext {
            config: &self.config,
            n_features: self.n_features,
            n_feature_mask: &self.feature_mask,
            split_criterion: &self.split_criterion,
            rng_state: &mut self.rng_state,
        };

        let candidates = split_logic::private::evaluate_split_candidates(
            histograms,
            self.config.feature_types.as_deref(),
            &ctx,
        );

        if candidates.is_empty() {
            return false;
        }

        let best_gain = candidates[0].1.gain;
        let second_best_gain = if candidates.len() > 1 {
            candidates[1].1.gain
        } else {
            0.0
        };

        // Check Hoeffding bound and adaptive depth.
        let ctx = split_logic::private::SplitContext {
            config: &self.config,
            n_features: self.n_features,
            n_feature_mask: &self.feature_mask,
            split_criterion: &self.split_criterion,
            rng_state: &mut self.rng_state,
        };

        if !split_logic::private::should_split_hoeffding(
            best_gain,
            second_best_gain,
            sample_count,
            &ctx,
        ) {
            if at_max_depth {
                if let Some(state) = self
                    .leaf_states
                    .get_mut(leaf_id.0 as usize)
                    .and_then(|o| o.as_mut())
                {
                    state.last_reeval_count = sample_count;
                }
            }
            return false;
        }

        // --- Execute the split ---
        let (best_feat_idx, ref best_candidate, ref fisher_order) = candidates[0];

        // Track split gain for feature importance.
        if best_feat_idx < self.split_gains.len() {
            self.split_gains[best_feat_idx] += best_candidate.gain;
        }

        let best_hist = &histograms.histograms[best_feat_idx];

        let left_value = leaf_weight(
            best_candidate.left_grad,
            best_candidate.left_hess,
            self.config.lambda,
        );
        let right_value = leaf_weight(
            best_candidate.right_grad,
            best_candidate.right_hess,
            self.config.lambda,
        );

        // Perform the split -- categorical or continuous.
        let (left_id, right_id) = if let Some(ref order) = fisher_order {
            let mut mask: u64 = 0;
            for &sorted_pos in order.iter().take(best_candidate.bin_idx + 1) {
                if sorted_pos < 64 {
                    mask |= 1u64 << sorted_pos;
                }
            }

            self.arena.split_leaf_categorical(
                leaf_id,
                best_feat_idx as u32,
                0.0,
                left_value,
                right_value,
                mask,
            )
        } else {
            let threshold = if best_candidate.bin_idx < best_hist.edges.edges.len() {
                best_hist.edges.edges[best_candidate.bin_idx]
            } else {
                f64::MAX
            };

            self.arena.split_leaf(
                leaf_id,
                best_feat_idx as u32,
                threshold,
                left_value,
                right_value,
            )
        };

        let parent_state = self
            .leaf_states
            .get_mut(leaf_id.0 as usize)
            .and_then(|o| o.take());
        let nf = n_features;

        // Ensure Vec is large enough for child NodeIds.
        let max_child = left_id.0.max(right_id.0) as usize;
        if self.leaf_states.len() <= max_child {
            self.leaf_states.resize_with(max_child + 1, || None);
        }

        if let Some(parent) = parent_state {
            if let Some(parent_hists) = parent.histograms {
                let edges_per_feature: Vec<crate::histogram::BinEdges> = parent_hists
                    .histograms
                    .iter()
                    .map(|h| h.edges.clone())
                    .collect();

                let left_hists = LeafHistograms::new(&edges_per_feature);
                let right_hists = LeafHistograms::new(&edges_per_feature);

                let ft = self.config.feature_types.as_deref();
                let child_binners_l = make_binners(nf, ft);
                let child_binners_r = make_binners(nf, ft);

                let left_model = parent.leaf_model.as_ref().map(|m| m.clone_warm());
                let right_model = parent.leaf_model.as_ref().map(|m| m.clone_warm());

                let left_state = LeafState {
                    histograms: Some(left_hists),
                    binners: child_binners_l,
                    bins_ready: true,
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    last_reeval_count: 0,
                    clip_grad_mean: 0.0,
                    clip_grad_m2: 0.0,
                    clip_grad_count: 0,
                    output_mean: 0.0,
                    output_m2: 0.0,
                    output_count: 0,
                    leaf_model: left_model,
                };

                let right_state = LeafState {
                    histograms: Some(right_hists),
                    binners: child_binners_r,
                    bins_ready: true,
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    last_reeval_count: 0,
                    clip_grad_mean: 0.0,
                    clip_grad_m2: 0.0,
                    clip_grad_count: 0,
                    output_mean: 0.0,
                    output_m2: 0.0,
                    output_count: 0,
                    leaf_model: right_model,
                };

                self.leaf_states[left_id.0 as usize] = Some(left_state);
                self.leaf_states[right_id.0 as usize] = Some(right_state);
            } else {
                let ft = self.config.feature_types.as_deref();
                let mut ls = LeafState::new_with_types(nf, ft);
                ls.leaf_model = parent.leaf_model.as_ref().map(|m| m.clone_warm());
                self.leaf_states[left_id.0 as usize] = Some(ls);
                let mut rs = LeafState::new_with_types(nf, ft);
                rs.leaf_model = parent.leaf_model.as_ref().map(|m| m.clone_warm());
                self.leaf_states[right_id.0 as usize] = Some(rs);
            }
        } else {
            let ft = self.config.feature_types.as_deref();
            let mut ls = LeafState::new_with_types(nf, ft);
            ls.leaf_model = self.make_leaf_model(left_id);
            self.leaf_states[left_id.0 as usize] = Some(ls);
            let mut rs = LeafState::new_with_types(nf, ft);
            rs.leaf_model = self.make_leaf_model(right_id);
            self.leaf_states[right_id.0 as usize] = Some(rs);
        }

        self.recompute_bandwidths();
        true
    }
}

impl StreamingTree for HoeffdingTree {
    fn train_one(&mut self, features: &[f64], gradient: f64, hessian: f64) {
        self.samples_seen += 1;

        let n_features = if let Some(n) = self.n_features {
            n
        } else {
            let n = features.len();
            self.n_features = Some(n);
            self.split_gains.resize(n, 0.0);

            if let Some(state) = self
                .leaf_states
                .get_mut(self.root.0 as usize)
                .and_then(|o| o.as_mut())
            {
                state.binners = make_binners(n, self.config.feature_types.as_deref());
            }
            n
        };

        debug_assert_eq!(
            features.len(),
            n_features,
            "feature count mismatch: got {} but expected {}",
            features.len(),
            n_features,
        );

        let leaf_id = self.route_to_leaf(features);
        self.arena.increment_sample_count(leaf_id);
        let sample_count = self.arena.get_sample_count(leaf_id);

        let idx = leaf_id.0 as usize;
        if self.leaf_states.len() <= idx {
            self.leaf_states.resize_with(idx + 1, || None);
        }
        if self.leaf_states[idx].is_none() {
            self.leaf_states[idx] = Some(LeafState::new_with_types(
                n_features,
                self.config.feature_types.as_deref(),
            ));
        }
        let state = self.leaf_states[idx].as_mut().unwrap();

        let gradient = if let Some(sigma) = self.config.gradient_clip_sigma {
            clip_gradient(state, gradient, sigma)
        } else {
            gradient
        };

        if !state.bins_ready {
            for (binner, &val) in state.binners.iter_mut().zip(features.iter()) {
                binner.observe(val);
            }

            if let Some(alpha) = self.config.leaf_decay_alpha {
                state.grad_sum = alpha * state.grad_sum + gradient;
                state.hess_sum = alpha * state.hess_sum + hessian;
            } else {
                state.grad_sum += gradient;
                state.hess_sum += hessian;
            }

            let lw = leaf_weight(state.grad_sum, state.hess_sum, self.config.lambda);
            self.arena.set_leaf_value(leaf_id, lw);

            if self.config.adaptive_leaf_bound.is_some() {
                update_output_stats(state, lw, self.config.leaf_decay_alpha);
            }

            if let Some(ref mut model) = state.leaf_model {
                model.update(features, gradient, hessian, self.config.lambda);
            }

            if sample_count >= self.config.grace_period as u64 {
                let edges_per_feature: Vec<crate::histogram::BinEdges> = state
                    .binners
                    .iter()
                    .map(|b| b.compute_edges(self.config.n_bins))
                    .collect();

                let mut histograms = LeafHistograms::new(&edges_per_feature);

                if let Some(alpha) = self.config.leaf_decay_alpha {
                    histograms.accumulate_with_decay(features, gradient, hessian, alpha);
                } else {
                    histograms.accumulate(features, gradient, hessian);
                }

                state.histograms = Some(histograms);
                state.bins_ready = true;
            }

            return;
        }

        if let Some(ref mut histograms) = state.histograms {
            if let Some(alpha) = self.config.leaf_decay_alpha {
                histograms.accumulate_with_decay(features, gradient, hessian, alpha);
            } else {
                histograms.accumulate(features, gradient, hessian);
            }
        }

        if let Some(alpha) = self.config.leaf_decay_alpha {
            state.grad_sum = alpha * state.grad_sum + gradient;
            state.hess_sum = alpha * state.hess_sum + hessian;
        } else {
            state.grad_sum += gradient;
            state.hess_sum += hessian;
        }
        let lw = leaf_weight(state.grad_sum, state.hess_sum, self.config.lambda);
        self.arena.set_leaf_value(leaf_id, lw);

        if self.config.adaptive_leaf_bound.is_some() {
            update_output_stats(state, lw, self.config.leaf_decay_alpha);
        }

        if let Some(ref mut model) = state.leaf_model {
            model.update(features, gradient, hessian, self.config.lambda);
        }

        if sample_count % (self.config.grace_period as u64) == 0 {
            self.attempt_split(leaf_id);
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let leaf_id = self.route_to_leaf(features);
        self.leaf_prediction(leaf_id, features)
    }

    #[inline]
    fn n_leaves(&self) -> usize {
        self.arena.n_leaves()
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        self.arena.reset();
        let root = self.arena.add_leaf(0);
        self.root = root;
        self.leaf_states.clear();

        let n_features = self.n_features.unwrap_or(0);
        self.leaf_states.resize_with(root.0 as usize + 1, || None);
        let mut root_state =
            LeafState::new_with_types(n_features, self.config.feature_types.as_deref());
        root_state.leaf_model = self.make_leaf_model(root);
        self.leaf_states[root.0 as usize] = Some(root_state);

        self.samples_seen = 0;
        self.feature_mask.clear();
        self.feature_mask_bits.clear();
        self.rng_state = self.config.seed;
        self.split_gains.iter_mut().for_each(|g| *g = 0.0);
        self.node_bandwidths.clear();
    }

    fn split_gains(&self) -> &[f64] {
        &self.split_gains
    }

    fn predict_with_variance(&self, features: &[f64]) -> (f64, f64) {
        let leaf_id = self.route_to_leaf(features);
        let value = self.leaf_prediction(leaf_id, features);
        if let Some(state) = self
            .leaf_states
            .get(leaf_id.0 as usize)
            .and_then(|o| o.as_ref())
        {
            let variance = 1.0 / (state.hess_sum + self.config.lambda);
            (value, variance)
        } else {
            (value, f64::INFINITY)
        }
    }
}

impl Clone for HoeffdingTree {
    fn clone(&self) -> Self {
        Self {
            arena: self.arena.clone(),
            root: self.root,
            config: self.config.clone(),
            leaf_states: self.leaf_states.clone(),
            n_features: self.n_features,
            samples_seen: self.samples_seen,
            split_criterion: self.split_criterion,
            feature_mask: self.feature_mask.clone(),
            feature_mask_bits: self.feature_mask_bits.clone(),
            rng_state: self.rng_state,
            split_gains: self.split_gains.clone(),
            node_bandwidths: self.node_bandwidths.clone(),
        }
    }
}

// SAFETY: All fields are Send + Sync. BinnerKind is a concrete enum with
// Send + Sync variants. XGBoostGain is stateless. Vec<Option<LeafState>>
// and Vec fields are trivially Send + Sync.
unsafe impl Send for HoeffdingTree {}
unsafe impl Sync for HoeffdingTree {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::builder::TreeConfig;
    use crate::tree::StreamingTree;

    #[test]
    fn single_sample_predict_not_nan() {
        let config = TreeConfig::new().grace_period(10);
        let mut tree = HoeffdingTree::new(config);

        let features = vec![1.0, 2.0, 3.0];
        tree.train_one(&features, -0.5, 1.0);

        let pred = tree.predict(&features);
        assert!(!pred.is_nan(), "prediction should not be NaN, got {}", pred);
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );

        assert!((pred - 0.25).abs() < 1e-10, "expected ~0.25, got {}", pred);
    }
}
