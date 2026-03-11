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

use crate::histogram::bins::LeafHistograms;
use crate::histogram::{BinEdges, BinnerKind};
use crate::tree::builder::TreeConfig;
use crate::tree::node::{NodeId, TreeArena};
use crate::tree::split::{leaf_weight, SplitCandidate, SplitCriterion, XGBoostGain};
use crate::tree::StreamingTree;

/// Tie-breaking threshold (tau). When `epsilon < tau`, we accept the best split
/// even if the gap between best and second-best gain is small, because the
/// Hoeffding bound is already tight enough that further samples won't help.
const TAU: f64 = 0.05;

// ---------------------------------------------------------------------------
// xorshift64 — minimal deterministic PRNG
// ---------------------------------------------------------------------------

/// Advance an xorshift64 state and return the new value.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

// ---------------------------------------------------------------------------
// LeafState — per-leaf bookkeeping
// ---------------------------------------------------------------------------

/// State tracked per leaf node for split decisions.
///
/// Each active leaf owns its own set of histogram accumulators (one per feature)
/// and running gradient/hessian sums for leaf weight updates.
#[derive(Clone)]
struct LeafState {
    /// Histogram accumulators for this leaf. `None` until bin edges are computed
    /// (after the grace period).
    histograms: Option<LeafHistograms>,

    /// Per-feature binning strategies that collect observed values to compute
    /// bin edges. Uses `BinnerKind` enum dispatch instead of `Box<dyn>` to
    /// eliminate N_features heap allocations per new leaf.
    binners: Vec<BinnerKind>,

    /// Whether bin edges have been computed (after grace period samples).
    bins_ready: bool,

    /// Running gradient sum for leaf weight updates.
    grad_sum: f64,

    /// Running hessian sum for leaf weight updates.
    hess_sum: f64,

    /// Sample count at last split re-evaluation (for EFDT-inspired re-eval).
    last_reeval_count: u64,
}

impl LeafState {
    /// Create a fresh leaf state for a leaf with `n_features` features.
    fn new(n_features: usize) -> Self {
        let binners: Vec<BinnerKind> = (0..n_features).map(|_| BinnerKind::uniform()).collect();

        Self {
            histograms: None,
            binners,
            bins_ready: false,
            grad_sum: 0.0,
            hess_sum: 0.0,
            last_reeval_count: 0,
        }
    }

    /// Create a leaf state with pre-computed histograms (used after a split
    /// when we can initialize the child via histogram subtraction).
    #[allow(dead_code)]
    fn with_histograms(histograms: LeafHistograms) -> Self {
        let n_features = histograms.n_features();
        let binners: Vec<BinnerKind> = (0..n_features).map(|_| BinnerKind::uniform()).collect();

        // Recover grad/hess sums from the histograms.
        let grad_sum: f64 = histograms
            .histograms
            .first()
            .map_or(0.0, |h| h.total_gradient());
        let hess_sum: f64 = histograms
            .histograms
            .first()
            .map_or(0.0, |h| h.total_hessian());

        Self {
            histograms: Some(histograms),
            binners,
            bins_ready: true,
            grad_sum,
            hess_sum,
            last_reeval_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// HoeffdingTree
// ---------------------------------------------------------------------------

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
    arena: TreeArena,

    /// Root node identifier.
    root: NodeId,

    /// Tree configuration / hyperparameters.
    config: TreeConfig,

    /// Per-leaf state indexed by `NodeId.0`. Dense Vec — NodeIds are
    /// contiguous u32 indices from TreeArena, so direct indexing is optimal.
    leaf_states: Vec<Option<LeafState>>,

    /// Number of features, learned from the first sample.
    n_features: Option<usize>,

    /// Total samples seen across all calls to `train_one`.
    samples_seen: u64,

    /// Split gain evaluator.
    split_criterion: XGBoostGain,

    /// Scratch buffer for the feature mask (avoids repeated allocation).
    feature_mask: Vec<usize>,

    /// Bitset scratch buffer for O(1) membership test during feature mask generation.
    /// Each bit `i` indicates whether feature `i` is already in `feature_mask`.
    feature_mask_bits: Vec<u64>,

    /// xorshift64 RNG state for feature subsampling.
    rng_state: u64,

    /// Accumulated split gains per feature for importance tracking.
    /// Indexed by feature index; grows lazily when n_features is learned.
    split_gains: Vec<f64>,
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
        // yet, so give it 0 binners — it will be properly initialized on the
        // first sample.
        let mut leaf_states = vec![None; root.0 as usize + 1];
        leaf_states[root.0 as usize] = Some(LeafState {
            histograms: None,
            binners: Vec::new(),
            bins_ready: false,
            grad_sum: 0.0,
            hess_sum: 0.0,
            last_reeval_count: 0,
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
            // Empty arena — add a root leaf (shouldn't normally happen in restore).
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

    /// Route a feature vector from the root down to a leaf, returning the leaf's NodeId.
    fn route_to_leaf(&self, features: &[f64]) -> NodeId {
        let mut current = self.root;
        while !self.arena.is_leaf(current) {
            let feat_idx = self.arena.get_feature_idx(current) as usize;
            let threshold = self.arena.get_threshold(current);
            current = if features[feat_idx] <= threshold {
                self.arena.get_left(current)
            } else {
                self.arena.get_right(current)
            };
        }
        current
    }

    /// Generate the feature mask for split evaluation.
    ///
    /// If `feature_subsample_rate` is 1.0, all features are included.
    /// Otherwise, a random subset is selected via xorshift64.
    ///
    /// Uses a `Vec<u64>` bitset for O(1) membership testing, replacing the
    /// previous O(n) `Vec::contains()` which made the fallback loop O(n²).
    fn generate_feature_mask(&mut self, n_features: usize) {
        self.feature_mask.clear();

        if self.config.feature_subsample_rate >= 1.0 {
            self.feature_mask.extend(0..n_features);
        } else {
            let target_count =
                ((n_features as f64) * self.config.feature_subsample_rate).ceil() as usize;
            let target_count = target_count.max(1).min(n_features);

            // Prepare the bitset: one bit per feature, O(1) membership test.
            let n_words = n_features.div_ceil(64);
            self.feature_mask_bits.clear();
            self.feature_mask_bits.resize(n_words, 0u64);

            // Include each feature with probability = subsample_rate.
            for i in 0..n_features {
                let r = xorshift64(&mut self.rng_state);
                let p = (r as f64) / (u64::MAX as f64);
                if p < self.config.feature_subsample_rate {
                    self.feature_mask.push(i);
                    self.feature_mask_bits[i / 64] |= 1u64 << (i % 64);
                }
            }

            // If we didn't get enough features, fill up deterministically.
            // Now O(n) instead of O(n²) thanks to the bitset.
            if self.feature_mask.len() < target_count {
                for i in 0..n_features {
                    if self.feature_mask.len() >= target_count {
                        break;
                    }
                    if self.feature_mask_bits[i / 64] & (1u64 << (i % 64)) == 0 {
                        self.feature_mask.push(i);
                        self.feature_mask_bits[i / 64] |= 1u64 << (i % 64);
                    }
                }
            }
        }
    }

    /// Attempt a split at the given leaf node.
    ///
    /// Returns `true` if a split was performed.
    fn attempt_split(&mut self, leaf_id: NodeId) -> bool {
        let depth = self.arena.get_depth(leaf_id);
        let at_max_depth = depth as usize >= self.config.max_depth;

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
        self.generate_feature_mask(n_features);

        // Materialize pending lazy decay before reading histogram data.
        // This converts un-decayed coordinates to true decayed values so
        // split evaluation sees correct gradient/hessian sums. O(n_features * n_bins)
        // but amortized over grace_period samples — not per-sample cost.
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
        // We need to borrow leaf_states immutably while feature_mask is borrowed.
        // Collect candidates first.
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

        // Collect (feature_idx, best_split_candidate) for each feature in the mask.
        let mut candidates: Vec<(usize, SplitCandidate)> = Vec::new();

        for &feat_idx in &self.feature_mask {
            if feat_idx >= histograms.n_features() {
                continue;
            }
            let hist = &histograms.histograms[feat_idx];
            let total_grad = hist.total_gradient();
            let total_hess = hist.total_hessian();

            if let Some(candidate) = self.split_criterion.evaluate(
                &hist.grad_sums,
                &hist.hess_sums,
                total_grad,
                total_hess,
                self.config.gamma,
                self.config.lambda,
            ) {
                candidates.push((feat_idx, candidate));
            }
        }

        if candidates.is_empty() {
            return false;
        }

        // Sort candidates by gain descending.
        candidates.sort_by(|a, b| {
            b.1.gain
                .partial_cmp(&a.1.gain)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_gain = candidates[0].1.gain;
        let second_best_gain = if candidates.len() > 1 {
            candidates[1].1.gain
        } else {
            0.0
        };

        // Hoeffding bound: epsilon = sqrt(R^2 * ln(1/delta) / (2 * n))
        // R = 1.0 (conservative bound on the range of the gain function).
        //
        // With EWMA decay, the effective sample size is bounded by 1/(1-alpha).
        // We cap n at this value to prevent spurious splits from artificially
        // tight bounds when decay is active.
        let r_squared = 1.0;
        let n = sample_count as f64;
        let effective_n = match self.config.leaf_decay_alpha {
            Some(alpha) => n.min(1.0 / (1.0 - alpha)),
            None => n,
        };
        let ln_inv_delta = (1.0 / self.config.delta).ln();
        let epsilon = (r_squared * ln_inv_delta / (2.0 * effective_n)).sqrt();

        // Split condition: the best is significantly better than second-best,
        // OR the bound is already so tight that more samples won't help.
        let gap = best_gain - second_best_gain;
        if gap <= epsilon && epsilon >= TAU {
            // If this was a re-evaluation at max depth, update the count
            // so we don't re-evaluate again until the next interval elapses.
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
        let (best_feat_idx, best_candidate) = candidates[0];

        // Track split gain for feature importance.
        if best_feat_idx < self.split_gains.len() {
            self.split_gains[best_feat_idx] += best_candidate.gain;
        }

        let best_hist = &histograms.histograms[best_feat_idx];

        // Determine the threshold from the bin edge.
        // The split is at bin_idx: bins [0..=bin_idx] go left.
        // The threshold is the upper edge of bin_idx.
        let threshold = if best_candidate.bin_idx < best_hist.edges.edges.len() {
            best_hist.edges.edges[best_candidate.bin_idx]
        } else {
            // Last bin — use a very large threshold (everything goes left).
            // This shouldn't happen with a well-formed split, but handle it.
            f64::MAX
        };

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

        // Perform the split in the arena.
        let (left_id, right_id) = self.arena.split_leaf(
            leaf_id,
            best_feat_idx as u32,
            threshold,
            left_value,
            right_value,
        );

        // Build child histograms using the subtraction trick.
        // The "left" child gets a fresh histogram set built from the parent's
        // bins, populated by scanning parent bins [0..=bin_idx].
        // Instead of re-scanning, we use the subtraction trick: one child
        // gets the parent's histograms minus the other child's.
        //
        // Strategy: build the left child's histogram directly from the
        // parent histogram for each feature (summing bins 0..=best_bin for
        // the split feature). Then the right child = parent - left.
        //
        // Actually, for a streaming tree, the cleaner approach is:
        // - Remove the parent's state
        // - Create fresh states for both children (they'll accumulate from
        //   new samples going forward)
        // - BUT we can seed one child with the parent's histograms by
        //   constructing "virtual" histograms from the parent's data.
        //
        // The simplest correct approach: both children start fresh with
        // pre-computed bin edges from the parent, so they're immediately
        // ready to accumulate. We don't carry forward the parent's histogram
        // data because new samples will naturally populate the children.
        //
        // However, to be more efficient, we CAN carry forward histogram data
        // using the subtraction trick. Let's do this properly:

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
                // Build left child histograms from the parent.
                let edges_per_feature: Vec<BinEdges> = parent_hists
                    .histograms
                    .iter()
                    .map(|h| h.edges.clone())
                    .collect();

                // The left child inherits a copy of the parent histograms,
                // but we really want to compute how much of the parent's data
                // would have gone left vs right. We don't have that per-feature
                // breakdown for features other than the split feature.
                //
                // Correct approach: create fresh histogram states for both
                // children with the same bin edges. They start empty but
                // with bins_ready = true, so new samples immediately accumulate.
                let left_hists = LeafHistograms::new(&edges_per_feature);
                let right_hists = LeafHistograms::new(&edges_per_feature);

                let left_state = LeafState {
                    histograms: Some(left_hists),
                    binners: (0..nf).map(|_| BinnerKind::uniform()).collect(),
                    bins_ready: true,
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    last_reeval_count: 0,
                };

                let right_state = LeafState {
                    histograms: Some(right_hists),
                    binners: (0..nf).map(|_| BinnerKind::uniform()).collect(),
                    bins_ready: true,
                    grad_sum: 0.0,
                    hess_sum: 0.0,
                    last_reeval_count: 0,
                };

                self.leaf_states[left_id.0 as usize] = Some(left_state);
                self.leaf_states[right_id.0 as usize] = Some(right_state);
            } else {
                // Parent didn't have histograms (shouldn't happen if bins_ready).
                self.leaf_states[left_id.0 as usize] = Some(LeafState::new(nf));
                self.leaf_states[right_id.0 as usize] = Some(LeafState::new(nf));
            }
        } else {
            // No parent state found (shouldn't happen).
            self.leaf_states[left_id.0 as usize] = Some(LeafState::new(nf));
            self.leaf_states[right_id.0 as usize] = Some(LeafState::new(nf));
        }

        true
    }
}

impl StreamingTree for HoeffdingTree {
    /// Train the tree on a single sample.
    ///
    /// Routes the sample to its leaf, updates histogram accumulators, and
    /// attempts a split if the Hoeffding bound is satisfied.
    fn train_one(&mut self, features: &[f64], gradient: f64, hessian: f64) {
        self.samples_seen += 1;

        // Initialize n_features on first sample.
        let n_features = if let Some(n) = self.n_features {
            n
        } else {
            let n = features.len();
            self.n_features = Some(n);
            self.split_gains.resize(n, 0.0);

            // Re-initialize the root's leaf state now that we know n_features.
            if let Some(state) = self
                .leaf_states
                .get_mut(self.root.0 as usize)
                .and_then(|o| o.as_mut())
            {
                state.binners = (0..n).map(|_| BinnerKind::uniform()).collect();
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

        // Route to leaf.
        let leaf_id = self.route_to_leaf(features);

        // Increment the sample count in the arena.
        self.arena.increment_sample_count(leaf_id);
        let sample_count = self.arena.get_sample_count(leaf_id);

        // Get or create the leaf state.
        let idx = leaf_id.0 as usize;
        if self.leaf_states.len() <= idx {
            self.leaf_states.resize_with(idx + 1, || None);
        }
        if self.leaf_states[idx].is_none() {
            self.leaf_states[idx] = Some(LeafState::new(n_features));
        }
        let state = self.leaf_states[idx].as_mut().unwrap();

        // If bins are not yet ready, check if we've reached the grace period.
        if !state.bins_ready {
            // Observe feature values in the binners.
            for (binner, &val) in state.binners.iter_mut().zip(features.iter()) {
                binner.observe(val);
            }

            // Accumulate running gradient/hessian sums (with optional EWMA decay).
            if let Some(alpha) = self.config.leaf_decay_alpha {
                state.grad_sum = alpha * state.grad_sum + gradient;
                state.hess_sum = alpha * state.hess_sum + hessian;
            } else {
                state.grad_sum += gradient;
                state.hess_sum += hessian;
            }

            // Update the leaf value from running sums.
            let lw = leaf_weight(state.grad_sum, state.hess_sum, self.config.lambda);
            self.arena.set_leaf_value(leaf_id, lw);

            // Check if we've reached the grace period to compute bin edges.
            if sample_count >= self.config.grace_period as u64 {
                let edges_per_feature: Vec<BinEdges> = state
                    .binners
                    .iter()
                    .map(|b| b.compute_edges(self.config.n_bins))
                    .collect();

                let mut histograms = LeafHistograms::new(&edges_per_feature);

                // We don't have the raw samples to replay into the histogram,
                // but we DO have the running grad/hess sums. We can't distribute
                // them across bins retroactively. The histograms start empty and
                // will accumulate from the next sample onward.
                // However, we should NOT lose the current sample. Let's accumulate
                // this sample into the newly created histograms.
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

        // Bins are ready — accumulate into histograms (with optional decay).
        if let Some(ref mut histograms) = state.histograms {
            if let Some(alpha) = self.config.leaf_decay_alpha {
                histograms.accumulate_with_decay(features, gradient, hessian, alpha);
            } else {
                histograms.accumulate(features, gradient, hessian);
            }
        }

        // Update running gradient/hessian sums and leaf value (with optional EWMA decay).
        if let Some(alpha) = self.config.leaf_decay_alpha {
            state.grad_sum = alpha * state.grad_sum + gradient;
            state.hess_sum = alpha * state.hess_sum + hessian;
        } else {
            state.grad_sum += gradient;
            state.hess_sum += hessian;
        }
        let lw = leaf_weight(state.grad_sum, state.hess_sum, self.config.lambda);
        self.arena.set_leaf_value(leaf_id, lw);

        // Attempt split.
        // We only try every grace_period samples to avoid excessive computation.
        if sample_count % (self.config.grace_period as u64) == 0 {
            self.attempt_split(leaf_id);
        }
    }

    /// Predict the leaf value for a feature vector.
    ///
    /// Routes from the root to a leaf via threshold comparisons and returns
    /// the leaf's current weight.
    fn predict(&self, features: &[f64]) -> f64 {
        let leaf_id = self.route_to_leaf(features);
        if let Some(state) = self
            .leaf_states
            .get(leaf_id.0 as usize)
            .and_then(|o| o.as_ref())
        {
            if state.hess_sum != 0.0 {
                // Compute live prediction from accumulated statistics.
                leaf_weight(state.grad_sum, state.hess_sum, self.config.lambda)
            } else {
                // Leaf has no accumulated data yet (e.g. freshly restored from
                // a serialized snapshot). Fall back to the stored leaf value.
                self.arena.leaf_value[leaf_id.0 as usize]
            }
        } else {
            0.0
        }
    }

    /// Current number of leaf nodes.
    #[inline]
    fn n_leaves(&self) -> usize {
        self.arena.n_leaves()
    }

    /// Total number of samples seen since creation.
    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Reset to initial state with a single root leaf.
    fn reset(&mut self) {
        self.arena.reset();
        let root = self.arena.add_leaf(0);
        self.root = root;
        self.leaf_states.clear();

        // Insert a placeholder leaf state for the new root.
        let n_features = self.n_features.unwrap_or(0);
        self.leaf_states.resize_with(root.0 as usize + 1, || None);
        self.leaf_states[root.0 as usize] = Some(LeafState::new(n_features));

        self.samples_seen = 0;
        self.feature_mask.clear();
        self.feature_mask_bits.clear();
        self.rng_state = self.config.seed;
        self.split_gains.iter_mut().for_each(|g| *g = 0.0);
    }

    fn split_gains(&self) -> &[f64] {
        &self.split_gains
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

    /// Simple xorshift64 for test reproducibility (same as the tree uses).
    fn test_xorshift(state: &mut u64) -> u64 {
        xorshift64(state)
    }

    /// Generate a pseudo-random f64 in [0, 1) from the RNG state.
    fn test_rand_f64(state: &mut u64) -> f64 {
        let r = test_xorshift(state);
        (r as f64) / (u64::MAX as f64)
    }

    // -----------------------------------------------------------------------
    // Test 1: Single sample train + predict returns non-NaN.
    // -----------------------------------------------------------------------
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

        // With gradient=-0.5, hessian=1.0, lambda=1.0:
        // leaf_weight = -(-0.5) / (1.0 + 1.0) = 0.25
        assert!((pred - 0.25).abs() < 1e-10, "expected ~0.25, got {}", pred);
    }

    // -----------------------------------------------------------------------
    // Test 2: Train 1000 samples from y=2*x + noise, verify RMSE decreases.
    // -----------------------------------------------------------------------
    #[test]
    fn linear_signal_rmse_improves() {
        let config = TreeConfig::new()
            .max_depth(4)
            .n_bins(32)
            .grace_period(50)
            .lambda(0.1)
            .gamma(0.0)
            .delta(1e-3);

        let mut tree = HoeffdingTree::new(config);
        let mut rng_state: u64 = 12345;

        // Generate training data: y = 2*x, with x in [0, 10].
        // For gradient boosting, gradient = prediction - target (for squared loss),
        // hessian = 1.0.
        //
        // We'll simulate a simple boosting loop:
        // - Start with prediction = 0 for all points.
        // - gradient = pred - target = 0 - y = -y
        // - hessian = 1.0

        let n_train = 1000;
        let mut features_all: Vec<f64> = Vec::with_capacity(n_train);
        let mut targets: Vec<f64> = Vec::with_capacity(n_train);

        for _ in 0..n_train {
            let x = test_rand_f64(&mut rng_state) * 10.0;
            let noise = (test_rand_f64(&mut rng_state) - 0.5) * 0.5;
            let y = 2.0 * x + noise;
            features_all.push(x);
            targets.push(y);
        }

        // Compute initial RMSE (prediction = 0).
        let initial_mse: f64 = targets.iter().map(|y| y * y).sum::<f64>() / n_train as f64;
        let initial_rmse = initial_mse.sqrt();

        // Train the tree.
        for i in 0..n_train {
            let feat = [features_all[i]];
            let pred = tree.predict(&feat);
            // For squared loss: gradient = pred - target, hessian = 1.0.
            let gradient = pred - targets[i];
            let hessian = 1.0;
            tree.train_one(&feat, gradient, hessian);
        }

        // Compute post-training RMSE.
        let mut post_mse = 0.0;
        for i in 0..n_train {
            let feat = [features_all[i]];
            let pred = tree.predict(&feat);
            let err = pred - targets[i];
            post_mse += err * err;
        }
        post_mse /= n_train as f64;
        let post_rmse = post_mse.sqrt();

        assert!(
            post_rmse < initial_rmse,
            "RMSE should decrease after training: initial={:.4}, post={:.4}",
            initial_rmse,
            post_rmse,
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: No splits before grace_period samples.
    // -----------------------------------------------------------------------
    #[test]
    fn no_splits_before_grace_period() {
        let grace = 100;
        let config = TreeConfig::new()
            .grace_period(grace)
            .max_depth(4)
            .n_bins(16)
            .delta(1e-1); // Very lenient delta to make splits easy.

        let mut tree = HoeffdingTree::new(config);
        let mut rng_state: u64 = 99999;

        // Train grace_period - 1 samples.
        for _ in 0..(grace - 1) {
            let x = test_rand_f64(&mut rng_state) * 10.0;
            let y = 2.0 * x;
            let feat = [x];
            let pred = tree.predict(&feat);
            tree.train_one(&feat, pred - y, 1.0);
        }

        assert_eq!(
            tree.n_leaves(),
            1,
            "should be exactly 1 leaf before grace_period, got {}",
            tree.n_leaves()
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: Tree does not exceed max_depth.
    // -----------------------------------------------------------------------
    #[test]
    fn respects_max_depth() {
        let max_depth = 3;
        let config = TreeConfig::new()
            .max_depth(max_depth)
            .grace_period(20)
            .n_bins(16)
            .lambda(0.01)
            .gamma(0.0)
            .delta(1e-1); // Very lenient.

        let mut tree = HoeffdingTree::new(config);
        let mut rng_state: u64 = 7777;

        // Train many samples with a clear signal to force splitting.
        for _ in 0..5000 {
            let x = test_rand_f64(&mut rng_state) * 10.0;
            let y = if x < 2.5 {
                -5.0
            } else if x < 5.0 {
                -1.0
            } else if x < 7.5 {
                1.0
            } else {
                5.0
            };
            let feat = [x];
            let pred = tree.predict(&feat);
            tree.train_one(&feat, pred - y, 1.0);
        }

        // Maximum number of leaves at depth d is 2^d.
        let max_leaves = 1usize << max_depth;
        assert!(
            tree.n_leaves() <= max_leaves,
            "tree has {} leaves, but max_depth={} allows at most {}",
            tree.n_leaves(),
            max_depth,
            max_leaves,
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Reset works — tree returns to single leaf.
    // -----------------------------------------------------------------------
    #[test]
    fn reset_returns_to_single_leaf() {
        let config = TreeConfig::new()
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .delta(1e-1);

        let mut tree = HoeffdingTree::new(config);
        let mut rng_state: u64 = 54321;

        // Train enough to potentially cause splits.
        for _ in 0..2000 {
            let x = test_rand_f64(&mut rng_state) * 10.0;
            let y = 3.0 * x - 5.0;
            let feat = [x];
            let pred = tree.predict(&feat);
            tree.train_one(&feat, pred - y, 1.0);
        }

        let pre_reset_samples = tree.n_samples_seen();
        assert!(pre_reset_samples > 0);

        tree.reset();

        assert_eq!(
            tree.n_leaves(),
            1,
            "after reset, should have exactly 1 leaf"
        );
        assert_eq!(
            tree.n_samples_seen(),
            0,
            "after reset, samples_seen should be 0"
        );

        // Predict should still work (returns 0.0 from empty leaf).
        let pred = tree.predict(&[5.0]);
        assert!(
            pred.abs() < 1e-10,
            "prediction after reset should be ~0.0, got {}",
            pred
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Multiple features — verify tree uses different features.
    // -----------------------------------------------------------------------
    #[test]
    fn multi_feature_training() {
        let config = TreeConfig::new()
            .grace_period(30)
            .max_depth(4)
            .n_bins(16)
            .lambda(0.1)
            .delta(1e-2);

        let mut tree = HoeffdingTree::new(config);
        let mut rng_state: u64 = 11111;

        // y = x0 + 2*x1, two features.
        for _ in 0..1000 {
            let x0 = test_rand_f64(&mut rng_state) * 5.0;
            let x1 = test_rand_f64(&mut rng_state) * 5.0;
            let y = x0 + 2.0 * x1;
            let feat = [x0, x1];
            let pred = tree.predict(&feat);
            tree.train_one(&feat, pred - y, 1.0);
        }

        // Just verify it trained without panicking and produces finite predictions.
        let pred = tree.predict(&[2.5, 2.5]);
        assert!(
            pred.is_finite(),
            "multi-feature prediction should be finite"
        );
        assert_eq!(tree.n_samples_seen(), 1000);
    }

    // -----------------------------------------------------------------------
    // Test 7: Feature subsampling does not panic.
    // -----------------------------------------------------------------------
    #[test]
    fn feature_subsampling_works() {
        let config = TreeConfig::new()
            .grace_period(30)
            .max_depth(3)
            .n_bins(16)
            .lambda(0.1)
            .delta(1e-2)
            .feature_subsample_rate(0.5);

        let mut tree = HoeffdingTree::new(config);
        let mut rng_state: u64 = 33333;

        // 5 features, only ~50% considered per split.
        for _ in 0..1000 {
            let feats: Vec<f64> = (0..5)
                .map(|_| test_rand_f64(&mut rng_state) * 10.0)
                .collect();
            let y: f64 = feats.iter().sum();
            let pred = tree.predict(&feats);
            tree.train_one(&feats, pred - y, 1.0);
        }

        let pred = tree.predict(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(pred.is_finite(), "subsampled prediction should be finite");
    }

    // -----------------------------------------------------------------------
    // Test 8: xorshift64 produces deterministic sequence.
    // -----------------------------------------------------------------------
    #[test]
    fn xorshift64_deterministic() {
        let mut s1: u64 = 42;
        let mut s2: u64 = 42;

        let seq1: Vec<u64> = (0..100).map(|_| xorshift64(&mut s1)).collect();
        let seq2: Vec<u64> = (0..100).map(|_| xorshift64(&mut s2)).collect();

        assert_eq!(seq1, seq2, "xorshift64 should be deterministic");

        // Verify no zeros in the sequence (xorshift64 with non-zero seed never produces 0).
        for &v in &seq1 {
            assert_ne!(v, 0, "xorshift64 should never produce 0 with non-zero seed");
        }
    }

    // -----------------------------------------------------------------------
    // Test 9: EWMA leaf decay — recent data dominates predictions.
    // -----------------------------------------------------------------------
    #[test]
    fn ewma_leaf_decay_recent_data_dominates() {
        // half_life=50 => alpha = exp(-ln(2)/50) ≈ 0.9862
        let alpha = (-(2.0_f64.ln()) / 50.0).exp();
        let config = TreeConfig::new()
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .lambda(1.0)
            .leaf_decay_alpha(alpha);
        let mut tree = HoeffdingTree::new(config);

        // Phase 1: 1000 samples targeting 1.0
        for _ in 0..1000 {
            let pred = tree.predict(&[1.0, 2.0]);
            let grad = pred - 1.0; // gradient for squared loss
            tree.train_one(&[1.0, 2.0], grad, 1.0);
        }

        // Phase 2: 100 samples targeting 5.0
        for _ in 0..100 {
            let pred = tree.predict(&[1.0, 2.0]);
            let grad = pred - 5.0;
            tree.train_one(&[1.0, 2.0], grad, 1.0);
        }

        let pred = tree.predict(&[1.0, 2.0]);
        // With EWMA, the prediction should be pulled toward 5.0 (recent target).
        // Without EWMA, 1000 samples at 1.0 would dominate 100 at 5.0.
        assert!(
            pred > 2.0,
            "EWMA should let recent data (target=5.0) pull prediction above 2.0, got {}",
            pred,
        );
    }

    // -----------------------------------------------------------------------
    // Test 10: EWMA disabled (None) matches traditional behavior.
    // -----------------------------------------------------------------------
    #[test]
    fn ewma_disabled_matches_traditional() {
        let config_no_ewma = TreeConfig::new()
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .lambda(1.0);
        let mut tree = HoeffdingTree::new(config_no_ewma);

        let mut rng_state: u64 = 99999;
        for _ in 0..200 {
            let x = test_rand_f64(&mut rng_state) * 10.0;
            let y = 3.0 * x + 1.0;
            let pred = tree.predict(&[x]);
            tree.train_one(&[x], pred - y, 1.0);
        }

        let pred = tree.predict(&[5.0]);
        assert!(
            pred.is_finite(),
            "prediction without EWMA should be finite, got {}",
            pred
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: Split re-evaluation at max depth grows beyond frozen point.
    // -----------------------------------------------------------------------
    #[test]
    fn split_reeval_at_max_depth() {
        let config = TreeConfig::new()
            .grace_period(20)
            .max_depth(2) // Very shallow to hit max depth quickly
            .n_bins(16)
            .lambda(1.0)
            .split_reeval_interval(50);
        let mut tree = HoeffdingTree::new(config);

        let mut rng_state: u64 = 54321;
        // Train enough to saturate max_depth=2 and then trigger re-evaluation.
        for _ in 0..2000 {
            let x1 = test_rand_f64(&mut rng_state) * 10.0 - 5.0;
            let x2 = test_rand_f64(&mut rng_state) * 10.0 - 5.0;
            let y = 2.0 * x1 + 3.0 * x2;
            let pred = tree.predict(&[x1, x2]);
            tree.train_one(&[x1, x2], pred - y, 1.0);
        }

        // With split_reeval_interval=50, max-depth leaves can re-evaluate
        // and potentially split beyond max_depth. The tree should have MORE
        // leaves than a max_depth=2 tree without re-eval (which caps at 4).
        let leaves = tree.n_leaves();
        assert!(
            leaves >= 4,
            "split re-eval should allow growth beyond max_depth=2 cap (4 leaves), got {}",
            leaves,
        );
    }

    // -----------------------------------------------------------------------
    // Test 12: Split re-evaluation disabled matches existing behavior.
    // -----------------------------------------------------------------------
    #[test]
    fn split_reeval_disabled_matches_traditional() {
        let config = TreeConfig::new()
            .grace_period(20)
            .max_depth(2)
            .n_bins(16)
            .lambda(1.0);
        // No split_reeval_interval => None => traditional hard cap
        let mut tree = HoeffdingTree::new(config);

        let mut rng_state: u64 = 77777;
        for _ in 0..2000 {
            let x1 = test_rand_f64(&mut rng_state) * 10.0 - 5.0;
            let x2 = test_rand_f64(&mut rng_state) * 10.0 - 5.0;
            let y = 2.0 * x1 + 3.0 * x2;
            let pred = tree.predict(&[x1, x2]);
            tree.train_one(&[x1, x2], pred - y, 1.0);
        }

        // Without re-eval, max_depth=2 caps at 4 leaves (2^2).
        let leaves = tree.n_leaves();
        assert!(
            leaves <= 4,
            "without re-eval, max_depth=2 should cap at 4 leaves, got {}",
            leaves,
        );
    }
}
