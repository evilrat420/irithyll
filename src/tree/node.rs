//! SoA arena-allocated node storage for streaming trees.
//!
//! Instead of per-node heap allocations, all node fields are stored in parallel
//! vectors indexed by [`NodeId`]. This layout is cache-friendly for batch
//! traversal and avoids pointer-chasing overhead inherent to linked structures.
//!
//! # Layout
//!
//! Every node occupies the same index across all vectors. Internal nodes use
//! `feature_idx`, `threshold`, `left`, and `right`. Leaf nodes use `leaf_value`.
//! The `is_leaf` flag discriminates between the two.

/// Index into the [`TreeArena`].
///
/// A thin wrapper around `u32`. The sentinel value [`NodeId::NONE`] (backed by
/// `u32::MAX`) represents an absent or unset node reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Sentinel value indicating "no node".
    pub const NONE: NodeId = NodeId(u32::MAX);

    /// Returns `true` if this is the sentinel [`NONE`](Self::NONE) value.
    #[inline]
    pub fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    /// Convert to `usize` for indexing into the arena vectors.
    #[inline]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

/// Structure-of-Arrays arena storage for tree nodes.
///
/// All node properties are stored in parallel vectors for cache efficiency.
/// Internal nodes have `feature_idx` + `threshold` + children.
/// Leaf nodes have `leaf_value`.
///
/// # Invariants
///
/// - All vectors have the same length at all times.
/// - A leaf node has `is_leaf[id] == true`, `left[id] == NONE`, `right[id] == NONE`.
/// - An internal node has `is_leaf[id] == false` and valid `left`/`right` children.
/// - `depth[id]` is set at allocation time and never changes.
#[derive(Debug, Clone)]
pub struct TreeArena {
    /// Feature index used for splitting (only meaningful for internal nodes).
    pub feature_idx: Vec<u32>,
    /// Split threshold (samples with feature <= threshold go left).
    pub threshold: Vec<f64>,
    /// Left child [`NodeId`] (`NONE` for leaves).
    pub left: Vec<NodeId>,
    /// Right child [`NodeId`] (`NONE` for leaves).
    pub right: Vec<NodeId>,
    /// Leaf prediction value.
    pub leaf_value: Vec<f64>,
    /// Whether this node is a leaf.
    pub is_leaf: Vec<bool>,
    /// Depth of this node in the tree (root = 0).
    pub depth: Vec<u16>,
    /// Number of samples routed through this node.
    pub sample_count: Vec<u64>,
}

impl TreeArena {
    /// Create an empty arena with no pre-allocated capacity.
    pub fn new() -> Self {
        Self {
            feature_idx: Vec::new(),
            threshold: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            leaf_value: Vec::new(),
            is_leaf: Vec::new(),
            depth: Vec::new(),
            sample_count: Vec::new(),
        }
    }

    /// Create an empty arena with pre-allocated capacity for `cap` nodes.
    ///
    /// This avoids reallocation when the maximum tree size is known upfront.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            feature_idx: Vec::with_capacity(cap),
            threshold: Vec::with_capacity(cap),
            left: Vec::with_capacity(cap),
            right: Vec::with_capacity(cap),
            leaf_value: Vec::with_capacity(cap),
            is_leaf: Vec::with_capacity(cap),
            depth: Vec::with_capacity(cap),
            sample_count: Vec::with_capacity(cap),
        }
    }

    /// Allocate a new leaf node with value 0.0 at the given depth.
    ///
    /// Returns the [`NodeId`] of the newly created leaf.
    pub fn add_leaf(&mut self, depth: u16) -> NodeId {
        let id = self.feature_idx.len() as u32;
        self.feature_idx.push(0);
        self.threshold.push(0.0);
        self.left.push(NodeId::NONE);
        self.right.push(NodeId::NONE);
        self.leaf_value.push(0.0);
        self.is_leaf.push(true);
        self.depth.push(depth);
        self.sample_count.push(0);
        NodeId(id)
    }

    /// Convert a leaf node into an internal (split) node, creating two new leaf
    /// children.
    ///
    /// The parent's `is_leaf` flag is cleared and its `feature_idx`/`threshold`
    /// are set. Two fresh leaf nodes are allocated at `depth = parent.depth + 1`
    /// with the specified initial values.
    ///
    /// # Panics
    ///
    /// Panics if `leaf_id` does not reference a current leaf node.
    ///
    /// # Returns
    ///
    /// `(left_id, right_id)` — the [`NodeId`]s of the two new children.
    pub fn split_leaf(
        &mut self,
        leaf_id: NodeId,
        feature_idx: u32,
        threshold: f64,
        left_value: f64,
        right_value: f64,
    ) -> (NodeId, NodeId) {
        let i = leaf_id.idx();
        assert!(
            self.is_leaf[i],
            "split_leaf called on non-leaf node {:?}",
            leaf_id
        );

        let child_depth = self.depth[i] + 1;

        // Allocate left child.
        let left_id = self.add_leaf(child_depth);
        self.leaf_value[left_id.idx()] = left_value;

        // Allocate right child.
        let right_id = self.add_leaf(child_depth);
        self.leaf_value[right_id.idx()] = right_value;

        // Convert the parent from a leaf into an internal node.
        self.is_leaf[i] = false;
        self.feature_idx[i] = feature_idx;
        self.threshold[i] = threshold;
        self.left[i] = left_id;
        self.right[i] = right_id;

        (left_id, right_id)
    }

    /// Returns `true` if the node at `id` is a leaf.
    #[inline]
    pub fn is_leaf(&self, id: NodeId) -> bool {
        self.is_leaf[id.idx()]
    }

    /// Return the leaf prediction value.
    ///
    /// # Panics
    ///
    /// Panics if `id` references an internal (non-leaf) node.
    #[inline]
    pub fn predict(&self, id: NodeId) -> f64 {
        let i = id.idx();
        assert!(self.is_leaf[i], "predict called on internal node {:?}", id);
        self.leaf_value[i]
    }

    /// Set the leaf prediction value for a leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `id` does not reference a leaf node.
    #[inline]
    pub fn set_leaf_value(&mut self, id: NodeId, value: f64) {
        let i = id.idx();
        assert!(
            self.is_leaf[i],
            "set_leaf_value called on internal node {:?}",
            id
        );
        self.leaf_value[i] = value;
    }

    /// Return the depth of the node.
    #[inline]
    pub fn get_depth(&self, id: NodeId) -> u16 {
        self.depth[id.idx()]
    }

    /// Return the feature index used for splitting at this node.
    #[inline]
    pub fn get_feature_idx(&self, id: NodeId) -> u32 {
        self.feature_idx[id.idx()]
    }

    /// Return the split threshold for this node.
    #[inline]
    pub fn get_threshold(&self, id: NodeId) -> f64 {
        self.threshold[id.idx()]
    }

    /// Return the left child [`NodeId`].
    #[inline]
    pub fn get_left(&self, id: NodeId) -> NodeId {
        self.left[id.idx()]
    }

    /// Return the right child [`NodeId`].
    #[inline]
    pub fn get_right(&self, id: NodeId) -> NodeId {
        self.right[id.idx()]
    }

    /// Return the sample count for this node.
    #[inline]
    pub fn get_sample_count(&self, id: NodeId) -> u64 {
        self.sample_count[id.idx()]
    }

    /// Increment the sample count for this node by one.
    #[inline]
    pub fn increment_sample_count(&mut self, id: NodeId) {
        self.sample_count[id.idx()] += 1;
    }

    /// Total number of allocated nodes (internal + leaf).
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.is_leaf.len()
    }

    /// Count of leaf nodes currently in the arena.
    pub fn n_leaves(&self) -> usize {
        self.is_leaf.iter().filter(|&&b| b).count()
    }

    /// Clear all storage, returning the arena to an empty state.
    ///
    /// Allocated memory is retained (capacity is preserved) so the arena can
    /// be reused without reallocating.
    pub fn reset(&mut self) {
        self.feature_idx.clear();
        self.threshold.clear();
        self.left.clear();
        self.right.clear();
        self.leaf_value.clear();
        self.is_leaf.clear();
        self.depth.clear();
        self.sample_count.clear();
    }
}

impl Default for TreeArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A freshly allocated leaf should be marked as a leaf with value 0.0,
    /// depth as requested, zero sample count, and NONE children.
    #[test]
    fn single_leaf() {
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);

        assert_eq!(root, NodeId(0));
        assert!(arena.is_leaf(root));
        assert_eq!(arena.predict(root), 0.0);
        assert_eq!(arena.get_depth(root), 0);
        assert_eq!(arena.get_sample_count(root), 0);
        assert_eq!(arena.get_left(root), NodeId::NONE);
        assert_eq!(arena.get_right(root), NodeId::NONE);
    }

    /// Splitting a leaf should convert the parent to an internal node and
    /// create two leaf children at depth + 1 with the specified values.
    #[test]
    fn split_leaf_basic() {
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);

        let (left, right) = arena.split_leaf(root, 3, 1.5, -0.25, 0.75);

        // Parent is no longer a leaf.
        assert!(!arena.is_leaf(root));
        assert_eq!(arena.get_feature_idx(root), 3);
        assert_eq!(arena.get_threshold(root), 1.5);
        assert_eq!(arena.get_left(root), left);
        assert_eq!(arena.get_right(root), right);

        // Children are leaves at depth 1 with the specified values.
        assert!(arena.is_leaf(left));
        assert_eq!(arena.predict(left), -0.25);
        assert_eq!(arena.get_depth(left), 1);

        assert!(arena.is_leaf(right));
        assert_eq!(arena.predict(right), 0.75);
        assert_eq!(arena.get_depth(right), 1);
    }

    /// Splitting a child node should grow the tree to three levels (depths 0, 1, 2).
    #[test]
    fn split_child_three_levels() {
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);

        // Level 1: split root.
        let (left, right) = arena.split_leaf(root, 0, 5.0, 0.0, 0.0);

        // Level 2: split the left child.
        let (ll, lr) = arena.split_leaf(left, 1, 2.0, -1.0, 1.0);

        // Root is internal.
        assert!(!arena.is_leaf(root));
        assert_eq!(arena.get_depth(root), 0);

        // Left child is now internal too.
        assert!(!arena.is_leaf(left));
        assert_eq!(arena.get_depth(left), 1);
        assert_eq!(arena.get_feature_idx(left), 1);
        assert_eq!(arena.get_threshold(left), 2.0);
        assert_eq!(arena.get_left(left), ll);
        assert_eq!(arena.get_right(left), lr);

        // Right child from the first split is still a leaf.
        assert!(arena.is_leaf(right));
        assert_eq!(arena.get_depth(right), 1);

        // Grandchildren are leaves at depth 2.
        assert!(arena.is_leaf(ll));
        assert_eq!(arena.get_depth(ll), 2);
        assert_eq!(arena.predict(ll), -1.0);

        assert!(arena.is_leaf(lr));
        assert_eq!(arena.get_depth(lr), 2);
        assert_eq!(arena.predict(lr), 1.0);
    }

    /// `n_nodes` and `n_leaves` should track allocations and splits correctly.
    #[test]
    fn node_and_leaf_counting() {
        let mut arena = TreeArena::new();

        // Empty arena.
        assert_eq!(arena.n_nodes(), 0);
        assert_eq!(arena.n_leaves(), 0);

        // Single root leaf.
        let root = arena.add_leaf(0);
        assert_eq!(arena.n_nodes(), 1);
        assert_eq!(arena.n_leaves(), 1);

        // Split root -> 3 nodes total (1 internal + 2 leaves).
        let (_left, right) = arena.split_leaf(root, 0, 1.0, 0.0, 0.0);
        assert_eq!(arena.n_nodes(), 3);
        assert_eq!(arena.n_leaves(), 2);

        // Split right child -> 5 nodes total (2 internal + 3 leaves).
        let _ = arena.split_leaf(right, 1, 2.0, 0.0, 0.0);
        assert_eq!(arena.n_nodes(), 5);
        assert_eq!(arena.n_leaves(), 3);
    }

    /// `NodeId::NONE` should report `is_none() == true` and should not collide
    /// with valid node indices in any reasonably-sized arena.
    #[test]
    fn node_id_none_sentinel() {
        let none = NodeId::NONE;
        assert!(none.is_none());
        assert_eq!(none.0, u32::MAX);

        let valid = NodeId(0);
        assert!(!valid.is_none());
        assert_ne!(valid, NodeId::NONE);
    }

    /// `reset` should clear all vectors, bringing node/leaf counts to zero,
    /// while preserving allocated capacity for reuse.
    #[test]
    fn reset_clears_everything() {
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);
        let _ = arena.split_leaf(root, 0, 1.0, 0.0, 0.0);

        assert_eq!(arena.n_nodes(), 3);

        arena.reset();

        assert_eq!(arena.n_nodes(), 0);
        assert_eq!(arena.n_leaves(), 0);

        // Capacity should be preserved (at least 3 from prior usage).
        assert!(arena.feature_idx.capacity() >= 3);
        assert!(arena.is_leaf.capacity() >= 3);

        // Should be able to reuse the arena normally after reset.
        let new_root = arena.add_leaf(0);
        assert_eq!(new_root, NodeId(0));
        assert_eq!(arena.n_nodes(), 1);
        assert_eq!(arena.n_leaves(), 1);
    }

    /// Sample count starts at zero and increments correctly per node.
    #[test]
    fn sample_count_tracking() {
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);

        assert_eq!(arena.get_sample_count(root), 0);

        arena.increment_sample_count(root);
        assert_eq!(arena.get_sample_count(root), 1);

        arena.increment_sample_count(root);
        arena.increment_sample_count(root);
        assert_eq!(arena.get_sample_count(root), 3);

        // Split and verify children start at zero independently.
        let (left, right) = arena.split_leaf(root, 0, 1.0, 0.0, 0.0);
        assert_eq!(arena.get_sample_count(left), 0);
        assert_eq!(arena.get_sample_count(right), 0);

        // Parent retains its count after split.
        assert_eq!(arena.get_sample_count(root), 3);

        arena.increment_sample_count(left);
        assert_eq!(arena.get_sample_count(left), 1);
        assert_eq!(arena.get_sample_count(right), 0);
    }

    /// `with_capacity` should pre-allocate without adding any nodes.
    #[test]
    fn with_capacity_preallocates() {
        let arena = TreeArena::with_capacity(64);

        // No nodes allocated yet.
        assert_eq!(arena.n_nodes(), 0);
        assert_eq!(arena.n_leaves(), 0);

        // But capacity is reserved.
        assert!(arena.feature_idx.capacity() >= 64);
        assert!(arena.threshold.capacity() >= 64);
        assert!(arena.left.capacity() >= 64);
        assert!(arena.right.capacity() >= 64);
        assert!(arena.leaf_value.capacity() >= 64);
        assert!(arena.is_leaf.capacity() >= 64);
        assert!(arena.depth.capacity() >= 64);
        assert!(arena.sample_count.capacity() >= 64);
    }

    /// `set_leaf_value` should update the prediction value of a leaf.
    #[test]
    fn set_leaf_value_updates() {
        let mut arena = TreeArena::new();
        let leaf = arena.add_leaf(0);

        assert_eq!(arena.predict(leaf), 0.0);

        arena.set_leaf_value(leaf, 42.5);
        assert_eq!(arena.predict(leaf), 42.5);

        arena.set_leaf_value(leaf, -3.25);
        assert_eq!(arena.predict(leaf), -3.25);
    }

    /// Calling `predict` on an internal node should panic.
    #[test]
    #[should_panic(expected = "predict called on internal node")]
    fn predict_panics_on_internal_node() {
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);
        let _ = arena.split_leaf(root, 0, 1.0, 0.0, 0.0);

        // root is now internal — this should panic.
        let _ = arena.predict(root);
    }

    /// Calling `set_leaf_value` on an internal node should panic.
    #[test]
    #[should_panic(expected = "set_leaf_value called on internal node")]
    fn set_leaf_value_panics_on_internal_node() {
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);
        let _ = arena.split_leaf(root, 0, 1.0, 0.0, 0.0);

        // root is now internal — this should panic.
        arena.set_leaf_value(root, 1.0);
    }

    /// Calling `split_leaf` on an internal node should panic.
    #[test]
    #[should_panic(expected = "split_leaf called on non-leaf node")]
    fn split_leaf_panics_on_internal_node() {
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);
        let _ = arena.split_leaf(root, 0, 1.0, 0.0, 0.0);

        // root is already internal — splitting again should panic.
        let _ = arena.split_leaf(root, 1, 2.0, 0.0, 0.0);
    }

    /// Default trait implementation should produce the same result as `new()`.
    #[test]
    fn default_matches_new() {
        let a = TreeArena::new();
        let b = TreeArena::default();

        assert_eq!(a.n_nodes(), b.n_nodes());
        assert_eq!(a.n_leaves(), b.n_leaves());
    }
}
