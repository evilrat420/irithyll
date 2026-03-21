//! Cache-optimized tree traversal for prediction.
//!
//! Standalone functions that operate on a [`TreeArena`] for traversal, prediction,
//! and structural queries. These are decoupled from [`crate::tree::StreamingTree`]
//! so they can be reused by the ensemble layer and tested independently.

use alloc::vec;
use alloc::vec::Vec;

use crate::tree::node::{NodeId, TreeArena};

/// Traverse the tree from the given root to a leaf, returning the leaf's [`NodeId`].
///
/// At each internal node the traversal goes **left** when
/// `features[feature_idx] <= threshold` and **right** otherwise.
///
/// # Panics
///
/// Panics (via bounds check) if `root` or any child index is out of range,
/// or if a feature index referenced by an internal node exceeds `features.len()`.
#[inline]
pub fn traverse_to_leaf(arena: &TreeArena, root: NodeId, features: &[f64]) -> NodeId {
    let mut current = root;
    loop {
        let idx = current.idx();
        if arena.is_leaf[idx] {
            return current;
        }
        let feat_idx = arena.feature_idx[idx] as usize;
        if features[feat_idx] <= arena.threshold[idx] {
            current = arena.left[idx];
        } else {
            current = arena.right[idx];
        }
    }
}

/// Predict the leaf value for a single feature vector, starting from `root`.
///
/// Equivalent to `arena.leaf_value[traverse_to_leaf(arena, root, features).idx()]`.
#[inline]
pub fn predict_from_root(arena: &TreeArena, root: NodeId, features: &[f64]) -> f64 {
    let leaf = traverse_to_leaf(arena, root, features);
    arena.leaf_value[leaf.idx()]
}

/// Batch prediction: compute the leaf value for each row in `feature_matrix`.
///
/// Returns a `Vec<f64>` with one prediction per row, in the same order as the
/// input matrix.
pub fn predict_batch(arena: &TreeArena, root: NodeId, feature_matrix: &[Vec<f64>]) -> Vec<f64> {
    feature_matrix
        .iter()
        .map(|features| predict_from_root(arena, root, features))
        .collect()
}

/// Collect every leaf [`NodeId`] reachable from `root` via depth-first traversal.
///
/// The returned order is deterministic (left subtree leaves appear before right
/// subtree leaves) but should not be relied upon for semantic meaning.
pub fn collect_leaves(arena: &TreeArena, root: NodeId) -> Vec<NodeId> {
    let mut leaves = Vec::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let idx = node.idx();
        if arena.is_leaf[idx] {
            leaves.push(node);
        } else {
            // Push right first so left is popped first (DFS left-to-right).
            stack.push(arena.right[idx]);
            stack.push(arena.left[idx]);
        }
    }
    leaves
}

/// Compute the maximum depth of any leaf reachable from `root`.
///
/// Returns `0` for a single-leaf tree (root is a leaf at depth 0).
/// Returns `0` if the tree is empty (no leaves found).
pub fn tree_depth(arena: &TreeArena, root: NodeId) -> u16 {
    let leaves = collect_leaves(arena, root);
    leaves
        .iter()
        .map(|id| arena.depth[id.idx()])
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::node::{NodeId, TreeArena};

    /// Helper: create an empty [`TreeArena`].
    fn empty_arena() -> TreeArena {
        TreeArena {
            feature_idx: Vec::new(),
            threshold: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            leaf_value: Vec::new(),
            is_leaf: Vec::new(),
            depth: Vec::new(),
            sample_count: Vec::new(),
            categorical_mask: Vec::new(),
        }
    }

    /// Push a leaf node into the arena, returning its [`NodeId`].
    fn push_leaf(arena: &mut TreeArena, value: f64, depth: u16) -> NodeId {
        let id = NodeId(arena.is_leaf.len() as u32);
        arena.feature_idx.push(0);
        arena.threshold.push(0.0);
        arena.left.push(NodeId::NONE);
        arena.right.push(NodeId::NONE);
        arena.leaf_value.push(value);
        arena.is_leaf.push(true);
        arena.depth.push(depth);
        arena.sample_count.push(0);
        arena.categorical_mask.push(None);
        id
    }

    /// Convert an existing leaf into an internal node by updating its fields
    /// in place. Useful for building trees top-down where the root is allocated
    /// first as a leaf and then split.
    fn convert_to_split(
        arena: &mut TreeArena,
        node: NodeId,
        feature: u32,
        threshold: f64,
        left: NodeId,
        right: NodeId,
    ) {
        let idx = node.idx();
        arena.feature_idx[idx] = feature;
        arena.threshold[idx] = threshold;
        arena.left[idx] = left;
        arena.right[idx] = right;
        arena.is_leaf[idx] = false;
    }

    // ------------------------------------------------------------------
    // 1. Single leaf tree: predict returns leaf value.
    // ------------------------------------------------------------------

    #[test]
    fn single_leaf_returns_value() {
        let mut arena = empty_arena();
        let root = push_leaf(&mut arena, 0.0, 0);

        // Any feature vector should land on the only leaf.
        assert_eq!(predict_from_root(&arena, root, &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn single_leaf_with_nonzero_value() {
        let mut arena = empty_arena();
        let root = push_leaf(&mut arena, 42.5, 0);

        assert_eq!(predict_from_root(&arena, root, &[]), 42.5);
    }

    // ------------------------------------------------------------------
    // 2. One split: verify left/right routing.
    // ------------------------------------------------------------------

    /// Build:
    /// ```text
    ///        [0] feat=0, thr=5.0
    ///        /                \
    ///   [1] leaf=-1.0    [2] leaf=+1.0
    /// ```
    fn build_one_split() -> (TreeArena, NodeId) {
        let mut arena = empty_arena();

        // Allocate root first as leaf, then children, then convert root.
        let root = push_leaf(&mut arena, 0.0, 0);
        let left_child = push_leaf(&mut arena, -1.0, 1);
        let right_child = push_leaf(&mut arena, 1.0, 1);
        convert_to_split(&mut arena, root, 0, 5.0, left_child, right_child);

        (arena, root)
    }

    #[test]
    fn one_split_goes_left() {
        let (arena, root) = build_one_split();
        // feature[0] = 3.0 <= 5.0 -> left leaf (-1.0)
        assert_eq!(predict_from_root(&arena, root, &[3.0]), -1.0);
    }

    #[test]
    fn one_split_goes_right() {
        let (arena, root) = build_one_split();
        // feature[0] = 7.0 > 5.0 -> right leaf (+1.0)
        assert_eq!(predict_from_root(&arena, root, &[7.0]), 1.0);
    }

    #[test]
    fn one_split_equal_goes_left() {
        let (arena, root) = build_one_split();
        // feature[0] = 5.0 == threshold -> left (<=)
        assert_eq!(predict_from_root(&arena, root, &[5.0]), -1.0);
    }

    // ------------------------------------------------------------------
    // 3. Two-level tree: three different feature vectors hit three leaves.
    // ------------------------------------------------------------------

    /// Build:
    /// ```text
    ///             [0] feat=0, thr=5.0
    ///            /                   \
    ///    [1] feat=1, thr=2.0     [2] leaf=10.0
    ///       /            \
    /// [3] leaf=-5.0  [4] leaf=3.0
    /// ```
    fn build_two_level() -> (TreeArena, NodeId) {
        let mut arena = empty_arena();

        let root = push_leaf(&mut arena, 0.0, 0); // id 0
        let inner = push_leaf(&mut arena, 0.0, 1); // id 1
        let right_leaf = push_leaf(&mut arena, 10.0, 1); // id 2
        let left_left = push_leaf(&mut arena, -5.0, 2); // id 3
        let left_right = push_leaf(&mut arena, 3.0, 2); // id 4

        convert_to_split(&mut arena, root, 0, 5.0, inner, right_leaf);
        convert_to_split(&mut arena, inner, 1, 2.0, left_left, left_right);

        (arena, root)
    }

    #[test]
    fn two_level_reaches_left_left() {
        let (arena, root) = build_two_level();
        // feat[0]=1.0 <= 5.0 -> go left; feat[1]=0.5 <= 2.0 -> go left => -5.0
        assert_eq!(predict_from_root(&arena, root, &[1.0, 0.5]), -5.0);
    }

    #[test]
    fn two_level_reaches_left_right() {
        let (arena, root) = build_two_level();
        // feat[0]=4.0 <= 5.0 -> go left; feat[1]=3.0 > 2.0 -> go right => 3.0
        assert_eq!(predict_from_root(&arena, root, &[4.0, 3.0]), 3.0);
    }

    #[test]
    fn two_level_reaches_right_leaf() {
        let (arena, root) = build_two_level();
        // feat[0]=8.0 > 5.0 -> go right => 10.0
        assert_eq!(predict_from_root(&arena, root, &[8.0, 999.0]), 10.0);
    }

    // ------------------------------------------------------------------
    // 4. Batch prediction: verify consistency with individual predictions.
    // ------------------------------------------------------------------

    #[test]
    fn batch_matches_individual() {
        let (arena, root) = build_two_level();

        let rows = vec![
            vec![1.0, 0.5],
            vec![4.0, 3.0],
            vec![8.0, 0.0],
            vec![5.0, 2.0], // both exactly on threshold -> left, left => -5.0
        ];

        let batch = predict_batch(&arena, root, &rows);

        for (i, row) in rows.iter().enumerate() {
            let individual = predict_from_root(&arena, root, row);
            assert_eq!(
                batch[i], individual,
                "batch[{}] = {} but individual = {} for features {:?}",
                i, batch[i], individual, row
            );
        }
    }

    #[test]
    fn batch_empty_input() {
        let (arena, root) = build_one_split();
        let result = predict_batch(&arena, root, &[]);
        assert!(result.is_empty());
    }

    // ------------------------------------------------------------------
    // 5. collect_leaves: correct leaf count for different tree shapes.
    // ------------------------------------------------------------------

    #[test]
    fn collect_leaves_single_leaf() {
        let mut arena = empty_arena();
        let root = push_leaf(&mut arena, 0.0, 0);
        let leaves = collect_leaves(&arena, root);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].idx(), root.idx());
    }

    #[test]
    fn collect_leaves_one_split() {
        let (arena, root) = build_one_split();
        let leaves = collect_leaves(&arena, root);
        assert_eq!(leaves.len(), 2);
    }

    #[test]
    fn collect_leaves_two_level() {
        let (arena, root) = build_two_level();
        let leaves = collect_leaves(&arena, root);
        // Three leaves: left-left(-5.0), left-right(3.0), right(10.0)
        assert_eq!(leaves.len(), 3);

        // Verify DFS left-to-right order by checking leaf values.
        let values: Vec<f64> = leaves.iter().map(|id| arena.leaf_value[id.idx()]).collect();
        assert_eq!(values, vec![-5.0, 3.0, 10.0]);
    }

    #[test]
    fn collect_leaves_balanced_depth2() {
        // Build a perfectly balanced tree of depth 2 with 4 leaves.
        //
        //             [0] feat=0, thr=5.0
        //            /                   \
        //    [1] feat=1, thr=2.0    [2] feat=1, thr=8.0
        //       /        \              /        \
        //  [3] 1.0    [4] 2.0     [5] 3.0    [6] 4.0
        let mut arena = empty_arena();

        let root = push_leaf(&mut arena, 0.0, 0);
        let left = push_leaf(&mut arena, 0.0, 1);
        let right = push_leaf(&mut arena, 0.0, 1);
        let ll = push_leaf(&mut arena, 1.0, 2);
        let lr = push_leaf(&mut arena, 2.0, 2);
        let rl = push_leaf(&mut arena, 3.0, 2);
        let rr = push_leaf(&mut arena, 4.0, 2);

        convert_to_split(&mut arena, root, 0, 5.0, left, right);
        convert_to_split(&mut arena, left, 1, 2.0, ll, lr);
        convert_to_split(&mut arena, right, 1, 8.0, rl, rr);

        let leaves = collect_leaves(&arena, root);
        assert_eq!(leaves.len(), 4);

        let values: Vec<f64> = leaves.iter().map(|id| arena.leaf_value[id.idx()]).collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ------------------------------------------------------------------
    // 6. tree_depth: verify for balanced and unbalanced trees.
    // ------------------------------------------------------------------

    #[test]
    fn depth_single_leaf() {
        let mut arena = empty_arena();
        let root = push_leaf(&mut arena, 0.0, 0);
        assert_eq!(tree_depth(&arena, root), 0);
    }

    #[test]
    fn depth_one_split() {
        let (arena, root) = build_one_split();
        assert_eq!(tree_depth(&arena, root), 1);
    }

    #[test]
    fn depth_two_level_unbalanced() {
        let (arena, root) = build_two_level();
        // Deepest leaf is at depth 2 (left-left and left-right).
        // Right leaf is at depth 1.
        assert_eq!(tree_depth(&arena, root), 2);
    }

    #[test]
    fn depth_left_skewed() {
        // Build a left-skewed chain of depth 4.
        //
        //  [0] split
        //  /       \
        // [1] split [2] leaf (depth=1)
        //  /    \
        // [3] split [4] leaf (depth=2)
        //  /    \
        // [5] split [6] leaf (depth=3)
        //  /    \
        // [7] leaf [8] leaf (depth=4)
        let mut arena = empty_arena();

        let n0 = push_leaf(&mut arena, 0.0, 0);
        let n1 = push_leaf(&mut arena, 0.0, 1);
        let n2 = push_leaf(&mut arena, 0.0, 1);
        let n3 = push_leaf(&mut arena, 0.0, 2);
        let n4 = push_leaf(&mut arena, 0.0, 2);
        let n5 = push_leaf(&mut arena, 0.0, 3);
        let n6 = push_leaf(&mut arena, 0.0, 3);
        let n7 = push_leaf(&mut arena, 0.0, 4);
        let n8 = push_leaf(&mut arena, 0.0, 4);

        convert_to_split(&mut arena, n0, 0, 1.0, n1, n2);
        convert_to_split(&mut arena, n1, 0, 2.0, n3, n4);
        convert_to_split(&mut arena, n3, 0, 3.0, n5, n6);
        convert_to_split(&mut arena, n5, 0, 4.0, n7, n8);

        assert_eq!(tree_depth(&arena, n0), 4);
        // 5 leaves total: n2(d=1), n4(d=2), n6(d=3), n7(d=4), n8(d=4)
        assert_eq!(collect_leaves(&arena, n0).len(), 5);
    }

    // ------------------------------------------------------------------
    // 7. Edge case: feature exactly equal to threshold goes left.
    // ------------------------------------------------------------------

    #[test]
    fn threshold_equality_goes_left() {
        let (arena, root) = build_one_split();
        let leaf = traverse_to_leaf(&arena, root, &[5.0]);
        // Left child is id 1, right child is id 2.
        assert_eq!(leaf.idx(), 1, "value == threshold must route left");
        assert_eq!(arena.leaf_value[leaf.idx()], -1.0);
    }

    #[test]
    fn threshold_equality_two_level() {
        let (arena, root) = build_two_level();
        // Both thresholds hit exactly: feat[0]=5.0 <= 5.0 -> left,
        // then feat[1]=2.0 <= 2.0 -> left => leaf at id 3, value -5.0.
        assert_eq!(predict_from_root(&arena, root, &[5.0, 2.0]), -5.0);
    }

    // ------------------------------------------------------------------
    // Extra: traverse_to_leaf returns the correct NodeId directly.
    // ------------------------------------------------------------------

    #[test]
    fn traverse_returns_correct_node_id() {
        let (arena, root) = build_two_level();

        // Route to left-left leaf (id 3).
        let leaf = traverse_to_leaf(&arena, root, &[0.0, 0.0]);
        assert_eq!(leaf.idx(), 3);

        // Route to left-right leaf (id 4).
        let leaf = traverse_to_leaf(&arena, root, &[0.0, 5.0]);
        assert_eq!(leaf.idx(), 4);

        // Route to right leaf (id 2).
        let leaf = traverse_to_leaf(&arena, root, &[10.0, 0.0]);
        assert_eq!(leaf.idx(), 2);
    }
}
