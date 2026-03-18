//! Branch-free tree traversal for packed nodes.
//!
//! The hot loop uses `get_unchecked` (bounds validated at `EnsembleView` construction)
//! and branchless child selection (`cmov` on x86, `csel` on ARM) for maximum throughput.

use crate::packed::PackedNode;

/// Traverse a single tree and return the leaf prediction value.
///
/// # Safety contract (upheld by `EnsembleView::from_bytes`)
///
/// - All node child indices are within `nodes.len()`.
/// - All feature indices are within `features.len()`.
/// - The tree is acyclic and every path terminates at a leaf.
///
/// These invariants are validated once during `EnsembleView::from_bytes()`. After
/// validation, traversal uses `get_unchecked` for zero-overhead indexing.
#[inline(always)]
pub fn predict_tree(nodes: &[PackedNode], features: &[f32]) -> f32 {
    let mut idx = 0u32;
    loop {
        // SAFETY: All indices validated during EnsembleView construction.
        let node = unsafe { nodes.get_unchecked(idx as usize) };
        if node.is_leaf() {
            return node.value;
        }
        let feat_idx = node.feature_idx() as usize;
        let feat_val = unsafe { *features.get_unchecked(feat_idx) };

        // Branchless child selection:
        // go_right = 1 if feat_val > threshold, 0 otherwise
        // idx = left + go_right * (right - left)
        // This compiles to cmov (x86) or csel (ARM) — no branch prediction miss.
        let go_right = (feat_val > node.value) as u32;
        let left = node.left_child() as u32;
        let right = node.right_child() as u32;
        idx = left + go_right * right.wrapping_sub(left);
    }
}

/// Predict 4 samples through one tree simultaneously.
///
/// Exploits CPU out-of-order execution with 4 independent traversal states.
/// Each sample follows its own path through the tree — the CPU can overlap
/// memory loads across samples since the data dependencies are independent.
#[inline]
pub fn predict_tree_x4(nodes: &[PackedNode], features: [&[f32]; 4]) -> [f32; 4] {
    let mut idx = [0u32; 4];
    let mut done = [false; 4];
    let mut result = [0.0f32; 4];

    loop {
        let mut all_done = true;
        for s in 0..4 {
            if done[s] {
                continue;
            }
            let node = unsafe { nodes.get_unchecked(idx[s] as usize) };
            if node.is_leaf() {
                result[s] = node.value;
                done[s] = true;
                continue;
            }
            all_done = false;
            let feat_idx = node.feature_idx() as usize;
            let feat_val = unsafe { *features[s].get_unchecked(feat_idx) };
            let go_right = (feat_val > node.value) as u32;
            let left = node.left_child() as u32;
            let right = node.right_child() as u32;
            idx[s] = left + go_right * right.wrapping_sub(left);
        }
        if all_done {
            return result;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::PackedNode;

    /// Build a simple tree:
    /// ```text
    ///     [0] feat=0, thr=5.0
    ///     /                \
    /// [1] leaf=-1.0    [2] leaf=1.0
    /// ```
    fn simple_tree() -> [PackedNode; 3] {
        [
            PackedNode::split(5.0, 0, 1, 2),
            PackedNode::leaf(-1.0),
            PackedNode::leaf(1.0),
        ]
    }

    /// Build a two-level tree:
    /// ```text
    ///          [0] feat=0, thr=5.0
    ///          /                \
    ///   [1] feat=1, thr=2.0   [2] leaf=10.0
    ///    /           \
    /// [3] leaf=-5.0  [4] leaf=3.0
    /// ```
    fn two_level_tree() -> [PackedNode; 5] {
        [
            PackedNode::split(5.0, 0, 1, 2),
            PackedNode::split(2.0, 1, 3, 4),
            PackedNode::leaf(10.0),
            PackedNode::leaf(-5.0),
            PackedNode::leaf(3.0),
        ]
    }

    #[test]
    fn single_leaf_tree() {
        let nodes = [PackedNode::leaf(42.0)];
        assert_eq!(predict_tree(&nodes, &[1.0, 2.0]), 42.0);
    }

    #[test]
    fn simple_tree_goes_left() {
        let nodes = simple_tree();
        // feat[0] = 3.0 <= 5.0 -> NOT > threshold -> go_right=0 -> left child
        // But wait: our comparison is `feat_val > threshold` for go_right.
        // feat[0]=3.0 > 5.0 is false -> go_right=0 -> idx = left = 1 -> leaf=-1.0
        assert_eq!(predict_tree(&nodes, &[3.0]), -1.0);
    }

    #[test]
    fn simple_tree_goes_right() {
        let nodes = simple_tree();
        // feat[0] = 7.0 > 5.0 -> go_right=1 -> right child -> leaf=1.0
        assert_eq!(predict_tree(&nodes, &[7.0]), 1.0);
    }

    #[test]
    fn simple_tree_equal_goes_left() {
        let nodes = simple_tree();
        // feat[0] = 5.0, NOT > 5.0 -> go_right=0 -> left -> leaf=-1.0
        // Note: this matches the SGBT convention (<=threshold goes left)
        assert_eq!(predict_tree(&nodes, &[5.0]), -1.0);
    }

    #[test]
    fn two_level_left_left() {
        let nodes = two_level_tree();
        // feat[0]=1.0, not > 5.0 -> left(1); feat[1]=0.5, not > 2.0 -> left(3) -> -5.0
        assert_eq!(predict_tree(&nodes, &[1.0, 0.5]), -5.0);
    }

    #[test]
    fn two_level_left_right() {
        let nodes = two_level_tree();
        // feat[0]=4.0, not > 5.0 -> left(1); feat[1]=3.0, > 2.0 -> right(4) -> 3.0
        assert_eq!(predict_tree(&nodes, &[4.0, 3.0]), 3.0);
    }

    #[test]
    fn two_level_right() {
        let nodes = two_level_tree();
        // feat[0]=8.0, > 5.0 -> right(2) -> leaf=10.0
        assert_eq!(predict_tree(&nodes, &[8.0, 999.0]), 10.0);
    }

    #[test]
    fn predict_x4_matches_single() {
        let nodes = two_level_tree();
        let f0: &[f32] = &[1.0, 0.5];
        let f1: &[f32] = &[4.0, 3.0];
        let f2: &[f32] = &[8.0, 0.0];
        let f3: &[f32] = &[5.0, 2.0];

        let batch = predict_tree_x4(&nodes, [f0, f1, f2, f3]);

        assert_eq!(batch[0], predict_tree(&nodes, f0));
        assert_eq!(batch[1], predict_tree(&nodes, f1));
        assert_eq!(batch[2], predict_tree(&nodes, f2));
        assert_eq!(batch[3], predict_tree(&nodes, f3));
    }
}
