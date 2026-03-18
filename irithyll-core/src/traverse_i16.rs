//! Branch-free tree traversal for quantized i16 packed nodes.
//!
//! Pure integer hot loop — zero floating-point operations. The caller pre-quantizes
//! features into `&[i16]` once, then passes them to each tree. Leaf values are
//! accumulated as i32 and dequantized once after all trees:
//!
//! ```text
//! prediction = base_prediction + sum(leaf_i16 as i32) as f64 / leaf_scale
//! ```
//!
//! On Cortex-M0+ (no FPU), the i16 comparison is a single `cmp` instruction (~1 cycle)
//! versus software float emulation (~20-50 cycles per `fcmp`).

use crate::packed_i16::PackedNodeI16;

/// Traverse a single tree using integer-only comparisons. Zero float ops.
///
/// `features_quantized` contains pre-quantized feature values (i16).
/// The caller is responsible for quantizing: `(feature_f32 * feature_scale[i]) as i16`
///
/// Returns the quantized leaf value (i16). The caller dequantizes once:
/// `prediction = sum(leaf_values) as f64 / leaf_scale`
///
/// # Safety contract (upheld by view-layer validation)
///
/// - All node child indices are within `nodes.len()`.
/// - All feature indices are within `features_quantized.len()`.
/// - The tree is acyclic and every path terminates at a leaf.
#[inline(always)]
pub fn predict_tree_i16(nodes: &[PackedNodeI16], features_quantized: &[i16]) -> i16 {
    let mut idx = 0u32;
    loop {
        // SAFETY: All indices validated during view construction.
        let node = unsafe { nodes.get_unchecked(idx as usize) };
        if node.is_leaf() {
            return node.value;
        }
        let feat_idx = node.feature_idx() as usize;
        let feat_val = unsafe { *features_quantized.get_unchecked(feat_idx) };

        // Pure integer comparison — 1 cycle on Cortex-M0+
        let go_right = (feat_val > node.value) as u32;
        let left = node.left_child() as u32;
        let right = node.right_child() as u32;
        idx = left + go_right * right.wrapping_sub(left);
    }
}

/// Predict 4 samples through one tree simultaneously. Integer-only.
///
/// Exploits CPU out-of-order execution with 4 independent traversal states.
/// Each sample follows its own path through the tree — the CPU can overlap
/// memory loads across samples since the data dependencies are independent.
#[inline]
pub fn predict_tree_i16_x4(nodes: &[PackedNodeI16], features: [&[i16]; 4]) -> [i16; 4] {
    let mut idx = [0u32; 4];
    let mut done = [false; 4];
    let mut result = [0i16; 4];

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

/// Traverse a single tree with inline feature quantization. No allocation needed.
///
/// Instead of pre-quantizing all features into an `&[i16]` buffer, this function
/// takes raw `&[f32]` features and `&[f32]` per-feature scales, performing one
/// `f32 → i16` cast per comparison. This trades one f32 multiply per tree node
/// for zero-allocation compatibility on `no_std` targets without `alloc`.
///
/// # Safety contract (upheld by view-layer validation)
///
/// - All node child indices are within `nodes.len()`.
/// - All feature indices are within `features.len()` and `feature_scales.len()`.
/// - The tree is acyclic and every path terminates at a leaf.
#[inline(always)]
pub fn predict_tree_i16_inline(
    nodes: &[PackedNodeI16],
    features: &[f32],
    feature_scales: &[f32],
) -> i16 {
    let mut idx = 0u32;
    loop {
        // SAFETY: All indices validated during view construction.
        let node = unsafe { nodes.get_unchecked(idx as usize) };
        if node.is_leaf() {
            return node.value;
        }
        let feat_idx = node.feature_idx() as usize;
        let feat_val = unsafe { *features.get_unchecked(feat_idx) };
        let scale = unsafe { *feature_scales.get_unchecked(feat_idx) };
        let quantized = (feat_val * scale) as i16;

        // Pure integer comparison after inline quantization
        let go_right = (quantized > node.value) as u32;
        let left = node.left_child() as u32;
        let right = node.right_child() as u32;
        idx = left + go_right * right.wrapping_sub(left);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed_i16::PackedNodeI16;

    /// Build a simple tree:
    /// ```text
    ///     [0] feat=0, thr=500
    ///     /                \
    /// [1] leaf=-100    [2] leaf=100
    /// ```
    fn simple_tree() -> [PackedNodeI16; 3] {
        [
            PackedNodeI16::split(500, 0, 1, 2),
            PackedNodeI16::leaf(-100),
            PackedNodeI16::leaf(100),
        ]
    }

    /// Build a two-level tree:
    /// ```text
    ///          [0] feat=0, thr=500
    ///          /                \
    ///   [1] feat=1, thr=200   [2] leaf=1000
    ///    /           \
    /// [3] leaf=-500  [4] leaf=300
    /// ```
    fn two_level_tree() -> [PackedNodeI16; 5] {
        [
            PackedNodeI16::split(500, 0, 1, 2),
            PackedNodeI16::split(200, 1, 3, 4),
            PackedNodeI16::leaf(1000),
            PackedNodeI16::leaf(-500),
            PackedNodeI16::leaf(300),
        ]
    }

    #[test]
    fn single_leaf_tree_i16() {
        let nodes = [PackedNodeI16::leaf(4200)];
        assert_eq!(predict_tree_i16(&nodes, &[100, 200]), 4200);
    }

    #[test]
    fn simple_tree_goes_left_i16() {
        let nodes = simple_tree();
        // feat[0] = 300 <= 500 -> NOT > threshold -> go_right=0 -> left child
        // idx = left = 1 -> leaf=-100
        assert_eq!(predict_tree_i16(&nodes, &[300]), -100);
    }

    #[test]
    fn simple_tree_goes_right_i16() {
        let nodes = simple_tree();
        // feat[0] = 700 > 500 -> go_right=1 -> right child -> leaf=100
        assert_eq!(predict_tree_i16(&nodes, &[700]), 100);
    }

    #[test]
    fn simple_tree_equal_goes_left_i16() {
        let nodes = simple_tree();
        // feat[0] = 500, NOT > 500 -> go_right=0 -> left -> leaf=-100
        // Matches SGBT convention: <= threshold goes left
        assert_eq!(predict_tree_i16(&nodes, &[500]), -100);
    }

    #[test]
    fn two_level_left_left_i16() {
        let nodes = two_level_tree();
        // feat[0]=100, not > 500 -> left(1); feat[1]=50, not > 200 -> left(3) -> -500
        assert_eq!(predict_tree_i16(&nodes, &[100, 50]), -500);
    }

    #[test]
    fn two_level_left_right_i16() {
        let nodes = two_level_tree();
        // feat[0]=400, not > 500 -> left(1); feat[1]=300, > 200 -> right(4) -> 300
        assert_eq!(predict_tree_i16(&nodes, &[400, 300]), 300);
    }

    #[test]
    fn two_level_right_i16() {
        let nodes = two_level_tree();
        // feat[0]=800, > 500 -> right(2) -> leaf=1000
        assert_eq!(predict_tree_i16(&nodes, &[800, 9999]), 1000);
    }

    #[test]
    fn predict_x4_matches_single_i16() {
        let nodes = two_level_tree();
        let f0: &[i16] = &[100, 50];
        let f1: &[i16] = &[400, 300];
        let f2: &[i16] = &[800, 0];
        let f3: &[i16] = &[500, 200];

        let batch = predict_tree_i16_x4(&nodes, [f0, f1, f2, f3]);

        assert_eq!(batch[0], predict_tree_i16(&nodes, f0));
        assert_eq!(batch[1], predict_tree_i16(&nodes, f1));
        assert_eq!(batch[2], predict_tree_i16(&nodes, f2));
        assert_eq!(batch[3], predict_tree_i16(&nodes, f3));
    }

    #[test]
    fn inline_matches_prequantized() {
        let nodes = simple_tree();
        // scale = 1.0, so f32 features equal i16 features after cast
        let features_f32: &[f32] = &[300.0];
        let scales: &[f32] = &[1.0];
        let features_i16: &[i16] = &[300];

        let inline_result = predict_tree_i16_inline(&nodes, features_f32, scales);
        let preq_result = predict_tree_i16(&nodes, features_i16);
        assert_eq!(inline_result, preq_result);
    }

    #[test]
    fn inline_with_scaling() {
        let nodes = simple_tree();
        // threshold is 500. feat[0] = 5.0 with scale = 200.0 -> quantized = 1000 > 500 -> right
        let features: &[f32] = &[5.0];
        let scales: &[f32] = &[200.0];
        assert_eq!(predict_tree_i16_inline(&nodes, features, scales), 100);

        // feat[0] = 1.0 with scale = 200.0 -> quantized = 200 <= 500 -> left
        let features2: &[f32] = &[1.0];
        assert_eq!(predict_tree_i16_inline(&nodes, features2, scales), -100);
    }
}
