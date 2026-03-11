//! TreeSHAP algorithm for single trees and ensembles.
//!
//! Implements the path-dependent TreeSHAP (Lundberg et al., 2020) which
//! computes exact Shapley values in O(L * D^2) per tree, where L is the
//! number of leaves and D is the tree depth.

use crate::tree::node::{NodeId, TreeArena};

/// Precompute the true cover (sum of leaf sample counts) for each node.
///
/// The arena's `sample_count` only tracks how many samples reached a node while
/// it was a leaf. After splitting, the internal node's count goes stale. This
/// function computes the correct cover by summing leaf counts bottom-up.
fn compute_covers(arena: &TreeArena, root: NodeId) -> Vec<f64> {
    let n = arena.n_nodes();
    let mut covers = vec![0.0; n];

    fn fill(arena: &TreeArena, node: NodeId, covers: &mut [f64]) -> f64 {
        let idx = node.idx();
        if arena.is_leaf[idx] {
            covers[idx] = arena.sample_count[idx] as f64;
            return covers[idx];
        }
        let left = fill(arena, arena.left[idx], covers);
        let right = fill(arena, arena.right[idx], covers);
        covers[idx] = left + right;
        covers[idx]
    }

    if !root.is_none() && root.idx() < n {
        fill(arena, root, &mut covers);
    }
    covers
}

/// Per-feature SHAP contributions for a single prediction.
///
/// # Invariant
///
/// `base_value + values.iter().sum::<f64>() ≈ model.predict(features)`
#[derive(Debug, Clone)]
pub struct ShapValues {
    /// Per-feature SHAP contribution (positive = pushes prediction up).
    pub values: Vec<f64>,
    /// Expected value of the model (mean prediction over training data).
    pub base_value: f64,
}

/// Named SHAP values with feature names attached.
#[derive(Debug, Clone)]
pub struct NamedShapValues {
    /// `(feature_name, shap_value)` pairs.
    pub values: Vec<(String, f64)>,
    /// Expected value of the model.
    pub base_value: f64,
}

// ---------------------------------------------------------------------------
// Path-dependent TreeSHAP internals
// ---------------------------------------------------------------------------

/// A single entry in the path tracking structure.
#[derive(Clone)]
struct PathEntry {
    /// Feature index at this node (-1 if root / no feature).
    feature_idx: i64,
    /// Fraction of zero-valued paths passing through this node.
    zero_fraction: f64,
    /// Fraction of one-valued paths passing through this node.
    one_fraction: f64,
    /// Proportion of training samples passing through this node.
    pweight: f64,
}

/// Extend the path by one node.
fn extend_path(path: &mut Vec<PathEntry>, zero_fraction: f64, one_fraction: f64, feature_idx: i64) {
    let depth = path.len();
    path.push(PathEntry {
        feature_idx,
        zero_fraction,
        one_fraction,
        pweight: if depth == 0 { 1.0 } else { 0.0 },
    });

    // Update path weights.
    for i in (1..depth + 1).rev() {
        path[i].pweight += one_fraction * path[i - 1].pweight * (i as f64) / ((depth + 1) as f64);
        path[i - 1].pweight =
            zero_fraction * path[i - 1].pweight * ((depth + 1 - i) as f64) / ((depth + 1) as f64);
    }
}

/// Unwind the path after returning from a subtree.
fn unwind_path(path: &mut Vec<PathEntry>, path_idx: usize) {
    let depth = path.len() - 1;
    let one_fraction = path[path_idx].one_fraction;
    let zero_fraction = path[path_idx].zero_fraction;

    let mut next_one_portion = path[depth].pweight;

    for i in (0..depth).rev() {
        if one_fraction != 0.0 {
            let tmp = path[i].pweight;
            path[i].pweight =
                next_one_portion * ((depth + 1 - i) as f64) / ((i + 1) as f64 * one_fraction);
            next_one_portion =
                tmp - path[i].pweight * zero_fraction * ((i + 1) as f64) / ((depth + 1 - i) as f64);
        } else {
            path[i].pweight =
                path[i].pweight * ((depth + 1 - i) as f64) / (zero_fraction * (i + 1) as f64);
        }
    }

    // Remove the entry at path_idx.
    for i in path_idx..depth {
        path[i] = path[i + 1].clone();
    }
    path.pop();
}

/// Compute the sum of unwound path weights for a given feature index.
fn unwound_path_sum(path: &[PathEntry], path_idx: usize) -> f64 {
    let depth = path.len() - 1;
    let one_fraction = path[path_idx].one_fraction;
    let zero_fraction = path[path_idx].zero_fraction;

    let mut total = 0.0;
    let mut next_one_portion = path[depth].pweight;

    for i in (0..depth).rev() {
        if one_fraction != 0.0 {
            let tmp = next_one_portion * ((depth + 1 - i) as f64) / ((i + 1) as f64 * one_fraction);
            total += tmp;
            next_one_portion =
                path[i].pweight - tmp * zero_fraction * ((i + 1) as f64) / ((depth + 1 - i) as f64);
        } else {
            total += path[i].pweight / (zero_fraction * (i + 1) as f64) * ((depth + 1 - i) as f64);
        }
    }
    total
}

/// Recursive TreeSHAP traversal.
fn tree_shap_recursive(
    arena: &TreeArena,
    covers: &[f64],
    node: NodeId,
    features: &[f64],
    shap_values: &mut [f64],
    path: &mut Vec<PathEntry>,
) {
    let idx = node.idx();

    if arena.is_leaf[idx] {
        // At a leaf — accumulate SHAP contributions.
        let leaf_value = arena.leaf_value[idx];
        for i in 1..path.len() {
            let w = unwound_path_sum(path, i);
            let feat = path[i].feature_idx;
            if feat >= 0 && (feat as usize) < shap_values.len() {
                shap_values[feat as usize] +=
                    w * (path[i].one_fraction - path[i].zero_fraction) * leaf_value;
            }
        }
        return;
    }

    // Internal node.
    let split_feat = arena.feature_idx[idx] as i64;
    let threshold = arena.threshold[idx];
    let left = arena.left[idx];
    let right = arena.right[idx];

    let left_cover = covers[left.idx()];
    let right_cover = covers[right.idx()];
    let node_cover = left_cover + right_cover;

    // No training data at this node — nothing to explain.
    if node_cover == 0.0 {
        return;
    }

    // Determine which child the sample goes to.
    let feat_val = if (split_feat as usize) < features.len() {
        features[split_feat as usize]
    } else {
        0.0
    };

    let (hot_child, cold_child, hot_cover, cold_cover) = if feat_val <= threshold {
        (left, right, left_cover, right_cover)
    } else {
        (right, left, right_cover, left_cover)
    };

    let hot_zero_fraction = hot_cover / node_cover;
    let cold_zero_fraction = cold_cover / node_cover;

    // Check if this feature appeared earlier in the path.
    let mut incoming_zero_fraction = 1.0;
    let mut incoming_one_fraction = 1.0;
    let mut duplicate_idx = None;

    for (i, entry) in path.iter().enumerate().skip(1) {
        if entry.feature_idx == split_feat {
            incoming_zero_fraction = entry.zero_fraction;
            incoming_one_fraction = entry.one_fraction;
            duplicate_idx = Some(i);
            break;
        }
    }

    if let Some(dup) = duplicate_idx {
        unwind_path(path, dup);
    }

    // If one child has zero cover, only recurse into the non-zero side.
    // The zero-cover subtree has no training data and contributes nothing;
    // extending the path with zero_fraction=0 would cause division-by-zero
    // in unwind_path.
    if hot_cover > 0.0 && cold_cover > 0.0 {
        // Standard case: both children have data.
        extend_path(
            path,
            hot_zero_fraction * incoming_zero_fraction,
            incoming_one_fraction,
            split_feat,
        );
        tree_shap_recursive(arena, covers, hot_child, features, shap_values, path);

        // Unwind hot extension, then re-extend with cold parameters.
        unwind_path(path, path.len() - 1);
        extend_path(
            path,
            cold_zero_fraction * incoming_zero_fraction,
            0.0, // sample doesn't go this way
            split_feat,
        );
        tree_shap_recursive(arena, covers, cold_child, features, shap_values, path);

        // Unwind cold extension.
        unwind_path(path, path.len() - 1);
    } else if hot_cover > 0.0 {
        // Only hot child has data — recurse without adding a path entry
        // for this feature (it has no decision power at this split).
        tree_shap_recursive(arena, covers, hot_child, features, shap_values, path);
    } else {
        // Only cold child has data.
        tree_shap_recursive(arena, covers, cold_child, features, shap_values, path);
    }

    // Restore duplicate if we unwound one earlier.
    if duplicate_idx.is_some() {
        extend_path(
            path,
            incoming_zero_fraction,
            incoming_one_fraction,
            split_feat,
        );
    }
}

/// Compute SHAP values for a single tree.
///
/// Returns per-feature SHAP contributions. The base value for a single tree
/// is the expected leaf value (weighted by sample count).
pub fn tree_shap_values(
    arena: &TreeArena,
    root: NodeId,
    features: &[f64],
    n_features: usize,
) -> ShapValues {
    let mut shap_values = vec![0.0; n_features];

    if arena.n_nodes() == 0 || root.is_none() {
        return ShapValues {
            values: shap_values,
            base_value: 0.0,
        };
    }

    // Precompute correct covers (sum of leaf sample counts) for each node.
    let covers = compute_covers(arena, root);

    // Compute base value: expected leaf value weighted by covers.
    let total_cover: f64 = arena
        .is_leaf
        .iter()
        .enumerate()
        .filter(|(_, &is_leaf)| is_leaf)
        .map(|(i, _)| covers[i])
        .sum();

    let base_value = if total_cover > 0.0 {
        arena
            .leaf_value
            .iter()
            .zip(arena.is_leaf.iter())
            .enumerate()
            .filter(|(_, (_, &is_leaf))| is_leaf)
            .map(|(i, (&val, _))| val * covers[i])
            .sum::<f64>()
            / total_cover
    } else {
        0.0
    };

    // Initialize path with a sentinel entry.
    let mut path = Vec::with_capacity(32);
    path.push(PathEntry {
        feature_idx: -1,
        zero_fraction: 1.0,
        one_fraction: 1.0,
        pweight: 1.0,
    });

    tree_shap_recursive(arena, &covers, root, features, &mut shap_values, &mut path);

    ShapValues {
        values: shap_values,
        base_value,
    }
}

/// Compute SHAP values for an SGBT ensemble.
///
/// Sums weighted SHAP contributions across all boosting steps.
/// The base_value is the ensemble's base_prediction.
pub fn ensemble_shap<L: crate::loss::Loss>(
    model: &crate::ensemble::SGBT<L>,
    features: &[f64],
) -> ShapValues {
    let n_features = model
        .config()
        .feature_names
        .as_ref()
        .map(|n| n.len())
        .unwrap_or_else(|| features.len());

    let lr = model.config().learning_rate;
    let mut total_shap = vec![0.0; n_features];

    for step in model.steps() {
        let slot = step.slot();
        let tree = slot.active_tree();
        let arena = tree.arena();
        let root = tree.root();

        if arena.n_nodes() == 0 {
            continue;
        }

        let tree_shap = tree_shap_values(arena, root, features, n_features);
        for (i, v) in tree_shap.values.iter().enumerate() {
            if i < total_shap.len() {
                total_shap[i] += lr * v;
            }
        }
    }

    ShapValues {
        values: total_shap,
        base_value: model.base_prediction(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::node::TreeArena;

    #[test]
    fn single_leaf_tree_all_shap_zero() {
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);
        arena.sample_count[root.idx()] = 100;
        arena.leaf_value[root.idx()] = 5.0;

        let shap = tree_shap_values(&arena, root, &[1.0, 2.0, 3.0], 3);
        assert!((shap.base_value - 5.0).abs() < 1e-10);
        for v in &shap.values {
            assert!(v.abs() < 1e-10, "single-leaf SHAP should be 0, got {v}");
        }
    }

    #[test]
    fn two_level_tree_shap_invariant() {
        // Build: root splits on feature 0 at threshold 0.5.
        //   Left leaf: value = -1.0, 60 samples
        //   Right leaf: value = 1.0, 40 samples
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);
        let (left, right) = arena.split_leaf(root, 0, 0.5, -1.0, 1.0);
        arena.sample_count[root.idx()] = 100;
        arena.sample_count[left.idx()] = 60;
        arena.sample_count[right.idx()] = 40;

        // Test: sample goes left (feature 0 = 0.3 <= 0.5).
        let features = [0.3, 5.0];
        let shap = tree_shap_values(&arena, root, &features, 2);

        // Base value = (-1.0 * 60 + 1.0 * 40) / 100 = -0.2
        let expected_base = -0.2;
        assert!(
            (shap.base_value - expected_base).abs() < 1e-10,
            "base_value: got {}, expected {}",
            shap.base_value,
            expected_base
        );

        // Prediction = -1.0 (left leaf).
        let prediction = -1.0;
        let shap_sum: f64 = shap.values.iter().sum();
        let reconstructed = shap.base_value + shap_sum;
        assert!(
            (reconstructed - prediction).abs() < 1e-8,
            "SHAP invariant violated: base({}) + sum({}) = {} != prediction({})",
            shap.base_value,
            shap_sum,
            reconstructed,
            prediction
        );

        // Feature 1 has no splits — its SHAP should be 0.
        assert!(
            shap.values[1].abs() < 1e-10,
            "non-split feature SHAP should be 0, got {}",
            shap.values[1]
        );
    }

    #[test]
    fn shap_invariant_right_path() {
        // Same tree, sample goes right (feature 0 = 0.7 > 0.5).
        let mut arena = TreeArena::new();
        let root = arena.add_leaf(0);
        let (left, right) = arena.split_leaf(root, 0, 0.5, -1.0, 1.0);
        arena.sample_count[root.idx()] = 100;
        arena.sample_count[left.idx()] = 60;
        arena.sample_count[right.idx()] = 40;

        let features = [0.7, 5.0];
        let shap = tree_shap_values(&arena, root, &features, 2);

        let prediction = 1.0; // right leaf
        let reconstructed = shap.base_value + shap.values.iter().sum::<f64>();
        assert!(
            (reconstructed - prediction).abs() < 1e-8,
            "SHAP invariant violated for right path: {} != {}",
            reconstructed,
            prediction
        );
    }

    #[test]
    fn empty_tree_returns_zeros() {
        let arena = TreeArena::new();
        let shap = tree_shap_values(&arena, NodeId::NONE, &[1.0], 1);
        assert_eq!(shap.base_value, 0.0);
        assert_eq!(shap.values.len(), 1);
        assert_eq!(shap.values[0], 0.0);
    }

    #[test]
    fn ensemble_shap_integration() {
        use crate::ensemble::config::SGBTConfig;
        use crate::ensemble::SGBT;

        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(3)
            .n_bins(8)
            .build()
            .unwrap();

        let mut model = SGBT::new(config);

        // Train on a simple linear signal: y = 2*x0 + 0.5*x1.
        let mut rng: u64 = 42;
        for _ in 0..200 {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let x0 = (rng >> 33) as f64 / (u32::MAX as f64);
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let x1 = (rng >> 33) as f64 / (u32::MAX as f64);
            let y = 2.0 * x0 + 0.5 * x1;
            model.train_one(&(&[x0, x1][..], y));
        }

        let features = [0.5, 0.5];
        let shap = ensemble_shap(&model, &features);

        // Verify SHAP invariant.
        let prediction = model.predict(&features);
        let reconstructed = shap.base_value + shap.values.iter().sum::<f64>();
        assert!(
            (reconstructed - prediction).abs() < 0.1,
            "ensemble SHAP invariant violated: {} != {} (diff={})",
            reconstructed,
            prediction,
            (reconstructed - prediction).abs()
        );
    }
}
