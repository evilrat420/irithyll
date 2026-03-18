//! Export trained SGBT models to the irithyll-core packed binary format.
//!
//! Converts the tree ensemble into a compact, zero-alloc-friendly binary
//! that can be loaded by [`irithyll_core::EnsembleView`] on embedded targets.
//!
//! # Usage
//!
//! ```no_run
//! use irithyll::{SGBTConfig, SGBT, Sample};
//! use irithyll::export_embedded::export_packed;
//!
//! let config = SGBTConfig::builder().n_steps(10).build().unwrap();
//! let mut model = SGBT::new(config);
//!
//! // ... train the model ...
//!
//! let packed = export_packed(&model, 3);
//! // `packed` can be written to flash, sent over network, etc.
//! // Load with `irithyll_core::EnsembleView::from_bytes(&packed)`
//! ```

use std::collections::VecDeque;

use crate::ensemble::SGBT;
use crate::loss::Loss;
use crate::tree::node::NodeId;
use irithyll_core::packed::{EnsembleHeader, PackedNode, TreeEntry};

/// Convert a trained SGBT model into the irithyll-core packed binary format.
///
/// # Arguments
///
/// * `model` - Trained SGBT model to export.
/// * `n_features` - Number of input features (must be specified explicitly
///   because the model doesn't track this -- Hoeffding trees accept any width).
///
/// # Panics
///
/// Panics if any tree has more than 65535 nodes (would overflow u16 indices).
pub fn export_packed<L: Loss>(model: &SGBT<L>, n_features: usize) -> Vec<u8> {
    let learning_rate = model.config().learning_rate;
    let n_trees = model.steps().len();

    // Phase 1: BFS-reindex each tree into contiguous PackedNode arrays.
    let mut all_tree_nodes: Vec<Vec<PackedNode>> = Vec::with_capacity(n_trees);

    for step in model.steps() {
        let arena = step.slot().active_tree().arena();
        let root = step.slot().active_tree().root();
        let packed_nodes = bfs_pack_tree(arena, root, learning_rate);
        all_tree_nodes.push(packed_nodes);
    }

    // Phase 2: Build the binary buffer.
    let header = EnsembleHeader {
        magic: EnsembleHeader::MAGIC,
        version: EnsembleHeader::VERSION,
        n_trees: n_trees as u16,
        n_features: n_features as u16,
        _reserved: 0,
        base_prediction: model.base_prediction() as f32,
    };

    // Build tree table with byte offsets
    let mut tree_table: Vec<TreeEntry> = Vec::with_capacity(n_trees);
    let mut byte_offset: u32 = 0;
    let node_size = core::mem::size_of::<PackedNode>() as u32;

    for tree_nodes in &all_tree_nodes {
        tree_table.push(TreeEntry {
            n_nodes: tree_nodes.len() as u32,
            offset: byte_offset,
        });
        byte_offset += tree_nodes.len() as u32 * node_size;
    }

    // Phase 3: Serialize to bytes.
    let header_size = core::mem::size_of::<EnsembleHeader>();
    let tree_table_size = n_trees * core::mem::size_of::<TreeEntry>();
    let nodes_size = byte_offset as usize;
    let total_size = header_size + tree_table_size + nodes_size;

    // Allocate aligned buffer (4-byte alignment required by EnsembleView)
    let mut buf: Vec<u8> = Vec::with_capacity(total_size);

    // Write header
    buf.extend_from_slice(as_bytes(&header));

    // Write tree table
    for entry in &tree_table {
        buf.extend_from_slice(as_bytes(entry));
    }

    // Write nodes
    for tree_nodes in &all_tree_nodes {
        for node in tree_nodes {
            buf.extend_from_slice(as_bytes(node));
        }
    }

    debug_assert_eq!(buf.len(), total_size);
    buf
}

/// BFS-walk a TreeArena from `root` and pack into contiguous PackedNodes.
///
/// Returns nodes in BFS order with root at index 0. All child indices
/// are remapped to BFS positions.
fn bfs_pack_tree(
    arena: &crate::tree::node::TreeArena,
    root: NodeId,
    learning_rate: f64,
) -> Vec<PackedNode> {
    if root.is_none() || arena.n_nodes() == 0 {
        // Empty tree -- single leaf with value 0
        return vec![PackedNode::leaf(0.0)];
    }

    // BFS to discover traversal order and assign contiguous indices.
    let mut queue = VecDeque::new();
    let mut bfs_order: Vec<NodeId> = Vec::new();

    queue.push_back(root);
    while let Some(node_id) = queue.pop_front() {
        bfs_order.push(node_id);
        let idx = node_id.idx();
        if !arena.is_leaf[idx] {
            queue.push_back(arena.left[idx]);
            queue.push_back(arena.right[idx]);
        }
    }

    let n_nodes = bfs_order.len();
    assert!(
        n_nodes <= u16::MAX as usize,
        "tree has {} nodes, exceeds u16::MAX (65535)",
        n_nodes
    );

    // Build old-NodeId -> new-BFS-index mapping.
    // Since NodeId.0 can be sparse (arena may have gaps), use a simple lookup.
    let max_id = bfs_order.iter().map(|id| id.0).max().unwrap_or(0) as usize;
    let mut id_to_bfs = vec![u16::MAX; max_id + 1];
    for (bfs_idx, &node_id) in bfs_order.iter().enumerate() {
        id_to_bfs[node_id.idx()] = bfs_idx as u16;
    }

    // Convert to PackedNodes
    let mut packed = Vec::with_capacity(n_nodes);
    for &node_id in &bfs_order {
        let idx = node_id.idx();
        if arena.is_leaf[idx] {
            packed.push(PackedNode::leaf(
                (learning_rate * arena.leaf_value[idx]) as f32,
            ));
        } else {
            let feature = arena.feature_idx[idx] as u16;
            let threshold = arena.threshold[idx] as f32;
            let left_bfs = id_to_bfs[arena.left[idx].idx()];
            let right_bfs = id_to_bfs[arena.right[idx].idx()];
            packed.push(PackedNode::split(threshold, feature, left_bfs, right_bfs));
        }
    }

    packed
}

/// Compare predictions between original SGBT and packed EnsembleView.
///
/// Returns the maximum absolute difference across all test samples.
/// A well-exported model should have max error < 1e-5 (f64->f32 precision loss).
///
/// # Panics
///
/// Panics if `packed` is not a valid packed binary.
pub fn validate_export<L: Loss>(model: &SGBT<L>, packed: &[u8], test_features: &[Vec<f64>]) -> f64 {
    let view = irithyll_core::EnsembleView::from_bytes(packed)
        .expect("validate_export: invalid packed binary");

    let mut max_diff: f64 = 0.0;

    for features_f64 in test_features {
        // Original model prediction (f64)
        let original = model.predict(features_f64);

        // Packed prediction (f32 -> f64 for comparison)
        let features_f32: Vec<f32> = features_f64.iter().map(|&v| v as f32).collect();
        let packed_pred = view.predict(&features_f32) as f64;

        let diff = (original - packed_pred).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    max_diff
}

/// Cast a `repr(C)` struct to its byte representation.
fn as_bytes<T: Sized>(val: &T) -> &[u8] {
    unsafe { core::slice::from_raw_parts(val as *const T as *const u8, core::mem::size_of::<T>()) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::config::SGBTConfig;
    use crate::sample::Sample;

    fn trained_model() -> SGBT {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(5)
            .max_depth(3)
            .n_bins(8)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);
        for i in 0..100 {
            let x = (i as f64) * 0.1;
            model.train_one(&Sample::new(vec![x, x * 2.0, x * 0.5], x * 3.0));
        }
        model
    }

    #[test]
    fn export_produces_valid_binary() {
        let model = trained_model();
        let packed = export_packed(&model, 3);

        // Should be parseable
        let view = irithyll_core::EnsembleView::from_bytes(&packed);
        assert!(view.is_ok(), "exported binary should be valid");

        let view = view.unwrap();
        assert_eq!(view.n_trees(), 5);
        assert_eq!(view.n_features(), 3);
    }

    #[test]
    fn export_preserves_base_prediction() {
        let model = trained_model();
        let packed = export_packed(&model, 3);
        let view = irithyll_core::EnsembleView::from_bytes(&packed).unwrap();

        let expected = model.base_prediction() as f32;
        assert!(
            (view.base_prediction() - expected).abs() < 1e-6,
            "base prediction mismatch: got {}, expected {}",
            view.base_prediction(),
            expected
        );
    }

    #[test]
    fn export_predictions_match_within_tolerance() {
        let model = trained_model();
        let packed = export_packed(&model, 3);

        let test_data: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                let x = (i as f64) * 0.2;
                vec![x, x * 2.0, x * 0.5]
            })
            .collect();

        let max_diff = validate_export(&model, &packed, &test_data);
        assert!(
            max_diff < 0.1,
            "max prediction difference {} exceeds tolerance",
            max_diff
        );
    }

    #[test]
    fn export_untrained_model() {
        let config = SGBTConfig::builder().n_steps(3).build().unwrap();
        let model = SGBT::new(config);
        let packed = export_packed(&model, 5);

        let view = irithyll_core::EnsembleView::from_bytes(&packed).unwrap();
        assert_eq!(view.n_trees(), 3);

        // Untrained: all predictions should be ~base_prediction
        let pred = view.predict(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(pred.is_finite());
    }

    #[test]
    fn binary_size_is_compact() {
        let model = trained_model();
        let packed = export_packed(&model, 3);

        // Header(16) + 5 trees * TreeEntry(8) + nodes * 12
        // Should be much smaller than JSON/bincode serialization
        let header_size = 16;
        let table_size = 5 * 8;
        let min_size = header_size + table_size + 5 * 12; // at least 1 node per tree
        assert!(
            packed.len() >= min_size,
            "packed binary too small: {} bytes",
            packed.len()
        );
        // Sanity: shouldn't be huge either (5 trees, max depth 3)
        assert!(
            packed.len() < 100_000,
            "packed binary unexpectedly large: {} bytes",
            packed.len()
        );
    }

    #[test]
    fn roundtrip_single_tree() {
        let config = SGBTConfig::builder()
            .n_steps(1)
            .learning_rate(0.05)
            .grace_period(5)
            .max_depth(2)
            .n_bins(8)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);
        for i in 0..50 {
            let x = (i as f64) * 0.1;
            model.train_one(&Sample::new(vec![x, x * 2.0], x + 1.0));
        }

        let packed = export_packed(&model, 2);
        let view = irithyll_core::EnsembleView::from_bytes(&packed).unwrap();
        assert_eq!(view.n_trees(), 1);

        // Verify prediction is finite and in a reasonable range
        let pred = view.predict(&[2.5, 5.0]);
        assert!(pred.is_finite());
    }
}
