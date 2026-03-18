//! Zero-copy, zero-alloc inference view over a packed ensemble binary.
//!
//! [`EnsembleView`] is constructed from a `&[u8]` buffer and validates the
//! entire structure on creation. After validation, all predictions are pure
//! pointer arithmetic with no allocation — suitable for embedded targets.

use crate::error::FormatError;
use crate::packed::{EnsembleHeader, PackedNode, TreeEntry};
use crate::traverse;

/// Zero-copy view over a packed ensemble binary.
///
/// All validation happens in [`from_bytes`](EnsembleView::from_bytes). After
/// construction, predictions use `get_unchecked` for zero-overhead indexing
/// (all bounds have been verified).
///
/// # Lifetime
///
/// The view borrows the input buffer — the buffer must outlive the view.
#[derive(Clone, Copy)]
pub struct EnsembleView<'a> {
    header: &'a EnsembleHeader,
    tree_table: &'a [TreeEntry],
    nodes: &'a [PackedNode],
}

impl<'a> EnsembleView<'a> {
    /// Parse and validate a packed ensemble binary.
    ///
    /// Validates:
    /// - Magic bytes match `"IRIT"`
    /// - Format version is supported
    /// - Buffer is large enough for header + tree table + all nodes
    /// - Every internal node's child indices are within bounds
    /// - Every internal node's feature index is < `n_features`
    ///
    /// # Errors
    ///
    /// Returns [`FormatError`] if any validation check fails.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, FormatError> {
        use core::mem::{align_of, size_of};

        let header_size = size_of::<EnsembleHeader>();
        if data.len() < header_size {
            return Err(FormatError::Truncated);
        }

        // Validate alignment — EnsembleHeader requires 4-byte alignment.
        // If the buffer isn't aligned, we can't safely cast. On embedded targets
        // this would be a hard fault.
        if (data.as_ptr() as usize) % align_of::<EnsembleHeader>() != 0 {
            return Err(FormatError::Unaligned);
        }

        // SAFETY: We've checked length and alignment.
        let header = unsafe { &*(data.as_ptr() as *const EnsembleHeader) };

        if header.magic != EnsembleHeader::MAGIC {
            return Err(FormatError::BadMagic);
        }
        if header.version != EnsembleHeader::VERSION {
            return Err(FormatError::UnsupportedVersion);
        }

        let n_trees = header.n_trees as usize;
        let tree_table_size = n_trees * size_of::<TreeEntry>();
        let tree_table_offset = header_size;

        if data.len() < tree_table_offset + tree_table_size {
            return Err(FormatError::Truncated);
        }

        // SAFETY: Alignment of TreeEntry is 4, same as EnsembleHeader, and
        // header_size (16) is a multiple of 4, so tree_table_ptr is aligned.
        let tree_table_ptr = unsafe { data.as_ptr().add(tree_table_offset) } as *const TreeEntry;
        let tree_table = unsafe { core::slice::from_raw_parts(tree_table_ptr, n_trees) };

        // Compute total nodes and validate tree table
        let nodes_base_offset = tree_table_offset + tree_table_size;
        let mut total_nodes: usize = 0;
        for entry in tree_table {
            total_nodes = total_nodes
                .checked_add(entry.n_nodes as usize)
                .ok_or(FormatError::Truncated)?;
        }

        let nodes_size = total_nodes
            .checked_mul(size_of::<PackedNode>())
            .ok_or(FormatError::Truncated)?;
        let total_required = nodes_base_offset
            .checked_add(nodes_size)
            .ok_or(FormatError::Truncated)?;
        if data.len() < total_required {
            return Err(FormatError::Truncated);
        }

        // Validate tree offsets and alignment
        for entry in tree_table {
            let node_byte_offset = entry.offset as usize;
            // Tree offset must be aligned to PackedNode size
            if node_byte_offset % size_of::<PackedNode>() != 0 {
                return Err(FormatError::MisalignedTreeOffset);
            }
            let tree_bytes = (entry.n_nodes as usize)
                .checked_mul(size_of::<PackedNode>())
                .ok_or(FormatError::Truncated)?;
            let tree_end = node_byte_offset
                .checked_add(tree_bytes)
                .ok_or(FormatError::Truncated)?;
            if tree_end > nodes_size {
                return Err(FormatError::Truncated);
            }
        }

        let nodes_ptr = unsafe { data.as_ptr().add(nodes_base_offset) } as *const PackedNode;
        let nodes = unsafe { core::slice::from_raw_parts(nodes_ptr, total_nodes) };

        // Validate every node's child indices and feature indices
        let n_features = header.n_features as usize;
        for (tree_idx, entry) in tree_table.iter().enumerate() {
            let tree_node_offset = entry.offset as usize / size_of::<PackedNode>();
            let tree_n_nodes = entry.n_nodes as usize;

            for local_idx in 0..tree_n_nodes {
                let global_idx = tree_node_offset + local_idx;
                let node = &nodes[global_idx];

                if !node.is_leaf() {
                    let left = node.left_child() as usize;
                    let right = node.right_child() as usize;

                    // Children must be within this tree's node range
                    if left >= tree_n_nodes || right >= tree_n_nodes {
                        return Err(FormatError::InvalidNodeIndex);
                    }

                    if n_features > 0 && node.feature_idx() as usize >= n_features {
                        return Err(FormatError::InvalidFeatureIndex);
                    }
                }
            }

            let _ = tree_idx; // suppress unused warning
        }

        Ok(Self {
            header,
            tree_table,
            nodes,
        })
    }

    /// Predict a single sample. Zero allocation.
    ///
    /// Returns `base_prediction + sum(tree_predictions)`.
    ///
    /// # Precondition
    ///
    /// `features.len()` **must** be `>= self.n_features()`. Passing fewer
    /// features than the model expects causes **undefined behavior** (out-of-bounds
    /// read via `get_unchecked` in the traversal hot path). A `debug_assert`
    /// catches this in debug builds.
    pub fn predict(&self, features: &[f32]) -> f32 {
        debug_assert!(
            features.len() >= self.header.n_features as usize,
            "predict: features.len() ({}) < n_features ({})",
            features.len(),
            self.header.n_features
        );
        let mut sum = self.header.base_prediction;
        for entry in self.tree_table {
            let start = entry.offset as usize / core::mem::size_of::<PackedNode>();
            let end = start + entry.n_nodes as usize;
            let tree_nodes = &self.nodes[start..end];
            sum += traverse::predict_tree(tree_nodes, features);
        }
        sum
    }

    /// Batch predict. Uses x4 interleaving when `samples.len() >= 4`.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() < samples.len()`.
    ///
    /// # Precondition
    ///
    /// Every sample must have `len() >= self.n_features()`. See [`predict`](Self::predict).
    pub fn predict_batch(&self, samples: &[&[f32]], out: &mut [f32]) {
        assert!(out.len() >= samples.len());

        let n = samples.len();
        let mut i = 0;

        // Process groups of 4 with interleaved traversal
        while i + 4 <= n {
            let batch = [samples[i], samples[i + 1], samples[i + 2], samples[i + 3]];
            // Initialize with base prediction
            let mut sums = [self.header.base_prediction; 4];
            for entry in self.tree_table {
                let start = entry.offset as usize / core::mem::size_of::<PackedNode>();
                let end = start + entry.n_nodes as usize;
                let tree_nodes = &self.nodes[start..end];
                let preds = traverse::predict_tree_x4(tree_nodes, batch);
                for j in 0..4 {
                    sums[j] += preds[j];
                }
            }
            out[i] = sums[0];
            out[i + 1] = sums[1];
            out[i + 2] = sums[2];
            out[i + 3] = sums[3];
            i += 4;
        }

        // Remainder
        while i < n {
            out[i] = self.predict(samples[i]);
            i += 1;
        }
    }

    /// Number of trees in the ensemble.
    #[inline]
    pub fn n_trees(&self) -> u16 {
        self.header.n_trees
    }

    /// Expected number of input features.
    #[inline]
    pub fn n_features(&self) -> u16 {
        self.header.n_features
    }

    /// Base prediction value.
    #[inline]
    pub fn base_prediction(&self) -> f32 {
        self.header.base_prediction
    }

    /// Total number of packed nodes across all trees.
    #[inline]
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl<'a> core::fmt::Debug for EnsembleView<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EnsembleView")
            .field("n_trees", &self.n_trees())
            .field("n_features", &self.n_features())
            .field("base_prediction", &self.base_prediction())
            .field("total_nodes", &self.total_nodes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::{EnsembleHeader, PackedNode, TreeEntry};
    use alloc::{format, vec, vec::Vec};
    use core::mem::size_of;

    /// Build a minimal valid packed binary with one tree, one leaf.
    fn build_single_leaf_binary(leaf_value: f32, base: f32) -> Vec<u8> {
        let header = EnsembleHeader {
            magic: EnsembleHeader::MAGIC,
            version: EnsembleHeader::VERSION,
            n_trees: 1,
            n_features: 1,
            _reserved: 0,
            base_prediction: base,
        };
        let entry = TreeEntry {
            n_nodes: 1,
            offset: 0,
        };
        let node = PackedNode::leaf(leaf_value);

        let mut buf = Vec::new();
        // Ensure 4-byte alignment by starting with a properly sized vec
        buf.extend_from_slice(as_bytes(&header));
        buf.extend_from_slice(as_bytes(&entry));
        buf.extend_from_slice(as_bytes(&node));
        buf
    }

    /// Build a binary with one tree: root splits on feat 0 at threshold 5.0.
    fn build_one_split_binary() -> Vec<u8> {
        let header = EnsembleHeader {
            magic: EnsembleHeader::MAGIC,
            version: EnsembleHeader::VERSION,
            n_trees: 1,
            n_features: 2,
            _reserved: 0,
            base_prediction: 0.0,
        };
        let entry = TreeEntry {
            n_nodes: 3,
            offset: 0,
        };
        let nodes = [
            PackedNode::split(5.0, 0, 1, 2),
            PackedNode::leaf(-1.0),
            PackedNode::leaf(1.0),
        ];

        let mut buf = Vec::new();
        buf.extend_from_slice(as_bytes(&header));
        buf.extend_from_slice(as_bytes(&entry));
        for n in &nodes {
            buf.extend_from_slice(as_bytes(n));
        }
        buf
    }

    /// Build a binary with two trees for testing multi-tree prediction.
    fn build_two_tree_binary() -> Vec<u8> {
        let header = EnsembleHeader {
            magic: EnsembleHeader::MAGIC,
            version: EnsembleHeader::VERSION,
            n_trees: 2,
            n_features: 2,
            _reserved: 0,
            base_prediction: 1.0,
        };
        // Tree 0: 3 nodes, offset 0
        // Tree 1: 1 node (leaf), offset 3*12=36
        let entries = [
            TreeEntry {
                n_nodes: 3,
                offset: 0,
            },
            TreeEntry {
                n_nodes: 1,
                offset: 3 * size_of::<PackedNode>() as u32,
            },
        ];
        let nodes = [
            // Tree 0
            PackedNode::split(5.0, 0, 1, 2),
            PackedNode::leaf(-1.0),
            PackedNode::leaf(1.0),
            // Tree 1
            PackedNode::leaf(0.5),
        ];

        let mut buf = Vec::new();
        buf.extend_from_slice(as_bytes(&header));
        for e in &entries {
            buf.extend_from_slice(as_bytes(e));
        }
        for n in &nodes {
            buf.extend_from_slice(as_bytes(n));
        }
        buf
    }

    /// Cast a repr(C) struct to bytes.
    fn as_bytes<T: Sized>(val: &T) -> &[u8] {
        unsafe { core::slice::from_raw_parts(val as *const T as *const u8, size_of::<T>()) }
    }

    #[test]
    fn parse_single_leaf() {
        let buf = build_single_leaf_binary(42.0, 0.0);
        let view = EnsembleView::from_bytes(&buf).unwrap();
        assert_eq!(view.n_trees(), 1);
        assert_eq!(view.n_features(), 1);
        assert_eq!(view.total_nodes(), 1);
    }

    #[test]
    fn predict_single_leaf() {
        let buf = build_single_leaf_binary(42.0, 10.0);
        let view = EnsembleView::from_bytes(&buf).unwrap();
        // prediction = base(10.0) + leaf(42.0) = 52.0
        let pred = view.predict(&[0.0]);
        assert!((pred - 52.0).abs() < 1e-6);
    }

    #[test]
    fn predict_one_split_left() {
        let buf = build_one_split_binary();
        let view = EnsembleView::from_bytes(&buf).unwrap();
        // feat[0]=3.0, not > 5.0 -> left -> leaf=-1.0; base=0.0
        let pred = view.predict(&[3.0, 0.0]);
        assert!((pred - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn predict_one_split_right() {
        let buf = build_one_split_binary();
        let view = EnsembleView::from_bytes(&buf).unwrap();
        // feat[0]=7.0, > 5.0 -> right -> leaf=1.0; base=0.0
        let pred = view.predict(&[7.0, 0.0]);
        assert!((pred - 1.0).abs() < 1e-6);
    }

    #[test]
    fn predict_two_trees() {
        let buf = build_two_tree_binary();
        let view = EnsembleView::from_bytes(&buf).unwrap();
        // feat[0]=3.0 -> tree0: left=-1.0; tree1: leaf=0.5; base=1.0
        // total = 1.0 + (-1.0) + 0.5 = 0.5
        let pred = view.predict(&[3.0, 0.0]);
        assert!((pred - 0.5).abs() < 1e-6);
    }

    #[test]
    fn predict_batch_matches_single() {
        let buf = build_two_tree_binary();
        let view = EnsembleView::from_bytes(&buf).unwrap();

        let samples: Vec<&[f32]> = vec![
            &[3.0, 0.0],
            &[7.0, 0.0],
            &[5.0, 0.0],
            &[0.0, 0.0],
            &[10.0, 0.0],
        ];
        let mut out = vec![0.0f32; 5];
        view.predict_batch(&samples, &mut out);

        for (i, &s) in samples.iter().enumerate() {
            let expected = view.predict(s);
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "batch[{}] = {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut buf = build_single_leaf_binary(0.0, 0.0);
        buf[0] = 0xFF; // corrupt magic
        assert_eq!(
            EnsembleView::from_bytes(&buf).unwrap_err(),
            FormatError::BadMagic
        );
    }

    #[test]
    fn truncated_buffer_is_rejected() {
        let buf = build_single_leaf_binary(0.0, 0.0);
        assert_eq!(
            EnsembleView::from_bytes(&buf[..4]).unwrap_err(),
            FormatError::Truncated
        );
    }

    #[test]
    fn bad_version_is_rejected() {
        let mut buf = build_single_leaf_binary(0.0, 0.0);
        // version is at offset 4 (after magic u32), 2 bytes LE
        buf[4] = 99;
        buf[5] = 0;
        assert_eq!(
            EnsembleView::from_bytes(&buf).unwrap_err(),
            FormatError::UnsupportedVersion
        );
    }

    #[test]
    fn invalid_child_index_is_rejected() {
        let header = EnsembleHeader {
            magic: EnsembleHeader::MAGIC,
            version: EnsembleHeader::VERSION,
            n_trees: 1,
            n_features: 2,
            _reserved: 0,
            base_prediction: 0.0,
        };
        let entry = TreeEntry {
            n_nodes: 3,
            offset: 0,
        };
        // Node 0 points to child 99, which is out of bounds (only 3 nodes)
        let nodes = [
            PackedNode::split(5.0, 0, 1, 99), // right child out of bounds
            PackedNode::leaf(-1.0),
            PackedNode::leaf(1.0),
        ];

        let mut buf = Vec::new();
        buf.extend_from_slice(as_bytes(&header));
        buf.extend_from_slice(as_bytes(&entry));
        for n in &nodes {
            buf.extend_from_slice(as_bytes(n));
        }

        assert_eq!(
            EnsembleView::from_bytes(&buf).unwrap_err(),
            FormatError::InvalidNodeIndex
        );
    }

    #[test]
    fn debug_format() {
        let buf = build_single_leaf_binary(0.0, 0.0);
        let view = EnsembleView::from_bytes(&buf).unwrap();
        let debug = format!("{:?}", view);
        assert!(debug.contains("EnsembleView"));
        assert!(debug.contains("n_trees"));
    }
}
