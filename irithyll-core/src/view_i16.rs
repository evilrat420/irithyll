//! Zero-copy, zero-alloc inference view over a quantized (int16) ensemble binary.
//!
//! [`QuantizedEnsembleView`] is the int16 equivalent of [`EnsembleView`](crate::EnsembleView).
//! After validation in [`from_bytes`](QuantizedEnsembleView::from_bytes), all traversal
//! uses integer-only comparisons. Features are quantized once per predict call (one f32
//! multiply per comparison), then each tree traversal is pure i16. Leaf values accumulate
//! as i32, dequantized once at the end.
//!
//! On Cortex-M0+ (no FPU), the `predict_prequantized` path eliminates **all** float ops
//! from the hot loop — the only float is the final `leaf_sum / leaf_scale`.

use crate::error::FormatError;
use crate::packed::TreeEntry;
use crate::packed_i16::{PackedNodeI16, QuantizedEnsembleHeader};
use crate::traverse_i16;

/// Zero-copy view over a quantized (int16) ensemble binary.
///
/// After validation in [`from_bytes`](Self::from_bytes), all traversal uses
/// integer-only comparisons. Features are quantized once per predict call,
/// then each tree traversal is pure i16. Leaf values accumulate as i32,
/// dequantized once at the end.
///
/// # Lifetime
///
/// The view borrows the input buffer — the buffer must outlive the view.
#[derive(Clone, Copy)]
pub struct QuantizedEnsembleView<'a> {
    header: &'a QuantizedEnsembleHeader,
    leaf_scale: f32,
    feature_scales: &'a [f32],
    tree_table: &'a [TreeEntry],
    nodes: &'a [PackedNodeI16],
}

impl<'a> QuantizedEnsembleView<'a> {
    /// Parse and validate a quantized ensemble binary.
    ///
    /// # Binary layout
    ///
    /// ```text
    /// [QuantizedEnsembleHeader: 16 bytes]   magic="IR16", version, n_trees, n_features, base_prediction
    /// [leaf_scale: f32]                      4 bytes — global scale for leaf dequantization
    /// [feature_scales: f32 x n_features]     n_features x 4 bytes — per-feature quantization scales
    /// [TreeEntry x n_trees: 8 bytes each]    same struct as f32 format
    /// [PackedNodeI16 x total_nodes: 8 bytes each]
    /// ```
    ///
    /// # Validates
    ///
    /// - Magic bytes match `"IR16"`
    /// - Format version is supported
    /// - Buffer is large enough for header + leaf_scale + feature_scales + tree table + all nodes
    /// - Every internal node's child indices are within bounds
    /// - Every internal node's feature index is < `n_features`
    /// - Tree offsets are aligned to `size_of::<PackedNodeI16>()`
    ///
    /// # Errors
    ///
    /// Returns [`FormatError`] if any validation check fails.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, FormatError> {
        use core::mem::{align_of, size_of};

        let header_size = size_of::<QuantizedEnsembleHeader>();
        if data.len() < header_size {
            return Err(FormatError::Truncated);
        }

        // Validate alignment — QuantizedEnsembleHeader requires 4-byte alignment.
        if (data.as_ptr() as usize) % align_of::<QuantizedEnsembleHeader>() != 0 {
            return Err(FormatError::Unaligned);
        }

        // SAFETY: We've checked length and alignment.
        let header = unsafe { &*(data.as_ptr() as *const QuantizedEnsembleHeader) };

        if header.magic != QuantizedEnsembleHeader::MAGIC {
            return Err(FormatError::BadMagic);
        }
        if header.version != QuantizedEnsembleHeader::VERSION {
            return Err(FormatError::UnsupportedVersion);
        }

        let n_trees = header.n_trees as usize;
        let n_features = header.n_features as usize;

        // After header: leaf_scale (4 bytes) + feature_scales (n_features * 4 bytes)
        let leaf_scale_offset = header_size;
        let leaf_scale_size = size_of::<f32>();
        let feature_scales_offset = leaf_scale_offset + leaf_scale_size;
        let feature_scales_size = n_features
            .checked_mul(size_of::<f32>())
            .ok_or(FormatError::Truncated)?;

        let tree_table_offset = feature_scales_offset
            .checked_add(feature_scales_size)
            .ok_or(FormatError::Truncated)?;
        let tree_table_size = n_trees
            .checked_mul(size_of::<TreeEntry>())
            .ok_or(FormatError::Truncated)?;

        let nodes_base_offset = tree_table_offset
            .checked_add(tree_table_size)
            .ok_or(FormatError::Truncated)?;

        // Check we have at least up to the tree table
        if data.len() < nodes_base_offset {
            return Err(FormatError::Truncated);
        }

        // Read leaf_scale
        // SAFETY: header is 4-byte aligned, header_size is 16 (multiple of 4),
        // so leaf_scale_offset is 4-byte aligned. f32 requires 4-byte alignment.
        let leaf_scale_ptr = unsafe { data.as_ptr().add(leaf_scale_offset) } as *const f32;
        let leaf_scale = unsafe { *leaf_scale_ptr };

        // Read feature_scales
        // SAFETY: leaf_scale_offset + 4 = feature_scales_offset, still 4-byte aligned.
        let feature_scales_ptr = unsafe { data.as_ptr().add(feature_scales_offset) } as *const f32;
        let feature_scales = unsafe { core::slice::from_raw_parts(feature_scales_ptr, n_features) };

        // Read tree table
        // SAFETY: feature_scales_offset + n_features*4 = tree_table_offset.
        // n_features*4 is a multiple of 4, so tree_table_offset is 4-byte aligned.
        // TreeEntry has align(4).
        let tree_table_ptr = unsafe { data.as_ptr().add(tree_table_offset) } as *const TreeEntry;
        let tree_table = unsafe { core::slice::from_raw_parts(tree_table_ptr, n_trees) };

        // Compute total nodes and validate
        let mut total_nodes: usize = 0;
        for entry in tree_table {
            total_nodes = total_nodes
                .checked_add(entry.n_nodes as usize)
                .ok_or(FormatError::Truncated)?;
        }

        let nodes_size = total_nodes
            .checked_mul(size_of::<PackedNodeI16>())
            .ok_or(FormatError::Truncated)?;
        let total_required = nodes_base_offset
            .checked_add(nodes_size)
            .ok_or(FormatError::Truncated)?;
        if data.len() < total_required {
            return Err(FormatError::Truncated);
        }

        // Validate tree offsets and alignment (must align to 8-byte PackedNodeI16 size)
        for entry in tree_table {
            let node_byte_offset = entry.offset as usize;
            if node_byte_offset % size_of::<PackedNodeI16>() != 0 {
                return Err(FormatError::MisalignedTreeOffset);
            }
            let tree_bytes = (entry.n_nodes as usize)
                .checked_mul(size_of::<PackedNodeI16>())
                .ok_or(FormatError::Truncated)?;
            let tree_end = node_byte_offset
                .checked_add(tree_bytes)
                .ok_or(FormatError::Truncated)?;
            if tree_end > nodes_size {
                return Err(FormatError::Truncated);
            }
        }

        // SAFETY: nodes_base_offset is 4-byte aligned (sum of 4-byte-aligned components).
        // PackedNodeI16 has align(4), so this is safe.
        let nodes_ptr = unsafe { data.as_ptr().add(nodes_base_offset) } as *const PackedNodeI16;
        let nodes = unsafe { core::slice::from_raw_parts(nodes_ptr, total_nodes) };

        // Validate every node's child indices and feature indices
        for entry in tree_table {
            let tree_node_offset = entry.offset as usize / size_of::<PackedNodeI16>();
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
        }

        Ok(Self {
            header,
            leaf_scale,
            feature_scales,
            tree_table,
            nodes,
        })
    }

    /// Predict a single sample with inline feature quantization. Zero allocation.
    ///
    /// Each tree comparison performs one f32 multiply to quantize the feature on-the-fly.
    /// Leaf values accumulate as i32 and are dequantized once at the end:
    /// `base_prediction + leaf_sum / leaf_scale`.
    ///
    /// For the pure-integer path (no f32 ops in the hot loop), use
    /// [`predict_prequantized`](Self::predict_prequantized).
    ///
    /// # Precondition
    ///
    /// `features.len()` **must** be `>= self.n_features()`. Passing fewer features
    /// causes **undefined behavior** via `get_unchecked`. A `debug_assert` catches
    /// this in debug builds.
    pub fn predict(&self, features: &[f32]) -> f32 {
        debug_assert!(
            features.len() >= self.header.n_features as usize,
            "predict: features.len() ({}) < n_features ({})",
            features.len(),
            self.header.n_features
        );

        let mut leaf_sum: i32 = 0;
        for entry in self.tree_table {
            let start = entry.offset as usize / core::mem::size_of::<PackedNodeI16>();
            let end = start + entry.n_nodes as usize;
            let tree_nodes = &self.nodes[start..end];
            leaf_sum +=
                traverse_i16::predict_tree_i16_inline(tree_nodes, features, self.feature_scales)
                    as i32;
        }

        self.header.base_prediction + (leaf_sum as f32) / self.leaf_scale
    }

    /// Predict a single sample from pre-quantized features. Pure integer hot loop.
    ///
    /// The caller is responsible for quantizing features beforehand:
    /// `features_i16[i] = (features_f32[i] * feature_scales[i]) as i16`
    ///
    /// This eliminates **all** float ops from the tree traversal — the only float
    /// operation is the final `base_prediction + leaf_sum / leaf_scale`.
    ///
    /// # Precondition
    ///
    /// `features_i16.len()` **must** be `>= self.n_features()`. Passing fewer features
    /// causes **undefined behavior** via `get_unchecked`. A `debug_assert` catches
    /// this in debug builds.
    pub fn predict_prequantized(&self, features_i16: &[i16]) -> f32 {
        debug_assert!(
            features_i16.len() >= self.header.n_features as usize,
            "predict_prequantized: features_i16.len() ({}) < n_features ({})",
            features_i16.len(),
            self.header.n_features
        );

        let mut leaf_sum: i32 = 0;
        for entry in self.tree_table {
            let start = entry.offset as usize / core::mem::size_of::<PackedNodeI16>();
            let end = start + entry.n_nodes as usize;
            let tree_nodes = &self.nodes[start..end];
            leaf_sum += traverse_i16::predict_tree_i16(tree_nodes, features_i16) as i32;
        }

        self.header.base_prediction + (leaf_sum as f32) / self.leaf_scale
    }

    /// Batch predict with inline quantization. Uses x4 interleaving when `samples.len() >= 4`.
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

        // No x4 interleaving for inline quantization — the inline path already
        // quantizes per comparison, so interleaving would require 4x the f32 ops.
        // Use simple sequential prediction.
        for (i, &s) in samples.iter().enumerate() {
            out[i] = self.predict(s);
        }
    }

    /// Batch predict from pre-quantized features. Uses x4 interleaving when `samples.len() >= 4`.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() < samples.len()`.
    ///
    /// # Precondition
    ///
    /// Every sample must have `len() >= self.n_features()`. See [`predict_prequantized`](Self::predict_prequantized).
    pub fn predict_batch_prequantized(&self, samples: &[&[i16]], out: &mut [f32]) {
        assert!(out.len() >= samples.len());

        let n = samples.len();
        let mut i = 0;

        // Process groups of 4 with interleaved traversal
        while i + 4 <= n {
            let batch = [samples[i], samples[i + 1], samples[i + 2], samples[i + 3]];
            let mut sums = [0i32; 4];
            for entry in self.tree_table {
                let start = entry.offset as usize / core::mem::size_of::<PackedNodeI16>();
                let end = start + entry.n_nodes as usize;
                let tree_nodes = &self.nodes[start..end];
                let preds = traverse_i16::predict_tree_i16_x4(tree_nodes, batch);
                for j in 0..4 {
                    sums[j] += preds[j] as i32;
                }
            }
            for j in 0..4 {
                out[i + j] = self.header.base_prediction + (sums[j] as f32) / self.leaf_scale;
            }
            i += 4;
        }

        // Remainder
        while i < n {
            out[i] = self.predict_prequantized(samples[i]);
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

    /// Global leaf scale factor for dequantization.
    #[inline]
    pub fn leaf_scale(&self) -> f32 {
        self.leaf_scale
    }

    /// Per-feature quantization scales.
    #[inline]
    pub fn feature_scales(&self) -> &[f32] {
        self.feature_scales
    }

    /// Total number of packed i16 nodes across all trees.
    #[inline]
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl<'a> core::fmt::Debug for QuantizedEnsembleView<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuantizedEnsembleView")
            .field("n_trees", &self.n_trees())
            .field("n_features", &self.n_features())
            .field("base_prediction", &self.base_prediction())
            .field("leaf_scale", &self.leaf_scale())
            .field("total_nodes", &self.total_nodes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::TreeEntry;
    use crate::packed_i16::{PackedNodeI16, QuantizedEnsembleHeader};
    use alloc::{format, vec, vec::Vec};
    use core::mem::size_of;

    /// Cast a repr(C) struct to bytes.
    fn as_bytes<T: Sized>(val: &T) -> &[u8] {
        unsafe { core::slice::from_raw_parts(val as *const T as *const u8, size_of::<T>()) }
    }

    /// Build a minimal valid quantized binary with one tree, one leaf.
    ///
    /// Layout: header(16) + leaf_scale(4) + feature_scales(n_features*4) + tree_entry(8) + node(8)
    fn build_single_leaf_binary(leaf_value: i16, base: f32, leaf_scale: f32) -> Vec<u8> {
        let header = QuantizedEnsembleHeader {
            magic: QuantizedEnsembleHeader::MAGIC,
            version: QuantizedEnsembleHeader::VERSION,
            n_trees: 1,
            n_features: 1,
            _reserved: 0,
            base_prediction: base,
        };
        let feature_scale: f32 = 1.0;
        let entry = TreeEntry {
            n_nodes: 1,
            offset: 0,
        };
        let node = PackedNodeI16::leaf(leaf_value);

        let mut buf = Vec::new();
        buf.extend_from_slice(as_bytes(&header));
        buf.extend_from_slice(as_bytes(&leaf_scale));
        buf.extend_from_slice(as_bytes(&feature_scale));
        buf.extend_from_slice(as_bytes(&entry));
        buf.extend_from_slice(as_bytes(&node));
        buf
    }

    /// Build a binary with one tree: root splits on feat 0 at quantized threshold 500.
    /// Feature scale = 100.0 (so raw feat 5.0 -> quantized 500).
    fn build_one_split_binary() -> Vec<u8> {
        let header = QuantizedEnsembleHeader {
            magic: QuantizedEnsembleHeader::MAGIC,
            version: QuantizedEnsembleHeader::VERSION,
            n_trees: 1,
            n_features: 2,
            _reserved: 0,
            base_prediction: 0.0,
        };
        let leaf_scale: f32 = 100.0;
        let feature_scales: [f32; 2] = [100.0, 100.0];
        let entry = TreeEntry {
            n_nodes: 3,
            offset: 0,
        };
        let nodes = [
            PackedNodeI16::split(500, 0, 1, 2), // threshold 500 = raw 5.0 * scale 100.0
            PackedNodeI16::leaf(-100),          // leaf = -100, dequantized = -100/100 = -1.0
            PackedNodeI16::leaf(100),           // leaf = 100, dequantized = 100/100 = 1.0
        ];

        let mut buf = Vec::new();
        buf.extend_from_slice(as_bytes(&header));
        buf.extend_from_slice(as_bytes(&leaf_scale));
        for s in &feature_scales {
            buf.extend_from_slice(as_bytes(s));
        }
        buf.extend_from_slice(as_bytes(&entry));
        for n in &nodes {
            buf.extend_from_slice(as_bytes(n));
        }
        buf
    }

    /// Build a binary with two trees for testing multi-tree prediction.
    fn build_two_tree_binary() -> Vec<u8> {
        let header = QuantizedEnsembleHeader {
            magic: QuantizedEnsembleHeader::MAGIC,
            version: QuantizedEnsembleHeader::VERSION,
            n_trees: 2,
            n_features: 2,
            _reserved: 0,
            base_prediction: 1.0,
        };
        let leaf_scale: f32 = 100.0;
        let feature_scales: [f32; 2] = [100.0, 100.0];
        // Tree 0: 3 nodes, offset 0
        // Tree 1: 1 node (leaf), offset 3 * 8 = 24 bytes
        let entries = [
            TreeEntry {
                n_nodes: 3,
                offset: 0,
            },
            TreeEntry {
                n_nodes: 1,
                offset: 3 * size_of::<PackedNodeI16>() as u32,
            },
        ];
        let nodes = [
            // Tree 0
            PackedNodeI16::split(500, 0, 1, 2),
            PackedNodeI16::leaf(-100), // -1.0
            PackedNodeI16::leaf(100),  // 1.0
            // Tree 1
            PackedNodeI16::leaf(50), // 0.5
        ];

        let mut buf = Vec::new();
        buf.extend_from_slice(as_bytes(&header));
        buf.extend_from_slice(as_bytes(&leaf_scale));
        for s in &feature_scales {
            buf.extend_from_slice(as_bytes(s));
        }
        for e in &entries {
            buf.extend_from_slice(as_bytes(e));
        }
        for n in &nodes {
            buf.extend_from_slice(as_bytes(n));
        }
        buf
    }

    #[test]
    fn parse_single_leaf_i16() {
        let buf = build_single_leaf_binary(42, 0.0, 100.0);
        let view = QuantizedEnsembleView::from_bytes(&buf).unwrap();
        assert_eq!(view.n_trees(), 1);
        assert_eq!(view.n_features(), 1);
        assert_eq!(view.total_nodes(), 1);
        assert_eq!(view.leaf_scale(), 100.0);
    }

    #[test]
    fn predict_single_leaf_i16() {
        // leaf=42, base=10.0, leaf_scale=100.0
        // prediction = 10.0 + 42/100.0 = 10.42
        let buf = build_single_leaf_binary(42, 10.0, 100.0);
        let view = QuantizedEnsembleView::from_bytes(&buf).unwrap();
        let pred = view.predict(&[0.0]);
        assert!((pred - 10.42).abs() < 1e-5, "expected 10.42, got {}", pred);
    }

    #[test]
    fn predict_one_split_left_i16() {
        let buf = build_one_split_binary();
        let view = QuantizedEnsembleView::from_bytes(&buf).unwrap();
        // feat[0]=3.0, scale=100.0 -> quantized=300, not > 500 -> left -> leaf=-100
        // prediction = 0.0 + (-100)/100.0 = -1.0
        let pred = view.predict(&[3.0, 0.0]);
        assert!((pred - (-1.0)).abs() < 1e-5, "expected -1.0, got {}", pred);
    }

    #[test]
    fn predict_one_split_right_i16() {
        let buf = build_one_split_binary();
        let view = QuantizedEnsembleView::from_bytes(&buf).unwrap();
        // feat[0]=7.0, scale=100.0 -> quantized=700, > 500 -> right -> leaf=100
        // prediction = 0.0 + 100/100.0 = 1.0
        let pred = view.predict(&[7.0, 0.0]);
        assert!((pred - 1.0).abs() < 1e-5, "expected 1.0, got {}", pred);
    }

    #[test]
    fn predict_two_trees_i16() {
        let buf = build_two_tree_binary();
        let view = QuantizedEnsembleView::from_bytes(&buf).unwrap();
        // feat[0]=3.0, scale=100 -> quantized=300, not > 500 -> tree0: left=-100
        // tree1: leaf=50
        // total = 1.0 + (-100 + 50)/100.0 = 1.0 + (-50)/100.0 = 1.0 - 0.5 = 0.5
        let pred = view.predict(&[3.0, 0.0]);
        assert!((pred - 0.5).abs() < 1e-5, "expected 0.5, got {}", pred);
    }

    #[test]
    fn predict_prequantized_matches_predict() {
        let buf = build_one_split_binary();
        let view = QuantizedEnsembleView::from_bytes(&buf).unwrap();

        // Test left path: feat[0]=3.0, scale=100.0 -> quantized=300
        let pred_inline = view.predict(&[3.0, 0.0]);
        let pred_preq = view.predict_prequantized(&[300, 0]);
        assert!(
            (pred_inline - pred_preq).abs() < 1e-5,
            "left: inline={}, prequantized={}",
            pred_inline,
            pred_preq
        );

        // Test right path: feat[0]=7.0, scale=100.0 -> quantized=700
        let pred_inline = view.predict(&[7.0, 0.0]);
        let pred_preq = view.predict_prequantized(&[700, 0]);
        assert!(
            (pred_inline - pred_preq).abs() < 1e-5,
            "right: inline={}, prequantized={}",
            pred_inline,
            pred_preq
        );
    }

    #[test]
    fn bad_magic_rejected_i16() {
        let mut buf = build_single_leaf_binary(0, 0.0, 100.0);
        buf[0] = 0xFF; // corrupt magic
        assert_eq!(
            QuantizedEnsembleView::from_bytes(&buf).unwrap_err(),
            FormatError::BadMagic
        );
    }

    #[test]
    fn truncated_rejected_i16() {
        let buf = build_single_leaf_binary(0, 0.0, 100.0);
        // Only pass the first 4 bytes — not even a full header
        assert_eq!(
            QuantizedEnsembleView::from_bytes(&buf[..4]).unwrap_err(),
            FormatError::Truncated
        );
    }

    #[test]
    fn debug_format_i16() {
        let buf = build_single_leaf_binary(0, 0.0, 100.0);
        let view = QuantizedEnsembleView::from_bytes(&buf).unwrap();
        let debug = format!("{:?}", view);
        assert!(
            debug.contains("QuantizedEnsembleView"),
            "missing struct name in debug: {}",
            debug
        );
        assert!(
            debug.contains("n_trees"),
            "missing n_trees in debug: {}",
            debug
        );
        assert!(
            debug.contains("leaf_scale"),
            "missing leaf_scale in debug: {}",
            debug
        );
    }

    #[test]
    fn predict_batch_matches_single_i16() {
        let buf = build_two_tree_binary();
        let view = QuantizedEnsembleView::from_bytes(&buf).unwrap();

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
    fn bad_version_rejected_i16() {
        let mut buf = build_single_leaf_binary(0, 0.0, 100.0);
        // version is at offset 4 (after magic u32), 2 bytes LE
        buf[4] = 99;
        buf[5] = 0;
        assert_eq!(
            QuantizedEnsembleView::from_bytes(&buf).unwrap_err(),
            FormatError::UnsupportedVersion
        );
    }

    #[test]
    fn invalid_child_index_rejected_i16() {
        let header = QuantizedEnsembleHeader {
            magic: QuantizedEnsembleHeader::MAGIC,
            version: QuantizedEnsembleHeader::VERSION,
            n_trees: 1,
            n_features: 2,
            _reserved: 0,
            base_prediction: 0.0,
        };
        let leaf_scale: f32 = 100.0;
        let feature_scales: [f32; 2] = [100.0, 100.0];
        let entry = TreeEntry {
            n_nodes: 3,
            offset: 0,
        };
        // Node 0 points to child 99, which is out of bounds (only 3 nodes)
        let nodes = [
            PackedNodeI16::split(500, 0, 1, 99),
            PackedNodeI16::leaf(-100),
            PackedNodeI16::leaf(100),
        ];

        let mut buf = Vec::new();
        buf.extend_from_slice(as_bytes(&header));
        buf.extend_from_slice(as_bytes(&leaf_scale));
        for s in &feature_scales {
            buf.extend_from_slice(as_bytes(s));
        }
        buf.extend_from_slice(as_bytes(&entry));
        for n in &nodes {
            buf.extend_from_slice(as_bytes(n));
        }

        assert_eq!(
            QuantizedEnsembleView::from_bytes(&buf).unwrap_err(),
            FormatError::InvalidNodeIndex
        );
    }
}
