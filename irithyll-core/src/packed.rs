//! 12-byte packed node format and ensemble binary layout.
//!
//! # Binary Format
//!
//! ```text
//! [EnsembleHeader: 16 bytes]
//! [TreeEntry × n_trees: 8 bytes each]
//! [PackedNode × total_nodes: 12 bytes each]
//! ```
//!
//! Learning rate is baked into leaf values at export time, eliminating one
//! multiply per tree during inference.

/// 12-byte packed decision tree node. AoS layout for cache-optimal inference.
///
/// 5 nodes per 64-byte cache line (60 bytes used, 4 bytes padding).
/// All fields for one traversal step are adjacent in memory — no cross-vector
/// striding like the SoA `TreeArena` used during training.
///
/// # Field layout
///
/// - `value`: split threshold (internal nodes) or leaf prediction (leaf nodes).
///   For leaves, the learning rate is already baked in: `value = lr * leaf_f64 as f32`.
/// - `children`: packed left (low u16) and right (high u16) child indices.
///   For leaves, this field is unused (set to 0).
/// - `feature_flags`: bit 15 = is_leaf flag. Bits 14:0 = feature index (max 32767).
///   For leaves, the feature index is unused.
/// - `_reserved`: padding for future use (categorical flag, metadata, etc.).
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedNode {
    /// Split threshold (internal) or prediction value (leaf, with lr baked in).
    pub value: f32,
    /// Packed children: left = low u16, right = high u16.
    pub children: u32,
    /// Bit 15 = is_leaf. Bits 14:0 = feature index.
    pub feature_flags: u16,
    /// Reserved for future use.
    pub _reserved: u16,
}

impl PackedNode {
    /// Bit mask for the is_leaf flag in `feature_flags`.
    pub const LEAF_FLAG: u16 = 0x8000;

    /// Create a leaf node with a prediction value.
    #[inline]
    pub const fn leaf(value: f32) -> Self {
        Self {
            value,
            children: 0,
            feature_flags: Self::LEAF_FLAG,
            _reserved: 0,
        }
    }

    /// Create an internal (split) node.
    #[inline]
    pub const fn split(threshold: f32, feature_idx: u16, left: u16, right: u16) -> Self {
        Self {
            value: threshold,
            children: (left as u32) | ((right as u32) << 16),
            feature_flags: feature_idx & 0x7FFF,
            _reserved: 0,
        }
    }

    /// Returns `true` if this is a leaf node.
    #[inline]
    pub const fn is_leaf(&self) -> bool {
        self.feature_flags & Self::LEAF_FLAG != 0
    }

    /// Feature index (bits 14:0). Only meaningful for internal nodes.
    #[inline]
    pub const fn feature_idx(&self) -> u16 {
        self.feature_flags & 0x7FFF
    }

    /// Left child index (low 16 bits of `children`).
    #[inline]
    pub const fn left_child(&self) -> u16 {
        self.children as u16
    }

    /// Right child index (high 16 bits of `children`).
    #[inline]
    pub const fn right_child(&self) -> u16 {
        (self.children >> 16) as u16
    }
}

/// Header for the packed ensemble binary format. 16 bytes, 4-byte aligned.
///
/// Appears at the start of every packed binary. Followed by `n_trees` [`TreeEntry`]
/// records and then contiguous [`PackedNode`] arrays.
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnsembleHeader {
    /// Magic bytes: `"IRIT"` in ASCII (little-endian u32: `0x54495249`).
    pub magic: u32,
    /// Binary format version. Currently `1`.
    pub version: u16,
    /// Number of trees in the ensemble.
    pub n_trees: u16,
    /// Expected number of input features.
    pub n_features: u16,
    /// Reserved padding.
    pub _reserved: u16,
    /// Base prediction (f64 quantized to f32). Added to the sum of tree predictions.
    pub base_prediction: f32,
}

impl EnsembleHeader {
    /// Magic value: "IRIT" in little-endian ASCII.
    pub const MAGIC: u32 = u32::from_le_bytes(*b"IRIT");
    /// Current format version.
    pub const VERSION: u16 = 1;
}

/// Tree table entry: metadata for one tree in the ensemble. 8 bytes.
///
/// The `n_nodes` field gives the number of [`PackedNode`]s in this tree.
/// The `offset` field is the byte offset from the start of the node data
/// region to this tree's first node.
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TreeEntry {
    /// Number of nodes in this tree.
    pub n_nodes: u32,
    /// Byte offset from nodes_base to this tree's first PackedNode.
    pub offset: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn packed_node_is_12_bytes() {
        assert_eq!(size_of::<PackedNode>(), 12);
    }

    #[test]
    fn packed_node_alignment_is_4() {
        assert_eq!(align_of::<PackedNode>(), 4);
    }

    #[test]
    fn ensemble_header_is_16_bytes() {
        assert_eq!(size_of::<EnsembleHeader>(), 16);
    }

    #[test]
    fn tree_entry_is_8_bytes() {
        assert_eq!(size_of::<TreeEntry>(), 8);
    }

    #[test]
    fn leaf_node_roundtrip() {
        let node = PackedNode::leaf(0.42);
        assert!(node.is_leaf());
        assert_eq!(node.value, 0.42);
        assert_eq!(node.children, 0);
    }

    #[test]
    fn split_node_roundtrip() {
        let node = PackedNode::split(1.5, 7, 1, 2);
        assert!(!node.is_leaf());
        assert_eq!(node.feature_idx(), 7);
        assert_eq!(node.value, 1.5);
        assert_eq!(node.left_child(), 1);
        assert_eq!(node.right_child(), 2);
    }

    #[test]
    fn max_feature_index() {
        // 15 bits = max 32767
        let node = PackedNode::split(0.0, 0x7FFF, 0, 0);
        assert_eq!(node.feature_idx(), 0x7FFF);
        assert!(!node.is_leaf());
    }

    #[test]
    fn max_child_indices() {
        let node = PackedNode::split(0.0, 0, u16::MAX, u16::MAX);
        assert_eq!(node.left_child(), u16::MAX);
        assert_eq!(node.right_child(), u16::MAX);
    }

    #[test]
    fn five_nodes_per_cache_line() {
        // 5 × 12 = 60 bytes, fits in 64-byte cache line
        assert!(5 * size_of::<PackedNode>() <= 64);
        // 6 × 12 = 72 bytes, does NOT fit
        assert!(6 * size_of::<PackedNode>() > 64);
    }

    #[test]
    fn header_magic_is_irit() {
        // "IRIT" as little-endian bytes: I=0x49, R=0x52, I=0x49, T=0x54
        let bytes = EnsembleHeader::MAGIC.to_le_bytes();
        assert_eq!(&bytes, b"IRIT");
    }
}
