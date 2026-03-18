//! 8-byte quantized packed node format for integer-only inference.
//!
//! # Binary Format
//!
//! ```text
//! [QuantizedEnsembleHeader: 16 bytes]
//! [leaf_scale: f32]                      // 4 bytes
//! [feature_scales: n_features × f32]     // n_features × 4 bytes
//! [TreeEntry × n_trees: 8 bytes each]
//! [PackedNodeI16 × total_nodes: 8 bytes each]
//! ```
//!
//! # Quantization scheme
//!
//! Per-feature quantization with shared scale: each feature gets its own scale
//! factor `scale_f = 32767.0 / max(|threshold_values_for_feature_f|)`. Thresholds
//! are stored as `(threshold * feature_scale[feat_idx]) as i16`. At predict time,
//! features are quantized once: `(feature * feature_scale[feat_idx]) as i16`.
//! Comparison is pure integer: `quantized_feature > quantized_threshold`.
//!
//! Leaf values use a separate global leaf scale: `leaf_i16 = (lr * leaf_f64 * leaf_scale) as i16`.
//! Final prediction: `base + sum(leaf_i16) / leaf_scale` — accumulate in i32,
//! dequantize once at the end.
//!
//! On Cortex-M0+ (no FPU), this eliminates all float ops from the hot loop.
//! The only float ops happen once at the end for dequantization.

/// 8-byte quantized decision tree node. Integer-only traversal for FPU-less targets.
///
/// 8 nodes per 64-byte cache line. All comparisons are i16 — no floating point
/// in the hot loop. On Cortex-M0+ (no FPU), this is ~25x faster than f32.
///
/// # Field layout
///
/// - `value`: quantized split threshold (internal nodes) or quantized leaf prediction (leaf nodes).
///   For splits: `(threshold * feature_scale[feat_idx]) as i16`.
///   For leaves: `(lr * leaf_f64 * leaf_scale) as i16`.
/// - `feature_flags`: bit 15 = is_leaf flag. Bits 14:0 = feature index (max 32767).
///   For leaves, the feature index is unused.
/// - `children`: packed left (low u16) and right (high u16) child indices.
///   For leaves, this field is unused (set to 0).
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedNodeI16 {
    /// Quantized split threshold (internal) or quantized leaf prediction (leaf).
    pub value: i16,
    /// Bit 15 = is_leaf. Bits 14:0 = feature index.
    pub feature_flags: u16,
    /// Packed children: left = low u16, right = high u16.
    pub children: u32,
}

impl PackedNodeI16 {
    /// Bit mask for the is_leaf flag in `feature_flags`.
    pub const LEAF_FLAG: u16 = 0x8000;

    /// Create a leaf node with a quantized prediction value.
    #[inline]
    pub const fn leaf(value: i16) -> Self {
        Self {
            value,
            children: 0,
            feature_flags: Self::LEAF_FLAG,
        }
    }

    /// Create an internal (split) node with a quantized threshold.
    #[inline]
    pub const fn split(threshold: i16, feature_idx: u16, left: u16, right: u16) -> Self {
        Self {
            value: threshold,
            children: (left as u32) | ((right as u32) << 16),
            feature_flags: feature_idx & 0x7FFF,
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

/// Header for quantized ensemble binary. 16 bytes, 4-byte aligned.
///
/// Uses magic `"IR16"` (0x49523136 LE) to distinguish from the f32 format (`"IRIT"`).
/// Appears at the start of every quantized binary, followed by `leaf_scale` (f32),
/// then `n_features` feature scale factors (f32 each), then the tree table and nodes.
///
/// Binary layout:
/// ```text
/// [QuantizedEnsembleHeader: 16 bytes]
/// [leaf_scale: f32]                      // 4 bytes
/// [feature_scales: n_features × f32]     // n_features × 4 bytes
/// [TreeEntry × n_trees: 8 bytes each]
/// [PackedNodeI16 × total_nodes: 8 bytes each]
/// ```
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantizedEnsembleHeader {
    /// Magic bytes: 0x49523136 ("IR16" in LE ASCII).
    pub magic: u32,
    /// Format version (1).
    pub version: u16,
    /// Number of trees.
    pub n_trees: u16,
    /// Number of input features.
    pub n_features: u16,
    /// Reserved.
    pub _reserved: u16,
    /// Base prediction (f32).
    pub base_prediction: f32,
}

impl QuantizedEnsembleHeader {
    /// Magic value: "IR16" in little-endian ASCII.
    pub const MAGIC: u32 = u32::from_le_bytes(*b"IR16");
    /// Current format version.
    pub const VERSION: u16 = 1;
}

// Re-export TreeEntry for convenience — same struct as packed.rs.
pub use crate::packed::TreeEntry as QuantizedTreeEntry;

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn packed_node_i16_is_8_bytes() {
        assert_eq!(size_of::<PackedNodeI16>(), 8);
    }

    #[test]
    fn packed_node_i16_alignment_is_4() {
        assert_eq!(align_of::<PackedNodeI16>(), 4);
    }

    #[test]
    fn quantized_header_is_16_bytes() {
        assert_eq!(size_of::<QuantizedEnsembleHeader>(), 16);
    }

    #[test]
    fn leaf_node_i16_roundtrip() {
        let node = PackedNodeI16::leaf(1234);
        assert!(node.is_leaf());
        assert_eq!(node.value, 1234);
        assert_eq!(node.children, 0);
    }

    #[test]
    fn split_node_i16_roundtrip() {
        let node = PackedNodeI16::split(5000, 7, 1, 2);
        assert!(!node.is_leaf());
        assert_eq!(node.feature_idx(), 7);
        assert_eq!(node.value, 5000);
        assert_eq!(node.left_child(), 1);
        assert_eq!(node.right_child(), 2);
    }

    #[test]
    fn max_feature_index_i16() {
        // 15 bits = max 32767
        let node = PackedNodeI16::split(0, 0x7FFF, 0, 0);
        assert_eq!(node.feature_idx(), 0x7FFF);
        assert!(!node.is_leaf());
    }

    #[test]
    fn max_child_indices_i16() {
        let node = PackedNodeI16::split(0, 0, u16::MAX, u16::MAX);
        assert_eq!(node.left_child(), u16::MAX);
        assert_eq!(node.right_child(), u16::MAX);
    }

    #[test]
    fn eight_nodes_per_cache_line() {
        // 8 x 8 = 64 bytes, fits exactly in 64-byte cache line
        assert!(8 * size_of::<PackedNodeI16>() <= 64);
    }

    #[test]
    fn header_magic_is_ir16() {
        // "IR16" as little-endian bytes: I=0x49, R=0x52, 1=0x31, 6=0x36
        let bytes = QuantizedEnsembleHeader::MAGIC.to_le_bytes();
        assert_eq!(&bytes, b"IR16");
    }

    #[test]
    fn leaf_negative_value_roundtrip() {
        let node = PackedNodeI16::leaf(-32000);
        assert!(node.is_leaf());
        assert_eq!(node.value, -32000);
    }

    #[test]
    fn split_negative_threshold_roundtrip() {
        let node = PackedNodeI16::split(-16000, 3, 10, 20);
        assert!(!node.is_leaf());
        assert_eq!(node.value, -16000);
        assert_eq!(node.feature_idx(), 3);
        assert_eq!(node.left_child(), 10);
        assert_eq!(node.right_child(), 20);
    }

    #[test]
    fn tree_entry_reused_from_packed() {
        // Verify QuantizedTreeEntry is the same type as TreeEntry
        let entry: QuantizedTreeEntry = QuantizedTreeEntry {
            n_nodes: 5,
            offset: 40,
        };
        assert_eq!(entry.n_nodes, 5);
        assert_eq!(entry.offset, 40);
        assert_eq!(size_of::<QuantizedTreeEntry>(), 8);
    }
}
