//! Minimal error types for the packed binary format.
//!
//! No `thiserror`, no `alloc` — just `core::fmt::Display` impls.

/// Errors that can occur when parsing or validating a packed ensemble binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatError {
    /// Magic bytes do not match `"IRIT"` (`0x54495249` LE).
    BadMagic,
    /// Format version is not supported by this build.
    UnsupportedVersion,
    /// Input buffer is too short to contain the declared structures.
    Truncated,
    /// Input buffer pointer is not aligned to 4 bytes.
    Unaligned,
    /// A node's child index points outside the node array bounds.
    InvalidNodeIndex,
    /// A node references a feature index >= `n_features`.
    InvalidFeatureIndex,
    /// A tree entry's byte offset is not aligned to `size_of::<PackedNode>()`.
    MisalignedTreeOffset,
}

impl core::fmt::Display for FormatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FormatError::BadMagic => write!(f, "bad magic: expected \"IRIT\""),
            FormatError::UnsupportedVersion => write!(f, "unsupported format version"),
            FormatError::Truncated => write!(f, "buffer truncated"),
            FormatError::Unaligned => write!(f, "buffer not 4-byte aligned"),
            FormatError::InvalidNodeIndex => write!(f, "node child index out of bounds"),
            FormatError::InvalidFeatureIndex => write!(f, "feature index exceeds n_features"),
            FormatError::MisalignedTreeOffset => {
                write!(f, "tree offset not aligned to node size")
            }
        }
    }
}
