//! Feature type declarations for streaming tree construction.

/// Declares whether a feature is continuous (default) or categorical.
///
/// Categorical features are handled differently in the tree construction:
/// - **Binning:** One bin per observed category value instead of equal-width bins.
/// - **Split evaluation:** Fisher optimal binary partitioning -- categories are sorted
///   by gradient\_sum/hessian\_sum ratio, then the best contiguous partition is found
///   using the same left-to-right XGBoost gain scan.
/// - **Routing:** Categorical splits use a `u64` bitmask where bit `i` set means
///   category `i` goes left. This supports up to 64 distinct category values per feature.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FeatureType {
    /// Numeric feature split by threshold comparisons (default).
    #[default]
    Continuous,
    /// Categorical feature split by bitmask partitioning.
    Categorical,
}
