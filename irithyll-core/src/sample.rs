//! Core observation trait and zero-copy sample types.

// ---------------------------------------------------------------------------
// Observation trait -- zero-copy training interface
// ---------------------------------------------------------------------------

/// Trait for anything that can be used as a training observation.
///
/// Implementors provide a feature slice and a target value. The optional
/// [`weight`](Self::weight) method defaults to 1.0 for uniform weighting.
///
/// This trait enables zero-copy training: callers can pass borrowed slices
/// directly via [`SampleRef`] or tuple impls without allocating.
///
/// # Built-in implementations
///
/// | Type | Allocates? |
/// |------|-----------|
/// | `SampleRef<'a>` | No -- borrows `&[f64]` |
/// | `(&[f64], f64)` | No -- tuple of slice + target |
/// | `(Vec<f64>, f64)` | Owns `Vec<f64>` (requires `alloc` feature) |
pub trait Observation {
    /// The feature values for this observation.
    fn features(&self) -> &[f64];
    /// The target value (regression) or class label (classification).
    fn target(&self) -> f64;
    /// Optional sample weight. Defaults to 1.0 (uniform).
    fn weight(&self) -> f64 {
        1.0
    }
}

// ---------------------------------------------------------------------------
// SampleRef -- zero-copy borrowing observation
// ---------------------------------------------------------------------------

/// A borrowed observation that avoids `Vec<f64>` allocation.
///
/// Use this when features are already available as a contiguous slice
/// (e.g., from Arrow arrays, memory-mapped data, or pre-allocated buffers).
#[derive(Debug, Clone, Copy)]
pub struct SampleRef<'a> {
    /// Borrowed feature slice.
    pub features: &'a [f64],
    /// Target value.
    pub target: f64,
    /// Sample weight (default 1.0).
    pub weight: f64,
}

impl<'a> SampleRef<'a> {
    /// Create a new sample reference with unit weight.
    #[inline]
    pub fn new(features: &'a [f64], target: f64) -> Self {
        Self {
            features,
            target,
            weight: 1.0,
        }
    }

    /// Create a new sample reference with explicit weight.
    #[inline]
    pub fn weighted(features: &'a [f64], target: f64, weight: f64) -> Self {
        Self {
            features,
            target,
            weight,
        }
    }
}

impl<'a> Observation for SampleRef<'a> {
    #[inline]
    fn features(&self) -> &[f64] {
        self.features
    }
    #[inline]
    fn target(&self) -> f64 {
        self.target
    }
    #[inline]
    fn weight(&self) -> f64 {
        self.weight
    }
}

// ---------------------------------------------------------------------------
// Tuple impls -- quick-and-dirty observations
// ---------------------------------------------------------------------------

impl Observation for (&[f64], f64) {
    #[inline]
    fn features(&self) -> &[f64] {
        self.0
    }
    #[inline]
    fn target(&self) -> f64 {
        self.1
    }
}

#[cfg(feature = "alloc")]
impl Observation for (alloc::vec::Vec<f64>, f64) {
    #[inline]
    fn features(&self) -> &[f64] {
        &self.0
    }
    #[inline]
    fn target(&self) -> f64 {
        self.1
    }
}

#[cfg(feature = "alloc")]
impl Observation for (&alloc::vec::Vec<f64>, f64) {
    #[inline]
    fn features(&self) -> &[f64] {
        self.0
    }
    #[inline]
    fn target(&self) -> f64 {
        self.1
    }
}

// ---------------------------------------------------------------------------
// Sample -- owned observation (requires alloc)
// ---------------------------------------------------------------------------

/// A single owned observation with feature vector and target value.
///
/// For regression, `target` is the continuous value to predict.
/// For binary classification, `target` is 0.0 or 1.0.
/// For multi-class, `target` is the class index as f64 (0.0, 1.0, 2.0, ...).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sample {
    /// Feature values for this observation.
    pub features: alloc::vec::Vec<f64>,
    /// Target value (regression) or class label (classification).
    pub target: f64,
    /// Optional sample weight (default 1.0).
    pub weight: f64,
}

#[cfg(feature = "alloc")]
impl Sample {
    /// Create a new sample with unit weight.
    #[inline]
    pub fn new(features: alloc::vec::Vec<f64>, target: f64) -> Self {
        Self {
            features,
            target,
            weight: 1.0,
        }
    }

    /// Create a new sample with explicit weight.
    #[inline]
    pub fn weighted(features: alloc::vec::Vec<f64>, target: f64, weight: f64) -> Self {
        Self {
            features,
            target,
            weight,
        }
    }

    /// Number of features in this sample.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.features.len()
    }
}

#[cfg(feature = "alloc")]
impl Observation for Sample {
    #[inline]
    fn features(&self) -> &[f64] {
        &self.features
    }
    #[inline]
    fn target(&self) -> f64 {
        self.target
    }
    #[inline]
    fn weight(&self) -> f64 {
        self.weight
    }
}

#[cfg(feature = "alloc")]
impl Observation for &Sample {
    #[inline]
    fn features(&self) -> &[f64] {
        &self.features
    }
    #[inline]
    fn target(&self) -> f64 {
        self.target
    }
    #[inline]
    fn weight(&self) -> f64 {
        self.weight
    }
}

// ---------------------------------------------------------------------------
// From impls -- conversion convenience
// ---------------------------------------------------------------------------

impl<'a> From<(&'a [f64], f64)> for SampleRef<'a> {
    fn from((features, target): (&'a [f64], f64)) -> Self {
        SampleRef::new(features, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_ref_new() {
        let features = [1.0, 2.0, 3.0];
        let obs = SampleRef::new(&features, 42.0);
        assert_eq!(obs.features(), &[1.0, 2.0, 3.0]);
        assert_eq!(obs.target(), 42.0);
        assert!((obs.weight() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sample_ref_weighted() {
        let features = [1.0, 2.0];
        let obs = SampleRef::weighted(&features, 5.0, 0.5);
        assert_eq!(obs.features(), &[1.0, 2.0]);
        assert_eq!(obs.target(), 5.0);
        assert!((obs.weight() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_tuple_slice_observation() {
        let features = [1.0, 2.0, 3.0];
        let obs = (&features[..], 42.0);
        assert_eq!(obs.features(), &[1.0, 2.0, 3.0]);
        assert_eq!(obs.target(), 42.0);
        assert!((obs.weight() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_from_tuple_to_sample_ref() {
        let features = [1.0, 2.0];
        let obs: SampleRef = (&features[..], 5.0).into();
        assert_eq!(obs.features(), &[1.0, 2.0]);
        assert_eq!(obs.target(), 5.0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_vec_tuple_observation() {
        let obs = (alloc::vec![1.0, 2.0, 3.0], 42.0);
        assert_eq!(obs.features(), &[1.0, 2.0, 3.0]);
        assert_eq!(obs.target(), 42.0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_vec_ref_tuple_observation() {
        let v = alloc::vec![1.0, 2.0, 3.0];
        let obs = (&v, 42.0);
        assert_eq!(obs.features(), &[1.0, 2.0, 3.0]);
        assert_eq!(obs.target(), 42.0);
    }
}
