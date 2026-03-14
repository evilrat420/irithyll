//! Core data types for streaming samples.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Observation trait -- zero-copy training interface
// ---------------------------------------------------------------------------

/// Trait for anything that can be used as a training observation.
///
/// Implementors provide a feature slice and a target value. The optional
/// [`weight`](Self::weight) method defaults to 1.0 for uniform weighting.
///
/// This trait enables zero-copy training: instead of requiring an owned
/// [`Sample`] (which forces a `Vec<f64>` allocation), callers can pass
/// borrowed slices directly via [`SampleRef`] or tuple impls.
///
/// # Built-in implementations
///
/// | Type | Allocates? |
/// |------|-----------|
/// | `Sample` | Already owns `Vec<f64>` |
/// | `SampleRef<'a>` | No -- borrows `&[f64]` |
/// | `(&[f64], f64)` | No -- tuple of slice + target |
/// | `(Vec<f64>, f64)` | Owns `Vec<f64>` |
///
/// # Example
///
/// ```
/// use irithyll::Observation;
///
/// // Zero-copy from a raw slice:
/// let features = [1.0, 2.0, 3.0];
/// let obs = (&features[..], 42.0);
/// assert_eq!(obs.features(), &[1.0, 2.0, 3.0]);
/// assert_eq!(obs.target(), 42.0);
/// assert_eq!(obs.weight(), 1.0);
/// ```
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
// Sample -- owned observation
// ---------------------------------------------------------------------------

/// A single observation with feature vector and target value.
///
/// For regression, `target` is the continuous value to predict.
/// For binary classification, `target` is 0.0 or 1.0.
/// For multi-class, `target` is the class index as f64 (0.0, 1.0, 2.0, ...).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Feature values for this observation.
    pub features: Vec<f64>,
    /// Target value (regression) or class label (classification).
    pub target: f64,
    /// Optional sample weight (default 1.0).
    pub weight: f64,
}

impl Sample {
    /// Create a new sample with unit weight.
    #[inline]
    pub fn new(features: Vec<f64>, target: f64) -> Self {
        Self {
            features,
            target,
            weight: 1.0,
        }
    }

    /// Create a new sample with explicit weight.
    #[inline]
    pub fn weighted(features: Vec<f64>, target: f64, weight: f64) -> Self {
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

impl std::fmt::Display for Sample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sample(features=[{}], target={})",
            self.features
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", "),
            self.target
        )
    }
}

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
// SampleRef -- zero-copy borrowing observation
// ---------------------------------------------------------------------------

/// A borrowed observation that avoids `Vec<f64>` allocation.
///
/// Use this when features are already available as a contiguous slice
/// (e.g., from Arrow arrays, memory-mapped data, or pre-allocated buffers).
///
/// # Example
///
/// ```
/// use irithyll::{Observation, SampleRef};
///
/// let features = [1.0, 2.0, 3.0];
/// let obs = SampleRef::new(&features, 42.0);
/// assert_eq!(obs.features(), &[1.0, 2.0, 3.0]);
/// assert_eq!(obs.target(), 42.0);
/// ```
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

impl Observation for (Vec<f64>, f64) {
    #[inline]
    fn features(&self) -> &[f64] {
        &self.0
    }
    #[inline]
    fn target(&self) -> f64 {
        self.1
    }
}

impl Observation for (&Vec<f64>, f64) {
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
// From impls -- conversion convenience
// ---------------------------------------------------------------------------

impl From<(Vec<f64>, f64)> for Sample {
    fn from((features, target): (Vec<f64>, f64)) -> Self {
        Sample::new(features, target)
    }
}

impl<'a> From<(&'a [f64], f64)> for SampleRef<'a> {
    fn from((features, target): (&'a [f64], f64)) -> Self {
        SampleRef::new(features, target)
    }
}
