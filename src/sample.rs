//! Core data types for streaming samples.

use serde::{Deserialize, Serialize};

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
