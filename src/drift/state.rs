//! Serializable drift detector state for model persistence.
//!
//! Captures the internal state of each drift detector variant so that
//! save/load cycles preserve accumulated statistics instead of creating
//! a fresh detector (which would lose warmup data and cause spurious drift).

use serde::{Deserialize, Serialize};

/// Serializable snapshot of a bucket in ADWIN's exponential histogram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdwinBucketState {
    pub total: f64,
    pub variance: f64,
    pub count: u64,
}

/// Serializable state for any drift detector variant.
///
/// Mirrors [`DriftDetectorType`](crate::ensemble::config::DriftDetectorType)
/// but captures running statistics instead of configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDetectorState {
    /// Page-Hinkley Test accumulated state.
    PageHinkley {
        running_mean: f64,
        sum_up: f64,
        min_sum_up: f64,
        sum_down: f64,
        min_sum_down: f64,
        count: u64,
    },
    /// ADWIN exponential histogram state.
    Adwin {
        rows: Vec<Vec<AdwinBucketState>>,
        total: f64,
        variance: f64,
        count: u64,
        width: u64,
    },
    /// DDM Welford running statistics state.
    Ddm {
        mean: f64,
        m2: f64,
        count: u64,
        min_p_plus_s: f64,
        min_s: f64,
    },
}
