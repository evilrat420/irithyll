//! Concept drift detection algorithms.
//!
//! Drift detectors monitor a stream of error values and signal when the
//! underlying distribution has changed, triggering tree replacement in SGBT.
//!
//! This is the `no_std`-compatible core module. The [`DriftDetector`] trait
//! requires the `alloc` feature for `Box`-returning methods; the signal enum
//! and state types are always available.

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub mod adwin;
pub mod ddm;
pub mod pht;

// ---------------------------------------------------------------------------
// DriftSignal
// ---------------------------------------------------------------------------

/// Signal emitted by a drift detector after observing a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DriftSignal {
    /// No significant change detected.
    Stable,
    /// Possible drift -- start training an alternate model.
    Warning,
    /// Confirmed drift -- replace the current model.
    Drift,
}

impl core::fmt::Display for DriftSignal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Stable => write!(f, "Stable"),
            Self::Warning => write!(f, "Warning"),
            Self::Drift => write!(f, "Drift"),
        }
    }
}

// ---------------------------------------------------------------------------
// DriftDetectorState (requires alloc for Vec/String)
// ---------------------------------------------------------------------------

/// Serializable state for any drift detector variant.
///
/// Captures running statistics so that save/load cycles preserve accumulated
/// warmup data instead of creating a fresh detector (which would cause
/// spurious drift signals).
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        rows: alloc::vec::Vec<alloc::vec::Vec<AdwinBucketState>>,
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

/// Serializable snapshot of a bucket in ADWIN's exponential histogram.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AdwinBucketState {
    pub total: f64,
    pub variance: f64,
    pub count: u64,
}

// ---------------------------------------------------------------------------
// DriftDetector trait (requires alloc for Box)
// ---------------------------------------------------------------------------

/// A sequential drift detector that monitors a stream of values.
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub trait DriftDetector: Send + Sync + 'static {
    /// Feed a new value and get the current drift signal.
    fn update(&mut self, value: f64) -> DriftSignal;

    /// Reset to initial state.
    fn reset(&mut self);

    /// Create a fresh detector with the same configuration but no state.
    fn clone_fresh(&self) -> alloc::boxed::Box<dyn DriftDetector>;

    /// Clone this detector including its internal state.
    ///
    /// Unlike [`clone_fresh`](Self::clone_fresh), this preserves accumulated
    /// statistics (running means, counters, etc.), producing a true deep copy.
    fn clone_boxed(&self) -> alloc::boxed::Box<dyn DriftDetector>;

    /// Current estimated mean of the monitored stream.
    fn estimated_mean(&self) -> f64;

    /// Serialize the detector's internal state for model persistence.
    ///
    /// Returns `None` if the detector does not support state serialization.
    /// Default implementation returns `None`.
    fn serialize_state(&self) -> Option<DriftDetectorState> {
        None
    }

    /// Restore the detector's internal state from a serialized snapshot.
    ///
    /// Returns `true` if the state was successfully restored. Default
    /// implementation returns `false` (unsupported).
    fn restore_state(&mut self, _state: &DriftDetectorState) -> bool {
        false
    }
}
