//! Concept drift detection algorithms.
//!
//! Drift detectors monitor a stream of error values and signal when the
//! underlying distribution has changed, triggering tree replacement in SGBT.

pub mod adwin;
pub mod ddm;
pub mod pht;
pub mod state;

/// Signal emitted by a drift detector after observing a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftSignal {
    /// No significant change detected.
    Stable,
    /// Possible drift -- start training an alternate model.
    Warning,
    /// Confirmed drift -- replace the current model.
    Drift,
}

impl std::fmt::Display for DriftSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "Stable"),
            Self::Warning => write!(f, "Warning"),
            Self::Drift => write!(f, "Drift"),
        }
    }
}

/// A sequential drift detector that monitors a stream of values.
pub trait DriftDetector: Send + Sync + 'static {
    /// Feed a new value and get the current drift signal.
    fn update(&mut self, value: f64) -> DriftSignal;

    /// Reset to initial state.
    fn reset(&mut self);

    /// Create a fresh detector with the same configuration but no state.
    fn clone_fresh(&self) -> Box<dyn DriftDetector>;

    /// Clone this detector including its internal state.
    ///
    /// Unlike [`clone_fresh`](Self::clone_fresh), this preserves accumulated
    /// statistics (running means, counters, etc.), producing a true deep copy.
    fn clone_boxed(&self) -> Box<dyn DriftDetector>;

    /// Current estimated mean of the monitored stream.
    fn estimated_mean(&self) -> f64;

    /// Serialize the detector's internal state for model persistence.
    ///
    /// Returns `None` if the detector does not support state serialization.
    /// Default implementation returns `None`.
    fn serialize_state(&self) -> Option<state::DriftDetectorState> {
        None
    }

    /// Restore the detector's internal state from a serialized snapshot.
    ///
    /// Returns `true` if the state was successfully restored. Default
    /// implementation returns `false` (unsupported).
    fn restore_state(&mut self, _state: &state::DriftDetectorState) -> bool {
        false
    }
}
