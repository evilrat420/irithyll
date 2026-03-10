//! Concept drift detection algorithms.
//!
//! Drift detectors monitor a stream of error values and signal when the
//! underlying distribution has changed, triggering tree replacement in SGBT.

pub mod adwin;
pub mod pht;
pub mod ddm;

/// Signal emitted by a drift detector after observing a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftSignal {
    /// No significant change detected.
    Stable,
    /// Possible drift — start training an alternate model.
    Warning,
    /// Confirmed drift — replace the current model.
    Drift,
}

/// A sequential drift detector that monitors a stream of values.
pub trait DriftDetector: Send + Sync + 'static {
    /// Feed a new value and get the current drift signal.
    fn update(&mut self, value: f64) -> DriftSignal;

    /// Reset to initial state.
    fn reset(&mut self);

    /// Create a fresh detector with the same configuration.
    fn clone_fresh(&self) -> Box<dyn DriftDetector>;

    /// Current estimated mean of the monitored stream.
    fn estimated_mean(&self) -> f64;
}
