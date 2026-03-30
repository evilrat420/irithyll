//! Concept drift detection algorithms.
//!
//! Drift detectors monitor a stream of error values and signal when the
//! underlying distribution has changed, triggering tree replacement in SGBT.
//!
//! Core types and implementations are defined in `irithyll-core` and
//! re-exported here for backward compatibility.

// Re-export core drift types and sub-modules.
pub use irithyll_core::drift::AdwinBucketState;
pub use irithyll_core::drift::DriftDetector;
pub use irithyll_core::drift::DriftDetectorState;
pub use irithyll_core::drift::DriftSignal;

pub use irithyll_core::drift::adwin;
pub use irithyll_core::drift::ddm;
pub use irithyll_core::drift::pht;

/// Backward-compatible `state` module re-exporting serializable drift state types.
pub mod state {
    pub use irithyll_core::drift::AdwinBucketState;
    pub use irithyll_core::drift::DriftDetectorState;
}
