//! Display implementations for SGBT config types.

use super::{DriftDetectorType, SGBTConfig};
use crate::alloc::fmt;

impl fmt::Display for DriftDetectorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PageHinkley { delta, lambda } => {
                write!(f, "PageHinkley(delta={}, lambda={})", delta, lambda)
            }
            Self::Adwin { delta } => write!(f, "Adwin(delta={})", delta),
            Self::Ddm {
                warning_level,
                drift_level,
                min_instances,
            } => write!(
                f,
                "Ddm(warning={}, drift={}, min_instances={})",
                warning_level, drift_level, min_instances
            ),
        }
    }
}

impl fmt::Display for SGBTConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SGBTConfig {{ steps={}, lr={}, depth={}, bins={}, variant={}, drift={} }}",
            self.n_steps,
            self.learning_rate,
            self.max_depth,
            self.n_bins,
            self.variant,
            self.drift_detector,
        )
    }
}
