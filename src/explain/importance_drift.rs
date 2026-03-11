//! Per-feature SHAP importance drift detection.
//!
//! Monitors per-feature |SHAP| values over time using drift detectors (PHT,
//! ADWIN, or DDM). Detects when the model's reasoning changes — e.g. a feature
//! that was previously unimportant becomes dominant, or vice versa — potentially
//! before accuracy degrades.
//!
//! This is a novel capability: no existing open-source streaming ML library
//! offers per-feature SHAP-based drift detection.
//!
//! # Usage
//!
//! ```text
//! let mut monitor = ImportanceDriftMonitor::new(n_features, detector_type);
//!
//! for sample in stream {
//!     let shap = model.explain(&sample.features);
//!     if let Some(report) = monitor.update(&shap) {
//!         if !report.drifted_features.is_empty() {
//!             println!("Feature importance drift on: {:?}", report.drifted_features);
//!         }
//!     }
//! }
//! ```

use crate::drift::{DriftDetector, DriftSignal};
use crate::ensemble::config::DriftDetectorType;
use crate::explain::treeshap::ShapValues;

/// Per-feature drift signal report.
///
/// Returned by [`ImportanceDriftMonitor::update`] when a SHAP explanation
/// is actually processed (not skipped by sample rate).
#[derive(Debug, Clone)]
pub struct ImportanceDriftReport {
    /// Drift signal for each feature.
    pub signals: Vec<DriftSignal>,
    /// Current estimated mean |SHAP| per feature from the detectors.
    pub means: Vec<f64>,
    /// Indices of features currently signaling Drift.
    pub drifted_features: Vec<usize>,
    /// Indices of features currently signaling Warning.
    pub warning_features: Vec<usize>,
}

/// Monitors per-feature |SHAP| values for importance drift.
///
/// Wraps one drift detector per feature. Feed SHAP explanations
/// (optionally sampled) and query which features have drifted.
///
/// This detects "model right for wrong reasons" scenarios where
/// accuracy hasn't degraded yet but the model's reasoning has shifted.
///
/// # Sample Rate
///
/// SHAP computation is O(L * D^2) per tree. Use `with_sample_rate` to
/// only process every Nth explanation, amortizing the cost while still
/// tracking importance trends.
pub struct ImportanceDriftMonitor {
    detectors: Vec<Box<dyn DriftDetector>>,
    detector_type: DriftDetectorType,
    n_features: usize,
    sample_rate: usize,
    sample_counter: u64,
    total_updates: u64,
}

// Manual Debug implementation since Box<dyn DriftDetector> doesn't derive Debug.
impl std::fmt::Debug for ImportanceDriftMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportanceDriftMonitor")
            .field("n_features", &self.n_features)
            .field("sample_rate", &self.sample_rate)
            .field("sample_counter", &self.sample_counter)
            .field("total_updates", &self.total_updates)
            .field("detector_type", &self.detector_type)
            .finish()
    }
}

impl ImportanceDriftMonitor {
    /// Create a monitor with one drift detector per feature.
    ///
    /// Every SHAP explanation will be processed (sample_rate = 1).
    pub fn new(n_features: usize, detector_type: DriftDetectorType) -> Self {
        Self::with_sample_rate(n_features, detector_type, 1)
    }

    /// Create a monitor that only processes every `sample_rate`-th explanation.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is zero or `n_features` is zero.
    pub fn with_sample_rate(
        n_features: usize,
        detector_type: DriftDetectorType,
        sample_rate: usize,
    ) -> Self {
        assert!(n_features > 0, "n_features must be > 0");
        assert!(sample_rate > 0, "sample_rate must be > 0");
        let detectors = (0..n_features).map(|_| detector_type.create()).collect();
        Self {
            detectors,
            detector_type,
            n_features,
            sample_rate,
            sample_counter: 0,
            total_updates: 0,
        }
    }

    /// Feed a SHAP explanation to the monitor.
    ///
    /// Returns `Some(report)` when the explanation is actually processed
    /// (according to `sample_rate`), or `None` when skipped.
    pub fn update(&mut self, shap: &ShapValues) -> Option<ImportanceDriftReport> {
        self.sample_counter += 1;
        if self.sample_counter % self.sample_rate as u64 != 0 {
            return None;
        }
        self.total_updates += 1;

        let mut signals = Vec::with_capacity(self.n_features);
        let mut means = Vec::with_capacity(self.n_features);
        let mut drifted = Vec::new();
        let mut warning = Vec::new();

        for (i, detector) in self.detectors.iter_mut().enumerate() {
            let abs_shap = shap.values.get(i).map(|v| v.abs()).unwrap_or(0.0);
            let signal = detector.update(abs_shap);
            means.push(detector.estimated_mean());

            match signal {
                DriftSignal::Drift => drifted.push(i),
                DriftSignal::Warning => warning.push(i),
                DriftSignal::Stable => {}
            }
            signals.push(signal);
        }

        Some(ImportanceDriftReport {
            signals,
            means,
            drifted_features: drifted,
            warning_features: warning,
        })
    }

    /// Reset all per-feature detectors to their initial state.
    pub fn reset(&mut self) {
        self.detectors = (0..self.n_features)
            .map(|_| self.detector_type.create())
            .collect();
        self.sample_counter = 0;
        self.total_updates = 0;
    }

    /// Number of features being monitored.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of SHAP explanations actually processed (after sampling).
    pub fn n_updates(&self) -> u64 {
        self.total_updates
    }

    /// Current estimated mean |SHAP| per feature from each detector.
    pub fn feature_means(&self) -> Vec<f64> {
        self.detectors.iter().map(|d| d.estimated_mean()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pht_type() -> DriftDetectorType {
        DriftDetectorType::PageHinkley {
            delta: 0.005,
            lambda: 50.0,
        }
    }

    #[test]
    fn stable_shap_no_drift() {
        let mut monitor = ImportanceDriftMonitor::new(3, pht_type());
        for _ in 0..500 {
            let shap = ShapValues {
                values: vec![1.0, 0.5, 0.3],
                base_value: 0.0,
            };
            if let Some(report) = monitor.update(&shap) {
                assert!(
                    report.drifted_features.is_empty(),
                    "unexpected drift: {:?}",
                    report.drifted_features
                );
            }
        }
        assert_eq!(monitor.n_updates(), 500);
    }

    #[test]
    fn shifted_shap_detects_drift() {
        let mut monitor = ImportanceDriftMonitor::new(3, pht_type());

        // Phase 1: stable importance
        for _ in 0..500 {
            let shap = ShapValues {
                values: vec![1.0, 0.5, 0.3],
                base_value: 0.0,
            };
            monitor.update(&shap);
        }

        // Phase 2: feature 0 jumps to 10x
        let mut drift_detected = false;
        for _ in 0..500 {
            let shap = ShapValues {
                values: vec![10.0, 0.5, 0.3],
                base_value: 0.0,
            };
            if let Some(report) = monitor.update(&shap) {
                if report.drifted_features.contains(&0) {
                    drift_detected = true;
                }
                // Features 1 and 2 should remain stable
                assert!(
                    !report.drifted_features.contains(&1),
                    "feature 1 should not drift"
                );
                assert!(
                    !report.drifted_features.contains(&2),
                    "feature 2 should not drift"
                );
            }
        }
        assert!(drift_detected, "drift should be detected on feature 0");
    }

    #[test]
    fn sample_rate_skips() {
        let mut monitor = ImportanceDriftMonitor::with_sample_rate(2, pht_type(), 5);
        let shap = ShapValues {
            values: vec![1.0, 0.5],
            base_value: 0.0,
        };

        let mut processed = 0;
        let mut skipped = 0;
        for _ in 0..20 {
            match monitor.update(&shap) {
                Some(_) => processed += 1,
                None => skipped += 1,
            }
        }
        assert_eq!(processed, 4); // 20 / 5 = 4
        assert_eq!(skipped, 16);
        assert_eq!(monitor.n_updates(), 4);
    }

    #[test]
    fn reset_clears_state() {
        let mut monitor = ImportanceDriftMonitor::new(2, pht_type());
        for _ in 0..100 {
            monitor.update(&ShapValues {
                values: vec![1.0, 2.0],
                base_value: 0.0,
            });
        }
        monitor.reset();
        assert_eq!(monitor.n_updates(), 0);
        assert_eq!(monitor.n_features(), 2);
    }

    #[test]
    fn feature_means_reports() {
        let mut monitor = ImportanceDriftMonitor::new(2, pht_type());
        for _ in 0..100 {
            monitor.update(&ShapValues {
                values: vec![5.0, 2.0],
                base_value: 0.0,
            });
        }
        let means = monitor.feature_means();
        assert_eq!(means.len(), 2);
        // After 100 constant inputs, mean should be close to the input values
        assert!((means[0] - 5.0).abs() < 1.0);
        assert!((means[1] - 2.0).abs() < 1.0);
    }

    #[test]
    fn handles_fewer_shap_values_than_features() {
        let mut monitor = ImportanceDriftMonitor::new(5, pht_type());
        // Only 2 values but 5 features — remaining should get 0.0
        let shap = ShapValues {
            values: vec![1.0, 2.0],
            base_value: 0.0,
        };
        let report = monitor.update(&shap).unwrap();
        assert_eq!(report.signals.len(), 5);
    }

    #[test]
    #[should_panic(expected = "n_features must be > 0")]
    fn zero_features_panics() {
        ImportanceDriftMonitor::new(0, pht_type());
    }

    #[test]
    #[should_panic(expected = "sample_rate must be > 0")]
    fn zero_sample_rate_panics() {
        ImportanceDriftMonitor::with_sample_rate(3, pht_type(), 0);
    }
}
