//! Online metric tracking for streaming model evaluation.
//!
//! All metrics are computed incrementally, one sample at a time, with O(1)
//! state. No past samples are stored. This module provides:
//!
//! - [`RegressionMetrics`] -- MAE, MSE, RMSE, R-squared via Welford's algorithm.
//! - [`ClassificationMetrics`] -- Accuracy, precision, recall, F1, log loss.
//! - [`FeatureImportance`] -- Accumulated split gain per feature.
//! - [`MetricSet`] -- Convenience wrapper combining regression and classification.

pub mod classification;
pub mod conformal;
pub mod ewma;
pub mod importance;
pub mod regression;
pub mod rolling;

pub use classification::ClassificationMetrics;
pub use importance::FeatureImportance;
pub use regression::RegressionMetrics;

/// Combined metric tracker that holds both regression and classification metrics.
///
/// Useful when evaluating a model that may be used for either task, or when
/// tracking both types of metrics simultaneously during experimentation.
///
/// # Example
///
/// ```
/// use irithyll::metrics::MetricSet;
///
/// let mut ms = MetricSet::new();
/// ms.update_regression(3.0, 2.8);
/// ms.update_classification(1, 1, 0.9);
///
/// assert!(ms.regression().mae() > 0.0);
/// assert!(ms.classification().accuracy() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct MetricSet {
    reg: RegressionMetrics,
    cls: ClassificationMetrics,
}

impl MetricSet {
    /// Create a new metric set with empty regression and classification trackers.
    pub fn new() -> Self {
        Self {
            reg: RegressionMetrics::new(),
            cls: ClassificationMetrics::new(),
        }
    }

    /// Update regression metrics with a (target, prediction) pair.
    pub fn update_regression(&mut self, target: f64, prediction: f64) {
        self.reg.update(target, prediction);
    }

    /// Update classification metrics with a single observation.
    ///
    /// - `target`: true class label.
    /// - `predicted`: predicted class label.
    /// - `predicted_proba`: model's predicted probability for the positive class.
    pub fn update_classification(&mut self, target: usize, predicted: usize, predicted_proba: f64) {
        self.cls.update(target, predicted, predicted_proba);
    }

    /// Reference to the regression metrics tracker.
    pub fn regression(&self) -> &RegressionMetrics {
        &self.reg
    }

    /// Reference to the classification metrics tracker.
    pub fn classification(&self) -> &ClassificationMetrics {
        &self.cls
    }

    /// Reset both regression and classification metrics to their initial states.
    pub fn reset(&mut self) {
        self.reg.reset();
        self.cls.reset();
    }
}

impl Default for MetricSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_set_new_is_empty() {
        let ms = MetricSet::new();
        assert_eq!(ms.regression().n_samples(), 0);
        assert_eq!(ms.classification().n_samples(), 0);
    }

    #[test]
    fn metric_set_update_regression() {
        let mut ms = MetricSet::new();
        ms.update_regression(5.0, 3.0);
        ms.update_regression(2.0, 2.0);
        assert_eq!(ms.regression().n_samples(), 2);
        assert!(ms.regression().mae() > 0.0);
        // Classification should be unaffected
        assert_eq!(ms.classification().n_samples(), 0);
    }

    #[test]
    fn metric_set_update_classification() {
        let mut ms = MetricSet::new();
        ms.update_classification(1, 1, 0.9);
        ms.update_classification(0, 0, 0.1);
        assert_eq!(ms.classification().n_samples(), 2);
        assert_eq!(ms.classification().accuracy(), 1.0);
        // Regression should be unaffected
        assert_eq!(ms.regression().n_samples(), 0);
    }

    #[test]
    fn metric_set_reset() {
        let mut ms = MetricSet::new();
        ms.update_regression(1.0, 2.0);
        ms.update_classification(1, 0, 0.3);
        ms.reset();
        assert_eq!(ms.regression().n_samples(), 0);
        assert_eq!(ms.classification().n_samples(), 0);
    }

    #[test]
    fn metric_set_default() {
        let ms = MetricSet::default();
        assert_eq!(ms.regression().n_samples(), 0);
        assert_eq!(ms.classification().n_samples(), 0);
    }
}
