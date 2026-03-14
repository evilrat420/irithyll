//! Streaming SHAP -- running mean of absolute SHAP values for online feature importance.

use crate::explain::treeshap::ShapValues;

/// Running mean of absolute SHAP values for online feature importance tracking.
///
/// Feed SHAP explanations as they are computed during training/prediction,
/// and query the running importance scores at any time.
///
/// # Example
///
/// ```text
/// let mut tracker = StreamingShap::new(n_features);
/// for sample in stream {
///     let shap = model.explain(&sample.features);
///     tracker.update(&shap);
/// }
/// let importances = tracker.importances(); // running mean |SHAP|
/// ```
#[derive(Debug, Clone)]
pub struct StreamingShap {
    running_mean: Vec<f64>,
    count: u64,
}

impl StreamingShap {
    /// Create a new tracker for the given number of features.
    pub fn new(n_features: usize) -> Self {
        Self {
            running_mean: vec![0.0; n_features],
            count: 0,
        }
    }

    /// Update with a new SHAP explanation, incrementing the running mean.
    pub fn update(&mut self, shap: &ShapValues) {
        self.count += 1;
        let n = self.count as f64;
        for (i, &v) in shap.values.iter().enumerate() {
            if i < self.running_mean.len() {
                self.running_mean[i] += (v.abs() - self.running_mean[i]) / n;
            }
        }
    }

    /// Current running mean of |SHAP| per feature.
    pub fn importances(&self) -> &[f64] {
        &self.running_mean
    }

    /// Number of explanations processed.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset to zero state.
    pub fn reset(&mut self) {
        self.running_mean.iter_mut().for_each(|v| *v = 0.0);
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_shap_basic() {
        let mut tracker = StreamingShap::new(2);

        tracker.update(&ShapValues {
            values: vec![1.0, -2.0],
            base_value: 0.0,
        });
        assert_eq!(tracker.count(), 1);
        assert!((tracker.importances()[0] - 1.0).abs() < 1e-10);
        assert!((tracker.importances()[1] - 2.0).abs() < 1e-10);

        tracker.update(&ShapValues {
            values: vec![3.0, -4.0],
            base_value: 0.0,
        });
        assert_eq!(tracker.count(), 2);
        // mean |SHAP| = (1+3)/2=2 for feat 0, (2+4)/2=3 for feat 1.
        assert!((tracker.importances()[0] - 2.0).abs() < 1e-10);
        assert!((tracker.importances()[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn streaming_shap_reset() {
        let mut tracker = StreamingShap::new(2);
        tracker.update(&ShapValues {
            values: vec![5.0, -3.0],
            base_value: 0.0,
        });
        tracker.reset();
        assert_eq!(tracker.count(), 0);
        assert_eq!(tracker.importances()[0], 0.0);
        assert_eq!(tracker.importances()[1], 0.0);
    }
}
