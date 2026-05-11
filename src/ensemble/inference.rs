use crate::ensemble::SGBT;
use crate::loss::Loss;

impl<L: Loss> SGBT<L> {
    /// Predict the raw output for a feature vector.
    ///
    /// Always uses sigmoid-blended soft routing with auto-calibrated per-feature
    /// bandwidths derived from median split threshold gaps. Features that have
    /// never been split on use hard routing (bandwidth = infinity).
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        if self.auto_bandwidths.is_empty() {
            for step in &self.steps {
                pred += self.config.learning_rate * step.predict(features);
            }
        } else {
            for step in &self.steps {
                pred += self.config.learning_rate
                    * step.predict_smooth_auto(features, &self.auto_bandwidths);
            }
        }
        pred
    }

    /// Predict using sigmoid-blended soft routing with an explicit bandwidth.
    ///
    /// Uses a single bandwidth for all features. For auto-calibrated per-feature
    /// bandwidths, use [`predict()`](SGBT::predict) which always uses smooth routing.
    pub fn predict_smooth(&self, features: &[f64], bandwidth: f64) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate * step.predict_smooth(features, bandwidth);
        }
        pred
    }

    /// Per-feature auto-calibrated bandwidths used by `predict()`.
    ///
    /// Empty before the first training sample. Each entry corresponds to a
    /// feature index; `f64::INFINITY` means that feature has no splits and
    /// uses hard routing.
    pub fn auto_bandwidths(&self) -> &[f64] {
        &self.auto_bandwidths
    }

    /// Predict with parent-leaf linear interpolation.
    ///
    /// Blends each leaf prediction with its parent's preserved prediction
    /// based on sample count, preventing stale predictions from fresh leaves.
    pub fn predict_interpolated(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate * step.predict_interpolated(features);
        }
        pred
    }

    /// Predict with sibling-based interpolation for feature-continuous predictions.
    ///
    /// At each split node near the threshold boundary, blends left and right
    /// subtree predictions linearly based on distance from the threshold.
    /// Uses auto-calibrated bandwidths as the interpolation margin.
    /// Predictions vary continuously as features change, eliminating
    /// step-function artifacts.
    pub fn predict_sibling_interpolated(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate
                * step.predict_sibling_interpolated(features, &self.auto_bandwidths);
        }
        pred
    }

    /// Predict with graduated active-shadow blending.
    ///
    /// Smoothly transitions between active and shadow trees during replacement,
    /// eliminating prediction dips. Requires `shadow_warmup` to be configured.
    /// When disabled, equivalent to `predict()`.
    pub fn predict_graduated(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate * step.predict_graduated(features);
        }
        pred
    }

    /// Predict with graduated blending + sibling interpolation (premium path).
    ///
    /// Combines graduated active-shadow handoff (no prediction dips during
    /// tree replacement) with feature-continuous sibling interpolation
    /// (no step-function artifacts near split boundaries).
    pub fn predict_graduated_sibling_interpolated(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate
                * step.predict_graduated_sibling_interpolated(features, &self.auto_bandwidths);
        }
        pred
    }

    /// Predict with loss transform applied (e.g., sigmoid for logistic loss).
    pub fn predict_transformed(&self, features: &[f64]) -> f64 {
        self.loss.predict_transform(self.predict(features))
    }

    /// Predict probability (alias for `predict_transformed`).
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        self.predict_transformed(features)
    }

    /// Predict with confidence estimation.
    ///
    /// Returns `(prediction, confidence)` where confidence = 1 / sqrt(sum_variance).
    /// Higher confidence indicates more certain predictions (leaves have seen
    /// more hessian mass). Confidence of 0.0 means the model has no information.
    ///
    /// The variance per tree is estimated as `1 / (H_sum + lambda)` at the
    /// leaf where the sample lands. The ensemble variance is the sum of
    /// per-tree variances (scaled by learning_rate²), and confidence is
    /// the reciprocal of the standard deviation.
    pub fn predict_with_confidence(&self, features: &[f64]) -> (f64, f64) {
        let mut pred = self.base_prediction;
        let mut total_variance = 0.0;
        let lr2 = self.config.learning_rate * self.config.learning_rate;

        for step in &self.steps {
            let (value, variance) = step.predict_with_variance(features);
            pred += self.config.learning_rate * value;
            total_variance += lr2 * variance;
        }

        let confidence = if total_variance > 0.0 && total_variance.is_finite() {
            1.0 / total_variance.sqrt()
        } else {
            0.0
        };

        (pred, confidence)
    }

    /// Batch prediction.
    pub fn predict_batch(&self, feature_matrix: &[Vec<f64>]) -> Vec<f64> {
        feature_matrix.iter().map(|f| self.predict(f)).collect()
    }
}
