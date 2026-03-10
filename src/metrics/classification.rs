//! Online classification metrics computed incrementally.
//!
//! All metrics are maintained with O(1) state using running counters.
//! No past samples are stored. Binary classification is assumed with
//! class 1 as positive and class 0 as negative.

/// Incrementally computed classification metrics.
///
/// Tracks accuracy, precision, recall, F1-score, and log loss over a
/// stream of observations. Designed for binary classification where
/// class 1 is the positive class.
///
/// # Example
///
/// ```
/// use irithyll::metrics::ClassificationMetrics;
///
/// let mut m = ClassificationMetrics::new();
/// m.update(1, 1, 0.9); // true positive
/// m.update(0, 0, 0.1); // true negative
/// assert!(m.accuracy() == 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// Total number of samples observed.
    n_total: u64,
    /// Number of correctly classified samples.
    n_correct: u64,
    /// True positives: target=1, predicted=1.
    tp: u64,
    /// False positives: target=0, predicted=1.
    fp: u64,
    /// False negatives: target=1, predicted=0.
    fn_count: u64,
    /// Running sum of per-sample log loss values.
    sum_log_loss: f64,
}

/// Minimum probability used when clipping to avoid log(0).
const CLIP_MIN: f64 = 1e-15;
/// Maximum probability used when clipping to avoid log(0).
const CLIP_MAX: f64 = 1.0 - 1e-15;

impl ClassificationMetrics {
    /// Create a new, empty metrics tracker.
    pub fn new() -> Self {
        Self {
            n_total: 0,
            n_correct: 0,
            tp: 0,
            fp: 0,
            fn_count: 0,
            sum_log_loss: 0.0,
        }
    }

    /// Update all metrics with a single observation.
    ///
    /// - `target`: true class label (0 or 1 for binary classification).
    /// - `predicted`: predicted class label (0 or 1).
    /// - `predicted_proba`: model's predicted probability for the positive class.
    ///   Clipped to `[1e-15, 1-1e-15]` for log loss computation.
    pub fn update(&mut self, target: usize, predicted: usize, predicted_proba: f64) {
        self.n_total += 1;

        if target == predicted {
            self.n_correct += 1;
        }

        // Binary confusion matrix updates (class 1 = positive)
        if target == 1 && predicted == 1 {
            self.tp += 1;
        } else if target == 0 && predicted == 1 {
            self.fp += 1;
        } else if target == 1 && predicted == 0 {
            self.fn_count += 1;
        }
        // target == 0 && predicted == 0 is TN, tracked implicitly

        // Log loss: -(y * ln(p) + (1-y) * ln(1-p))
        let p = predicted_proba.clamp(CLIP_MIN, CLIP_MAX);
        let y = if target == 1 { 1.0 } else { 0.0 };
        let sample_loss = -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
        self.sum_log_loss += sample_loss;
    }

    /// Classification accuracy: correct / total.
    /// Returns 0.0 if no samples have been observed.
    pub fn accuracy(&self) -> f64 {
        if self.n_total == 0 {
            return 0.0;
        }
        self.n_correct as f64 / self.n_total as f64
    }

    /// Precision: TP / (TP + FP).
    /// Returns 0.0 if no positive predictions have been made.
    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 {
            return 0.0;
        }
        self.tp as f64 / denom as f64
    }

    /// Recall (sensitivity): TP / (TP + FN).
    /// Returns 0.0 if no positive samples exist.
    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_count;
        if denom == 0 {
            return 0.0;
        }
        self.tp as f64 / denom as f64
    }

    /// F1-score: harmonic mean of precision and recall.
    /// Returns 0.0 if both precision and recall are zero.
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        let sum = p + r;
        if sum == 0.0 {
            return 0.0;
        }
        2.0 * p * r / sum
    }

    /// Mean log loss (cross-entropy loss) over all observed samples.
    /// Returns 0.0 if no samples have been observed.
    pub fn log_loss(&self) -> f64 {
        if self.n_total == 0 {
            return 0.0;
        }
        self.sum_log_loss / self.n_total as f64
    }

    /// Number of samples observed so far.
    pub fn n_samples(&self) -> u64 {
        self.n_total
    }

    /// Reset all metrics to the initial empty state.
    pub fn reset(&mut self) {
        self.n_total = 0;
        self.n_correct = 0;
        self.tp = 0;
        self.fp = 0;
        self.fn_count = 0;
        self.sum_log_loss = 0.0;
    }
}

impl Default for ClassificationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn empty_state_returns_zeros() {
        let m = ClassificationMetrics::new();
        assert_eq!(m.accuracy(), 0.0);
        assert_eq!(m.precision(), 0.0);
        assert_eq!(m.recall(), 0.0);
        assert_eq!(m.f1(), 0.0);
        assert_eq!(m.log_loss(), 0.0);
        assert_eq!(m.n_samples(), 0);
    }

    #[test]
    fn perfect_accuracy() {
        let mut m = ClassificationMetrics::new();
        m.update(1, 1, 0.99);
        m.update(0, 0, 0.01);
        m.update(1, 1, 0.95);
        m.update(0, 0, 0.05);
        assert!(approx_eq(m.accuracy(), 1.0));
    }

    #[test]
    fn fifty_percent_accuracy() {
        let mut m = ClassificationMetrics::new();
        m.update(1, 1, 0.9);  // correct
        m.update(0, 1, 0.8);  // wrong
        m.update(1, 0, 0.2);  // wrong
        m.update(0, 0, 0.1);  // correct
        assert!(approx_eq(m.accuracy(), 0.5));
    }

    #[test]
    fn precision_all_true_positives() {
        let mut m = ClassificationMetrics::new();
        m.update(1, 1, 0.9);
        m.update(1, 1, 0.8);
        m.update(0, 0, 0.1); // TN, does not affect precision denominator
        // TP=2, FP=0 => precision = 1.0
        assert!(approx_eq(m.precision(), 1.0));
    }

    #[test]
    fn precision_all_false_positives() {
        let mut m = ClassificationMetrics::new();
        m.update(0, 1, 0.9); // FP
        m.update(0, 1, 0.8); // FP
        // TP=0, FP=2 => precision = 0.0
        assert!(approx_eq(m.precision(), 0.0));
    }

    #[test]
    fn recall_all_positives_detected() {
        let mut m = ClassificationMetrics::new();
        m.update(1, 1, 0.9);
        m.update(1, 1, 0.8);
        // TP=2, FN=0 => recall = 1.0
        assert!(approx_eq(m.recall(), 1.0));
    }

    #[test]
    fn recall_no_positives_detected() {
        let mut m = ClassificationMetrics::new();
        m.update(1, 0, 0.3);
        m.update(1, 0, 0.2);
        // TP=0, FN=2 => recall = 0.0
        assert!(approx_eq(m.recall(), 0.0));
    }

    #[test]
    fn f1_mixed_case() {
        let mut m = ClassificationMetrics::new();
        m.update(1, 1, 0.9); // TP
        m.update(1, 0, 0.3); // FN
        m.update(0, 1, 0.7); // FP
        m.update(0, 0, 0.1); // TN
        // TP=1, FP=1, FN=1
        // precision = 1/2 = 0.5, recall = 1/2 = 0.5
        // f1 = 2*0.5*0.5 / (0.5+0.5) = 0.5
        assert!(approx_eq(m.f1(), 0.5));
    }

    #[test]
    fn f1_zero_when_no_predictions() {
        let mut m = ClassificationMetrics::new();
        m.update(0, 0, 0.1);
        m.update(0, 0, 0.2);
        // No positive predictions, no positive targets => precision=0, recall=0, f1=0
        assert_eq!(m.f1(), 0.0);
    }

    #[test]
    fn log_loss_near_perfect_predictions() {
        let mut m = ClassificationMetrics::new();
        m.update(1, 1, 0.999);
        m.update(0, 0, 0.001);
        // Both predictions are near-perfect, so log loss should be very small
        assert!(m.log_loss() < 0.01);
        assert!(m.log_loss() > 0.0);
    }

    #[test]
    fn log_loss_random_predictions() {
        let mut m = ClassificationMetrics::new();
        // Predicting 0.5 for everything => log loss = -ln(0.5) = ln(2) ~ 0.693
        m.update(1, 1, 0.5);
        m.update(0, 0, 0.5);
        let expected = 2.0_f64.ln();
        assert!(approx_eq(m.log_loss(), expected));
    }

    #[test]
    fn log_loss_clips_extreme_probabilities() {
        let mut m = ClassificationMetrics::new();
        // Probability of exactly 0.0 or 1.0 should be clipped, not produce inf/NaN
        m.update(1, 1, 1.0);
        m.update(0, 0, 0.0);
        assert!(m.log_loss().is_finite());
        assert!(m.log_loss() >= 0.0);
    }

    #[test]
    fn reset_clears_state() {
        let mut m = ClassificationMetrics::new();
        m.update(1, 1, 0.9);
        m.update(0, 1, 0.8);
        m.update(1, 0, 0.2);
        m.reset();
        assert_eq!(m.n_samples(), 0);
        assert_eq!(m.accuracy(), 0.0);
        assert_eq!(m.precision(), 0.0);
        assert_eq!(m.recall(), 0.0);
        assert_eq!(m.f1(), 0.0);
        assert_eq!(m.log_loss(), 0.0);
    }

    #[test]
    fn default_is_empty() {
        let m = ClassificationMetrics::default();
        assert_eq!(m.n_samples(), 0);
        assert_eq!(m.accuracy(), 0.0);
    }

    #[test]
    fn log_loss_asymmetric() {
        // Confident wrong prediction should have high log loss
        let mut m = ClassificationMetrics::new();
        m.update(1, 0, 0.01); // target=1 but predicted proba=0.01
        // loss = -(1.0 * ln(0.01) + 0.0 * ln(0.99)) = -ln(0.01) = ln(100) ~ 4.605
        let expected = 100.0_f64.ln();
        assert!(approx_eq(m.log_loss(), expected));
    }
}
