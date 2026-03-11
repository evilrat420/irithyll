//! Windowed (rolling) metrics over the last N observations.
//!
//! Uses a circular buffer + revert pattern: when the buffer is full, the oldest
//! observation is reverted from running accumulators before the new one is added.
//! All metric queries reflect only the most recent `window_size` samples.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Rolling Regression Metrics
// ---------------------------------------------------------------------------

/// Windowed regression metrics over the last N observations.
///
/// Maintains MAE, MSE, RMSE, and R-squared over a sliding window using
/// the revert pattern with a `VecDeque` circular buffer. Welford's algorithm
/// tracks target variance for streaming R-squared.
///
/// # Example
///
/// ```
/// use irithyll::metrics::rolling::RollingRegressionMetrics;
///
/// let mut m = RollingRegressionMetrics::new(3);
/// m.update(1.0, 1.5); // window: [(1,1.5)]
/// m.update(2.0, 2.5); // window: [(1,1.5), (2,2.5)]
/// m.update(3.0, 3.5); // window: [(1,1.5), (2,2.5), (3,3.5)]
/// m.update(4.0, 4.5); // oldest (1,1.5) evicted
/// assert_eq!(m.n_samples(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct RollingRegressionMetrics {
    window_size: usize,
    buffer: VecDeque<(f64, f64)>,
    count: u64,
    sum_abs_error: f64,
    sum_sq_error: f64,
    /// Sliding Welford running mean of targets.
    target_mean: f64,
    /// Sliding Welford running M2 of targets.
    target_m2: f64,
}

impl RollingRegressionMetrics {
    /// Create a new rolling metrics tracker with the given window size.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is zero.
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size + 1),
            count: 0,
            sum_abs_error: 0.0,
            sum_sq_error: 0.0,
            target_mean: 0.0,
            target_m2: 0.0,
        }
    }

    /// Update metrics with a new (target, prediction) pair.
    ///
    /// If the window is full, the oldest observation is reverted first.
    pub fn update(&mut self, target: f64, prediction: f64) {
        // Evict oldest if at capacity
        if self.buffer.len() == self.window_size {
            let (old_t, old_p) = self.buffer.pop_front().unwrap();
            self.revert(old_t, old_p);
        }

        // Add new observation
        self.count += 1;
        let error = target - prediction;
        self.sum_abs_error += error.abs();
        self.sum_sq_error += error * error;

        // Forward Welford step for target
        let delta = target - self.target_mean;
        self.target_mean += delta / self.count as f64;
        let delta2 = target - self.target_mean;
        self.target_m2 += delta * delta2;

        self.buffer.push_back((target, prediction));
    }

    /// Revert the effect of a single (target, prediction) observation.
    fn revert(&mut self, target: f64, prediction: f64) {
        if self.count == 0 {
            return;
        }

        let error = target - prediction;
        self.sum_abs_error -= error.abs();
        self.sum_sq_error -= error * error;

        // Reverse Welford for target mean/m2
        if self.count == 1 {
            // Removing the last sample resets to empty state
            self.target_mean = 0.0;
            self.target_m2 = 0.0;
        } else {
            let n = self.count as f64;
            let old_mean = self.target_mean;
            self.target_mean = (old_mean * n - target) / (n - 1.0);
            let delta_old = target - old_mean;
            let delta_new = target - self.target_mean;
            self.target_m2 -= delta_old * delta_new;
        }

        self.count -= 1;
    }

    /// Mean Absolute Error over the current window.
    pub fn mae(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_abs_error / self.count as f64
    }

    /// Mean Squared Error over the current window.
    pub fn mse(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_sq_error / self.count as f64
    }

    /// Root Mean Squared Error over the current window.
    pub fn rmse(&self) -> f64 {
        self.mse().sqrt()
    }

    /// R-squared over the current window.
    ///
    /// Returns 0.0 if fewer than 2 samples in the window or if target
    /// variance is zero.
    pub fn r_squared(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        if self.target_m2 <= 0.0 {
            return 0.0;
        }
        1.0 - self.sum_sq_error / self.target_m2
    }

    /// Number of samples currently in the window.
    pub fn n_samples(&self) -> u64 {
        self.count
    }

    /// The configured window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Whether the window is completely filled.
    pub fn is_full(&self) -> bool {
        self.count as usize == self.window_size
    }

    /// Reset all metrics and clear the buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.count = 0;
        self.sum_abs_error = 0.0;
        self.sum_sq_error = 0.0;
        self.target_mean = 0.0;
        self.target_m2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Rolling Classification Metrics
// ---------------------------------------------------------------------------

/// Minimum probability used when clipping to avoid log(0).
const CLIP_MIN: f64 = 1e-15;
/// Maximum probability used when clipping to avoid log(0).
const CLIP_MAX: f64 = 1.0 - 1e-15;

/// Windowed classification metrics over the last N observations.
///
/// Tracks accuracy, precision, recall, F1, and log loss over a sliding window.
/// Binary classification with class 1 as positive.
///
/// # Example
///
/// ```
/// use irithyll::metrics::rolling::RollingClassificationMetrics;
///
/// let mut m = RollingClassificationMetrics::new(100);
/// m.update(1, 1, 0.9); // TP
/// m.update(0, 0, 0.1); // TN
/// assert_eq!(m.accuracy(), 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct RollingClassificationMetrics {
    window_size: usize,
    buffer: VecDeque<(usize, usize, f64)>, // (target, predicted, proba)
    n_correct: u64,
    tp: u64,
    fp: u64,
    fn_count: u64,
    sum_log_loss: f64,
    count: u64,
}

impl RollingClassificationMetrics {
    /// Create a new rolling classification metrics tracker.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is zero.
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size + 1),
            n_correct: 0,
            tp: 0,
            fp: 0,
            fn_count: 0,
            sum_log_loss: 0.0,
            count: 0,
        }
    }

    /// Update metrics with a single observation, evicting the oldest if full.
    pub fn update(&mut self, target: usize, predicted: usize, predicted_proba: f64) {
        if self.buffer.len() == self.window_size {
            let (old_t, old_p, old_proba) = self.buffer.pop_front().unwrap();
            self.revert(old_t, old_p, old_proba);
        }

        self.count += 1;
        if target == predicted {
            self.n_correct += 1;
        }
        if target == 1 && predicted == 1 {
            self.tp += 1;
        } else if target == 0 && predicted == 1 {
            self.fp += 1;
        } else if target == 1 && predicted == 0 {
            self.fn_count += 1;
        }

        let p = predicted_proba.clamp(CLIP_MIN, CLIP_MAX);
        let y = if target == 1 { 1.0 } else { 0.0 };
        self.sum_log_loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());

        self.buffer.push_back((target, predicted, predicted_proba));
    }

    fn revert(&mut self, target: usize, predicted: usize, predicted_proba: f64) {
        if self.count == 0 {
            return;
        }
        self.count -= 1;
        if target == predicted {
            self.n_correct -= 1;
        }
        if target == 1 && predicted == 1 {
            self.tp -= 1;
        } else if target == 0 && predicted == 1 {
            self.fp -= 1;
        } else if target == 1 && predicted == 0 {
            self.fn_count -= 1;
        }

        let p = predicted_proba.clamp(CLIP_MIN, CLIP_MAX);
        let y = if target == 1 { 1.0 } else { 0.0 };
        self.sum_log_loss -= -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }

    /// Classification accuracy over the current window.
    pub fn accuracy(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.n_correct as f64 / self.count as f64
    }

    /// Precision: TP / (TP + FP).
    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 {
            return 0.0;
        }
        self.tp as f64 / denom as f64
    }

    /// Recall: TP / (TP + FN).
    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_count;
        if denom == 0 {
            return 0.0;
        }
        self.tp as f64 / denom as f64
    }

    /// F1-score: harmonic mean of precision and recall.
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        let sum = p + r;
        if sum == 0.0 {
            return 0.0;
        }
        2.0 * p * r / sum
    }

    /// Mean log loss over the current window.
    pub fn log_loss(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_log_loss / self.count as f64
    }

    /// Number of samples in the current window.
    pub fn n_samples(&self) -> u64 {
        self.count
    }

    /// The configured window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Whether the window is completely filled.
    pub fn is_full(&self) -> bool {
        self.count as usize == self.window_size
    }

    /// Reset all metrics and clear the buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.count = 0;
        self.n_correct = 0;
        self.tp = 0;
        self.fp = 0;
        self.fn_count = 0;
        self.sum_log_loss = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // ---- Regression ----

    #[test]
    fn rolling_reg_within_window() {
        let mut m = RollingRegressionMetrics::new(10);
        m.update(1.0, 2.0); // err=1
        m.update(3.0, 1.0); // err=2
        m.update(5.0, 4.0); // err=1
        assert_eq!(m.n_samples(), 3);
        assert!(!m.is_full());
        assert!(approx_eq(m.mae(), 4.0 / 3.0));
        assert!(approx_eq(m.mse(), 6.0 / 3.0));
    }

    #[test]
    fn rolling_reg_slides() {
        let mut m = RollingRegressionMetrics::new(2);
        m.update(1.0, 2.0); // err=1
        m.update(3.0, 1.0); // err=2
        assert!(m.is_full());
        assert!(approx_eq(m.mae(), 1.5)); // (1+2)/2

        // Evict (1,2), add (5,4) err=1
        m.update(5.0, 4.0);
        assert_eq!(m.n_samples(), 2);
        assert!(approx_eq(m.mae(), 1.5)); // (2+1)/2
    }

    #[test]
    fn rolling_reg_matches_manual() {
        // Feed 10 samples with window=3, verify last 3 match manual computation
        let mut m = RollingRegressionMetrics::new(3);
        let data = [
            (1.0, 1.5),
            (2.0, 2.5),
            (3.0, 3.5),
            (4.0, 4.5),
            (5.0, 5.5),
            (6.0, 6.5),
            (7.0, 7.5),
            (8.0, 8.5),
            (9.0, 9.5),
            (10.0, 10.5),
        ];
        for &(t, p) in &data {
            m.update(t, p);
        }
        assert_eq!(m.n_samples(), 3);
        // Last 3: (8,8.5), (9,9.5), (10,10.5) all errors=0.5
        assert!(approx_eq(m.mae(), 0.5));
        assert!(approx_eq(m.mse(), 0.25));
    }

    #[test]
    fn rolling_reg_r_squared() {
        let mut m = RollingRegressionMetrics::new(4);
        // Perfect predictions
        m.update(2.0, 2.0);
        m.update(4.0, 4.0);
        m.update(6.0, 6.0);
        m.update(8.0, 8.0);
        assert!(approx_eq(m.r_squared(), 1.0));
    }

    #[test]
    fn rolling_reg_reset() {
        let mut m = RollingRegressionMetrics::new(5);
        m.update(1.0, 2.0);
        m.update(3.0, 4.0);
        m.reset();
        assert_eq!(m.n_samples(), 0);
        assert_eq!(m.mae(), 0.0);
        assert!(!m.is_full());
    }

    #[test]
    fn rolling_reg_window_size() {
        let m = RollingRegressionMetrics::new(42);
        assert_eq!(m.window_size(), 42);
    }

    // ---- Classification ----

    #[test]
    fn rolling_cls_accuracy() {
        let mut m = RollingClassificationMetrics::new(3);
        m.update(1, 1, 0.9); // correct
        m.update(0, 0, 0.1); // correct
        m.update(1, 0, 0.3); // wrong
        assert!(approx_eq(m.accuracy(), 2.0 / 3.0));

        // Evict (1,1,0.9) — was correct, add another correct
        m.update(0, 0, 0.2); // correct
                             // Window: [(0,0), (1,0), (0,0)] => 2 correct of 3
        assert!(approx_eq(m.accuracy(), 2.0 / 3.0));
    }

    #[test]
    fn rolling_cls_precision_recall() {
        let mut m = RollingClassificationMetrics::new(4);
        m.update(1, 1, 0.9); // TP
        m.update(0, 1, 0.7); // FP
        m.update(1, 0, 0.3); // FN
        m.update(0, 0, 0.1); // TN
                             // TP=1, FP=1, FN=1
        assert!(approx_eq(m.precision(), 0.5));
        assert!(approx_eq(m.recall(), 0.5));
        assert!(approx_eq(m.f1(), 0.5));
    }

    #[test]
    fn rolling_cls_slides() {
        let mut m = RollingClassificationMetrics::new(2);
        m.update(1, 1, 0.9); // TP
        m.update(0, 1, 0.7); // FP
        assert!(approx_eq(m.precision(), 0.5)); // 1/(1+1)

        // Evict TP, add TN
        m.update(0, 0, 0.1);
        // Window: [(0,1), (0,0)] => TP=0, FP=1
        assert!(approx_eq(m.precision(), 0.0));
    }

    #[test]
    fn rolling_cls_reset() {
        let mut m = RollingClassificationMetrics::new(5);
        m.update(1, 1, 0.9);
        m.reset();
        assert_eq!(m.n_samples(), 0);
        assert_eq!(m.accuracy(), 0.0);
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn rolling_reg_zero_window() {
        RollingRegressionMetrics::new(0);
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn rolling_cls_zero_window() {
        RollingClassificationMetrics::new(0);
    }
}
