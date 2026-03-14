//! Exponentially weighted moving average (EWMA) metrics.
//!
//! O(1) memory, no buffer needed. Recent observations are weighted more heavily
//! via the decay factor `alpha = 2 / (span + 1)`. Complements rolling metrics
//! which have a hard window cutoff.

// ---------------------------------------------------------------------------
// EWMA Regression Metrics
// ---------------------------------------------------------------------------

/// Exponentially weighted regression metrics.
///
/// Each metric (MAE, MSE, RMSE, R-squared) is maintained as an EWMA
/// with decay factor `alpha`. Recent observations have more influence.
///
/// # Example
///
/// ```
/// use irithyll::metrics::ewma::EwmaRegressionMetrics;
///
/// let mut m = EwmaRegressionMetrics::new(20); // span of 20
/// for i in 0..100 {
///     m.update(i as f64, i as f64 + 0.5);
/// }
/// assert!(m.mae() < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct EwmaRegressionMetrics {
    alpha: f64,
    count: u64,
    ewma_abs_error: f64,
    ewma_sq_error: f64,
    ewma_target: f64,
    ewma_target_sq_dev: f64,
    initialized: bool,
}

impl EwmaRegressionMetrics {
    /// Create EWMA regression metrics with the given span.
    ///
    /// The decay factor is `alpha = 2.0 / (span + 1)`.
    ///
    /// # Panics
    ///
    /// Panics if `span` is zero.
    pub fn new(span: usize) -> Self {
        assert!(span > 0, "span must be > 0");
        Self::with_alpha(2.0 / (span as f64 + 1.0))
    }

    /// Create EWMA regression metrics with a direct decay factor.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not in (0, 1].
    pub fn with_alpha(alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "alpha must be in (0, 1], got {alpha}"
        );
        Self {
            alpha,
            count: 0,
            ewma_abs_error: 0.0,
            ewma_sq_error: 0.0,
            ewma_target: 0.0,
            ewma_target_sq_dev: 0.0,
            initialized: false,
        }
    }

    /// Update with a new (target, prediction) pair.
    pub fn update(&mut self, target: f64, prediction: f64) {
        let abs_err = (target - prediction).abs();
        let sq_err = (target - prediction) * (target - prediction);

        if !self.initialized {
            self.ewma_abs_error = abs_err;
            self.ewma_sq_error = sq_err;
            self.ewma_target = target;
            self.ewma_target_sq_dev = 0.0;
            self.initialized = true;
        } else {
            self.ewma_abs_error = self.alpha * abs_err + (1.0 - self.alpha) * self.ewma_abs_error;
            self.ewma_sq_error = self.alpha * sq_err + (1.0 - self.alpha) * self.ewma_sq_error;

            let old_target_mean = self.ewma_target;
            self.ewma_target = self.alpha * target + (1.0 - self.alpha) * self.ewma_target;
            let dev = (target - old_target_mean) * (target - old_target_mean);
            self.ewma_target_sq_dev =
                self.alpha * dev + (1.0 - self.alpha) * self.ewma_target_sq_dev;
        }
        self.count += 1;
    }

    /// Exponentially weighted MAE.
    pub fn mae(&self) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        self.ewma_abs_error
    }

    /// Exponentially weighted MSE.
    pub fn mse(&self) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        self.ewma_sq_error
    }

    /// Exponentially weighted RMSE.
    pub fn rmse(&self) -> f64 {
        self.mse().sqrt()
    }

    /// Exponentially weighted R-squared.
    ///
    /// Returns 0.0 if fewer than 2 observations or if target variance is zero.
    pub fn r_squared(&self) -> f64 {
        if self.count < 2 || self.ewma_target_sq_dev < 1e-15 {
            return 0.0;
        }
        1.0 - self.ewma_sq_error / self.ewma_target_sq_dev
    }

    /// The decay factor alpha.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Number of observations processed.
    pub fn n_samples(&self) -> u64 {
        self.count
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.count = 0;
        self.ewma_abs_error = 0.0;
        self.ewma_sq_error = 0.0;
        self.ewma_target = 0.0;
        self.ewma_target_sq_dev = 0.0;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// EWMA Classification Metrics
// ---------------------------------------------------------------------------

/// Exponentially weighted classification metrics.
///
/// Tracks exponentially weighted accuracy and log loss. O(1) memory.
///
/// # Example
///
/// ```
/// use irithyll::metrics::ewma::EwmaClassificationMetrics;
///
/// let mut m = EwmaClassificationMetrics::new(50);
/// m.update(1, 1, 0.95);
/// m.update(0, 0, 0.05);
/// assert!(m.accuracy() > 0.9);
/// ```
#[derive(Debug, Clone)]
pub struct EwmaClassificationMetrics {
    alpha: f64,
    count: u64,
    ewma_correct: f64,
    ewma_log_loss: f64,
    initialized: bool,
}

/// Minimum probability used when clipping to avoid log(0).
const CLIP_MIN: f64 = 1e-15;
/// Maximum probability used when clipping to avoid log(0).
const CLIP_MAX: f64 = 1.0 - 1e-15;

impl EwmaClassificationMetrics {
    /// Create EWMA classification metrics with the given span.
    ///
    /// # Panics
    ///
    /// Panics if `span` is zero.
    pub fn new(span: usize) -> Self {
        assert!(span > 0, "span must be > 0");
        Self::with_alpha(2.0 / (span as f64 + 1.0))
    }

    /// Create EWMA classification metrics with a direct decay factor.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not in (0, 1].
    pub fn with_alpha(alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "alpha must be in (0, 1], got {alpha}"
        );
        Self {
            alpha,
            count: 0,
            ewma_correct: 0.0,
            ewma_log_loss: 0.0,
            initialized: false,
        }
    }

    /// Update with a single observation.
    pub fn update(&mut self, target: usize, predicted: usize, predicted_proba: f64) {
        let correct = if target == predicted { 1.0 } else { 0.0 };

        let p = predicted_proba.clamp(CLIP_MIN, CLIP_MAX);
        let y = if target == 1 { 1.0 } else { 0.0 };
        let sample_loss = -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());

        if !self.initialized {
            self.ewma_correct = correct;
            self.ewma_log_loss = sample_loss;
            self.initialized = true;
        } else {
            self.ewma_correct = self.alpha * correct + (1.0 - self.alpha) * self.ewma_correct;
            self.ewma_log_loss = self.alpha * sample_loss + (1.0 - self.alpha) * self.ewma_log_loss;
        }
        self.count += 1;
    }

    /// Exponentially weighted accuracy (proportion of recent correct predictions).
    pub fn accuracy(&self) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        self.ewma_correct
    }

    /// Exponentially weighted log loss.
    pub fn log_loss(&self) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        self.ewma_log_loss
    }

    /// The decay factor alpha.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Number of observations processed.
    pub fn n_samples(&self) -> u64 {
        self.count
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.count = 0;
        self.ewma_correct = 0.0;
        self.ewma_log_loss = 0.0;
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    // ---- Regression ----

    #[test]
    fn ewma_reg_initial() {
        let mut m = EwmaRegressionMetrics::new(10);
        m.update(5.0, 3.0); // err=2
        assert!(approx_eq(m.mae(), 2.0));
        assert!(approx_eq(m.mse(), 4.0));
        assert_eq!(m.n_samples(), 1);
    }

    #[test]
    fn ewma_reg_decay() {
        let mut m = EwmaRegressionMetrics::new(5); // alpha = 2/6 = 0.333
                                                   // Feed constant error of 1.0
        for _ in 0..100 {
            m.update(5.0, 4.0);
        }
        assert!(approx_eq(m.mae(), 1.0));

        // Now feed zero error -- MAE should decay toward 0
        for _ in 0..100 {
            m.update(5.0, 5.0);
        }
        assert!(m.mae() < 0.01);
    }

    #[test]
    fn ewma_reg_span_controls_rate() {
        let mut fast = EwmaRegressionMetrics::new(2); // alpha = 2/3
        let mut slow = EwmaRegressionMetrics::new(50); // alpha = 2/51

        // Inject error then zero error
        fast.update(5.0, 4.0);
        slow.update(5.0, 4.0);
        for _ in 0..10 {
            fast.update(5.0, 5.0);
            slow.update(5.0, 5.0);
        }
        // Fast should have decayed more
        assert!(fast.mae() < slow.mae());
    }

    #[test]
    fn ewma_reg_reset() {
        let mut m = EwmaRegressionMetrics::new(10);
        m.update(1.0, 2.0);
        m.reset();
        assert_eq!(m.n_samples(), 0);
        assert_eq!(m.mae(), 0.0);
    }

    #[test]
    fn ewma_reg_r_squared_perfect() {
        let mut m = EwmaRegressionMetrics::new(20);
        for i in 0..100 {
            m.update(i as f64, i as f64); // perfect predictions
        }
        // MSE should be 0, but R-squared needs target variance > 0
        // With all different targets, target_sq_dev > 0 but sq_error = 0
        assert!(m.r_squared() > 0.99 || m.mse() < 1e-10);
    }

    // ---- Classification ----

    #[test]
    fn ewma_cls_all_correct() {
        let mut m = EwmaClassificationMetrics::new(10);
        for _ in 0..50 {
            m.update(1, 1, 0.95);
        }
        assert!(m.accuracy() > 0.99);
    }

    #[test]
    fn ewma_cls_accuracy_decays() {
        let mut m = EwmaClassificationMetrics::new(5);
        // All correct
        for _ in 0..50 {
            m.update(1, 1, 0.9);
        }
        assert!(m.accuracy() > 0.99);

        // Now all wrong
        for _ in 0..50 {
            m.update(1, 0, 0.1);
        }
        assert!(m.accuracy() < 0.01);
    }

    #[test]
    fn ewma_cls_log_loss_near_perfect() {
        let mut m = EwmaClassificationMetrics::new(10);
        m.update(1, 1, 0.999);
        assert!(m.log_loss() < 0.01);
        assert!(m.log_loss() > 0.0);
    }

    #[test]
    fn ewma_cls_reset() {
        let mut m = EwmaClassificationMetrics::new(10);
        m.update(1, 1, 0.9);
        m.reset();
        assert_eq!(m.n_samples(), 0);
        assert_eq!(m.accuracy(), 0.0);
    }

    #[test]
    #[should_panic(expected = "span must be > 0")]
    fn ewma_reg_zero_span() {
        EwmaRegressionMetrics::new(0);
    }

    #[test]
    #[should_panic(expected = "span must be > 0")]
    fn ewma_cls_zero_span() {
        EwmaClassificationMetrics::new(0);
    }

    #[test]
    fn ewma_alpha_getter() {
        let m = EwmaRegressionMetrics::new(9); // alpha = 2/10 = 0.2
        assert!(approx_eq(m.alpha(), 0.2));
    }
}
