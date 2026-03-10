//! Online regression metrics computed incrementally.
//!
//! All metrics are maintained with O(1) state using running sums and
//! Welford's online algorithm for variance estimation. No past samples
//! are stored.

/// Incrementally computed regression metrics.
///
/// Tracks MAE, MSE, RMSE, and R-squared over a stream of (target, prediction)
/// pairs. Target variance is estimated via Welford's online algorithm to
/// support streaming R-squared computation.
///
/// # Example
///
/// ```
/// use irithyll::metrics::RegressionMetrics;
///
/// let mut m = RegressionMetrics::new();
/// m.update(3.0, 2.5);
/// m.update(1.0, 1.5);
/// assert!(m.mae() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct RegressionMetrics {
    /// Number of samples observed.
    count: u64,
    /// Running sum of |target - prediction|.
    sum_abs_error: f64,
    /// Running sum of (target - prediction)^2.
    sum_sq_error: f64,
    /// Welford running mean of targets.
    target_mean: f64,
    /// Welford running sum of squared deviations of targets (M2).
    target_m2: f64,
}

impl RegressionMetrics {
    /// Create a new, empty metrics tracker.
    pub fn new() -> Self {
        Self {
            count: 0,
            sum_abs_error: 0.0,
            sum_sq_error: 0.0,
            target_mean: 0.0,
            target_m2: 0.0,
        }
    }

    /// Update all metrics with a single (target, prediction) observation.
    pub fn update(&mut self, target: f64, prediction: f64) {
        self.count += 1;

        let error = target - prediction;
        self.sum_abs_error += error.abs();
        self.sum_sq_error += error * error;

        // Welford's online algorithm for target variance
        let delta = target - self.target_mean;
        self.target_mean += delta / self.count as f64;
        let delta2 = target - self.target_mean;
        self.target_m2 += delta * delta2;
    }

    /// Mean Absolute Error. Returns 0.0 if no samples have been observed.
    pub fn mae(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_abs_error / self.count as f64
    }

    /// Mean Squared Error. Returns 0.0 if no samples have been observed.
    pub fn mse(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_sq_error / self.count as f64
    }

    /// Root Mean Squared Error. Returns 0.0 if no samples have been observed.
    pub fn rmse(&self) -> f64 {
        self.mse().sqrt()
    }

    /// Coefficient of determination (R-squared).
    ///
    /// Uses Welford's running variance of targets as the denominator.
    /// Returns 0.0 if fewer than 2 samples have been observed or if the
    /// target variance is zero (constant target).
    ///
    /// Formula: `1.0 - sum_sq_error / target_m2`
    pub fn r_squared(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        if self.target_m2 == 0.0 {
            // Constant target: if predictions are also perfect, R^2 is
            // meaningless; return 0.0 to avoid division by zero.
            return 0.0;
        }
        1.0 - self.sum_sq_error / self.target_m2
    }

    /// Number of samples observed so far.
    pub fn n_samples(&self) -> u64 {
        self.count
    }

    /// Reset all metrics to the initial empty state.
    pub fn reset(&mut self) {
        self.count = 0;
        self.sum_abs_error = 0.0;
        self.sum_sq_error = 0.0;
        self.target_mean = 0.0;
        self.target_m2 = 0.0;
    }
}

impl Default for RegressionMetrics {
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
        let m = RegressionMetrics::new();
        assert_eq!(m.mae(), 0.0);
        assert_eq!(m.mse(), 0.0);
        assert_eq!(m.rmse(), 0.0);
        assert_eq!(m.r_squared(), 0.0);
        assert_eq!(m.n_samples(), 0);
    }

    #[test]
    fn single_sample() {
        let mut m = RegressionMetrics::new();
        m.update(5.0, 3.0);
        assert!(approx_eq(m.mae(), 2.0));
        assert!(approx_eq(m.mse(), 4.0));
        assert!(approx_eq(m.rmse(), 2.0));
        // R-squared requires at least 2 samples
        assert_eq!(m.r_squared(), 0.0);
        assert_eq!(m.n_samples(), 1);
    }

    #[test]
    fn mae_multiple_samples() {
        let mut m = RegressionMetrics::new();
        // errors: |1-2|=1, |3-1|=2, |5-4|=1  => MAE = 4/3
        m.update(1.0, 2.0);
        m.update(3.0, 1.0);
        m.update(5.0, 4.0);
        assert!(approx_eq(m.mae(), 4.0 / 3.0));
    }

    #[test]
    fn mse_multiple_samples() {
        let mut m = RegressionMetrics::new();
        // errors: (1-2)^2=1, (3-1)^2=4, (5-4)^2=1 => MSE = 6/3 = 2
        m.update(1.0, 2.0);
        m.update(3.0, 1.0);
        m.update(5.0, 4.0);
        assert!(approx_eq(m.mse(), 2.0));
    }

    #[test]
    fn rmse_multiple_samples() {
        let mut m = RegressionMetrics::new();
        m.update(1.0, 2.0);
        m.update(3.0, 1.0);
        m.update(5.0, 4.0);
        assert!(approx_eq(m.rmse(), 2.0_f64.sqrt()));
    }

    #[test]
    fn r_squared_perfect_prediction() {
        let mut m = RegressionMetrics::new();
        m.update(1.0, 1.0);
        m.update(2.0, 2.0);
        m.update(3.0, 3.0);
        m.update(4.0, 4.0);
        assert!(approx_eq(m.r_squared(), 1.0));
    }

    #[test]
    fn r_squared_mean_prediction() {
        // If predictions are all the mean of targets, R^2 should be ~0
        let mut m = RegressionMetrics::new();
        let targets = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        for &t in &targets {
            m.update(t, mean);
        }
        // sum_sq_error = sum((t - mean)^2) = 1+1+0+1+1 = (1+1+0+1+4) wait:
        // (1-3)^2=4, (2-3)^2=1, (3-3)^2=0, (4-3)^2=1, (5-3)^2=4 => 10
        // target_m2 = same = 10 (population variance sum)
        // R^2 = 1 - 10/10 = 0.0
        assert!(approx_eq(m.r_squared(), 0.0));
    }

    #[test]
    fn r_squared_partial_fit() {
        let mut m = RegressionMetrics::new();
        // targets: [2, 4, 6, 8], predictions: [2.5, 3.5, 5.5, 8.5]
        // errors: -0.5, 0.5, 0.5, -0.5 => sq: 0.25*4 = 1.0
        // target_mean = 5.0, target_m2 = (2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2 = 9+1+1+9 = 20
        // R^2 = 1 - 1.0/20.0 = 0.95
        m.update(2.0, 2.5);
        m.update(4.0, 3.5);
        m.update(6.0, 5.5);
        m.update(8.0, 8.5);
        assert!(approx_eq(m.r_squared(), 0.95));
    }

    #[test]
    fn r_squared_negative() {
        // Predictions worse than predicting the mean => R^2 < 0
        let mut m = RegressionMetrics::new();
        // targets: [1, 2, 3], mean=2, target_m2 = 1+0+1=2
        // predictions: [10, 10, 10] => sum_sq_error = 81+64+49 = 194
        m.update(1.0, 10.0);
        m.update(2.0, 10.0);
        m.update(3.0, 10.0);
        // R^2 = 1 - 194/2 = 1 - 97 = -96
        assert!(m.r_squared() < 0.0);
    }

    #[test]
    fn r_squared_constant_target() {
        let mut m = RegressionMetrics::new();
        m.update(5.0, 5.0);
        m.update(5.0, 5.0);
        m.update(5.0, 5.0);
        // target_m2 = 0 => returns 0.0
        assert_eq!(m.r_squared(), 0.0);
    }

    #[test]
    fn reset_clears_state() {
        let mut m = RegressionMetrics::new();
        m.update(1.0, 2.0);
        m.update(3.0, 4.0);
        m.reset();
        assert_eq!(m.n_samples(), 0);
        assert_eq!(m.mae(), 0.0);
        assert_eq!(m.mse(), 0.0);
        assert_eq!(m.rmse(), 0.0);
        assert_eq!(m.r_squared(), 0.0);
    }

    #[test]
    fn nan_inputs_propagate() {
        let mut m = RegressionMetrics::new();
        m.update(f64::NAN, 1.0);
        assert!(m.mae().is_nan());
        assert!(m.mse().is_nan());
    }

    #[test]
    fn large_sample_count() {
        let mut m = RegressionMetrics::new();
        for i in 0..10_000 {
            let t = i as f64;
            let p = t + 1.0; // constant error of 1.0
            m.update(t, p);
        }
        assert!(approx_eq(m.mae(), 1.0));
        assert!(approx_eq(m.mse(), 1.0));
        assert!(approx_eq(m.rmse(), 1.0));
        assert_eq!(m.n_samples(), 10_000);
    }

    #[test]
    fn default_is_empty() {
        let m = RegressionMetrics::default();
        assert_eq!(m.n_samples(), 0);
        assert_eq!(m.mae(), 0.0);
    }
}
