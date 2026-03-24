//! Streaming quantile estimation via stochastic gradient descent on pinball loss.
//!
//! The [`StreamingQuantileTracker`] maintains an online estimate of any quantile
//! of a streaming distribution. Each observation triggers one SGD step on the
//! pinball (quantile) loss, requiring O(1) time and space per update.

/// Tracks a single quantile of a streaming distribution.
///
/// Uses the subgradient of the pinball loss:
/// ```text
/// if value > estimate:  estimate += lr * quantile
/// if value <= estimate: estimate -= lr * (1 - quantile)
/// ```
///
/// This converges to the true quantile for any stationary distribution and
/// tracks slowly drifting distributions.
///
/// # Example
///
/// ```
/// use irithyll::metrics::quantile_tracker::StreamingQuantileTracker;
///
/// let mut tracker = StreamingQuantileTracker::new(0.5, 0.5);
/// // Feed symmetric pairs to avoid ordering bias
/// for _ in 0..20 {
///     for i in 0..50 {
///         tracker.update(i as f64);
///         tracker.update(99.0 - i as f64);
///     }
/// }
/// // Median of 0..99 is ~49.5
/// assert!((tracker.estimate() - 49.5).abs() < 10.0);
/// ```
#[derive(Debug, Clone)]
pub struct StreamingQuantileTracker {
    /// Target quantile in (0, 1).
    quantile: f64,
    /// Current quantile estimate.
    estimate: f64,
    /// Step size for SGD updates.
    lr: f64,
    /// Number of updates processed.
    n_updates: u64,
}

impl StreamingQuantileTracker {
    /// Create a new streaming quantile tracker.
    ///
    /// - `quantile`: target quantile in (0, 1), e.g. 0.5 for median, 0.9 for 90th percentile.
    /// - `lr`: learning rate / step size. Larger values adapt faster but oscillate more.
    ///
    /// # Panics
    ///
    /// Panics if `quantile` is not in (0, 1) or `lr` is not positive.
    pub fn new(quantile: f64, lr: f64) -> Self {
        assert!(
            quantile > 0.0 && quantile < 1.0,
            "quantile must be in (0, 1), got {quantile}"
        );
        assert!(lr > 0.0, "lr must be > 0, got {lr}");
        Self {
            quantile,
            estimate: 0.0,
            lr,
            n_updates: 0,
        }
    }

    /// Track the median (quantile=0.5) with default learning rate 0.01.
    pub fn default_median() -> Self {
        Self::new(0.5, 0.01)
    }

    /// Track the 90th percentile with default learning rate 0.01.
    pub fn default_90th() -> Self {
        Self::new(0.9, 0.01)
    }

    /// Update with a new observed value (one SGD step on pinball loss).
    pub fn update(&mut self, value: f64) {
        if value > self.estimate {
            self.estimate += self.lr * self.quantile;
        } else {
            self.estimate -= self.lr * (1.0 - self.quantile);
        }
        self.n_updates += 1;
    }

    /// Current quantile estimate.
    pub fn estimate(&self) -> f64 {
        self.estimate
    }

    /// Target quantile.
    pub fn quantile(&self) -> f64 {
        self.quantile
    }

    /// Number of updates processed.
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Reset to initial state (estimate = 0).
    pub fn reset(&mut self) {
        self.estimate = 0.0;
        self.n_updates = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_converges() {
        // Feed alternating high/low values from uniform [0, 999] to avoid
        // sequential ordering bias. With lr=0.5 and many samples, the tracker
        // should converge near the true median (499.5).
        let mut tracker = StreamingQuantileTracker::new(0.5, 0.5);
        for _ in 0..20 {
            // Feed in symmetric pairs to avoid order bias
            for i in 0..500 {
                tracker.update(i as f64); // low half
                tracker.update(999.0 - i as f64); // high half
            }
        }
        assert!(
            (tracker.estimate() - 499.5).abs() < 50.0,
            "median estimate {} should be near 499.5",
            tracker.estimate()
        );
    }

    #[test]
    fn percentile_90th_converges() {
        // Feed symmetric pairs. P90 of uniform [0, 999] is ~899.
        let mut tracker = StreamingQuantileTracker::new(0.9, 0.5);
        for _ in 0..20 {
            for i in 0..500 {
                tracker.update(i as f64);
                tracker.update(999.0 - i as f64);
            }
        }
        assert!(
            (tracker.estimate() - 899.0).abs() < 60.0,
            "P90 estimate {} should be near 899",
            tracker.estimate()
        );
    }

    #[test]
    fn reacts_to_distribution_shift() {
        let mut tracker = StreamingQuantileTracker::new(0.5, 1.0);

        // Phase 1: values around 100
        for _ in 0..2000 {
            tracker.update(100.0);
        }
        let est_before = tracker.estimate();

        // Phase 2: shift to values around 500
        for _ in 0..2000 {
            tracker.update(500.0);
        }
        let est_after = tracker.estimate();

        assert!(
            est_after > est_before,
            "estimate should increase after shift: before={est_before}, after={est_after}"
        );
        assert!(
            (est_after - 500.0).abs() < 50.0,
            "estimate {} should track toward 500 after shift",
            est_after
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut tracker = StreamingQuantileTracker::default_median();
        for i in 0..100 {
            tracker.update(i as f64);
        }
        assert!(tracker.n_updates() > 0);
        assert!(tracker.estimate().abs() > 0.0);

        tracker.reset();
        assert_eq!(tracker.n_updates(), 0);
        assert!((tracker.estimate()).abs() < 1e-10);
    }

    #[test]
    fn estimate_starts_at_zero() {
        let tracker = StreamingQuantileTracker::default_90th();
        assert!((tracker.estimate()).abs() < 1e-10);
        assert_eq!(tracker.n_updates(), 0);
    }

    #[test]
    fn lr_controls_adaptation_speed() {
        let mut fast = StreamingQuantileTracker::new(0.5, 1.0);
        let mut slow = StreamingQuantileTracker::new(0.5, 0.01);

        // Feed the same data
        for _ in 0..100 {
            fast.update(100.0);
            slow.update(100.0);
        }

        // Fast learner should be closer to 100
        assert!(
            (fast.estimate() - 100.0).abs() < (slow.estimate() - 100.0).abs(),
            "fast ({}) should be closer to 100 than slow ({})",
            fast.estimate(),
            slow.estimate()
        );
    }
}
