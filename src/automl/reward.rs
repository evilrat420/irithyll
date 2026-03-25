//! Reward normalization for bandit-guided hyperparameter selection.
//!
//! Converts raw metric values (where lower is better) into [0, 1] rewards
//! suitable for Thompson Sampling and other bandit algorithms. Uses a running
//! EWMA baseline to normalize relative to recently observed performance.

/// Normalizes metric values to [0, 1] for Thompson Sampling consumption.
///
/// Uses a running baseline (EWMA of metric values seen so far) to normalize:
/// `reward = (1 - metric_value / baseline).clamp(0, 1)`
///
/// Lower metric values (better performance) produce higher rewards.
///
/// # Examples
///
/// ```
/// use irithyll::automl::RewardNormalizer;
///
/// let mut norm = RewardNormalizer::new(0.1);
///
/// // First value establishes baseline, returns 0.5 (neutral).
/// let r0 = norm.normalize(10.0);
/// assert!((r0 - 0.5).abs() < 1e-12);
///
/// // Much better (lower) metric produces higher reward.
/// let r1 = norm.normalize(2.0);
/// assert!(r1 > 0.5);
///
/// // Much worse (higher) metric produces lower reward.
/// let r2 = norm.normalize(50.0);
/// assert!(r2 < 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct RewardNormalizer {
    baseline: f64,
    alpha: f64,
    initialized: bool,
}

impl RewardNormalizer {
    /// Create a new reward normalizer with the given EWMA decay factor.
    ///
    /// `alpha` controls how quickly the baseline adapts to new metric values.
    /// Higher alpha means faster adaptation (more weight on recent values).
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not in (0, 1].
    pub fn new(alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "RewardNormalizer alpha must be in (0, 1], got {alpha}"
        );
        Self {
            baseline: 0.0,
            alpha,
            initialized: false,
        }
    }

    /// Create a new reward normalizer from an EWMA span.
    ///
    /// Computes `alpha = 2 / (span + 1)`, which gives the EWMA an effective
    /// window of approximately `span` observations.
    ///
    /// # Panics
    ///
    /// Panics if `span == 0`.
    pub fn with_span(span: usize) -> Self {
        assert!(span > 0, "RewardNormalizer span must be > 0, got {span}");
        let alpha = 2.0 / (span as f64 + 1.0);
        Self::new(alpha)
    }

    /// Normalize a metric value to a [0, 1] reward.
    ///
    /// - On the first call, sets the baseline and returns 0.5 (neutral).
    /// - On subsequent calls, updates the EWMA baseline and computes:
    ///   `reward = (1 - metric_value / baseline).clamp(0, 1)`
    /// - If the baseline is zero, returns 0.5 to avoid division by zero.
    pub fn normalize(&mut self, metric_value: f64) -> f64 {
        if !self.initialized {
            self.baseline = metric_value;
            self.initialized = true;
            return 0.5;
        }

        // Update baseline with EWMA.
        self.baseline = self.alpha * metric_value + (1.0 - self.alpha) * self.baseline;

        // Guard against zero baseline.
        if self.baseline == 0.0 {
            return 0.5;
        }

        // Lower metric_value relative to baseline -> higher reward.
        (1.0 - metric_value / self.baseline).clamp(0.0, 1.0)
    }

    /// Current baseline value (EWMA of metric values).
    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    /// Reset the normalizer to its initial (uninitialized) state.
    pub fn reset(&mut self) {
        self.baseline = 0.0;
        self.initialized = false;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// First normalized value should return 0.5 (neutral).
    #[test]
    fn normalizer_first_value_returns_half() {
        let mut norm = RewardNormalizer::new(0.1);
        let reward = norm.normalize(10.0);
        assert!(
            (reward - 0.5).abs() < 1e-12,
            "first value should return 0.5, got {reward}"
        );
    }

    /// Lower metric values (better performance) should produce higher rewards.
    #[test]
    fn better_performance_higher_reward() {
        let mut norm = RewardNormalizer::new(0.01);
        norm.normalize(10.0); // baseline = 10.0

        // Metric value well below baseline should yield high reward.
        // With alpha=0.01: baseline becomes 0.01*5 + 0.99*10 = 9.95
        // reward = 1 - 5/9.95 = 0.497... still < 0.5 with alpha=0.01
        // Use a very small value to ensure reward > 0.5.
        let reward = norm.normalize(1.0);
        // baseline = 0.01*1 + 0.99*10 = 9.91
        // reward = 1 - 1/9.91 = 0.899
        assert!(
            reward > 0.5,
            "lower metric should produce reward > 0.5, got {reward}"
        );
    }

    /// Higher metric values (worse performance) should produce lower rewards.
    #[test]
    fn worse_performance_lower_reward() {
        let mut norm = RewardNormalizer::new(0.01);
        norm.normalize(10.0); // baseline = 10.0

        // Metric value well above baseline should yield low reward.
        let reward = norm.normalize(20.0);
        assert!(
            reward < 0.5,
            "higher metric should produce reward < 0.5, got {reward}"
        );
    }

    /// Reward should always be clamped to [0, 1].
    #[test]
    fn reward_always_clamped() {
        let mut norm = RewardNormalizer::new(0.1);
        norm.normalize(1.0); // baseline = 1.0

        // Very large metric (much worse than baseline).
        let reward_bad = norm.normalize(1000.0);
        assert!(
            reward_bad >= 0.0 && reward_bad <= 1.0,
            "reward should be in [0, 1], got {reward_bad}"
        );

        // Very small metric (much better than baseline).
        let reward_good = norm.normalize(0.001);
        assert!(
            reward_good >= 0.0 && reward_good <= 1.0,
            "reward should be in [0, 1], got {reward_good}"
        );

        // Negative metric value.
        let reward_neg = norm.normalize(-5.0);
        assert!(
            reward_neg >= 0.0 && reward_neg <= 1.0,
            "reward should be in [0, 1] even for negative metric, got {reward_neg}"
        );
    }

    /// Baseline should track toward recent values over time.
    #[test]
    fn baseline_tracks_values() {
        let mut norm = RewardNormalizer::new(0.5); // fast-adapting alpha
        norm.normalize(10.0);
        assert!(
            (norm.baseline() - 10.0).abs() < 1e-12,
            "initial baseline should be 10.0, got {}",
            norm.baseline()
        );

        // Feed consistently lower values; baseline should decrease.
        for _ in 0..20 {
            norm.normalize(2.0);
        }
        assert!(
            norm.baseline() < 5.0,
            "baseline should have moved toward 2.0, got {}",
            norm.baseline()
        );

        // Feed consistently higher values; baseline should increase.
        for _ in 0..20 {
            norm.normalize(100.0);
        }
        assert!(
            norm.baseline() > 50.0,
            "baseline should have moved toward 100.0, got {}",
            norm.baseline()
        );
    }

    /// Reset should clear all state, making the next normalize return 0.5.
    #[test]
    fn reset_clears_state() {
        let mut norm = RewardNormalizer::new(0.1);
        norm.normalize(10.0);
        norm.normalize(5.0);
        assert!(
            norm.baseline() > 0.0,
            "baseline should be non-zero before reset"
        );

        norm.reset();
        assert!(
            (norm.baseline() - 0.0).abs() < 1e-12,
            "baseline should be 0 after reset, got {}",
            norm.baseline()
        );

        // Next normalize should behave as first call.
        let reward = norm.normalize(42.0);
        assert!(
            (reward - 0.5).abs() < 1e-12,
            "after reset, first value should return 0.5, got {reward}"
        );
    }

    /// When baseline is exactly zero, normalize should return 0.5.
    #[test]
    fn zero_baseline_returns_half() {
        let mut norm = RewardNormalizer::new(1.0); // alpha=1.0 means baseline = last value
        norm.normalize(0.0); // baseline = 0.0

        // Second call: baseline is now 0.0 (EWMA with alpha=1.0: baseline = alpha*0 + 0*baseline = 0)
        let reward = norm.normalize(5.0);
        // After EWMA update: baseline = 1.0 * 5.0 + 0.0 * 0.0 = 5.0, so not zero.
        // Let's create the actual zero-baseline scenario.
        let mut norm2 = RewardNormalizer::new(0.5);
        norm2.normalize(0.0); // baseline = 0.0

        // Next call: baseline = 0.5 * metric + 0.5 * 0.0 = 0.5 * metric.
        // For baseline to stay 0.0, we need metric = 0.0 too.
        let reward2 = norm2.normalize(0.0);
        // baseline = 0.5 * 0 + 0.5 * 0 = 0, so should return 0.5.
        assert!(
            (reward2 - 0.5).abs() < 1e-12,
            "zero baseline should return 0.5, got {reward2}"
        );

        // Also verify the first normalize with 0.0 returns 0.5.
        let mut norm3 = RewardNormalizer::new(0.1);
        let reward3 = norm3.normalize(0.0);
        assert!(
            (reward3 - 0.5).abs() < 1e-12,
            "first value of 0.0 should return 0.5, got {reward3}"
        );

        // Verify non-zero reward with actually zero baseline.
        assert!(reward.is_finite(), "reward should be finite, got {reward}");
    }

    /// Verify with_span computes alpha correctly.
    #[test]
    fn with_span_alpha_computation() {
        let norm = RewardNormalizer::with_span(19); // alpha = 2/(19+1) = 0.1
        assert!(
            (norm.alpha - 0.1).abs() < 1e-12,
            "span=19 should give alpha=0.1, got {}",
            norm.alpha
        );

        let norm2 = RewardNormalizer::with_span(1); // alpha = 2/(1+1) = 1.0
        assert!(
            (norm2.alpha - 1.0).abs() < 1e-12,
            "span=1 should give alpha=1.0, got {}",
            norm2.alpha
        );
    }

    /// Verify new() panics on invalid alpha.
    #[test]
    #[should_panic(expected = "alpha must be in (0, 1]")]
    fn new_panics_on_zero_alpha() {
        RewardNormalizer::new(0.0);
    }

    /// Verify new() panics on alpha > 1.
    #[test]
    #[should_panic(expected = "alpha must be in (0, 1]")]
    fn new_panics_on_alpha_over_one() {
        RewardNormalizer::new(1.5);
    }

    /// Verify with_span panics on zero span.
    #[test]
    #[should_panic(expected = "span must be > 0")]
    fn with_span_panics_on_zero() {
        RewardNormalizer::with_span(0);
    }
}
