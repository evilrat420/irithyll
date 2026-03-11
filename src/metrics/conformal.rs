//! Adaptive Conformal Inference (ACI) for prediction intervals.
//!
//! Implements the ACI algorithm from Gibbs & Candes (2021) which provides
//! distribution-free coverage guarantees even under concept drift. The tracker
//! monitors whether observed values fall within prediction intervals and adapts
//! the miscoverage rate `alpha_t` online.
//!
//! # Usage
//!
//! Train two SGBT models — one for the low quantile and one for the high
//! quantile — and feed their predictions along with the actual target into
//! the ACI tracker. Use `effective_quantiles()` to determine what quantile
//! levels to use for the next prediction interval.

/// Adaptive Conformal Inference (ACI) tracker for prediction intervals.
///
/// Monitors coverage of prediction intervals and adapts the miscoverage
/// rate `alpha_t` online to maintain the target coverage probability,
/// even under distribution shift.
///
/// # Algorithm
///
/// After each observation:
///
/// ```text
/// err_t = indicator(target NOT in [lower, upper])
/// alpha_t += gamma * (target_alpha - err_t)
/// alpha_t = clamp(alpha_t, 0.001, 0.999)
/// ```
///
/// # Example
///
/// ```
/// use irithyll::metrics::conformal::AdaptiveConformalInterval;
///
/// let mut aci = AdaptiveConformalInterval::new(0.1, 0.01);
/// // After each prediction:
/// aci.update(5.0, 3.0, 7.0); // target=5, interval=[3,7] => covered
/// let (low_q, high_q) = aci.effective_quantiles();
/// // Use low_q and high_q as tau for the next quantile predictions
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveConformalInterval {
    /// Target miscoverage rate (e.g., 0.1 for 90% coverage).
    target_alpha: f64,
    /// Current adaptive miscoverage rate.
    alpha_t: f64,
    /// Step size for adaptation.
    gamma: f64,
    /// Running count of covered observations.
    n_covered: u64,
    /// Total observations.
    n_total: u64,
}

impl AdaptiveConformalInterval {
    /// Create a new ACI tracker.
    ///
    /// - `target_alpha`: desired miscoverage rate (e.g., 0.1 for 90% coverage).
    /// - `gamma`: adaptation step size. Larger values track drift faster but
    ///   with more variance. Typical range: 0.001 to 0.05.
    ///
    /// # Panics
    ///
    /// Panics if `target_alpha` is not in (0, 1) or `gamma` is not positive.
    pub fn new(target_alpha: f64, gamma: f64) -> Self {
        assert!(
            target_alpha > 0.0 && target_alpha < 1.0,
            "target_alpha must be in (0, 1), got {target_alpha}"
        );
        assert!(gamma > 0.0, "gamma must be > 0, got {gamma}");
        Self {
            target_alpha,
            alpha_t: target_alpha,
            gamma,
            n_covered: 0,
            n_total: 0,
        }
    }

    /// Update with a new observation and its prediction interval.
    ///
    /// - `target`: the actual observed value.
    /// - `lower`: lower bound of the prediction interval.
    /// - `upper`: upper bound of the prediction interval.
    pub fn update(&mut self, target: f64, lower: f64, upper: f64) {
        self.n_total += 1;
        let covered = target >= lower && target <= upper;
        if covered {
            self.n_covered += 1;
        }

        let err = if covered { 0.0 } else { 1.0 };
        self.alpha_t += self.gamma * (self.target_alpha - err);
        self.alpha_t = self.alpha_t.clamp(0.001, 0.999);
    }

    /// Effective quantile levels for the next prediction interval.
    ///
    /// Returns `(low_tau, high_tau)` where:
    /// - `low_tau = alpha_t / 2`
    /// - `high_tau = 1 - alpha_t / 2`
    ///
    /// Use these as the tau parameters for your low/high quantile models.
    pub fn effective_quantiles(&self) -> (f64, f64) {
        (self.alpha_t / 2.0, 1.0 - self.alpha_t / 2.0)
    }

    /// Empirical coverage rate: fraction of targets that fell within intervals.
    pub fn empirical_coverage(&self) -> f64 {
        if self.n_total == 0 {
            return 0.0;
        }
        self.n_covered as f64 / self.n_total as f64
    }

    /// Current adaptive miscoverage rate.
    pub fn current_alpha(&self) -> f64 {
        self.alpha_t
    }

    /// Target miscoverage rate.
    pub fn target_alpha(&self) -> f64 {
        self.target_alpha
    }

    /// Number of observations processed.
    pub fn n_samples(&self) -> u64 {
        self.n_total
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.alpha_t = self.target_alpha;
        self.n_covered = 0;
        self.n_total = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn aci_perfect_coverage() {
        // All targets inside interval => err=0 each time
        // alpha_t += gamma * (alpha - 0) = gamma * alpha (positive)
        // So alpha_t increases, meaning quantiles spread LESS (narrower intervals)
        let mut aci = AdaptiveConformalInterval::new(0.1, 0.01);
        let initial = aci.current_alpha();
        for _ in 0..100 {
            aci.update(5.0, 0.0, 10.0); // always covered
        }
        assert!(aci.current_alpha() > initial);
        assert!((aci.empirical_coverage() - 1.0).abs() < EPS);
    }

    #[test]
    fn aci_no_coverage() {
        // All targets outside interval => err=1 each time
        // alpha_t += gamma * (alpha - 1) = gamma * (alpha - 1) (negative)
        // So alpha_t decreases, meaning quantiles spread MORE (wider intervals)
        let mut aci = AdaptiveConformalInterval::new(0.1, 0.01);
        let initial = aci.current_alpha();
        for _ in 0..100 {
            aci.update(100.0, 0.0, 10.0); // never covered
        }
        assert!(aci.current_alpha() < initial);
        assert!((aci.empirical_coverage()).abs() < EPS);
    }

    #[test]
    fn aci_converges_to_target() {
        // Feed data where exactly 90% are covered (alpha=0.1 => 90% coverage)
        let mut aci = AdaptiveConformalInterval::new(0.1, 0.005);
        for i in 0..10_000 {
            if i % 10 == 0 {
                // 10% not covered
                aci.update(100.0, 0.0, 10.0);
            } else {
                // 90% covered
                aci.update(5.0, 0.0, 10.0);
            }
        }
        // alpha_t should stabilize near target_alpha
        assert!((aci.current_alpha() - 0.1).abs() < 0.05);
    }

    #[test]
    fn aci_effective_quantiles() {
        let aci = AdaptiveConformalInterval::new(0.1, 0.01);
        let (lo, hi) = aci.effective_quantiles();
        assert!((lo - 0.05).abs() < EPS);
        assert!((hi - 0.95).abs() < EPS);
    }

    #[test]
    fn aci_clamp_lower() {
        let mut aci = AdaptiveConformalInterval::new(0.1, 1.0); // huge gamma
                                                                // Many no-coverage => alpha_t should hit floor at 0.001
        for _ in 0..1000 {
            aci.update(100.0, 0.0, 10.0);
        }
        assert!((aci.current_alpha() - 0.001).abs() < EPS);
    }

    #[test]
    fn aci_clamp_upper() {
        let mut aci = AdaptiveConformalInterval::new(0.9, 1.0); // huge gamma
                                                                // Many full-coverage => alpha_t should hit ceiling at 0.999
        for _ in 0..1000 {
            aci.update(5.0, 0.0, 10.0);
        }
        assert!((aci.current_alpha() - 0.999).abs() < EPS);
    }

    #[test]
    fn aci_reset() {
        let mut aci = AdaptiveConformalInterval::new(0.1, 0.01);
        for _ in 0..50 {
            aci.update(5.0, 0.0, 10.0);
        }
        aci.reset();
        assert_eq!(aci.n_samples(), 0);
        assert!((aci.current_alpha() - 0.1).abs() < EPS);
        assert!((aci.empirical_coverage()).abs() < EPS);
    }

    #[test]
    #[should_panic(expected = "target_alpha must be in (0, 1)")]
    fn aci_invalid_alpha() {
        AdaptiveConformalInterval::new(0.0, 0.01);
    }

    #[test]
    #[should_panic(expected = "gamma must be > 0")]
    fn aci_invalid_gamma() {
        AdaptiveConformalInterval::new(0.1, 0.0);
    }
}
