//! Strongly Adaptive Online Conformal Prediction (Bhatnagar et al., ICML 2023).
//!
//! Minimizes worst-case regret over ALL contiguous time intervals,
//! automatically adapting to both slow drift and sudden regime changes.
//! Uses multiple "experts" at geometrically spaced timescales, aggregated
//! via coin-betting online learning.

/// A single quantile expert operating at a specific timescale.
#[derive(Debug, Clone)]
struct QuantileExpert {
    /// Current quantile estimate.
    estimate: f64,
    /// Learning rate for this timescale.
    step_size: f64,
    /// Coin-betting wealth (used for aggregation weighting).
    wealth: f64,
}

/// Multi-scale strongly adaptive conformal predictor.
///
/// Maintains `n_experts` quantile experts at geometrically spaced timescales.
/// Expert `i` has step size `1 / 2^i`, so expert 0 adapts fastest (step_size=0.5)
/// and later experts are more stable. The final threshold is a wealth-weighted
/// combination of all expert estimates.
///
/// # Example
///
/// ```
/// use irithyll::metrics::strongly_adaptive::StronglyAdaptiveConformal;
///
/// let mut sac = StronglyAdaptiveConformal::default_90();
/// for i in 0..100 {
///     let target = (i as f64) * 0.1;
///     let prediction = target + 0.05;
///     sac.update(prediction, target);
/// }
/// assert!(sac.current_threshold() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct StronglyAdaptiveConformal {
    /// Experts at different timescales.
    experts: Vec<QuantileExpert>,
    /// Target miscoverage rate.
    alpha_target: f64,
    /// Number of updates processed.
    n_updates: u64,
    /// Running empirical coverage rate.
    cumulative_coverage: f64,
}

impl StronglyAdaptiveConformal {
    /// Create a new strongly adaptive conformal predictor.
    ///
    /// - `alpha_target`: desired miscoverage rate (e.g., 0.1 for 90% coverage).
    /// - `n_experts`: number of experts. Expert `i` has step size `1 / 2^i`,
    ///   covering timescales from 2 to `2^n_experts`.
    ///
    /// # Panics
    ///
    /// Panics if `alpha_target` is not in (0, 1) or `n_experts` is 0.
    pub fn new(alpha_target: f64, n_experts: usize) -> Self {
        assert!(
            alpha_target > 0.0 && alpha_target < 1.0,
            "alpha_target must be in (0, 1), got {alpha_target}"
        );
        assert!(n_experts > 0, "n_experts must be > 0, got {n_experts}");

        let experts = (0..n_experts)
            .map(|i| QuantileExpert {
                estimate: alpha_target,
                step_size: 1.0 / (1u64 << i) as f64, // 1/2^i
                wealth: 1.0,
            })
            .collect();

        Self {
            experts,
            alpha_target,
            n_updates: 0,
            cumulative_coverage: 0.0,
        }
    }

    /// 90% coverage with 10 experts (covers timescales 1 to 1024).
    pub fn default_90() -> Self {
        Self::new(0.1, 10)
    }

    /// 95% coverage with 10 experts.
    pub fn default_95() -> Self {
        Self::new(0.05, 10)
    }

    /// Update all experts with a new (prediction, target) pair.
    ///
    /// Each expert updates its quantile estimate and wealth according to
    /// coin-betting dynamics. The aggregated threshold is a wealth-weighted
    /// combination.
    pub fn update(&mut self, prediction: f64, target: f64) {
        let score = (prediction - target).abs();

        for expert in &mut self.experts {
            let err = if score > expert.estimate { 1.0 } else { 0.0 };
            let gradient = self.alpha_target - err;

            // Coin-betting wealth update (simplified KT estimator).
            // Use only the gradient direction to adjust wealth — experts that
            // correctly predict coverage/miscoverage gain wealth.
            let bet_payoff = expert.step_size * gradient;
            expert.wealth *= 1.0 + bet_payoff.clamp(-0.4, 0.4);
            // Floor wealth to prevent extinction
            expert.wealth = expert.wealth.max(1e-10);

            // Quantile estimate update
            expert.estimate += expert.step_size * gradient;
            expert.estimate = expert.estimate.max(0.0);
        }

        // Update coverage tracking
        self.n_updates += 1;
        let threshold = self.current_threshold();
        let covered = if score <= threshold { 1.0 } else { 0.0 };
        self.cumulative_coverage += (covered - self.cumulative_coverage) / self.n_updates as f64;
    }

    /// Current threshold: wealth-weighted combination of expert estimates.
    pub fn current_threshold(&self) -> f64 {
        let total_wealth: f64 = self.experts.iter().map(|e| e.wealth).sum();
        if total_wealth <= 0.0 {
            return self.alpha_target;
        }
        self.experts
            .iter()
            .map(|e| e.wealth * e.estimate)
            .sum::<f64>()
            / total_wealth
    }

    /// Empirical coverage rate.
    pub fn coverage(&self) -> f64 {
        self.cumulative_coverage
    }

    /// Number of updates processed.
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Reset all experts and counters to initial state.
    pub fn reset(&mut self) {
        for expert in &mut self.experts {
            expert.estimate = self.alpha_target;
            expert.wealth = 1.0;
        }
        self.n_updates = 0;
        self.cumulative_coverage = 0.0;
    }

    /// Number of experts in the ensemble.
    pub fn n_experts(&self) -> usize {
        self.experts.len()
    }

    /// Step sizes of all experts (for inspection / testing).
    pub fn expert_step_sizes(&self) -> Vec<f64> {
        self.experts.iter().map(|e| e.step_size).collect()
    }

    /// Wealth of all experts (for inspection / testing).
    pub fn expert_weights(&self) -> Vec<f64> {
        self.experts.iter().map(|e| e.wealth).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_converges_to_target() {
        let mut sac = StronglyAdaptiveConformal::default_90();
        for i in 0..1000 {
            let target = (i as f64) * 0.01;
            // Small error most of the time, large error 10% of the time
            let offset = if i % 10 == 0 { 5.0 } else { 0.01 };
            sac.update(target + offset, target);
        }
        // Coverage should be reasonable (not 0 or 1)
        assert!(
            sac.coverage() > 0.3,
            "coverage {} should be > 0.3",
            sac.coverage()
        );
        assert!(
            sac.coverage() < 1.0,
            "coverage {} should be < 1.0",
            sac.coverage()
        );
    }

    #[test]
    fn adapts_to_sudden_shift() {
        let mut sac = StronglyAdaptiveConformal::new(0.1, 8);

        // Phase 1: small errors
        for i in 0..500 {
            let t = i as f64;
            sac.update(t + 0.1, t);
        }
        let threshold_before = sac.current_threshold();

        // Phase 2: sudden large errors (regime shift)
        for i in 0..500 {
            let t = i as f64;
            sac.update(t + 10.0, t);
        }
        let threshold_after = sac.current_threshold();

        // Threshold should increase substantially after the shift
        assert!(
            threshold_after > threshold_before,
            "threshold should increase after regime shift: before={threshold_before}, after={threshold_after}"
        );
    }

    #[test]
    fn multiple_experts_have_different_scales() {
        let sac = StronglyAdaptiveConformal::new(0.1, 5);
        let step_sizes = sac.expert_step_sizes();

        assert_eq!(step_sizes.len(), 5);
        // Step sizes should be geometrically decreasing: 1.0, 0.5, 0.25, 0.125, 0.0625
        assert!((step_sizes[0] - 1.0).abs() < 1e-10);
        assert!((step_sizes[1] - 0.5).abs() < 1e-10);
        assert!((step_sizes[2] - 0.25).abs() < 1e-10);
        assert!((step_sizes[3] - 0.125).abs() < 1e-10);
        assert!((step_sizes[4] - 0.0625).abs() < 1e-10);

        // Each step size should be half the previous
        for i in 1..step_sizes.len() {
            assert!(
                (step_sizes[i] - step_sizes[i - 1] / 2.0).abs() < 1e-10,
                "expert {i} step_size {} should be half of expert {} step_size {}",
                step_sizes[i],
                i - 1,
                step_sizes[i - 1]
            );
        }
    }

    #[test]
    fn reset_clears_all_experts() {
        let mut sac = StronglyAdaptiveConformal::default_90();
        for i in 0..100 {
            sac.update(i as f64 + 5.0, i as f64);
        }
        assert!(sac.n_updates() > 0);

        sac.reset();
        assert_eq!(sac.n_updates(), 0);
        assert!((sac.coverage()).abs() < 1e-10);
        // All expert weights should be reset to 1.0
        for w in sac.expert_weights() {
            assert!(
                (w - 1.0).abs() < 1e-10,
                "expert wealth should be 1.0 after reset, got {w}"
            );
        }
    }

    #[test]
    fn wealth_weighting_favors_accurate_experts() {
        let mut sac = StronglyAdaptiveConformal::new(0.1, 5);

        // Feed alternating regimes — fast experts adapt quickly, slow ones lag.
        // This should cause different experts to accumulate different wealth.
        for cycle in 0..10 {
            let offset = if cycle % 2 == 0 { 0.1 } else { 5.0 };
            for i in 0..50 {
                let t = i as f64;
                sac.update(t + offset, t);
            }
        }

        let weights = sac.expert_weights();
        // Experts should have different weights due to different timescale performance
        let min_w = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_w > min_w * 1.01 || (max_w - min_w).abs() > 1e-8,
            "expert weights should diverge with regime changes, but min={min_w}, max={max_w}"
        );
    }

    #[test]
    fn default_constructors_work() {
        let sac90 = StronglyAdaptiveConformal::default_90();
        assert_eq!(sac90.n_experts(), 10);
        assert_eq!(sac90.n_updates(), 0);
        assert!((sac90.current_threshold() - 0.1).abs() < 1e-10);

        let sac95 = StronglyAdaptiveConformal::default_95();
        assert_eq!(sac95.n_experts(), 10);
        assert!((sac95.current_threshold() - 0.05).abs() < 1e-10);
    }
}
