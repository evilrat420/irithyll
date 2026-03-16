//! Thompson Sampling with Beta-Bernoulli conjugate posterior.
//!
//! Thompson Sampling is a Bayesian approach to the multi-armed bandit problem
//! that maintains a Beta posterior distribution for each arm's reward
//! probability. At each step it samples from each arm's posterior and selects
//! the arm with the highest sample — naturally balancing exploration (uncertain
//! arms have wide posteriors) with exploitation (well-rewarded arms have
//! posteriors concentrated near high values).
//!
//! # Algorithm
//!
//! For each arm *a* with posterior `Beta(alpha_a, beta_a)`:
//!
//! 1. **Select:** Sample `theta_a ~ Beta(alpha_a, beta_a)` for every arm.
//!    Return `argmax theta_a`.
//! 2. **Update:** On observing reward *r* in \[0, 1\] for arm *a*:
//!    - `alpha_a += r`
//!    - `beta_a += 1 - r`
//!
//! # Prior
//!
//! The default prior is `Beta(1, 1)` (uniform on \[0, 1\]), expressing no
//! initial preference. Use [`ThompsonSampling::with_prior`] for an informative
//! prior.
//!
//! # Example
//!
//! ```
//! use irithyll::bandits::{Bandit, ThompsonSampling};
//!
//! let mut ts = ThompsonSampling::new(3);
//! for _ in 0..200 {
//!     let arm = ts.select_arm();
//!     let reward = if arm == 0 { 0.9 } else { 0.1 };
//!     ts.update(arm, reward);
//! }
//! // Arm 0 should be selected most often.
//! assert_eq!(
//!     ts.arm_counts().iter().enumerate().max_by_key(|(_, &c)| c).unwrap().0,
//!     0
//! );
//! ```

use super::{beta_sample, Bandit};

/// Thompson Sampling bandit with Beta-Bernoulli conjugate posterior.
///
/// Each arm maintains a `Beta(alpha, beta)` posterior updated with observed
/// rewards in `[0, 1]`. Arm selection samples from each posterior and picks
/// the argmax, providing probability-matching exploration.
///
/// # Fields
///
/// - `alphas` / `betas` — posterior parameters per arm.
/// - `values` — running mean reward per arm (for [`Bandit::arm_values`]).
/// - `counts` — number of times each arm has been pulled.
#[derive(Debug, Clone)]
pub struct ThompsonSampling {
    n_arms: usize,
    alphas: Vec<f64>,
    betas: Vec<f64>,
    counts: Vec<u64>,
    values: Vec<f64>,
    total_pulls: u64,
    rng_state: u64,
    /// Initial alpha prior, stored for [`Bandit::reset`].
    alpha_prior: f64,
    /// Initial beta prior, stored for [`Bandit::reset`].
    beta_prior: f64,
}

impl ThompsonSampling {
    /// Create a new `ThompsonSampling` bandit with the default uniform
    /// `Beta(1, 1)` prior and seed 42.
    pub fn new(n_arms: usize) -> Self {
        Self::build(n_arms, 1.0, 1.0, 42)
    }

    /// Create a new `ThompsonSampling` bandit with a custom RNG seed and
    /// the default `Beta(1, 1)` prior.
    pub fn with_seed(n_arms: usize, seed: u64) -> Self {
        Self::build(n_arms, 1.0, 1.0, seed)
    }

    /// Create a new `ThompsonSampling` bandit with a custom `Beta(alpha, beta)`
    /// prior for all arms and seed 42.
    ///
    /// # Panics
    ///
    /// Panics if `alpha_prior` or `beta_prior` are not positive.
    pub fn with_prior(n_arms: usize, alpha_prior: f64, beta_prior: f64) -> Self {
        assert!(
            alpha_prior > 0.0,
            "alpha_prior must be positive, got {alpha_prior}"
        );
        assert!(
            beta_prior > 0.0,
            "beta_prior must be positive, got {beta_prior}"
        );
        Self::build(n_arms, alpha_prior, beta_prior, 42)
    }

    /// Internal constructor.
    fn build(n_arms: usize, alpha_prior: f64, beta_prior: f64, seed: u64) -> Self {
        Self {
            n_arms,
            alphas: vec![alpha_prior; n_arms],
            betas: vec![beta_prior; n_arms],
            counts: vec![0; n_arms],
            values: vec![0.0; n_arms],
            total_pulls: 0,
            rng_state: seed,
            alpha_prior,
            beta_prior,
        }
    }

    /// Per-arm posterior alpha parameters.
    #[inline]
    pub fn arm_alphas(&self) -> &[f64] {
        &self.alphas
    }

    /// Per-arm posterior beta parameters.
    #[inline]
    pub fn arm_betas(&self) -> &[f64] {
        &self.betas
    }
}

impl Bandit for ThompsonSampling {
    fn select_arm(&mut self) -> usize {
        let mut best_arm = 0;
        let mut best_sample = f64::NEG_INFINITY;

        for a in 0..self.n_arms {
            let sample = beta_sample(self.alphas[a], self.betas[a], &mut self.rng_state);
            if sample > best_sample {
                best_sample = sample;
                best_arm = a;
            }
        }

        best_arm
    }

    fn update(&mut self, arm: usize, reward: f64) {
        // Clamp reward to [0, 1] for Beta posterior validity.
        let r = reward.clamp(0.0, 1.0);

        // Update pull count and total.
        self.counts[arm] += 1;
        self.total_pulls += 1;

        // Incremental mean: value = value + (r - value) / count.
        let n = self.counts[arm] as f64;
        self.values[arm] += (r - self.values[arm]) / n;

        // Update posterior: alpha += r, beta += (1 - r).
        self.alphas[arm] += r;
        self.betas[arm] += 1.0 - r;
    }

    #[inline]
    fn n_arms(&self) -> usize {
        self.n_arms
    }

    #[inline]
    fn n_pulls(&self) -> u64 {
        self.total_pulls
    }

    fn reset(&mut self) {
        self.alphas.fill(self.alpha_prior);
        self.betas.fill(self.beta_prior);
        self.counts.fill(0);
        self.values.fill(0.0);
        self.total_pulls = 0;
    }

    #[inline]
    fn arm_values(&self) -> &[f64] {
        &self.values
    }

    #[inline]
    fn arm_counts(&self) -> &[u64] {
        &self.counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_best_arm_over_time() {
        let mut ts = ThompsonSampling::with_seed(4, 123);

        for _ in 0..300 {
            let arm = ts.select_arm();
            // Arm 0 always returns reward 1.0, others return 0.0.
            let reward = if arm == 0 { 1.0 } else { 0.0 };
            ts.update(arm, reward);
        }

        let counts = ts.arm_counts();
        assert!(
            counts[0] > counts[1] && counts[0] > counts[2] && counts[0] > counts[3],
            "arm 0 (always reward 1.0) should have the most pulls, got counts={:?}",
            counts
        );
    }

    #[test]
    fn prior_is_uniform() {
        let ts = ThompsonSampling::new(5);

        for a in 0..5 {
            assert_eq!(
                ts.arm_alphas()[a],
                1.0,
                "fresh alpha[{a}] should be 1.0, got {}",
                ts.arm_alphas()[a]
            );
            assert_eq!(
                ts.arm_betas()[a],
                1.0,
                "fresh beta[{a}] should be 1.0, got {}",
                ts.arm_betas()[a]
            );
        }
    }

    #[test]
    fn update_adjusts_posterior() {
        let mut ts = ThompsonSampling::new(3);

        ts.update(0, 1.0);

        assert_eq!(
            ts.arm_alphas()[0],
            2.0,
            "alpha[0] after reward=1.0 should be 2.0, got {}",
            ts.arm_alphas()[0]
        );
        assert_eq!(
            ts.arm_betas()[0],
            1.0,
            "beta[0] after reward=1.0 should be 1.0 (unchanged), got {}",
            ts.arm_betas()[0]
        );

        // Other arms should be untouched.
        assert_eq!(
            ts.arm_alphas()[1],
            1.0,
            "alpha[1] should remain at prior, got {}",
            ts.arm_alphas()[1]
        );
        assert_eq!(
            ts.arm_betas()[1],
            1.0,
            "beta[1] should remain at prior, got {}",
            ts.arm_betas()[1]
        );
    }

    #[test]
    fn reset_restores_prior() {
        let mut ts = ThompsonSampling::with_prior(3, 2.0, 5.0);

        // Train for a while.
        for _ in 0..50 {
            let arm = ts.select_arm();
            ts.update(arm, 0.7);
        }
        assert!(ts.n_pulls() > 0, "should have pulls before reset");

        // Reset.
        ts.reset();

        assert_eq!(ts.n_pulls(), 0, "total_pulls should be 0 after reset");
        for a in 0..3 {
            assert_eq!(
                ts.arm_alphas()[a],
                2.0,
                "alpha[{a}] should be restored to prior 2.0 after reset, got {}",
                ts.arm_alphas()[a]
            );
            assert_eq!(
                ts.arm_betas()[a],
                5.0,
                "beta[{a}] should be restored to prior 5.0 after reset, got {}",
                ts.arm_betas()[a]
            );
            assert_eq!(
                ts.arm_counts()[a],
                0,
                "count[{a}] should be 0 after reset, got {}",
                ts.arm_counts()[a]
            );
            assert_eq!(
                ts.arm_values()[a],
                0.0,
                "value[{a}] should be 0.0 after reset, got {}",
                ts.arm_values()[a]
            );
        }
    }

    #[test]
    fn custom_prior_works() {
        let ts = ThompsonSampling::with_prior(3, 2.0, 5.0);

        for a in 0..3 {
            assert_eq!(
                ts.arm_alphas()[a],
                2.0,
                "alpha[{a}] should start at custom prior 2.0, got {}",
                ts.arm_alphas()[a]
            );
            assert_eq!(
                ts.arm_betas()[a],
                5.0,
                "beta[{a}] should start at custom prior 5.0, got {}",
                ts.arm_betas()[a]
            );
        }
    }

    #[test]
    fn rewards_clamped_to_unit_interval() {
        let mut ts = ThompsonSampling::new(2);

        // Reward > 1.0 should be clamped to 1.0.
        ts.update(0, 5.0);
        assert_eq!(
            ts.arm_alphas()[0],
            2.0,
            "alpha[0] after reward=5.0 (clamped to 1.0) should be 2.0, got {}",
            ts.arm_alphas()[0]
        );
        assert_eq!(
            ts.arm_betas()[0],
            1.0,
            "beta[0] after reward=5.0 (clamped) should be 1.0, got {}",
            ts.arm_betas()[0]
        );

        // Reward < 0.0 should be clamped to 0.0.
        ts.update(1, -3.0);
        assert_eq!(
            ts.arm_alphas()[1],
            1.0,
            "alpha[1] after reward=-3.0 (clamped to 0.0) should be 1.0, got {}",
            ts.arm_alphas()[1]
        );
        assert_eq!(
            ts.arm_betas()[1],
            2.0,
            "beta[1] after reward=-3.0 (clamped) should be 2.0, got {}",
            ts.arm_betas()[1]
        );
    }

    #[test]
    fn arm_values_track_mean() {
        let mut ts = ThompsonSampling::new(2);

        // Arm 0: rewards 0.2, 0.4, 0.6 -> mean = 0.4
        ts.update(0, 0.2);
        ts.update(0, 0.4);
        ts.update(0, 0.6);

        let expected_mean = (0.2 + 0.4 + 0.6) / 3.0;
        let actual = ts.arm_values()[0];
        assert!(
            (actual - expected_mean).abs() < 1e-12,
            "arm 0 mean should be ~{expected_mean}, got {actual}"
        );

        // Arm 1: single reward 0.8 -> mean = 0.8
        ts.update(1, 0.8);
        assert!(
            (ts.arm_values()[1] - 0.8).abs() < 1e-12,
            "arm 1 mean should be 0.8 after single update, got {}",
            ts.arm_values()[1]
        );

        // Verify counts are correct.
        assert_eq!(
            ts.arm_counts()[0],
            3,
            "arm 0 should have 3 pulls, got {}",
            ts.arm_counts()[0]
        );
        assert_eq!(
            ts.arm_counts()[1],
            1,
            "arm 1 should have 1 pull, got {}",
            ts.arm_counts()[1]
        );
        assert_eq!(
            ts.n_pulls(),
            4,
            "total pulls should be 4, got {}",
            ts.n_pulls()
        );
    }
}
