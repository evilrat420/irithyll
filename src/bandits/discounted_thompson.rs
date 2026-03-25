//! Thompson Sampling with exponential discounting for non-stationary bandits.
//!
//! Standard Thompson Sampling assumes stationary reward distributions — the
//! best arm stays the best forever. In many real-world settings (e.g., concept
//! drift in AutoML, changing user preferences), the optimal arm shifts over
//! time. Discounted Thompson Sampling (Qi et al., 2023) addresses this by
//! exponentially decaying posterior parameters toward the prior on every step,
//! giving recent observations more influence than old ones.
//!
//! # Algorithm
//!
//! Before each update, **all** arms' posteriors are decayed toward the prior:
//!
//! ```text
//! for all arms a:
//!     alpha_a = alpha_prior + discount * (alpha_a - alpha_prior)
//!     beta_a  = beta_prior  + discount * (beta_a  - beta_prior)
//! ```
//!
//! Then the selected arm is updated normally:
//!
//! ```text
//! alpha_arm += reward
//! beta_arm  += 1 - reward
//! ```
//!
//! The `discount` parameter controls the forgetting rate:
//! - `discount = 1.0` — no forgetting (equivalent to standard Thompson Sampling).
//! - `discount = 0.99` — aggressive forgetting, adapts quickly.
//! - `discount = 0.999` — gentle forgetting, adapts slowly.
//!
//! # Example
//!
//! ```
//! use irithyll::bandits::{Bandit, DiscountedThompsonSampling};
//!
//! let mut dts = DiscountedThompsonSampling::new(3, 0.99);
//! // Arm 0 is best initially.
//! for _ in 0..100 {
//!     let arm = dts.select_arm();
//!     let reward = if arm == 0 { 0.9 } else { 0.1 };
//!     dts.update(arm, reward);
//! }
//! // Now arm 2 becomes best — discounting lets us adapt.
//! for _ in 0..200 {
//!     let arm = dts.select_arm();
//!     let reward = if arm == 2 { 0.9 } else { 0.1 };
//!     dts.update(arm, reward);
//! }
//! assert_eq!(
//!     dts.arm_counts().iter().enumerate().max_by_key(|(_, &c)| c).unwrap().0,
//!     2
//! );
//! ```
//!
//! # Reference
//!
//! Qi, Y. et al. (2023). "Discounted Thompson Sampling for Non-Stationary
//! Bandits."

use super::{beta_sample, Bandit};

/// Thompson Sampling with exponential discounting (Qi+2023).
///
/// Extends standard Beta-Bernoulli Thompson Sampling with a forgetting
/// factor that decays posterior parameters toward the prior on every step.
/// This allows the algorithm to adapt in non-stationary environments where
/// the best arm can change over time.
///
/// Used in irithyll's AutoML module to select hyperparameter configurations
/// when the optimal configuration may shift due to concept drift.
#[derive(Debug, Clone)]
pub struct DiscountedThompsonSampling {
    n_arms: usize,
    alphas: Vec<f64>,
    betas: Vec<f64>,
    counts: Vec<u64>,
    values: Vec<f64>,
    total_pulls: u64,
    rng_state: u64,
    /// Initial alpha prior, stored for decay target and [`Bandit::reset`].
    alpha_prior: f64,
    /// Initial beta prior, stored for decay target and [`Bandit::reset`].
    beta_prior: f64,
    /// Forgetting factor in (0, 1]. Values closer to 1 forget more slowly.
    discount: f64,
}

impl DiscountedThompsonSampling {
    /// Create a new `DiscountedThompsonSampling` bandit with the default
    /// uniform `Beta(1, 1)` prior and seed 42.
    ///
    /// # Panics
    ///
    /// Panics if `discount` is not in `(0, 1]` or if `n_arms` is zero.
    pub fn new(n_arms: usize, discount: f64) -> Self {
        Self::build(n_arms, 1.0, 1.0, discount, 42)
    }

    /// Create a new `DiscountedThompsonSampling` bandit with a custom RNG
    /// seed and the default `Beta(1, 1)` prior.
    ///
    /// # Panics
    ///
    /// Panics if `discount` is not in `(0, 1]` or if `n_arms` is zero.
    pub fn with_seed(n_arms: usize, discount: f64, seed: u64) -> Self {
        Self::build(n_arms, 1.0, 1.0, discount, seed)
    }

    /// Create a new `DiscountedThompsonSampling` bandit with a custom
    /// `Beta(alpha, beta)` prior for all arms and seed 42.
    ///
    /// # Panics
    ///
    /// Panics if `alpha_prior` or `beta_prior` are not positive, if
    /// `discount` is not in `(0, 1]`, or if `n_arms` is zero.
    pub fn with_prior(n_arms: usize, discount: f64, alpha_prior: f64, beta_prior: f64) -> Self {
        assert!(
            alpha_prior > 0.0,
            "alpha_prior must be positive, got {alpha_prior}"
        );
        assert!(
            beta_prior > 0.0,
            "beta_prior must be positive, got {beta_prior}"
        );
        Self::build(n_arms, alpha_prior, beta_prior, discount, 42)
    }

    /// Internal constructor.
    fn build(n_arms: usize, alpha_prior: f64, beta_prior: f64, discount: f64, seed: u64) -> Self {
        assert!(n_arms > 0, "n_arms must be at least 1, got {n_arms}");
        assert!(
            discount > 0.0 && discount <= 1.0,
            "discount must be in (0, 1], got {discount}"
        );
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
            discount,
        }
    }

    /// The discount (forgetting) factor.
    #[inline]
    pub fn discount(&self) -> f64 {
        self.discount
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

impl Bandit for DiscountedThompsonSampling {
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

        // Decay ALL arms toward the prior.
        for a in 0..self.n_arms {
            self.alphas[a] = self.alpha_prior + self.discount * (self.alphas[a] - self.alpha_prior);
            self.betas[a] = self.beta_prior + self.discount * (self.betas[a] - self.beta_prior);
        }

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
    fn new_creates_valid() {
        let dts = DiscountedThompsonSampling::new(4, 0.99);

        assert_eq!(dts.n_arms(), 4, "should have 4 arms");
        assert_eq!(dts.n_pulls(), 0, "should start with 0 pulls");
        assert_eq!(dts.discount(), 0.99, "discount should be 0.99");

        for a in 0..4 {
            assert_eq!(
                dts.arm_alphas()[a],
                1.0,
                "alpha[{a}] should start at prior 1.0, got {}",
                dts.arm_alphas()[a]
            );
            assert_eq!(
                dts.arm_betas()[a],
                1.0,
                "beta[{a}] should start at prior 1.0, got {}",
                dts.arm_betas()[a]
            );
            assert_eq!(
                dts.arm_counts()[a],
                0,
                "count[{a}] should start at 0, got {}",
                dts.arm_counts()[a]
            );
            assert_eq!(
                dts.arm_values()[a],
                0.0,
                "value[{a}] should start at 0.0, got {}",
                dts.arm_values()[a]
            );
        }
    }

    #[test]
    fn discount_decays_posteriors() {
        let mut dts = DiscountedThompsonSampling::with_seed(2, 0.95, 100);
        let mut ts = super::super::ThompsonSampling::with_seed(2, 100);

        // Feed identical rewards to arm 0 for both.
        for _ in 0..50 {
            dts.update(0, 1.0);
            ts.update(0, 1.0);
        }

        // Discounted version should have smaller alpha for arm 0 because
        // old observations are forgotten.
        assert!(
            dts.arm_alphas()[0] < ts.arm_alphas()[0],
            "discounted alpha[0] ({}) should be less than standard alpha[0] ({}) \
             because old observations are decayed",
            dts.arm_alphas()[0],
            ts.arm_alphas()[0]
        );
    }

    #[test]
    fn discount_one_equals_standard() {
        // With discount=1.0, discounted TS should match standard TS exactly.
        let mut dts = DiscountedThompsonSampling::with_seed(3, 1.0, 42);
        let mut ts = super::super::ThompsonSampling::with_seed(3, 42);

        // Run both with identical update sequences.
        let rewards = [0.5, 1.0, 0.0, 0.7, 0.3, 0.9, 0.1, 0.8, 0.2, 0.6];
        for (i, &r) in rewards.iter().enumerate() {
            let arm = i % 3;
            dts.update(arm, r);
            ts.update(arm, r);
        }

        for a in 0..3 {
            assert!(
                (dts.arm_alphas()[a] - ts.arm_alphas()[a]).abs() < 1e-12,
                "with discount=1.0, alpha[{a}] should match standard TS: \
                 discounted={}, standard={}",
                dts.arm_alphas()[a],
                ts.arm_alphas()[a]
            );
            assert!(
                (dts.arm_betas()[a] - ts.arm_betas()[a]).abs() < 1e-12,
                "with discount=1.0, beta[{a}] should match standard TS: \
                 discounted={}, standard={}",
                dts.arm_betas()[a],
                ts.arm_betas()[a]
            );
        }
    }

    #[test]
    fn adapts_to_change() {
        // Phase 1: arm 0 is best (reward 0.9 vs 0.1).
        // Phase 2: arm 1 becomes best (reward 0.9 vs 0.1).
        // Discounted TS should adapt and pull arm 1 more in phase 2
        // compared to standard TS.
        let seed = 777;
        let mut dts = DiscountedThompsonSampling::with_seed(2, 0.98, seed);
        let mut ts = super::super::ThompsonSampling::with_seed(2, seed);

        // Phase 1: arm 0 is best.
        for _ in 0..500 {
            let arm_d = dts.select_arm();
            let r_d = if arm_d == 0 { 0.9 } else { 0.1 };
            dts.update(arm_d, r_d);

            let arm_s = ts.select_arm();
            let r_s = if arm_s == 0 { 0.9 } else { 0.1 };
            ts.update(arm_s, r_s);
        }

        // Phase 2: arm 1 is best. Track phase-2 selections.
        let mut dts_arm1_count: u64 = 0;
        let mut ts_arm1_count: u64 = 0;

        for _ in 0..500 {
            let arm_d = dts.select_arm();
            let r_d = if arm_d == 1 { 0.9 } else { 0.1 };
            dts.update(arm_d, r_d);
            if arm_d == 1 {
                dts_arm1_count += 1;
            }

            let arm_s = ts.select_arm();
            let r_s = if arm_s == 1 { 0.9 } else { 0.1 };
            ts.update(arm_s, r_s);
            if arm_s == 1 {
                ts_arm1_count += 1;
            }
        }

        assert!(
            dts_arm1_count > ts_arm1_count,
            "discounted TS should select arm 1 more often in phase 2 \
             (dts={dts_arm1_count}, standard={ts_arm1_count}) because it \
             forgets the old arm-0 dominance"
        );
    }

    #[test]
    #[should_panic(expected = "discount must be in (0, 1]")]
    fn invalid_discount_zero_panics() {
        DiscountedThompsonSampling::new(2, 0.0);
    }

    #[test]
    #[should_panic(expected = "discount must be in (0, 1]")]
    fn invalid_discount_above_one_panics() {
        DiscountedThompsonSampling::new(2, 1.5);
    }

    #[test]
    #[should_panic(expected = "n_arms must be at least 1")]
    fn invalid_n_arms_panics() {
        DiscountedThompsonSampling::new(0, 0.99);
    }

    #[test]
    fn reward_clamped() {
        let mut dts = DiscountedThompsonSampling::new(2, 0.99);

        // Reward > 1.0 should be clamped to 1.0.
        dts.update(0, 5.0);
        // After decay: alpha[0] = 1.0 + 0.99*(1.0-1.0) = 1.0, then +1.0 = 2.0
        assert!(
            (dts.arm_alphas()[0] - 2.0).abs() < 1e-12,
            "alpha[0] after reward=5.0 (clamped to 1.0) should be 2.0, got {}",
            dts.arm_alphas()[0]
        );
        // beta[0] = 1.0 + 0.99*(1.0-1.0) = 1.0, then +0.0 = 1.0
        assert!(
            (dts.arm_betas()[0] - 1.0).abs() < 1e-12,
            "beta[0] after reward=5.0 (clamped to 1.0) should be 1.0, got {}",
            dts.arm_betas()[0]
        );

        // Reward < 0.0 should be clamped to 0.0.
        // Reset to clean state for arm 1 check.
        dts.reset();
        dts.update(1, -3.0);
        // After decay: alpha[1] = 1.0, then +0.0 = 1.0
        assert!(
            (dts.arm_alphas()[1] - 1.0).abs() < 1e-12,
            "alpha[1] after reward=-3.0 (clamped to 0.0) should be 1.0, got {}",
            dts.arm_alphas()[1]
        );
        // beta[1] = 1.0, then +1.0 = 2.0
        assert!(
            (dts.arm_betas()[1] - 2.0).abs() < 1e-12,
            "beta[1] after reward=-3.0 (clamped to 0.0) should be 2.0, got {}",
            dts.arm_betas()[1]
        );
    }

    #[test]
    fn reset_restores_prior() {
        let mut dts = DiscountedThompsonSampling::with_prior(3, 0.95, 2.0, 5.0);

        // Train for a while.
        for _ in 0..50 {
            let arm = dts.select_arm();
            dts.update(arm, 0.7);
        }
        assert!(dts.n_pulls() > 0, "should have pulls before reset");

        // Reset.
        dts.reset();

        assert_eq!(dts.n_pulls(), 0, "total_pulls should be 0 after reset");
        for a in 0..3 {
            assert_eq!(
                dts.arm_alphas()[a],
                2.0,
                "alpha[{a}] should be restored to prior 2.0 after reset, got {}",
                dts.arm_alphas()[a]
            );
            assert_eq!(
                dts.arm_betas()[a],
                5.0,
                "beta[{a}] should be restored to prior 5.0 after reset, got {}",
                dts.arm_betas()[a]
            );
            assert_eq!(
                dts.arm_counts()[a],
                0,
                "count[{a}] should be 0 after reset, got {}",
                dts.arm_counts()[a]
            );
            assert_eq!(
                dts.arm_values()[a],
                0.0,
                "value[{a}] should be 0.0 after reset, got {}",
                dts.arm_values()[a]
            );
        }
    }

    #[test]
    fn with_prior_custom() {
        let dts = DiscountedThompsonSampling::with_prior(3, 0.99, 3.0, 7.0);

        assert_eq!(dts.discount(), 0.99, "discount should be 0.99");
        for a in 0..3 {
            assert_eq!(
                dts.arm_alphas()[a],
                3.0,
                "alpha[{a}] should start at custom prior 3.0, got {}",
                dts.arm_alphas()[a]
            );
            assert_eq!(
                dts.arm_betas()[a],
                7.0,
                "beta[{a}] should start at custom prior 7.0, got {}",
                dts.arm_betas()[a]
            );
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let seed = 314159;
        let mut dts1 = DiscountedThompsonSampling::with_seed(4, 0.99, seed);
        let mut dts2 = DiscountedThompsonSampling::with_seed(4, 0.99, seed);

        let mut arms1 = Vec::new();
        let mut arms2 = Vec::new();

        for _ in 0..100 {
            let a1 = dts1.select_arm();
            let a2 = dts2.select_arm();
            arms1.push(a1);
            arms2.push(a2);

            // Give same deterministic reward based on arm.
            let r = (a1 as f64) * 0.25;
            dts1.update(a1, r);
            dts2.update(a2, r);
        }

        assert_eq!(
            arms1, arms2,
            "same seed should produce identical arm selections"
        );
    }
}
