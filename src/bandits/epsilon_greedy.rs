//! Epsilon-greedy bandit with optional exponential decay.

use super::{xorshift64, xorshift64_f64, Bandit};

/// Epsilon-greedy multi-armed bandit.
///
/// With probability `epsilon`, selects a uniformly random arm (exploration).
/// Otherwise, selects the arm with the highest estimated value (exploitation),
/// breaking ties by lowest index.
///
/// When constructed with [`EpsilonGreedy::with_decay`], epsilon decays
/// exponentially as `initial_epsilon * decay^t` after each update, where `t`
/// is the total number of pulls so far.
///
/// # Example
///
/// ```
/// use irithyll::bandits::{Bandit, EpsilonGreedy};
///
/// let mut bandit = EpsilonGreedy::new(3, 0.1);
/// for _ in 0..100 {
///     let arm = bandit.select_arm();
///     let reward = if arm == 0 { 1.0 } else { 0.0 };
///     bandit.update(arm, reward);
/// }
/// assert!(bandit.arm_values()[0] > bandit.arm_values()[1]);
/// ```
#[derive(Clone, Debug)]
pub struct EpsilonGreedy {
    n_arms: usize,
    epsilon: f64,
    values: Vec<f64>,
    counts: Vec<u64>,
    total_pulls: u64,
    rng_state: u64,
    decay: Option<f64>,
    initial_epsilon: f64,
}

impl EpsilonGreedy {
    /// Create an epsilon-greedy bandit with fixed epsilon and default seed (42).
    pub fn new(n_arms: usize, epsilon: f64) -> Self {
        Self::with_seed(n_arms, epsilon, 42)
    }

    /// Create an epsilon-greedy bandit with a custom RNG seed.
    pub fn with_seed(n_arms: usize, epsilon: f64, seed: u64) -> Self {
        assert!(n_arms > 0, "must have at least one arm");
        Self {
            n_arms,
            epsilon,
            values: vec![0.0; n_arms],
            counts: vec![0; n_arms],
            total_pulls: 0,
            rng_state: seed,
            decay: None,
            initial_epsilon: epsilon,
        }
    }

    /// Create an epsilon-greedy bandit with exponential epsilon decay.
    ///
    /// After each update, epsilon becomes `initial_epsilon * decay^t` where
    /// `t` is the total number of pulls. A typical decay value is `0.999`.
    pub fn with_decay(n_arms: usize, epsilon: f64, decay: f64) -> Self {
        assert!(n_arms > 0, "must have at least one arm");
        Self {
            n_arms,
            epsilon,
            values: vec![0.0; n_arms],
            counts: vec![0; n_arms],
            total_pulls: 0,
            rng_state: 42,
            decay: Some(decay),
            initial_epsilon: epsilon,
        }
    }

    /// Current epsilon value (may differ from initial if decay is active).
    pub fn current_epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl Bandit for EpsilonGreedy {
    fn select_arm(&mut self) -> usize {
        let r = xorshift64_f64(&mut self.rng_state);
        if r < self.epsilon {
            // Explore: uniform random arm.
            let raw = xorshift64(&mut self.rng_state);
            (raw as usize) % self.n_arms
        } else {
            // Exploit: argmax of values (ties broken by lowest index).
            let mut best_arm = 0;
            let mut best_val = self.values[0];
            for i in 1..self.n_arms {
                if self.values[i] > best_val {
                    best_val = self.values[i];
                    best_arm = i;
                }
            }
            best_arm
        }
    }

    fn update(&mut self, arm: usize, reward: f64) {
        assert!(arm < self.n_arms, "arm index out of bounds");
        self.counts[arm] += 1;
        self.total_pulls += 1;
        // Incremental mean update.
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm] as f64;
        // Decay epsilon if configured.
        if let Some(d) = self.decay {
            self.epsilon = self.initial_epsilon * d.powi(self.total_pulls as i32);
        }
    }

    fn n_arms(&self) -> usize {
        self.n_arms
    }

    fn n_pulls(&self) -> u64 {
        self.total_pulls
    }

    fn reset(&mut self) {
        self.values.fill(0.0);
        self.counts.fill(0);
        self.total_pulls = 0;
        self.epsilon = self.initial_epsilon;
    }

    fn arm_values(&self) -> &[f64] {
        &self.values
    }

    fn arm_counts(&self) -> &[u64] {
        &self.counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_best_arm_over_time() {
        let mut bandit = EpsilonGreedy::with_seed(4, 0.1, 7);
        for _ in 0..500 {
            let arm = bandit.select_arm();
            let reward = if arm == 0 { 1.0 } else { 0.0 };
            bandit.update(arm, reward);
        }
        let counts = bandit.arm_counts();
        let arm0_pulls = counts[0];
        for (i, &c) in counts.iter().enumerate().skip(1) {
            assert!(
                arm0_pulls > c,
                "arm 0 should have more pulls than arm {}: {} vs {}",
                i,
                arm0_pulls,
                c
            );
        }
    }

    #[test]
    fn epsilon_zero_always_exploits() {
        let mut bandit = EpsilonGreedy::with_seed(3, 0.0, 99);
        // Manually set arm 1 as the best.
        bandit.update(0, 0.2);
        bandit.update(1, 0.9);
        bandit.update(2, 0.5);

        // With epsilon=0, every subsequent selection should pick arm 1.
        for _ in 0..100 {
            let arm = bandit.select_arm();
            assert_eq!(
                arm, 1,
                "epsilon=0 should always exploit the best arm (1), got {}",
                arm
            );
        }
    }

    #[test]
    fn decay_reduces_epsilon() {
        let mut bandit = EpsilonGreedy::with_decay(3, 0.5, 0.99);
        let initial = bandit.current_epsilon();

        for _ in 0..50 {
            let arm = bandit.select_arm();
            bandit.update(arm, 0.5);
        }
        let after_50 = bandit.current_epsilon();
        assert!(
            after_50 < initial,
            "epsilon should decrease with decay: initial={}, after 50 pulls={}",
            initial,
            after_50
        );

        let expected = 0.5 * 0.99_f64.powi(50);
        assert!(
            (after_50 - expected).abs() < 1e-10,
            "epsilon after 50 pulls should be ~{}, got {}",
            expected,
            after_50
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut bandit = EpsilonGreedy::with_decay(3, 0.5, 0.99);
        for _ in 0..100 {
            let arm = bandit.select_arm();
            bandit.update(arm, 1.0);
        }
        assert!(bandit.n_pulls() > 0, "should have pulls before reset");

        bandit.reset();

        assert_eq!(bandit.n_pulls(), 0, "total pulls should be 0 after reset");
        assert!(
            bandit.arm_counts().iter().all(|&c| c == 0),
            "all arm counts should be 0 after reset"
        );
        assert!(
            bandit.arm_values().iter().all(|&v| v == 0.0),
            "all arm values should be 0.0 after reset"
        );
        assert!(
            (bandit.current_epsilon() - 0.5).abs() < f64::EPSILON,
            "epsilon should be restored to initial (0.5) after reset, got {}",
            bandit.current_epsilon()
        );
    }

    #[test]
    fn all_arms_explored_with_high_epsilon() {
        let mut bandit = EpsilonGreedy::with_seed(5, 1.0, 314);
        for _ in 0..100 {
            let arm = bandit.select_arm();
            bandit.update(arm, 0.5);
        }
        for (i, &c) in bandit.arm_counts().iter().enumerate() {
            assert!(
                c > 0,
                "arm {} should have been explored at least once with epsilon=1.0, count={}",
                i,
                c
            );
        }
    }

    #[test]
    fn incremental_mean_correct() {
        let mut bandit = EpsilonGreedy::with_seed(2, 0.0, 42);

        // Feed known rewards to arm 0: 1.0, 3.0, 2.0 -> mean = 2.0
        bandit.update(0, 1.0);
        bandit.update(0, 3.0);
        bandit.update(0, 2.0);

        let expected_mean = 2.0;
        let actual = bandit.arm_values()[0];
        assert!(
            (actual - expected_mean).abs() < 1e-12,
            "arm 0 mean should be {}, got {}",
            expected_mean,
            actual
        );
        assert_eq!(
            bandit.arm_counts()[0],
            3,
            "arm 0 should have 3 pulls, got {}",
            bandit.arm_counts()[0]
        );

        // Feed known rewards to arm 1: 0.5, 0.5 -> mean = 0.5
        bandit.update(1, 0.5);
        bandit.update(1, 0.5);

        let actual_1 = bandit.arm_values()[1];
        assert!(
            (actual_1 - 0.5).abs() < 1e-12,
            "arm 1 mean should be 0.5, got {}",
            actual_1
        );
        assert_eq!(
            bandit.n_pulls(),
            5,
            "total pulls should be 5, got {}",
            bandit.n_pulls()
        );
    }
}
