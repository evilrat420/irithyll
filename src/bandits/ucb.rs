//! Upper Confidence Bound bandits: UCB1 and UCB-Tuned.

use super::Bandit;

// ---------------------------------------------------------------------------
// UCB1 (Auer et al., 2002)
// ---------------------------------------------------------------------------

/// Upper Confidence Bound 1 bandit (Auer et al., 2002).
///
/// Selects the arm maximizing `Q(a) + sqrt(2 * ln(t) / N(a))` where `Q(a)` is
/// the estimated reward, `t` is the total number of pulls, and `N(a)` is the
/// pull count for arm `a`. Arms with zero pulls are selected first in
/// round-robin order.
///
/// # Example
///
/// ```
/// use irithyll::bandits::{Bandit, UCB1};
///
/// let mut ucb = UCB1::new(3);
/// for _ in 0..100 {
///     let arm = ucb.select_arm();
///     let reward = if arm == 0 { 0.9 } else { 0.1 };
///     ucb.update(arm, reward);
/// }
/// assert_eq!(
///     ucb.arm_counts().iter().enumerate().max_by_key(|(_, c)| *c).unwrap().0,
///     0,
/// );
/// ```
#[derive(Clone, Debug)]
pub struct UCB1 {
    n_arms: usize,
    values: Vec<f64>,
    counts: Vec<u64>,
    total_pulls: u64,
}

impl UCB1 {
    /// Create a new UCB1 bandit with `n_arms` arms.
    ///
    /// # Panics
    ///
    /// Panics if `n_arms == 0`.
    pub fn new(n_arms: usize) -> Self {
        assert!(n_arms > 0, "UCB1 requires at least 1 arm");
        Self {
            n_arms,
            values: vec![0.0; n_arms],
            counts: vec![0; n_arms],
            total_pulls: 0,
        }
    }
}

impl Bandit for UCB1 {
    fn select_arm(&mut self) -> usize {
        // Round-robin for arms that haven't been pulled yet.
        for (i, &c) in self.counts.iter().enumerate() {
            if c == 0 {
                return i;
            }
        }

        let ln_t = (self.total_pulls as f64).ln();
        let mut best_arm = 0;
        let mut best_value = f64::NEG_INFINITY;

        for a in 0..self.n_arms {
            let ucb = self.values[a] + (2.0 * ln_t / self.counts[a] as f64).sqrt();
            if ucb > best_value {
                best_value = ucb;
                best_arm = a;
            }
        }

        best_arm
    }

    fn update(&mut self, arm: usize, reward: f64) {
        self.counts[arm] += 1;
        self.total_pulls += 1;
        // Incremental mean: Q_new = Q_old + (reward - Q_old) / N
        let n = self.counts[arm] as f64;
        self.values[arm] += (reward - self.values[arm]) / n;
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
    }

    fn arm_values(&self) -> &[f64] {
        &self.values
    }

    fn arm_counts(&self) -> &[u64] {
        &self.counts
    }
}

// ---------------------------------------------------------------------------
// UCB-Tuned (Auer et al., 2002 — improved)
// ---------------------------------------------------------------------------

/// Upper Confidence Bound Tuned bandit (Auer et al., 2002).
///
/// An improvement over [`UCB1`] that uses per-arm variance estimates to produce
/// tighter confidence bounds. Selects the arm maximizing:
///
/// ```text
/// Q(a) + sqrt(ln(t) / N(a) * min(0.25, V_a))
/// ```
///
/// where `V_a = (sum_sq[a] / N(a)) - Q(a)^2 + sqrt(2 * ln(t) / N(a))` is the
/// estimated variance plus an exploration bonus.
///
/// # Example
///
/// ```
/// use irithyll::bandits::{Bandit, UCBTuned};
///
/// let mut ucb = UCBTuned::new(3);
/// for _ in 0..100 {
///     let arm = ucb.select_arm();
///     let reward = if arm == 0 { 0.9 } else { 0.1 };
///     ucb.update(arm, reward);
/// }
/// assert_eq!(
///     ucb.arm_counts().iter().enumerate().max_by_key(|(_, c)| *c).unwrap().0,
///     0,
/// );
/// ```
#[derive(Clone, Debug)]
pub struct UCBTuned {
    n_arms: usize,
    values: Vec<f64>,
    counts: Vec<u64>,
    total_pulls: u64,
    sum_sq: Vec<f64>,
}

impl UCBTuned {
    /// Create a new UCB-Tuned bandit with `n_arms` arms.
    ///
    /// # Panics
    ///
    /// Panics if `n_arms == 0`.
    pub fn new(n_arms: usize) -> Self {
        assert!(n_arms > 0, "UCBTuned requires at least 1 arm");
        Self {
            n_arms,
            values: vec![0.0; n_arms],
            counts: vec![0; n_arms],
            total_pulls: 0,
            sum_sq: vec![0.0; n_arms],
        }
    }
}

impl Bandit for UCBTuned {
    fn select_arm(&mut self) -> usize {
        // Round-robin for arms that haven't been pulled yet.
        for (i, &c) in self.counts.iter().enumerate() {
            if c == 0 {
                return i;
            }
        }

        let ln_t = (self.total_pulls as f64).ln();
        let mut best_arm = 0;
        let mut best_value = f64::NEG_INFINITY;

        for a in 0..self.n_arms {
            let n_a = self.counts[a] as f64;
            let mean = self.values[a];

            // Variance estimate: E[X^2] - E[X]^2 + exploration bonus
            let variance = self.sum_sq[a] / n_a - mean * mean + (2.0 * ln_t / n_a).sqrt();
            let v_a = variance.min(0.25);

            let ucb = mean + (ln_t / n_a * v_a).sqrt();
            if ucb > best_value {
                best_value = ucb;
                best_arm = a;
            }
        }

        best_arm
    }

    fn update(&mut self, arm: usize, reward: f64) {
        self.counts[arm] += 1;
        self.total_pulls += 1;
        self.sum_sq[arm] += reward * reward;
        // Incremental mean: Q_new = Q_old + (reward - Q_old) / N
        let n = self.counts[arm] as f64;
        self.values[arm] += (reward - self.values[arm]) / n;
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
        self.sum_sq.fill(0.0);
    }

    fn arm_values(&self) -> &[f64] {
        &self.values
    }

    fn arm_counts(&self) -> &[u64] {
        &self.counts
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- UCB1 tests --------------------------------------------------------

    #[test]
    fn ucb1_selects_best_arm() {
        let mut bandit = UCB1::new(3);
        let rewards = [0.9, 0.1, 0.5];

        for _ in 0..500 {
            let arm = bandit.select_arm();
            bandit.update(arm, rewards[arm]);
        }

        let counts = bandit.arm_counts();
        assert!(
            counts[0] > counts[1] && counts[0] > counts[2],
            "arm 0 (reward=0.9) should have most pulls, got counts {:?}",
            counts
        );
    }

    #[test]
    fn ucb1_explores_all_arms_first() {
        let mut bandit = UCB1::new(5);
        let mut seen = vec![false; 5];

        for _ in 0..5 {
            let arm = bandit.select_arm();
            seen[arm] = true;
            bandit.update(arm, 0.5);
        }

        assert!(
            seen.iter().all(|&s| s),
            "first n_arms pulls should cover all arms (round-robin), seen {:?}",
            seen
        );
    }

    #[test]
    fn ucb1_reset_clears_state() {
        let mut bandit = UCB1::new(3);

        for _ in 0..50 {
            let arm = bandit.select_arm();
            bandit.update(arm, 0.5);
        }

        bandit.reset();

        assert_eq!(
            bandit.n_pulls(),
            0,
            "total pulls should be 0 after reset, got {}",
            bandit.n_pulls()
        );
        assert!(
            bandit.arm_values().iter().all(|&v| v == 0.0),
            "all values should be 0.0 after reset, got {:?}",
            bandit.arm_values()
        );
        assert!(
            bandit.arm_counts().iter().all(|&c| c == 0),
            "all counts should be 0 after reset, got {:?}",
            bandit.arm_counts()
        );
    }

    #[test]
    fn ucb1_incremental_mean_correct() {
        let mut bandit = UCB1::new(2);

        // Manually feed arm 0 a known sequence: 1.0, 0.0, 0.5
        // Expected mean after each: 1.0, 0.5, 0.5
        bandit.update(0, 1.0);
        assert!(
            (bandit.arm_values()[0] - 1.0).abs() < 1e-12,
            "after [1.0]: expected mean 1.0, got {}",
            bandit.arm_values()[0]
        );

        bandit.update(0, 0.0);
        assert!(
            (bandit.arm_values()[0] - 0.5).abs() < 1e-12,
            "after [1.0, 0.0]: expected mean 0.5, got {}",
            bandit.arm_values()[0]
        );

        bandit.update(0, 0.5);
        assert!(
            (bandit.arm_values()[0] - 0.5).abs() < 1e-12,
            "after [1.0, 0.0, 0.5]: expected mean 0.5, got {}",
            bandit.arm_values()[0]
        );

        // Arm 1 untouched
        assert_eq!(
            bandit.arm_values()[1],
            0.0,
            "arm 1 should still have value 0.0, got {}",
            bandit.arm_values()[1]
        );
        assert_eq!(
            bandit.arm_counts()[1],
            0,
            "arm 1 should still have count 0, got {}",
            bandit.arm_counts()[1]
        );
    }

    #[test]
    fn ucb1_single_arm_always_selected() {
        let mut bandit = UCB1::new(1);

        for _ in 0..100 {
            let arm = bandit.select_arm();
            assert_eq!(arm, 0, "single-arm bandit should always select arm 0");
            bandit.update(arm, 0.5);
        }

        assert_eq!(
            bandit.arm_counts()[0],
            100,
            "arm 0 should have 100 pulls, got {}",
            bandit.arm_counts()[0]
        );
    }

    // -- UCBTuned tests ----------------------------------------------------

    #[test]
    fn ucb_tuned_selects_best_arm() {
        let mut bandit = UCBTuned::new(3);
        let rewards = [0.9, 0.1, 0.5];

        for _ in 0..500 {
            let arm = bandit.select_arm();
            bandit.update(arm, rewards[arm]);
        }

        let counts = bandit.arm_counts();
        assert!(
            counts[0] > counts[1] && counts[0] > counts[2],
            "arm 0 (reward=0.9) should have most pulls, got counts {:?}",
            counts
        );
    }

    #[test]
    fn ucb_tuned_explores_all_arms_first() {
        let mut bandit = UCBTuned::new(5);
        let mut seen = vec![false; 5];

        for _ in 0..5 {
            let arm = bandit.select_arm();
            seen[arm] = true;
            bandit.update(arm, 0.5);
        }

        assert!(
            seen.iter().all(|&s| s),
            "first n_arms pulls should cover all arms (round-robin), seen {:?}",
            seen
        );
    }

    #[test]
    fn ucb_tuned_reset_clears_state() {
        let mut bandit = UCBTuned::new(3);

        for _ in 0..50 {
            let arm = bandit.select_arm();
            bandit.update(arm, 0.5);
        }

        bandit.reset();

        assert_eq!(
            bandit.n_pulls(),
            0,
            "total pulls should be 0 after reset, got {}",
            bandit.n_pulls()
        );
        assert!(
            bandit.arm_values().iter().all(|&v| v == 0.0),
            "all values should be 0.0 after reset, got {:?}",
            bandit.arm_values()
        );
        assert!(
            bandit.arm_counts().iter().all(|&c| c == 0),
            "all counts should be 0 after reset, got {:?}",
            bandit.arm_counts()
        );
    }

    #[test]
    fn ucb_tuned_variance_tracking() {
        let mut bandit = UCBTuned::new(2);

        // Feed arm 0: 1.0, 0.0 -> sum_sq should be 1.0 + 0.0 = 1.0
        bandit.update(0, 1.0);
        bandit.update(0, 0.0);

        // Access sum_sq directly
        assert!(
            (bandit.sum_sq[0] - 1.0).abs() < 1e-12,
            "sum_sq[0] after [1.0, 0.0] should be 1.0, got {}",
            bandit.sum_sq[0]
        );

        // Mean should be 0.5, variance = E[X^2] - E[X]^2 = 0.5 - 0.25 = 0.25
        let mean = bandit.arm_values()[0];
        let empirical_var = bandit.sum_sq[0] / bandit.arm_counts()[0] as f64 - mean * mean;
        assert!(
            (empirical_var - 0.25).abs() < 1e-12,
            "empirical variance after [1.0, 0.0] should be 0.25, got {}",
            empirical_var
        );

        // Feed arm 0: 0.5 -> sum_sq should be 1.0 + 0.25 = 1.25
        bandit.update(0, 0.5);
        assert!(
            (bandit.sum_sq[0] - 1.25).abs() < 1e-12,
            "sum_sq[0] after [1.0, 0.0, 0.5] should be 1.25, got {}",
            bandit.sum_sq[0]
        );

        // Arm 1 untouched
        assert!(
            (bandit.sum_sq[1]).abs() < 1e-12,
            "sum_sq[1] should be 0.0, got {}",
            bandit.sum_sq[1]
        );
    }
}
