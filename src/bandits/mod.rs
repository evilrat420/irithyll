//! Multi-armed bandit algorithms for online decision-making.
//!
//! This module provides both context-free and contextual bandit algorithms
//! for the exploration-exploitation trade-off in streaming settings.
//!
//! # Context-free bandits
//!
//! Implement the [`Bandit`] trait — select arms based only on past rewards:
//!
//! - [`EpsilonGreedy`] — random exploration with probability ε (optionally decaying).
//! - [`UCB1`] — upper confidence bound (Auer et al., 2002).
//! - [`UCBTuned`] — UCB with per-arm variance estimates for tighter bounds.
//! - [`ThompsonSampling`] — Bayesian arm selection via Beta posterior sampling.
//!
//! # Contextual bandits
//!
//! Implement the [`ContextualBandit`] trait — use feature context for arm selection:
//!
//! - [`LinUCB`] — linear upper confidence bound with per-arm ridge regression
//!   (Li et al., 2010).
//!
//! # Example
//!
//! ```
//! use irithyll::bandits::{Bandit, UCB1};
//!
//! let mut bandit = UCB1::new(3); // 3 arms
//! for _ in 0..100 {
//!     let arm = bandit.select_arm();
//!     let reward = if arm == 1 { 0.8 } else { 0.3 }; // arm 1 is best
//!     bandit.update(arm, reward);
//! }
//! assert_eq!(bandit.arm_counts().iter().copied().max(), bandit.arm_counts().get(1).copied());
//! ```

mod epsilon_greedy;
mod lin_ucb;
mod thompson;
mod ucb;

pub use epsilon_greedy::EpsilonGreedy;
pub use lin_ucb::LinUCB;
pub use thompson::ThompsonSampling;
pub use ucb::{UCBTuned, UCB1};

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// A multi-armed bandit that selects arms to maximize cumulative reward.
///
/// All context-free bandit algorithms implement this trait. Arm indices are
/// `0..n_arms`.
pub trait Bandit: Send + Sync {
    /// Select the next arm to pull.
    ///
    /// May mutate internal RNG state.
    fn select_arm(&mut self) -> usize;

    /// Update the bandit with observed reward for the given arm.
    ///
    /// Rewards should typically be in `[0, 1]` for Thompson Sampling (Beta
    /// posterior), but UCB and epsilon-greedy work with arbitrary real rewards.
    fn update(&mut self, arm: usize, reward: f64);

    /// Number of arms.
    fn n_arms(&self) -> usize;

    /// Total number of pulls across all arms.
    fn n_pulls(&self) -> u64;

    /// Reset to initial state (all counts and estimates zeroed).
    fn reset(&mut self);

    /// Per-arm estimated reward values.
    fn arm_values(&self) -> &[f64];

    /// Per-arm pull counts.
    fn arm_counts(&self) -> &[u64];
}

/// A contextual bandit that uses feature vectors for arm selection.
///
/// Unlike [`Bandit`], contextual bandits condition their arm selection on
/// an observed feature vector (context), enabling personalized decisions.
pub trait ContextualBandit: Send + Sync {
    /// Select the next arm given context features.
    fn select_arm(&mut self, context: &[f64]) -> usize;

    /// Update with observed reward for the chosen arm and context.
    fn update(&mut self, arm: usize, context: &[f64], reward: f64);

    /// Number of arms.
    fn n_arms(&self) -> usize;

    /// Total number of pulls across all arms.
    fn n_pulls(&self) -> u64;

    /// Reset to initial state.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// Shared RNG utilities (xorshift64, no external dep)
// ---------------------------------------------------------------------------

/// Advance xorshift64 state and return raw u64.
#[inline]
pub(crate) fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Uniform f64 in [0, 1) from xorshift64.
#[inline]
pub(crate) fn xorshift64_f64(state: &mut u64) -> f64 {
    // Use 53-bit mantissa for full f64 precision.
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Standard normal via Box-Muller transform (returns one sample).
pub(crate) fn standard_normal(state: &mut u64) -> f64 {
    loop {
        let u1 = xorshift64_f64(state);
        let u2 = xorshift64_f64(state);
        if u1 > 0.0 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            return r * theta.cos();
        }
    }
}

/// Gamma(shape, 1) sample via Marsaglia-Tsang (2000).
///
/// Requires `shape > 0`. For shape < 1, uses the boost:
/// `Gamma(a, 1) = Gamma(a+1, 1) * U^(1/a)`.
pub(crate) fn gamma_sample(shape: f64, state: &mut u64) -> f64 {
    debug_assert!(shape > 0.0, "gamma shape must be positive");

    if shape < 1.0 {
        // Boost for shape < 1
        let u = xorshift64_f64(state);
        return gamma_sample(shape + 1.0, state) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = standard_normal(state);
        let v_base = 1.0 + c * x;
        if v_base <= 0.0 {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = xorshift64_f64(state);

        // Squeeze test
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        // Full test
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Beta(alpha, beta) sample from two independent Gamma samples.
pub(crate) fn beta_sample(alpha: f64, beta: f64, state: &mut u64) -> f64 {
    let x = gamma_sample(alpha, state);
    let y = gamma_sample(beta, state);
    x / (x + y)
}

// ---------------------------------------------------------------------------
// Tests for shared utilities
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift64_deterministic() {
        let mut s1: u64 = 42;
        let mut s2: u64 = 42;
        let a: Vec<u64> = (0..50).map(|_| xorshift64(&mut s1)).collect();
        let b: Vec<u64> = (0..50).map(|_| xorshift64(&mut s2)).collect();
        assert_eq!(a, b, "same seed should produce identical sequence");
    }

    #[test]
    fn xorshift64_f64_in_unit_interval() {
        let mut s: u64 = 123;
        for _ in 0..1000 {
            let v = xorshift64_f64(&mut s);
            assert!(
                (0.0..1.0).contains(&v),
                "xorshift64_f64 should be in [0, 1), got {}",
                v
            );
        }
    }

    #[test]
    fn standard_normal_finite() {
        let mut s: u64 = 99;
        for _ in 0..1000 {
            let v = standard_normal(&mut s);
            assert!(v.is_finite(), "standard_normal should be finite, got {}", v);
        }
    }

    #[test]
    fn gamma_sample_positive() {
        let mut s: u64 = 77;
        for shape in [0.5, 1.0, 2.0, 5.0, 10.0] {
            for _ in 0..200 {
                let v = gamma_sample(shape, &mut s);
                assert!(
                    v > 0.0 && v.is_finite(),
                    "gamma({}) should be positive finite, got {}",
                    shape,
                    v
                );
            }
        }
    }

    #[test]
    fn beta_sample_in_unit_interval() {
        let mut s: u64 = 55;
        for (a, b) in [(1.0, 1.0), (2.0, 5.0), (0.5, 0.5), (10.0, 10.0)] {
            for _ in 0..200 {
                let v = beta_sample(a, b, &mut s);
                assert!(
                    (0.0..=1.0).contains(&v) && v.is_finite(),
                    "beta({}, {}) should be in [0, 1], got {}",
                    a,
                    b,
                    v
                );
            }
        }
    }

    #[test]
    fn beta_sample_mean_approximately_correct() {
        let mut s: u64 = 200;
        let alpha = 3.0;
        let beta = 7.0;
        let n = 5000;
        let sum: f64 = (0..n).map(|_| beta_sample(alpha, beta, &mut s)).sum();
        let mean = sum / n as f64;
        let expected = alpha / (alpha + beta); // 0.3
        assert!(
            (mean - expected).abs() < 0.05,
            "beta({}, {}) mean should be ~{}, got {}",
            alpha,
            beta,
            expected,
            mean
        );
    }
}
