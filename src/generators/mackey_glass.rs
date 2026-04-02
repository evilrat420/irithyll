//! Mackey-Glass chaotic time series generator (Mackey & Glass, 1977).
//!
//! Generates samples from the Mackey-Glass delay differential equation:
//!
//! ```text
//! dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
//! ```
//!
//! Default parameters: `beta=0.2, gamma=0.1, n=10, tau=17`.
//!
//! Features are a lag-embedding `[x(t-L+1), ..., x(t)]` and the target is `x(t+1)`.
//! This is the standard reservoir computing benchmark (Jaeger 2001, Lukoševičius 2009).

use super::{Rng, StreamGenerator, TaskType};

/// Mackey-Glass chaotic time series generator.
///
/// # Parameters
/// - `seed`: PRNG seed (used for initial conditions)
/// - `tau`: delay parameter (default 17)
/// - `n_lags`: number of lag features in the embedding (default 6)
/// - `warmup`: number of initial steps to discard (default 500)
#[derive(Debug, Clone)]
pub struct MackeyGlass {
    /// Pre-generated time series.
    series: Vec<f64>,
    /// Current read position in the series.
    cursor: usize,
    /// Number of lag-embedding features.
    n_lags: usize,
}

impl MackeyGlass {
    /// Create a Mackey-Glass generator with default parameters.
    ///
    /// `tau=17, n_lags=6, warmup=500`, generates enough data for ~100K samples.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, 17, 6, 500, 100_000)
    }

    /// Create with custom parameters.
    ///
    /// - `tau`: delay (governs chaotic behavior; tau>=17 is chaotic)
    /// - `n_lags`: lag-embedding dimension (number of features)
    /// - `warmup`: transient steps to skip
    /// - `max_samples`: maximum number of samples to pre-generate
    pub fn with_config(
        seed: u64,
        tau: usize,
        n_lags: usize,
        warmup: usize,
        max_samples: usize,
    ) -> Self {
        let mut rng = Rng::new(seed);
        let beta = 0.2;
        let gamma = 0.1;
        let n_exp = 10.0;

        // Total steps: warmup + lag window + samples + 1 (for one-step-ahead target)
        let total = warmup + n_lags + max_samples + 1;
        let mut x = vec![0.0; total.max(tau + 1)];

        // Initialize first tau steps with small random perturbations around 0.9
        for val in x.iter_mut().take(tau) {
            *val = 0.9 + rng.uniform_range(-0.1, 0.1);
        }

        // Discrete approximation of the delay-differential equation
        for t in tau..total {
            let x_tau = x[t - tau];
            x[t] = x[t - 1] + beta * x_tau / (1.0 + x_tau.powf(n_exp)) - gamma * x[t - 1];
        }

        // Keep the post-warmup portion
        let start = warmup;
        let series: Vec<f64> = x[start..total].to_vec();

        Self {
            series,
            cursor: n_lags, // first valid position for full lag window
            n_lags,
        }
    }
}

impl StreamGenerator for MackeyGlass {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Lag-embedding: [x(t-n_lags+1), ..., x(t)]
        let start = self.cursor - self.n_lags;
        let features: Vec<f64> = self.series[start..self.cursor].to_vec();

        // Target: x(t+1)
        let target = if self.cursor < self.series.len() - 1 {
            self.series[self.cursor + 1]
        } else {
            // Wrap around if exhausted (shouldn't happen with proper max_samples)
            self.series[self.cursor]
        };

        // Advance cursor, wrap if needed
        self.cursor += 1;
        if self.cursor >= self.series.len() - 1 {
            self.cursor = self.n_lags;
        }

        (features, target)
    }

    fn n_features(&self) -> usize {
        self.n_lags
    }

    fn task_type(&self) -> TaskType {
        TaskType::Regression
    }

    fn drift_occurred(&self) -> bool {
        false // Mackey-Glass is stationary (no concept drift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mackey_glass_produces_correct_n_features() {
        let mut gen = MackeyGlass::new(42);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            6,
            "MackeyGlass default should produce 6 lag features, got {}",
            features.len()
        );
    }

    #[test]
    fn mackey_glass_custom_lags() {
        let mut gen = MackeyGlass::with_config(42, 17, 10, 100, 1000);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            10,
            "MackeyGlass with 10 lags should produce 10 features"
        );
    }

    #[test]
    fn mackey_glass_task_type_is_regression() {
        let gen = MackeyGlass::new(42);
        assert_eq!(gen.task_type(), TaskType::Regression);
    }

    #[test]
    fn mackey_glass_produces_finite_values() {
        let mut gen = MackeyGlass::new(123);
        for i in 0..5000 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(
                    f.is_finite(),
                    "feature {} at sample {} is not finite: {}",
                    j,
                    i,
                    f
                );
            }
            assert!(
                target.is_finite(),
                "target at sample {} is not finite: {}",
                i,
                target
            );
        }
    }

    #[test]
    fn mackey_glass_values_bounded() {
        // Mackey-Glass with tau=17 should stay bounded approximately in [0.2, 1.5]
        let mut gen = MackeyGlass::new(42);
        for i in 0..5000 {
            let (features, target) = gen.next_sample();
            for (j, &f) in features.iter().enumerate() {
                assert!(
                    f.abs() < 5.0,
                    "feature {} at sample {} has unexpected magnitude: {}",
                    j,
                    i,
                    f
                );
            }
            assert!(
                target.abs() < 5.0,
                "target at sample {} has unexpected magnitude: {}",
                i,
                target
            );
        }
    }

    #[test]
    fn mackey_glass_no_drift() {
        let mut gen = MackeyGlass::new(42);
        for _ in 0..500 {
            gen.next_sample();
            assert!(
                !gen.drift_occurred(),
                "MackeyGlass should never signal drift"
            );
        }
    }

    #[test]
    fn mackey_glass_deterministic_with_same_seed() {
        let mut gen1 = MackeyGlass::new(42);
        let mut gen2 = MackeyGlass::new(42);
        for _ in 0..200 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }

    #[test]
    fn mackey_glass_series_is_chaotic() {
        // With tau=17, the series should not be constant or periodic over short windows
        let mut gen = MackeyGlass::new(42);
        let mut values = Vec::new();
        for _ in 0..100 {
            let (_, target) = gen.next_sample();
            values.push(target);
        }
        // Check that values have non-trivial variance
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let var: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        assert!(
            var > 1e-6,
            "Mackey-Glass series should have non-trivial variance, got {}",
            var
        );
    }
}
