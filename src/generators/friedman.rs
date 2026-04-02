//! Friedman #1 with Drift (Friedman, 1991).
//!
//! Ten features uniform in `[0, 1]`, only the first 5 are relevant.
//! The target function is:
//!
//! ```text
//! y = c0*sin(pi*x1*x2) + c1*(x3 - 0.5)^2 + c2*x4 + c3*x5 + noise
//! ```
//!
//! Default coefficients `[10, 20, 10, 5]` rotate slowly over time to introduce
//! gradual concept drift.

use super::{Rng, StreamGenerator, TaskType};

/// Friedman #1 regression generator with optional gradual drift.
///
/// # Parameters
/// - `seed`: PRNG seed
/// - `noise_std`: standard deviation of Gaussian noise (default 1.0)
/// - `drift_magnitude`: per-sample coefficient rotation magnitude (default 0.0)
#[derive(Debug, Clone)]
pub struct Friedman {
    rng: Rng,
    coefficients: [f64; 4],
    noise_std: f64,
    drift_magnitude: f64,
    sample_idx: usize,
    drift_flag: bool,
}

impl Friedman {
    /// Default Friedman #1 coefficients.
    pub const DEFAULT_COEFFICIENTS: [f64; 4] = [10.0, 20.0, 10.0, 5.0];

    /// Create a new Friedman #1 generator with default settings (no drift).
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, 1.0, 0.0)
    }

    /// Create with custom noise and drift magnitude.
    ///
    /// `drift_magnitude` controls how fast the coefficients rotate. A value of
    /// `0.001` produces noticeable drift over tens of thousands of samples.
    pub fn with_config(seed: u64, noise_std: f64, drift_magnitude: f64) -> Self {
        Self {
            rng: Rng::new(seed),
            coefficients: Self::DEFAULT_COEFFICIENTS,
            noise_std,
            drift_magnitude,
            sample_idx: 0,
            drift_flag: false,
        }
    }
}

impl StreamGenerator for Friedman {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Gradual drift: rotate coefficients
        self.drift_flag = false;
        if self.drift_magnitude > 0.0 {
            let t = self.sample_idx as f64;
            // Slowly shift each coefficient using sinusoidal modulation
            self.coefficients[0] =
                Self::DEFAULT_COEFFICIENTS[0] + self.drift_magnitude * (0.001 * t).sin() * 5.0;
            self.coefficients[1] =
                Self::DEFAULT_COEFFICIENTS[1] + self.drift_magnitude * (0.0007 * t).cos() * 10.0;
            self.coefficients[2] =
                Self::DEFAULT_COEFFICIENTS[2] + self.drift_magnitude * (0.0013 * t).sin() * 5.0;
            self.coefficients[3] =
                Self::DEFAULT_COEFFICIENTS[3] + self.drift_magnitude * (0.0009 * t).cos() * 2.5;

            if self.sample_idx > 0 && self.sample_idx % 500 == 0 {
                self.drift_flag = true;
            }
        }

        // 10 features uniform [0, 1], only first 5 are relevant
        let x: Vec<f64> = (0..10).map(|_| self.rng.uniform()).collect();

        let c = &self.coefficients;
        let y = c[0] * (std::f64::consts::PI * x[0] * x[1]).sin()
            + c[1] * (x[2] - 0.5).powi(2)
            + c[2] * x[3]
            + c[3] * x[4]
            + self.rng.normal(0.0, self.noise_std);

        self.sample_idx += 1;
        (x, y)
    }

    fn n_features(&self) -> usize {
        10
    }

    fn task_type(&self) -> TaskType {
        TaskType::Regression
    }

    fn drift_occurred(&self) -> bool {
        self.drift_flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn friedman_produces_correct_n_features() {
        let mut gen = Friedman::new(42);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            10,
            "Friedman should produce 10 features, got {}",
            features.len()
        );
    }

    #[test]
    fn friedman_task_type_is_regression() {
        let gen = Friedman::new(42);
        assert_eq!(
            gen.task_type(),
            TaskType::Regression,
            "Friedman task type should be Regression"
        );
    }

    #[test]
    fn friedman_produces_finite_values() {
        let mut gen = Friedman::with_config(123, 1.0, 0.001);
        for i in 0..2000 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(f.is_finite(), "feature {} at sample {} is not finite", j, i);
            }
            assert!(target.is_finite(), "target at sample {} is not finite", i);
        }
    }

    #[test]
    fn friedman_features_in_unit_range() {
        let mut gen = Friedman::new(42);
        for _ in 0..500 {
            let (features, _) = gen.next_sample();
            for (j, &f) in features.iter().enumerate() {
                assert!(
                    (0.0..1.0).contains(&f),
                    "feature {} should be in [0, 1), got {}",
                    j,
                    f
                );
            }
        }
    }

    #[test]
    fn friedman_no_drift_by_default() {
        let mut gen = Friedman::new(42);
        for _ in 0..1000 {
            gen.next_sample();
            assert!(
                !gen.drift_occurred(),
                "no drift should occur with default settings"
            );
        }
    }

    #[test]
    fn friedman_drift_when_configured() {
        let mut gen = Friedman::with_config(42, 1.0, 1.0);
        let mut drift_count = 0;
        for _ in 0..5000 {
            gen.next_sample();
            if gen.drift_occurred() {
                drift_count += 1;
            }
        }
        assert!(
            drift_count >= 5,
            "expected drift signals with magnitude > 0, got {}",
            drift_count
        );
    }

    #[test]
    fn friedman_deterministic_with_same_seed() {
        let mut gen1 = Friedman::new(42);
        let mut gen2 = Friedman::new(42);
        for _ in 0..200 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }

    #[test]
    fn friedman_target_depends_on_first_five_features() {
        // With zero noise and no drift, the target should be purely deterministic
        // from the first 5 features. Run two generators with the same first samples.
        let mut gen = Friedman::with_config(42, 0.0, 0.0);
        let (x, y) = gen.next_sample();

        // Manually compute expected target
        let c = Friedman::DEFAULT_COEFFICIENTS;
        let expected = c[0] * (std::f64::consts::PI * x[0] * x[1]).sin()
            + c[1] * (x[2] - 0.5).powi(2)
            + c[2] * x[3]
            + c[3] * x[4];

        assert!(
            (y - expected).abs() < 1e-10,
            "target should match manual computation: expected {}, got {}",
            expected,
            y
        );
    }
}
