//! Rotating Hyperplane generator (Hulten et al., KDD 2001).
//!
//! Features are uniform in `[0, 1]`. The label is `1` if the weighted sum of
//! features exceeds a threshold, else `0`. Gradual drift is introduced by
//! continuously rotating a subset of the weight vector.

use super::{Rng, StreamGenerator, TaskType};

/// Rotating Hyperplane stream generator with gradual concept drift.
///
/// # Parameters
/// - `seed`: PRNG seed
/// - `n_features`: dimensionality (default 10)
/// - `n_drifting`: number of features whose weights rotate (default 3)
/// - `magnitude`: rotation step size per sample (default 0.01)
/// - `noise`: standard deviation of Gaussian noise added to the decision (default 0.05)
#[derive(Debug, Clone)]
pub struct Hyperplane {
    rng: Rng,
    n_feat: usize,
    n_drifting: usize,
    magnitude: f64,
    noise_std: f64,
    weights: Vec<f64>,
    sample_idx: usize,
    drift_flag: bool,
}

impl Hyperplane {
    /// Create a new Hyperplane generator.
    ///
    /// - `n_features`: number of input features
    /// - `n_drifting`: how many features have drifting weights (clamped to `n_features`)
    /// - `magnitude`: drift step per sample
    /// - `noise`: label noise standard deviation
    pub fn new(
        seed: u64,
        n_features: usize,
        n_drifting: usize,
        magnitude: f64,
        noise: f64,
    ) -> Self {
        let mut rng = Rng::new(seed);
        let n_drifting = n_drifting.min(n_features);
        let weights: Vec<f64> = (0..n_features)
            .map(|i| (i as f64 + 1.0) / n_features as f64 + rng.uniform_range(-0.1, 0.1))
            .collect();
        Self {
            rng,
            n_feat: n_features,
            n_drifting,
            magnitude,
            noise_std: noise,
            weights,
            sample_idx: 0,
            drift_flag: false,
        }
    }
}

impl StreamGenerator for Hyperplane {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Rotate drifting weights
        self.drift_flag = false;
        if self.magnitude > 0.0 && self.n_drifting > 0 {
            for i in 0..self.n_drifting {
                let direction = if i % 2 == 0 { 1.0 } else { -1.0 };
                self.weights[i] += direction * self.magnitude;
                // Keep weights positive and bounded
                self.weights[i] = self.weights[i].clamp(0.01, 10.0);
            }
            // Signal drift every 100 samples (gradual)
            if self.sample_idx > 0 && self.sample_idx % 100 == 0 {
                self.drift_flag = true;
            }
        }

        let x: Vec<f64> = (0..self.n_feat).map(|_| self.rng.uniform()).collect();

        let dot: f64 = x.iter().zip(self.weights.iter()).map(|(a, b)| a * b).sum();
        let threshold: f64 = self.weights.iter().sum::<f64>() * 0.5;
        let noise = self.rng.normal(0.0, self.noise_std);

        let label = if dot + noise > threshold { 1.0 } else { 0.0 };

        self.sample_idx += 1;
        (x, label)
    }

    fn n_features(&self) -> usize {
        self.n_feat
    }

    fn task_type(&self) -> TaskType {
        TaskType::BinaryClassification
    }

    fn drift_occurred(&self) -> bool {
        self.drift_flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyperplane_produces_correct_n_features() {
        let mut gen = Hyperplane::new(42, 10, 3, 0.01, 0.05);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            10,
            "Hyperplane should produce 10 features, got {}",
            features.len()
        );
    }

    #[test]
    fn hyperplane_custom_dimensionality() {
        let mut gen = Hyperplane::new(42, 25, 5, 0.01, 0.0);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            25,
            "Hyperplane with d=25 should produce 25 features"
        );
    }

    #[test]
    fn hyperplane_task_type_is_binary() {
        let gen = Hyperplane::new(42, 10, 3, 0.01, 0.05);
        assert_eq!(gen.task_type(), TaskType::BinaryClassification);
    }

    #[test]
    fn hyperplane_produces_finite_values() {
        let mut gen = Hyperplane::new(123, 10, 3, 0.01, 0.05);
        for i in 0..2000 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(f.is_finite(), "feature {} at sample {} is not finite", j, i);
            }
            assert!(target.is_finite(), "target at sample {} is not finite", i);
        }
    }

    #[test]
    fn hyperplane_features_in_unit_range() {
        let mut gen = Hyperplane::new(42, 10, 3, 0.01, 0.0);
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
    fn hyperplane_gradual_drift_signals() {
        let mut gen = Hyperplane::new(42, 10, 3, 0.01, 0.05);
        let mut drift_count = 0;
        for _ in 0..1000 {
            gen.next_sample();
            if gen.drift_occurred() {
                drift_count += 1;
            }
        }
        // With gradual drift every 100 samples, expect ~9 drift signals in 1000 samples
        assert!(
            drift_count >= 5,
            "expected multiple gradual drift signals, got {}",
            drift_count
        );
    }

    #[test]
    fn hyperplane_no_drift_when_magnitude_zero() {
        let mut gen = Hyperplane::new(42, 10, 3, 0.0, 0.05);
        for _ in 0..500 {
            gen.next_sample();
            assert!(
                !gen.drift_occurred(),
                "no drift should occur when magnitude is 0"
            );
        }
    }

    #[test]
    fn hyperplane_deterministic_with_same_seed() {
        let mut gen1 = Hyperplane::new(42, 10, 3, 0.01, 0.05);
        let mut gen2 = Hyperplane::new(42, 10, 3, 0.01, 0.05);
        for _ in 0..200 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }
}
