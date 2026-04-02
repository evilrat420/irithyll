//! Random RBF with Drift generator (Bifet et al., MOA 2010).
//!
//! Generates samples from Gaussian clusters (centroids) in a configurable
//! feature space. Each centroid is assigned a class and a weight. Samples are
//! drawn from randomly selected centroids with Gaussian spread. Gradual drift
//! is introduced by moving centroids over time.

use super::{Rng, StreamGenerator, TaskType};

/// A single centroid in the RBF generator.
#[derive(Debug, Clone)]
struct Centroid {
    center: Vec<f64>,
    class: usize,
    weight: f64,
    /// Velocity vector for drift.
    velocity: Vec<f64>,
}

/// Random RBF stream generator with drifting centroids.
///
/// # Parameters
/// - `seed`: PRNG seed
/// - `n_centroids`: number of Gaussian clusters (default 50)
/// - `n_features`: dimensionality (default 10)
/// - `n_classes`: number of distinct classes (default 2)
/// - `drift_speed`: magnitude of centroid movement per sample (default 0.0001)
#[derive(Debug, Clone)]
pub struct RandomRBF {
    rng: Rng,
    centroids: Vec<Centroid>,
    n_feat: usize,
    n_cls: usize,
    drift_speed: f64,
    /// Cumulative weight for weighted sampling.
    cumulative_weights: Vec<f64>,
    sample_idx: usize,
    drift_flag: bool,
}

impl RandomRBF {
    /// Create a new Random RBF generator with default settings.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, 50, 10, 2, 0.0001)
    }

    /// Create with custom parameters.
    pub fn with_config(
        seed: u64,
        n_centroids: usize,
        n_features: usize,
        n_classes: usize,
        drift_speed: f64,
    ) -> Self {
        let mut rng = Rng::new(seed);

        let mut centroids = Vec::with_capacity(n_centroids);
        for _ in 0..n_centroids {
            let center: Vec<f64> = (0..n_features).map(|_| rng.uniform()).collect();
            let class = rng.uniform_int(n_classes);
            let weight = rng.uniform_range(0.1, 1.0);
            let velocity: Vec<f64> = (0..n_features).map(|_| rng.normal(0.0, 1.0)).collect();
            // Normalize velocity to unit length
            let norm: f64 = velocity
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                .max(1e-10);
            let velocity: Vec<f64> = velocity.iter().map(|v| v / norm).collect();
            centroids.push(Centroid {
                center,
                class,
                weight,
                velocity,
            });
        }

        let cumulative_weights = Self::compute_cumulative(&centroids);

        Self {
            rng,
            centroids,
            n_feat: n_features,
            n_cls: n_classes,
            drift_speed,
            cumulative_weights,
            sample_idx: 0,
            drift_flag: false,
        }
    }

    fn compute_cumulative(centroids: &[Centroid]) -> Vec<f64> {
        let mut cum = Vec::with_capacity(centroids.len());
        let mut total = 0.0;
        for c in centroids {
            total += c.weight;
            cum.push(total);
        }
        cum
    }

    /// Select a centroid by weighted sampling.
    fn select_centroid(&mut self) -> usize {
        let total = *self.cumulative_weights.last().unwrap_or(&1.0);
        let r = self.rng.uniform_range(0.0, total);
        self.cumulative_weights
            .iter()
            .position(|&cw| cw >= r)
            .unwrap_or(self.centroids.len() - 1)
    }
}

impl StreamGenerator for RandomRBF {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Move centroids (gradual drift)
        self.drift_flag = false;
        if self.drift_speed > 0.0 {
            for c in &mut self.centroids {
                for (pos, vel) in c.center.iter_mut().zip(c.velocity.iter()) {
                    *pos += vel * self.drift_speed;
                    // Clamp to [0, 1] to keep centroids in feature space
                    *pos = pos.clamp(0.0, 1.0);
                }
            }
            if self.sample_idx > 0 && self.sample_idx % 100 == 0 {
                self.drift_flag = true;
            }
        }

        // Sample from a randomly selected centroid
        let ci = self.select_centroid();
        let centroid = &self.centroids[ci];
        let class = centroid.class;

        // Generate features: center + Gaussian noise with std = 0.1
        let spread = 0.1;
        let features: Vec<f64> = centroid
            .center
            .iter()
            .map(|&c| c + self.rng.normal(0.0, spread))
            .collect();

        self.sample_idx += 1;
        (features, class as f64)
    }

    fn n_features(&self) -> usize {
        self.n_feat
    }

    fn task_type(&self) -> TaskType {
        if self.n_cls == 2 {
            TaskType::BinaryClassification
        } else {
            TaskType::MulticlassClassification {
                n_classes: self.n_cls,
            }
        }
    }

    fn drift_occurred(&self) -> bool {
        self.drift_flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_produces_correct_n_features() {
        let mut gen = RandomRBF::new(42);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            10,
            "RandomRBF default should produce 10 features, got {}",
            features.len()
        );
    }

    #[test]
    fn rbf_custom_dimensions() {
        let mut gen = RandomRBF::with_config(42, 20, 5, 3, 0.001);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            5,
            "RandomRBF with d=5 should produce 5 features"
        );
    }

    #[test]
    fn rbf_task_type_binary_default() {
        let gen = RandomRBF::new(42);
        assert_eq!(
            gen.task_type(),
            TaskType::BinaryClassification,
            "default RandomRBF should be binary classification"
        );
    }

    #[test]
    fn rbf_task_type_multiclass() {
        let gen = RandomRBF::with_config(42, 50, 10, 5, 0.0);
        assert_eq!(
            gen.task_type(),
            TaskType::MulticlassClassification { n_classes: 5 },
            "RandomRBF with 5 classes should be multiclass"
        );
    }

    #[test]
    fn rbf_produces_finite_values() {
        let mut gen = RandomRBF::new(123);
        for i in 0..1000 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(f.is_finite(), "feature {} at sample {} is not finite", j, i);
            }
            assert!(target.is_finite(), "target at sample {} is not finite", i);
        }
    }

    #[test]
    fn rbf_labels_in_valid_range() {
        let n_classes = 4;
        let mut gen = RandomRBF::with_config(42, 30, 10, n_classes, 0.0);
        for _ in 0..500 {
            let (_, target) = gen.next_sample();
            let c = target as usize;
            assert!(
                c < n_classes,
                "label should be in 0..{}, got {}",
                n_classes,
                target
            );
        }
    }

    #[test]
    fn rbf_gradual_drift_signals() {
        let mut gen = RandomRBF::with_config(42, 50, 10, 2, 0.001);
        let mut drift_count = 0;
        for _ in 0..1000 {
            gen.next_sample();
            if gen.drift_occurred() {
                drift_count += 1;
            }
        }
        assert!(
            drift_count >= 5,
            "expected multiple gradual drift signals, got {}",
            drift_count
        );
    }

    #[test]
    fn rbf_no_drift_when_speed_zero() {
        let mut gen = RandomRBF::with_config(42, 50, 10, 2, 0.0);
        for _ in 0..500 {
            gen.next_sample();
            assert!(!gen.drift_occurred(), "no drift when speed is 0");
        }
    }

    #[test]
    fn rbf_deterministic_with_same_seed() {
        let mut gen1 = RandomRBF::new(42);
        let mut gen2 = RandomRBF::new(42);
        for _ in 0..200 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }
}
