//! SEA Concepts generator (Street & Kim, KDD 2001).
//!
//! Three features uniform in `[0, 10]`, where feature 3 is irrelevant noise.
//! The label is `1` if `f1 + f2 <= threshold`, else `0`. Four concepts with
//! thresholds `{8, 9, 7, 9.5}` produce abrupt drift at configurable change points.

use super::{Rng, StreamGenerator, TaskType};

/// SEA Concepts stream generator.
///
/// # Parameters
/// - `seed`: PRNG seed for reproducibility
/// - `change_points`: sample indices at which the concept switches (default: `[12500, 25000, 37500]`)
/// - `noise_rate`: probability of flipping the label (default: `0.1`)
#[derive(Debug, Clone)]
pub struct SEA {
    rng: Rng,
    thresholds: [f64; 4],
    change_points: Vec<usize>,
    noise_rate: f64,
    sample_idx: usize,
    current_concept: usize,
    drift_flag: bool,
}

impl SEA {
    /// Default thresholds for the 4 SEA concepts.
    pub const DEFAULT_THRESHOLDS: [f64; 4] = [8.0, 9.0, 7.0, 9.5];

    /// Create a new SEA generator with default settings.
    ///
    /// Change points at `[12500, 25000, 37500]`, noise rate `0.1`.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, vec![12_500, 25_000, 37_500], 0.1)
    }

    /// Create a SEA generator with custom change points and noise rate.
    pub fn with_config(seed: u64, change_points: Vec<usize>, noise_rate: f64) -> Self {
        Self {
            rng: Rng::new(seed),
            thresholds: Self::DEFAULT_THRESHOLDS,
            change_points,
            noise_rate,
            sample_idx: 0,
            current_concept: 0,
            drift_flag: false,
        }
    }
}

impl StreamGenerator for SEA {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Check for drift
        self.drift_flag = false;
        if self.current_concept < self.change_points.len()
            && self.sample_idx == self.change_points[self.current_concept]
        {
            self.current_concept += 1;
            self.drift_flag = true;
        }

        let concept_idx = self.current_concept.min(self.thresholds.len() - 1);
        let threshold = self.thresholds[concept_idx];

        let f1 = self.rng.uniform_range(0.0, 10.0);
        let f2 = self.rng.uniform_range(0.0, 10.0);
        let f3 = self.rng.uniform_range(0.0, 10.0); // irrelevant

        let mut label = if f1 + f2 <= threshold { 1.0 } else { 0.0 };

        // Noise: flip label with probability noise_rate
        if self.rng.bernoulli(self.noise_rate) {
            label = 1.0 - label;
        }

        self.sample_idx += 1;
        (vec![f1, f2, f3], label)
    }

    fn n_features(&self) -> usize {
        3
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
    fn sea_produces_correct_n_features() {
        let mut gen = SEA::new(42);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            3,
            "SEA should produce exactly 3 features, got {}",
            features.len()
        );
    }

    #[test]
    fn sea_task_type_is_binary_classification() {
        let gen = SEA::new(42);
        assert_eq!(
            gen.task_type(),
            TaskType::BinaryClassification,
            "SEA task type should be BinaryClassification"
        );
    }

    #[test]
    fn sea_produces_finite_values() {
        let mut gen = SEA::new(123);
        for i in 0..1000 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(f.is_finite(), "feature {} at sample {} is not finite", j, i);
            }
            assert!(target.is_finite(), "target at sample {} is not finite", i);
        }
    }

    #[test]
    fn sea_labels_are_binary() {
        let mut gen = SEA::new(99);
        for _ in 0..500 {
            let (_, target) = gen.next_sample();
            assert!(
                target == 0.0 || target == 1.0,
                "SEA label should be 0.0 or 1.0, got {}",
                target
            );
        }
    }

    #[test]
    fn sea_drift_at_change_points() {
        let mut gen = SEA::with_config(42, vec![100, 200], 0.0);

        // No drift before change point
        for i in 0..100 {
            gen.next_sample();
            if i < 99 {
                assert!(
                    !gen.drift_occurred(),
                    "no drift expected before change point at sample {}",
                    i
                );
            }
        }
        // Drift at sample 100
        gen.next_sample();
        assert!(
            gen.drift_occurred(),
            "drift should occur at the first change point (sample 100)"
        );

        // No drift between change points
        for _ in 102..200 {
            gen.next_sample();
            assert!(
                !gen.drift_occurred(),
                "no drift expected between change points"
            );
        }
    }

    #[test]
    fn sea_deterministic_with_same_seed() {
        let mut gen1 = SEA::new(42);
        let mut gen2 = SEA::new(42);
        for _ in 0..200 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }

    #[test]
    fn sea_features_in_range() {
        let mut gen = SEA::new(77);
        for _ in 0..500 {
            let (features, _) = gen.next_sample();
            for (j, &f) in features.iter().enumerate() {
                assert!(
                    (0.0..=10.0).contains(&f),
                    "SEA feature {} should be in [0,10], got {}",
                    j,
                    f
                );
            }
        }
    }
}
