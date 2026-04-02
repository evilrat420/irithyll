//! Waveform generator (Breiman et al., 1984).
//!
//! Three-class classification with 21 features (optionally 40 with 19 noise features).
//! Each class is a mixture of two triangular wave templates plus Gaussian noise.
//! The Bayes error rate is approximately 14%.

use super::{Rng, StreamGenerator, TaskType};

/// Triangular wave template h1: peak at position 7.
fn h1(i: usize) -> f64 {
    let i = i as f64;
    if i <= 7.0 { i } else { 14.0 - i }.max(0.0)
}

/// Triangular wave template h2: peak at position 11.
fn h2(i: usize) -> f64 {
    let i = i as f64;
    if i <= 11.0 {
        (i - 4.0).max(0.0)
    } else {
        (18.0 - i).max(0.0)
    }
}

/// Triangular wave template h3: peak at position 15.
fn h3(i: usize) -> f64 {
    let i = i as f64;
    if i <= 15.0 {
        (i - 8.0).max(0.0)
    } else {
        (22.0 - i).max(0.0)
    }
}

/// Waveform stream generator for 3-class classification.
///
/// # Parameters
/// - `seed`: PRNG seed
/// - `has_noise_features`: if `true`, appends 19 `N(0,1)` noise features (40 total)
#[derive(Debug, Clone)]
pub struct Waveform {
    rng: Rng,
    has_noise: bool,
}

impl Waveform {
    /// Create a new Waveform generator with 21 features (no noise features).
    pub fn new(seed: u64) -> Self {
        Self::with_noise(seed, false)
    }

    /// Create a Waveform generator, optionally with 19 noise features.
    pub fn with_noise(seed: u64, has_noise_features: bool) -> Self {
        Self {
            rng: Rng::new(seed),
            has_noise: has_noise_features,
        }
    }
}

impl StreamGenerator for Waveform {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Choose class 0, 1, or 2
        let class = self.rng.uniform_int(3);
        let u1 = self.rng.uniform();
        let u2 = 1.0 - u1;

        let n_base = 21;
        let n_total = if self.has_noise { 40 } else { 21 };
        let mut features = Vec::with_capacity(n_total);

        // Template pair depends on class:
        //   class 0: u1*h1 + u2*h2
        //   class 1: u1*h1 + u2*h3
        //   class 2: u1*h2 + u2*h3
        for i in 1..=n_base {
            let base = match class {
                0 => u1 * h1(i) + u2 * h2(i),
                1 => u1 * h1(i) + u2 * h3(i),
                2 => u1 * h2(i) + u2 * h3(i),
                _ => unreachable!(),
            };
            features.push(base + self.rng.normal(0.0, 1.0));
        }

        // Optional noise features
        if self.has_noise {
            for _ in 0..19 {
                features.push(self.rng.normal(0.0, 1.0));
            }
        }

        (features, class as f64)
    }

    fn n_features(&self) -> usize {
        if self.has_noise {
            40
        } else {
            21
        }
    }

    fn task_type(&self) -> TaskType {
        TaskType::MulticlassClassification { n_classes: 3 }
    }

    fn drift_occurred(&self) -> bool {
        false // Waveform has no concept drift
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waveform_produces_correct_n_features_default() {
        let mut gen = Waveform::new(42);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            21,
            "Waveform (no noise) should produce 21 features, got {}",
            features.len()
        );
    }

    #[test]
    fn waveform_produces_correct_n_features_with_noise() {
        let mut gen = Waveform::with_noise(42, true);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            40,
            "Waveform (with noise) should produce 40 features, got {}",
            features.len()
        );
    }

    #[test]
    fn waveform_task_type_is_multiclass_3() {
        let gen = Waveform::new(42);
        assert_eq!(
            gen.task_type(),
            TaskType::MulticlassClassification { n_classes: 3 },
            "Waveform task type should be 3-class multiclass"
        );
    }

    #[test]
    fn waveform_labels_in_valid_range() {
        let mut gen = Waveform::new(42);
        for _ in 0..1000 {
            let (_, target) = gen.next_sample();
            let c = target as usize;
            assert!(c < 3, "Waveform class should be 0, 1, or 2, got {}", target);
            assert_eq!(target, c as f64, "Waveform class should be an integer");
        }
    }

    #[test]
    fn waveform_produces_finite_values() {
        let mut gen = Waveform::with_noise(123, true);
        for i in 0..1000 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(f.is_finite(), "feature {} at sample {} is not finite", j, i);
            }
            assert!(target.is_finite(), "target at sample {} is not finite", i);
        }
    }

    #[test]
    fn waveform_no_drift() {
        let mut gen = Waveform::new(42);
        for _ in 0..500 {
            gen.next_sample();
            assert!(!gen.drift_occurred(), "Waveform should never signal drift");
        }
    }

    #[test]
    fn waveform_all_classes_appear() {
        let mut gen = Waveform::new(42);
        let mut seen = [false; 3];
        for _ in 0..300 {
            let (_, target) = gen.next_sample();
            seen[target as usize] = true;
        }
        for (c, &s) in seen.iter().enumerate() {
            assert!(s, "class {} was never generated in 300 samples", c);
        }
    }

    #[test]
    fn waveform_deterministic_with_same_seed() {
        let mut gen1 = Waveform::new(42);
        let mut gen2 = Waveform::new(42);
        for _ in 0..200 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }

    #[test]
    fn waveform_templates_have_correct_peaks() {
        // h1 peaks at 7, h2 at 11, h3 at 15
        assert_eq!(h1(7), 7.0, "h1 should peak at position 7");
        assert_eq!(h2(11), 7.0, "h2 should peak at position 11");
        assert_eq!(h3(15), 7.0, "h3 should peak at position 15");
        assert_eq!(h1(0), 0.0, "h1(0) should be 0");
        assert_eq!(h1(14), 0.0, "h1(14) should be 0");
    }
}
