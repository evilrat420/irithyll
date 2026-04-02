//! LED Generator (Breiman et al., 1984).
//!
//! Simulates a 7-segment LED display for digits 0-9 with 17 irrelevant noise
//! features (24 total). Each segment bit is flipped with a configurable noise
//! probability. Concept drift permutes the segment indices.

use super::{Rng, StreamGenerator, TaskType};

/// 7-segment encodings for digits 0-9.
/// Each row is `[a, b, c, d, e, f, g]` where 1 = segment on.
///
/// ```text
///  _a_
/// |f  |b
///  _g_
/// |e  |c
///  _d_
/// ```
const LED_SEGMENTS: [[u8; 7]; 10] = [
    [1, 1, 1, 1, 1, 1, 0], // 0
    [0, 1, 1, 0, 0, 0, 0], // 1
    [1, 1, 0, 1, 1, 0, 1], // 2
    [1, 1, 1, 1, 0, 0, 1], // 3
    [0, 1, 1, 0, 0, 1, 1], // 4
    [1, 0, 1, 1, 0, 1, 1], // 5
    [1, 0, 1, 1, 1, 1, 1], // 6
    [1, 1, 1, 0, 0, 0, 0], // 7
    [1, 1, 1, 1, 1, 1, 1], // 8
    [1, 1, 1, 1, 0, 1, 1], // 9
];

/// LED stream generator for 10-class digit classification.
///
/// # Parameters
/// - `seed`: PRNG seed
/// - `noise_rate`: probability of flipping each segment bit (default 0.1)
/// - `change_points`: sample indices at which segment permutation changes
#[derive(Debug, Clone)]
pub struct LED {
    rng: Rng,
    noise_rate: f64,
    /// Permutation mapping: segment index -> actual feature position
    permutation: [usize; 7],
    change_points: Vec<usize>,
    sample_idx: usize,
    next_change: usize,
    drift_flag: bool,
}

impl LED {
    /// Create a new LED generator with default noise rate 0.1 and no drift.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, 0.1, Vec::new())
    }

    /// Create an LED generator with custom noise and change points.
    ///
    /// At each change point, the segment-to-feature mapping is randomly permuted.
    pub fn with_config(seed: u64, noise_rate: f64, change_points: Vec<usize>) -> Self {
        Self {
            rng: Rng::new(seed),
            noise_rate,
            permutation: [0, 1, 2, 3, 4, 5, 6], // identity
            change_points,
            sample_idx: 0,
            next_change: 0,
            drift_flag: false,
        }
    }

    /// Fisher-Yates shuffle of the segment permutation.
    fn shuffle_permutation(&mut self) {
        for i in (1..7).rev() {
            let j = self.rng.uniform_int(i + 1);
            self.permutation.swap(i, j);
        }
    }
}

impl StreamGenerator for LED {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Check for drift
        self.drift_flag = false;
        if self.next_change < self.change_points.len()
            && self.sample_idx == self.change_points[self.next_change]
        {
            self.shuffle_permutation();
            self.next_change += 1;
            self.drift_flag = true;
        }

        // Choose a random digit
        let digit = self.rng.uniform_int(10);
        let segments = &LED_SEGMENTS[digit];

        // Build the 24-feature vector: 7 segment bits (permuted) + 17 noise
        let mut features = vec![0.0; 24];

        for (seg_idx, &seg_val) in segments.iter().enumerate() {
            let mut val = seg_val as f64;
            // Flip with noise probability
            if self.rng.bernoulli(self.noise_rate) {
                val = 1.0 - val;
            }
            features[self.permutation[seg_idx]] = val;
        }

        // Noise features (indices 7..24): random 0/1
        for feat in features.iter_mut().skip(7) {
            *feat = if self.rng.bernoulli(0.5) { 1.0 } else { 0.0 };
        }

        self.sample_idx += 1;
        (features, digit as f64)
    }

    fn n_features(&self) -> usize {
        24
    }

    fn task_type(&self) -> TaskType {
        TaskType::MulticlassClassification { n_classes: 10 }
    }

    fn drift_occurred(&self) -> bool {
        self.drift_flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn led_produces_correct_n_features() {
        let mut gen = LED::new(42);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            24,
            "LED should produce 24 features, got {}",
            features.len()
        );
    }

    #[test]
    fn led_task_type_is_multiclass_10() {
        let gen = LED::new(42);
        assert_eq!(
            gen.task_type(),
            TaskType::MulticlassClassification { n_classes: 10 },
            "LED task type should be 10-class multiclass"
        );
    }

    #[test]
    fn led_labels_in_valid_range() {
        let mut gen = LED::new(42);
        for _ in 0..1000 {
            let (_, target) = gen.next_sample();
            let digit = target as usize;
            assert!(digit < 10, "LED digit should be in 0..9, got {}", target);
            assert_eq!(
                target, digit as f64,
                "LED digit should be an integer, got {}",
                target
            );
        }
    }

    #[test]
    fn led_features_are_binary() {
        let mut gen = LED::new(42);
        for _ in 0..500 {
            let (features, _) = gen.next_sample();
            for (j, &f) in features.iter().enumerate() {
                assert!(
                    f == 0.0 || f == 1.0,
                    "LED feature {} should be 0 or 1, got {}",
                    j,
                    f
                );
            }
        }
    }

    #[test]
    fn led_produces_finite_values() {
        let mut gen = LED::new(123);
        for i in 0..500 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(f.is_finite(), "feature {} at sample {} is not finite", j, i);
            }
            assert!(target.is_finite(), "target at sample {} is not finite", i);
        }
    }

    #[test]
    fn led_drift_at_change_points() {
        let mut gen = LED::with_config(42, 0.1, vec![50, 100]);

        for _ in 0..50 {
            gen.next_sample();
        }
        gen.next_sample();
        assert!(
            gen.drift_occurred(),
            "drift should occur at change point 50"
        );

        for _ in 52..100 {
            gen.next_sample();
            assert!(
                !gen.drift_occurred(),
                "no drift expected between change points"
            );
        }
    }

    #[test]
    fn led_deterministic_with_same_seed() {
        let mut gen1 = LED::new(42);
        let mut gen2 = LED::new(42);
        for _ in 0..200 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }

    #[test]
    fn led_all_digits_appear() {
        let mut gen = LED::new(42);
        let mut seen = [false; 10];
        for _ in 0..1000 {
            let (_, target) = gen.next_sample();
            seen[target as usize] = true;
        }
        for (d, &s) in seen.iter().enumerate() {
            assert!(s, "digit {} was never generated in 1000 samples", d);
        }
    }
}
