//! Needle-in-a-haystack stream generator.
//!
//! Produces a stream where one informative sample (the "needle") is embedded
//! at a fixed position within a long sequence of noise samples (the "haystack").
//! The model must retain the needle signal across many distractors.
//!
//! # Protocol
//!
//! Each epoch of `haystack_size` samples:
//! - Sample `needle_pos` contains the needle: features are distinctly separated
//!   from noise (controlled by `distinctiveness`), and the target is `needle_target`.
//! - All other samples are distractors: features are uniform noise, targets are
//!   white noise in `[-0.5, 0.5]`.
//!
//! The needle position is fixed (not randomized) to make the benchmark deterministic
//! and repeatable. The informative value is the same every epoch, so a model with
//! sufficient memory can learn to reproduce `needle_target` after the needle position.
//!
//! `drift_occurred()` returns `true` at the start of each new epoch.

use super::{Rng, StreamGenerator, TaskType};

/// Needle-in-a-haystack stream generator.
///
/// # Parameters
/// - `seed`: PRNG seed for reproducibility
/// - `n_features`: number of input features (default: 8)
/// - `haystack_size`: total samples per epoch including the needle (default: 256)
/// - `needle_pos`: index of the needle within each epoch (default: 0)
/// - `distinctiveness`: feature separation for the needle (default: 3.0)
/// - `needle_target`: regression target for the needle sample (default: 1.0)
#[derive(Debug, Clone)]
pub struct NeedleStream {
    rng: Rng,
    /// Number of input features.
    n_features: usize,
    /// Total samples per epoch (including needle).
    haystack_size: usize,
    /// Index of the needle within each epoch.
    needle_pos: usize,
    /// Feature magnitude for the needle (distractors are uniform in [0, 1]).
    distinctiveness: f64,
    /// Regression target for needle samples.
    needle_target: f64,
    /// Current index within the epoch.
    pos: usize,
    /// Drift flag for epoch transitions.
    drift_flag: bool,
}

impl NeedleStream {
    /// Default number of features.
    pub const DEFAULT_N_FEATURES: usize = 8;
    /// Default haystack size (samples per epoch).
    pub const DEFAULT_HAYSTACK_SIZE: usize = 256;
    /// Default needle position within each epoch.
    pub const DEFAULT_NEEDLE_POS: usize = 0;
    /// Default feature distinctiveness multiplier.
    pub const DEFAULT_DISTINCTIVENESS: f64 = 3.0;
    /// Default needle target value.
    pub const DEFAULT_NEEDLE_TARGET: f64 = 1.0;

    /// Create a needle stream.
    ///
    /// - `seed`: PRNG seed.
    /// - `n_features`: number of input features.
    /// - `haystack_size`: total samples per epoch (including the needle at index 0).
    pub fn new(seed: u64, n_features: usize, haystack_size: usize) -> Self {
        Self::with_config(
            seed,
            n_features,
            haystack_size,
            Self::DEFAULT_NEEDLE_POS,
            Self::DEFAULT_DISTINCTIVENESS,
            Self::DEFAULT_NEEDLE_TARGET,
        )
    }

    /// Create a needle stream with custom parameters.
    ///
    /// # Panics
    ///
    /// Panics if `haystack_size == 0`, `n_features == 0`, or
    /// `needle_pos >= haystack_size`.
    pub fn with_config(
        seed: u64,
        n_features: usize,
        haystack_size: usize,
        needle_pos: usize,
        distinctiveness: f64,
        needle_target: f64,
    ) -> Self {
        assert!(n_features > 0, "n_features must be > 0");
        assert!(haystack_size > 0, "haystack_size must be > 0");
        assert!(
            needle_pos < haystack_size,
            "needle_pos ({}) must be < haystack_size ({})",
            needle_pos,
            haystack_size
        );

        Self {
            rng: Rng::new(seed),
            n_features,
            haystack_size,
            needle_pos,
            distinctiveness,
            needle_target,
            pos: 0,
            drift_flag: false,
        }
    }

    /// Whether the current position (before the next `next_sample` call) is the needle.
    pub fn at_needle(&self) -> bool {
        self.pos == self.needle_pos
    }

    /// Haystack size (samples per epoch, including the needle).
    pub fn haystack_size(&self) -> usize {
        self.haystack_size
    }

    /// Needle position within each epoch.
    pub fn needle_pos(&self) -> usize {
        self.needle_pos
    }
}

impl StreamGenerator for NeedleStream {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Mark epoch start.
        self.drift_flag = self.pos == 0 && self.haystack_size > 0;

        let is_needle = self.pos == self.needle_pos;

        let (features, target) = if is_needle {
            // Needle: all features at `distinctiveness` (highly distinctive vs noise in [0,1]).
            let feats = vec![self.distinctiveness; self.n_features];
            (feats, self.needle_target)
        } else {
            // Distractor: uniform noise features, white-noise target.
            let feats: Vec<f64> = (0..self.n_features).map(|_| self.rng.uniform()).collect();
            let noise_target = self.rng.uniform_range(-0.5, 0.5);
            (feats, noise_target)
        };

        self.pos += 1;
        if self.pos >= self.haystack_size {
            self.pos = 0;
        }

        (features, target)
    }

    fn n_features(&self) -> usize {
        self.n_features
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
    fn needle_produces_correct_n_features() {
        let mut gen = NeedleStream::new(
            42,
            NeedleStream::DEFAULT_N_FEATURES,
            NeedleStream::DEFAULT_HAYSTACK_SIZE,
        );
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            NeedleStream::DEFAULT_N_FEATURES,
            "features should have {} dims, got {}",
            NeedleStream::DEFAULT_N_FEATURES,
            features.len()
        );
    }

    #[test]
    fn needle_task_type_is_regression() {
        let gen = NeedleStream::new(42, 8, 64);
        assert_eq!(gen.task_type(), TaskType::Regression);
    }

    #[test]
    fn needle_produces_finite_values() {
        let mut gen = NeedleStream::new(77, 8, 64);
        for i in 0..512 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(f.is_finite(), "feature {} at sample {} is not finite", j, i);
            }
            assert!(target.is_finite(), "target at sample {} is not finite", i);
        }
    }

    #[test]
    fn needle_deterministic_with_same_seed() {
        let mut gen1 = NeedleStream::new(99, 8, 64);
        let mut gen2 = NeedleStream::new(99, 8, 64);
        for _ in 0..512 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }

    #[test]
    fn needle_at_expected_position() {
        // Default: needle_pos=0, so the first sample of each epoch is the needle.
        let n = 16;
        let mut gen = NeedleStream::with_config(1, 4, n, 0, 3.0, 5.0);

        // First sample is needle.
        let (features, target) = gen.next_sample();
        assert!(
            (target - 5.0).abs() < 1e-12,
            "needle target should be 5.0, got {}",
            target
        );
        for (j, &f) in features.iter().enumerate() {
            assert!(
                (f - 3.0).abs() < 1e-12,
                "needle feature {} should be 3.0, got {}",
                j,
                f
            );
        }
    }

    #[test]
    fn needle_middle_position() {
        // Place needle in the middle of a small haystack.
        let haystack = 10;
        let needle_pos = 5;
        let mut gen = NeedleStream::with_config(2, 4, haystack, needle_pos, 4.0, 9.0);

        for i in 0..haystack {
            let (features, target) = gen.next_sample();
            if i == needle_pos {
                assert!(
                    (target - 9.0).abs() < 1e-12,
                    "needle at pos {} should have target 9.0, got {}",
                    i,
                    target
                );
                for (j, &f) in features.iter().enumerate() {
                    assert!(
                        (f - 4.0).abs() < 1e-12,
                        "needle feature {} at pos {} should be 4.0, got {}",
                        j,
                        i,
                        f
                    );
                }
            } else {
                // Distractors should have feature values in [0, 1] (not at distinctiveness).
                for &f in features.iter() {
                    assert!(
                        (0.0..=1.0).contains(&f),
                        "distractor feature should be in [0,1], got {}",
                        f
                    );
                }
            }
        }
    }

    #[test]
    fn needle_epoch_drift_flag() {
        let haystack = 8;
        let mut gen = NeedleStream::with_config(3, 4, haystack, 0, 2.0, 1.0);

        // First sample of epoch 0: drift_flag set.
        gen.next_sample();
        assert!(gen.drift_occurred(), "drift expected at start of epoch 0");

        // Samples 1..haystack-1: no drift.
        for i in 1..haystack {
            gen.next_sample();
            assert!(!gen.drift_occurred(), "no drift expected at sample {}", i);
        }

        // First sample of epoch 1: drift_flag set again.
        gen.next_sample();
        assert!(gen.drift_occurred(), "drift expected at start of epoch 1");
    }
}
