//! Progressive validation with configurable holdout strategies.
//!
//! Unlike prequential evaluation (which trains on every sample), progressive
//! validation can hold out some samples for pure evaluation -- they are never
//! trained on. This provides a more conservative estimate of generalisation
//! performance at the cost of reduced training data.
//!
//! # Holdout Strategies
//!
//! - [`HoldoutStrategy::None`] -- train on everything (equivalent to prequential).
//! - [`HoldoutStrategy::Periodic`] -- every N-th sample is test-only.
//! - [`HoldoutStrategy::Random`] -- each sample has probability *p* of being
//!   held out, using a deterministic xorshift64 PRNG for reproducibility.

use crate::learner::StreamingLearner;
use crate::metrics::regression::RegressionMetrics;
use crate::sample::Observation;

// ---------------------------------------------------------------------------
// Holdout Strategy
// ---------------------------------------------------------------------------

/// Strategy for deciding which samples are held out (test-only, never trained on).
///
/// All strategies evaluate every sample; the distinction is whether the model
/// is trained on it afterward.
#[derive(Debug, Clone)]
pub enum HoldoutStrategy {
    /// Every sample is used for both test and train (equivalent to prequential).
    None,
    /// Every N-th sample is test-only (not trained on).
    ///
    /// The remaining `(N-1)/N` fraction of samples are used for training.
    Periodic {
        /// Holdout period. Every `period`-th sample is test-only.
        period: usize,
    },
    /// Each sample has probability `holdout_fraction` of being test-only.
    ///
    /// Uses a deterministic xorshift64 PRNG seeded with `seed` for
    /// reproducibility. The same seed always produces the same holdout pattern.
    Random {
        /// Probability that a sample is held out (not trained on).
        holdout_fraction: f64,
        /// Seed for the deterministic PRNG.
        seed: u64,
    },
}

// ---------------------------------------------------------------------------
// xorshift64 PRNG
// ---------------------------------------------------------------------------

/// Deterministic xorshift64 PRNG.
///
/// Fast, minimal state, reproducible. Same algorithm used in irithyll's
/// BaggedSGBT for Poisson sampling.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Progressive Validator
// ---------------------------------------------------------------------------

/// Progressive validator with configurable holdout.
///
/// Every sample is always evaluated (predict + record metrics). The holdout
/// strategy controls which samples the model trains on afterward. This lets
/// you measure model performance on data it will never learn from.
///
/// # Semantics
///
/// For each sample:
/// 1. **Predict** -- query the model.
/// 2. **Record** -- update regression metrics with (target, prediction).
/// 3. **Holdout check** -- decide whether to train based on the strategy.
/// 4. **Train** (conditionally) -- feed the sample to the model if not held out.
///
/// # Example
///
/// ```
/// use irithyll::evaluation::{ProgressiveValidator, HoldoutStrategy};
/// use irithyll::learner::{SGBTLearner, StreamingLearner};
/// use irithyll::SGBTConfig;
///
/// let config = SGBTConfig::builder()
///     .n_steps(5)
///     .learning_rate(0.1)
///     .grace_period(10)
///     .build()
///     .unwrap();
/// let mut model = SGBTLearner::from_config(config);
///
/// let holdout = HoldoutStrategy::Periodic { period: 5 };
/// let mut validator = ProgressiveValidator::new(holdout);
///
/// for i in 0..200 {
///     let x = [i as f64, (i as f64).sin()];
///     let y = i as f64 * 0.5 + 1.0;
///     validator.step(&mut model, &x, y);
/// }
///
/// // ~80% of samples were trained on, ~20% were holdout-only
/// assert!(validator.holdout_fraction_actual() > 0.15);
/// assert!(validator.holdout_fraction_actual() < 0.25);
/// ```
#[derive(Debug, Clone)]
pub struct ProgressiveValidator {
    holdout: HoldoutStrategy,
    regression: RegressionMetrics,
    samples_seen: u64,
    samples_evaluated: u64,
    samples_trained: u64,
    rng_state: u64,
}

impl ProgressiveValidator {
    /// Create a new progressive validator with the given holdout strategy.
    ///
    /// # Panics
    ///
    /// - Panics if `HoldoutStrategy::Periodic { period }` has `period == 0`.
    /// - Panics if `HoldoutStrategy::Random { holdout_fraction, .. }` has
    ///   `holdout_fraction` outside `[0.0, 1.0]`.
    /// - Panics if `HoldoutStrategy::Random { seed, .. }` has `seed == 0`
    ///   (xorshift64 requires a non-zero seed).
    pub fn new(holdout: HoldoutStrategy) -> Self {
        let rng_state = match &holdout {
            HoldoutStrategy::None => 1, // unused, but non-zero
            HoldoutStrategy::Periodic { period } => {
                assert!(*period > 0, "holdout period must be > 0");
                1 // unused
            }
            HoldoutStrategy::Random {
                holdout_fraction,
                seed,
            } => {
                assert!(
                    *holdout_fraction >= 0.0 && *holdout_fraction <= 1.0,
                    "holdout_fraction must be in [0.0, 1.0], got {}",
                    holdout_fraction
                );
                assert!(*seed != 0, "xorshift64 seed must be non-zero");
                *seed
            }
        };

        Self {
            holdout,
            regression: RegressionMetrics::new(),
            samples_seen: 0,
            samples_evaluated: 0,
            samples_trained: 0,
            rng_state,
        }
    }

    /// Process a single sample: always evaluate, conditionally train.
    ///
    /// Returns the prediction (made before any potential training).
    ///
    /// The model is always queried for a prediction and metrics are always
    /// recorded. Whether the model trains on this sample depends on the
    /// holdout strategy:
    ///
    /// - `None`: always train.
    /// - `Periodic { period }`: skip training every `period`-th sample.
    /// - `Random { holdout_fraction, .. }`: skip training with probability
    ///   `holdout_fraction`.
    pub fn step(&mut self, model: &mut dyn StreamingLearner, features: &[f64], target: f64) -> f64 {
        // 1. Predict
        let prediction = model.predict(features);

        // 2. Always record metrics
        self.regression.update(target, prediction);
        self.samples_evaluated += 1;

        // 3. Decide whether to train based on holdout strategy
        let should_train = match &self.holdout {
            HoldoutStrategy::None => true,
            HoldoutStrategy::Periodic { period } => {
                // Every period-th sample is held out (not trained on)
                self.samples_seen % *period as u64 != 0
            }
            HoldoutStrategy::Random {
                holdout_fraction, ..
            } => {
                let rand_val = xorshift64(&mut self.rng_state) as f64 / u64::MAX as f64;
                rand_val >= *holdout_fraction
            }
        };

        // 4. Conditionally train
        if should_train {
            model.train(features, target);
            self.samples_trained += 1;
        }

        // 5. Increment counter
        self.samples_seen += 1;

        prediction
    }

    /// Run the full evaluation protocol over a slice of observations.
    ///
    /// Iterates over `data`, calling [`step`](Self::step) for each observation.
    pub fn evaluate<O: Observation>(&mut self, model: &mut dyn StreamingLearner, data: &[O]) {
        for obs in data {
            self.step(model, obs.features(), obs.target());
        }
    }

    /// Reference to the cumulative regression metrics.
    pub fn regression_metrics(&self) -> &RegressionMetrics {
        &self.regression
    }

    /// Total number of samples processed.
    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Number of samples evaluated (always equals `samples_seen`).
    pub fn samples_evaluated(&self) -> u64 {
        self.samples_evaluated
    }

    /// Number of samples the model was trained on.
    pub fn samples_trained(&self) -> u64 {
        self.samples_trained
    }

    /// Actual holdout fraction: `1.0 - (samples_trained / samples_seen)`.
    ///
    /// Returns 0.0 if no samples have been seen yet.
    pub fn holdout_fraction_actual(&self) -> f64 {
        if self.samples_seen == 0 {
            return 0.0;
        }
        1.0 - (self.samples_trained as f64 / self.samples_seen as f64)
    }

    /// Reset all metrics and counters to the initial state.
    ///
    /// The holdout strategy is preserved. For `Random`, the PRNG state is
    /// reset to the original seed.
    pub fn reset(&mut self) {
        self.regression.reset();
        self.samples_seen = 0;
        self.samples_evaluated = 0;
        self.samples_trained = 0;

        // Reset PRNG to original seed for reproducibility
        if let HoldoutStrategy::Random { seed, .. } = &self.holdout {
            self.rng_state = *seed;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::StreamingLearner;
    use crate::sample::Sample;

    /// Mock learner that always predicts a fixed constant.
    struct MockLearner {
        prediction: f64,
        n: u64,
    }

    impl MockLearner {
        fn new(prediction: f64) -> Self {
            Self { prediction, n: 0 }
        }
    }

    impl StreamingLearner for MockLearner {
        fn train_one(&mut self, _features: &[f64], _target: f64, _weight: f64) {
            self.n += 1;
        }

        fn predict(&self, _features: &[f64]) -> f64 {
            self.prediction
        }

        fn n_samples_seen(&self) -> u64 {
            self.n
        }

        fn reset(&mut self) {
            self.n = 0;
        }
    }

    #[test]
    fn holdout_none_trains_on_everything() {
        let mut model = MockLearner::new(0.0);
        let mut validator = ProgressiveValidator::new(HoldoutStrategy::None);

        for i in 0..100 {
            validator.step(&mut model, &[i as f64], i as f64);
        }

        assert_eq!(validator.samples_seen(), 100);
        assert_eq!(validator.samples_evaluated(), 100);
        assert_eq!(validator.samples_trained(), 100);
        assert!(
            (validator.holdout_fraction_actual() - 0.0).abs() < 1e-12,
            "holdout fraction should be 0.0 for None strategy"
        );
        assert_eq!(model.n_samples_seen(), 100);
    }

    #[test]
    fn holdout_periodic_holds_out_every_nth() {
        let mut model = MockLearner::new(0.0);
        let holdout = HoldoutStrategy::Periodic { period: 5 };
        let mut validator = ProgressiveValidator::new(holdout);

        for i in 0..100 {
            validator.step(&mut model, &[i as f64], i as f64);
        }

        assert_eq!(validator.samples_seen(), 100);
        assert_eq!(validator.samples_evaluated(), 100);
        // Held out at samples_seen=0,5,10,...,95 => 20 held out, 80 trained
        assert_eq!(validator.samples_trained(), 80);
        assert_eq!(model.n_samples_seen(), 80);

        let actual = validator.holdout_fraction_actual();
        assert!(
            (actual - 0.2).abs() < 0.01,
            "expected holdout ~0.2, got {}",
            actual
        );
    }

    #[test]
    fn holdout_random_approximates_target_fraction() {
        let mut model = MockLearner::new(0.0);
        let holdout = HoldoutStrategy::Random {
            holdout_fraction: 0.3,
            seed: 42,
        };
        let mut validator = ProgressiveValidator::new(holdout);

        for i in 0..10_000 {
            validator.step(&mut model, &[i as f64], i as f64);
        }

        assert_eq!(validator.samples_seen(), 10_000);
        assert_eq!(validator.samples_evaluated(), 10_000);

        let actual = validator.holdout_fraction_actual();
        assert!(
            (actual - 0.3).abs() < 0.05,
            "expected holdout ~0.3, got {} (tolerance 0.05)",
            actual
        );
    }

    #[test]
    fn holdout_random_deterministic_with_same_seed() {
        let run = |seed: u64| -> Vec<bool> {
            let mut model = MockLearner::new(0.0);
            let holdout = HoldoutStrategy::Random {
                holdout_fraction: 0.5,
                seed,
            };
            let mut validator = ProgressiveValidator::new(holdout);
            let mut pattern = Vec::new();
            for i in 0..50 {
                let prev_trained = validator.samples_trained();
                validator.step(&mut model, &[i as f64], i as f64);
                // Record whether this specific sample was trained on
                pattern.push(validator.samples_trained() > prev_trained);
            }
            pattern
        };

        let run1 = run(12345);
        let run2 = run(12345);
        assert_eq!(
            run1, run2,
            "same seed should produce identical holdout patterns"
        );

        // Different seed should (very likely) produce different pattern
        let run3 = run(99999);
        assert_ne!(
            run1, run3,
            "different seeds should produce different holdout patterns"
        );
    }

    #[test]
    fn step_returns_prediction() {
        let mut model = MockLearner::new(7.5);
        let mut validator = ProgressiveValidator::new(HoldoutStrategy::None);

        let pred = validator.step(&mut model, &[1.0], 5.0);
        assert!(
            (pred - 7.5).abs() < 1e-12,
            "step should return the model's prediction, got {}",
            pred
        );
    }

    #[test]
    fn evaluate_over_observations() {
        let mut model = MockLearner::new(0.0);
        let mut validator = ProgressiveValidator::new(HoldoutStrategy::None);

        let data: Vec<Sample> = (0..30)
            .map(|i| Sample::new(vec![i as f64], i as f64))
            .collect();

        validator.evaluate(&mut model, &data);

        assert_eq!(validator.samples_seen(), 30);
        assert_eq!(validator.samples_evaluated(), 30);
        assert_eq!(validator.samples_trained(), 30);
    }

    #[test]
    fn reset_clears_all_state() {
        let mut model = MockLearner::new(0.0);
        let holdout = HoldoutStrategy::Random {
            holdout_fraction: 0.2,
            seed: 42,
        };
        let mut validator = ProgressiveValidator::new(holdout);

        for i in 0..50 {
            validator.step(&mut model, &[i as f64], i as f64);
        }

        assert!(validator.samples_seen() > 0);
        validator.reset();

        assert_eq!(validator.samples_seen(), 0);
        assert_eq!(validator.samples_evaluated(), 0);
        assert_eq!(validator.samples_trained(), 0);
        assert_eq!(validator.regression_metrics().n_samples(), 0);
        assert!(
            (validator.holdout_fraction_actual() - 0.0).abs() < 1e-12,
            "holdout fraction should be 0.0 after reset"
        );
    }

    #[test]
    fn reset_restores_prng_seed() {
        let holdout = HoldoutStrategy::Random {
            holdout_fraction: 0.5,
            seed: 42,
        };

        let mut model1 = MockLearner::new(0.0);
        let mut validator = ProgressiveValidator::new(holdout);

        // Run once
        for i in 0..20 {
            validator.step(&mut model1, &[i as f64], i as f64);
        }
        let trained_first_run = validator.samples_trained();

        // Reset and run again
        validator.reset();
        let mut model2 = MockLearner::new(0.0);
        for i in 0..20 {
            validator.step(&mut model2, &[i as f64], i as f64);
        }
        let trained_second_run = validator.samples_trained();

        assert_eq!(
            trained_first_run, trained_second_run,
            "reset should restore PRNG state, producing identical holdout patterns"
        );
    }

    #[test]
    fn metrics_always_recorded_even_for_holdout() {
        let mut model = MockLearner::new(0.0);
        let holdout = HoldoutStrategy::Periodic { period: 2 };
        let mut validator = ProgressiveValidator::new(holdout);

        for i in 0..10 {
            validator.step(&mut model, &[i as f64], i as f64);
        }

        // All 10 samples should be evaluated
        assert_eq!(validator.samples_evaluated(), 10);
        assert_eq!(validator.regression_metrics().n_samples(), 10);

        // But only ~half are trained on
        // Held out: 0, 2, 4, 6, 8 => 5 held out, 5 trained
        assert_eq!(validator.samples_trained(), 5);
    }

    #[test]
    #[should_panic(expected = "holdout period must be > 0")]
    fn periodic_zero_period_panics() {
        ProgressiveValidator::new(HoldoutStrategy::Periodic { period: 0 });
    }

    #[test]
    #[should_panic(expected = "holdout_fraction must be in [0.0, 1.0]")]
    fn random_invalid_fraction_panics() {
        ProgressiveValidator::new(HoldoutStrategy::Random {
            holdout_fraction: 1.5,
            seed: 42,
        });
    }

    #[test]
    #[should_panic(expected = "xorshift64 seed must be non-zero")]
    fn random_zero_seed_panics() {
        ProgressiveValidator::new(HoldoutStrategy::Random {
            holdout_fraction: 0.5,
            seed: 0,
        });
    }

    #[test]
    fn holdout_fraction_zero_means_all_train() {
        let mut model = MockLearner::new(0.0);
        let holdout = HoldoutStrategy::Random {
            holdout_fraction: 0.0,
            seed: 42,
        };
        let mut validator = ProgressiveValidator::new(holdout);

        for i in 0..100 {
            validator.step(&mut model, &[i as f64], i as f64);
        }

        assert_eq!(validator.samples_trained(), 100);
    }

    #[test]
    fn holdout_fraction_one_means_none_train() {
        let mut model = MockLearner::new(0.0);
        let holdout = HoldoutStrategy::Random {
            holdout_fraction: 1.0,
            seed: 42,
        };
        let mut validator = ProgressiveValidator::new(holdout);

        for i in 0..100 {
            validator.step(&mut model, &[i as f64], i as f64);
        }

        assert_eq!(validator.samples_trained(), 0);
        assert_eq!(model.n_samples_seen(), 0);
    }
}
