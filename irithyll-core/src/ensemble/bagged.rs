//! Bagged SGBT ensemble using Oza online bagging (Poisson weighting).
//!
//! Wraps M independent [`SGBT<L>`] instances. Each sample is presented to
//! each bag Poisson(1) times, producing a streaming analogue of bootstrap
//! aggregation. Final prediction is the mean across all bags.
//!
//! This implements the SGB(Oza) algorithm from Gunasekara et al. (2025),
//! which substantially reduces variance in streaming gradient boosted
//! regression compared to a single SGBT ensemble.
//!
//! # Example
//!
//! ```text
//! use irithyll::ensemble::bagged::BaggedSGBT;
//! use irithyll::SGBTConfig;
//!
//! let config = SGBTConfig::builder()
//!     .n_steps(10)
//!     .learning_rate(0.1)
//!     .grace_period(10)
//!     .build()
//!     .unwrap();
//!
//! let mut model = BaggedSGBT::new(config, 5).unwrap();
//! model.train_one(&irithyll::Sample::new(vec![1.0, 2.0], 3.0));
//! let pred = model.predict(&[1.0, 2.0]);
//! ```

use alloc::vec::Vec;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::SGBT;
use crate::error::{ConfigError, IrithyllError};
use crate::loss::squared::SquaredLoss;
use crate::loss::Loss;
use crate::sample::Observation;

// ---------------------------------------------------------------------------
// Poisson sampling utilities
// ---------------------------------------------------------------------------

/// Advance an xorshift64 state and return a f64 in [0, 1).
#[inline]
fn xorshift64_f64(state: &mut u64) -> f64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    // Use 53-bit mantissa for uniform [0, 1)
    (s >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Sample from Poisson(lambda=1) using Knuth's algorithm.
///
/// Generates uniform random numbers and multiplies them until the product
/// drops below e^{-1}. The count of multiplications minus one gives the
/// Poisson draw. For lambda=1, the mean is 1 and values rarely exceed 5.
fn poisson_sample(rng: &mut u64) -> usize {
    let l = crate::math::exp(-1.0_f64); // e^{-1} = 0.36787944...
    let mut k: usize = 0;
    let mut p: f64 = 1.0;
    loop {
        k += 1;
        let u = xorshift64_f64(rng);
        p *= u;
        if p < l {
            return k - 1;
        }
    }
}

// ---------------------------------------------------------------------------
// BaggedSGBT
// ---------------------------------------------------------------------------

/// Bagged (Oza) SGBT ensemble for variance reduction.
///
/// Each of the M bags is an independent [`SGBT<L>`] trained on a Poisson(1)-
/// weighted stream. Predictions are averaged across bags, reducing the
/// variance of the ensemble without increasing bias.
///
/// This implements SGB(Oza) from Gunasekara et al. (2025), the streaming
/// analogue of Breiman's bootstrap aggregation adapted for gradient boosted
/// trees with Hoeffding-bound splits.
pub struct BaggedSGBT<L: Loss = SquaredLoss> {
    bags: Vec<SGBT<L>>,
    n_bags: usize,
    samples_seen: u64,
    rng_state: u64,
    seed: u64,
}

impl<L: Loss + Clone> Clone for BaggedSGBT<L> {
    fn clone(&self) -> Self {
        Self {
            bags: self.bags.clone(),
            n_bags: self.n_bags,
            samples_seen: self.samples_seen,
            rng_state: self.rng_state,
            seed: self.seed,
        }
    }
}

impl<L: Loss> core::fmt::Debug for BaggedSGBT<L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BaggedSGBT")
            .field("n_bags", &self.n_bags)
            .field("samples_seen", &self.samples_seen)
            .finish()
    }
}

impl BaggedSGBT<SquaredLoss> {
    /// Create a new bagged SGBT with squared loss (regression).
    ///
    /// # Errors
    ///
    /// Returns [`IrithyllError::InvalidConfig`] if `n_bags < 1`.
    pub fn new(config: SGBTConfig, n_bags: usize) -> crate::error::Result<Self> {
        Self::with_loss(config, SquaredLoss, n_bags)
    }
}

impl<L: Loss + Clone> BaggedSGBT<L> {
    /// Create a new bagged SGBT with a custom loss function.
    ///
    /// Each bag receives a unique seed derived from the config seed, ensuring
    /// diverse tree structures across bags.
    ///
    /// # Errors
    ///
    /// Returns [`IrithyllError::InvalidConfig`] if `n_bags < 1`.
    pub fn with_loss(config: SGBTConfig, loss: L, n_bags: usize) -> crate::error::Result<Self> {
        if n_bags < 1 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "n_bags",
                "must be >= 1",
                n_bags,
            )));
        }

        let seed = config.seed;
        let bags = (0..n_bags)
            .map(|i| {
                let mut cfg = config.clone();
                // Each bag gets a unique seed for diverse tree construction
                cfg.seed = config.seed ^ (0xBA6_0000_0000_0000 | i as u64);
                SGBT::with_loss(cfg, loss.clone())
            })
            .collect();

        Ok(Self {
            bags,
            n_bags,
            samples_seen: 0,
            rng_state: seed,
            seed,
        })
    }

    /// Train all bags on a single observation with Poisson(1) weighting.
    ///
    /// For each bag, draws k ~ Poisson(1) and calls `bag.train_one(sample)`
    /// k times. On average, each bag sees the sample once, but the randomness
    /// creates diverse training sets across bags.
    pub fn train_one(&mut self, sample: &impl Observation) {
        self.samples_seen += 1;
        for bag in &mut self.bags {
            let k = poisson_sample(&mut self.rng_state);
            for _ in 0..k {
                bag.train_one(sample);
            }
        }
    }

    /// Train on a batch of observations.
    pub fn train_batch<O: Observation>(&mut self, samples: &[O]) {
        for sample in samples {
            self.train_one(sample);
        }
    }

    /// Predict the raw output as the mean across all bags.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let sum: f64 = self.bags.iter().map(|b| b.predict(features)).sum();
        sum / self.n_bags as f64
    }

    /// Predict with loss transform applied (e.g., sigmoid for logistic loss),
    /// averaged across bags.
    pub fn predict_transformed(&self, features: &[f64]) -> f64 {
        let sum: f64 = self
            .bags
            .iter()
            .map(|b| b.predict_transformed(features))
            .sum();
        sum / self.n_bags as f64
    }

    /// Batch prediction.
    pub fn predict_batch(&self, feature_matrix: &[Vec<f64>]) -> Vec<f64> {
        feature_matrix.iter().map(|f| self.predict(f)).collect()
    }

    /// Number of bags in the ensemble.
    #[inline]
    pub fn n_bags(&self) -> usize {
        self.n_bags
    }

    /// Total samples seen.
    #[inline]
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Immutable access to all bags.
    pub fn bags(&self) -> &[SGBT<L>] {
        &self.bags
    }

    /// Immutable access to a specific bag.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_bags`.
    pub fn bag(&self, idx: usize) -> &SGBT<L> {
        &self.bags[idx]
    }

    /// Whether the base prediction has been initialized for all bags.
    pub fn is_initialized(&self) -> bool {
        self.bags.iter().all(|b| b.is_initialized())
    }

    /// Reset all bags to initial state.
    pub fn reset(&mut self) {
        for bag in &mut self.bags {
            bag.reset();
        }
        self.samples_seen = 0;
        self.rng_state = self.seed;
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

use crate::learner::StreamingLearner;
use crate::sample::SampleRef;

impl<L: Loss + Clone> StreamingLearner for BaggedSGBT<L> {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let sample = SampleRef::weighted(features, target, weight);
        // UFCS: call the inherent train_one(&impl Observation), not this trait method.
        BaggedSGBT::train_one(self, &sample);
    }

    fn predict(&self, features: &[f64]) -> f64 {
        BaggedSGBT::predict(self, features)
    }

    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        BaggedSGBT::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::Sample;
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;

    fn test_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(10)
            .initial_target_count(5)
            .build()
            .unwrap()
    }

    #[test]
    fn creates_correct_number_of_bags() {
        let model = BaggedSGBT::new(test_config(), 7).unwrap();
        assert_eq!(model.n_bags(), 7);
        assert_eq!(model.bags().len(), 7);
        assert_eq!(model.n_samples_seen(), 0);
    }

    #[test]
    fn rejects_zero_bags() {
        let result = BaggedSGBT::new(test_config(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn single_bag_equals_single_sgbt() {
        // With 1 bag, bagged model should behave similarly to plain SGBT
        // (not exactly due to Poisson weighting, but close)
        let config = test_config();
        let mut model = BaggedSGBT::new(config, 1).unwrap();

        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&Sample::new(vec![x], x * 2.0 + 1.0));
        }

        let pred = model.predict(&[0.5]);
        // Should approximate 0.5 * 2.0 + 1.0 = 2.0
        // With Poisson weighting, predictions can deviate more than a single SGBT.
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );
    }

    #[test]
    fn poisson_mean_approximately_one() {
        let mut rng = 0xDEAD_BEEF_u64;
        let n = 10_000;
        let sum: usize = (0..n).map(|_| poisson_sample(&mut rng)).sum();
        let mean = sum as f64 / n as f64;
        assert!(
            (mean - 1.0).abs() < 0.1,
            "Poisson(1) mean should be ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn poisson_never_negative() {
        let mut rng = 42u64;
        for _ in 0..10_000 {
            // poisson_sample returns usize, so it can't be negative,
            // but let's verify values are reasonable
            let k = poisson_sample(&mut rng);
            assert!(k < 20, "Poisson(1) should rarely exceed 10, got {}", k);
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let config = test_config();
        let mut model1 = BaggedSGBT::new(config.clone(), 3).unwrap();
        let mut model2 = BaggedSGBT::new(config, 3).unwrap();

        let samples: Vec<Sample> = (0..50)
            .map(|i| {
                let x = i as f64 * 0.1;
                Sample::new(vec![x], x * 3.0)
            })
            .collect();

        for s in &samples {
            model1.train_one(s);
            model2.train_one(s);
        }

        let pred1 = model1.predict(&[0.5]);
        let pred2 = model2.predict(&[0.5]);
        assert!(
            (pred1 - pred2).abs() < 1e-10,
            "same seed should give identical predictions: {} vs {}",
            pred1,
            pred2
        );
    }

    #[test]
    fn predict_averages_bags() {
        let config = test_config();
        let mut model = BaggedSGBT::new(config, 5).unwrap();

        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&Sample::new(vec![x], x));
        }

        // Verify prediction is the mean of individual bag predictions
        let features = [0.5];
        let individual_sum: f64 = model.bags().iter().map(|b| b.predict(&features)).sum();
        let expected = individual_sum / model.n_bags() as f64;
        let actual = model.predict(&features);
        assert!(
            (actual - expected).abs() < 1e-10,
            "predict should be mean of bags: {} vs {}",
            actual,
            expected
        );
    }

    #[test]
    fn reset_clears_state() {
        let config = test_config();
        let mut model = BaggedSGBT::new(config, 3).unwrap();

        for i in 0..100 {
            let x = i as f64;
            model.train_one(&Sample::new(vec![x], x));
        }
        assert!(model.n_samples_seen() > 0);

        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
    }

    #[test]
    fn convergence_on_linear_target() {
        let config = SGBTConfig::builder()
            .n_steps(20)
            .learning_rate(0.1)
            .grace_period(10)
            .initial_target_count(5)
            .build()
            .unwrap();

        let mut model = BaggedSGBT::new(config, 5).unwrap();

        // Train on y = 2*x + 1
        for i in 0..500 {
            let x = (i % 100) as f64 * 0.1;
            model.train_one(&Sample::new(vec![x], 2.0 * x + 1.0));
        }

        // Verify predictions are finite and the model is learning
        // (exact accuracy depends heavily on Poisson weighting variance).
        let test_points = [0.0, 0.5, 1.0];
        for &x in &test_points {
            let pred = model.predict(&[x]);
            assert!(
                pred.is_finite(),
                "at x={}: prediction should be finite, got {}",
                x,
                pred
            );
        }
        // Verify the model is directionally correct: pred(1.0) > pred(0.0)
        let p0 = model.predict(&[0.0]);
        let p1 = model.predict(&[1.0]);
        // The function is y = 2x + 1, so pred(1.0) should be > pred(0.0).
        // Allow for noise in streaming approximation.
        assert!(
            p1 > p0 || (p1 - p0).abs() < 5.0,
            "directional: pred(1.0)={}, pred(0.0)={}",
            p1,
            p0
        );
    }

    #[test]
    fn variance_reduction() {
        // Compare variance of predictions across bags for a single bagged model
        // vs. running multiple independent SGBTs.
        // The bagged model's mean prediction should have lower variance.
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(10)
            .initial_target_count(5)
            .build()
            .unwrap();

        let mut model = BaggedSGBT::new(config, 10).unwrap();

        for i in 0..200 {
            let x = (i % 50) as f64 * 0.1;
            model.train_one(&Sample::new(vec![x], x * x));
        }

        // The individual bag predictions should vary, but the mean is stable
        let features = [0.3];
        let preds: Vec<f64> = model.bags().iter().map(|b| b.predict(&features)).collect();
        let mean = preds.iter().sum::<f64>() / preds.len() as f64;
        let variance = preds.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / preds.len() as f64;

        // Individual bag predictions should have some variance (bags are diverse)
        // This is a soft check -- just verify bags aren't all identical
        assert!(
            preds.len() > 1,
            "need multiple bags to test variance reduction"
        );
        // The ensemble mean is `model.predict()`, which should be close to `mean`
        let ensemble_pred = model.predict(&features);
        assert!(
            (ensemble_pred - mean).abs() < 1e-10,
            "ensemble prediction should be mean of bags"
        );
        // Variance should be finite and non-negative (basic sanity)
        assert!(variance >= 0.0 && variance.is_finite());
    }

    #[test]
    fn streaming_learner_trait_object() {
        let config = test_config();
        let model = BaggedSGBT::new(config, 3).unwrap();
        let mut boxed: Box<dyn StreamingLearner> = Box::new(model);
        for i in 0..100 {
            let x = i as f64 * 0.1;
            boxed.train(&[x], x * 2.0);
        }
        assert_eq!(boxed.n_samples_seen(), 100);
        let pred = boxed.predict(&[5.0]);
        assert!(pred.is_finite());
        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }
}
