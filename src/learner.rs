//! Unified streaming learner trait for polymorphic model composition.
//!
//! [`StreamingLearner`] is an **object-safe** trait that abstracts over any
//! online/streaming machine learning model -- gradient boosted trees, linear
//! models, Naive Bayes, Mondrian forests, or anything else that can ingest
//! samples one at a time and produce predictions.
//!
//! # Motivation
//!
//! Stacking ensembles and meta-learners need to treat heterogeneous base
//! models uniformly: train them on the same stream, collect their predictions
//! as features for a combiner, and manage their lifecycle (reset, clone,
//! serialization). `StreamingLearner` provides exactly this interface.
//!
//! # Object Safety
//!
//! The trait is deliberately object-safe: every method uses `&self` /
//! `&mut self` with concrete return types (no generics on methods, no
//! `Self`-by-value in non-`Sized` positions). This means you can store
//! `Box<dyn StreamingLearner>` in a `Vec`, enabling runtime-polymorphic
//! stacking without monomorphization.
//!
//! # Usage
//!
//! ```
//! use irithyll::learner::{StreamingLearner, SGBTLearner};
//! use irithyll::SGBTConfig;
//!
//! // Create a base learner from config
//! let config = SGBTConfig::builder()
//!     .n_steps(10)
//!     .learning_rate(0.1)
//!     .build()
//!     .unwrap();
//! let mut learner = SGBTLearner::from_config(config);
//!
//! // Train incrementally
//! learner.train(&[1.0, 2.0], 3.0);
//! learner.train(&[4.0, 5.0], 6.0);
//!
//! // Predict
//! let pred = learner.predict(&[1.0, 2.0]);
//! assert!(pred.is_finite());
//!
//! // Use as trait object for stacking
//! let boxed: Box<dyn StreamingLearner> = Box::new(learner);
//! assert_eq!(boxed.n_samples_seen(), 2);
//! ```

use std::fmt;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::SGBT;
use crate::loss::squared::SquaredLoss;
use crate::loss::Loss;
use crate::sample::SampleRef;

// ---------------------------------------------------------------------------
// StreamingLearner trait
// ---------------------------------------------------------------------------

/// Object-safe trait for any streaming (online) machine learning model.
///
/// All methods use `&self` or `&mut self` with concrete return types,
/// ensuring the trait can be used behind `Box<dyn StreamingLearner>` for
/// runtime-polymorphic stacking ensembles.
///
/// The `Send + Sync` supertraits allow learners to be shared across threads
/// (e.g., for parallel prediction in async pipelines).
///
/// # Required Methods
///
/// | Method | Purpose |
/// |--------|---------|
/// | [`train_one`](Self::train_one) | Ingest a single weighted observation |
/// | [`predict`](Self::predict) | Produce a prediction for a feature vector |
/// | [`n_samples_seen`](Self::n_samples_seen) | Total observations ingested so far |
/// | [`reset`](Self::reset) | Clear all learned state, returning to a fresh model |
///
/// # Default Methods
///
/// | Method | Purpose |
/// |--------|---------|
/// | [`train`](Self::train) | Convenience wrapper calling `train_one` with unit weight |
/// | [`predict_batch`](Self::predict_batch) | Map `predict` over a slice of feature vectors |
pub trait StreamingLearner: Send + Sync {
    /// Train on a single observation with explicit sample weight.
    ///
    /// This is the fundamental training primitive. All streaming models must
    /// support weighted incremental updates -- even if the weight is simply
    /// used to scale gradient contributions.
    ///
    /// # Arguments
    ///
    /// * `features` -- feature vector for this observation
    /// * `target` -- target value (regression) or class label (classification)
    /// * `weight` -- sample weight (1.0 for uniform weighting)
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64);

    /// Predict the target for the given feature vector.
    ///
    /// Returns the raw model output (no loss transform applied). For SGBT
    /// this is the sum of tree predictions; for linear models this is the
    /// dot product plus bias.
    fn predict(&self, features: &[f64]) -> f64;

    /// Total number of observations trained on since creation or last reset.
    fn n_samples_seen(&self) -> u64;

    /// Reset the model to its initial (untrained) state.
    ///
    /// After calling `reset()`, the model should behave identically to a
    /// freshly constructed instance with the same configuration. In particular,
    /// `n_samples_seen()` must return 0.
    fn reset(&mut self);

    /// Train on a single observation with unit weight.
    ///
    /// Convenience wrapper around [`train_one`](Self::train_one) that passes
    /// `weight = 1.0`. This is the most common training call in practice.
    fn train(&mut self, features: &[f64], target: f64) {
        self.train_one(features, target, 1.0);
    }

    /// Predict for each row in a feature matrix.
    ///
    /// Returns a `Vec<f64>` with one prediction per input row. The default
    /// implementation simply maps [`predict`](Self::predict) over the slices;
    /// concrete implementations may override this for SIMD or batch-optimized
    /// prediction paths.
    ///
    /// # Arguments
    ///
    /// * `feature_matrix` -- each element is a feature vector (one row)
    fn predict_batch(&self, feature_matrix: &[&[f64]]) -> Vec<f64> {
        feature_matrix.iter().map(|row| self.predict(row)).collect()
    }
}

// ---------------------------------------------------------------------------
// SGBTLearner -- adapter wrapping SGBT<L> into StreamingLearner
// ---------------------------------------------------------------------------

/// Adapter that wraps an [`SGBT`] ensemble into the [`StreamingLearner`] trait.
///
/// This is the primary way to use SGBT models in polymorphic stacking
/// ensembles. The loss function `L` is monomorphized at compile time for
/// zero-cost gradient dispatch, while the `StreamingLearner` trait provides
/// the uniform interface needed by meta-learners.
///
/// # Type Parameter
///
/// * `L` -- loss function type, defaulting to [`SquaredLoss`] for regression.
///   Any `L: Loss + Clone` is supported.
///
/// # Examples
///
/// ```
/// use irithyll::learner::SGBTLearner;
/// use irithyll::SGBTConfig;
///
/// // Default squared loss:
/// let config = SGBTConfig::builder().n_steps(5).build().unwrap();
/// let learner = SGBTLearner::from_config(config);
/// ```
///
/// ```
/// use irithyll::learner::SGBTLearner;
/// use irithyll::{SGBTConfig, SGBT};
/// use irithyll::loss::logistic::LogisticLoss;
///
/// // Custom loss via wrapping an existing SGBT:
/// let config = SGBTConfig::builder().n_steps(5).build().unwrap();
/// let model = SGBT::with_loss(config, LogisticLoss);
/// let learner = SGBTLearner::new(model);
/// ```
pub struct SGBTLearner<L: Loss = SquaredLoss> {
    inner: SGBT<L>,
}

impl<L: Loss> SGBTLearner<L> {
    /// Wrap an existing [`SGBT`] model into a `SGBTLearner`.
    ///
    /// The model retains all of its current state (trained trees, samples
    /// seen, etc.). This enables wrapping a partially-trained model.
    #[inline]
    pub fn new(model: SGBT<L>) -> Self {
        Self { inner: model }
    }

    /// Immutable access to the underlying [`SGBT`] model.
    ///
    /// Useful for inspecting model internals (config, base prediction,
    /// number of steps) without consuming the adapter.
    #[inline]
    pub fn inner(&self) -> &SGBT<L> {
        &self.inner
    }

    /// Mutable access to the underlying [`SGBT`] model.
    ///
    /// Enables calling SGBT-specific methods (e.g., `predict_transformed`,
    /// serialization) that are not part of the `StreamingLearner` interface.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut SGBT<L> {
        &mut self.inner
    }

    /// Consume the adapter and return the underlying [`SGBT`] model.
    ///
    /// This is useful when you need to serialize the model or switch from
    /// polymorphic back to monomorphic usage.
    #[inline]
    pub fn into_inner(self) -> SGBT<L> {
        self.inner
    }
}

impl SGBTLearner<SquaredLoss> {
    /// Create a new `SGBTLearner` with squared loss from a configuration.
    ///
    /// This is the most common constructor for regression tasks. For custom
    /// losses, construct the [`SGBT`] first with [`SGBT::with_loss`] and
    /// wrap it via [`SGBTLearner::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use irithyll::learner::{SGBTLearner, StreamingLearner};
    /// use irithyll::SGBTConfig;
    ///
    /// let config = SGBTConfig::builder()
    ///     .n_steps(10)
    ///     .learning_rate(0.05)
    ///     .build()
    ///     .unwrap();
    /// let learner = SGBTLearner::from_config(config);
    /// assert_eq!(learner.n_samples_seen(), 0);
    /// ```
    #[inline]
    pub fn from_config(config: SGBTConfig) -> Self {
        Self {
            inner: SGBT::new(config),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl for SGBTLearner
// ---------------------------------------------------------------------------

impl<L: Loss> StreamingLearner for SGBTLearner<L> {
    #[inline]
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let sample = SampleRef::weighted(features, target, weight);
        self.inner.train_one(&sample);
    }

    #[inline]
    fn predict(&self, features: &[f64]) -> f64 {
        self.inner.predict(features)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.inner.n_samples_seen()
    }

    #[inline]
    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ---------------------------------------------------------------------------
// Clone impl -- manual to match irithyll patterns (no derive)
// ---------------------------------------------------------------------------

impl<L: Loss + Clone> Clone for SGBTLearner<L> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Debug impl -- manual, avoids requiring Debug on L
// ---------------------------------------------------------------------------

impl<L: Loss> fmt::Debug for SGBTLearner<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SGBTLearner")
            .field("inner", &self.inner)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SGBTConfig;

    /// Shared minimal config for tests.
    fn default_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .build()
            .unwrap()
    }

    #[test]
    fn test_sgbt_learner_creation() {
        let learner = SGBTLearner::from_config(default_config());
        assert_eq!(learner.n_samples_seen(), 0);
        // Untrained model predicts zero (no base prediction initialized).
        let pred = learner.predict(&[1.0, 2.0, 3.0]);
        assert!(pred.abs() < 1e-12);
    }

    #[test]
    fn test_trait_object_safety() {
        // Verify that Box<dyn StreamingLearner> compiles and works.
        let learner = SGBTLearner::from_config(default_config());
        let mut boxed: Box<dyn StreamingLearner> = Box::new(learner);

        // Train through the trait object.
        boxed.train(&[1.0, 2.0, 3.0], 5.0);
        assert_eq!(boxed.n_samples_seen(), 1);

        // Predict through the trait object.
        let pred = boxed.predict(&[1.0, 2.0, 3.0]);
        assert!(pred.is_finite());

        // Reset through the trait object.
        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }

    #[test]
    fn test_train_and_predict() {
        let mut learner = SGBTLearner::from_config(default_config());

        // Record prediction before training.
        let features = [1.0, 2.0, 3.0];
        let pred_before = learner.predict(&features);

        // Train on enough samples to initialize base prediction and grow trees.
        for i in 0..100 {
            learner.train(&features, (i as f64) * 0.1);
        }
        assert_eq!(learner.n_samples_seen(), 100);

        // Prediction should have changed from the untrained state.
        let pred_after = learner.predict(&features);
        assert!(
            (pred_after - pred_before).abs() > 1e-6,
            "prediction should change after training: before={}, after={}",
            pred_before,
            pred_after,
        );
        assert!(pred_after.is_finite());
    }

    #[test]
    fn test_reset() {
        let mut learner = SGBTLearner::from_config(default_config());

        // Train some samples.
        for i in 0..50 {
            learner.train(&[1.0, 2.0], i as f64);
        }
        assert_eq!(learner.n_samples_seen(), 50);

        // Reset and verify state is cleared.
        learner.reset();
        assert_eq!(learner.n_samples_seen(), 0);

        // After reset, predictions should return to zero (uninitialised base).
        let pred = learner.predict(&[1.0, 2.0]);
        assert!(
            pred.abs() < 1e-12,
            "prediction after reset should be zero, got {}",
            pred,
        );
    }

    #[test]
    fn test_predict_batch() {
        let mut learner = SGBTLearner::from_config(default_config());

        // Train a bit so predictions are non-trivial.
        for i in 0..60 {
            learner.train(&[i as f64, (i as f64) * 0.5], i as f64);
        }

        let rows: Vec<&[f64]> = vec![&[1.0, 0.5], &[10.0, 5.0], &[30.0, 15.0]];
        let batch = learner.predict_batch(&rows);

        // Batch results should match individual predictions exactly.
        assert_eq!(batch.len(), rows.len());
        for (i, row) in rows.iter().enumerate() {
            let individual = learner.predict(row);
            assert!(
                (batch[i] - individual).abs() < 1e-12,
                "batch[{}]={} != individual={}",
                i,
                batch[i],
                individual,
            );
        }
    }

    #[test]
    fn test_inner_access() {
        let config = default_config();
        let learner = SGBTLearner::from_config(config.clone());

        // inner() should return the SGBT reference with matching config.
        let sgbt = learner.inner();
        assert_eq!(sgbt.n_samples_seen(), 0);
        assert_eq!(sgbt.config().n_steps, config.n_steps);
    }

    #[test]
    fn test_clone() {
        let mut learner = SGBTLearner::from_config(default_config());

        // Train the original.
        for i in 0..80 {
            learner.train(&[i as f64, (i as f64) * 2.0], i as f64);
        }

        // Clone should be an independent copy.
        let mut cloned = learner.clone();
        assert_eq!(cloned.n_samples_seen(), learner.n_samples_seen());

        // Predictions should match immediately after clone.
        let features = [5.0, 10.0];
        let pred_orig = learner.predict(&features);
        let pred_clone = cloned.predict(&features);
        assert!(
            (pred_orig - pred_clone).abs() < 1e-12,
            "clone prediction should match original: {} vs {}",
            pred_orig,
            pred_clone,
        );

        // Training the clone should not affect the original.
        for i in 0..50 {
            cloned.train(&[i as f64, (i as f64) * 2.0], 999.0);
        }
        assert_eq!(learner.n_samples_seen(), 80);
        assert_eq!(cloned.n_samples_seen(), 130);

        // Predictions should now diverge.
        let pred_orig_after = learner.predict(&features);
        let pred_clone_after = cloned.predict(&features);
        assert!(
            (pred_orig - pred_orig_after).abs() < 1e-12,
            "original should be unchanged after training clone",
        );
        assert!(
            (pred_clone_after - pred_clone).abs() > 1e-6,
            "clone prediction should change after further training",
        );
    }
}
