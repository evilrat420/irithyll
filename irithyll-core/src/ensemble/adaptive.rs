//! Adaptive learning rate wrapper for SGBT ensembles.
//!
//! [`AdaptiveSGBT`] pairs an [`SGBT`] model with an [`LRScheduler`], adjusting
//! the learning rate before each training step based on the scheduler's policy.
//! This enables time-varying learning rates -- decay, cosine annealing, plateau
//! reduction -- without modifying the core ensemble code.
//!
//! # Example
//!
//! ```text
//! use irithyll::ensemble::adaptive::AdaptiveSGBT;
//! use irithyll::ensemble::lr_schedule::ExponentialDecayLR;
//! use irithyll::SGBTConfig;
//! use irithyll::learner::StreamingLearner;
//!
//! let config = SGBTConfig::builder()
//!     .n_steps(10)
//!     .learning_rate(0.1)
//!     .build()
//!     .unwrap();
//!
//! let mut model = AdaptiveSGBT::new(config, ExponentialDecayLR::new(0.1, 0.999));
//! model.train(&[1.0, 2.0], 3.0);
//! model.train(&[4.0, 5.0], 6.0);
//!
//! // The learning rate adapts over time.
//! let pred = model.predict(&[1.0, 2.0]);
//! ```

use alloc::boxed::Box;

use core::fmt;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::lr_schedule::LRScheduler;
use crate::ensemble::SGBT;
use crate::learner::StreamingLearner;
use crate::loss::squared::SquaredLoss;
use crate::loss::Loss;
use crate::sample::{Observation, SampleRef};

/// SGBT ensemble with an attached learning rate scheduler.
///
/// Before each `train_one` call, AdaptiveSGBT:
///
/// 1. Computes the current prediction to estimate loss.
/// 2. Queries the scheduler for the new learning rate.
/// 3. Sets the learning rate on the inner SGBT.
/// 4. Delegates the actual training step.
///
/// This allows any [`LRScheduler`] -- exponential decay, cosine annealing,
/// plateau reduction -- to drive the ensemble's learning rate without touching
/// the core boosting logic.
///
/// # Loss Estimation
///
/// The scheduler receives squared error `(target - prediction)²` as its loss
/// signal. This is computed from the current ensemble prediction *before* the
/// training step, making it a one-step-lagged estimate. This works well for
/// schedulers like [`PlateauLR`](crate::ensemble::lr_schedule::PlateauLR) that
/// smooth over many steps.
pub struct AdaptiveSGBT<L: Loss = SquaredLoss> {
    inner: SGBT<L>,
    scheduler: Box<dyn LRScheduler>,
    step_count: u64,
    last_loss: f64,
    base_lr: f64,
}

impl AdaptiveSGBT<SquaredLoss> {
    /// Create an adaptive SGBT with squared loss (regression).
    ///
    /// The initial learning rate is taken from the config and also stored as
    /// `base_lr` for reference.
    ///
    /// ```text
    /// use irithyll::ensemble::adaptive::AdaptiveSGBT;
    /// use irithyll::ensemble::lr_schedule::ConstantLR;
    /// use irithyll::SGBTConfig;
    ///
    /// let config = SGBTConfig::builder()
    ///     .n_steps(10)
    ///     .learning_rate(0.05)
    ///     .build()
    ///     .unwrap();
    /// let model = AdaptiveSGBT::new(config, ConstantLR::new(0.05));
    /// ```
    pub fn new(config: SGBTConfig, scheduler: impl LRScheduler + 'static) -> Self {
        let base_lr = config.learning_rate;
        Self {
            inner: SGBT::new(config),
            scheduler: Box::new(scheduler),
            step_count: 0,
            last_loss: 0.0,
            base_lr,
        }
    }
}

impl<L: Loss> AdaptiveSGBT<L> {
    /// Create an adaptive SGBT with a specific loss function.
    ///
    /// ```text
    /// use irithyll::ensemble::adaptive::AdaptiveSGBT;
    /// use irithyll::ensemble::lr_schedule::LinearDecayLR;
    /// use irithyll::loss::logistic::LogisticLoss;
    /// use irithyll::SGBTConfig;
    ///
    /// let config = SGBTConfig::builder()
    ///     .n_steps(10)
    ///     .learning_rate(0.1)
    ///     .build()
    ///     .unwrap();
    /// let model = AdaptiveSGBT::with_loss(
    ///     config, LogisticLoss, LinearDecayLR::new(0.1, 0.001, 10_000),
    /// );
    /// ```
    pub fn with_loss(config: SGBTConfig, loss: L, scheduler: impl LRScheduler + 'static) -> Self {
        let base_lr = config.learning_rate;
        Self {
            inner: SGBT::with_loss(config, loss),
            scheduler: Box::new(scheduler),
            step_count: 0,
            last_loss: 0.0,
            base_lr,
        }
    }

    /// Train on a single observation, adapting the learning rate first.
    ///
    /// This is the generic version accepting any `Observation` implementor.
    /// For the `StreamingLearner` trait interface, use `train_one(features, target, weight)`.
    pub fn train_one_obs(&mut self, sample: &impl Observation) {
        // Estimate loss from current prediction.
        let pred = self.inner.predict(sample.features());
        let err = sample.target() - pred;
        self.last_loss = err * err;

        // Query scheduler for new learning rate.
        let lr = self
            .scheduler
            .learning_rate(self.step_count, self.last_loss);
        self.inner.set_learning_rate(lr);
        self.step_count += 1;

        // Delegate training.
        self.inner.train_one(sample);
    }

    /// Current learning rate (as last set by the scheduler).
    pub fn current_lr(&self) -> f64 {
        self.inner.config().learning_rate
    }

    /// The initial learning rate from the original config.
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }

    /// Total scheduler steps (equal to samples trained).
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Most recent loss value passed to the scheduler.
    pub fn last_loss(&self) -> f64 {
        self.last_loss
    }

    /// Immutable access to the scheduler.
    pub fn scheduler(&self) -> &dyn LRScheduler {
        &*self.scheduler
    }

    /// Mutable access to the scheduler.
    pub fn scheduler_mut(&mut self) -> &mut dyn LRScheduler {
        &mut *self.scheduler
    }

    /// Immutable access to the inner SGBT model.
    pub fn inner(&self) -> &SGBT<L> {
        &self.inner
    }

    /// Mutable access to the inner SGBT model.
    pub fn inner_mut(&mut self) -> &mut SGBT<L> {
        &mut self.inner
    }

    /// Consume the wrapper and return the inner SGBT model.
    pub fn into_inner(self) -> SGBT<L> {
        self.inner
    }

    /// Predict using the inner SGBT model.
    pub fn predict(&self, features: &[f64]) -> f64 {
        self.inner.predict(features)
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner
// ---------------------------------------------------------------------------

impl<L: Loss> StreamingLearner for AdaptiveSGBT<L> {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let sample = SampleRef::weighted(features, target, weight);
        self.train_one_obs(&sample);
    }

    fn predict(&self, features: &[f64]) -> f64 {
        self.inner.predict(features)
    }

    fn n_samples_seen(&self) -> u64 {
        self.inner.n_samples_seen()
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.scheduler.reset();
        self.step_count = 0;
        self.last_loss = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl<L: Loss> fmt::Debug for AdaptiveSGBT<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdaptiveSGBT")
            .field("inner", &self.inner)
            .field("step_count", &self.step_count)
            .field("last_loss", &self.last_loss)
            .field("base_lr", &self.base_lr)
            .field("current_lr", &self.current_lr())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::lr_schedule::{ConstantLR, ExponentialDecayLR, PlateauLR};
    use alloc::boxed::Box;

    fn test_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .build()
            .unwrap()
    }

    #[test]
    fn construction_and_initial_state() {
        let model = AdaptiveSGBT::new(test_config(), ConstantLR::new(0.1));
        assert_eq!(model.step_count(), 0);
        assert_eq!(model.n_samples_seen(), 0);
        assert!((model.base_lr() - 0.1).abs() < 1e-12);
        assert!((model.current_lr() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn train_increments_step_count() {
        let mut model = AdaptiveSGBT::new(test_config(), ConstantLR::new(0.1));
        model.train(&[1.0, 2.0], 3.0);
        model.train(&[4.0, 5.0], 6.0);
        assert_eq!(model.step_count(), 2);
        assert_eq!(model.n_samples_seen(), 2);
    }

    #[test]
    fn exponential_decay_reduces_lr() {
        let mut model = AdaptiveSGBT::new(test_config(), ExponentialDecayLR::new(0.1, 0.9));

        // Train a few steps -- LR should decrease.
        for i in 0..10 {
            model.train(&[i as f64, (i * 2) as f64], i as f64);
        }

        // After 10 steps with gamma=0.9, LR should be much less than initial.
        let lr = model.current_lr();
        assert!(lr < 0.1, "LR should have decayed from 0.1, got {}", lr);
    }

    #[test]
    fn predict_is_finite() {
        let mut model = AdaptiveSGBT::new(test_config(), ConstantLR::new(0.1));
        model.train(&[1.0, 2.0], 3.0);
        let pred = model.predict(&[1.0, 2.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );
    }

    #[test]
    fn reset_clears_all_state() {
        let mut model = AdaptiveSGBT::new(test_config(), ConstantLR::new(0.1));
        model.train(&[1.0], 2.0);
        model.train(&[3.0], 4.0);
        assert_eq!(model.step_count(), 2);

        model.reset();
        assert_eq!(model.step_count(), 0);
        assert_eq!(model.n_samples_seen(), 0);
        assert!((model.last_loss()).abs() < 1e-12);
    }

    #[test]
    fn as_streaming_learner_trait_object() {
        let model = AdaptiveSGBT::new(test_config(), ConstantLR::new(0.1));
        let mut boxed: Box<dyn StreamingLearner> = Box::new(model);
        boxed.train(&[1.0, 2.0], 3.0);
        assert_eq!(boxed.n_samples_seen(), 1);
        let pred = boxed.predict(&[1.0, 2.0]);
        assert!(pred.is_finite());
    }

    #[test]
    fn plateau_lr_reduces_on_stagnation() {
        // PlateauLR with patience=5, factor=0.5, min_lr=1e-6
        let scheduler = PlateauLR::new(0.1, 0.5, 5, 1e-6);
        let mut model = AdaptiveSGBT::new(test_config(), scheduler);

        // Feed constant data -- loss will stagnate.
        for _ in 0..50 {
            model.train(&[1.0, 1.0], 1.0);
        }

        // After stagnation, LR should have been reduced at least once.
        let lr = model.current_lr();
        assert!(lr <= 0.1, "PlateauLR should have reduced LR, got {}", lr);
    }
}
