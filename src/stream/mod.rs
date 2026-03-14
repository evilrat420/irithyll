//! Async streaming infrastructure for tokio-native sample ingestion.
//!
//! This module provides the async interface for running an [`SGBT`] model as
//! a long-lived training service. Samples arrive through a bounded channel,
//! the model trains incrementally on each one, and concurrent read-only
//! prediction access is available at all times via [`Predictor`] handles.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────┐     mpsc      ┌───────────┐
//! │ Senders  │───(bounded)──>│ AsyncSGBT │  (write lock per sample)
//! └──────────┘    channel    │  .run()    │
//!                            └─────┬─────┘
//!                                  │
//!                       Arc<RwLock<SGBT<L>>>
//!                                  │
//!                            ┌─────┴─────┐
//!                            │ Predictor │  (read lock per predict)
//!                            └───────────┘
//! ```
//!
//! # Lifecycle
//!
//! 1. Create an `AsyncSGBT` via [`new`](AsyncSGBT::new) or
//!    [`with_capacity`](AsyncSGBT::with_capacity).
//! 2. Clone sender handles via [`sender`](AsyncSGBT::sender) and predictor
//!    handles via [`predictor`](AsyncSGBT::predictor).
//! 3. Spawn [`run`](AsyncSGBT::run) (or [`run_with_callback`](AsyncSGBT::run_with_callback))
//!    on a tokio task. The loop starts by dropping its internal sender copy
//!    so that the channel closes cleanly once all external senders are dropped.
//! 4. Feed samples from any number of async tasks.
//! 5. Drop all senders to signal shutdown; the training loop drains remaining
//!    buffered samples and returns `Ok(())`.
//!
//! # Example
//!
//! ```no_run
//! use irithyll::{SGBTConfig, Sample};
//! use irithyll::stream::AsyncSGBT;
//!
//! # async fn example() -> irithyll::error::Result<()> {
//! let config = SGBTConfig::builder()
//!     .n_steps(50)
//!     .learning_rate(0.1)
//!     .build()?;
//!
//! let mut runner = AsyncSGBT::new(config);
//! let sender = runner.sender();
//! let predictor = runner.predictor();
//!
//! // Spawn the training loop.
//! let train_handle = tokio::spawn(async move { runner.run().await });
//!
//! // Feed samples from any async context.
//! sender.send(Sample::new(vec![1.0, 2.0], 3.0)).await?;
//!
//! // Predict concurrently while training proceeds.
//! let pred = predictor.predict(&[1.0, 2.0]);
//!
//! // Drop sender to signal shutdown; training loop returns Ok(()).
//! drop(sender);
//! train_handle.await.unwrap()?;
//! # Ok(())
//! # }
//! ```

pub mod adapters;
pub mod channel;

use std::fmt;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::debug;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::SGBT;
use crate::error::Result;
use crate::loss::squared::SquaredLoss;
use crate::loss::Loss;

pub use adapters::{Prediction, PredictionStream};
pub use channel::{SampleReceiver, SampleSender};

/// Default bounded channel capacity when none is specified.
const DEFAULT_CHANNEL_CAPACITY: usize = 1024;

// ---------------------------------------------------------------------------
// Predictor
// ---------------------------------------------------------------------------

/// A concurrent, read-only prediction handle to a shared [`SGBT`] model.
///
/// Obtained via [`AsyncSGBT::predictor`]. Each prediction acquires a read lock
/// on the underlying `RwLock<SGBT<L>>`, allowing multiple predictors to operate
/// concurrently and in parallel with the training loop (which holds a write
/// lock only briefly per sample).
///
/// `Predictor` is `Clone`, `Send`, and `Sync` -- share it freely across tasks.
pub struct Predictor<L: Loss = SquaredLoss> {
    pub(crate) model: Arc<RwLock<SGBT<L>>>,
}

// Manual Clone impl -- cloning the Arc doesn't require L: Clone.
impl<L: Loss> Clone for Predictor<L> {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
        }
    }
}

impl<L: Loss> fmt::Debug for Predictor<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Predictor")
            .field("n_samples_seen", &self.model.read().n_samples_seen())
            .finish()
    }
}

impl<L: Loss> Predictor<L> {
    /// Predict the raw model output for a feature vector.
    ///
    /// Acquires a read lock on the shared model. Returns the unscaled
    /// ensemble prediction (base + weighted sum of tree outputs).
    #[inline]
    pub fn predict(&self, features: &[f64]) -> f64 {
        self.model.read().predict(features)
    }

    /// Predict with the loss function's transform applied.
    ///
    /// For regression (squared loss) this is identity; for binary
    /// classification (logistic loss) this applies the sigmoid.
    #[inline]
    pub fn predict_transformed(&self, features: &[f64]) -> f64 {
        self.model.read().predict_transformed(features)
    }

    /// Number of samples the model has been trained on so far.
    #[inline]
    pub fn n_samples_seen(&self) -> u64 {
        self.model.read().n_samples_seen()
    }

    /// Whether the model's base prediction has been initialized.
    ///
    /// Returns `false` until enough initial samples have been collected
    /// (typically 50) to compute the base constant.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.model.read().is_initialized()
    }
}

// ---------------------------------------------------------------------------
// AsyncSGBT
// ---------------------------------------------------------------------------

/// Async wrapper around [`SGBT`] for tokio-native streaming training.
///
/// `AsyncSGBT` owns the shared model and the receiving end of a bounded
/// sample channel. Call [`run`](Self::run) to start the training loop,
/// which consumes samples from the channel and trains incrementally.
///
/// Generic over `L: Loss` so the training loop benefits from monomorphized
/// gradient/hessian dispatch (no vtable overhead).
///
/// Prediction handles ([`Predictor`]) and sender handles ([`SampleSender`])
/// can be obtained before starting the loop and used concurrently from
/// other tasks.
///
/// # Shutdown
///
/// When [`run`](Self::run) is called, it drops the internal sender copy
/// so that the channel closes as soon as all external senders are dropped.
/// The loop then drains any remaining buffered samples and returns `Ok(())`.
pub struct AsyncSGBT<L: Loss = SquaredLoss> {
    /// Shared model, protected by a parking_lot RwLock.
    model: Arc<RwLock<SGBT<L>>>,
    /// Receiving end of the sample channel.
    receiver: Option<SampleReceiver>,
    /// Sending end, kept so callers can clone it via `sender()`.
    /// Wrapped in Option so `run()` can drop it before entering the loop,
    /// ensuring the channel closes when all external senders are dropped.
    sender: Option<SampleSender>,
}

impl AsyncSGBT<SquaredLoss> {
    /// Create a new async SGBT runner with the default channel capacity (1024).
    ///
    /// Uses squared loss (regression). For other loss functions, use
    /// [`with_loss`](AsyncSGBT::with_loss) or [`with_loss_and_capacity`](AsyncSGBT::with_loss_and_capacity).
    pub fn new(config: SGBTConfig) -> Self {
        Self::with_capacity(config, DEFAULT_CHANNEL_CAPACITY)
    }

    /// Create a new async SGBT runner with a custom channel capacity.
    ///
    /// Uses squared loss (regression).
    pub fn with_capacity(config: SGBTConfig, capacity: usize) -> Self {
        let model = SGBT::new(config);
        let shared = Arc::new(RwLock::new(model));
        let (sender, receiver) = channel::bounded(capacity);

        Self {
            model: shared,
            receiver: Some(receiver),
            sender: Some(sender),
        }
    }
}

impl<L: Loss> AsyncSGBT<L> {
    /// Create a new async SGBT runner with a specific loss function.
    ///
    /// ```no_run
    /// use irithyll::SGBTConfig;
    /// use irithyll::stream::AsyncSGBT;
    /// use irithyll::loss::logistic::LogisticLoss;
    ///
    /// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
    /// let runner = AsyncSGBT::with_loss(config, LogisticLoss);
    /// ```
    pub fn with_loss(config: SGBTConfig, loss: L) -> Self {
        Self::with_loss_and_capacity(config, loss, DEFAULT_CHANNEL_CAPACITY)
    }

    /// Create a new async SGBT runner with a specific loss and channel capacity.
    pub fn with_loss_and_capacity(config: SGBTConfig, loss: L, capacity: usize) -> Self {
        let model = SGBT::with_loss(config, loss);
        let shared = Arc::new(RwLock::new(model));
        let (sender, receiver) = channel::bounded(capacity);

        Self {
            model: shared,
            receiver: Some(receiver),
            sender: Some(sender),
        }
    }

    /// Obtain a clonable sender handle for feeding samples into the channel.
    ///
    /// Multiple senders can be created (via `Clone`) and used from different
    /// async tasks. The training loop runs until all external senders are
    /// dropped.
    ///
    /// # Panics
    ///
    /// Panics if called after [`run`](Self::run) has already started, since
    /// the internal sender is consumed at that point.
    pub fn sender(&self) -> SampleSender {
        self.sender
            .as_ref()
            .expect("sender() called after run() consumed the internal sender")
            .clone()
    }

    /// Obtain a concurrent prediction handle to the shared model.
    ///
    /// The predictor can be cloned and used from any thread or task while
    /// the training loop is running.
    pub fn predictor(&self) -> Predictor<L> {
        Predictor {
            model: Arc::clone(&self.model),
        }
    }

    /// Run the main training loop.
    ///
    /// Receives samples from the bounded channel and trains the model
    /// incrementally. For each sample:
    ///
    /// 1. Acquire write lock on the shared `SGBT<L>`.
    /// 2. Call `train_one(&sample)`.
    /// 3. Release the lock.
    ///
    /// Before entering the loop, the internal sender is dropped so that the
    /// channel closes cleanly when all external senders are dropped.
    ///
    /// Returns `Ok(())` when the channel closes (all senders have been
    /// dropped and all buffered samples have been consumed).
    ///
    /// # Logging
    ///
    /// Emits a `tracing::debug!` message every 1000 samples with the
    /// current sample count.
    ///
    /// # Panics
    ///
    /// Panics if called more than once (the receiver is consumed on first call).
    pub async fn run(&mut self) -> Result<()> {
        // Drop our sender so the channel closes when external senders drop.
        self.sender.take();

        let receiver = self
            .receiver
            .take()
            .expect("run() called more than once: receiver already consumed");

        self.run_inner(receiver, None::<fn(u64)>).await
    }

    /// Run the training loop with a callback invoked after each sample.
    ///
    /// Behaves identically to [`run`](Self::run), but calls `callback`
    /// with the current `n_samples_seen()` count after training each sample.
    /// Useful for progress bars, metrics collection, or adaptive control.
    ///
    /// The callback runs synchronously within the training task -- keep it
    /// fast to avoid blocking the loop.
    ///
    /// # Panics
    ///
    /// Panics if called more than once (the receiver is consumed on first call).
    pub async fn run_with_callback<F>(&mut self, callback: F) -> Result<()>
    where
        F: Fn(u64),
    {
        // Drop our sender so the channel closes when external senders drop.
        self.sender.take();

        let receiver = self
            .receiver
            .take()
            .expect("run_with_callback() called more than once: receiver already consumed");

        self.run_inner(receiver, Some(callback)).await
    }

    /// Internal training loop shared by `run` and `run_with_callback`.
    async fn run_inner<F>(&self, mut receiver: SampleReceiver, callback: Option<F>) -> Result<()>
    where
        F: Fn(u64),
    {
        while let Some(sample) = receiver.recv().await {
            let seen;
            {
                let mut model = self.model.write();
                model.train_one(&sample);
                seen = model.n_samples_seen();
            }

            if let Some(ref cb) = callback {
                cb(seen);
            }

            if seen % 1000 == 0 {
                debug!(samples_seen = seen, "async training progress");
            }
        }

        let total = self.model.read().n_samples_seen();
        debug!(total_samples = total, "async training loop completed");

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::config::SGBTConfig;
    use crate::sample::Sample;

    use std::sync::atomic::{AtomicU64, Ordering};

    fn default_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(3)
            .n_bins(8)
            .build()
            .unwrap()
    }

    fn sample(x: f64) -> Sample {
        Sample::new(vec![x, x * 0.5], x * 2.0)
    }

    // 1. Basic lifecycle: send samples, run loop, verify training.
    #[tokio::test]
    async fn basic_lifecycle() {
        let mut runner = AsyncSGBT::new(default_config());
        let sender = runner.sender();
        let predictor = runner.predictor();

        // Initially untrained.
        assert_eq!(predictor.n_samples_seen(), 0);
        assert!(!predictor.is_initialized());

        let handle = tokio::spawn(async move { runner.run().await });

        for i in 0..20 {
            sender.send(sample(i as f64)).await.unwrap();
        }
        drop(sender);

        handle.await.unwrap().unwrap();
        assert_eq!(predictor.n_samples_seen(), 20);
    }

    // 2. Predictor works concurrently with training.
    #[tokio::test]
    async fn concurrent_predict_during_training() {
        let mut runner = AsyncSGBT::new(default_config());
        let sender = runner.sender();
        let predictor = runner.predictor();

        let pred_handle = tokio::spawn({
            let predictor = predictor.clone();
            async move {
                // Keep predicting while training runs.
                let mut predictions = Vec::new();
                for _ in 0..50 {
                    let p = predictor.predict(&[1.0, 0.5]);
                    predictions.push(p);
                    tokio::task::yield_now().await;
                }
                predictions
            }
        });

        let train_handle = tokio::spawn(async move { runner.run().await });

        for i in 0..100 {
            sender.send(sample(i as f64)).await.unwrap();
        }
        drop(sender);

        let predictions = pred_handle.await.unwrap();
        train_handle.await.unwrap().unwrap();

        // All predictions should be finite.
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    // 3. run returns Ok(()) when channel closes immediately (no samples).
    #[tokio::test]
    async fn run_returns_ok_on_empty_channel() {
        let mut runner = AsyncSGBT::new(default_config());
        let sender = runner.sender();
        // Drop the only external sender immediately.
        drop(sender);

        // run() drops the internal sender, so the channel is fully closed.
        let result = runner.run().await;
        assert!(result.is_ok());
        assert_eq!(runner.model.read().n_samples_seen(), 0);
    }

    // 4. with_capacity creates channel with specified size.
    #[tokio::test]
    async fn with_capacity_custom() {
        let mut runner = AsyncSGBT::with_capacity(default_config(), 2);
        let sender = runner.sender();

        let handle = tokio::spawn(async move { runner.run().await });

        // Channel capacity 2: should be able to send 2 without blocking.
        sender.send(sample(1.0)).await.unwrap();
        sender.send(sample(2.0)).await.unwrap();
        drop(sender);

        handle.await.unwrap().unwrap();
    }

    // 5. Multiple senders from different tasks.
    #[tokio::test]
    async fn multiple_senders() {
        let mut runner = AsyncSGBT::new(default_config());
        let sender1 = runner.sender();
        let sender2 = runner.sender();
        let predictor = runner.predictor();

        let handle = tokio::spawn(async move { runner.run().await });

        let h1 = tokio::spawn(async move {
            for i in 0..10 {
                sender1.send(sample(i as f64)).await.unwrap();
            }
        });

        let h2 = tokio::spawn(async move {
            for i in 10..20 {
                sender2.send(sample(i as f64)).await.unwrap();
            }
        });

        h1.await.unwrap();
        h2.await.unwrap();

        // Both senders dropped (moved into tasks that completed).
        // run() already dropped its internal sender, so the channel closes.
        // Wait for the training loop to drain and finish.
        handle.await.unwrap().unwrap();

        assert_eq!(predictor.n_samples_seen(), 20);
    }

    // 6. run_with_callback invokes callback for each sample.
    #[tokio::test]
    async fn run_with_callback_invokes() {
        let mut runner = AsyncSGBT::new(default_config());
        let sender = runner.sender();

        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = Arc::clone(&counter);

        let handle = tokio::spawn(async move {
            runner
                .run_with_callback(move |_seen| {
                    counter_clone.fetch_add(1, Ordering::Relaxed);
                })
                .await
        });

        for i in 0..15 {
            sender.send(sample(i as f64)).await.unwrap();
        }
        drop(sender);

        handle.await.unwrap().unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 15);
    }

    // 7. Callback receives correct sample counts.
    #[tokio::test]
    async fn callback_receives_correct_counts() {
        let mut runner = AsyncSGBT::new(default_config());
        let sender = runner.sender();

        let counts = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let counts_clone = Arc::clone(&counts);

        let handle = tokio::spawn(async move {
            runner
                .run_with_callback(move |seen| {
                    counts_clone.lock().push(seen);
                })
                .await
        });

        for i in 0..5 {
            sender.send(sample(i as f64)).await.unwrap();
        }
        drop(sender);

        handle.await.unwrap().unwrap();

        let recorded = counts.lock().clone();
        assert_eq!(recorded.len(), 5);
        // Counts should be monotonically increasing.
        for window in recorded.windows(2) {
            assert!(window[1] > window[0]);
        }
        assert_eq!(*recorded.last().unwrap(), 5);
    }

    // 8. Predictor clone is independent but sees same model.
    #[tokio::test]
    async fn predictor_clone_independent() {
        let runner = AsyncSGBT::new(default_config());
        let p1 = runner.predictor();
        let p2 = p1.clone();

        // Both should return the same prediction (same underlying model).
        let pred1 = p1.predict(&[1.0, 2.0]);
        let pred2 = p2.predict(&[1.0, 2.0]);
        assert!((pred1 - pred2).abs() < f64::EPSILON);
    }

    // 9. predict_transformed works through Predictor.
    #[tokio::test]
    async fn predictor_predict_transformed() {
        let runner = AsyncSGBT::new(default_config());
        let predictor = runner.predictor();

        // For squared loss, predict_transformed == predict (identity transform).
        let raw = predictor.predict(&[1.0, 2.0]);
        let transformed = predictor.predict_transformed(&[1.0, 2.0]);
        assert!((raw - transformed).abs() < f64::EPSILON);
    }

    // 10. Predictor is Send + Sync (compile-time check).
    #[test]
    fn predictor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Predictor>();
    }

    // 11. AsyncSGBT is Send (required for tokio::spawn).
    #[test]
    fn async_sgbt_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AsyncSGBT>();
    }

    // 12. Training actually improves predictions.
    #[tokio::test]
    async fn training_improves_predictions() {
        let mut runner = AsyncSGBT::new(default_config());
        let sender = runner.sender();
        let predictor = runner.predictor();

        let handle = tokio::spawn(async move { runner.run().await });

        // Prediction before training.
        let pred_before = predictor.predict(&[5.0, 2.5]);

        // Send consistent data: target = 10.0.
        for _ in 0..100 {
            sender
                .send(Sample::new(vec![5.0, 2.5], 10.0))
                .await
                .unwrap();
        }

        // Give the training loop time to process.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let pred_after = predictor.predict(&[5.0, 2.5]);
        drop(sender);

        handle.await.unwrap().unwrap();

        // After training on constant target 10.0, prediction should be
        // closer to 10.0 than the initial (0.0).
        assert!(
            (pred_after - 10.0).abs() < (pred_before - 10.0).abs(),
            "prediction should improve: before={}, after={}, target=10.0",
            pred_before,
            pred_after
        );
    }

    // 13. with_loss creates async runner with custom loss.
    #[tokio::test]
    async fn with_loss_creates_runner() {
        use crate::loss::logistic::LogisticLoss;

        let config = default_config();
        let mut runner = AsyncSGBT::with_loss(config, LogisticLoss);
        let sender = runner.sender();
        let predictor = runner.predictor();

        // Sigmoid(0) = 0.5 for logistic loss
        let pred = predictor.predict_transformed(&[1.0, 2.0]);
        assert!(
            (pred - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5, got {}",
            pred
        );

        let handle = tokio::spawn(async move { runner.run().await });
        drop(sender);
        handle.await.unwrap().unwrap();
    }
}
