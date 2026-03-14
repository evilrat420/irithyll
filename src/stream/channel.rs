//! Bounded channel adapters for backpressure-aware sample ingestion.
//!
//! Provides [`SampleSender`] and [`SampleReceiver`], thin wrappers around
//! tokio's bounded mpsc channel that surface irithyll-typed errors and
//! integrate cleanly with the async training loop in [`super::AsyncSGBT`].
//!
//! The bounded channel enforces backpressure: if the training loop falls
//! behind, senders will await until capacity is available, preventing
//! unbounded memory growth from fast data sources.

use std::task::{Context, Poll};

use crate::error::{IrithyllError, Result};
use crate::sample::Sample;
use tokio::sync::mpsc;

/// A clonable sender handle for streaming [`Sample`]s into the training loop.
///
/// Wraps a [`tokio::sync::mpsc::Sender<Sample>`] and maps channel errors to
/// [`IrithyllError::ChannelClosed`]. Clone this freely to feed samples from
/// multiple async tasks.
///
/// # Example
///
/// ```no_run
/// # use irithyll::sample::Sample;
/// # async fn example(sender: irithyll::stream::SampleSender) -> irithyll::error::Result<()> {
/// sender.send(Sample::new(vec![1.0, 2.0], 3.0)).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct SampleSender {
    inner: mpsc::Sender<Sample>,
}

impl SampleSender {
    /// Create a new sender from a tokio mpsc sender.
    pub(crate) fn new(inner: mpsc::Sender<Sample>) -> Self {
        Self { inner }
    }

    /// Send a single sample into the training channel.
    ///
    /// Awaits if the channel is full (backpressure). Returns
    /// [`IrithyllError::ChannelClosed`] if the receiver has been dropped.
    pub async fn send(&self, sample: Sample) -> Result<()> {
        self.inner
            .send(sample)
            .await
            .map_err(|_| IrithyllError::ChannelClosed)
    }

    /// Send a batch of samples sequentially into the training channel.
    ///
    /// Each sample is sent in order; backpressure applies per-sample.
    /// Returns [`IrithyllError::ChannelClosed`] if the receiver is dropped
    /// before all samples are sent.
    pub async fn send_batch(&self, samples: Vec<Sample>) -> Result<()> {
        for sample in samples {
            self.send(sample).await?;
        }
        Ok(())
    }

    /// Returns `true` if the receiver has been dropped.
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

/// The receiving end of the sample channel, consumed by the training loop.
///
/// Wraps a [`tokio::sync::mpsc::Receiver<Sample>`]. Not clonable -- only
/// one consumer (the [`AsyncSGBT`](super::AsyncSGBT) training loop) should
/// own this.
#[derive(Debug)]
pub struct SampleReceiver {
    inner: mpsc::Receiver<Sample>,
}

impl SampleReceiver {
    /// Create a new receiver from a tokio mpsc receiver.
    pub(crate) fn new(inner: mpsc::Receiver<Sample>) -> Self {
        Self { inner }
    }

    /// Receive the next sample from the channel.
    ///
    /// Returns `None` when all senders have been dropped and the channel
    /// is drained -- this is the clean shutdown signal for the training loop.
    pub async fn recv(&mut self) -> Option<Sample> {
        self.inner.recv().await
    }

    /// Poll for the next sample without creating an intermediate future.
    ///
    /// This is the non-async counterpart of [`recv`](Self::recv), designed
    /// for use inside manual [`Stream`](futures_core::Stream) implementations
    /// like [`PredictionStream`](super::adapters::PredictionStream).
    pub(crate) fn poll_recv(&mut self, cx: &mut Context<'_>) -> Poll<Option<Sample>> {
        self.inner.poll_recv(cx)
    }
}

/// Create a bounded sample channel with the given capacity.
///
/// Returns a `(SampleSender, SampleReceiver)` pair. The sender is clonable;
/// the receiver is not.
pub(crate) fn bounded(capacity: usize) -> (SampleSender, SampleReceiver) {
    let (tx, rx) = mpsc::channel(capacity);
    (SampleSender::new(tx), SampleReceiver::new(rx))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::Sample;

    fn sample(x: f64) -> Sample {
        Sample::new(vec![x], x * 2.0)
    }

    // 1. Basic send/recv round-trip.
    #[tokio::test]
    async fn send_recv_round_trip() {
        let (tx, mut rx) = bounded(16);
        tx.send(sample(1.0)).await.unwrap();
        let s = rx.recv().await.unwrap();
        assert!((s.features[0] - 1.0).abs() < f64::EPSILON);
        assert!((s.target - 2.0).abs() < f64::EPSILON);
    }

    // 2. Recv returns None after all senders are dropped.
    #[tokio::test]
    async fn recv_none_on_closed() {
        let (tx, mut rx) = bounded(16);
        drop(tx);
        assert!(rx.recv().await.is_none());
    }

    // 3. Send fails after receiver is dropped.
    #[tokio::test]
    async fn send_fails_on_closed_receiver() {
        let (tx, rx) = bounded(16);
        drop(rx);
        let result = tx.send(sample(1.0)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            IrithyllError::ChannelClosed => {}
            other => panic!("expected ChannelClosed, got {:?}", other),
        }
    }

    // 4. is_closed reflects receiver state.
    #[tokio::test]
    async fn is_closed_reflects_state() {
        let (tx, rx) = bounded(16);
        assert!(!tx.is_closed());
        drop(rx);
        assert!(tx.is_closed());
    }

    // 5. send_batch sends all samples in order.
    #[tokio::test]
    async fn send_batch_preserves_order() {
        let (tx, mut rx) = bounded(16);
        let batch: Vec<Sample> = (0..5).map(|i| sample(i as f64)).collect();
        tx.send_batch(batch).await.unwrap();
        drop(tx);

        let mut received = Vec::new();
        while let Some(s) = rx.recv().await {
            received.push(s);
        }
        assert_eq!(received.len(), 5);
        for (i, s) in received.iter().enumerate() {
            assert!((s.features[0] - i as f64).abs() < f64::EPSILON);
        }
    }

    // 6. send_batch fails partway if receiver drops mid-batch.
    #[tokio::test]
    async fn send_batch_fails_on_mid_drop() {
        let (tx, rx) = bounded(2);
        drop(rx);
        let batch: Vec<Sample> = (0..10).map(|i| sample(i as f64)).collect();
        let result = tx.send_batch(batch).await;
        assert!(result.is_err());
    }

    // 7. Cloned sender works independently.
    #[tokio::test]
    async fn cloned_sender_works() {
        let (tx, mut rx) = bounded(16);
        let tx2 = tx.clone();

        tx.send(sample(1.0)).await.unwrap();
        tx2.send(sample(2.0)).await.unwrap();

        let s1 = rx.recv().await.unwrap();
        let s2 = rx.recv().await.unwrap();
        assert!((s1.features[0] - 1.0).abs() < f64::EPSILON);
        assert!((s2.features[0] - 2.0).abs() < f64::EPSILON);
    }

    // 8. Bounded channel applies backpressure (capacity=1).
    #[tokio::test]
    async fn bounded_backpressure() {
        let (tx, mut rx) = bounded(1);

        // Fill the single slot.
        tx.send(sample(1.0)).await.unwrap();

        // Spawn a task that sends another -- it should block until we recv.
        let tx_clone = tx.clone();
        let handle = tokio::spawn(async move {
            tx_clone.send(sample(2.0)).await.unwrap();
        });

        // Small yield to let the spawned task attempt the send.
        tokio::task::yield_now().await;

        // Drain the first, unblocking the sender.
        let s1 = rx.recv().await.unwrap();
        assert!((s1.features[0] - 1.0).abs() < f64::EPSILON);

        handle.await.unwrap();

        let s2 = rx.recv().await.unwrap();
        assert!((s2.features[0] - 2.0).abs() < f64::EPSILON);
    }

    // 9. Channel with large capacity handles burst.
    #[tokio::test]
    async fn large_burst() {
        let (tx, mut rx) = bounded(1024);
        for i in 0..1000 {
            tx.send(sample(i as f64)).await.unwrap();
        }
        drop(tx);

        let mut count = 0u64;
        while rx.recv().await.is_some() {
            count += 1;
        }
        assert_eq!(count, 1000);
    }

    // 10. SampleSender is Send + Sync (compile-time check).
    #[test]
    fn sender_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SampleSender>();
    }
}
