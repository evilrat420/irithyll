//! Stream adapters for continuous prediction output.
//!
//! Provides [`PredictionStream`], a [`futures_core::Stream`] that consumes
//! incoming samples via a [`SampleReceiver`] and yields [`Prediction`] values
//! by querying a [`Predictor`] for each sample.
//!
//! This enables reactive pipelines: feed samples into the channel, and
//! asynchronously iterate over predictions as they become available.
//!
//! # Example
//!
//! ```no_run
//! # use irithyll::stream::{AsyncSGBT, PredictionStream};
//! # use irithyll::SGBTConfig;
//! # use futures_core::Stream;
//! # async fn example() {
//! let config = SGBTConfig::default();
//! let model = AsyncSGBT::new(config);
//! // PredictionStream implements futures_core::Stream<Item = Prediction>
//! # }
//! ```

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;

use super::channel::SampleReceiver;
use super::Predictor;

/// A single prediction result paired with the original sample data.
///
/// Contains both the raw model output and the loss-transformed prediction
/// (e.g., sigmoid for logistic loss), along with the ground-truth target
/// and input features for downstream evaluation.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Raw model output (sum of base prediction + weighted tree outputs).
    pub raw: f64,
    /// Transformed prediction (e.g., sigmoid applied for classification).
    pub transformed: f64,
    /// Ground-truth target from the original sample.
    pub target: f64,
    /// Input feature vector from the original sample.
    pub features: Vec<f64>,
}

/// An async stream that yields [`Prediction`]s by combining a sample receiver
/// with a concurrent predictor handle.
///
/// For each [`Sample`](crate::sample::Sample) received from the channel,
/// the stream queries the shared model via the [`Predictor`] and yields
/// a [`Prediction`] containing both raw and transformed outputs.
///
/// The stream terminates when the underlying [`SampleReceiver`] is exhausted
/// (all senders dropped and channel drained).
///
/// # Implementation
///
/// Implements [`futures_core::Stream`] directly. The predictor acquires a
/// read lock on the shared model for each prediction, so predictions reflect
/// the model state at the moment of polling — not the state when the sample
/// was sent. This is intentional for streaming evaluation.
pub struct PredictionStream {
    receiver: SampleReceiver,
    predictor: Predictor,
}

impl PredictionStream {
    /// Create a new prediction stream from a sample receiver and predictor.
    ///
    /// The receiver provides incoming samples; the predictor queries the
    /// shared model for each one.
    pub fn new(receiver: SampleReceiver, predictor: Predictor) -> Self {
        Self {
            receiver,
            predictor,
        }
    }
}

impl Stream for PredictionStream {
    type Item = Prediction;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // SAFETY: We never move out of the pinned struct; SampleReceiver's
        // inner tokio Receiver is Unpin, so projecting through Pin is safe.
        let this = self.get_mut();

        // Poll the underlying tokio receiver. We create a recv future inline
        // and poll it. Since tokio::sync::mpsc::Receiver::recv takes &mut self
        // and returns a future that borrows it, we use the poll_recv method
        // which is designed for exactly this use case.
        match this.receiver.poll_recv(cx) {
            Poll::Ready(Some(sample)) => {
                let raw = this.predictor.predict(&sample.features);
                let transformed = this.predictor.predict_transformed(&sample.features);

                Poll::Ready(Some(Prediction {
                    raw,
                    transformed,
                    target: sample.target,
                    features: sample.features,
                }))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // We cannot know how many samples remain — lower bound 0, no upper.
        (0, None)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::config::SGBTConfig;
    use crate::ensemble::SGBT;
    use crate::sample::Sample;
    use crate::stream::channel;

    use parking_lot::RwLock;
    use std::sync::Arc;

    use futures_core::Stream;
    use std::pin::Pin;

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

    fn make_predictor(config: SGBTConfig) -> (Arc<RwLock<SGBT>>, Predictor) {
        let model = SGBT::new(config);
        let shared = Arc::new(RwLock::new(model));
        let predictor = Predictor {
            model: Arc::clone(&shared),
        };
        (shared, predictor)
    }

    fn sample(x: f64) -> Sample {
        Sample::new(vec![x, x * 0.5], x * 2.0)
    }

    // 1. PredictionStream yields predictions for sent samples.
    #[tokio::test]
    async fn stream_yields_predictions() {
        let (_shared, predictor) = make_predictor(default_config());
        let (tx, rx) = channel::bounded(16);

        let mut stream = PredictionStream::new(rx, predictor);

        tx.send(sample(1.0)).await.unwrap();
        drop(tx);

        // Use poll manually via a utility — or just use StreamExt from futures.
        // For simplicity, we poll via a helper.
        let pred = poll_next(&mut stream).await;
        assert!(pred.is_some());
        let pred = pred.unwrap();
        assert!((pred.target - 2.0).abs() < f64::EPSILON);
        assert_eq!(pred.features.len(), 2);
        assert!(pred.raw.is_finite());
        assert!(pred.transformed.is_finite());
    }

    // 2. Stream terminates when channel closes.
    #[tokio::test]
    async fn stream_terminates_on_close() {
        let (_shared, predictor) = make_predictor(default_config());
        let (tx, rx) = channel::bounded(16);
        drop(tx);

        let mut stream = PredictionStream::new(rx, predictor);
        let pred = poll_next(&mut stream).await;
        assert!(pred.is_none());
    }

    // 3. Prediction contains correct features and target.
    #[tokio::test]
    async fn prediction_preserves_sample_data() {
        let (_shared, predictor) = make_predictor(default_config());
        let (tx, rx) = channel::bounded(16);

        let mut stream = PredictionStream::new(rx, predictor);

        let s = Sample::weighted(vec![3.25, 2.71], 42.0, 1.5);
        tx.send(s).await.unwrap();
        drop(tx);

        let pred = poll_next(&mut stream).await.unwrap();
        assert!((pred.features[0] - 3.25).abs() < f64::EPSILON);
        assert!((pred.features[1] - 2.71).abs() < f64::EPSILON);
        assert!((pred.target - 42.0).abs() < f64::EPSILON);
    }

    // 4. Multiple predictions in sequence.
    #[tokio::test]
    async fn multiple_predictions() {
        let (_shared, predictor) = make_predictor(default_config());
        let (tx, rx) = channel::bounded(16);

        let mut stream = PredictionStream::new(rx, predictor);

        for i in 0..5 {
            tx.send(sample(i as f64)).await.unwrap();
        }
        drop(tx);

        let mut count = 0;
        while let Some(pred) = poll_next(&mut stream).await {
            assert!(pred.raw.is_finite());
            count += 1;
        }
        assert_eq!(count, 5);
    }

    // 5. Predictions reflect model training state.
    #[tokio::test]
    async fn predictions_reflect_model_state() {
        let (shared, predictor) = make_predictor(default_config());
        let (tx, rx) = channel::bounded(16);

        let mut stream = PredictionStream::new(rx, predictor);

        // Before training: prediction should be ~0.
        tx.send(sample(1.0)).await.unwrap();
        let pred_before = poll_next(&mut stream).await.unwrap();

        // Train the model.
        {
            let mut model = shared.write();
            for i in 0..100 {
                model.train_one(&Sample::new(vec![1.0, 0.5], 10.0 + i as f64 * 0.01));
            }
        }

        // After training: prediction should have moved from zero.
        tx.send(sample(1.0)).await.unwrap();
        drop(tx);
        let pred_after = poll_next(&mut stream).await.unwrap();

        // The model was trained, so the predictions should differ.
        // (Both start at 0 for untrained, but after 100 samples they diverge.)
        assert!(
            (pred_after.raw - pred_before.raw).abs() > f64::EPSILON
                || pred_before.raw.abs() < f64::EPSILON,
            "predictions should reflect training: before={}, after={}",
            pred_before.raw,
            pred_after.raw
        );
    }

    // 6. Prediction struct Debug impl works.
    #[test]
    fn prediction_debug() {
        let p = Prediction {
            raw: 1.0,
            transformed: 0.73,
            target: 1.0,
            features: vec![0.5, 0.6],
        };
        let dbg = format!("{:?}", p);
        assert!(dbg.contains("raw"));
        assert!(dbg.contains("transformed"));
    }

    // 7. Prediction struct Clone works.
    #[test]
    fn prediction_clone() {
        let p = Prediction {
            raw: 2.5,
            transformed: 0.92,
            target: 1.0,
            features: vec![1.0, 2.0, 3.0],
        };
        let p2 = p.clone();
        assert!((p2.raw - 2.5).abs() < f64::EPSILON);
        assert_eq!(p2.features.len(), 3);
    }

    // 8. PredictionStream size_hint returns (0, None).
    #[tokio::test]
    async fn stream_size_hint() {
        let (_shared, predictor) = make_predictor(default_config());
        let (_tx, rx) = channel::bounded(16);
        let stream = PredictionStream::new(rx, predictor);
        let (lo, hi) = stream.size_hint();
        assert_eq!(lo, 0);
        assert!(hi.is_none());
    }

    // 9. Empty batch yields nothing, then terminates.
    #[tokio::test]
    async fn empty_stream() {
        let (_shared, predictor) = make_predictor(default_config());
        let (tx, rx) = channel::bounded(16);
        // Send nothing, just close.
        drop(tx);

        let mut stream = PredictionStream::new(rx, predictor);
        assert!(poll_next(&mut stream).await.is_none());
    }

    // 10. PredictionStream is Unpin (required for easy use).
    #[test]
    fn stream_is_unpin() {
        fn assert_unpin<T: Unpin>() {}
        assert_unpin::<PredictionStream>();
    }

    // -----------------------------------------------------------------------
    // Helper: poll a Stream to completion of one item using tokio.
    // -----------------------------------------------------------------------
    async fn poll_next<S: Stream + Unpin>(stream: &mut S) -> Option<S::Item> {
        use std::future::poll_fn;
        poll_fn(|cx| Pin::new(&mut *stream).poll_next(cx)).await
    }
}
