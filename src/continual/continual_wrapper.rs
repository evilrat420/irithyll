//! Drift-aware continual learning wrapper for any [`StreamingLearner`].
//!
//! [`ContinualLearner`] wraps an opaque streaming model and monitors
//! prediction error via a pluggable [`DriftDetector`]. When drift is
//! detected the inner model is reset (or partially reset), allowing it
//! to adapt to the new data regime without accumulating stale knowledge.
//!
//! Because [`StreamingLearner`] is intentionally opaque -- no access to
//! raw parameters or gradients -- the wrapper uses **prequential error**
//! (predict-then-train) as the drift signal source. Models that expose
//! parameters can compose with `ContinualStrategy` directly; this
//! wrapper provides the outer orchestration layer.
//!
//! # Prequential Protocol
//!
//! On every `train_one` call the wrapper:
//!
//! 1. **Predicts** first (before the model has seen this sample).
//! 2. Computes absolute prediction error `|pred - target|`.
//! 3. Feeds the error to the drift detector.
//! 4. If the detector signals `Drift` and `reset_on_drift` is enabled,
//!    resets the inner model so it can re-learn from scratch.
//! 5. Trains the inner model on the sample (whether or not a reset occurred).
//!
//! This is the standard **prequential evaluation** protocol used in
//! streaming ML literature (Gama et al., 2013).

use crate::learner::StreamingLearner;
use irithyll_core::drift::{DriftDetector, DriftSignal};

use std::fmt;

// ---------------------------------------------------------------------------
// ContinualLearner
// ---------------------------------------------------------------------------

/// Wraps any [`StreamingLearner`] with drift-detected continual adaptation.
///
/// Since `StreamingLearner` is opaque (no access to raw parameters or
/// gradients), `ContinualLearner` uses prediction error to drive drift
/// detection, which triggers model reset on the underlying learner.
///
/// For models that **do** expose parameters (neural models), the
/// `ContinualStrategy` trait can be applied
/// directly. This wrapper provides the higher-level orchestration layer.
///
/// # Example
///
/// ```
/// use irithyll::continual::ContinualLearner;
/// use irithyll::{linear, StreamingLearner};
/// use irithyll_core::drift::pht::PageHinkleyTest;
///
/// let mut cl = ContinualLearner::new(linear(0.01))
///     .with_drift_detector(PageHinkleyTest::new());
///
/// for i in 0..100 {
///     cl.train(&[i as f64], i as f64 * 2.0);
/// }
/// let pred = cl.predict(&[50.0]);
/// assert!(pred.is_finite());
/// ```
pub struct ContinualLearner {
    /// The wrapped streaming model.
    inner: Box<dyn StreamingLearner>,
    /// Optional drift detector fed with prediction errors.
    drift_detector: Option<Box<dyn DriftDetector>>,
    /// Whether to reset the inner model on drift (default: true).
    reset_on_drift: bool,
    /// Total training samples seen (including across resets).
    n_samples: u64,
    /// Number of drift events detected.
    drift_count: u64,
    /// Most recent drift signal from the detector.
    last_drift_signal: DriftSignal,
}

impl ContinualLearner {
    /// Wrap a streaming learner with continual learning capabilities.
    ///
    /// The returned wrapper has no drift detector attached by default --
    /// call [`with_drift_detector`](Self::with_drift_detector) to enable
    /// drift-aware behaviour.
    pub fn new(learner: impl StreamingLearner + 'static) -> Self {
        Self {
            inner: Box::new(learner),
            drift_detector: None,
            reset_on_drift: true,
            n_samples: 0,
            drift_count: 0,
            last_drift_signal: DriftSignal::Stable,
        }
    }

    /// Wrap a boxed streaming learner.
    ///
    /// Use this when the learner is already behind a
    /// `Box<dyn StreamingLearner>`.
    pub fn from_boxed(learner: Box<dyn StreamingLearner>) -> Self {
        Self {
            inner: learner,
            drift_detector: None,
            reset_on_drift: true,
            n_samples: 0,
            drift_count: 0,
            last_drift_signal: DriftSignal::Stable,
        }
    }

    // -----------------------------------------------------------------------
    // Builder methods
    // -----------------------------------------------------------------------

    /// Attach a drift detector that monitors prediction error.
    ///
    /// The detector receives `|prediction - target|` on every training
    /// sample (prequential protocol).
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll::continual::ContinualLearner;
    /// use irithyll::linear;
    /// use irithyll_core::drift::pht::PageHinkleyTest;
    ///
    /// let cl = ContinualLearner::new(linear(0.01))
    ///     .with_drift_detector(PageHinkleyTest::new());
    /// ```
    pub fn with_drift_detector(mut self, detector: impl DriftDetector + 'static) -> Self {
        self.drift_detector = Some(Box::new(detector));
        self
    }

    /// Attach a boxed drift detector.
    pub fn with_drift_detector_boxed(mut self, detector: Box<dyn DriftDetector>) -> Self {
        self.drift_detector = Some(detector);
        self
    }

    /// Set whether the inner model is reset when drift is detected.
    ///
    /// Default: `true`. When set to `false`, the wrapper still tracks
    /// drift events and signals but does not reset the model.
    pub fn with_reset_on_drift(mut self, reset: bool) -> Self {
        self.reset_on_drift = reset;
        self
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of drift events detected since creation (or last reset).
    #[inline]
    pub fn drift_count(&self) -> u64 {
        self.drift_count
    }

    /// Most recent drift signal from the detector.
    ///
    /// Returns [`DriftSignal::Stable`] if no detector is attached or no
    /// samples have been processed.
    #[inline]
    pub fn last_signal(&self) -> DriftSignal {
        self.last_drift_signal
    }

    /// Whether the wrapper is configured to reset on drift.
    #[inline]
    pub fn reset_on_drift(&self) -> bool {
        self.reset_on_drift
    }

    /// Immutable reference to the wrapped streaming learner.
    #[inline]
    pub fn inner(&self) -> &dyn StreamingLearner {
        &*self.inner
    }

    /// Mutable reference to the wrapped streaming learner.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut dyn StreamingLearner {
        &mut *self.inner
    }

    /// Whether a drift detector is attached.
    #[inline]
    pub fn has_drift_detector(&self) -> bool {
        self.drift_detector.is_some()
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

impl StreamingLearner for ContinualLearner {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // Step 1: Prequential prediction (before this sample updates the model).
        let pred = self.inner.predict(features);

        // Step 2: Feed absolute prediction error to drift detector.
        if let Some(ref mut detector) = self.drift_detector {
            let error = (pred - target).abs();
            let signal = detector.update(error);
            self.last_drift_signal = signal;

            // Step 3: Handle drift.
            if signal == DriftSignal::Drift {
                self.drift_count += 1;

                if self.reset_on_drift {
                    self.inner.reset();
                }
            }
        }

        // Step 4: Train the inner model (always, even after reset).
        self.inner.train_one(features, target, weight);

        // Step 5: Increment our own sample counter.
        self.n_samples += 1;
    }

    #[inline]
    fn predict(&self, features: &[f64]) -> f64 {
        self.inner.predict(features)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.inner.reset();
        if let Some(ref mut detector) = self.drift_detector {
            detector.reset();
        }
        self.n_samples = 0;
        self.drift_count = 0;
        self.last_drift_signal = DriftSignal::Stable;
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

impl fmt::Debug for ContinualLearner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContinualLearner")
            .field("n_samples", &self.n_samples)
            .field("drift_count", &self.drift_count)
            .field("last_signal", &self.last_drift_signal)
            .field("reset_on_drift", &self.reset_on_drift)
            .field("has_detector", &self.drift_detector.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Factory function
// ---------------------------------------------------------------------------

/// Wrap any streaming learner with drift-detected continual adaptation.
///
/// Returns a [`ContinualLearner`] with no drift detector attached.
/// Chain [`ContinualLearner::with_drift_detector`] to enable detection.
///
/// # Example
///
/// ```
/// use irithyll::continual::continual;
/// use irithyll::{esn, StreamingLearner};
/// use irithyll_core::drift::pht::PageHinkleyTest;
///
/// let mut cl = continual(esn(50, 0.9))
///     .with_drift_detector(PageHinkleyTest::new());
///
/// for i in 0..60 {
///     cl.train(&[i as f64 * 0.1], 0.0);
/// }
/// let pred = cl.predict(&[1.0]);
/// assert!(pred.is_finite());
/// ```
pub fn continual(learner: impl StreamingLearner + 'static) -> ContinualLearner {
    ContinualLearner::new(learner)
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for ContinualLearner {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        // Cannot access inner learner diagnostics through Box<dyn StreamingLearner>.
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use irithyll_core::drift::pht::PageHinkleyTest;

    // A trivial learner for testing: tracks a running mean of targets.
    struct MeanLearner {
        sum: f64,
        count: u64,
    }

    impl MeanLearner {
        fn new() -> Self {
            Self { sum: 0.0, count: 0 }
        }
    }

    impl StreamingLearner for MeanLearner {
        fn train_one(&mut self, _features: &[f64], target: f64, _weight: f64) {
            self.sum += target;
            self.count += 1;
        }

        fn predict(&self, _features: &[f64]) -> f64 {
            if self.count == 0 {
                return 0.0;
            }
            self.sum / self.count as f64
        }

        fn n_samples_seen(&self) -> u64 {
            self.count
        }

        fn reset(&mut self) {
            self.sum = 0.0;
            self.count = 0;
        }
    }

    // MeanLearner is trivially thread-safe.
    unsafe impl Send for MeanLearner {}
    unsafe impl Sync for MeanLearner {}

    #[test]
    fn wraps_learner_transparently() {
        let mut cl = ContinualLearner::new(MeanLearner::new());

        // Train with known values.
        cl.train(&[1.0], 10.0);
        cl.train(&[2.0], 20.0);

        assert_eq!(cl.n_samples_seen(), 2);

        // Predict should return the mean of targets (from inner MeanLearner).
        let pred = cl.predict(&[0.0]);
        assert!(
            (pred - 15.0).abs() < 1e-6,
            "expected mean ~15.0, got {}",
            pred
        );
    }

    #[test]
    fn drift_detection_triggers_on_error_spike() {
        // Use a very sensitive PHT to trigger quickly.
        let pht = PageHinkleyTest::with_params(0.001, 5.0);
        let mut cl = ContinualLearner::new(MeanLearner::new()).with_drift_detector(pht);

        // Phase 1: Train on stable data (target ~ 1.0).
        // MeanLearner will converge toward 1.0, so prediction error stays small.
        for _ in 0..200 {
            cl.train(&[0.0], 1.0);
        }
        let drifts_before = cl.drift_count();

        // Phase 2: Sudden regime shift (target -> 1000.0).
        // Prediction is ~1.0 but target is 1000.0 => error ~999 => triggers drift.
        let mut detected = false;
        for _ in 0..200 {
            cl.train(&[0.0], 1000.0);
            if cl.drift_count() > drifts_before {
                detected = true;
                break;
            }
        }

        assert!(detected, "drift should be detected on sudden error spike");
    }

    #[test]
    fn drift_count_increments() {
        let pht = PageHinkleyTest::with_params(0.001, 5.0);
        let mut cl = ContinualLearner::new(MeanLearner::new()).with_drift_detector(pht);

        assert_eq!(cl.drift_count(), 0);

        // Phase 1: stable.
        for _ in 0..200 {
            cl.train(&[0.0], 1.0);
        }

        // Phase 2: shift to trigger drift.
        for _ in 0..200 {
            cl.train(&[0.0], 1000.0);
        }

        assert!(
            cl.drift_count() >= 1,
            "drift_count should be >= 1 after regime shift, got {}",
            cl.drift_count()
        );
    }

    #[test]
    fn reset_on_drift_resets_inner_model() {
        let pht = PageHinkleyTest::with_params(0.001, 5.0);
        let mut cl = ContinualLearner::new(MeanLearner::new())
            .with_drift_detector(pht)
            .with_reset_on_drift(true);

        // Phase 1: stable training.
        for _ in 0..200 {
            cl.train(&[0.0], 1.0);
        }

        // Inner model has accumulated samples.
        assert!(
            cl.inner().n_samples_seen() > 0,
            "inner should have samples before drift"
        );

        // Phase 2: trigger drift.
        for _ in 0..200 {
            cl.train(&[0.0], 1000.0);
        }

        // After drift + reset, the inner model was reset and then re-trained
        // on the post-drift samples. Its count should be less than the total.
        assert!(
            cl.inner().n_samples_seen() < cl.n_samples_seen(),
            "inner model samples ({}) should be less than total ({}) after reset",
            cl.inner().n_samples_seen(),
            cl.n_samples_seen()
        );
    }

    #[test]
    fn no_drift_detector_works_fine() {
        // No detector attached -- pure pass-through.
        let mut cl = ContinualLearner::new(MeanLearner::new());

        cl.train(&[0.0], 5.0);
        cl.train(&[0.0], 15.0);
        assert_eq!(cl.n_samples_seen(), 2);

        let pred = cl.predict(&[0.0]);
        assert!(
            (pred - 10.0).abs() < 1e-6,
            "pass-through should work without detector: got {}",
            pred
        );

        assert_eq!(cl.drift_count(), 0);
        assert_eq!(cl.last_signal(), DriftSignal::Stable);
    }

    #[test]
    fn predict_is_side_effect_free() {
        let pht = PageHinkleyTest::new();
        let mut cl = ContinualLearner::new(MeanLearner::new()).with_drift_detector(pht);

        cl.train(&[0.0], 10.0);
        let n_before = cl.n_samples_seen();
        let drift_before = cl.drift_count();
        let signal_before = cl.last_signal();

        // Multiple predictions should not change any state.
        let _ = cl.predict(&[0.0]);
        let _ = cl.predict(&[0.0]);
        let _ = cl.predict(&[0.0]);

        assert_eq!(
            cl.n_samples_seen(),
            n_before,
            "predict should not change n_samples"
        );
        assert_eq!(
            cl.drift_count(),
            drift_before,
            "predict should not change drift_count"
        );
        assert_eq!(
            cl.last_signal(),
            signal_before,
            "predict should not change last_signal"
        );
    }

    #[test]
    fn n_samples_tracks_correctly() {
        let mut cl = ContinualLearner::new(MeanLearner::new());

        assert_eq!(cl.n_samples_seen(), 0);

        for i in 1..=50 {
            cl.train(&[0.0], i as f64);
            assert_eq!(
                cl.n_samples_seen(),
                i,
                "n_samples should be {} after {} trains",
                i,
                i
            );
        }
    }

    #[test]
    fn inner_access_works() {
        let mut cl = ContinualLearner::new(MeanLearner::new());

        cl.train(&[0.0], 10.0);
        cl.train(&[0.0], 20.0);

        // inner() should reflect the model's state.
        assert_eq!(cl.inner().n_samples_seen(), 2);

        // inner_mut() should allow modification.
        cl.inner_mut().reset();
        assert_eq!(cl.inner().n_samples_seen(), 0);
    }

    #[test]
    fn reset_clears_everything() {
        let pht = PageHinkleyTest::with_params(0.001, 5.0);
        let mut cl = ContinualLearner::new(MeanLearner::new()).with_drift_detector(pht);

        // Train and trigger drift.
        for _ in 0..200 {
            cl.train(&[0.0], 1.0);
        }
        for _ in 0..200 {
            cl.train(&[0.0], 1000.0);
        }

        // Some state should have accumulated.
        assert!(cl.n_samples_seen() > 0);

        // Full reset.
        cl.reset();

        assert_eq!(
            cl.n_samples_seen(),
            0,
            "n_samples should be zero after reset"
        );
        assert_eq!(
            cl.drift_count(),
            0,
            "drift_count should be zero after reset"
        );
        assert_eq!(
            cl.last_signal(),
            DriftSignal::Stable,
            "last_signal should be Stable after reset"
        );
        assert_eq!(
            cl.inner().n_samples_seen(),
            0,
            "inner model should be reset"
        );
    }

    #[test]
    fn pipeline_composition_works() {
        use crate::pipeline::Pipeline;

        let cl = continual(MeanLearner::new());
        let mut pipeline = Pipeline::builder().learner(cl);

        pipeline.train(&[1.0, 2.0], 10.0);
        pipeline.train(&[3.0, 4.0], 20.0);

        assert_eq!(pipeline.n_samples_seen(), 2);

        let pred = pipeline.predict(&[5.0, 6.0]);
        assert!(pred.is_finite(), "pipeline prediction should be finite");
    }

    #[test]
    fn factory_function_creates_wrapper() {
        let mut cl = continual(MeanLearner::new());

        cl.train(&[0.0], 42.0);
        assert_eq!(cl.n_samples_seen(), 1);

        let pred = cl.predict(&[0.0]);
        assert!(
            (pred - 42.0).abs() < 1e-6,
            "factory-created wrapper should work: got {}",
            pred
        );
    }

    #[test]
    fn with_reset_on_drift_false_does_not_reset() {
        let pht = PageHinkleyTest::with_params(0.001, 5.0);
        let mut cl = ContinualLearner::new(MeanLearner::new())
            .with_drift_detector(pht)
            .with_reset_on_drift(false);

        // Phase 1: stable.
        for _ in 0..200 {
            cl.train(&[0.0], 1.0);
        }
        let inner_count_before_shift = cl.inner().n_samples_seen();

        // Phase 2: trigger drift (but reset is disabled).
        for _ in 0..200 {
            cl.train(&[0.0], 1000.0);
        }

        // Drift should be detected but inner model NOT reset -- so inner
        // count should equal total wrapper count (all samples accumulated).
        assert!(
            cl.drift_count() >= 1,
            "drift should still be detected even with reset_on_drift=false"
        );
        assert_eq!(
            cl.inner().n_samples_seen(),
            cl.n_samples_seen(),
            "inner model should NOT have been reset (reset_on_drift=false): inner={}, total={}",
            cl.inner().n_samples_seen(),
            cl.n_samples_seen()
        );
        assert!(
            cl.inner().n_samples_seen() > inner_count_before_shift,
            "inner should have continued accumulating samples"
        );
    }

    #[test]
    fn as_trait_object() {
        // ContinualLearner should work behind Box<dyn StreamingLearner>.
        let cl = ContinualLearner::new(MeanLearner::new());
        let mut boxed: Box<dyn StreamingLearner> = Box::new(cl);

        boxed.train(&[0.0], 7.0);
        assert_eq!(boxed.n_samples_seen(), 1);

        let pred = boxed.predict(&[0.0]);
        assert!(
            (pred - 7.0).abs() < 1e-6,
            "trait object predict should work: got {}",
            pred
        );
    }

    #[test]
    fn debug_format_is_informative() {
        let cl =
            ContinualLearner::new(MeanLearner::new()).with_drift_detector(PageHinkleyTest::new());

        let debug = format!("{:?}", cl);
        assert!(
            debug.contains("ContinualLearner"),
            "debug output should contain struct name"
        );
        assert!(
            debug.contains("drift_count"),
            "debug output should contain drift_count field"
        );
        assert!(
            debug.contains("has_detector"),
            "debug output should contain has_detector field"
        );
    }
}
