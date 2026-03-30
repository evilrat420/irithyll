//! Generic drift-aware wrapper for any [`StreamingLearner`].
//!
//! [`DriftAware`] wraps a **concrete** streaming learner with automatic
//! drift detection and reset, preserving the inner type for zero-cost
//! delegation and typed access.
//!
//! Unlike [`ContinualLearner`](crate::continual::ContinualLearner) which
//! type-erases the inner model behind `Box<dyn StreamingLearner>`,
//! `DriftAware<M>` is monomorphic -- the compiler can inline and
//! devirtualize all delegation calls.
//!
//! # Prequential Protocol
//!
//! On every `train_one` call the wrapper:
//!
//! 1. **Predicts** first (before the model has seen this sample).
//! 2. Computes absolute prediction error `|pred - target|`.
//! 3. Feeds the error to the drift detector.
//! 4. If the detector signals [`DriftSignal::Drift`]: resets the inner
//!    model **and** replaces the detector with a fresh instance (via the
//!    factory closure).
//! 5. Trains the inner model on the sample (whether or not a reset
//!    occurred).
//!
//! This turns any stationary learner into a drift-aware streaming learner.

use std::fmt;

use irithyll_core::drift::{DriftDetector, DriftSignal};
use irithyll_core::learner::StreamingLearner;

// ---------------------------------------------------------------------------
// DriftAware
// ---------------------------------------------------------------------------

/// Wraps any [`StreamingLearner`] with automatic drift detection and reset.
///
/// On each `train_one()`, the wrapper:
/// 1. Predicts (test-then-train protocol)
/// 2. Feeds prediction error to the drift detector
/// 3. On [`DriftSignal::Drift`]: resets the inner model and detector
/// 4. Trains the inner model
///
/// This turns any stationary learner into a drift-aware streaming learner.
///
/// # Type Parameter
///
/// * `M` -- any concrete type implementing [`StreamingLearner`]. The type is
///   preserved so callers can access model-specific methods via
///   [`inner()`](Self::inner) / [`inner_mut()`](Self::inner_mut).
///
/// # Example
///
/// ```
/// use irithyll::drift::drift_aware::DriftAware;
/// use irithyll::learner::{SGBTLearner, StreamingLearner};
/// use irithyll::SGBTConfig;
///
/// let config = SGBTConfig::builder()
///     .n_steps(5)
///     .learning_rate(0.1)
///     .build()
///     .unwrap();
/// let mut da = DriftAware::with_ddm(SGBTLearner::from_config(config));
///
/// for i in 0..100 {
///     da.train(&[i as f64], i as f64 * 0.5);
/// }
/// assert_eq!(da.n_samples_seen(), 100);
/// ```
pub struct DriftAware<M: StreamingLearner> {
    inner: M,
    detector: Box<dyn DriftDetector>,
    /// Factory to create fresh detectors on reset.
    detector_factory: Box<dyn Fn() -> Box<dyn DriftDetector> + Send + Sync>,
    n_drifts: u64,
    n_samples: u64,
}

// ---------------------------------------------------------------------------
// Constructors and accessors
// ---------------------------------------------------------------------------

impl<M: StreamingLearner> DriftAware<M> {
    /// Wrap a streaming learner with an existing boxed drift detector.
    ///
    /// The detector's [`clone_fresh`](DriftDetector::clone_fresh) method is
    /// used as the factory for creating replacement detectors after drift.
    ///
    /// # Arguments
    ///
    /// * `model` -- the inner streaming learner to wrap.
    /// * `detector` -- a boxed drift detector to monitor prediction errors.
    pub fn new(model: M, detector: Box<dyn DriftDetector>) -> Self {
        // Capture a fresh copy of the detector for the factory closure.
        let factory_seed = detector.clone_fresh();
        let detector_factory: Box<dyn Fn() -> Box<dyn DriftDetector> + Send + Sync> =
            Box::new(move || factory_seed.clone_fresh());

        Self {
            inner: model,
            detector,
            detector_factory,
            n_drifts: 0,
            n_samples: 0,
        }
    }

    /// Convenience constructor that attaches a default
    /// [`Ddm`](irithyll_core::drift::ddm::Ddm) detector.
    ///
    /// Uses `Ddm::new()` which defaults to `warning_level = 2.0`,
    /// `drift_level = 3.0`, `min_instances = 30`.
    pub fn with_ddm(model: M) -> Self {
        use irithyll_core::drift::ddm::Ddm;

        let detector: Box<dyn DriftDetector> = Box::new(Ddm::new());
        let detector_factory: Box<dyn Fn() -> Box<dyn DriftDetector> + Send + Sync> =
            Box::new(|| Box::new(Ddm::new()));

        Self {
            inner: model,
            detector,
            detector_factory,
            n_drifts: 0,
            n_samples: 0,
        }
    }

    /// Number of drift events detected since creation or last reset.
    #[inline]
    pub fn n_drifts(&self) -> u64 {
        self.n_drifts
    }

    /// Immutable reference to the inner streaming learner.
    #[inline]
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Mutable reference to the inner streaming learner.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    /// Consume the wrapper and return the inner streaming learner.
    #[inline]
    pub fn into_inner(self) -> M {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

impl<M: StreamingLearner> StreamingLearner for DriftAware<M> {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // Step 1: Prequential prediction (before this sample updates the model).
        let pred = self.inner.predict(features);

        // Step 2: Feed absolute prediction error to drift detector.
        let error = (pred - target).abs();
        let signal = self.detector.update(error);

        // Step 3: Handle drift -- reset both inner model and detector.
        if signal == DriftSignal::Drift {
            self.n_drifts += 1;
            self.inner.reset();
            self.detector = (self.detector_factory)();
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
        self.detector = (self.detector_factory)();
        self.n_samples = 0;
        self.n_drifts = 0;
    }

    fn diagnostics_array(&self) -> [f64; 5] {
        self.inner.diagnostics_array()
    }

    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64) {
        self.inner.adjust_config(lr_multiplier, lambda_delta);
    }

    fn apply_structural_change(&mut self, depth_delta: i32, steps_delta: i32) {
        self.inner.apply_structural_change(depth_delta, steps_delta);
    }

    fn replacement_count(&self) -> u64 {
        self.n_drifts
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl<M: StreamingLearner> crate::automl::DiagnosticSource for DriftAware<M> {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        // Delegate to inner if it implements DiagnosticSource via the
        // diagnostics_array() StreamingLearner method.
        let arr = self.inner.diagnostics_array();
        // All zeros means the inner model does not provide diagnostics.
        if arr == [0.0; 5] {
            return None;
        }
        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: arr[0],
            regularization_sensitivity: arr[1],
            depth_sufficiency: arr[2],
            effective_dof: arr[3],
            uncertainty: arr[4],
        })
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

impl<M: StreamingLearner + fmt::Debug> fmt::Debug for DriftAware<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DriftAware")
            .field("inner", &self.inner)
            .field("n_drifts", &self.n_drifts)
            .field("n_samples", &self.n_samples)
            .finish_non_exhaustive()
    }
}
