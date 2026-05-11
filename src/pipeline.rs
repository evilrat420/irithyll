//! Composable streaming pipelines for preprocessing → learning chains.
//!
//! A [`Pipeline`] chains one or more [`StreamingPreprocessor`] steps with a
//! terminal [`StreamingLearner`], producing a single unit that itself
//! implements `StreamingLearner`. This enables nested composition -- pipelines
//! can participate in stacking ensembles, be boxed as `Box<dyn StreamingLearner>`,
//! or be used anywhere a learner is expected.
//!
//! # Design
//!
//! During **training** (`train_one`), each preprocessor receives the features,
//! updates its internal statistics, and outputs transformed features for the
//! next stage. The final transformed features reach the learner.
//!
//! During **prediction** (`predict`), preprocessors only *transform* -- they do
//! **not** update their statistics. This matches the standard ML convention
//! where test-time transforms use frozen statistics.
//!
//! # Example
//!
//! ```
//! use irithyll::preprocessing::IncrementalNormalizer;
//! use irithyll::pipeline::{Pipeline, StreamingPreprocessor};
//! use irithyll::learner::StreamingLearner;
//! use irithyll::learners::StreamingLinearModel;
//!
//! // Build a normalizer → linear model pipeline.
//! let mut pipeline = Pipeline::builder()
//!     .pipe(IncrementalNormalizer::new())
//!     .learner(StreamingLinearModel::new(0.01));
//!
//! // Train through the pipeline -- normalizer updates, then learner trains.
//! pipeline.train(&[100.0, 0.5], 42.0);
//! pipeline.train(&[200.0, 1.5], 84.0);
//!
//! // Predict -- normalizer transforms (no update), then learner predicts.
//! let pred = pipeline.predict(&[150.0, 1.0]);
//! assert!(pred.is_finite());
//! ```

use crate::learner::StreamingLearner;
use crate::preprocessing::StreamingTargetPreprocessor;

// ---------------------------------------------------------------------------
// StreamingPreprocessor trait
// ---------------------------------------------------------------------------

/// Object-safe trait for streaming feature transformers.
///
/// A `StreamingPreprocessor` maintains running statistics that are updated
/// during training and applied (without update) during prediction. This
/// separation ensures test-time transforms use frozen statistics.
///
/// # Object Safety
///
/// All methods use `&self` / `&mut self` with concrete return types,
/// allowing `Box<dyn StreamingPreprocessor>` for runtime-polymorphic pipelines.
///
/// # Implementors
///
/// - [`IncrementalNormalizer`](crate::preprocessing::IncrementalNormalizer) --
///   Welford online standardization (zero-mean, unit-variance).
pub trait StreamingPreprocessor: Send + Sync {
    /// Update internal statistics from this sample and return transformed features.
    ///
    /// Called during training. The preprocessor incorporates the sample into
    /// its running statistics (e.g., mean/variance) and returns the transformed
    /// features for the next pipeline stage.
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64>;

    /// Transform features using current statistics without updating them.
    ///
    /// Called during prediction. Statistics remain frozen so that test-time
    /// behaviour is deterministic with respect to the training data seen so far.
    fn transform(&self, features: &[f64]) -> Vec<f64>;

    /// Number of output features, or `None` if unknown until the first sample.
    fn output_dim(&self) -> Option<usize>;

    /// Reset to initial (untrained) state.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// PipelineBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing [`Pipeline`] instances.
///
/// Chain preprocessor steps with [`pipe`](Self::pipe), then terminate with
/// [`learner`](Self::learner) to produce the final `Pipeline`.
///
/// ```
/// use irithyll::preprocessing::IncrementalNormalizer;
/// use irithyll::pipeline::PipelineBuilder;
/// use irithyll::learners::StreamingLinearModel;
///
/// let pipeline = PipelineBuilder::new()
///     .pipe(IncrementalNormalizer::new())
///     .learner(StreamingLinearModel::new(0.01));
/// ```
pub struct PipelineBuilder {
    preprocessors: Vec<Box<dyn StreamingPreprocessor>>,
    target_preprocessor: Option<Box<dyn StreamingTargetPreprocessor>>,
}

impl PipelineBuilder {
    /// Create an empty pipeline builder with no preprocessor steps.
    pub fn new() -> Self {
        Self {
            preprocessors: Vec::new(),
            target_preprocessor: None,
        }
    }

    /// Append a preprocessor step to the pipeline.
    ///
    /// Steps execute in the order they are added: the first `pipe` call
    /// receives raw features, the second receives the output of the first, etc.
    pub fn pipe(mut self, preprocessor: impl StreamingPreprocessor + 'static) -> Self {
        self.preprocessors.push(Box::new(preprocessor));
        self
    }

    /// Attach an optional target preprocessor to the pipeline.
    ///
    /// The target preprocessor transforms the regression target before the
    /// learner sees it during training, and inverts the transformation on
    /// the predict output so that predictions are in the original target scale.
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll::preprocessing::{TargetScaler, IncrementalNormalizer};
    /// use irithyll::pipeline::PipelineBuilder;
    /// use irithyll::learners::StreamingLinearModel;
    /// use irithyll::StreamingLearner;
    ///
    /// let mut pipeline = PipelineBuilder::new()
    ///     .pipe(IncrementalNormalizer::new())
    ///     .target_preprocessor(TargetScaler::new())
    ///     .learner(StreamingLinearModel::new(0.01));
    ///
    /// pipeline.train(&[100.0, 0.5], 1000.0);
    /// let pred = pipeline.predict(&[100.0, 0.5]);
    /// // pred is in the original target scale (not z-scored).
    /// assert!(pred.is_finite());
    /// ```
    pub fn target_preprocessor(mut self, tp: impl StreamingTargetPreprocessor + 'static) -> Self {
        self.target_preprocessor = Some(Box::new(tp));
        self
    }

    /// Terminate the pipeline with a learner, producing a [`Pipeline`].
    ///
    /// The learner receives features that have been transformed by all
    /// preceding preprocessor steps.
    pub fn learner(self, learner: impl StreamingLearner + 'static) -> Pipeline {
        Pipeline {
            preprocessors: self.preprocessors,
            learner: Box::new(learner),
            target_preprocessor: self.target_preprocessor,
            samples_seen: 0,
        }
    }

    /// Terminate the pipeline with a boxed learner, producing a [`Pipeline`].
    ///
    /// Use this when the learner is already behind a `Box<dyn StreamingLearner>`.
    pub fn learner_boxed(self, learner: Box<dyn StreamingLearner>) -> Pipeline {
        Pipeline {
            preprocessors: self.preprocessors,
            learner,
            target_preprocessor: self.target_preprocessor,
            samples_seen: 0,
        }
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A streaming preprocessing → learning pipeline.
///
/// Chains zero or more [`StreamingPreprocessor`] steps with a terminal
/// [`StreamingLearner`]. The pipeline itself implements `StreamingLearner`,
/// enabling nested composition -- pipelines can participate in stacking
/// ensembles, be stored as `Box<dyn StreamingLearner>`, or sit inside
/// other pipelines.
///
/// # Training vs Prediction
///
/// | Phase | Preprocessors | Learner |
/// |-------|---------------|---------|
/// | `train_one` / `train` | `update_and_transform` | `train_one` |
/// | `predict` | `transform` (no update) | `predict` |
///
/// This matches the standard convention: preprocessor statistics are updated
/// only during training, then frozen during prediction.
///
/// # Construction
///
/// Use [`Pipeline::builder`] or the free function [`pipe`](crate::pipe) to
/// start building a pipeline:
///
/// ```
/// use irithyll::preprocessing::IncrementalNormalizer;
/// use irithyll::pipeline::Pipeline;
/// use irithyll::learners::StreamingLinearModel;
/// use irithyll::StreamingLearner;
///
/// let mut p = Pipeline::builder()
///     .pipe(IncrementalNormalizer::new())
///     .learner(StreamingLinearModel::new(0.01));
///
/// p.train(&[10.0, 20.0], 5.0);
/// let pred = p.predict(&[10.0, 20.0]);
/// ```
pub struct Pipeline {
    preprocessors: Vec<Box<dyn StreamingPreprocessor>>,
    learner: Box<dyn StreamingLearner>,
    /// Optional target preprocessor: applied to the target before `train_one`
    /// and inverted on the output of `predict`.
    target_preprocessor: Option<Box<dyn StreamingTargetPreprocessor>>,
    samples_seen: u64,
}

impl Pipeline {
    /// Start building a pipeline.
    ///
    /// Returns a [`PipelineBuilder`] that collects preprocessor steps.
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    /// Number of preprocessor steps in this pipeline.
    pub fn n_preprocessors(&self) -> usize {
        self.preprocessors.len()
    }

    /// Access the terminal learner.
    pub fn learner(&self) -> &dyn StreamingLearner {
        &*self.learner
    }

    /// Mutable access to the terminal learner.
    pub fn learner_mut(&mut self) -> &mut dyn StreamingLearner {
        &mut *self.learner
    }

    /// Run features through all preprocessors (transform only, no update).
    fn transform_features(&self, features: &[f64]) -> Vec<f64> {
        let mut x = features.to_vec();
        for preprocessor in &self.preprocessors {
            x = preprocessor.transform(&x);
        }
        x
    }

    /// Run features through all preprocessors (update + transform).
    fn update_and_transform_features(&mut self, features: &[f64]) -> Vec<f64> {
        let mut x = features.to_vec();
        for preprocessor in &mut self.preprocessors {
            x = preprocessor.update_and_transform(&x);
        }
        x
    }
}

impl StreamingLearner for Pipeline {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let x = self.update_and_transform_features(features);
        // === Wave 7-3 target preprocessor ===
        let transformed_target = if let Some(tp) = self.target_preprocessor.as_mut() {
            tp.fit_transform(target)
        } else {
            target
        };
        self.learner.train_one(&x, transformed_target, weight);
        self.samples_seen += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let x = self.transform_features(features);
        let raw = self.learner.predict(&x);
        // === Wave 7-3 target preprocessor ===
        if let Some(tp) = self.target_preprocessor.as_ref() {
            tp.inverse_transform(raw)
        } else {
            raw
        }
    }

    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        for preprocessor in &mut self.preprocessors {
            preprocessor.reset();
        }
        self.learner.reset();
        // === Wave 7-3 target preprocessor ===
        if let Some(tp) = self.target_preprocessor.as_mut() {
            tp.reset();
        }
        self.samples_seen = 0;
    }

    #[allow(deprecated)]
    fn diagnostics_array(&self) -> [f64; 5] {
        self.learner.diagnostics_array()
    }

    #[allow(deprecated)]
    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64) {
        self.learner.adjust_config(lr_multiplier, lambda_delta);
    }

    #[allow(deprecated)]
    fn apply_structural_change(&mut self, depth_delta: i32, steps_delta: i32) {
        self.learner
            .apply_structural_change(depth_delta, steps_delta);
    }

    #[allow(deprecated)]
    fn replacement_count(&self) -> u64 {
        self.learner.replacement_count()
    }
}

// Pipeline is Send + Sync because its fields are:
// - Vec<Box<dyn StreamingPreprocessor>>: Send + Sync (trait bound)
// - Box<dyn StreamingLearner>: Send + Sync (trait bound)
// - u64: Send + Sync

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("n_preprocessors", &self.preprocessors.len())
            .field(
                "has_target_preprocessor",
                &self.target_preprocessor.is_some(),
            )
            .field("samples_seen", &self.samples_seen)
            .finish()
    }
}

use std::fmt;

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for Pipeline {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::IncrementalNormalizer;

    const EPS: f64 = 1e-6;

    // A trivial preprocessor for testing: multiplies all features by a constant.
    #[derive(Clone)]
    struct ScalePreprocessor {
        factor: f64,
        dim: Option<usize>,
    }

    impl ScalePreprocessor {
        fn new(factor: f64) -> Self {
            Self { factor, dim: None }
        }
    }

    impl StreamingPreprocessor for ScalePreprocessor {
        fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
            self.dim = Some(features.len());
            features.iter().map(|&x| x * self.factor).collect()
        }

        fn transform(&self, features: &[f64]) -> Vec<f64> {
            features.iter().map(|&x| x * self.factor).collect()
        }

        fn output_dim(&self) -> Option<usize> {
            self.dim
        }

        fn reset(&mut self) {
            self.dim = None;
        }
    }

    // A trivial learner for testing: returns the mean of features as prediction.
    struct MeanLearner {
        samples: u64,
    }

    impl MeanLearner {
        fn new() -> Self {
            Self { samples: 0 }
        }
    }

    impl StreamingLearner for MeanLearner {
        fn train_one(&mut self, _features: &[f64], _target: f64, _weight: f64) {
            self.samples += 1;
        }

        fn predict(&self, features: &[f64]) -> f64 {
            if features.is_empty() {
                return 0.0;
            }
            features.iter().sum::<f64>() / features.len() as f64
        }

        fn n_samples_seen(&self) -> u64 {
            self.samples
        }

        fn reset(&mut self) {
            self.samples = 0;
        }
    }

    // MeanLearner is Send+Sync by composition (f64, u64 fields only).

    #[test]
    fn builder_creates_pipeline() {
        let p = Pipeline::builder()
            .pipe(ScalePreprocessor::new(2.0))
            .learner(MeanLearner::new());

        assert_eq!(p.n_preprocessors(), 1);
        assert_eq!(p.n_samples_seen(), 0);
    }

    #[test]
    fn pipeline_trains_and_predicts() {
        let mut p = Pipeline::builder()
            .pipe(ScalePreprocessor::new(2.0))
            .learner(MeanLearner::new());

        p.train(&[1.0, 2.0, 3.0], 0.0);
        assert_eq!(p.n_samples_seen(), 1);

        // ScalePreprocessor doubles features: [1, 2, 3] -> [2, 4, 6]
        // MeanLearner returns mean: (2 + 4 + 6) / 3 = 4.0
        let pred = p.predict(&[1.0, 2.0, 3.0]);
        assert!((pred - 4.0).abs() < EPS, "pred = {}", pred);
    }

    #[test]
    fn multi_preprocessor_chaining() {
        let mut p = Pipeline::builder()
            .pipe(ScalePreprocessor::new(2.0))
            .pipe(ScalePreprocessor::new(3.0))
            .learner(MeanLearner::new());

        p.train(&[1.0, 1.0], 0.0);

        // [1, 1] -> *2 -> [2, 2] -> *3 -> [6, 6]
        // MeanLearner returns 6.0
        let pred = p.predict(&[1.0, 1.0]);
        assert!((pred - 6.0).abs() < EPS, "pred = {}", pred);
    }

    #[test]
    fn predict_does_not_update_preprocessor() {
        let mut p = Pipeline::builder()
            .pipe(ScalePreprocessor::new(2.0))
            .learner(MeanLearner::new());

        // Before any training, output_dim is None.
        assert_eq!(p.preprocessors[0].output_dim(), None);

        // predict should NOT call update_and_transform.
        let _ = p.predict(&[1.0, 2.0]);
        assert_eq!(p.preprocessors[0].output_dim(), None);
        assert_eq!(p.n_samples_seen(), 0);

        // train should update.
        p.train(&[1.0, 2.0], 0.0);
        assert_eq!(p.preprocessors[0].output_dim(), Some(2));
        assert_eq!(p.n_samples_seen(), 1);
    }

    #[test]
    fn reset_clears_all_state() {
        let mut p = Pipeline::builder()
            .pipe(ScalePreprocessor::new(2.0))
            .learner(MeanLearner::new());

        p.train(&[1.0], 0.0);
        p.train(&[2.0], 0.0);
        assert_eq!(p.n_samples_seen(), 2);
        assert_eq!(p.preprocessors[0].output_dim(), Some(1));

        p.reset();
        assert_eq!(p.n_samples_seen(), 0);
        assert_eq!(p.preprocessors[0].output_dim(), None);
        assert_eq!(p.learner().n_samples_seen(), 0);
    }

    #[test]
    fn pipeline_as_trait_object() {
        let p = Pipeline::builder()
            .pipe(ScalePreprocessor::new(1.0))
            .learner(MeanLearner::new());

        // Must compile: Pipeline behind Box<dyn StreamingLearner>.
        let mut boxed: Box<dyn StreamingLearner> = Box::new(p);
        boxed.train(&[5.0, 10.0], 0.0);
        let pred = boxed.predict(&[5.0, 10.0]);
        assert!((pred - 7.5).abs() < EPS);
        assert_eq!(boxed.n_samples_seen(), 1);
    }

    #[test]
    fn pipeline_with_normalizer() {
        let mut p = Pipeline::builder()
            .pipe(IncrementalNormalizer::new())
            .learner(MeanLearner::new());

        // Feed several samples to build up normalizer statistics.
        for i in 0..100 {
            p.train(&[i as f64, (i as f64) * 2.0], 0.0);
        }

        // Predict on the mean -- normalized features should be near zero.
        let pred = p.predict(&[49.5, 99.0]);
        assert!(
            pred.abs() < 0.5,
            "prediction on mean features should be near zero, got {}",
            pred
        );
    }

    #[test]
    fn empty_preprocessor_pipeline() {
        // Pipeline with no preprocessors -- features pass straight to learner.
        let mut p = Pipeline::builder().learner(MeanLearner::new());

        assert_eq!(p.n_preprocessors(), 0);
        p.train(&[10.0, 20.0], 0.0);
        let pred = p.predict(&[10.0, 20.0]);
        assert!((pred - 15.0).abs() < EPS);
    }

    #[test]
    fn learner_boxed_constructor() {
        let learner: Box<dyn StreamingLearner> = Box::new(MeanLearner::new());
        let mut p = Pipeline::builder()
            .pipe(ScalePreprocessor::new(2.0))
            .learner_boxed(learner);

        p.train(&[3.0], 0.0);
        let pred = p.predict(&[3.0]);
        assert!((pred - 6.0).abs() < EPS);
    }

    // ---------------------------------------------------------------------------
    // Wave 7-3: target preprocessor wiring tests
    // ---------------------------------------------------------------------------

    #[test]
    fn pipeline_with_target_preprocessor_inverts_on_predict() {
        use crate::preprocessing::{StreamingTargetPreprocessor, TargetScaler};

        // Use a simple identity learner: predict returns the target it was trained on.
        // We approximate this by training a MeanLearner with a constant target.
        // Strategy: train pipeline with target preprocessor on a constant target (5.0),
        // then verify that predict output is back in the original scale.
        //
        // TargetScaler with a single distinct value: after 1 sample the mean=5.0,
        // std=sqrt(variance_floor). fit_transform(5.0) = 0.0. The learner trains on 0.0.
        // predict returns MeanLearner output (mean of features). inverse_transform of that
        // value brings it back. The key invariant tested here: predict() inverts the
        // target transform so the output is in original-target units.
        // Drive a primary scaler and a parallel reference scaler through the same
        // sequence of fit_transform calls, so the reference matches the pipeline
        // scaler's post-train state. predict() uses post-train statistics, so the
        // expected value must be computed from those same statistics.
        let mut scaler = TargetScaler::new();
        let mut reference = TargetScaler::new();
        for i in 0..100_i32 {
            scaler.fit_transform(i as f64);
            reference.fit_transform(i as f64);
        }
        let warm_mean = scaler.mean();

        let mut pipeline = Pipeline::builder()
            .target_preprocessor(scaler)
            .learner_boxed(Box::new(MeanLearner::new()));

        // Pipeline.train() advances its scaler by one sample; mirror on reference.
        pipeline.train(&[1.0, 2.0], warm_mean);
        reference.fit_transform(warm_mean);

        let pred = pipeline.predict(&[1.0, 2.0]);
        // MeanLearner predicts mean(1, 2) = 1.5 in z-space; inverse → 1.5 * σ + μ
        // computed from the reference scaler's post-train statistics.
        let expected = reference.inverse_transform(1.5);
        assert!(
            (pred - expected).abs() < 1e-9,
            "predict should invert target scaler: expected {}, got {}",
            expected,
            pred
        );
    }

    #[test]
    fn pipeline_target_preprocessor_reset_clears_scaler() {
        use crate::preprocessing::TargetScaler;

        let mut pipeline = Pipeline::builder()
            .target_preprocessor(TargetScaler::new())
            .learner_boxed(Box::new(MeanLearner::new()));

        for i in 0..10_i32 {
            pipeline.train(&[1.0], i as f64);
        }
        // After training, scaler has non-trivial stats.
        assert_eq!(pipeline.n_samples_seen(), 10);

        pipeline.reset();
        assert_eq!(pipeline.n_samples_seen(), 0);
        // After reset, scaler returns to initial state: transforming any value
        // should still produce a finite result (cold-start behaviour).
        let pred = pipeline.predict(&[1.0]);
        assert!(
            pred.is_finite(),
            "predict after reset should be finite, got {}",
            pred
        );
    }

    // ---------------------------------------------------------------------------
    // make_pipeline! macro tests (Wave 7-2)
    // ---------------------------------------------------------------------------

    #[test]
    fn make_pipeline_compiles_with_one_stage() {
        // Degenerate case: a single learner, no preprocessors.
        let mut p = crate::make_pipeline!(MeanLearner::new());
        assert_eq!(
            p.n_preprocessors(),
            0,
            "no preprocessors in single-stage pipeline"
        );
        p.train(&[3.0, 7.0], 0.0);
        let pred = p.predict(&[3.0, 7.0]);
        // MeanLearner returns mean of features: (3+7)/2 = 5
        assert!(
            (pred - 5.0).abs() < EPS,
            "single-stage pipeline pred = {pred}"
        );
    }

    #[test]
    fn make_pipeline_chains_three_stages() {
        // Three stages: two preprocessors + one learner.
        // ScalePreprocessor(2) then ScalePreprocessor(3): [1,1] -> [2,2] -> [6,6]
        // MeanLearner returns 6.0
        let mut p = crate::make_pipeline!(
            ScalePreprocessor::new(2.0) =>
            ScalePreprocessor::new(3.0) =>
            MeanLearner::new()
        );
        assert_eq!(
            p.n_preprocessors(),
            2,
            "two preprocessors in three-stage pipeline"
        );
        p.train(&[1.0, 1.0], 0.0);
        let pred = p.predict(&[1.0, 1.0]);
        assert!(
            (pred - 6.0).abs() < EPS,
            "three-stage make_pipeline! expected 6.0, got {pred}"
        );
    }

    #[test]
    fn make_pipeline_two_stages() {
        // One preprocessor + learner: [2,4] -> *2 -> [4,8] -> mean -> 6.0
        let mut p = crate::make_pipeline!(
            ScalePreprocessor::new(2.0) =>
            MeanLearner::new()
        );
        assert_eq!(
            p.n_preprocessors(),
            1,
            "one preprocessor in two-stage pipeline"
        );
        p.train(&[2.0, 4.0], 0.0);
        let pred = p.predict(&[2.0, 4.0]);
        assert!(
            (pred - 6.0).abs() < EPS,
            "two-stage make_pipeline! expected 6.0, got {pred}"
        );
    }

    #[test]
    fn make_pipeline_implements_streaming_learner() {
        // Pipeline produced by make_pipeline! must satisfy StreamingLearner.
        let mut p: Box<dyn StreamingLearner> =
            Box::new(crate::make_pipeline!(ScalePreprocessor::new(1.0) => MeanLearner::new()));
        p.train(&[5.0, 10.0], 0.0);
        let pred = p.predict(&[5.0, 10.0]);
        assert!(
            (pred - 7.5).abs() < EPS,
            "boxed make_pipeline! pred = {pred}"
        );
    }
}
