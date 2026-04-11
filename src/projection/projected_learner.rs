//! Projection-based wrapper for streaming learners.
//!
//! [`ProjectedLearner`] composes a [`SubspaceTracker`] (PAST algorithm) with
//! an [`IncrementalNormalizer`] and any inner [`StreamingLearner`]. The wrapper
//! normalizes inputs, projects them to a lower-dimensional subspace, and feeds
//! the projected features to the inner model.
//!
//! # Projection Update Strategy
//!
//! The projection update has two paths, selected automatically based on
//! whether the inner model exposes RLS readout weights via
//! [`readout_weights()`](StreamingLearner::readout_weights):
//!
//! - **Supervised** (inner model has RLS readout): Uses the prediction
//!   gradient `dL/dW = -2 * residual * outer(x, beta)` to steer W toward
//!   prediction-relevant directions. This finds signal even when noise
//!   dimensions have comparable variance to signal dimensions.
//!
//! - **Unsupervised** (inner model has no RLS readout): Falls back to pure
//!   PAST (streaming PCA), tracking directions of maximum variance. This
//!   works well when the signal lives in the top principal components.
//!
//! # Data Flow
//!
//! ```text
//! x (d_in)
//!   -> normalizer.update(x)
//!   -> z = normalizer.transform(x)      (zero-mean, unit-var)
//!   -> p = tracker.project(z)            (rank-dim projection)
//!   -> inner.predict(p) / inner.train(p, target)
//!   -> if inner has readout weights:
//!        tracker.supervised_update(z, residual, beta, lr)
//!      else:
//!        tracker.update(z)               (pure PCA)
//! ```
//!
//! During warmup (first `config.warmup` samples) the normalizer accumulates
//! statistics but the projection is not updated. The inner model always sees
//! rank-dimensional features, even during warmup, so there is no
//! dimension mismatch at the warmup-to-projection transition.

use crate::error::{ConfigError, IrithyllError, Result};
use crate::learner::StreamingLearner;
use crate::preprocessing::IncrementalNormalizer;
use crate::projection::SubspaceTracker;

use std::fmt;

// ===========================================================================
// ProjectionConfig
// ===========================================================================

/// Configuration for online projection learning.
///
/// Controls the PAST subspace tracker and the warmup period for the
/// normalizer. Use [`ProjectionConfig::builder()`] for ergonomic construction
/// with validation.
///
/// # Defaults
///
/// | Parameter | Default | Description |
/// |-----------|---------|-------------|
/// | `rank` | 8 | Projection rank (output dimension) |
/// | `lambda` | 0.9999 | PAST forgetting factor (half-life ~6931 samples) |
/// | `delta` | 100.0 | Initial P diagonal scaling |
/// | `warmup` | 200 | Warmup samples (normalizer only, no PAST updates) |
/// | `seed` | 42 | RNG seed for Xavier initialization |
/// | `supervised_lr` | 0.01 | Learning rate for supervised projection gradient |
#[derive(Clone, Debug)]
pub struct ProjectionConfig {
    /// Projection rank (output dimension). Default: 8.
    pub rank: usize,
    /// PAST forgetting factor (half-life ~6931 samples). Default: 0.9999.
    pub lambda: f64,
    /// Initial P diagonal scaling. Default: 100.0.
    pub delta: f64,
    /// Warmup samples (pass raw features before updating PAST). Default: 200.
    pub warmup: usize,
    /// RNG seed. Default: 42.
    pub seed: u64,
    /// Learning rate for supervised projection gradient. Default: 0.01.
    ///
    /// When the inner model exposes readout weights, this learning rate
    /// controls how fast W adapts to prediction-relevant directions.
    /// Decoupled from `lambda` because the projection learning timescale
    /// and forgetting timescale serve different purposes.
    pub supervised_lr: f64,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            lambda: 0.9999,
            delta: 100.0,
            warmup: 200,
            seed: 42,
            supervised_lr: 0.001,
        }
    }
}

impl ProjectionConfig {
    /// Start building a `ProjectionConfig` with default values.
    pub fn builder() -> ProjectionConfigBuilder {
        ProjectionConfigBuilder::new()
    }
}

// ===========================================================================
// ProjectionConfigBuilder
// ===========================================================================

/// Builder for [`ProjectionConfig`] with validation on [`build()`](Self::build).
#[derive(Debug, Clone)]
pub struct ProjectionConfigBuilder {
    config: ProjectionConfig,
}

impl ProjectionConfigBuilder {
    /// Create a new builder with default hyperparameters.
    pub fn new() -> Self {
        Self {
            config: ProjectionConfig::default(),
        }
    }

    /// Set the projection rank (output dimension).
    pub fn rank(mut self, rank: usize) -> Self {
        self.config.rank = rank;
        self
    }

    /// Set the PAST forgetting factor (lambda).
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.config.lambda = lambda;
        self
    }

    /// Set the initial P diagonal scaling (delta).
    pub fn delta(mut self, delta: f64) -> Self {
        self.config.delta = delta;
        self
    }

    /// Set the warmup period (samples before PAST updates begin).
    pub fn warmup(mut self, warmup: usize) -> Self {
        self.config.warmup = warmup;
        self
    }

    /// Set the RNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Set the supervised projection learning rate. Default: 0.01.
    ///
    /// Controls how fast W adapts toward prediction-relevant directions
    /// when the inner model exposes readout weights. Higher values give
    /// faster adaptation but may overshoot on noisy data.
    pub fn supervised_lr(mut self, lr: f64) -> Self {
        self.config.supervised_lr = lr;
        self
    }

    /// Validate and build the configuration.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if any parameter is out of range.
    pub fn build(self) -> Result<ProjectionConfig> {
        let c = &self.config;

        if c.rank < 1 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "rank",
                "must be >= 1",
                c.rank,
            )));
        }
        if c.lambda <= 0.0 || c.lambda > 1.0 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "lambda",
                "must be in (0, 1]",
                c.lambda,
            )));
        }
        if c.delta <= 0.0 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "delta",
                "must be > 0",
                c.delta,
            )));
        }
        if c.seed == 0 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "seed",
                "must be non-zero (xorshift64)",
                c.seed,
            )));
        }
        if c.supervised_lr <= 0.0 || c.supervised_lr > 1.0 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "supervised_lr",
                "must be in (0, 1]",
                c.supervised_lr,
            )));
        }

        Ok(self.config)
    }
}

impl Default for ProjectionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// ProjectedLearner
// ===========================================================================

/// Wraps any [`StreamingLearner`] with online projection learning.
///
/// The wrapper normalizes raw inputs, projects them through an adaptive
/// subspace tracker, and feeds the reduced features to the inner model.
///
/// When the inner model provides readout weights (via
/// [`readout_weights()`](StreamingLearner::readout_weights)), the projection
/// uses **supervised gradient updates** that steer the subspace toward
/// prediction-relevant directions. Otherwise, it falls back to unsupervised
/// PAST (streaming PCA), tracking directions of maximum variance.
///
/// The inner model always sees `rank`-dimensional features, regardless of
/// the original input dimension. This means the inner model must be configured
/// for `rank` input dimensions, not `d_in`.
///
/// # Example
///
/// ```
/// use irithyll::projection::{ProjectedLearner, ProjectionConfig};
/// use irithyll::{rls, StreamingLearner};
///
/// let config = ProjectionConfig::builder()
///     .rank(4)
///     .warmup(20)
///     .build()
///     .unwrap();
///
/// // Inner model sees rank=4 features
/// let mut model = ProjectedLearner::new(Box::new(rls(0.99)), 10, config);
///
/// for i in 0..100 {
///     let x = vec![i as f64 * 0.01; 10];
///     model.train(&x, i as f64);
/// }
/// let pred = model.predict(&[0.5; 10]);
/// assert!(pred.is_finite());
/// ```
pub struct ProjectedLearner {
    /// The wrapped streaming model (sees rank-dim features).
    inner: Box<dyn StreamingLearner>,
    /// PAST subspace tracker (d_in -> rank).
    tracker: SubspaceTracker,
    /// Welford normalizer for input standardization.
    normalizer: IncrementalNormalizer,
    /// Configuration snapshot.
    config: ProjectionConfig,
    /// Total samples processed.
    n_samples: u64,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl ProjectedLearner {
    /// Create a new projected learner from a boxed inner model.
    ///
    /// # Arguments
    ///
    /// * `inner` -- the downstream streaming learner (must accept `rank`-dim input).
    /// * `d_in` -- the original input dimensionality.
    /// * `config` -- projection configuration.
    ///
    /// # Panics
    ///
    /// Panics if `config.rank > d_in` (delegated to `SubspaceTracker::new`).
    pub fn new(inner: Box<dyn StreamingLearner>, d_in: usize, config: ProjectionConfig) -> Self {
        let tracker =
            SubspaceTracker::new(d_in, config.rank, config.lambda, config.delta, config.seed);
        let normalizer = IncrementalNormalizer::new();
        Self {
            inner,
            tracker,
            normalizer,
            config,
            n_samples: 0,
        }
    }

    /// Convenience constructor that accepts any concrete `StreamingLearner`.
    ///
    /// Equivalent to `ProjectedLearner::new(Box::new(inner), d_in, config)`.
    pub fn from_learner(
        inner: impl StreamingLearner + 'static,
        d_in: usize,
        config: ProjectionConfig,
    ) -> Self {
        Self::new(Box::new(inner), d_in, config)
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Immutable reference to the PAST subspace tracker.
    #[inline]
    pub fn tracker(&self) -> &SubspaceTracker {
        &self.tracker
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

    /// The projection configuration.
    #[inline]
    pub fn config(&self) -> &ProjectionConfig {
        &self.config
    }

    /// Whether the warmup period has completed.
    #[inline]
    pub fn warmup_complete(&self) -> bool {
        self.n_samples >= self.config.warmup as u64
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

impl StreamingLearner for ProjectedLearner {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // 1. Update normalizer statistics with raw input.
        self.normalizer.update(features);

        // 2. Normalize input.
        let normed = self.normalizer.transform(features);

        // 3. Project to rank dimensions.
        let projected = self.tracker.project(&normed);

        // 4. Train inner model on projected features.
        self.inner.train_one(&projected, target, weight);

        // 5. Update projection after warmup.
        //    - Supervised path: if the inner model exposes RLS readout weights
        //      whose length matches the projection rank, use the prediction
        //      gradient to steer W toward signal directions.
        //    - Unsupervised path: fall back to pure PAST (reconstruction-based PCA).
        //
        //    The length check is necessary because models like Mamba/ESN/TTT have
        //    an RLS readout that operates on their internal state (hidden size),
        //    not the projected features.  Only when the inner model is a plain RLS
        //    (or similar) whose weight vector matches the projection rank can we
        //    use it as the gradient direction for W.
        if self.n_samples >= self.config.warmup as u64 {
            if let Some(beta) = self.inner.readout_weights() {
                if beta.len() == self.tracker.rank() {
                    let pred = self.inner.predict(&projected);
                    let residual = target - pred;
                    // Ramp supervised LR: inner model's readout weights are
                    // unreliable early on. Linear ramp from 0 to full LR over
                    // the first 1000 post-warmup samples avoids early noise.
                    let post_warmup = self.n_samples - self.config.warmup as u64;
                    let ramp = (post_warmup as f64 / 1000.0).min(1.0);
                    let lr = self.config.supervised_lr * ramp;
                    self.tracker.supervised_update(&normed, residual, beta, lr);
                } else {
                    // Readout dimension != projection rank: supervised gradient
                    // is not meaningful.  Fall back to unsupervised PAST.
                    self.tracker.update(&normed, 0.0);
                }
            } else {
                self.tracker.update(&normed, 0.0);
            }
        }

        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        // Before any training, the normalizer has no statistics.
        // Use raw features through the (random) projection -- the inner model
        // is also untrained, so the prediction is meaningless anyway.
        if self.normalizer.count() == 0 {
            let projected = self.tracker.project(features);
            return self.inner.predict(&projected);
        }
        let normed = self.normalizer.transform(features);
        let projected = self.tracker.project(&normed);
        self.inner.predict(&projected)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.tracker.reset();
        self.normalizer = IncrementalNormalizer::new();
        self.n_samples = 0;
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
        self.inner.replacement_count()
    }

    fn check_proactive_prune(&mut self) -> bool {
        self.inner.check_proactive_prune()
    }

    fn set_prune_half_life(&mut self, hl: usize) {
        self.inner.set_prune_half_life(hl);
    }

    fn readout_weights(&self) -> Option<&[f64]> {
        self.inner.readout_weights()
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for ProjectedLearner {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        // Cannot access inner learner diagnostics through Box<dyn StreamingLearner>.
        None
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

impl fmt::Debug for ProjectedLearner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProjectedLearner")
            .field("d_in", &self.tracker.d_in())
            .field("rank", &self.config.rank)
            .field("warmup", &self.config.warmup)
            .field("n_samples", &self.n_samples)
            .field("warmup_complete", &self.warmup_complete())
            .finish()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

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

    // MeanLearner is Send+Sync by composition (f64, u64 fields only).

    // -----------------------------------------------------------------------
    // Config builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn default_config_builds() {
        let config = ProjectionConfig::builder().build().unwrap();
        assert_eq!(config.rank, 8);
        assert!((config.lambda - 0.9999).abs() < 1e-12);
        assert!((config.delta - 100.0).abs() < 1e-12);
        assert_eq!(config.warmup, 200);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn custom_config_builds() {
        let config = ProjectionConfig::builder()
            .rank(4)
            .lambda(0.995)
            .delta(50.0)
            .warmup(20)
            .seed(123)
            .build()
            .unwrap();
        assert_eq!(config.rank, 4);
        assert!((config.lambda - 0.995).abs() < 1e-12);
        assert!((config.delta - 50.0).abs() < 1e-12);
        assert_eq!(config.warmup, 20);
        assert_eq!(config.seed, 123);
    }

    #[test]
    fn zero_rank_fails() {
        let result = ProjectionConfig::builder().rank(0).build();
        assert!(result.is_err(), "rank=0 should fail validation");
    }

    #[test]
    fn lambda_zero_fails() {
        let result = ProjectionConfig::builder().lambda(0.0).build();
        assert!(result.is_err(), "lambda=0 should fail validation");
    }

    #[test]
    fn lambda_above_one_fails() {
        let result = ProjectionConfig::builder().lambda(1.01).build();
        assert!(result.is_err(), "lambda>1 should fail validation");
    }

    #[test]
    fn delta_zero_fails() {
        let result = ProjectionConfig::builder().delta(0.0).build();
        assert!(result.is_err(), "delta=0 should fail validation");
    }

    #[test]
    fn seed_zero_fails() {
        let result = ProjectionConfig::builder().seed(0).build();
        assert!(result.is_err(), "seed=0 should fail validation");
    }

    // -----------------------------------------------------------------------
    // ProjectedLearner tests
    // -----------------------------------------------------------------------

    fn make_projected(d_in: usize, rank: usize, warmup: usize) -> ProjectedLearner {
        let config = ProjectionConfig::builder()
            .rank(rank)
            .warmup(warmup)
            .build()
            .unwrap();
        ProjectedLearner::from_learner(MeanLearner::new(), d_in, config)
    }

    #[test]
    fn train_and_predict_basic() {
        let mut model = make_projected(6, 3, 5);

        for i in 0..20 {
            let x = vec![i as f64 * 0.1; 6];
            model.train(&x, i as f64);
        }
        assert_eq!(model.n_samples_seen(), 20);

        let pred = model.predict(&[0.5; 6]);
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );
    }

    #[test]
    fn predict_before_training_returns_finite() {
        // Before any training, predict should still return a finite value
        // (using raw features through the random projection).
        let model = make_projected(4, 2, 5);
        let pred = model.predict(&[1.0; 4]);
        assert!(
            pred.is_finite(),
            "predict before training should return finite value, got {}",
            pred
        );
    }

    #[test]
    fn warmup_transition_is_seamless() {
        let warmup = 10;
        let mut model = make_projected(8, 4, warmup);

        // Train through warmup
        for i in 0..warmup + 20 {
            let x = vec![(i as f64 * 0.1).sin(); 8];
            model.train(&x, i as f64);
        }

        assert!(model.warmup_complete(), "warmup should be complete");
        assert_eq!(model.n_samples_seen(), (warmup + 20) as u64);

        // PAST should have received updates after warmup
        assert!(
            model.tracker().n_samples() > 0,
            "tracker should have samples after warmup"
        );
    }

    #[test]
    fn reset_clears_everything() {
        let mut model = make_projected(6, 3, 5);

        for i in 0..30 {
            model.train(&[i as f64; 6], i as f64);
        }
        assert!(model.n_samples_seen() > 0);
        assert!(model.warmup_complete());

        model.reset();

        assert_eq!(
            model.n_samples_seen(),
            0,
            "n_samples should be 0 after reset"
        );
        assert!(
            !model.warmup_complete(),
            "warmup should not be complete after reset"
        );
        assert_eq!(model.tracker().n_samples(), 0, "tracker should be reset");
        assert_eq!(
            model.inner().n_samples_seen(),
            0,
            "inner model should be reset"
        );
    }

    #[test]
    fn accessors_work() {
        let model = make_projected(10, 4, 20);

        assert_eq!(model.tracker().d_in(), 10);
        assert_eq!(model.tracker().rank(), 4);
        assert_eq!(model.config().rank, 4);
        assert_eq!(model.config().warmup, 20);
        assert_eq!(model.inner().n_samples_seen(), 0);
    }

    #[test]
    fn from_boxed_constructor_works() {
        let inner: Box<dyn StreamingLearner> = Box::new(MeanLearner::new());
        let config = ProjectionConfig::builder().rank(3).build().unwrap();
        let mut model = ProjectedLearner::new(inner, 6, config);

        model.train(&[1.0; 6], 5.0);
        model.train(&[2.0; 6], 10.0);

        assert_eq!(model.n_samples_seen(), 2);
        let pred = model.predict(&[1.5; 6]);
        assert!(pred.is_finite());
    }

    #[test]
    fn as_trait_object() {
        let model = make_projected(6, 3, 5);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(model);

        boxed.train(&[1.0; 6], 7.0);
        assert_eq!(boxed.n_samples_seen(), 1);

        let pred = boxed.predict(&[1.0; 6]);
        assert!(
            pred.is_finite(),
            "trait object predict should work: got {}",
            pred
        );
    }

    #[test]
    fn debug_format_is_informative() {
        let model = make_projected(10, 4, 20);
        let debug = format!("{:?}", model);
        assert!(
            debug.contains("ProjectedLearner"),
            "debug should contain struct name"
        );
        assert!(debug.contains("rank"), "debug should contain rank field");
        assert!(debug.contains("d_in"), "debug should contain d_in field");
    }

    #[test]
    fn inner_model_sees_rank_dim_features() {
        // Verify inner model gets rank-dim input by checking that it trains
        // successfully even though raw input is d_in-dim.
        let mut model = make_projected(20, 5, 10);

        for i in 0..50 {
            let x = vec![i as f64 * 0.02; 20];
            model.train(&x, i as f64);
        }

        // If inner model received d_in-dim features, the MeanLearner would
        // still work (it ignores features), but the subspace tracker would
        // have been exercised correctly.
        assert_eq!(model.inner().n_samples_seen(), 50);
        assert_eq!(model.tracker().n_samples(), 40); // 50 - 10 warmup
    }

    #[test]
    fn predict_is_side_effect_free() {
        let mut model = make_projected(6, 3, 5);

        for i in 0..10 {
            model.train(&[i as f64; 6], i as f64);
        }

        let n_before = model.n_samples_seen();
        let tracker_n_before = model.tracker().n_samples();

        // Multiple predictions should not change any state.
        let _ = model.predict(&[0.5; 6]);
        let _ = model.predict(&[0.5; 6]);
        let _ = model.predict(&[0.5; 6]);

        assert_eq!(
            model.n_samples_seen(),
            n_before,
            "predict should not change n_samples"
        );
        assert_eq!(
            model.tracker().n_samples(),
            tracker_n_before,
            "predict should not change tracker samples"
        );
    }
}
