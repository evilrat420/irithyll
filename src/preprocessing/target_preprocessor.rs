//! Streaming target preprocessors: transformers that operate on the target
//! variable before learning and invert the transformation on prediction output.
//!
//! [`StreamingTargetPreprocessor`] is the symmetric counterpart to
//! [`StreamingPreprocessor`](crate::pipeline::StreamingPreprocessor). Feature
//! preprocessors transform the input space; target preprocessors transform the
//! output space.
//!
//! # Why target preprocessing matters
//!
//! Regression targets can span many orders of magnitude (e.g., house prices)
//! or have heavy-tailed distributions. Training directly on raw targets
//! can cause gradient instability or slow convergence. By transforming the
//! target before the learner and inverting on predict, the learner operates in
//! a well-conditioned space while the pipeline still returns predictions in the
//! original scale.
//!
//! # Invertibility contract
//!
//! Every implementor MUST satisfy:
//! ```text
//! inverse_transform(fit_transform(y)) ≈ y   for all valid y
//! ```
//! where "valid" is defined per implementor. See individual implementors for
//! domain restrictions.
//!
//! # Implementors
//!
//! | Type | Transform | Inverse | Domain restriction |
//! |------|-----------|---------|-------------------|
//! | [`TargetScaler`] | z-score | un-z-score | none |
//! | [`TargetLog1pTransform`] | `log1p(y)` | `expm1(y')` | `y >= -1.0` |
//! | [`TargetEncoderPreprocessor`] | smoothed target mean | global mean | categorical columns |
//!
//! # Pipeline integration
//!
//! Wire a target preprocessor into a [`Pipeline`](crate::pipeline::Pipeline)
//! using [`PipelineBuilder::target_preprocessor`](crate::pipeline::PipelineBuilder::target_preprocessor):
//!
//! ```
//! use irithyll::preprocessing::{TargetScaler, IncrementalNormalizer};
//! use irithyll::pipeline::Pipeline;
//! use irithyll::learners::StreamingLinearModel;
//! use irithyll::StreamingLearner;
//!
//! let mut pipeline = Pipeline::builder()
//!     .pipe(IncrementalNormalizer::new())
//!     .target_preprocessor(TargetScaler::new())
//!     .learner(StreamingLinearModel::new(0.01));
//!
//! pipeline.train(&[100.0, 0.5], 1000.0);
//! let pred = pipeline.predict(&[100.0, 0.5]);
//! // pred is in the original target scale (not z-scored).
//! assert!(pred.is_finite());
//! ```

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Object-safe trait for streaming target transformers.
///
/// A `StreamingTargetPreprocessor` maintains running statistics that update
/// with each call to [`fit_transform`](Self::fit_transform). During prediction,
/// [`inverse_transform`](Self::inverse_transform) reverts the transformation
/// so that the pipeline output is in the original target scale.
///
/// # Object Safety
///
/// All methods use `&self` / `&mut self` with concrete argument and return
/// types, enabling `Box<dyn StreamingTargetPreprocessor>` for runtime-polymorphic
/// pipelines.
///
/// # Invertibility
///
/// Implementors must guarantee `inverse_transform(fit_transform(y)) ≈ y` for
/// all `y` in the declared domain. When the domain is restricted (e.g., `y >= -1.0`
/// for log-transform), the restriction must be documented clearly.
pub trait StreamingTargetPreprocessor: Send + Sync {
    /// Update running statistics from this target and return the transformed value.
    ///
    /// Called during training. The preprocessor incorporates `target` into its
    /// running state (e.g., Welford mean/variance) and returns the transformed
    /// value that the downstream learner will train on.
    fn fit_transform(&mut self, target: f64) -> f64;

    /// Invert a transformed target back to the original scale.
    ///
    /// Called during prediction. The result is the prediction in the original
    /// target units. Running statistics remain unchanged.
    ///
    /// Implementors must document any domain restrictions that make the inverse
    /// undefined (e.g., negative inputs to `expm1`).
    fn inverse_transform(&self, transformed: f64) -> f64;

    /// Reset to initial (untrained) state.
    ///
    /// Clears accumulated statistics. The preprocessor behaves as if no
    /// samples have been seen.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// TargetScaler  (z-score)
// ---------------------------------------------------------------------------

/// Streaming z-score scaler for regression targets.
///
/// Maintains a Welford online mean and variance, transforming each target to
/// zero-mean, unit-variance space:
///
/// ```text
/// z = (y - mean) / sqrt(variance + floor)
/// ```
///
/// The variance floor (default `1e-10`) prevents division by zero when all
/// targets seen so far are identical.
///
/// # Invertibility
///
/// The inverse is exact for any finite input:
/// ```text
/// y = z * sqrt(variance + floor) + mean
/// ```
///
/// No domain restriction. All finite `f64` values are valid.
///
/// # Cold start
///
/// Before any samples are seen, the first call to `fit_transform` returns
/// `0.0` (i.e., the first sample is treated as the mean). The Welford
/// accumulator initialises from the first observation, matching the convention
/// in [`IncrementalNormalizer`](crate::preprocessing::IncrementalNormalizer).
///
/// # Example
///
/// ```
/// use irithyll::preprocessing::TargetScaler;
/// use irithyll::preprocessing::StreamingTargetPreprocessor;
///
/// let mut scaler = TargetScaler::new();
///
/// // Inject several targets to build statistics.
/// for &y in &[10.0_f64, 20.0, 30.0, 40.0, 50.0] {
///     scaler.fit_transform(y);
/// }
///
/// // Fit-transform and invert should recover the original.
/// let original = 35.0;
/// let transformed = scaler.fit_transform(original);
/// let recovered = scaler.inverse_transform(transformed);
/// // After incorporating 35.0 the stats shift slightly, so recovered ≈ 35.0.
/// // For a pure round-trip without updating stats, transform + inverse:
/// let mut scaler2 = TargetScaler::new();
/// for &y in &[10.0_f64, 20.0, 30.0, 40.0, 50.0, 35.0] {
///     scaler2.fit_transform(y);
/// }
/// let t = (35.0 - scaler2.mean()) / scaler2.std();
/// let r = scaler2.inverse_transform(t);
/// assert!((r - 35.0).abs() < 1e-9);
/// ```
#[derive(Clone, Debug)]
pub struct TargetScaler {
    /// Number of samples seen (n for Welford).
    count: u64,
    /// Welford running mean.
    mean: f64,
    /// Welford running M2 (sum of squared deviations).
    m2: f64,
    /// Minimum variance to use in denominator (prevents /0).
    variance_floor: f64,
}

impl TargetScaler {
    /// Create a `TargetScaler` with the default variance floor of `1e-10`.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            variance_floor: 1e-10,
        }
    }

    /// Create a `TargetScaler` with a custom variance floor.
    ///
    /// The variance floor must be positive. Values around `1e-10` to `1e-6`
    /// are typical; larger values act as regularisation.
    ///
    /// # Panics
    ///
    /// Panics if `variance_floor <= 0.0`.
    pub fn with_variance_floor(variance_floor: f64) -> Self {
        assert!(
            variance_floor > 0.0,
            "TargetScaler: variance_floor must be > 0.0, got {}",
            variance_floor
        );
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            variance_floor,
        }
    }

    /// Current running mean of all targets seen so far.
    ///
    /// Returns `0.0` before any samples.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Current running standard deviation (sample std, Bessel-corrected for n>1).
    ///
    /// Returns `sqrt(variance_floor)` before any samples.
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Current running variance (sample variance, Bessel-corrected for n>1).
    ///
    /// Returns `variance_floor` before any samples.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            self.variance_floor
        } else {
            (self.m2 / (self.count - 1) as f64).max(self.variance_floor)
        }
    }
}

impl Default for TargetScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingTargetPreprocessor for TargetScaler {
    fn fit_transform(&mut self, target: f64) -> f64 {
        // Welford online update.
        self.count += 1;
        let delta = target - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = target - self.mean;
        self.m2 += delta * delta2;

        // Transform uses the *updated* statistics.
        (target - self.mean) / self.std()
    }

    fn inverse_transform(&self, transformed: f64) -> f64 {
        transformed * self.std() + self.mean
    }

    fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// TargetLog1pTransform
// ---------------------------------------------------------------------------

/// Streaming `log1p` transform for non-negative regression targets.
///
/// Applies `log(1 + y)` during training and `exp(y') - 1` during prediction.
/// This is useful for targets that are positive, right-skewed, and span
/// several orders of magnitude (e.g., counts, sales volumes, prices).
///
/// # Domain restriction
///
/// The forward transform requires `y >= -1.0`.  Values below this produce
/// `NaN` because `log(1 + y)` is undefined for `1 + y <= 0`. The inverse is
/// defined for all finite inputs, but very large `transformed` values will
/// produce `inf` (saturating the `f64` exponent).
///
/// If your targets can be negative and < -1.0, use [`TargetScaler`] instead.
///
/// # Invertibility
///
/// For `y >= -1.0`:
/// ```text
/// inverse_transform(fit_transform(y)) == expm1(log1p(y)) == y
/// ```
/// This is exact to floating-point precision (no accumulated statistics).
///
/// # Cold start
///
/// This transform is stateless — no Welford accumulator. It can be used
/// from the very first sample without a warm-up period.
///
/// # Example
///
/// ```
/// use irithyll::preprocessing::TargetLog1pTransform;
/// use irithyll::preprocessing::StreamingTargetPreprocessor;
///
/// let mut t = TargetLog1pTransform::new();
///
/// let original = 1000.0_f64;
/// let transformed = t.fit_transform(original);
/// let recovered = t.inverse_transform(transformed);
/// assert!((recovered - original).abs() < 1e-9);
///
/// // Domain: y >= -1.0 is required.
/// let transformed_zero = t.fit_transform(0.0);
/// assert_eq!(transformed_zero, 0.0); // log1p(0) = 0
/// let recovered_zero = t.inverse_transform(transformed_zero);
/// assert!((recovered_zero - 0.0).abs() < 1e-12);
/// ```
#[derive(Clone, Debug, Default)]
pub struct TargetLog1pTransform;

impl TargetLog1pTransform {
    /// Create a `TargetLog1pTransform` instance.
    pub fn new() -> Self {
        Self
    }
}

impl StreamingTargetPreprocessor for TargetLog1pTransform {
    /// Apply `log(1 + target)` and return.
    ///
    /// # Domain restriction
    ///
    /// Returns `NaN` if `target < -1.0`. This is intentional — passing
    /// out-of-domain values is a caller error, not silently corrected.
    fn fit_transform(&mut self, target: f64) -> f64 {
        target.ln_1p()
    }

    /// Apply `exp(transformed) - 1` and return.
    ///
    /// Exact inverse of `log1p` for any finite `transformed`.
    fn inverse_transform(&self, transformed: f64) -> f64 {
        transformed.exp_m1()
    }

    fn reset(&mut self) {
        // Stateless — nothing to reset.
    }
}

// ---------------------------------------------------------------------------
// TargetEncoderPreprocessor
// ---------------------------------------------------------------------------

/// Streaming target preprocessor adapter for [`TargetEncoder`](crate::preprocessing::TargetEncoder).
///
/// Wraps a `TargetEncoder` so it can participate in the
/// [`StreamingTargetPreprocessor`] protocol. This enables `TargetEncoder` to
/// be first-class in a [`Pipeline`](crate::pipeline::Pipeline) via
/// [`PipelineBuilder::target_preprocessor`](crate::pipeline::PipelineBuilder::target_preprocessor).
///
/// # How it works
///
/// The adapter stores the feature vector alongside the target on each
/// `fit_transform` call. On `inverse_transform`, it returns the global mean
/// — the "best constant" prediction in the original target space.
///
/// # Invertibility note
///
/// `TargetEncoder` is a *feature-space* transformer (it encodes features using
/// the target), not a target-space transformer. A true round-trip inverse
/// (recovering the exact original target from an encoded feature vector) does
/// not exist without knowing which category generated the target. The
/// `inverse_transform` therefore returns the global mean — an approximation
/// that is consistent with the category-encoding forward direction. This is
/// documented here so users are not surprised.
///
/// If you need a true target-space transform with an exact inverse, use
/// [`TargetScaler`] or [`TargetLog1pTransform`] instead.
///
/// # Example
///
/// ```
/// use irithyll::preprocessing::{TargetEncoder, TargetEncoderPreprocessor};
/// use irithyll::preprocessing::StreamingTargetPreprocessor;
///
/// // Feature 0 is categorical.
/// let enc = TargetEncoder::new(vec![0]);
/// let mut tp = TargetEncoderPreprocessor::new(enc, vec![1.0, 5.0]);
///
/// // Train: category 1.0 → target ~10.
/// for _ in 0..50 {
///     tp.fit_transform(10.0);
/// }
///
/// // The global mean is ≈ 10.0; inverse_transform returns it.
/// let approx = tp.inverse_transform(0.0);
/// assert!((approx - 10.0).abs() < 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct TargetEncoderPreprocessor {
    inner: crate::preprocessing::TargetEncoder,
    /// Feature vector used in the most recent `fit_transform` call.
    /// Updated on every call so the encoder processes the correct features.
    last_features: Vec<f64>,
}

impl TargetEncoderPreprocessor {
    /// Wrap a `TargetEncoder`.
    ///
    /// `initial_features` is the feature vector used for the very first
    /// `fit_transform` call (before any sample has been seen). This avoids
    /// a special-case cold-start path.
    pub fn new(inner: crate::preprocessing::TargetEncoder, initial_features: Vec<f64>) -> Self {
        Self {
            inner,
            last_features: initial_features,
        }
    }

    /// Access the underlying `TargetEncoder`.
    pub fn encoder(&self) -> &crate::preprocessing::TargetEncoder {
        &self.inner
    }

    /// Mutable access to the underlying `TargetEncoder`.
    pub fn encoder_mut(&mut self) -> &mut crate::preprocessing::TargetEncoder {
        &mut self.inner
    }

    /// Update `last_features` so the next `fit_transform` encodes the correct
    /// category. Call this before `fit_transform` when the feature vector
    /// changes between samples.
    pub fn set_features(&mut self, features: Vec<f64>) {
        self.last_features = features;
    }
}

impl StreamingTargetPreprocessor for TargetEncoderPreprocessor {
    fn fit_transform(&mut self, target: f64) -> f64 {
        self.inner.update(&self.last_features, target);
        target // The target itself is not transformed; features are.
    }

    /// Returns the global mean -- the best constant prediction in original
    /// target scale. This is an approximation; see struct-level doc.
    fn inverse_transform(&self, _transformed: f64) -> f64 {
        self.inner.global_mean()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // -- TargetScaler -------------------------------------------------------

    #[test]
    fn target_scaler_fit_transform_then_inverse_recovers_original() {
        // After seeing enough samples, inverse_transform(fit_transform(y))
        // should return approximately the original target.
        // Note: fit_transform updates running stats with y, then z-scores with
        // updated stats. The inverse uses the same (now updated) stats, so the
        // round-trip is exact.
        let mut scaler = TargetScaler::new();

        // Prime the scaler with several diverse samples.
        for &y in &[1.0_f64, 2.0, 3.0, 4.0, 5.0] {
            scaler.fit_transform(y);
        }

        let test_targets = [0.0_f64, 6.0, 3.0, -1.0, 100.0];
        for &original in &test_targets {
            let transformed = scaler.fit_transform(original);
            let recovered = scaler.inverse_transform(transformed);
            assert!(
                (recovered - original).abs() < EPS,
                "target_scaler round-trip failed for {}: transformed={}, recovered={}",
                original,
                transformed,
                recovered
            );
        }
    }

    #[test]
    fn target_scaler_reset_clears_state() {
        let mut scaler = TargetScaler::new();
        for &y in &[10.0_f64, 20.0, 30.0] {
            scaler.fit_transform(y);
        }
        assert!(scaler.mean() != 0.0);
        scaler.reset();
        assert_eq!(scaler.mean(), 0.0);
        assert_eq!(scaler.variance(), scaler.variance_floor);
    }

    #[test]
    fn target_scaler_cold_start_is_finite() {
        let mut scaler = TargetScaler::new();
        let t = scaler.fit_transform(42.0);
        assert!(t.is_finite(), "first fit_transform should be finite");
        let r = scaler.inverse_transform(t);
        assert!(
            (r - 42.0).abs() < EPS,
            "cold-start round-trip failed: {}",
            r
        );
    }

    #[test]
    fn target_scaler_accumulates_statistics_online() {
        // The Welford scaler updates mean and variance with each call to
        // fit_transform, using the UPDATED statistics to z-score.  The output
        // sequence is therefore heteroscedastic (each sample is scored against
        // different stats), so their sample variance is NOT 1.0.
        //
        // What IS invariant: after N samples the Welford variance estimate equals
        // the standard sample variance of the N observed targets. We verify this
        // offline to confirm the accumulator is correct.
        let mut scaler = TargetScaler::new();
        let n = 100_usize;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        for &y in &data {
            scaler.fit_transform(y);
        }

        // Offline reference: sample mean and sample variance of data.
        let ref_mean: f64 = data.iter().sum::<f64>() / n as f64;
        let ref_var: f64 =
            data.iter().map(|&y| (y - ref_mean).powi(2)).sum::<f64>() / (n - 1) as f64;

        assert!(
            (scaler.mean() - ref_mean).abs() < 1e-9,
            "Welford mean diverged from reference: {} vs {}",
            scaler.mean(),
            ref_mean
        );
        assert!(
            (scaler.variance() - ref_var).abs() < 1e-6,
            "Welford variance diverged from reference: {} vs {}",
            scaler.variance(),
            ref_var
        );
    }

    // -- TargetLog1pTransform -----------------------------------------------

    #[test]
    fn target_log1p_transform_inverse_correct_for_positive() {
        let mut t = TargetLog1pTransform::new();
        let test_values = [0.0_f64, 0.5, 1.0, 10.0, 100.0, 1_000_000.0];
        for &original in &test_values {
            let transformed = t.fit_transform(original);
            let recovered = t.inverse_transform(transformed);
            assert!(
                (recovered - original).abs() < EPS,
                "log1p round-trip failed for {}: transformed={}, recovered={}",
                original,
                transformed,
                recovered
            );
        }
    }

    #[test]
    fn target_log1p_transform_boundary_zero() {
        let mut t = TargetLog1pTransform::new();
        let transformed = t.fit_transform(0.0);
        assert_eq!(transformed, 0.0, "log1p(0+1) = log(1) = 0");
        let recovered = t.inverse_transform(transformed);
        assert!(
            recovered.abs() < EPS,
            "expm1(0) should be 0, got {}",
            recovered
        );
    }

    #[test]
    fn target_log1p_transform_out_of_domain_produces_nan() {
        let mut t = TargetLog1pTransform::new();
        // y = -2.0 → log1p(-2.0) = log(-1.0) = NaN
        let result = t.fit_transform(-2.0);
        assert!(
            result.is_nan(),
            "out-of-domain value should produce NaN, got {}",
            result
        );
    }

    #[test]
    fn target_log1p_transform_stateless_reset_no_op() {
        let mut t = TargetLog1pTransform::new();
        t.fit_transform(5.0);
        t.fit_transform(10.0);
        t.reset(); // no-op for stateless transform
                   // After reset the transform should behave identically.
        let transformed = t.fit_transform(42.0);
        let recovered = t.inverse_transform(transformed);
        assert!(
            (recovered - 42.0).abs() < EPS,
            "after reset round-trip failed: {}",
            recovered
        );
    }

    // -- TargetEncoderPreprocessor ------------------------------------------

    #[test]
    fn target_encoder_handles_categorical_streaming() {
        use crate::preprocessing::TargetEncoder;

        let enc = TargetEncoder::new(vec![0]);
        let mut tp = TargetEncoderPreprocessor::new(enc, vec![1.0, 5.0]);

        // Category 1.0 → target ~10.0 (50 samples).
        for _ in 0..50 {
            tp.fit_transform(10.0);
        }
        // Change features to category 2.0.
        tp.set_features(vec![2.0, 5.0]);
        // Category 2.0 → target ~20.0 (50 samples).
        for _ in 0..50 {
            tp.fit_transform(20.0);
        }

        // Global mean should be ≈ 15.0 (equal mix of 10 and 20).
        let global_mean = tp.encoder().global_mean();
        assert!(
            (global_mean - 15.0).abs() < 0.5,
            "global mean expected ~15.0, got {}",
            global_mean
        );
        // inverse_transform returns the global mean.
        let approx = tp.inverse_transform(0.0);
        assert!(
            (approx - global_mean).abs() < EPS,
            "inverse_transform should return global_mean, got {}",
            approx
        );
    }

    #[test]
    fn target_encoder_preprocessor_reset_clears_encoder() {
        use crate::preprocessing::TargetEncoder;

        let enc = TargetEncoder::new(vec![0]);
        let mut tp = TargetEncoderPreprocessor::new(enc, vec![1.0]);
        for _ in 0..10 {
            tp.fit_transform(5.0);
        }
        assert!(tp.encoder().global_mean() != 0.0);
        tp.reset();
        assert_eq!(tp.encoder().global_mean(), 0.0);
    }
}
