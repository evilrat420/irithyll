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

use alloc::vec::Vec;

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
/// | [`diagnostics_array`](Self::diagnostics_array) | Raw diagnostic signals for adaptive tuning (all zeros by default) |
/// | [`adjust_config`](Self::adjust_config) | Apply smooth LR/lambda adjustments (no-op by default) |
/// | [`apply_structural_change`](Self::apply_structural_change) | Apply depth/steps changes at replacement boundaries (no-op by default) |
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

    /// Raw diagnostic signals for adaptive tuning.
    ///
    /// Returns `[residual_alignment, reg_sensitivity, depth_sufficiency,
    /// effective_dof, uncertainty]`. These five signals drive the
    /// [`DiagnosticAdaptor`] in the auto-builder pipeline.
    ///
    /// Default: all zeros (model does not provide diagnostics). Models with
    /// internal diagnostic caches (e.g. SGBT, DistributionalSGBT) override
    /// this to return real computed values.
    fn diagnostics_array(&self) -> [f64; 5] {
        [0.0; 5]
    }

    /// Apply smooth learning rate and regularization adjustments.
    ///
    /// * `lr_multiplier` -- scales the current learning rate (1.0 = no change,
    ///   0.99 = 1% decrease, 1.01 = 1% increase).
    /// * `lambda_delta` -- added to the L2 regularization parameter
    ///   (0.0 = no change, positive = increase, negative = decrease).
    ///
    /// Default: no-op. Override for models with adjustable hyperparameters
    /// (e.g. SGBT, DistributionalSGBT).
    fn adjust_config(&mut self, _lr_multiplier: f64, _lambda_delta: f64) {}

    /// Apply structural changes at model replacement boundaries.
    ///
    /// * `depth_delta` -- adjust maximum tree depth (+1, -1, or 0).
    /// * `steps_delta` -- adjust number of ensemble steps (+2, -2, or 0).
    ///
    /// Structural changes take effect on the *next* tree replacement, not
    /// immediately. Default: no-op for models without structural config.
    fn apply_structural_change(&mut self, _depth_delta: i32, _steps_delta: i32) {}
}
