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
//!
//! # Capability Traits
//!
//! Opt-in capability traits narrow the required bound to the actual capability
//! a wrapper or algorithm needs, enabling cleaner type-level documentation:
//!
//! - [`HasReadout`] -- models with a linear RLS readout layer (neural models,
//!   KRLS, RLS). Used by `ProjectedLearner` supervised-projection path.
//! - [`Tunable`] -- models that expose diagnostics and accept LR / lambda
//!   adjustments from AutoML components.
//! - [`Structural`] -- models whose capacity can grow or shrink at runtime
//!   (tree ensembles: SGBT, ARF, Mondrian).

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
/// | [`replacement_count`](Self::replacement_count) | Total internal model replacements (0 by default) |
/// | [`readout_weights`](Self::readout_weights) | RLS readout weights for supervised projection (`None` by default) |
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

    /// Train on a single observation with an explicit distillation weight.
    ///
    /// Used by the knowledge-distillation path (`distill` feature) to replay
    /// pseudo-targets from dominated candidates into the winner's model with a
    /// down-weighted loss contribution.
    ///
    /// The default implementation delegates to [`train_one`](Self::train_one),
    /// forwarding `weight` directly. Models that support weighted training
    /// (e.g. `DistributionalSGBT`) therefore use the weight correctly. Models
    /// that internally ignore the weight field still compile without changes --
    /// the default is correct and transparent for both cases.
    ///
    /// Non-distillation consumers are unaffected: this method is not called by
    /// any non-distillation code path.
    fn train_one_weighted(&mut self, features: &[f64], target: f64, weight: f64) {
        self.train_one(features, target, weight);
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
    /// diagnostic adaptor in the auto-builder pipeline.
    ///
    /// Default: all zeros (model does not provide diagnostics). Models with
    /// internal diagnostic caches (e.g. SGBT, DistributionalSGBT) override
    /// this to return real computed values.
    ///
    /// # Deprecation
    ///
    /// Prefer `<T as Tunable>::diagnostics_array(model)` when the concrete type
    /// is known, or hold a `Box<dyn Tunable>` when dynamic dispatch over only
    /// tunable models is required. This shim keeps trait-object callers
    /// (`Box<dyn StreamingLearner>`) working until v11.0 removes it.
    #[deprecated(
        since = "10.0.0",
        note = "use the `Tunable` capability trait instead: `<T as Tunable>::diagnostics_array(model)` or hold `Box<dyn Tunable>`"
    )]
    #[doc(hidden)]
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
    ///
    /// # Deprecation
    ///
    /// Prefer `<T as Tunable>::adjust_config(model, lr, lambda)` when the
    /// concrete type is known. This shim keeps existing trait-object callers
    /// working until v11.0 removes it.
    #[deprecated(
        since = "10.0.0",
        note = "use the `Tunable` capability trait instead: `<T as Tunable>::adjust_config(model, lr_mult, lambda_delta)` or hold `Box<dyn Tunable>`"
    )]
    #[doc(hidden)]
    fn adjust_config(&mut self, _lr_multiplier: f64, _lambda_delta: f64) {}

    /// Apply structural changes at model replacement boundaries.
    ///
    /// * `depth_delta` -- adjust maximum tree depth (+1, -1, or 0).
    /// * `steps_delta` -- adjust number of ensemble steps (+2, -2, or 0).
    ///
    /// Structural changes take effect on the *next* tree replacement, not
    /// immediately. Default: no-op for models without structural config.
    ///
    /// # Deprecation
    ///
    /// Prefer `<T as Structural>::apply_structural_change(model, ...)` when
    /// the concrete type is known. This shim keeps existing trait-object
    /// callers working until v11.0 removes it.
    #[deprecated(
        since = "10.0.0",
        note = "use the `Structural` capability trait instead: `<T as Structural>::apply_structural_change(model, depth_delta, steps_delta)` or hold `Box<dyn Structural>`"
    )]
    #[doc(hidden)]
    fn apply_structural_change(&mut self, _depth_delta: i32, _steps_delta: i32) {}

    /// Total number of internal model replacements (e.g. tree replacements
    /// triggered by drift detection or max-tree-samples).
    ///
    /// External callers (e.g. the auto-builder) use this to detect when a
    /// structural boundary has occurred and apply queued structural changes.
    /// Default: 0 for models without replacement semantics.
    ///
    /// # Deprecation
    ///
    /// Prefer `<T as Structural>::replacement_count(model)` when the concrete
    /// type is known. This shim keeps existing trait-object callers working
    /// until v11.0 removes it.
    #[deprecated(
        since = "10.0.0",
        note = "use the `Structural` capability trait instead: `<T as Structural>::replacement_count(model)` or hold `Box<dyn Structural>`"
    )]
    #[doc(hidden)]
    fn replacement_count(&self) -> u64 {
        0
    }

    /// Manually trigger a proactive prune check.
    ///
    /// Returns `true` if an internal component was pruned/replaced.
    /// Default: no-op (returns `false`).
    ///
    /// # Deprecation
    ///
    /// Prefer `<T as Structural>::check_proactive_prune(model)` when the
    /// concrete type is known. This shim keeps existing trait-object callers
    /// working until v11.0 removes it.
    #[deprecated(
        since = "10.0.0",
        note = "use the `Structural` capability trait instead: `<T as Structural>::check_proactive_prune(model)` or hold `Box<dyn Structural>`"
    )]
    #[doc(hidden)]
    fn check_proactive_prune(&mut self) -> bool {
        false
    }

    /// Dynamically set the contribution accuracy EWMA half-life.
    ///
    /// Recomputes `prune_alpha` so each correction batch contributes equally
    /// regardless of size. Default: no-op.
    ///
    /// # Deprecation
    ///
    /// Prefer `<T as Structural>::set_prune_half_life(model, hl)` when the
    /// concrete type is known. This shim keeps existing trait-object callers
    /// working until v11.0 removes it.
    #[deprecated(
        since = "10.0.0",
        note = "use the `Structural` capability trait instead: `<T as Structural>::set_prune_half_life(model, hl)` or hold `Box<dyn Structural>`"
    )]
    #[doc(hidden)]
    fn set_prune_half_life(&mut self, _hl: usize) {}

    /// Return the readout weight vector for supervised projection, if available.
    ///
    /// Models with an RLS readout layer return `Some(&weights)`. Models
    /// without (KAN, SpikeNet, SGBT, etc.) return `None`. Used by
    /// `ProjectedLearner` for supervised projection updates.
    ///
    /// # Deprecation
    ///
    /// Prefer `<T as HasReadout>::readout_weights(model)` when the concrete
    /// type is known, or hold a `Box<dyn HasReadout>`. This shim keeps
    /// existing trait-object callers working until v11.0 removes it.
    #[deprecated(
        since = "10.0.0",
        note = "use the `HasReadout` capability trait instead: `<T as HasReadout>::readout_weights(model)` or hold `Box<dyn HasReadout>`"
    )]
    #[doc(hidden)]
    fn readout_weights(&self) -> Option<&[f64]> {
        None
    }

    /// Optional tree-level structure diagnostics.
    ///
    /// Returns per-tree: `(depth, n_leaves, leaf_weight_mean, leaf_weight_std, samples_seen)`.
    /// Default: empty vec (model has no trees).
    ///
    /// # Deprecation
    ///
    /// Prefer `<T as Structural>::tree_structure(model)` when the concrete type
    /// is known. This shim keeps existing trait-object callers working until
    /// v11.0 removes it.
    #[deprecated(
        since = "10.0.0",
        note = "use the `Structural` capability trait instead: `<T as Structural>::tree_structure(model)` or hold `Box<dyn Structural>`"
    )]
    #[doc(hidden)]
    fn tree_structure(&self) -> Vec<(usize, usize, f64, f64, u64)> {
        Vec::new()
    }
}

// ===========================================================================
// Capability traits
// ===========================================================================

/// Models that expose a linear readout weight vector.
///
/// Implemented by models with an RLS readout layer: neural models (ESN, Mamba,
/// KAN, TTT, sLSTM, HGRN2, mGRADE, attention variants), kernel models (KRLS),
/// and linear models (RLS). Used by `ProjectedLearner` for supervised
/// projection updates.
///
/// # Object Safety
///
/// This trait is object-safe. `Box<dyn HasReadout>` is a legal type.
pub trait HasReadout: StreamingLearner {
    /// The linear readout weight vector.
    ///
    /// For RLS-family models this is the full coefficient vector including
    /// the bias term if one is used. Length matches the model's internal
    /// feature dimensionality.
    fn readout_weights(&self) -> &[f64];
}

/// Models that expose diagnostics and accept smooth hyperparameter adjustments.
///
/// Implemented by models touched by AutoML components: SGBT, DistributionalSGBT,
/// RLS, KAN, TTT, ESN, mGRADE, and any model with tunable LR or regularization.
///
/// # Object Safety
///
/// This trait is object-safe. `Box<dyn Tunable>` is a legal type.
pub trait Tunable: StreamingLearner {
    /// Raw diagnostic signals for adaptive tuning.
    ///
    /// Returns `[residual_alignment, reg_sensitivity, depth_sufficiency,
    /// effective_dof, uncertainty]`. These five signals drive the diagnostic
    /// adaptor in the AutoML pipeline.
    fn diagnostics_array(&self) -> [f64; 5];

    /// Apply smooth learning rate and regularization adjustments.
    ///
    /// * `lr_multiplier` -- scales the current learning rate (1.0 = no
    ///   change, 0.99 = 1% decrease, 1.01 = 1% increase).
    /// * `lambda_delta` -- additive delta applied to the L2 regularization
    ///   parameter (0.0 = no change, positive = increase regularization).
    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64);
}

/// Models whose internal capacity can grow or shrink at runtime.
///
/// Implemented by tree ensemble models: SGBT, AdaptiveRandomForest,
/// DistributionalSGBT, BaggedSGBT, stacked ensembles that delegate to trees.
///
/// # Object Safety
///
/// This trait is object-safe. `Box<dyn Structural>` is a legal type.
pub trait Structural: StreamingLearner {
    /// Apply depth/step changes that take effect at the next tree replacement.
    ///
    /// * `depth_delta` -- signed adjustment to maximum tree depth (+1, -1, 0).
    /// * `steps_delta` -- signed adjustment to ensemble step count (+2, -2, 0).
    fn apply_structural_change(&mut self, depth_delta: i32, steps_delta: i32);

    /// Total number of internal model replacements since creation or last reset.
    ///
    /// External callers use this counter to detect replacement boundaries and
    /// apply queued structural changes.
    fn replacement_count(&self) -> u64;

    /// Manually trigger a proactive prune check.
    ///
    /// Returns `true` if an internal component was pruned or replaced.
    /// Defaults to `false` (no-op) for models without proactive pruning.
    fn check_proactive_prune(&mut self) -> bool {
        false
    }

    /// Dynamically set the contribution-accuracy EWMA half-life.
    ///
    /// Recomputes `prune_alpha` so each correction batch contributes equally
    /// regardless of batch size. Default: no-op for models without an EWMA
    /// prune accumulator.
    fn set_prune_half_life(&mut self, _hl: usize) {}

    /// Per-tree structure diagnostics.
    ///
    /// Returns one tuple per tree:
    /// `(max_depth, n_leaves, leaf_weight_mean, leaf_weight_std, samples_seen)`.
    /// Defaults to an empty vec for models without tree diagnostics.
    fn tree_structure(&self) -> Vec<(usize, usize, f64, f64, u64)> {
        Vec::new()
    }
}

// ===========================================================================
// Object-safety smoke tests
// ===========================================================================
//
// These functions never run -- they only compile-check object safety.
// If any of the traits above inadvertently gains a non-object-safe method,
// the test below will fail to compile.

#[cfg(test)]
mod _object_safety_tests {
    use super::{HasReadout, StreamingLearner, Structural, Tunable};
    use alloc::boxed::Box;

    fn _object_safe_streaming_learner(_: Box<dyn StreamingLearner>) {}
    fn _object_safe_has_readout(_: Box<dyn HasReadout>) {}
    fn _object_safe_tunable(_: Box<dyn Tunable>) {}
    fn _object_safe_structural(_: Box<dyn Structural>) {}
}
