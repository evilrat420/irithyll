//! SGBT ensemble orchestrator — the core boosting loop.
//!
//! Implements Streaming Gradient Boosted Trees (Gunasekara et al., 2024):
//! a sequence of boosting steps, each owning a streaming tree and drift detector,
//! with automatic tree replacement when concept drift is detected.
//!
//! # Algorithm
//!
//! For each incoming sample `(x, y)`:
//! 1. Compute the current ensemble prediction: `F(x) = base + lr * Σ tree_s(x)`
//! 2. For each boosting step `s = 1..N`:
//!    - Compute gradient `g = loss.gradient(y, current_pred)`
//!    - Compute hessian `h = loss.hessian(y, current_pred)`
//!    - Feed `(x, g, h)` to tree `s` (which internally uses weighted squared loss)
//!    - Update `current_pred += lr * tree_s.predict(x)`
//! 3. The ensemble adapts incrementally, with each tree targeting the residual
//!    of all preceding trees.

pub mod bagged;
pub mod config;
pub mod multi_target;
pub mod multiclass;
pub mod parallel;
pub mod quantile_regressor;
pub mod replacement;
pub mod step;
pub mod variants;

use std::fmt;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::step::BoostingStep;
use crate::loss::squared::SquaredLoss;
use crate::loss::Loss;
use crate::sample::Observation;
#[allow(unused_imports)] // Used in doc links + tests
use crate::sample::Sample;
use crate::tree::builder::TreeConfig;

/// Type alias for an SGBT model using dynamic (boxed) loss dispatch.
///
/// Use this when the loss function is determined at runtime (e.g., when
/// deserializing a model from JSON where the loss type is stored as a tag).
///
/// For compile-time loss dispatch (preferred for performance), use
/// `SGBT<LogisticLoss>`, `SGBT<HuberLoss>`, etc.
pub type DynSGBT = SGBT<Box<dyn Loss>>;

/// Streaming Gradient Boosted Trees ensemble.
///
/// The primary entry point for training and prediction. Generic over `L: Loss`
/// so the loss function's gradient/hessian calls are monomorphized (inlined)
/// into the boosting hot loop — no virtual dispatch overhead.
///
/// The default type parameter `L = SquaredLoss` means `SGBT::new(config)`
/// creates a regression model without specifying the loss type explicitly.
///
/// # Examples
///
/// ```
/// use irithyll::{SGBTConfig, SGBT};
///
/// // Regression with squared loss (default):
/// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
/// let model = SGBT::new(config);
/// ```
///
/// ```
/// use irithyll::{SGBTConfig, SGBT};
/// use irithyll::loss::logistic::LogisticLoss;
///
/// // Classification with logistic loss — no Box::new()!
/// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
/// let model = SGBT::with_loss(config, LogisticLoss);
/// ```
pub struct SGBT<L: Loss = SquaredLoss> {
    /// Configuration.
    config: SGBTConfig,
    /// Boosting steps (one tree + drift detector each).
    steps: Vec<BoostingStep>,
    /// Loss function (monomorphized — no vtable).
    loss: L,
    /// Base prediction (initial constant, computed from first batch of targets).
    base_prediction: f64,
    /// Whether base_prediction has been initialized.
    base_initialized: bool,
    /// Running collection of initial targets for computing base_prediction.
    initial_targets: Vec<f64>,
    /// Number of initial targets to collect before setting base_prediction.
    initial_target_count: usize,
    /// Total samples trained.
    samples_seen: u64,
    /// RNG state for variant skip logic.
    rng_state: u64,
}

impl<L: Loss + Clone> Clone for SGBT<L> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            steps: self.steps.clone(),
            loss: self.loss.clone(),
            base_prediction: self.base_prediction,
            base_initialized: self.base_initialized,
            initial_targets: self.initial_targets.clone(),
            initial_target_count: self.initial_target_count,
            samples_seen: self.samples_seen,
            rng_state: self.rng_state,
        }
    }
}

impl<L: Loss> fmt::Debug for SGBT<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SGBT")
            .field("n_steps", &self.steps.len())
            .field("samples_seen", &self.samples_seen)
            .field("base_prediction", &self.base_prediction)
            .field("base_initialized", &self.base_initialized)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor for the default loss (SquaredLoss)
// ---------------------------------------------------------------------------

impl SGBT<SquaredLoss> {
    /// Create a new SGBT ensemble with squared loss (regression).
    ///
    /// This is the most common constructor. For classification or custom
    /// losses, use [`with_loss`](SGBT::with_loss).
    pub fn new(config: SGBTConfig) -> Self {
        Self::with_loss(config, SquaredLoss)
    }
}

// ---------------------------------------------------------------------------
// General impl for all Loss types
// ---------------------------------------------------------------------------

impl<L: Loss> SGBT<L> {
    /// Create a new SGBT ensemble with a specific loss function.
    ///
    /// The loss is stored by value (monomorphized), giving zero-cost
    /// gradient/hessian dispatch.
    ///
    /// ```
    /// use irithyll::{SGBTConfig, SGBT};
    /// use irithyll::loss::logistic::LogisticLoss;
    ///
    /// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
    /// let model = SGBT::with_loss(config, LogisticLoss);
    /// ```
    pub fn with_loss(config: SGBTConfig, loss: L) -> Self {
        let leaf_decay_alpha = config
            .leaf_half_life
            .map(|hl| (-(2.0_f64.ln()) / hl as f64).exp());

        let tree_config = TreeConfig::new()
            .max_depth(config.max_depth)
            .n_bins(config.n_bins)
            .lambda(config.lambda)
            .gamma(config.gamma)
            .grace_period(config.grace_period)
            .delta(config.delta)
            .feature_subsample_rate(config.feature_subsample_rate)
            .leaf_decay_alpha_opt(leaf_decay_alpha)
            .split_reeval_interval_opt(config.split_reeval_interval)
            .feature_types_opt(config.feature_types.clone())
            .gradient_clip_sigma_opt(config.gradient_clip_sigma)
            .monotone_constraints_opt(config.monotone_constraints.clone());

        let max_tree_samples = config.max_tree_samples;

        let steps: Vec<BoostingStep> = (0..config.n_steps)
            .map(|i| {
                let mut tc = tree_config.clone();
                tc.seed = config.seed ^ (i as u64);
                let detector = config.drift_detector.create();
                BoostingStep::new_with_max_samples(tc, detector, max_tree_samples)
            })
            .collect();

        let seed = config.seed;
        let initial_target_count = config.initial_target_count;
        Self {
            config,
            steps,
            loss,
            base_prediction: 0.0,
            base_initialized: false,
            initial_targets: Vec::new(),
            initial_target_count,
            samples_seen: 0,
            rng_state: seed,
        }
    }

    /// Train on a single observation.
    ///
    /// Accepts any type implementing [`Observation`], including [`Sample`],
    /// [`SampleRef`](crate::SampleRef), or tuples like `(&[f64], f64)` for
    /// zero-copy training.
    pub fn train_one(&mut self, sample: &impl Observation) {
        self.samples_seen += 1;
        let target = sample.target();
        let features = sample.features();

        // Initialize base prediction from first few targets
        if !self.base_initialized {
            self.initial_targets.push(target);
            if self.initial_targets.len() >= self.initial_target_count {
                self.base_prediction = self.loss.initial_prediction(&self.initial_targets);
                self.base_initialized = true;
                self.initial_targets.clear();
                self.initial_targets.shrink_to_fit();
            }
        }

        // Current prediction starts from base
        let mut current_pred = self.base_prediction;

        // Sequential boosting: each step targets the residual of all prior steps
        for step in &mut self.steps {
            let gradient = self.loss.gradient(target, current_pred);
            let hessian = self.loss.hessian(target, current_pred);
            let train_count = self
                .config
                .variant
                .train_count(hessian, &mut self.rng_state);

            let step_pred = step.train_and_predict(features, gradient, hessian, train_count);

            current_pred += self.config.learning_rate * step_pred;
        }
    }

    /// Train on a batch of observations.
    pub fn train_batch<O: Observation>(&mut self, samples: &[O]) {
        for sample in samples {
            self.train_one(sample);
        }
    }

    /// Train on a batch with periodic callback for cooperative yielding.
    ///
    /// The callback is invoked every `interval` samples with the number of
    /// samples processed so far. This allows long-running training to yield
    /// to other tasks in an async runtime, update progress bars, or perform
    /// periodic checkpointing.
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll::{SGBTConfig, SGBT};
    ///
    /// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
    /// let mut model = SGBT::new(config);
    /// let data: Vec<(Vec<f64>, f64)> = Vec::new(); // your data
    ///
    /// model.train_batch_with_callback(&data, 1000, |processed| {
    ///     println!("Trained {} samples", processed);
    /// });
    /// ```
    pub fn train_batch_with_callback<O: Observation, F: FnMut(usize)>(
        &mut self,
        samples: &[O],
        interval: usize,
        mut callback: F,
    ) {
        let interval = interval.max(1); // Prevent zero interval
        for (i, sample) in samples.iter().enumerate() {
            self.train_one(sample);
            if (i + 1) % interval == 0 {
                callback(i + 1);
            }
        }
        // Final callback if the total isn't a multiple of interval
        let total = samples.len();
        if total % interval != 0 {
            callback(total);
        }
    }

    /// Train on a random subsample of a batch using reservoir sampling.
    ///
    /// When `max_samples < samples.len()`, selects a representative subset
    /// using Algorithm R (Vitter, 1985) — a uniform random sample without
    /// replacement. The selected samples are then trained in their original
    /// order to preserve sequential dependencies.
    ///
    /// This is ideal for large replay buffers where training on the full
    /// dataset is prohibitively slow but a representative subset gives
    /// equivalent model quality (e.g., 1M of 4.3M samples with R²=0.997).
    ///
    /// When `max_samples >= samples.len()`, all samples are trained.
    pub fn train_batch_subsampled<O: Observation>(&mut self, samples: &[O], max_samples: usize) {
        if max_samples >= samples.len() {
            self.train_batch(samples);
            return;
        }

        // Reservoir sampling (Algorithm R) to select indices
        let mut reservoir: Vec<usize> = (0..max_samples).collect();
        let mut rng = self.rng_state;

        for i in max_samples..samples.len() {
            // Generate random index in [0, i]
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng % (i as u64 + 1)) as usize;
            if j < max_samples {
                reservoir[j] = i;
            }
        }

        self.rng_state = rng;

        // Sort to preserve original order (important for EWMA/drift state)
        reservoir.sort_unstable();

        // Train on the selected subset
        for &idx in &reservoir {
            self.train_one(&samples[idx]);
        }
    }

    /// Train on a batch with both subsampling and periodic callbacks.
    ///
    /// Combines reservoir subsampling with cooperative yield points.
    /// Ideal for long-running daemon training where you need both
    /// efficiency (subsampling) and cooperation (yielding).
    pub fn train_batch_subsampled_with_callback<O: Observation, F: FnMut(usize)>(
        &mut self,
        samples: &[O],
        max_samples: usize,
        interval: usize,
        mut callback: F,
    ) {
        if max_samples >= samples.len() {
            self.train_batch_with_callback(samples, interval, callback);
            return;
        }

        // Reservoir sampling
        let mut reservoir: Vec<usize> = (0..max_samples).collect();
        let mut rng = self.rng_state;

        for i in max_samples..samples.len() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng % (i as u64 + 1)) as usize;
            if j < max_samples {
                reservoir[j] = i;
            }
        }

        self.rng_state = rng;
        reservoir.sort_unstable();

        let interval = interval.max(1);
        for (i, &idx) in reservoir.iter().enumerate() {
            self.train_one(&samples[idx]);
            if (i + 1) % interval == 0 {
                callback(i + 1);
            }
        }
        let total = reservoir.len();
        if total % interval != 0 {
            callback(total);
        }
    }

    /// Predict the raw output for a feature vector.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate * step.predict(features);
        }
        pred
    }

    /// Predict with loss transform applied (e.g., sigmoid for logistic loss).
    pub fn predict_transformed(&self, features: &[f64]) -> f64 {
        self.loss.predict_transform(self.predict(features))
    }

    /// Predict probability (alias for `predict_transformed`).
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        self.predict_transformed(features)
    }

    /// Predict with confidence estimation.
    ///
    /// Returns `(prediction, confidence)` where confidence = 1 / sqrt(sum_variance).
    /// Higher confidence indicates more certain predictions (leaves have seen
    /// more hessian mass). Confidence of 0.0 means the model has no information.
    ///
    /// This enables execution engines to modulate aggressiveness:
    /// - High confidence + favorable prediction → act immediately
    /// - Low confidence → fall back to simpler models or wait for more data
    ///
    /// The variance per tree is estimated as `1 / (H_sum + lambda)` at the
    /// leaf where the sample lands. The ensemble variance is the sum of
    /// per-tree variances (scaled by learning_rate²), and confidence is
    /// the reciprocal of the standard deviation.
    pub fn predict_with_confidence(&self, features: &[f64]) -> (f64, f64) {
        let mut pred = self.base_prediction;
        let mut total_variance = 0.0;
        let lr2 = self.config.learning_rate * self.config.learning_rate;

        for step in &self.steps {
            let (value, variance) = step.predict_with_variance(features);
            pred += self.config.learning_rate * value;
            total_variance += lr2 * variance;
        }

        let confidence = if total_variance > 0.0 && total_variance.is_finite() {
            1.0 / total_variance.sqrt()
        } else {
            0.0
        };

        (pred, confidence)
    }

    /// Batch prediction.
    pub fn predict_batch(&self, feature_matrix: &[Vec<f64>]) -> Vec<f64> {
        feature_matrix.iter().map(|f| self.predict(f)).collect()
    }

    /// Number of boosting steps.
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Total trees (active + alternates).
    pub fn n_trees(&self) -> usize {
        self.steps.len() + self.steps.iter().filter(|s| s.has_alternate()).count()
    }

    /// Total leaves across all active trees.
    pub fn total_leaves(&self) -> usize {
        self.steps.iter().map(|s| s.n_leaves()).sum()
    }

    /// Total samples trained.
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// The current base prediction.
    pub fn base_prediction(&self) -> f64 {
        self.base_prediction
    }

    /// Whether the base prediction has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.base_initialized
    }

    /// Access the configuration.
    pub fn config(&self) -> &SGBTConfig {
        &self.config
    }

    /// Immutable access to the boosting steps.
    ///
    /// Useful for model inspection and export (e.g., ONNX serialization).
    pub fn steps(&self) -> &[BoostingStep] {
        &self.steps
    }

    /// Immutable access to the loss function.
    pub fn loss(&self) -> &L {
        &self.loss
    }

    /// Feature importances based on accumulated split gains across all trees.
    ///
    /// Returns normalized importances (sum to 1.0) indexed by feature.
    /// Returns an empty Vec if no splits have occurred yet.
    pub fn feature_importances(&self) -> Vec<f64> {
        // Aggregate split gains across all boosting steps.
        let mut totals: Vec<f64> = Vec::new();
        for step in &self.steps {
            let gains = step.slot().split_gains();
            if totals.is_empty() && !gains.is_empty() {
                totals.resize(gains.len(), 0.0);
            }
            for (i, &g) in gains.iter().enumerate() {
                if i < totals.len() {
                    totals[i] += g;
                }
            }
        }

        // Normalize to sum to 1.0.
        let sum: f64 = totals.iter().sum();
        if sum > 0.0 {
            totals.iter_mut().for_each(|v| *v /= sum);
        }
        totals
    }

    /// Feature names, if configured.
    pub fn feature_names(&self) -> Option<&[String]> {
        self.config.feature_names.as_deref()
    }

    /// Feature importances paired with their names.
    ///
    /// Returns `None` if feature names are not configured. Otherwise returns
    /// `(name, importance)` pairs sorted by importance descending.
    pub fn named_feature_importances(&self) -> Option<Vec<(String, f64)>> {
        let names = self.config.feature_names.as_ref()?;
        let importances = self.feature_importances();
        let mut pairs: Vec<(String, f64)> = names
            .iter()
            .zip(importances.iter().chain(std::iter::repeat(&0.0)))
            .map(|(n, &v)| (n.clone(), v))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Some(pairs)
    }

    /// Train on a single sample with named features.
    ///
    /// Converts a `HashMap<String, f64>` of named features into a positional
    /// vector using the configured feature names. Missing features default to 0.0.
    ///
    /// # Panics
    ///
    /// Panics if `feature_names` is not configured.
    pub fn train_one_named(
        &mut self,
        features: &std::collections::HashMap<String, f64>,
        target: f64,
    ) {
        let names = self
            .config
            .feature_names
            .as_ref()
            .expect("train_one_named requires feature_names to be configured");
        let vec: Vec<f64> = names
            .iter()
            .map(|name| features.get(name).copied().unwrap_or(0.0))
            .collect();
        self.train_one(&(&vec[..], target));
    }

    /// Predict with named features.
    ///
    /// Converts named features into a positional vector, same as `train_one_named`.
    ///
    /// # Panics
    ///
    /// Panics if `feature_names` is not configured.
    pub fn predict_named(&self, features: &std::collections::HashMap<String, f64>) -> f64 {
        let names = self
            .config
            .feature_names
            .as_ref()
            .expect("predict_named requires feature_names to be configured");
        let vec: Vec<f64> = names
            .iter()
            .map(|name| features.get(name).copied().unwrap_or(0.0))
            .collect();
        self.predict(&vec)
    }

    /// Compute per-feature SHAP explanations for a prediction.
    ///
    /// Returns [`ShapValues`](crate::explain::treeshap::ShapValues) containing
    /// per-feature contributions and a base value. The invariant holds:
    /// `base_value + sum(values) ≈ self.predict(features)`.
    pub fn explain(&self, features: &[f64]) -> crate::explain::treeshap::ShapValues {
        crate::explain::treeshap::ensemble_shap(self, features)
    }

    /// Compute named SHAP explanations (requires `feature_names` configured).
    ///
    /// Returns `None` if feature names are not set. Otherwise returns
    /// [`NamedShapValues`](crate::explain::treeshap::NamedShapValues) with
    /// `(name, contribution)` pairs sorted by absolute contribution descending.
    pub fn explain_named(
        &self,
        features: &[f64],
    ) -> Option<crate::explain::treeshap::NamedShapValues> {
        let names = self.config.feature_names.as_ref()?;
        let shap = self.explain(features);
        let mut pairs: Vec<(String, f64)> = names
            .iter()
            .zip(shap.values.iter().chain(std::iter::repeat(&0.0)))
            .map(|(n, &v)| (n.clone(), v))
            .collect();
        pairs.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(crate::explain::treeshap::NamedShapValues {
            values: pairs,
            base_value: shap.base_value,
        })
    }

    /// Reset the ensemble to initial state.
    pub fn reset(&mut self) {
        for step in &mut self.steps {
            step.reset();
        }
        self.base_prediction = 0.0;
        self.base_initialized = false;
        self.initial_targets.clear();
        self.samples_seen = 0;
        self.rng_state = self.config.seed;
    }

    /// Serialize the model into a [`ModelState`](crate::serde_support::ModelState).
    ///
    /// Auto-detects the [`LossType`](crate::loss::LossType) from the loss
    /// function's [`Loss::loss_type()`] implementation.
    ///
    /// # Errors
    ///
    /// Returns [`IrithyllError::Serialization`](crate::IrithyllError::Serialization)
    /// if the loss does not implement `loss_type()` (returns `None`). For custom
    /// losses, use [`to_model_state_with`](Self::to_model_state_with) instead.
    #[cfg(any(feature = "serde-json", feature = "serde-bincode"))]
    pub fn to_model_state(&self) -> crate::error::Result<crate::serde_support::ModelState> {
        let loss_type = self.loss.loss_type().ok_or_else(|| {
            crate::error::IrithyllError::Serialization(
                "cannot auto-detect loss type for serialization: \
                 implement Loss::loss_type() or use to_model_state_with()"
                    .into(),
            )
        })?;
        Ok(self.to_model_state_with(loss_type))
    }

    /// Serialize the model with an explicit [`LossType`](crate::loss::LossType) tag.
    ///
    /// Use this for custom loss functions that don't implement `loss_type()`.
    #[cfg(any(feature = "serde-json", feature = "serde-bincode"))]
    pub fn to_model_state_with(
        &self,
        loss_type: crate::loss::LossType,
    ) -> crate::serde_support::ModelState {
        use crate::serde_support::{ModelState, StepSnapshot, TreeSnapshot};

        fn snapshot_tree(tree: &crate::tree::hoeffding::HoeffdingTree) -> TreeSnapshot {
            use crate::tree::StreamingTree;
            let arena = tree.arena();
            TreeSnapshot {
                feature_idx: arena.feature_idx.clone(),
                threshold: arena.threshold.clone(),
                left: arena.left.iter().map(|id| id.0).collect(),
                right: arena.right.iter().map(|id| id.0).collect(),
                leaf_value: arena.leaf_value.clone(),
                is_leaf: arena.is_leaf.clone(),
                depth: arena.depth.clone(),
                sample_count: arena.sample_count.clone(),
                n_features: tree.n_features(),
                samples_seen: tree.n_samples_seen(),
                rng_state: tree.rng_state(),
                categorical_mask: arena.categorical_mask.clone(),
            }
        }

        let steps = self
            .steps
            .iter()
            .map(|step| {
                let slot = step.slot();
                let tree_snap = snapshot_tree(slot.active_tree());
                let alt_snap = slot.alternate_tree().map(snapshot_tree);
                let drift_state = slot.detector().serialize_state();
                let alt_drift_state = slot.alt_detector().and_then(|d| d.serialize_state());
                StepSnapshot {
                    tree: tree_snap,
                    alternate_tree: alt_snap,
                    drift_state,
                    alt_drift_state,
                }
            })
            .collect();

        ModelState {
            config: self.config.clone(),
            loss_type,
            base_prediction: self.base_prediction,
            base_initialized: self.base_initialized,
            initial_targets: self.initial_targets.clone(),
            initial_target_count: self.initial_target_count,
            samples_seen: self.samples_seen,
            rng_state: self.rng_state,
            steps,
        }
    }
}

// ---------------------------------------------------------------------------
// DynSGBT: deserialization returns a dynamically-dispatched model
// ---------------------------------------------------------------------------

#[cfg(any(feature = "serde-json", feature = "serde-bincode"))]
impl SGBT<Box<dyn Loss>> {
    /// Reconstruct an SGBT model from a [`ModelState`](crate::serde_support::ModelState).
    ///
    /// Returns a [`DynSGBT`] (`SGBT<Box<dyn Loss>>`) because the concrete
    /// loss type is determined at runtime from the serialized tag.
    ///
    /// Rebuilds the full ensemble including tree topology and leaf values.
    /// Histogram accumulators are left empty and will rebuild from continued
    /// training. If drift detector state was serialized, it is restored;
    /// otherwise a fresh detector is created from the config.
    pub fn from_model_state(state: crate::serde_support::ModelState) -> Self {
        use crate::ensemble::replacement::TreeSlot;
        use crate::serde_support::TreeSnapshot;
        use crate::tree::hoeffding::HoeffdingTree;
        use crate::tree::node::{NodeId, TreeArena};

        fn rebuild_tree(snapshot: &TreeSnapshot, tree_config: TreeConfig) -> HoeffdingTree {
            let mut arena = TreeArena::new();
            let n = snapshot.feature_idx.len();

            // Rebuild arena from parallel vecs.
            for i in 0..n {
                arena.feature_idx.push(snapshot.feature_idx[i]);
                arena.threshold.push(snapshot.threshold[i]);
                arena.left.push(NodeId(snapshot.left[i]));
                arena.right.push(NodeId(snapshot.right[i]));
                arena.leaf_value.push(snapshot.leaf_value[i]);
                arena.is_leaf.push(snapshot.is_leaf[i]);
                arena.depth.push(snapshot.depth[i]);
                arena.sample_count.push(snapshot.sample_count[i]);
                // Backward compat: old snapshots have empty categorical_mask
                let mask = snapshot.categorical_mask.get(i).copied().flatten();
                arena.categorical_mask.push(mask);
            }

            HoeffdingTree::from_arena(
                tree_config,
                arena,
                snapshot.n_features,
                snapshot.samples_seen,
                snapshot.rng_state,
            )
        }

        let loss = state.loss_type.into_loss();

        let leaf_decay_alpha = state
            .config
            .leaf_half_life
            .map(|hl| (-(2.0_f64.ln()) / hl as f64).exp());
        let max_tree_samples = state.config.max_tree_samples;

        let steps = state
            .steps
            .iter()
            .enumerate()
            .map(|(i, step_snap)| {
                let tree_config = TreeConfig::new()
                    .max_depth(state.config.max_depth)
                    .n_bins(state.config.n_bins)
                    .lambda(state.config.lambda)
                    .gamma(state.config.gamma)
                    .grace_period(state.config.grace_period)
                    .delta(state.config.delta)
                    .feature_subsample_rate(state.config.feature_subsample_rate)
                    .leaf_decay_alpha_opt(leaf_decay_alpha)
                    .split_reeval_interval_opt(state.config.split_reeval_interval)
                    .feature_types_opt(state.config.feature_types.clone())
                    .gradient_clip_sigma_opt(state.config.gradient_clip_sigma)
                    .monotone_constraints_opt(state.config.monotone_constraints.clone())
                    .seed(state.config.seed ^ (i as u64));

                let active = rebuild_tree(&step_snap.tree, tree_config.clone());
                let alternate = step_snap
                    .alternate_tree
                    .as_ref()
                    .map(|snap| rebuild_tree(snap, tree_config.clone()));

                let mut detector = state.config.drift_detector.create();
                if let Some(ref ds) = step_snap.drift_state {
                    detector.restore_state(ds);
                }
                let mut slot = TreeSlot::from_trees(
                    active,
                    alternate,
                    tree_config,
                    detector,
                    max_tree_samples,
                );
                if let Some(ref ads) = step_snap.alt_drift_state {
                    if let Some(alt_det) = slot.alt_detector_mut() {
                        alt_det.restore_state(ads);
                    }
                }
                BoostingStep::from_slot(slot)
            })
            .collect();

        Self {
            config: state.config,
            steps,
            loss,
            base_prediction: state.base_prediction,
            base_initialized: state.base_initialized,
            initial_targets: state.initial_targets,
            initial_target_count: state.initial_target_count,
            samples_seen: state.samples_seen,
            rng_state: state.rng_state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .build()
            .unwrap()
    }

    #[test]
    fn new_model_predicts_zero() {
        let model = SGBT::new(default_config());
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(pred.abs() < 1e-12);
    }

    #[test]
    fn train_one_does_not_panic() {
        let mut model = SGBT::new(default_config());
        model.train_one(&Sample::new(vec![1.0, 2.0, 3.0], 5.0));
        assert_eq!(model.n_samples_seen(), 1);
    }

    #[test]
    fn prediction_changes_after_training() {
        let mut model = SGBT::new(default_config());
        let features = vec![1.0, 2.0, 3.0];
        for i in 0..100 {
            model.train_one(&Sample::new(features.clone(), (i as f64) * 0.1));
        }
        let pred = model.predict(&features);
        assert!(pred.is_finite());
    }

    #[test]
    fn linear_signal_rmse_improves() {
        let config = SGBTConfig::builder()
            .n_steps(20)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(3)
            .n_bins(16)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        let mut rng: u64 = 12345;
        let mut early_errors = Vec::new();
        let mut late_errors = Vec::new();

        for i in 0..500 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x1 = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x2 = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            let target = 2.0 * x1 + 3.0 * x2;

            let pred = model.predict(&[x1, x2]);
            let error = (pred - target).powi(2);

            if (50..150).contains(&i) {
                early_errors.push(error);
            }
            if i >= 400 {
                late_errors.push(error);
            }

            model.train_one(&Sample::new(vec![x1, x2], target));
        }

        let early_rmse = (early_errors.iter().sum::<f64>() / early_errors.len() as f64).sqrt();
        let late_rmse = (late_errors.iter().sum::<f64>() / late_errors.len() as f64).sqrt();

        assert!(
            late_rmse < early_rmse,
            "RMSE should decrease: early={:.4}, late={:.4}",
            early_rmse,
            late_rmse
        );
    }

    #[test]
    fn train_batch_equivalent_to_sequential() {
        let config = default_config();
        let mut model_seq = SGBT::new(config.clone());
        let mut model_batch = SGBT::new(config);

        let samples: Vec<Sample> = (0..20)
            .map(|i| {
                let x = i as f64 * 0.5;
                Sample::new(vec![x, x * 2.0], x * 3.0)
            })
            .collect();

        for s in &samples {
            model_seq.train_one(s);
        }
        model_batch.train_batch(&samples);

        let pred_seq = model_seq.predict(&[1.0, 2.0]);
        let pred_batch = model_batch.predict(&[1.0, 2.0]);

        assert!(
            (pred_seq - pred_batch).abs() < 1e-10,
            "seq={}, batch={}",
            pred_seq,
            pred_batch
        );
    }

    #[test]
    fn reset_returns_to_initial() {
        let mut model = SGBT::new(default_config());
        for i in 0..100 {
            model.train_one(&Sample::new(vec![1.0, 2.0], i as f64));
        }
        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.is_initialized());
        assert!(model.predict(&[1.0, 2.0]).abs() < 1e-12);
    }

    #[test]
    fn base_prediction_initializes() {
        let mut model = SGBT::new(default_config());
        for i in 0..50 {
            model.train_one(&Sample::new(vec![1.0], i as f64 + 100.0));
        }
        assert!(model.is_initialized());
        let expected = (100.0 + 149.0) / 2.0;
        assert!((model.base_prediction() - expected).abs() < 1.0);
    }

    #[test]
    fn with_loss_uses_custom_loss() {
        use crate::loss::logistic::LogisticLoss;
        let model = SGBT::with_loss(default_config(), LogisticLoss);
        let pred = model.predict_transformed(&[1.0, 2.0]);
        assert!(
            (pred - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5, got {}",
            pred
        );
    }

    #[test]
    fn ewma_config_propagates_and_trains() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(3)
            .n_bins(16)
            .leaf_half_life(50)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        for i in 0..200 {
            let x = (i as f64) * 0.1;
            model.train_one(&Sample::new(vec![x, x * 2.0], x * 3.0));
        }

        let pred = model.predict(&[1.0, 2.0]);
        assert!(
            pred.is_finite(),
            "EWMA-enabled model should produce finite predictions, got {}",
            pred
        );
    }

    #[test]
    fn max_tree_samples_config_propagates() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(3)
            .n_bins(16)
            .max_tree_samples(200)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        for i in 0..500 {
            let x = (i as f64) * 0.1;
            model.train_one(&Sample::new(vec![x, x * 2.0], x * 3.0));
        }

        let pred = model.predict(&[1.0, 2.0]);
        assert!(
            pred.is_finite(),
            "max_tree_samples model should produce finite predictions, got {}",
            pred
        );
    }

    #[test]
    fn split_reeval_config_propagates() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(2)
            .n_bins(16)
            .split_reeval_interval(50)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        let mut rng: u64 = 12345;
        for _ in 0..1000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x1 = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x2 = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            let target = 2.0 * x1 + 3.0 * x2;
            model.train_one(&Sample::new(vec![x1, x2], target));
        }

        let pred = model.predict(&[1.0, 2.0]);
        assert!(
            pred.is_finite(),
            "split re-eval model should produce finite predictions, got {}",
            pred
        );
    }

    #[test]
    fn loss_accessor_works() {
        use crate::loss::logistic::LogisticLoss;
        let model = SGBT::with_loss(default_config(), LogisticLoss);
        // Verify we can access the concrete loss type
        let _loss: &LogisticLoss = model.loss();
        assert_eq!(_loss.n_outputs(), 1);
    }

    #[test]
    fn clone_produces_independent_copy() {
        let config = default_config();
        let mut model = SGBT::new(config);

        // Train the original on some data
        let mut rng: u64 = 99999;
        for _ in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            let target = 2.0 * x + 1.0;
            model.train_one(&Sample::new(vec![x], target));
        }

        // Clone the model
        let mut cloned = model.clone();

        // Both should produce identical predictions
        let test_features = [3.0];
        let pred_original = model.predict(&test_features);
        let pred_cloned = cloned.predict(&test_features);
        assert!(
            (pred_original - pred_cloned).abs() < 1e-12,
            "clone should predict identically: original={pred_original}, cloned={pred_cloned}"
        );

        // Train only the clone further — models should diverge
        for _ in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            let target = -3.0 * x + 5.0; // Different relationship
            cloned.train_one(&Sample::new(vec![x], target));
        }

        let pred_original_after = model.predict(&test_features);
        let pred_cloned_after = cloned.predict(&test_features);

        // Original should be unchanged
        assert!(
            (pred_original - pred_original_after).abs() < 1e-12,
            "original should be unchanged after training clone"
        );

        // Clone should have diverged
        assert!(
            (pred_original_after - pred_cloned_after).abs() > 1e-6,
            "clone should diverge after independent training"
        );
    }

    // -------------------------------------------------------------------
    // predict_with_confidence returns finite values
    // -------------------------------------------------------------------
    #[test]
    fn predict_with_confidence_finite() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .grace_period(10)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        // Train enough to initialize
        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&(&[x, x * 2.0][..], x + 1.0));
        }

        let (pred, confidence) = model.predict_with_confidence(&[1.0, 2.0]);
        assert!(pred.is_finite(), "prediction should be finite");
        assert!(confidence.is_finite(), "confidence should be finite");
        assert!(
            confidence > 0.0,
            "confidence should be positive after training"
        );
    }

    // -------------------------------------------------------------------
    // predict_with_confidence positive after training
    // -------------------------------------------------------------------
    #[test]
    fn predict_with_confidence_positive_after_training() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .grace_period(10)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        // Train enough to initialize and build structure
        for i in 0..200 {
            let x = i as f64 * 0.05;
            model.train_one(&(&[x][..], x * 2.0));
        }

        let (pred, confidence) = model.predict_with_confidence(&[1.0]);

        assert!(pred.is_finite(), "prediction should be finite");
        assert!(
            confidence > 0.0 && confidence.is_finite(),
            "confidence should be finite and positive, got {}",
            confidence,
        );

        // Multiple queries should give consistent confidence
        let (pred2, conf2) = model.predict_with_confidence(&[1.0]);
        assert!(
            (pred - pred2).abs() < 1e-12,
            "same input should give same prediction"
        );
        assert!(
            (confidence - conf2).abs() < 1e-12,
            "same input should give same confidence"
        );
    }

    // -------------------------------------------------------------------
    // predict_with_confidence agrees with predict on point estimate
    // -------------------------------------------------------------------
    #[test]
    fn predict_with_confidence_matches_predict() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .grace_period(10)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        for i in 0..200 {
            let x = (i as f64 - 100.0) * 0.01;
            model.train_one(&(&[x, x * x][..], x * 3.0 + 1.0));
        }

        let pred = model.predict(&[0.5, 0.25]);
        let (conf_pred, _) = model.predict_with_confidence(&[0.5, 0.25]);

        assert!(
            (pred - conf_pred).abs() < 1e-10,
            "prediction mismatch: predict()={} vs predict_with_confidence()={}",
            pred,
            conf_pred,
        );
    }

    // -------------------------------------------------------------------
    // gradient clipping config round-trips through builder
    // -------------------------------------------------------------------
    #[test]
    fn gradient_clip_config_builder() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .gradient_clip_sigma(3.0)
            .build()
            .unwrap();

        assert_eq!(config.gradient_clip_sigma, Some(3.0));
    }

    // -------------------------------------------------------------------
    // monotonic constraints config round-trips through builder
    // -------------------------------------------------------------------
    #[test]
    fn monotone_constraints_config_builder() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .monotone_constraints(vec![1, -1, 0])
            .build()
            .unwrap();

        assert_eq!(config.monotone_constraints, Some(vec![1, -1, 0]));
    }

    // -------------------------------------------------------------------
    // monotonic constraints validation rejects invalid values
    // -------------------------------------------------------------------
    #[test]
    fn monotone_constraints_invalid_value_rejected() {
        let result = SGBTConfig::builder()
            .n_steps(10)
            .monotone_constraints(vec![1, 2, 0])
            .build();

        assert!(result.is_err(), "constraint value 2 should be rejected");
    }

    // -------------------------------------------------------------------
    // gradient clipping validation rejects non-positive sigma
    // -------------------------------------------------------------------
    #[test]
    fn gradient_clip_sigma_negative_rejected() {
        let result = SGBTConfig::builder()
            .n_steps(10)
            .gradient_clip_sigma(-1.0)
            .build();

        assert!(result.is_err(), "negative sigma should be rejected");
    }

    // -------------------------------------------------------------------
    // gradient clipping ensemble-level reduces outlier impact
    // -------------------------------------------------------------------
    #[test]
    fn gradient_clipping_reduces_outlier_impact() {
        // Without clipping
        let config_no_clip = SGBTConfig::builder()
            .n_steps(5)
            .grace_period(10)
            .build()
            .unwrap();
        let mut model_no_clip = SGBT::new(config_no_clip);

        // With clipping
        let config_clip = SGBTConfig::builder()
            .n_steps(5)
            .grace_period(10)
            .gradient_clip_sigma(3.0)
            .build()
            .unwrap();
        let mut model_clip = SGBT::new(config_clip);

        // Train both on identical normal data
        for i in 0..100 {
            let x = (i as f64) * 0.01;
            let sample = (&[x][..], x * 2.0);
            model_no_clip.train_one(&sample);
            model_clip.train_one(&sample);
        }

        let pred_no_clip_before = model_no_clip.predict(&[0.5]);
        let pred_clip_before = model_clip.predict(&[0.5]);

        // Inject outlier
        let outlier = (&[0.5_f64][..], 10000.0);
        model_no_clip.train_one(&outlier);
        model_clip.train_one(&outlier);

        let pred_no_clip_after = model_no_clip.predict(&[0.5]);
        let pred_clip_after = model_clip.predict(&[0.5]);

        let delta_no_clip = (pred_no_clip_after - pred_no_clip_before).abs();
        let delta_clip = (pred_clip_after - pred_clip_before).abs();

        // Clipped model should be less affected by the outlier
        assert!(
            delta_clip <= delta_no_clip + 1e-10,
            "clipped model should be less affected: delta_clip={}, delta_no_clip={}",
            delta_clip,
            delta_no_clip,
        );
    }

    // -------------------------------------------------------------------
    // train_batch_with_callback fires at correct intervals
    // -------------------------------------------------------------------
    #[test]
    fn train_batch_with_callback_fires() {
        let config = SGBTConfig::builder()
            .n_steps(3)
            .grace_period(5)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        let data: Vec<(Vec<f64>, f64)> = (0..25)
            .map(|i| (vec![i as f64 * 0.1], i as f64 * 0.5))
            .collect();

        let mut callbacks = Vec::new();
        model.train_batch_with_callback(&data, 10, |n| {
            callbacks.push(n);
        });

        // Should fire at 10, 20, and 25 (final)
        assert_eq!(callbacks, vec![10, 20, 25]);
    }

    // -------------------------------------------------------------------
    // train_batch_subsampled produces deterministic subset
    // -------------------------------------------------------------------
    #[test]
    fn train_batch_subsampled_trains_subset() {
        let config = SGBTConfig::builder()
            .n_steps(3)
            .grace_period(5)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        let data: Vec<(Vec<f64>, f64)> = (0..100)
            .map(|i| (vec![i as f64 * 0.01], i as f64 * 0.1))
            .collect();

        // Train on only 20 of 100 samples
        model.train_batch_subsampled(&data, 20);

        // Model should have seen some samples
        assert!(
            model.n_samples_seen() > 0,
            "model should have trained on subset"
        );
        assert!(
            model.n_samples_seen() <= 20,
            "model should have trained at most 20 samples, got {}",
            model.n_samples_seen(),
        );
    }

    // -------------------------------------------------------------------
    // train_batch_subsampled full dataset = train_batch
    // -------------------------------------------------------------------
    #[test]
    fn train_batch_subsampled_full_equals_batch() {
        let config1 = SGBTConfig::builder()
            .n_steps(3)
            .grace_period(5)
            .build()
            .unwrap();
        let config2 = config1.clone();

        let mut model1 = SGBT::new(config1);
        let mut model2 = SGBT::new(config2);

        let data: Vec<(Vec<f64>, f64)> = (0..50)
            .map(|i| (vec![i as f64 * 0.1], i as f64 * 0.5))
            .collect();

        model1.train_batch(&data);
        model2.train_batch_subsampled(&data, 1000); // max_samples > data.len()

        // Both should have identical state
        assert_eq!(model1.n_samples_seen(), model2.n_samples_seen());
        let pred1 = model1.predict(&[2.5]);
        let pred2 = model2.predict(&[2.5]);
        assert!(
            (pred1 - pred2).abs() < 1e-12,
            "full subsample should equal batch: {} vs {}",
            pred1,
            pred2,
        );
    }

    // -------------------------------------------------------------------
    // train_batch_subsampled_with_callback combines both
    // -------------------------------------------------------------------
    #[test]
    fn train_batch_subsampled_with_callback_works() {
        let config = SGBTConfig::builder()
            .n_steps(3)
            .grace_period(5)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        let data: Vec<(Vec<f64>, f64)> = (0..200)
            .map(|i| (vec![i as f64 * 0.01], i as f64 * 0.1))
            .collect();

        let mut callbacks = Vec::new();
        model.train_batch_subsampled_with_callback(&data, 50, 10, |n| {
            callbacks.push(n);
        });

        // Should have trained ~50 samples with callbacks at 10, 20, 30, 40, 50
        assert!(!callbacks.is_empty(), "should have received callbacks");
        assert_eq!(
            *callbacks.last().unwrap(),
            50,
            "final callback should be total samples"
        );
    }
}
