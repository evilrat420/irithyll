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

pub mod config;
pub mod step;
pub mod replacement;
pub mod multiclass;
pub mod variants;
pub mod parallel;

use std::fmt;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::step::BoostingStep;
use crate::loss::Loss;
use crate::loss::squared::SquaredLoss;
use crate::sample::Sample;
use crate::tree::builder::TreeConfig;

/// Streaming Gradient Boosted Trees ensemble.
///
/// The primary entry point for training and prediction. Supports regression
/// (default) and binary classification via pluggable loss functions.
pub struct SGBT {
    /// Configuration.
    config: SGBTConfig,
    /// Boosting steps (one tree + drift detector each).
    steps: Vec<BoostingStep>,
    /// Loss function.
    loss: Box<dyn Loss>,
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

impl fmt::Debug for SGBT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SGBT")
            .field("n_steps", &self.steps.len())
            .field("samples_seen", &self.samples_seen)
            .field("base_prediction", &self.base_prediction)
            .field("base_initialized", &self.base_initialized)
            .finish()
    }
}

impl SGBT {
    /// Create a new SGBT ensemble with squared loss (regression).
    pub fn new(config: SGBTConfig) -> Self {
        let tree_config = TreeConfig::new()
            .max_depth(config.max_depth)
            .n_bins(config.n_bins)
            .lambda(config.lambda)
            .gamma(config.gamma)
            .grace_period(config.grace_period)
            .delta(config.delta)
            .feature_subsample_rate(config.feature_subsample_rate);

        let steps: Vec<BoostingStep> = (0..config.n_steps)
            .map(|i| {
                let mut tc = tree_config.clone();
                tc.seed = config.seed ^ (i as u64);
                let detector = config.drift_detector.create();
                BoostingStep::new(tc, detector)
            })
            .collect();

        let seed = config.seed;
        let initial_target_count = config.initial_target_count;
        Self {
            config,
            steps,
            loss: Box::new(SquaredLoss),
            base_prediction: 0.0,
            base_initialized: false,
            initial_targets: Vec::new(),
            initial_target_count,
            samples_seen: 0,
            rng_state: seed,
        }
    }

    /// Create a new SGBT ensemble with a custom loss function.
    pub fn with_loss(config: SGBTConfig, loss: Box<dyn Loss>) -> Self {
        let mut model = Self::new(config);
        model.loss = loss;
        model
    }

    /// Train on a single sample.
    pub fn train_one(&mut self, sample: &Sample) {
        self.samples_seen += 1;

        // Initialize base prediction from first few targets
        if !self.base_initialized {
            self.initial_targets.push(sample.target);
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
            let gradient = self.loss.gradient(sample.target, current_pred);
            let hessian = self.loss.hessian(sample.target, current_pred);
            let train_count = self.config.variant.train_count(hessian, &mut self.rng_state);

            let step_pred = step.train_and_predict(
                &sample.features,
                gradient,
                hessian,
                train_count,
            );

            current_pred += self.config.learning_rate * step_pred;
        }
    }

    /// Train on a batch of samples.
    pub fn train_batch(&mut self, samples: &[Sample]) {
        for sample in samples {
            self.train_one(sample);
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
    /// Captures all internal state needed to reconstruct the model for
    /// prediction and continued training. The caller must supply a
    /// [`LossType`](crate::serde_support::LossType) tag matching the loss
    /// function used during training.
    #[cfg(feature = "serde-json")]
    pub fn to_model_state(
        &self,
        loss_type: crate::serde_support::LossType,
    ) -> crate::serde_support::ModelState {
        use crate::serde_support::{ModelState, StepSnapshot, TreeSnapshot};

        fn snapshot_tree(
            tree: &crate::tree::hoeffding::HoeffdingTree,
        ) -> TreeSnapshot {
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
            }
        }

        let steps = self
            .steps
            .iter()
            .map(|step| {
                let slot = step.slot();
                let tree_snap = snapshot_tree(slot.active_tree());
                let alt_snap = slot.alternate_tree().map(snapshot_tree);
                StepSnapshot {
                    tree: tree_snap,
                    alternate_tree: alt_snap,
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

    /// Reconstruct an SGBT model from a [`ModelState`](crate::serde_support::ModelState).
    ///
    /// Rebuilds the full ensemble including tree topology and leaf values.
    /// Histogram accumulators are left empty and will rebuild from continued
    /// training. The drift detector for each step is freshly created from
    /// the config (its internal state is not serialized).
    #[cfg(feature = "serde-json")]
    pub fn from_model_state(state: crate::serde_support::ModelState) -> Self {
        use crate::serde_support::TreeSnapshot;
        use crate::tree::hoeffding::HoeffdingTree;
        use crate::tree::node::{NodeId, TreeArena};
        use crate::ensemble::replacement::TreeSlot;

        fn rebuild_tree(
            snapshot: &TreeSnapshot,
            tree_config: TreeConfig,
        ) -> HoeffdingTree {
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
                    .seed(state.config.seed ^ (i as u64));

                let active = rebuild_tree(&step_snap.tree, tree_config.clone());
                let alternate = step_snap
                    .alternate_tree
                    .as_ref()
                    .map(|snap| rebuild_tree(snap, tree_config.clone()));

                let detector = state.config.drift_detector.create();
                let slot = TreeSlot::from_trees(active, alternate, tree_config, detector);
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
            early_rmse, late_rmse
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
            pred_seq, pred_batch
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
        let model = SGBT::with_loss(default_config(), Box::new(LogisticLoss));
        let pred = model.predict_transformed(&[1.0, 2.0]);
        assert!((pred - 0.5).abs() < 1e-6, "sigmoid(0) should be 0.5, got {}", pred);
    }
}
