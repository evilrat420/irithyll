//! SGBT ensemble orchestrator -- the core boosting loop.
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

/// Field-access helpers for internal ensemble state.
pub mod accessors;
pub mod adaptive;
pub mod adaptive_forest;
pub mod bagged;
pub mod config;
pub mod core;
pub mod diagnostics;
pub mod distributional;
/// Fast packed-node inference path for SGBT ensembles.
pub mod inference;
/// Ensemble inspection utilities (feature importance, leaf paths).
pub mod inspection;
pub mod lr_schedule;
pub mod moe;
pub mod moe_distributional;
pub mod multi_target;
pub mod multiclass;
pub mod parallel;
pub mod quantile_regressor;
pub mod replacement;
pub mod stacked;
pub mod step;
/// Core training loop implementation.
pub mod train;
pub mod variants;

// Re-export core types from the core module
pub(crate) use core::DiagnosticCache;
pub use core::SGBT;

use std::collections::VecDeque;

use crate::ensemble::step::BoostingStep;
use crate::loss::Loss;
#[allow(unused_imports)]
use crate::sample::Sample;

/// Type alias for an SGBT model using dynamic (boxed) loss dispatch.
///
/// Use this when the loss function is determined at runtime (e.g., when
/// deserializing a model from JSON where the loss type is stored as a tag).
///
/// For compile-time loss dispatch (preferred for performance), use
/// `SGBT<LogisticLoss>`, `SGBT<HuberLoss>`, etc.
pub type DynSGBT = SGBT<Box<dyn Loss>>;

impl<L: Loss> SGBT<L> {
    /// Full ensemble diagnostics with per-tree contributions for a given input.
    ///
    /// Computes tree structure metrics, feature importance (split-count based),
    /// per-tree contributions (`lr * tree.predict(features)`), and replacement
    /// counts across all boosting steps.
    pub fn diagnostics(
        &self,
        features: &[f64],
    ) -> crate::ensemble::diagnostics::EnsembleDiagnostics {
        crate::ensemble::diagnostics::build_ensemble_diagnostics(
            &self.steps,
            self.base_prediction,
            self.config.learning_rate,
            self.samples_seen,
            Some(features),
        )
    }

    /// Ensemble diagnostics without per-tree contributions.
    ///
    /// Same as [`diagnostics()`](Self::diagnostics) but all `contribution`
    /// fields are set to 0.0. Use when you only need structure/importance
    /// and don't have a feature vector handy.
    pub fn diagnostics_overview(&self) -> crate::ensemble::diagnostics::EnsembleDiagnostics {
        crate::ensemble::diagnostics::build_ensemble_diagnostics(
            &self.steps,
            self.base_prediction,
            self.config.learning_rate,
            self.samples_seen,
            None,
        )
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
        use crate::serde_support::{ModelState, StepSnapshot};

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
            rolling_mean_error: self.rolling_mean_error,
            contribution_ewma: self.contribution_ewma.clone(),
            low_contrib_count: self.low_contrib_count.clone(),
            rolling_contribution_sigma: self.rolling_contribution_sigma,
        }
    }
}

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

        let loss = state.loss_type.into_loss();

        let leaf_decay_alpha = state
            .config
            .leaf_half_life
            .map(|hl| (-(2.0_f64.ln()) / hl as f64).exp());
        let max_tree_samples = state.config.max_tree_samples;

        let base_tree_config = crate::ensemble::config::build_tree_config(&state.config)
            .leaf_decay_alpha_opt(leaf_decay_alpha);

        let steps: Vec<BoostingStep> = state
            .steps
            .iter()
            .enumerate()
            .map(|(i, step_snap)| {
                let tree_config = base_tree_config
                    .clone()
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

        let n = steps.len();
        let has_pruning = state.config.quality_prune_alpha.is_some();

        let contribution_ewma = if !state.contribution_ewma.is_empty() {
            state.contribution_ewma
        } else if has_pruning {
            vec![0.0; n]
        } else {
            Vec::new()
        };
        let low_contrib_count = if !state.low_contrib_count.is_empty() {
            state.low_contrib_count
        } else if has_pruning {
            vec![0; n]
        } else {
            Vec::new()
        };

        let prune_alpha = if state.config.proactive_prune_interval.is_some() {
            let hl = state.config.prune_half_life.unwrap_or_else(|| {
                if let Some((base_mts, _)) = state.config.adaptive_mts {
                    base_mts as usize
                } else if let Some(mts) = state.config.max_tree_samples {
                    mts as usize
                } else {
                    state.config.grace_period.max(1)
                }
            });
            1.0 - (-2.0 / hl.max(1) as f64).exp()
        } else {
            0.01
        };

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
            contribution_ewma,
            low_contrib_count,
            rolling_mean_error: state.rolling_mean_error,
            auto_bandwidths: Vec::new(),
            last_replacement_sum: 0,
            rolling_contribution_sigma: state.rolling_contribution_sigma,
            sigma_ring: VecDeque::new(),
            mts_replacement_sum: 0,
            diag: DiagnosticCache {
                contribution_accuracy: vec![0.0; n],
                prune_alpha,
                ..Default::default()
            },
        }
    }
}

#[cfg(any(feature = "serde-json", feature = "serde-bincode"))]
impl<L: Loss> SGBT<L> {
    /// Reconstruct an `SGBT<L>` from a [`ModelState`](crate::serde_support::ModelState)
    /// using a caller-supplied concrete loss function.
    ///
    /// This is the generic counterpart of [`SGBT::<Box<dyn Loss>>::from_model_state`]:
    /// it ignores the `loss_type` tag in the state and uses the provided `loss`
    /// directly, allowing reconstruction into a monomorphized type like
    /// `SGBT<SoftmaxLoss>` or `SGBT<SquaredLoss>`.
    pub fn from_model_state_with_loss(state: crate::serde_support::ModelState, loss: L) -> Self {
        use crate::ensemble::replacement::TreeSlot;

        let leaf_decay_alpha = state
            .config
            .leaf_half_life
            .map(|hl| (-(2.0_f64.ln()) / hl as f64).exp());
        let max_tree_samples = state.config.max_tree_samples;

        let base_tree_config = crate::ensemble::config::build_tree_config(&state.config)
            .leaf_decay_alpha_opt(leaf_decay_alpha);

        let steps: Vec<BoostingStep> = state
            .steps
            .iter()
            .enumerate()
            .map(|(i, step_snap)| {
                let tree_config = base_tree_config
                    .clone()
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

        let n = steps.len();
        let has_pruning = state.config.quality_prune_alpha.is_some();

        let contribution_ewma = if !state.contribution_ewma.is_empty() {
            state.contribution_ewma
        } else if has_pruning {
            vec![0.0; n]
        } else {
            Vec::new()
        };
        let low_contrib_count = if !state.low_contrib_count.is_empty() {
            state.low_contrib_count
        } else if has_pruning {
            vec![0; n]
        } else {
            Vec::new()
        };

        let prune_alpha = if state.config.proactive_prune_interval.is_some() {
            let hl = state.config.prune_half_life.unwrap_or_else(|| {
                if let Some((base_mts, _)) = state.config.adaptive_mts {
                    base_mts as usize
                } else if let Some(mts) = state.config.max_tree_samples {
                    mts as usize
                } else {
                    state.config.grace_period.max(1)
                }
            });
            1.0 - (-2.0 / hl.max(1) as f64).exp()
        } else {
            0.01
        };

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
            contribution_ewma,
            low_contrib_count,
            rolling_mean_error: state.rolling_mean_error,
            auto_bandwidths: Vec::new(),
            last_replacement_sum: 0,
            rolling_contribution_sigma: state.rolling_contribution_sigma,
            sigma_ring: VecDeque::new(),
            mts_replacement_sum: 0,
            diag: DiagnosticCache {
                contribution_accuracy: vec![0.0; n],
                prune_alpha,
                ..Default::default()
            },
        }
    }
}

#[cfg(any(feature = "serde-json", feature = "serde-bincode"))]
pub(crate) fn snapshot_tree(
    tree: &crate::tree::hoeffding::HoeffdingTree,
) -> crate::serde_support::TreeSnapshot {
    use crate::serde_support::TreeSnapshot;
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

#[cfg(any(feature = "serde-json", feature = "serde-bincode"))]
pub(crate) fn rebuild_tree(
    snapshot: &crate::serde_support::TreeSnapshot,
    tree_config: crate::tree::builder::TreeConfig,
) -> crate::tree::hoeffding::HoeffdingTree {
    use crate::tree::hoeffding::HoeffdingTree;
    use crate::tree::node::{NodeId, TreeArena};

    let mut arena = TreeArena::new();
    let n = snapshot.feature_idx.len();

    for i in 0..n {
        arena.feature_idx.push(snapshot.feature_idx[i]);
        arena.threshold.push(snapshot.threshold[i]);
        arena.left.push(NodeId(snapshot.left[i]));
        arena.right.push(NodeId(snapshot.right[i]));
        arena.leaf_value.push(snapshot.leaf_value[i]);
        arena.is_leaf.push(snapshot.is_leaf[i]);
        arena.depth.push(snapshot.depth[i]);
        arena.sample_count.push(snapshot.sample_count[i]);
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

impl<L: Loss> crate::automl::DiagnosticSource for SGBT<L> {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.diag.cached_residual_alignment,
            regularization_sensitivity: self.diag.cached_reg_sensitivity,
            depth_sufficiency: self.diag.cached_depth_sufficiency,
            effective_dof: self.diag.cached_effective_dof,
            uncertainty: self.rolling_contribution_sigma,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::config::SGBTConfig;

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
        let _loss: &LogisticLoss = model.loss();
        assert_eq!(_loss.n_outputs(), 1);
    }

    #[test]
    fn clone_produces_independent_copy() {
        let config = default_config();
        let mut model = SGBT::new(config);

        let mut rng: u64 = 99999;
        for _ in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            let target = 2.0 * x + 1.0;
            model.train_one(&Sample::new(vec![x], target));
        }

        let mut cloned = model.clone();

        let test_features = [3.0];
        let pred_original = model.predict(&test_features);
        let pred_cloned = cloned.predict(&test_features);
        assert!(
            (pred_original - pred_cloned).abs() < 1e-12,
            "clone should predict identically: original={pred_original}, cloned={pred_cloned}"
        );

        for _ in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            let target = -3.0 * x + 5.0;
            cloned.train_one(&Sample::new(vec![x], target));
        }

        let pred_original_after = model.predict(&test_features);
        let pred_cloned_after = cloned.predict(&test_features);

        assert!(
            (pred_original - pred_original_after).abs() < 1e-12,
            "original should be unchanged after training clone"
        );

        assert!(
            (pred_original_after - pred_cloned_after).abs() > 1e-6,
            "clone should diverge after independent training"
        );
    }

    #[test]
    fn predict_with_confidence_finite() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .grace_period(10)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

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

    #[test]
    fn predict_with_confidence_positive_after_training() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .grace_period(10)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

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

    #[test]
    fn predict_with_confidence_matches_predict() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .grace_period(10)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        for i in 0..200 {
            let x = i as f64 * 0.05;
            model.train_one(&(&[x][..], x * 2.0));
        }

        let (pred_with_conf, _) = model.predict_with_confidence(&[1.0]);
        let pred = model.predict(&[1.0]);

        assert!(
            (pred_with_conf - pred).abs() < 1e-12,
            "predict_with_confidence prediction should match predict"
        );
    }

    #[test]
    fn feature_importances_sums_to_one() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(5)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&Sample::new(vec![x, x * 0.5, x * 2.0], x + 1.0));
        }

        let imps = model.feature_importances();
        if !imps.is_empty() {
            let sum: f64 = imps.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "importances should sum to 1.0, got {}",
                sum
            );
        }
    }

    #[cfg(any(feature = "serde-json", feature = "serde-bincode"))]
    #[test]
    fn snapshot_restore_preserves_tree_config_knobs() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(4)
            .n_bins(16)
            .seed(99)
            .adaptive_leaf_bound(3.0)
            .max_leaf_output(5.0)
            .min_hessian_sum(1.0)
            .hoeffding_r(0.5)
            .gradient_clip_sigma(3.0)
            .build()
            .unwrap();

        let mut model = SGBT::new(config.clone());
        for i in 0..100 {
            let x = i as f64 * 0.05;
            model.train_one(&(vec![x, x * 0.3], x + 0.5));
        }

        let snapshot = model.to_model_state().unwrap();
        let restored = SGBT::from_model_state(snapshot);

        let rc = restored.config();
        assert_eq!(
            rc.adaptive_leaf_bound,
            Some(3.0),
            "adaptive_leaf_bound lost on restore"
        );
        assert_eq!(
            rc.max_leaf_output,
            Some(5.0),
            "max_leaf_output lost on restore"
        );
        assert_eq!(
            rc.min_hessian_sum,
            Some(1.0),
            "min_hessian_sum lost on restore"
        );
        assert_eq!(rc.hoeffding_r, Some(0.5), "hoeffding_r lost on restore");
        assert_eq!(
            rc.gradient_clip_sigma,
            Some(3.0),
            "gradient_clip_sigma lost on restore"
        );
        assert_eq!(rc.n_steps, config.n_steps, "n_steps lost on restore");
        assert_eq!(rc.max_depth, config.max_depth, "max_depth lost on restore");

        let test_x = vec![1.0, 0.3];
        assert!(
            restored.predict(&test_x).is_finite(),
            "restored model prediction should be finite"
        );
    }

    #[test]
    fn sgbt_contribution_sigma_exposed() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(5)
            .adaptive_mts(500, 1.0)
            .build()
            .unwrap();
        let mut model = SGBT::new(config);

        assert!(
            model.contribution_sigma().abs() < 1e-15,
            "contribution_sigma should be 0 before training, got {}",
            model.contribution_sigma()
        );

        for i in 0..200 {
            let x = i as f64 * 0.05;
            model.train_one(&(vec![x, x * 0.5], x * 2.0 + 1.0));
        }

        let sigma = model.contribution_sigma();
        assert!(
            sigma > 0.0,
            "contribution_sigma should be > 0 after training with adaptive_mts, got {}",
            sigma
        );
        assert!(
            sigma.is_finite(),
            "contribution_sigma should be finite, got {}",
            sigma
        );
    }
}
