use crate::ensemble::config::SGBTConfig;
use crate::ensemble::step::BoostingStep;
use crate::ensemble::SGBT;
use crate::loss::Loss;

impl<L: Loss> SGBT<L> {
    /// Number of boosting steps (trees) in the ensemble.
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

    /// Current tree contribution standard deviation (honest uncertainty).
    ///
    /// This is the EWMA of per-sample contribution sigma across trees,
    /// computed from the `adaptive_mts` machinery. Reflects how much
    /// individual trees disagree — higher values indicate more model
    /// uncertainty / regime change.
    ///
    /// Returns 0.0 if `adaptive_mts` is not enabled or the model has
    /// not yet been trained.
    #[inline]
    pub fn contribution_sigma(&self) -> f64 {
        self.rolling_contribution_sigma
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

    /// Set the learning rate for future boosting rounds.
    ///
    /// This allows external schedulers to adapt the rate over time without
    /// rebuilding the model.
    ///
    /// # Panics
    ///
    /// Panics if `lr` is not in `(0.0, 1.0]` or is not finite.
    #[inline]
    pub fn set_learning_rate(&mut self, lr: f64) {
        assert!(
            lr > 0.0 && lr <= 1.0 && lr.is_finite(),
            "learning_rate must be in (0.0, 1.0], got {}",
            lr
        );
        self.config.learning_rate = lr;
    }

    /// Set the L2 regularization parameter (lambda) for future boosting rounds.
    ///
    /// Higher lambda increases regularization, shrinking leaf weights toward
    /// zero. Takes effect immediately for subsequent leaf weight computations.
    ///
    /// # Arguments
    ///
    /// * `lambda` -- new L2 regularization value (must be >= 0)
    #[inline]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.config.lambda = lambda.max(0.0);
    }

    /// Set the maximum tree depth for future replacement trees.
    ///
    /// Existing trees are not affected -- only new trees created during
    /// drift-triggered or proactive replacement will use the updated depth.
    ///
    /// # Arguments
    ///
    /// * `depth` -- new maximum depth (clamped to 1..=20)
    #[inline]
    pub fn set_max_depth(&mut self, depth: usize) {
        self.config.max_depth = depth.clamp(1, 20);
    }

    /// Adjust the number of boosting steps (trees in the ensemble).
    ///
    /// - **Growing** (`n > current`): appends fresh trees using the current config.
    /// - **Shrinking** (`n < current`): removes trailing steps (newest trees).
    /// - Clamped to `3..=1000` to prevent degenerate ensembles.
    pub fn set_n_steps(&mut self, n: usize) {
        let n = n.clamp(3, 1000);
        let current = self.steps.len();
        if n > current {
            let leaf_decay_alpha = self
                .config
                .leaf_half_life
                .map(|hl| (-(2.0_f64.ln()) / hl as f64).exp());
            let tree_config = crate::ensemble::config::build_tree_config(&self.config)
                .leaf_decay_alpha_opt(leaf_decay_alpha);
            let mts = self.config.max_tree_samples;
            let shadow_warmup = self.config.shadow_warmup.unwrap_or(0);
            for i in current..n {
                let mut tc = tree_config.clone();
                tc.seed = self.config.seed ^ (i as u64);
                let detector = self.config.drift_detector.create();
                let step = if shadow_warmup > 0 {
                    BoostingStep::new_with_graduated(tc, detector, mts, shadow_warmup)
                } else {
                    BoostingStep::new_with_max_samples(tc, detector, mts)
                };
                self.steps.push(step);
            }
        } else if n < current {
            self.steps.truncate(n);
        }
        self.diag.contribution_accuracy.resize(n, 0.0);
        self.config.n_steps = n;
    }

    /// Total tree replacements across all boosting steps.
    pub fn total_replacements(&self) -> u64 {
        self.steps.iter().map(|s| s.slot().replacements()).sum()
    }

    /// Manually trigger a proactive prune check.
    ///
    /// Finds the worst mature tree (past grace period) and replaces it if its
    /// contribution accuracy is negative (accuracy-based) or its prediction
    /// variance is minimal (variance-based). The contribution accuracy EWMAs
    /// are updated every sample inside `train_one()`; this method only performs
    /// the replacement decision.
    ///
    /// Returns `true` if a tree was replaced, `false` otherwise.
    pub fn check_proactive_prune(&mut self) -> bool {
        if self.steps.len() <= 1 {
            return false;
        }
        if self.config.accuracy_based_pruning {
            let grace_period = self.config.grace_period as u64;
            let worst = self
                .steps
                .iter()
                .enumerate()
                .zip(self.diag.contribution_accuracy.iter())
                .filter(|((_, step), _)| step.slot().n_samples_seen() >= grace_period)
                .min_by(|((_, _), a), ((_, _), b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(((worst_idx, _), &worst_acc)) = worst {
                if worst_acc < 0.0 {
                    self.steps[worst_idx].slot_mut().replace_active();
                    self.diag.contribution_accuracy[worst_idx] = 0.0;
                    return true;
                }
            }
            false
        } else {
            let worst_idx = self
                .steps
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let a_std = a.slot().prediction_std();
                    let b_std = b.slot().prediction_std();
                    a_std
                        .partial_cmp(&b_std)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.steps[worst_idx].slot_mut().replace_active();
            true
        }
    }

    /// Dynamically set the contribution accuracy EWMA half-life.
    ///
    /// Recomputes `prune_alpha` from the given half-life so each correction
    /// batch contributes equally regardless of size.
    pub fn set_prune_half_life(&mut self, hl: usize) {
        self.diag.prune_alpha = 1.0 - (-2.0 / hl.max(1) as f64).exp();
    }
}
