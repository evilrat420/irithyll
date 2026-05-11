use crate::ensemble::SGBT;
use crate::loss::Loss;
use crate::sample::Observation;
use crate::tree::node::NodeId;

impl<L: Loss> SGBT<L> {
    /// Train on a single observation.
    ///
    /// Accepts any type implementing [`Observation`], including
    /// [`SampleRef`](crate::SampleRef) or tuples like `(&[f64], f64)` for
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

        // Adaptive MTS: accumulate sigma_ratio into ring buffer.
        // Effective MTS is only recomputed at tree replacement boundaries.
        if self.config.adaptive_mts.is_some() {
            let contribution_sigma = self.compute_contribution_sigma(features);
            const CONTRIBUTION_SIGMA_ALPHA: f64 = 0.001;
            self.rolling_contribution_sigma = (1.0 - CONTRIBUTION_SIGMA_ALPHA)
                * self.rolling_contribution_sigma
                + CONTRIBUTION_SIGMA_ALPHA * contribution_sigma;

            let sigma_ratio = if self.rolling_contribution_sigma > 1e-12 {
                contribution_sigma / self.rolling_contribution_sigma
            } else {
                1.0
            };
            let cap = self.config.grace_period;
            if self.sigma_ring.len() >= cap {
                self.sigma_ring.pop_front();
            }
            self.sigma_ring.push_back(sigma_ratio);
        }

        // Current prediction starts from base
        let mut current_pred = self.base_prediction;

        let prune_alpha = self.config.quality_prune_alpha;
        let prune_threshold = self.config.quality_prune_threshold;
        let prune_patience = self.config.quality_prune_patience;

        // Error-weighted sample importance: compute weight from prediction error
        let error_weight = if let Some(ew_alpha) = self.config.error_weight_alpha {
            let abs_error = (target - current_pred).abs();
            if self.rolling_mean_error > 1e-15 {
                let w = (1.0 + abs_error / (self.rolling_mean_error + 1e-15)).min(10.0);
                self.rolling_mean_error =
                    ew_alpha * abs_error + (1.0 - ew_alpha) * self.rolling_mean_error;
                w
            } else {
                self.rolling_mean_error = abs_error.max(1e-15);
                1.0 // first sample, no reweighting
            }
        } else {
            1.0
        };

        // Sequential boosting: each step targets the residual of all prior steps
        for s in 0..self.steps.len() {
            let gradient = self.loss.gradient(target, current_pred) * error_weight;
            let hessian = self.loss.hessian(target, current_pred) * error_weight;
            let train_count = self
                .config
                .variant
                .train_count(hessian, &mut self.rng_state);

            let step_pred =
                self.steps[s].train_and_predict(features, gradient, hessian, train_count);

            current_pred += self.config.learning_rate * step_pred;

            // Quality-based tree pruning: track contribution and replace dead wood
            if let Some(alpha) = prune_alpha {
                let contribution = (self.config.learning_rate * step_pred).abs();
                self.contribution_ewma[s] =
                    alpha * contribution + (1.0 - alpha) * self.contribution_ewma[s];

                if self.contribution_ewma[s] < prune_threshold {
                    self.low_contrib_count[s] += 1;
                    if self.low_contrib_count[s] >= prune_patience {
                        self.steps[s].reset();
                        self.contribution_ewma[s] = 0.0;
                        self.low_contrib_count[s] = 0;
                    }
                } else {
                    self.low_contrib_count[s] = 0;
                }
            }
        }

        // Proactive pruning: replace worst-contributing tree every N samples.
        if let Some(interval) = self.config.proactive_prune_interval {
            // Update contribution accuracy EWMAs (used by accuracy-based pruning).
            if self.config.accuracy_based_pruning {
                let mut ensemble_pred = self.base_prediction;
                for step in self.steps.iter() {
                    ensemble_pred += self.config.learning_rate * step.predict(features);
                }
                let residual = target - ensemble_pred;
                let sign = residual.signum();
                for (i, step) in self.steps.iter().enumerate() {
                    let contribution = self.config.learning_rate * step.predict(features);
                    let alignment = contribution * sign;
                    self.diag.contribution_accuracy[i] = self.diag.prune_alpha * alignment
                        + (1.0 - self.diag.prune_alpha) * self.diag.contribution_accuracy[i];
                }
            }

            // Interval-based fallback: fire prune check every N samples.
            if interval > 0 && self.samples_seen % interval == 0 {
                self.check_proactive_prune();
            }
        }

        // End-of-cycle adaptive MTS: update effective MTS at replacement boundaries.
        if let Some((base_mts, k)) = self.config.adaptive_mts {
            let current_sum: u64 = self.steps.iter().map(|s| s.slot().replacements()).sum();
            if current_sum != self.mts_replacement_sum {
                self.mts_replacement_sum = current_sum;
                if !self.sigma_ring.is_empty() {
                    let mean_sigma =
                        self.sigma_ring.iter().sum::<f64>() / self.sigma_ring.len() as f64;
                    let floor = (base_mts as f64 * self.config.adaptive_mts_floor).max(100.0);
                    let effective_mts =
                        (base_mts as f64 / (1.0 + k * mean_sigma)).max(floor) as u64;
                    for step in &mut self.steps {
                        step.slot_mut().set_max_tree_samples(Some(effective_mts));
                    }
                }
            }
        }

        // Update diagnostic cache for config_diagnostics() signals
        self.update_diagnostic_cache(features);

        // Refresh auto-bandwidths when trees have been replaced or not yet computed.
        self.refresh_bandwidths();
    }

    /// Update cached diagnostic signals from tree internals.
    ///
    /// Computes four signals used by the auto-builder:
    /// - **residual_alignment**: cosine similarity of consecutive tree contributions
    /// - **regularization_sensitivity**: mean |G|/(H+λ)² across leaves
    /// - **depth_sufficiency**: F-statistic (between-leaf / within-leaf variance)
    /// - **effective_dof**: trace(H/(H+λ)) across all leaves
    pub(crate) fn update_diagnostic_cache(&mut self, features: &[f64]) {
        let lambda = self.config.lambda;
        let lr = self.config.learning_rate;
        let n_steps = self.steps.len();

        // 1. Residual alignment: cosine similarity of consecutive contribution vectors
        let mut contributions = Vec::with_capacity(n_steps);
        for step in &self.steps {
            contributions.push(lr * step.predict(features));
        }

        if !self.diag.prev_contributions.is_empty()
            && self.diag.prev_contributions.len() == contributions.len()
            && !self.diag.prev_prev_contributions.is_empty()
            && self.diag.prev_prev_contributions.len() == contributions.len()
        {
            // Delta-based alignment: cosine similarity of consecutive *changes*
            // in the contribution vector, not the raw vectors themselves.
            // This prevents saturation when contributions change slowly.
            let delta_curr: Vec<f64> = contributions
                .iter()
                .zip(&self.diag.prev_contributions)
                .map(|(a, b)| a - b)
                .collect();
            let delta_prev: Vec<f64> = self
                .diag
                .prev_contributions
                .iter()
                .zip(&self.diag.prev_prev_contributions)
                .map(|(a, b)| a - b)
                .collect();

            let dot: f64 = delta_curr.iter().zip(&delta_prev).map(|(a, b)| a * b).sum();
            let norm_curr: f64 = delta_curr.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_prev: f64 = delta_prev.iter().map(|x| x * x).sum::<f64>().sqrt();
            self.diag.cached_residual_alignment = if norm_curr > 1e-15 && norm_prev > 1e-15 {
                dot / (norm_curr * norm_prev)
            } else {
                0.0
            };
        }
        self.diag.prev_prev_contributions =
            std::mem::replace(&mut self.diag.prev_contributions, contributions);

        // 2-4. Leaf traversal for reg_sensitivity, depth_sufficiency, effective_dof
        let mut total_sensitivity = 0.0;
        let mut total_dof = 0.0;
        let mut leaf_weights: Vec<f64> = Vec::new();
        let mut leaf_within_vars: Vec<f64> = Vec::new();
        let mut n_leaves_total: u64 = 0;

        for step in &self.steps {
            let tree = step.slot().active_tree();
            let arena = tree.arena();

            for node_idx in 0..arena.n_nodes() {
                let nid = NodeId(node_idx as u32);
                if arena.is_leaf(nid) {
                    if let Some((g, h)) = tree.leaf_grad_hess(nid) {
                        let denom = h + lambda;
                        if denom.abs() > 1e-15 {
                            // Reg sensitivity: |G| / (H+λ)²
                            total_sensitivity += g.abs() / (denom * denom);
                            // Effective DOF: H / (H+λ)
                            total_dof += h / denom;
                            // Leaf weight: w* = -G/(H+λ)
                            leaf_weights.push(-g / denom);
                            // Within-leaf variance: 1/(H+λ)
                            leaf_within_vars.push(1.0 / denom);
                            n_leaves_total += 1;
                        }
                    }
                }
            }
        }

        if n_leaves_total > 0 {
            let n = n_leaves_total as f64;
            self.diag.cached_reg_sensitivity = total_sensitivity / n;
            self.diag.cached_effective_dof = total_dof;

            // Depth sufficiency: F = between_var / within_var
            let mean_weight = leaf_weights.iter().sum::<f64>() / n;
            let between_var = leaf_weights
                .iter()
                .map(|w| (w - mean_weight).powi(2))
                .sum::<f64>()
                / (n - 1.0).max(1.0);
            let within_var = leaf_within_vars.iter().sum::<f64>() / n;
            self.diag.cached_depth_sufficiency = between_var / within_var.max(1e-15);
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
    pub fn train_batch_with_callback<O: Observation, F: FnMut(usize)>(
        &mut self,
        samples: &[O],
        interval: usize,
        mut callback: F,
    ) {
        let interval = interval.max(1);
        for (i, sample) in samples.iter().enumerate() {
            self.train_one(sample);
            if (i + 1) % interval == 0 {
                callback(i + 1);
            }
        }
        let total = samples.len();
        if total % interval != 0 {
            callback(total);
        }
    }

    /// Train on a random subsample of a batch using reservoir sampling.
    ///
    /// When `max_samples < samples.len()`, selects a representative subset
    /// using Algorithm R (Vitter, 1985) -- a uniform random sample without
    /// replacement. The selected samples are then trained in their original
    /// order to preserve sequential dependencies.
    pub fn train_batch_subsampled<O: Observation>(&mut self, samples: &[O], max_samples: usize) {
        if max_samples >= samples.len() {
            self.train_batch(samples);
            return;
        }

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

        for &idx in &reservoir {
            self.train_one(&samples[idx]);
        }
    }

    /// Train on a batch with both subsampling and periodic callbacks.
    ///
    /// Combines reservoir subsampling with cooperative yield points.
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
}
