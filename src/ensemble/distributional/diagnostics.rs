//! Distributional SGBT diagnostic structures and methods.

use crate::ensemble::config::ScaleMode;
use crate::ensemble::step::BoostingStep;

use super::DistributionalSGBT;

/// Per-tree diagnostic summary.
#[derive(Debug, Clone)]
pub struct DistributionalTreeDiagnostic {
    /// Number of leaf nodes in this tree.
    pub n_leaves: usize,
    /// Maximum depth reached by any leaf.
    pub max_depth_reached: usize,
    /// Total samples this tree has seen.
    pub samples_seen: u64,
    /// Leaf weight statistics: `(min, max, mean, std)`.
    pub leaf_weight_stats: (f64, f64, f64, f64),
    /// Feature indices this tree has split on (non-zero gain).
    pub split_features: Vec<usize>,
    /// Per-leaf sample counts showing data distribution across leaves.
    pub leaf_sample_counts: Vec<u64>,
    /// Running mean of predictions from this tree (Welford online).
    pub prediction_mean: f64,
    /// Running standard deviation of predictions from this tree.
    pub prediction_std: f64,
}

/// Full model diagnostics for [`DistributionalSGBT`].
///
/// Contains per-tree summaries, feature usage, base predictions, and
/// empirical σ state.
#[derive(Debug, Clone)]
pub struct ModelDiagnostics {
    /// Per-tree diagnostic summaries (location trees first, then scale trees).
    pub trees: Vec<DistributionalTreeDiagnostic>,
    /// Location trees only (view into `trees`).
    pub location_trees: Vec<DistributionalTreeDiagnostic>,
    /// Scale trees only (view into `trees`).
    pub scale_trees: Vec<DistributionalTreeDiagnostic>,
    /// How many trees each feature is used in (split count per feature).
    pub feature_split_counts: Vec<usize>,
    /// Base prediction for location (mean).
    pub location_base: f64,
    /// Base prediction for scale (log-sigma).
    pub scale_base: f64,
    /// Current empirical σ (`sqrt(ewma_sq_err)`), always available.
    pub empirical_sigma: f64,
    /// Scale mode in use.
    pub scale_mode: ScaleMode,
    /// Number of scale trees that actually split (>1 leaf). 0 = frozen chain.
    pub scale_trees_active: usize,
    /// Per-feature auto-calibrated bandwidths for smooth prediction.
    /// `f64::INFINITY` means that feature uses hard routing.
    pub auto_bandwidths: Vec<f64>,
    /// Ensemble-level gradient running mean.
    pub ensemble_grad_mean: f64,
    /// Ensemble-level gradient standard deviation.
    pub ensemble_grad_std: f64,
}

/// Decomposed prediction showing each tree's contribution.
#[derive(Debug, Clone)]
pub struct DecomposedPrediction {
    /// Base location prediction (mean of initial targets).
    pub location_base: f64,
    /// Base scale prediction (log-sigma of initial targets).
    pub scale_base: f64,
    /// Per-step location contributions: `learning_rate * tree_prediction`.
    /// `location_base + sum(location_contributions)` = μ.
    pub location_contributions: Vec<f64>,
    /// Per-step scale contributions: `learning_rate * tree_prediction`.
    /// `scale_base + sum(scale_contributions)` = log(σ).
    pub scale_contributions: Vec<f64>,
}

impl DecomposedPrediction {
    /// Reconstruct the final μ from base + contributions.
    pub fn mu(&self) -> f64 {
        self.location_base + self.location_contributions.iter().sum::<f64>()
    }

    /// Reconstruct the final log(σ) from base + contributions.
    pub fn log_sigma(&self) -> f64 {
        self.scale_base + self.scale_contributions.iter().sum::<f64>()
    }

    /// Reconstruct the final σ (exponentiated).
    pub fn sigma(&self) -> f64 {
        self.log_sigma().exp().max(1e-8)
    }
}

impl DistributionalSGBT {
    /// Update cached diagnostic signals from tree internals.
    ///
    /// Computes four signals used by the auto-builder:
    /// - **residual_alignment**: cosine similarity of consecutive tree contributions
    /// - **regularization_sensitivity**: mean |G|/(H+λ)² across leaves
    /// - **depth_sufficiency**: F-statistic (between-leaf / within-leaf variance)
    /// - **effective_dof**: trace(H/(H+λ)) across all leaves
    pub(crate) fn update_diagnostic_cache(&mut self, features: &[f64]) {
        use crate::tree::node::NodeId;

        let lambda = self.config.lambda;
        let lr = self.config.learning_rate;
        let n_steps = self.location_steps.len();

        // 1. Residual alignment: cosine similarity of consecutive contribution vectors
        let mut contributions = Vec::with_capacity(n_steps);
        for step in &self.location_steps {
            contributions.push(lr * step.predict(features));
        }

        if !self.prev_contributions.is_empty()
            && self.prev_contributions.len() == contributions.len()
            && !self.prev_prev_contributions.is_empty()
            && self.prev_prev_contributions.len() == contributions.len()
        {
            // Delta-based alignment: cosine similarity of consecutive *changes*
            // in the contribution vector, not the raw vectors themselves.
            // This prevents saturation when contributions change slowly.
            let delta_curr: Vec<f64> = contributions
                .iter()
                .zip(&self.prev_contributions)
                .map(|(a, b)| a - b)
                .collect();
            let delta_prev: Vec<f64> = self
                .prev_contributions
                .iter()
                .zip(&self.prev_prev_contributions)
                .map(|(a, b)| a - b)
                .collect();

            let dot: f64 = delta_curr.iter().zip(&delta_prev).map(|(a, b)| a * b).sum();
            let norm_curr: f64 = delta_curr.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_prev: f64 = delta_prev.iter().map(|x| x * x).sum::<f64>().sqrt();
            self.cached_residual_alignment = if norm_curr > 1e-15 && norm_prev > 1e-15 {
                dot / (norm_curr * norm_prev)
            } else {
                0.0
            };
        }
        self.prev_prev_contributions =
            core::mem::replace(&mut self.prev_contributions, contributions);

        // 2-4. Leaf traversal for reg_sensitivity, depth_sufficiency, effective_dof
        let mut total_sensitivity = 0.0;
        let mut total_dof = 0.0;
        let mut leaf_weights: Vec<f64> = Vec::new();
        let mut leaf_within_vars: Vec<f64> = Vec::new();
        let mut n_leaves_total: u64 = 0;

        for step in &self.location_steps {
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
            self.cached_reg_sensitivity = total_sensitivity / n;
            self.cached_effective_dof = total_dof;

            // Depth sufficiency: F = between_var / within_var
            let mean_weight = leaf_weights.iter().sum::<f64>() / n;
            let between_var = leaf_weights
                .iter()
                .map(|w| (w - mean_weight).powi(2))
                .sum::<f64>()
                / (n - 1.0).max(1.0);
            let within_var = leaf_within_vars.iter().sum::<f64>() / n;
            self.cached_depth_sufficiency = between_var / within_var.max(1e-15);
        }
    }

    /// Full model diagnostics: per-tree structure, feature usage, base predictions.
    ///
    /// The `trees` vector contains location trees first (indices `0..n_steps`),
    /// then scale trees (`n_steps..2*n_steps`).
    ///
    /// `scale_trees_active` counts how many scale trees have actually split
    /// (more than 1 leaf). If this is 0, the scale chain is effectively frozen.
    pub fn diagnostics(&self) -> ModelDiagnostics {
        let n = self.location_steps.len();
        let mut trees = Vec::with_capacity(2 * n);
        let mut feature_split_counts: Vec<usize> = Vec::new();

        fn collect_tree_diags(
            steps: &[BoostingStep],
            trees: &mut Vec<DistributionalTreeDiagnostic>,
            feature_split_counts: &mut Vec<usize>,
        ) {
            for step in steps {
                let slot = step.slot();
                let tree = slot.active_tree();
                let arena = tree.arena();

                let leaf_values: Vec<f64> = (0..arena.is_leaf.len())
                    .filter(|&i| arena.is_leaf[i])
                    .map(|i| arena.leaf_value[i])
                    .collect();

                let leaf_sample_counts: Vec<u64> = (0..arena.is_leaf.len())
                    .filter(|&i| arena.is_leaf[i])
                    .map(|i| arena.sample_count[i])
                    .collect();

                let max_depth_reached = (0..arena.is_leaf.len())
                    .filter(|&i| arena.is_leaf[i])
                    .map(|i| arena.depth[i] as usize)
                    .max()
                    .unwrap_or(0);

                let leaf_weight_stats = if leaf_values.is_empty() {
                    (0.0, 0.0, 0.0, 0.0)
                } else {
                    let min = leaf_values.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = leaf_values
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let sum: f64 = leaf_values.iter().sum();
                    let mean = sum / leaf_values.len() as f64;
                    let var: f64 = leaf_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / leaf_values.len() as f64;
                    (min, max, mean, var.sqrt())
                };

                let gains = slot.split_gains();
                let split_features: Vec<usize> = gains
                    .iter()
                    .enumerate()
                    .filter(|(_, &g)| g > 0.0)
                    .map(|(i, _)| i)
                    .collect();

                if !gains.is_empty() {
                    if feature_split_counts.is_empty() {
                        feature_split_counts.resize(gains.len(), 0);
                    }
                    for &fi in &split_features {
                        if fi < feature_split_counts.len() {
                            feature_split_counts[fi] += 1;
                        }
                    }
                }

                trees.push(DistributionalTreeDiagnostic {
                    n_leaves: leaf_values.len(),
                    max_depth_reached,
                    samples_seen: step.n_samples_seen(),
                    leaf_weight_stats,
                    split_features,
                    leaf_sample_counts,
                    prediction_mean: slot.prediction_mean(),
                    prediction_std: slot.prediction_std(),
                });
            }
        }

        collect_tree_diags(&self.location_steps, &mut trees, &mut feature_split_counts);
        collect_tree_diags(&self.scale_steps, &mut trees, &mut feature_split_counts);

        let location_trees = trees[..n].to_vec();
        let scale_trees = trees[n..].to_vec();
        let scale_trees_active = scale_trees.iter().filter(|t| t.n_leaves > 1).count();

        ModelDiagnostics {
            trees,
            location_trees,
            scale_trees,
            feature_split_counts,
            location_base: self.location_base,
            scale_base: self.scale_base,
            empirical_sigma: self.ewma_sq_err.sqrt(),
            scale_mode: self.scale_mode,
            scale_trees_active,
            auto_bandwidths: self.auto_bandwidths.clone(),
            ensemble_grad_mean: self.ensemble_grad_mean,
            ensemble_grad_std: self.ensemble_grad_std(),
        }
    }

    /// Ensemble-level diagnostics (location + optional scale) with per-tree
    /// contributions for a given input.
    ///
    /// Returns [`DistributionalDiagnostics`](crate::ensemble::diagnostics::DistributionalDiagnostics)
    /// containing location ensemble diagnostics, optional scale diagnostics
    /// (when in `TreeChain` mode), current `honest_sigma`, and the rolling
    /// `honest_sigma` baseline.
    pub fn ensemble_diagnostics(
        &self,
        features: &[f64],
    ) -> crate::ensemble::diagnostics::DistributionalDiagnostics {
        use crate::ensemble::diagnostics::build_ensemble_diagnostics;

        let location = build_ensemble_diagnostics(
            &self.location_steps,
            self.location_base,
            self.config.learning_rate,
            self.samples_seen,
            Some(features),
        );

        let scale = match self.scale_mode {
            ScaleMode::TreeChain => Some(build_ensemble_diagnostics(
                &self.scale_steps,
                self.scale_base,
                self.config.learning_rate,
                self.samples_seen,
                Some(features),
            )),
            ScaleMode::Empirical => None,
        };

        let honest_sigma = self.compute_honest_sigma(features);

        // Compute effective_mts from ring mean (end-of-cycle evaluation)
        let effective_mts = self.config.adaptive_mts.map(|(base_mts, k)| {
            if self.sigma_ring.is_empty() {
                return base_mts;
            }
            let mean_sigma = self.sigma_ring.iter().sum::<f64>() / self.sigma_ring.len() as f64;
            let floor = (base_mts as f64 * self.config.adaptive_mts_floor).max(100.0);
            (base_mts as f64 / (1.0 + k * mean_sigma)).max(floor) as u64
        });

        crate::ensemble::diagnostics::DistributionalDiagnostics {
            location,
            scale,
            honest_sigma,
            rolling_honest_sigma_mean: self.rolling_honest_sigma_mean,
            effective_mts,
        }
    }

    /// Per-tree contribution to the final prediction.
    ///
    /// Returns two vectors: location contributions and scale contributions.
    /// Each entry is `learning_rate * tree_prediction` -- the additive
    /// contribution of that boosting step to the final μ or log(σ).
    ///
    /// Summing `location_base + sum(location_contributions)` recovers μ.
    /// Summing `scale_base + sum(scale_contributions)` recovers log(σ).
    ///
    /// In `Empirical` scale mode, `scale_base` is `ln(empirical_sigma)` and
    /// `scale_contributions` are all zero (σ is not tree-derived).
    pub fn predict_decomposed(&self, features: &[f64]) -> DecomposedPrediction {
        let lr = self.config.learning_rate;
        let location: Vec<f64> = self
            .location_steps
            .iter()
            .map(|s| lr * s.predict(features))
            .collect();

        let (sb, scale) = match self.scale_mode {
            ScaleMode::Empirical => {
                let empirical_sigma = self.ewma_sq_err.sqrt().max(1e-8);
                (empirical_sigma.ln(), vec![0.0; self.location_steps.len()])
            }
            ScaleMode::TreeChain => {
                let s: Vec<f64> = self
                    .scale_steps
                    .iter()
                    .map(|s| lr * s.predict(features))
                    .collect();
                (self.scale_base, s)
            }
        };

        DecomposedPrediction {
            location_base: self.location_base,
            scale_base: sb,
            location_contributions: location,
            scale_contributions: scale,
        }
    }

    /// Feature importances based on accumulated split gains across all trees.
    ///
    /// Aggregates gains from both location and scale ensembles, then
    /// normalizes to sum to 1.0. Indexed by feature.
    /// Returns an empty Vec if no splits have occurred yet.
    pub fn feature_importances(&self) -> Vec<f64> {
        let mut totals: Vec<f64> = Vec::new();
        for steps in [&self.location_steps, &self.scale_steps] {
            for step in steps {
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
        }
        let sum: f64 = totals.iter().sum();
        if sum > 0.0 {
            totals.iter_mut().for_each(|v| *v /= sum);
        }
        totals
    }

    /// Feature importances split by ensemble: `(location_importances, scale_importances)`.
    ///
    /// Each vector is independently normalized to sum to 1.0.
    /// Useful for understanding which features drive the mean vs. the uncertainty.
    pub fn feature_importances_split(&self) -> (Vec<f64>, Vec<f64>) {
        fn aggregate(steps: &[BoostingStep]) -> Vec<f64> {
            let mut totals: Vec<f64> = Vec::new();
            for step in steps {
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
            let sum: f64 = totals.iter().sum();
            if sum > 0.0 {
                totals.iter_mut().for_each(|v| *v /= sum);
            }
            totals
        }

        (
            aggregate(&self.location_steps),
            aggregate(&self.scale_steps),
        )
    }
}
