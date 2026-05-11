use crate::ensemble::step::BoostingStep;
use crate::ensemble::SGBT;
use crate::loss::Loss;

impl<L: Loss> SGBT<L> {
    /// Immutable access to boosting steps.
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

        let sum: f64 = totals.iter().sum();
        if sum > 0.0 {
            totals.iter_mut().for_each(|v| *v /= sum);
        } else {
            // No splits have accumulated any gain yet — return empty per contract.
            // A non-empty zero vec would violate the "sums to 1.0" invariant that
            // callers (and the public API docs) rely on.
            totals.clear();
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

    /// Refresh auto-bandwidths if any tree has been replaced since last computation.
    pub(crate) fn refresh_bandwidths(&mut self) {
        let current_sum: u64 = self.steps.iter().map(|s| s.slot().replacements()).sum();
        if current_sum != self.last_replacement_sum || self.auto_bandwidths.is_empty() {
            self.auto_bandwidths = self.compute_auto_bandwidths();
            self.last_replacement_sum = current_sum;
        }
    }

    /// Compute per-feature auto-calibrated bandwidths from all trees.
    ///
    /// For each feature, collects all split thresholds across all trees,
    /// computes the median gap between consecutive unique thresholds, and
    /// returns `median_gap * K` (K = 2.0).
    ///
    /// Edge cases:
    /// - Feature with < 3 unique thresholds: `range / n_bins * K`
    /// - Feature never split on (< 2 unique thresholds): `f64::INFINITY` (hard routing)
    fn compute_auto_bandwidths(&self) -> Vec<f64> {
        const K: f64 = 2.0;

        let n_features = self
            .steps
            .iter()
            .filter_map(|s| s.slot().active_tree().n_features())
            .max()
            .unwrap_or(0);

        if n_features == 0 {
            return Vec::new();
        }

        let mut all_thresholds: Vec<Vec<f64>> = vec![Vec::new(); n_features];

        for step in &self.steps {
            let tree_thresholds = step
                .slot()
                .active_tree()
                .collect_split_thresholds_per_feature();
            for (i, ts) in tree_thresholds.into_iter().enumerate() {
                if i < n_features {
                    all_thresholds[i].extend(ts);
                }
            }
        }

        let n_bins = self.config.n_bins as f64;

        all_thresholds
            .iter()
            .map(|ts| {
                if ts.is_empty() {
                    return f64::INFINITY;
                }

                let mut sorted = ts.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

                if sorted.len() < 2 {
                    return f64::INFINITY;
                }

                let mut gaps: Vec<f64> = sorted.windows(2).map(|w| w[1] - w[0]).collect();

                if sorted.len() < 3 {
                    let range = sorted.last().unwrap() - sorted.first().unwrap();
                    if range < 1e-15 {
                        return f64::INFINITY;
                    }
                    return (range / n_bins) * K;
                }

                gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median_gap = if gaps.len() % 2 == 0 {
                    (gaps[gaps.len() / 2 - 1] + gaps[gaps.len() / 2]) / 2.0
                } else {
                    gaps[gaps.len() / 2]
                };

                if median_gap < 1e-15 {
                    f64::INFINITY
                } else {
                    median_gap * K
                }
            })
            .collect()
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
        self.auto_bandwidths.clear();
        self.last_replacement_sum = 0;
        self.rolling_contribution_sigma = 0.0;
        self.sigma_ring.clear();
        self.mts_replacement_sum = 0;
        self.diag.prev_contributions.clear();
        self.diag.prev_prev_contributions.clear();
        self.diag.cached_residual_alignment = 0.0;
        self.diag.cached_reg_sensitivity = 0.0;
        self.diag.cached_depth_sufficiency = 0.0;
        self.diag.cached_effective_dof = 0.0;
        self.diag.contribution_accuracy = vec![0.0; self.steps.len()];
    }
}
