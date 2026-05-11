//! Training logic for distributional SGBT.

use super::DistributionalSGBT;

impl DistributionalSGBT {
    /// Empirical-σ training: location trees only, σ from EWMA of squared errors.
    pub(crate) fn train_one_empirical(&mut self, target: f64, features: &[f64]) {
        // Current location prediction (before this step's update)
        let mut mu = self.location_base;
        for s in 0..self.location_steps.len() {
            mu += self.config.learning_rate * self.location_steps[s].predict(features);
        }

        // Update rolling honest_sigma mean (tree contribution variance EWMA)
        let honest_sigma = self.compute_honest_sigma(features);
        const HONEST_SIGMA_ALPHA: f64 = 0.001;
        self.rolling_honest_sigma_mean = (1.0 - HONEST_SIGMA_ALPHA)
            * self.rolling_honest_sigma_mean
            + HONEST_SIGMA_ALPHA * honest_sigma;

        // Empirical sigma from EWMA of squared prediction errors
        let err = target - mu;
        let alpha = self.empirical_sigma_alpha;
        self.ewma_sq_err = (1.0 - alpha) * self.ewma_sq_err + alpha * err * err;
        let empirical_sigma = self.ewma_sq_err.sqrt().max(1e-8);

        // Compute σ-ratio for uncertainty-modulated learning rate (PD controller)
        let sigma_ratio = if self.uncertainty_modulated_lr {
            // Compute sigma velocity (derivative of sigma over time)
            let d_sigma = empirical_sigma - self.prev_sigma;
            self.prev_sigma = empirical_sigma;

            // EWMA-smooth the velocity (same alpha as empirical sigma for synchronization)
            self.sigma_velocity = (1.0 - alpha) * self.sigma_velocity + alpha * d_sigma;

            // Adaptive derivative gain: self-calibrating, no config needed
            let k_d = if self.rolling_sigma_mean > 1e-12 {
                self.sigma_velocity.abs() / self.rolling_sigma_mean
            } else {
                0.0
            };

            // PD ratio: proportional + derivative
            let pd_sigma = empirical_sigma + k_d * self.sigma_velocity;
            let ratio = (pd_sigma / self.rolling_sigma_mean).clamp(0.1, 10.0);

            // Update rolling sigma mean with slow EWMA
            const SIGMA_EWMA_ALPHA: f64 = 0.001;
            self.rolling_sigma_mean = (1.0 - SIGMA_EWMA_ALPHA) * self.rolling_sigma_mean
                + SIGMA_EWMA_ALPHA * empirical_sigma;

            ratio
        } else {
            1.0
        };

        let base_lr = self.config.learning_rate;

        // Train location steps only -- no scale trees needed
        let mut mu_accum = self.location_base;
        for s in 0..self.location_steps.len() {
            let (g_mu, h_mu) = self.location_gradient(mu_accum, target);
            // Welford update for ensemble gradient stats
            self.update_ensemble_grad_stats(g_mu);
            let train_count = self.config.variant.train_count(h_mu, &mut self.rng_state);
            let loc_pred =
                self.location_steps[s].train_and_predict(features, g_mu, h_mu, train_count);
            mu_accum += (base_lr * sigma_ratio) * loc_pred;
        }

        // Refresh packed cache if interval reached
        self.maybe_refresh_packed_cache();
    }

    /// Tree-chain training: full NGBoost dual-chain with location + scale trees.
    pub(crate) fn train_one_tree_chain(&mut self, target: f64, features: &[f64]) {
        let mut mu = self.location_base;
        let mut log_sigma = self.scale_base;

        // Update rolling honest_sigma mean (tree contribution variance EWMA)
        let honest_sigma = self.compute_honest_sigma(features);
        const HONEST_SIGMA_ALPHA: f64 = 0.001;
        self.rolling_honest_sigma_mean = (1.0 - HONEST_SIGMA_ALPHA)
            * self.rolling_honest_sigma_mean
            + HONEST_SIGMA_ALPHA * honest_sigma;

        // Compute σ-ratio for uncertainty-modulated learning rate (PD controller).
        let sigma_ratio = if self.uncertainty_modulated_lr {
            let current_sigma = log_sigma.exp().max(1e-8);

            // Compute sigma velocity (derivative of sigma over time)
            let d_sigma = current_sigma - self.prev_sigma;
            self.prev_sigma = current_sigma;

            // EWMA-smooth the velocity (same alpha as empirical sigma for synchronization)
            let alpha = self.empirical_sigma_alpha;
            self.sigma_velocity = (1.0 - alpha) * self.sigma_velocity + alpha * d_sigma;

            // Adaptive derivative gain: self-calibrating, no config needed
            let k_d = if self.rolling_sigma_mean > 1e-12 {
                self.sigma_velocity.abs() / self.rolling_sigma_mean
            } else {
                0.0
            };

            // PD ratio: proportional + derivative
            let pd_sigma = current_sigma + k_d * self.sigma_velocity;
            let ratio = (pd_sigma / self.rolling_sigma_mean).clamp(0.1, 10.0);

            const SIGMA_EWMA_ALPHA: f64 = 0.001;
            self.rolling_sigma_mean = (1.0 - SIGMA_EWMA_ALPHA) * self.rolling_sigma_mean
                + SIGMA_EWMA_ALPHA * current_sigma;

            ratio
        } else {
            1.0
        };

        let base_lr = self.config.learning_rate;

        // Sequential boosting: both ensembles target their respective residuals
        for s in 0..self.location_steps.len() {
            let sigma = log_sigma.exp().max(1e-8);
            let z = (target - mu) / sigma;

            // Location gradients (squared or Huber loss w.r.t. mu)
            let (g_mu, h_mu) = self.location_gradient(mu, target);
            // Welford update for ensemble gradient stats
            self.update_ensemble_grad_stats(g_mu);

            // Scale gradients (NLL w.r.t. log_sigma)
            let g_sigma = 1.0 - z * z;
            let h_sigma = (2.0 * z * z).clamp(0.01, 100.0);

            let train_count = self.config.variant.train_count(h_mu, &mut self.rng_state);

            // Train location step -- σ-modulated LR when enabled
            let loc_pred =
                self.location_steps[s].train_and_predict(features, g_mu, h_mu, train_count);
            mu += (base_lr * sigma_ratio) * loc_pred;

            // Train scale step -- ALWAYS at unmodulated base rate.
            let scale_pred =
                self.scale_steps[s].train_and_predict(features, g_sigma, h_sigma, train_count);
            log_sigma += base_lr * scale_pred;
        }

        // Also update empirical sigma tracker for diagnostics
        let err = target - mu;
        let alpha = self.empirical_sigma_alpha;
        self.ewma_sq_err = (1.0 - alpha) * self.ewma_sq_err + alpha * err * err;

        // Refresh packed cache if interval reached
        self.maybe_refresh_packed_cache();
    }

    /// Compute location gradient (squared or adaptive Huber).
    ///
    /// When `huber_k` is configured, uses Huber loss with adaptive
    /// `delta = k * empirical_sigma` for bounded gradients.
    #[inline]
    pub(crate) fn location_gradient(&self, mu: f64, target: f64) -> (f64, f64) {
        if let Some(k) = self.config.huber_k {
            let delta = k * self.ewma_sq_err.sqrt().max(1e-8);
            let residual = mu - target;
            if residual.abs() <= delta {
                (residual, 1.0)
            } else {
                (delta * residual.signum(), 1e-6)
            }
        } else {
            (mu - target, 1.0)
        }
    }

    /// Welford update for ensemble-level gradient statistics.
    #[inline]
    pub(crate) fn update_ensemble_grad_stats(&mut self, gradient: f64) {
        self.ensemble_grad_count += 1;
        let delta = gradient - self.ensemble_grad_mean;
        self.ensemble_grad_mean += delta / self.ensemble_grad_count as f64;
        let delta2 = gradient - self.ensemble_grad_mean;
        self.ensemble_grad_m2 += delta * delta2;
    }

    /// Ensemble-level gradient standard deviation.
    pub fn ensemble_grad_std(&self) -> f64 {
        if self.ensemble_grad_count < 2 {
            return 0.0;
        }
        (self.ensemble_grad_m2 / (self.ensemble_grad_count - 1) as f64)
            .sqrt()
            .max(0.0)
    }

    /// Ensemble-level gradient mean.
    pub fn ensemble_grad_mean(&self) -> f64 {
        self.ensemble_grad_mean
    }

    /// Check if packed cache should be refreshed and do so if interval reached.
    pub(crate) fn maybe_refresh_packed_cache(&mut self) {
        if self.packed_refresh_interval > 0 {
            self.samples_since_refresh += 1;
            if self.samples_since_refresh >= self.packed_refresh_interval {
                self.refresh_packed_cache();
                self.samples_since_refresh = 0;
            }
        }
    }

    /// Re-export the location ensemble into the packed cache.
    pub(crate) fn refresh_packed_cache(&mut self) {
        // Determine n_features from the first tree that has been initialized
        let n_features = self
            .location_steps
            .iter()
            .filter_map(|s| s.slot().active_tree().n_features())
            .max()
            .unwrap_or(0);

        if n_features == 0 {
            return;
        }

        let (bytes, base) = crate::export_embedded::export_distributional_packed(self, n_features);
        self.packed_cache = Some(crate::ensemble::distributional::PackedInferenceCache {
            bytes,
            base,
            n_features,
        });
    }

    /// Refresh auto-bandwidths if any tree has been replaced since last computation.
    pub(crate) fn refresh_bandwidths(&mut self) {
        let current_sum: u64 = self
            .location_steps
            .iter()
            .chain(self.scale_steps.iter())
            .map(|s| s.slot().replacements())
            .sum();
        if current_sum != self.last_replacement_sum || self.auto_bandwidths.is_empty() {
            self.auto_bandwidths = self.compute_auto_bandwidths();
            self.last_replacement_sum = current_sum;
        }
    }

    /// Compute per-feature auto-calibrated bandwidths from all trees.
    fn compute_auto_bandwidths(&self) -> Vec<f64> {
        const K: f64 = 2.0;

        let n_features = self
            .location_steps
            .iter()
            .chain(self.scale_steps.iter())
            .filter_map(|s| s.slot().active_tree().n_features())
            .max()
            .unwrap_or(0);

        if n_features == 0 {
            return Vec::new();
        }

        let mut all_thresholds: Vec<Vec<f64>> = vec![Vec::new(); n_features];

        for step in self.location_steps.iter().chain(self.scale_steps.iter()) {
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
}
