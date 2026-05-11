//! Prediction and inference methods for distributional SGBT.

use super::{DistributionalSGBT, GaussianPrediction};
use crate::ensemble::config::ScaleMode;

impl DistributionalSGBT {
    /// Full-tree location prediction (fallback when packed cache is unavailable).
    fn predict_full_trees(&self, features: &[f64]) -> f64 {
        let mut mu = self.location_base;
        if self.auto_bandwidths.is_empty() {
            for s in 0..self.location_steps.len() {
                mu += self.config.learning_rate * self.location_steps[s].predict(features);
            }
        } else {
            for s in 0..self.location_steps.len() {
                mu += self.config.learning_rate
                    * self.location_steps[s].predict_smooth_auto(features, &self.auto_bandwidths);
            }
        }
        mu
    }

    /// Predict the full Gaussian distribution for a feature vector.
    ///
    /// When a packed cache is available, uses it for the location (μ) prediction
    /// via contiguous BFS-packed memory traversal. Falls back to full tree
    /// traversal if the cache is absent or produces non-finite results.
    ///
    /// Sigma computation always uses the primary path (EWMA or scale chain)
    /// and is unaffected by the packed cache.
    pub fn predict(&self, features: &[f64]) -> GaussianPrediction {
        // Try packed cache for mu if available
        let mu = if let Some(ref cache) = self.packed_cache {
            let features_f32: Vec<f32> = features.iter().map(|&v| v as f32).collect();
            match irithyll_core::EnsembleView::from_bytes(&cache.bytes) {
                Ok(view) => {
                    let packed_mu = cache.base + view.predict(&features_f32) as f64;
                    if packed_mu.is_finite() {
                        packed_mu
                    } else {
                        self.predict_full_trees(features)
                    }
                }
                Err(_) => self.predict_full_trees(features),
            }
        } else {
            self.predict_full_trees(features)
        };

        let (sigma, log_sigma) = match self.scale_mode {
            ScaleMode::Empirical => {
                let s = self.ewma_sq_err.sqrt().max(1e-8);
                (s, s.ln())
            }
            ScaleMode::TreeChain => {
                let mut ls = self.scale_base;
                if self.auto_bandwidths.is_empty() {
                    for s in 0..self.scale_steps.len() {
                        ls += self.config.learning_rate * self.scale_steps[s].predict(features);
                    }
                } else {
                    for s in 0..self.scale_steps.len() {
                        ls += self.config.learning_rate
                            * self.scale_steps[s]
                                .predict_smooth_auto(features, &self.auto_bandwidths);
                    }
                }
                (ls.exp().max(1e-8), ls)
            }
        };

        let honest_sigma = self.compute_honest_sigma(features);

        GaussianPrediction {
            mu,
            sigma,
            log_sigma,
            honest_sigma,
        }
    }

    /// Predict using sigmoid-blended soft routing for smooth interpolation.
    ///
    /// Instead of hard left/right routing at tree split nodes, each split
    /// uses sigmoid blending: `alpha = sigmoid((threshold - feature) / bandwidth)`.
    /// The result is a continuous function that varies smoothly with every
    /// feature change.
    ///
    /// `bandwidth` controls transition sharpness: smaller = sharper (closer
    /// to hard splits), larger = smoother.
    pub fn predict_smooth(&self, features: &[f64], bandwidth: f64) -> GaussianPrediction {
        let mut mu = self.location_base;
        for s in 0..self.location_steps.len() {
            mu += self.config.learning_rate
                * self.location_steps[s].predict_smooth(features, bandwidth);
        }

        let (sigma, log_sigma) = match self.scale_mode {
            ScaleMode::Empirical => {
                let s = self.ewma_sq_err.sqrt().max(1e-8);
                (s, s.ln())
            }
            ScaleMode::TreeChain => {
                let mut ls = self.scale_base;
                for s in 0..self.scale_steps.len() {
                    ls += self.config.learning_rate
                        * self.scale_steps[s].predict_smooth(features, bandwidth);
                }
                (ls.exp().max(1e-8), ls)
            }
        };

        let honest_sigma = self.compute_honest_sigma(features);

        GaussianPrediction {
            mu,
            sigma,
            log_sigma,
            honest_sigma,
        }
    }

    /// Predict with parent-leaf linear interpolation.
    ///
    /// Blends each leaf prediction with its parent's preserved prediction
    /// based on sample count, preventing stale predictions from fresh leaves.
    pub fn predict_interpolated(&self, features: &[f64]) -> GaussianPrediction {
        let mut mu = self.location_base;
        for s in 0..self.location_steps.len() {
            mu += self.config.learning_rate * self.location_steps[s].predict_interpolated(features);
        }

        let (sigma, log_sigma) = match self.scale_mode {
            ScaleMode::Empirical => {
                let s = self.ewma_sq_err.sqrt().max(1e-8);
                (s, s.ln())
            }
            ScaleMode::TreeChain => {
                let mut ls = self.scale_base;
                for s in 0..self.scale_steps.len() {
                    ls += self.config.learning_rate
                        * self.scale_steps[s].predict_interpolated(features);
                }
                (ls.exp().max(1e-8), ls)
            }
        };

        let honest_sigma = self.compute_honest_sigma(features);

        GaussianPrediction {
            mu,
            sigma,
            log_sigma,
            honest_sigma,
        }
    }

    /// Predict with sibling-based interpolation for feature-continuous predictions.
    ///
    /// At each split node near the threshold boundary, blends left and right
    /// subtree predictions linearly. Uses auto-calibrated bandwidths as the
    /// interpolation margin. Predictions vary continuously as features change.
    pub fn predict_sibling_interpolated(&self, features: &[f64]) -> GaussianPrediction {
        let mut mu = self.location_base;
        for s in 0..self.location_steps.len() {
            mu += self.config.learning_rate
                * self.location_steps[s]
                    .predict_sibling_interpolated(features, &self.auto_bandwidths);
        }

        let (sigma, log_sigma) = match self.scale_mode {
            ScaleMode::Empirical => {
                let s = self.ewma_sq_err.sqrt().max(1e-8);
                (s, s.ln())
            }
            ScaleMode::TreeChain => {
                let mut ls = self.scale_base;
                for s in 0..self.scale_steps.len() {
                    ls += self.config.learning_rate
                        * self.scale_steps[s]
                            .predict_sibling_interpolated(features, &self.auto_bandwidths);
                }
                (ls.exp().max(1e-8), ls)
            }
        };

        let honest_sigma = self.compute_honest_sigma(features);

        GaussianPrediction {
            mu,
            sigma,
            log_sigma,
            honest_sigma,
        }
    }

    /// Predict with graduated active-shadow blending.
    ///
    /// Smoothly transitions between active and shadow trees during replacement.
    /// Requires `shadow_warmup` to be configured.
    pub fn predict_graduated(&self, features: &[f64]) -> GaussianPrediction {
        let mut mu = self.location_base;
        for s in 0..self.location_steps.len() {
            mu += self.config.learning_rate * self.location_steps[s].predict_graduated(features);
        }

        let (sigma, log_sigma) = match self.scale_mode {
            ScaleMode::Empirical => {
                let s = self.ewma_sq_err.sqrt().max(1e-8);
                (s, s.ln())
            }
            ScaleMode::TreeChain => {
                let mut ls = self.scale_base;
                for s in 0..self.scale_steps.len() {
                    ls +=
                        self.config.learning_rate * self.scale_steps[s].predict_graduated(features);
                }
                (ls.exp().max(1e-8), ls)
            }
        };

        let honest_sigma = self.compute_honest_sigma(features);

        GaussianPrediction {
            mu,
            sigma,
            log_sigma,
            honest_sigma,
        }
    }

    /// Predict with graduated blending + sibling interpolation (premium path).
    pub fn predict_graduated_sibling_interpolated(&self, features: &[f64]) -> GaussianPrediction {
        let mut mu = self.location_base;
        for s in 0..self.location_steps.len() {
            mu += self.config.learning_rate
                * self.location_steps[s]
                    .predict_graduated_sibling_interpolated(features, &self.auto_bandwidths);
        }

        let (sigma, log_sigma) = match self.scale_mode {
            ScaleMode::Empirical => {
                let s = self.ewma_sq_err.sqrt().max(1e-8);
                (s, s.ln())
            }
            ScaleMode::TreeChain => {
                let mut ls = self.scale_base;
                for s in 0..self.scale_steps.len() {
                    ls += self.config.learning_rate
                        * self.scale_steps[s].predict_graduated_sibling_interpolated(
                            features,
                            &self.auto_bandwidths,
                        );
                }
                (ls.exp().max(1e-8), ls)
            }
        };

        let honest_sigma = self.compute_honest_sigma(features);

        GaussianPrediction {
            mu,
            sigma,
            log_sigma,
            honest_sigma,
        }
    }

    /// Predict using per-node auto-bandwidth soft routing.
    ///
    /// Every prediction is a continuous weighted blend instead of a
    /// piecewise-constant step function. No training changes.
    pub fn predict_soft_routed(&self, features: &[f64]) -> GaussianPrediction {
        let mut mu = self.location_base;
        for step in &self.location_steps {
            mu += self.config.learning_rate * step.predict_soft_routed(features);
        }

        let (sigma, log_sigma) = match self.scale_mode {
            ScaleMode::Empirical => {
                let s = self.ewma_sq_err.sqrt().max(1e-8);
                (s, s.ln())
            }
            ScaleMode::TreeChain => {
                let mut ls = self.scale_base;
                for step in &self.scale_steps {
                    ls += self.config.learning_rate * step.predict_soft_routed(features);
                }
                (ls.exp().max(1e-8), ls)
            }
        };

        let honest_sigma = self.compute_honest_sigma(features);

        GaussianPrediction {
            mu,
            sigma,
            log_sigma,
            honest_sigma,
        }
    }

    /// Predict with σ-ratio diagnostic exposed.
    ///
    /// Returns `(mu, sigma, sigma_ratio)` where `sigma_ratio` is
    /// `current_sigma / rolling_sigma_mean` -- the multiplier applied to the
    /// location learning rate when uncertainty_modulated_lr is enabled.
    ///
    /// When σ-modulation is disabled, `sigma_ratio` is always `1.0`.
    pub fn predict_distributional(&self, features: &[f64]) -> (f64, f64, f64) {
        let pred = self.predict(features);
        let sigma_ratio = if self.uncertainty_modulated_lr {
            (pred.sigma / self.rolling_sigma_mean).clamp(0.1, 10.0)
        } else {
            1.0
        };
        (pred.mu, pred.sigma, sigma_ratio)
    }

    /// Current empirical sigma (`sqrt(ewma_sq_err)`).
    ///
    /// Returns the model's recent error magnitude. Available in both scale modes.
    #[inline]
    pub fn empirical_sigma(&self) -> f64 {
        self.ewma_sq_err.sqrt()
    }

    /// Current scale mode.
    #[inline]
    pub fn scale_mode(&self) -> ScaleMode {
        self.scale_mode
    }

    /// Current σ velocity -- the EWMA-smoothed derivative of empirical σ.
    ///
    /// Positive values indicate growing prediction errors (model deteriorating
    /// or regime change). Negative values indicate improving predictions.
    /// Only meaningful when `ScaleMode::Empirical` is active.
    #[inline]
    pub fn sigma_velocity(&self) -> f64 {
        self.sigma_velocity
    }

    /// Predict the mean (location parameter) only.
    #[inline]
    pub fn predict_mu(&self, features: &[f64]) -> f64 {
        self.predict(features).mu
    }

    /// Predict the standard deviation (scale parameter) only.
    #[inline]
    pub fn predict_sigma(&self, features: &[f64]) -> f64 {
        self.predict(features).sigma
    }

    /// Predict a symmetric confidence interval.
    ///
    /// `confidence` is the Z-score multiplier:
    /// - 1.0 → 68% CI
    /// - 1.96 → 95% CI
    /// - 2.576 → 99% CI
    pub fn predict_interval(&self, features: &[f64], confidence: f64) -> (f64, f64) {
        let pred = self.predict(features);
        (pred.lower(confidence), pred.upper(confidence))
    }

    /// Batch prediction.
    pub fn predict_batch(&self, feature_matrix: &[Vec<f64>]) -> Vec<GaussianPrediction> {
        feature_matrix.iter().map(|f| self.predict(f)).collect()
    }

    /// Train on a batch of observations.
    pub fn train_batch<O: crate::sample::Observation>(&mut self, samples: &[O]) {
        for sample in samples {
            self.train_one(sample);
        }
    }

    /// Train on a batch with periodic callback.
    pub fn train_batch_with_callback<O: crate::sample::Observation, F: FnMut(usize)>(
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
}
