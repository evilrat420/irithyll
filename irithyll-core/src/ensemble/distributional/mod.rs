//! Distributional SGBT -- outputs Gaussian N(μ, σ²) instead of a point estimate.
//!
//! [`DistributionalSGBT`] supports two scale estimation modes via
//! [`ScaleMode`]:
//!
//! ## Empirical σ (default)
//!
//! Tracks an EWMA of squared prediction errors:
//!
//! ```text
//! err = target - mu
//! ewma_sq_err = alpha * err² + (1 - alpha) * ewma_sq_err
//! sigma = sqrt(ewma_sq_err)
//! ```
//!
//! Always calibrated (σ literally *is* recent error magnitude), zero tuning,
//! O(1) memory and compute.  When `uncertainty_modulated_lr` is enabled,
//! high recent errors → σ large → location LR scales up → faster correction.
//!
//! ## Tree chain (NGBoost-style)
//!
//! Maintains two independent tree ensembles: one for location (μ), one for
//! scale (log σ).  Gives feature-conditional uncertainty but requires strong
//! scale-gradient signal for the trees to split.
//!
//! # References
//!
//! Duan et al. (2020). "NGBoost: Natural Gradient Boosting for Probabilistic Prediction."

mod diagnostics;
mod inference;
mod training;

#[cfg(test)]
mod tests;

pub use diagnostics::{DecomposedPrediction, DistributionalTreeDiagnostic, ModelDiagnostics};

use alloc::vec::Vec;

use crate::ensemble::config::{SGBTConfig, ScaleMode};
use crate::ensemble::step::BoostingStep;
use crate::sample::{Observation, SampleRef};

/// Cached packed f32 binary for fast location-only inference.
///
/// Re-exported periodically from the location ensemble. Predictions use
/// contiguous BFS-packed memory for cache-optimal tree traversal.
struct PackedInferenceCache {
    bytes: Vec<u8>,
    base: f64,
    n_features: usize,
}

impl Clone for PackedInferenceCache {
    fn clone(&self) -> Self {
        Self {
            bytes: self.bytes.clone(),
            base: self.base,
            n_features: self.n_features,
        }
    }
}

/// Prediction from a distributional model: full Gaussian N(μ, σ²).
#[derive(Debug, Clone, Copy)]
pub struct GaussianPrediction {
    /// Location parameter (mean).
    pub mu: f64,
    /// Scale parameter (standard deviation, always > 0).
    pub sigma: f64,
    /// Log of scale parameter (raw model output for scale ensemble).
    pub log_sigma: f64,
    /// Tree contribution variance (epistemic uncertainty).
    ///
    /// Standard deviation of individual location-tree contributions,
    /// computed via one-pass Welford variance with Bessel's correction.
    /// Reacts instantly when trees disagree (no EWMA lag), making it
    /// superior to empirical sigma for regime-change detection.
    ///
    /// Zero when the model has 0 or 1 active location trees.
    pub honest_sigma: f64,
}

impl GaussianPrediction {
    /// Lower bound of a symmetric confidence interval.
    ///
    /// For 95% CI, use `z = 1.96`.
    #[inline]
    pub fn lower(&self, z: f64) -> f64 {
        self.mu - z * self.sigma
    }

    /// Upper bound of a symmetric confidence interval.
    #[inline]
    pub fn upper(&self, z: f64) -> f64 {
        self.mu + z * self.sigma
    }
}

/// NGBoost-style distributional streaming gradient boosted trees.
///
/// Outputs a full Gaussian predictive distribution N(μ, σ²) by maintaining two
/// independent ensembles -- one for location (mean) and one for scale (log-sigma).
///
/// # Example
///
/// ```text
/// use irithyll_core::SGBTConfig;
/// use irithyll_core::ensemble::distributional::DistributionalSGBT;
///
/// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
/// let mut model = DistributionalSGBT::new(config);
///
/// // Train on streaming data
/// model.train_one(&(vec![1.0, 2.0], 3.5));
///
/// // Get full distributional prediction
/// let pred = model.predict(&[1.0, 2.0]);
/// println!("mean={}, sigma={}", pred.mu, pred.sigma);
/// ```
pub struct DistributionalSGBT {
    config: SGBTConfig,
    location_steps: Vec<BoostingStep>,
    scale_steps: Vec<BoostingStep>,
    location_base: f64,
    scale_base: f64,
    base_initialized: bool,
    initial_targets: Vec<f64>,
    initial_target_count: usize,
    samples_seen: u64,
    rng_state: u64,
    uncertainty_modulated_lr: bool,
    rolling_sigma_mean: f64,
    scale_mode: ScaleMode,
    ewma_sq_err: f64,
    empirical_sigma_alpha: f64,
    prev_sigma: f64,
    sigma_velocity: f64,
    auto_bandwidths: Vec<f64>,
    last_replacement_sum: u64,
    ensemble_grad_mean: f64,
    ensemble_grad_m2: f64,
    ensemble_grad_count: u64,
    rolling_honest_sigma_mean: f64,
    packed_cache: Option<PackedInferenceCache>,
    samples_since_refresh: u64,
    packed_refresh_interval: u64,
}

impl Clone for DistributionalSGBT {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            location_steps: self.location_steps.clone(),
            scale_steps: self.scale_steps.clone(),
            location_base: self.location_base,
            scale_base: self.scale_base,
            base_initialized: self.base_initialized,
            initial_targets: self.initial_targets.clone(),
            initial_target_count: self.initial_target_count,
            samples_seen: self.samples_seen,
            rng_state: self.rng_state,
            uncertainty_modulated_lr: self.uncertainty_modulated_lr,
            rolling_sigma_mean: self.rolling_sigma_mean,
            scale_mode: self.scale_mode,
            ewma_sq_err: self.ewma_sq_err,
            empirical_sigma_alpha: self.empirical_sigma_alpha,
            prev_sigma: self.prev_sigma,
            sigma_velocity: self.sigma_velocity,
            auto_bandwidths: self.auto_bandwidths.clone(),
            last_replacement_sum: self.last_replacement_sum,
            ensemble_grad_mean: self.ensemble_grad_mean,
            ensemble_grad_m2: self.ensemble_grad_m2,
            ensemble_grad_count: self.ensemble_grad_count,
            rolling_honest_sigma_mean: self.rolling_honest_sigma_mean,
            packed_cache: self.packed_cache.clone(),
            samples_since_refresh: self.samples_since_refresh,
            packed_refresh_interval: self.packed_refresh_interval,
        }
    }
}

impl core::fmt::Debug for DistributionalSGBT {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut s = f.debug_struct("DistributionalSGBT");
        s.field("n_steps", &self.location_steps.len())
            .field("samples_seen", &self.samples_seen)
            .field("location_base", &self.location_base)
            .field("scale_mode", &self.scale_mode)
            .field("base_initialized", &self.base_initialized);
        match self.scale_mode {
            ScaleMode::Empirical => {
                s.field("empirical_sigma", &crate::math::sqrt(self.ewma_sq_err));
            }
            ScaleMode::TreeChain => {
                s.field("scale_base", &self.scale_base);
            }
        }
        if self.uncertainty_modulated_lr {
            s.field("rolling_sigma_mean", &self.rolling_sigma_mean);
        }
        s.finish()
    }
}

impl DistributionalSGBT {
    /// Create a new distributional SGBT model.
    pub fn new(config: SGBTConfig) -> Self {
        let n_steps = config.n_steps;
        let initial_target_count = config.initial_target_count;
        let seed = config.seed;
        let uncertainty_modulated_lr = config.uncertainty_modulated_lr;
        let scale_mode = config.scale_mode;

        let leaf_decay_alpha = config
            .leaf_half_life
            .map(|hl| crate::math::exp(-crate::math::ln(2.0) / hl as f64));
        let tree_config = crate::ensemble::config::build_tree_config(&config)
            .leaf_decay_alpha_opt(leaf_decay_alpha);
        let max_tree_samples = config.max_tree_samples;
        let shadow_warmup = config.shadow_warmup.unwrap_or(0);

        let build_steps = |salt: u64| -> Vec<BoostingStep> {
            (0..n_steps)
                .map(|i| {
                    let mut tc = tree_config.clone();
                    tc.seed = seed ^ salt ^ (i as u64);
                    let detector = config.drift_detector.create();
                    if shadow_warmup > 0 {
                        BoostingStep::new_with_graduated(
                            tc,
                            detector,
                            max_tree_samples,
                            shadow_warmup,
                        )
                    } else {
                        BoostingStep::new_with_max_samples(tc, detector, max_tree_samples)
                    }
                })
                .collect()
        };

        let location_steps = build_steps(0);
        let scale_steps = build_steps(0xD15C_A1E5_5CA1_E000);

        Self {
            config,
            location_steps,
            scale_steps,
            location_base: 0.0,
            scale_base: 0.0,
            base_initialized: false,
            initial_targets: Vec::with_capacity(initial_target_count),
            initial_target_count,
            samples_seen: 0,
            rng_state: 1u64.wrapping_add(seed),
            uncertainty_modulated_lr,
            rolling_sigma_mean: 1.0,
            scale_mode,
            ewma_sq_err: 0.0,
            empirical_sigma_alpha: 0.05,
            prev_sigma: 0.0,
            sigma_velocity: 0.0,
            auto_bandwidths: Vec::new(),
            last_replacement_sum: 0,
            ensemble_grad_mean: 0.0,
            ensemble_grad_m2: 0.0,
            ensemble_grad_count: 0,
            rolling_honest_sigma_mean: 1.0,
            packed_cache: None,
            samples_since_refresh: 0,
            packed_refresh_interval: 1000,
        }
    }

    /// Access the configuration.
    pub fn config(&self) -> &SGBTConfig {
        &self.config
    }

    /// Train on a single observation.
    pub fn train_one(&mut self, obs: &impl Observation) {
        training::train_distributional_one(self, obs);
    }

    /// Train on a batch of observations.
    pub fn train_batch(&mut self, samples: &[(Vec<f64>, f64)]) {
        for (features, target) in samples {
            self.train_one(&(features.clone(), *target));
        }
    }

    /// Predict the full distributional output for a single sample.
    pub fn predict(&self, features: &[f64]) -> GaussianPrediction {
        inference::predict_distributional(self, features)
    }

    /// Predict distributional output for multiple samples (batch).
    pub fn predict_batch(&self, batch: &[Vec<f64>]) -> Vec<GaussianPrediction> {
        batch.iter().map(|f| self.predict(f)).collect()
    }

    /// Predict a confidence interval with the given z-score.
    ///
    /// Returns `(lower, upper)` for a symmetric interval around μ.
    pub fn predict_interval(&self, features: &[f64], z: f64) -> (f64, f64) {
        let pred = self.predict(features);
        (pred.lower(z), pred.upper(z))
    }

    /// Tuple form: `(μ, σ, σ_ratio)` where σ_ratio is 1.0 if uncertainty
    /// modulation is disabled, otherwise the ratio for LR scaling.
    pub fn predict_distributional(&self, features: &[f64]) -> (f64, f64, f64) {
        let pred = self.predict(features);
        let ratio = if self.uncertainty_modulated_lr {
            (pred.honest_sigma / self.rolling_honest_sigma_mean).clamp(0.1, 10.0)
        } else {
            1.0
        };
        (pred.mu, pred.sigma, ratio)
    }

    /// Predict with sigmoid-blended soft routing for smooth interpolation.
    ///
    /// Instead of hard left/right routing at tree split nodes, each split
    /// uses sigmoid blending: `alpha = sigmoid((threshold - feature) / bandwidth)`.
    /// The result is a continuous function that varies smoothly with every
    /// feature change.
    ///
    /// `bandwidth` controls transition sharpness: smaller = sharper (closer
    /// to hard splits), larger = smoother.
    pub fn predict_smooth(&self, features: &[f64], bandwidth: f64) -> GaussianPrediction {
        inference::predict_smooth(self, features, bandwidth)
    }

    /// Predict with parent-leaf linear interpolation.
    ///
    /// Blends each leaf prediction with its parent's preserved prediction
    /// based on sample count, preventing stale predictions from fresh leaves.
    pub fn predict_interpolated(&self, features: &[f64]) -> GaussianPrediction {
        inference::predict_interpolated(self, features)
    }

    /// Predict with sibling-based interpolation for feature-continuous predictions.
    ///
    /// At each split node near the threshold boundary, blends left and right
    /// subtree predictions linearly. Uses auto-calibrated bandwidths as the
    /// interpolation margin. Predictions vary continuously as features change.
    pub fn predict_sibling_interpolated(&self, features: &[f64]) -> GaussianPrediction {
        inference::predict_sibling_interpolated(self, features)
    }

    /// Is the model initialized with base predictions?
    pub fn is_initialized(&self) -> bool {
        self.base_initialized
    }

    /// Number of location ensemble trees.
    pub fn n_location_trees(&self) -> usize {
        self.location_steps.len()
    }

    /// Number of scale ensemble trees.
    pub fn n_scale_trees(&self) -> usize {
        self.scale_steps.len()
    }

    /// Total number of trees in both ensembles.
    pub fn n_trees(&self) -> usize {
        self.location_steps.len() + self.scale_steps.len()
    }

    /// Total samples seen.
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Is uncertainty-modulated learning rate enabled?
    pub fn is_uncertainty_modulated(&self) -> bool {
        self.uncertainty_modulated_lr
    }

    /// Current rolling mean of predicted σ.
    pub fn rolling_sigma_mean(&self) -> f64 {
        self.rolling_sigma_mean
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.location_steps.clear();
        self.scale_steps.clear();
        self.location_base = 0.0;
        self.scale_base = 0.0;
        self.base_initialized = false;
        self.initial_targets.clear();
        self.samples_seen = 0;
        self.rng_state = 1u64.wrapping_add(self.config.seed);
        self.rolling_sigma_mean = 1.0;
        self.ewma_sq_err = 0.0;
        self.prev_sigma = 0.0;
        self.sigma_velocity = 0.0;
        self.auto_bandwidths.clear();
        self.ensemble_grad_mean = 0.0;
        self.ensemble_grad_m2 = 0.0;
        self.ensemble_grad_count = 0;
        self.rolling_honest_sigma_mean = 1.0;
        self.packed_cache = None;
    }

    /// Full model diagnostics.
    pub fn diagnostics(&self) -> ModelDiagnostics {
        diagnostics::compute_diagnostics(self)
    }

    /// Decomposed prediction (per-tree contributions).
    pub fn predict_decomposed(&self, features: &[f64]) -> DecomposedPrediction {
        diagnostics::decompose_prediction(self, features)
    }

    /// Feature importances (location + scale combined).
    pub fn feature_importances(&self) -> Vec<f64> {
        diagnostics::compute_feature_importances(self, false)
    }

    /// Feature importances split by ensemble.
    pub fn feature_importances_split(&self) -> (Vec<f64>, Vec<f64>) {
        let location = diagnostics::compute_feature_importances(self, true);
        let scale = diagnostics::compute_feature_importances_scale(self);
        (location, scale)
    }

    /// Compute honest_sigma from current location tree predictions.
    #[allow(dead_code)]
    fn compute_honest_sigma(&self, features: &[f64]) -> f64 {
        if self.location_steps.len() < 2 {
            return 0.0;
        }

        let preds: Vec<f64> = self
            .location_steps
            .iter()
            .map(|s| s.predict(features))
            .collect();

        let n = preds.len() as f64;
        let mean = preds.iter().sum::<f64>() / n;
        let var = preds
            .iter()
            .map(|p| {
                let d = p - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0).max(1.0);
        crate::math::sqrt(var)
    }
}

impl crate::learner::StreamingLearner for DistributionalSGBT {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let sample = SampleRef::weighted(features, target, weight);
        DistributionalSGBT::train_one(self, &sample);
    }

    fn predict(&self, features: &[f64]) -> f64 {
        DistributionalSGBT::predict(self, features).mu
    }

    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        DistributionalSGBT::reset(self);
    }
}
