//! Distributional SGBT -- outputs Gaussian N(μ, σ²) instead of a point estimate.
//!
//! [`DistributionalSGBT`] supports two scale estimation modes via
//! [`crate::ensemble::config::ScaleMode`]:
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
#[cfg(test)]
mod tests;
mod training;

pub use diagnostics::{DecomposedPrediction, DistributionalTreeDiagnostic, ModelDiagnostics};

use crate::ensemble::config::{SGBTConfig, ScaleMode};
use crate::ensemble::step::BoostingStep;
use crate::sample::{Observation, SampleRef};
use std::collections::VecDeque;

/// Cached packed f32 binary for fast location-only inference.
///
/// Re-exported periodically from the location ensemble. Predictions use
/// contiguous BFS-packed memory for cache-optimal tree traversal.
pub(crate) struct PackedInferenceCache {
    pub bytes: Vec<u8>,
    pub base: f64,
    pub n_features: usize,
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
/// ```
/// use irithyll::SGBTConfig;
/// use irithyll::ensemble::distributional::DistributionalSGBT;
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
    pub(crate) config: SGBTConfig,
    pub(crate) location_steps: Vec<BoostingStep>,
    pub(crate) scale_steps: Vec<BoostingStep>,
    pub(crate) location_base: f64,
    pub(crate) scale_base: f64,
    pub(crate) base_initialized: bool,
    pub(crate) initial_targets: Vec<f64>,
    pub(crate) initial_target_count: usize,
    pub(crate) samples_seen: u64,
    pub(crate) rng_state: u64,
    pub(crate) uncertainty_modulated_lr: bool,
    pub(crate) rolling_sigma_mean: f64,
    pub(crate) scale_mode: ScaleMode,
    pub(crate) ewma_sq_err: f64,
    pub(crate) empirical_sigma_alpha: f64,
    pub(crate) prev_sigma: f64,
    pub(crate) sigma_velocity: f64,
    pub(crate) auto_bandwidths: Vec<f64>,
    pub(crate) last_replacement_sum: u64,
    pub(crate) ensemble_grad_mean: f64,
    pub(crate) ensemble_grad_m2: f64,
    pub(crate) ensemble_grad_count: u64,
    pub(crate) rolling_honest_sigma_mean: f64,
    pub(crate) sigma_ring: VecDeque<f64>,
    pub(crate) mts_replacement_sum: u64,
    pub(crate) packed_cache: Option<PackedInferenceCache>,
    pub(crate) samples_since_refresh: u64,
    pub(crate) packed_refresh_interval: u64,
    pub(crate) prev_contributions: Vec<f64>,
    pub(crate) prev_prev_contributions: Vec<f64>,
    pub(crate) cached_residual_alignment: f64,
    pub(crate) cached_reg_sensitivity: f64,
    pub(crate) cached_depth_sufficiency: f64,
    pub(crate) cached_effective_dof: f64,
    pub(crate) contribution_accuracy: Vec<f64>,
    pub(crate) prune_alpha: f64,
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
            sigma_ring: self.sigma_ring.clone(),
            mts_replacement_sum: self.mts_replacement_sum,
            packed_cache: self.packed_cache.clone(),
            samples_since_refresh: self.samples_since_refresh,
            packed_refresh_interval: self.packed_refresh_interval,
            prev_contributions: self.prev_contributions.clone(),
            prev_prev_contributions: self.prev_prev_contributions.clone(),
            cached_residual_alignment: self.cached_residual_alignment,
            cached_reg_sensitivity: self.cached_reg_sensitivity,
            cached_depth_sufficiency: self.cached_depth_sufficiency,
            cached_effective_dof: self.cached_effective_dof,
            contribution_accuracy: self.contribution_accuracy.clone(),
            prune_alpha: self.prune_alpha,
        }
    }
}

impl std::fmt::Debug for DistributionalSGBT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("DistributionalSGBT");
        s.field("n_steps", &self.location_steps.len())
            .field("samples_seen", &self.samples_seen)
            .field("location_base", &self.location_base)
            .field("scale_mode", &self.scale_mode)
            .field("base_initialized", &self.base_initialized);
        match self.scale_mode {
            ScaleMode::Empirical => {
                s.field("empirical_sigma", &self.ewma_sq_err.sqrt());
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
    /// Create a new distributional SGBT with the given configuration.
    ///
    /// When `scale_mode` is `Empirical` (default), scale trees are still allocated
    /// but never trained — only the EWMA error tracker produces σ.  When
    /// `scale_mode` is `TreeChain`, both location and scale ensembles are active.
    pub fn new(config: SGBTConfig) -> Self {
        let leaf_decay_alpha = config
            .leaf_half_life
            .map(|hl| (-(2.0_f64.ln()) / hl as f64).exp());

        let tree_config = crate::ensemble::config::build_tree_config(&config)
            .leaf_decay_alpha_opt(leaf_decay_alpha);

        let max_tree_samples = config.max_tree_samples;

        // Location ensemble (seed offset: 0)
        let shadow_warmup = config.shadow_warmup.unwrap_or(0);
        let location_steps: Vec<BoostingStep> = (0..config.n_steps)
            .map(|i| {
                let mut tc = tree_config.clone();
                tc.seed = config.seed ^ (i as u64);
                let detector = config.drift_detector.create();
                if shadow_warmup > 0 {
                    BoostingStep::new_with_graduated(tc, detector, max_tree_samples, shadow_warmup)
                } else {
                    BoostingStep::new_with_max_samples(tc, detector, max_tree_samples)
                }
            })
            .collect();

        // Scale ensemble (seed offset: 0xSCALE) -- only trained in TreeChain mode
        let scale_steps: Vec<BoostingStep> = (0..config.n_steps)
            .map(|i| {
                let mut tc = tree_config.clone();
                tc.seed = config.seed ^ (i as u64) ^ 0x0005_CA1E_0000_0000;
                let detector = config.drift_detector.create();
                if shadow_warmup > 0 {
                    BoostingStep::new_with_graduated(tc, detector, max_tree_samples, shadow_warmup)
                } else {
                    BoostingStep::new_with_max_samples(tc, detector, max_tree_samples)
                }
            })
            .collect();

        let seed = config.seed;
        let initial_target_count = config.initial_target_count;
        let uncertainty_modulated_lr = config.uncertainty_modulated_lr;
        let scale_mode = config.scale_mode;
        let empirical_sigma_alpha = config.empirical_sigma_alpha;
        let packed_refresh_interval = config.packed_refresh_interval;
        let n_steps = config.n_steps;
        let prune_alpha = if config.proactive_prune_interval.is_some() {
            let hl = config.prune_half_life.unwrap_or_else(|| {
                if let Some((base_mts, _)) = config.adaptive_mts {
                    base_mts as usize
                } else if let Some(mts) = config.max_tree_samples {
                    mts as usize
                } else {
                    config.grace_period.max(1)
                }
            });
            1.0 - (-2.0 / hl.max(1) as f64).exp()
        } else {
            0.01
        };
        Self {
            config,
            location_steps,
            scale_steps,
            location_base: 0.0,
            scale_base: 0.0,
            base_initialized: false,
            initial_targets: Vec::new(),
            initial_target_count,
            samples_seen: 0,
            rng_state: seed,
            uncertainty_modulated_lr,
            rolling_sigma_mean: 1.0,
            scale_mode,
            ewma_sq_err: 1.0,
            empirical_sigma_alpha,
            prev_sigma: 0.0,
            sigma_velocity: 0.0,
            auto_bandwidths: Vec::new(),
            last_replacement_sum: 0,
            ensemble_grad_mean: 0.0,
            ensemble_grad_m2: 0.0,
            ensemble_grad_count: 0,
            rolling_honest_sigma_mean: 0.0,
            sigma_ring: VecDeque::new(),
            mts_replacement_sum: 0,
            packed_cache: None,
            samples_since_refresh: 0,
            packed_refresh_interval,
            prev_contributions: Vec::new(),
            prev_prev_contributions: Vec::new(),
            cached_residual_alignment: 0.0,
            cached_reg_sensitivity: 0.0,
            cached_depth_sufficiency: 0.0,
            cached_effective_dof: 0.0,
            contribution_accuracy: vec![0.0; n_steps],
            prune_alpha,
        }
    }

    /// Compute honest_sigma from location tree contributions for a feature vector.
    ///
    /// Returns the Bessel-corrected standard deviation of individual tree
    /// contributions (`learning_rate * tree.predict(features)`). This is
    /// O(n_steps) — the same cost as a single prediction pass.
    pub(crate) fn compute_honest_sigma(&self, features: &[f64]) -> f64 {
        let n = self.location_steps.len();
        if n <= 1 {
            return 0.0;
        }
        let lr = self.config.learning_rate;
        let mut sum = 0.0_f64;
        let mut sq_sum = 0.0_f64;
        for step in &self.location_steps {
            let c = lr * step.predict(features);
            sum += c;
            sq_sum += c * c;
        }
        let nf = n as f64;
        let mean_c = sum / nf;
        let var = (sq_sum / nf) - (mean_c * mean_c);
        let var_corrected = var * nf / (nf - 1.0);
        var_corrected.max(0.0).sqrt()
    }

    /// Train on a single observation.
    pub fn train_one(&mut self, sample: &impl Observation) {
        self.samples_seen += 1;
        let target = sample.target();
        let features = sample.features();

        // Initialize base predictions from first few targets
        if !self.base_initialized {
            self.initial_targets.push(target);
            if self.initial_targets.len() >= self.initial_target_count {
                // Location base = mean
                let sum: f64 = self.initial_targets.iter().sum();
                let mean = sum / self.initial_targets.len() as f64;
                self.location_base = mean;

                // Scale base = log(std) -- clamped for stability
                let var: f64 = self
                    .initial_targets
                    .iter()
                    .map(|&y| (y - mean) * (y - mean))
                    .sum::<f64>()
                    / self.initial_targets.len() as f64;
                let initial_std = var.sqrt().max(1e-6);
                self.scale_base = initial_std.ln();

                // Initialize rolling sigma mean and ewma from initial targets std
                self.rolling_sigma_mean = initial_std;
                self.ewma_sq_err = var.max(1e-12);

                // Initialize PD sigma state
                self.prev_sigma = initial_std;
                self.sigma_velocity = 0.0;

                self.base_initialized = true;
                self.initial_targets.clear();
                self.initial_targets.shrink_to_fit();
            }
            return;
        }

        // Adaptive MTS: accumulate sigma_ratio into ring buffer.
        if self.config.adaptive_mts.is_some() {
            let sigma_ratio = if self.rolling_honest_sigma_mean > 1e-12 {
                let honest_sigma = self.compute_honest_sigma(features);
                honest_sigma / self.rolling_honest_sigma_mean
            } else {
                1.0
            };
            let cap = self.config.grace_period;
            if self.sigma_ring.len() >= cap {
                self.sigma_ring.pop_front();
            }
            self.sigma_ring.push_back(sigma_ratio);
        }

        match self.scale_mode {
            ScaleMode::Empirical => self.train_one_empirical(target, features),
            ScaleMode::TreeChain => self.train_one_tree_chain(target, features),
        }

        // Proactive pruning
        if let Some(interval) = self.config.proactive_prune_interval {
            if self.config.accuracy_based_pruning {
                let mut location_pred = self.location_base;
                for step in self.location_steps.iter() {
                    location_pred += self.config.learning_rate * step.predict(features);
                }
                let residual = target - location_pred;
                let sign = residual.signum();
                for (i, step) in self.location_steps.iter().enumerate() {
                    let contribution = self.config.learning_rate * step.predict(features);
                    let alignment = contribution * sign;
                    self.contribution_accuracy[i] = self.prune_alpha * alignment
                        + (1.0 - self.prune_alpha) * self.contribution_accuracy[i];
                }
            }

            if interval > 0 && self.samples_seen % interval == 0 {
                self.check_proactive_prune();
            }
        }

        // End-of-cycle adaptive MTS
        if let Some((base_mts, k)) = self.config.adaptive_mts {
            let current_sum: u64 = self
                .location_steps
                .iter()
                .chain(self.scale_steps.iter())
                .map(|s| s.slot().replacements())
                .sum();
            if current_sum != self.mts_replacement_sum {
                self.mts_replacement_sum = current_sum;
                if !self.sigma_ring.is_empty() {
                    let mean_sigma =
                        self.sigma_ring.iter().sum::<f64>() / self.sigma_ring.len() as f64;
                    let floor = (base_mts as f64 * self.config.adaptive_mts_floor).max(100.0);
                    let effective_mts =
                        (base_mts as f64 / (1.0 + k * mean_sigma)).max(floor) as u64;
                    for step in &mut self.location_steps {
                        step.slot_mut().set_max_tree_samples(Some(effective_mts));
                    }
                    for step in &mut self.scale_steps {
                        step.slot_mut().set_max_tree_samples(Some(effective_mts));
                    }
                }
            }
        }

        // Update diagnostic cache
        self.update_diagnostic_cache(features);

        // Refresh auto-bandwidths
        self.refresh_bandwidths();
    }

    /// Enable or reconfigure the packed inference cache at runtime.
    pub fn enable_packed_cache(&mut self, interval: u64) {
        self.packed_refresh_interval = interval;
        self.samples_since_refresh = 0;
        if interval > 0 && self.base_initialized {
            self.refresh_packed_cache();
        } else if interval == 0 {
            self.packed_cache = None;
        }
    }

    /// Whether the packed inference cache is currently populated.
    #[inline]
    pub fn has_packed_cache(&self) -> bool {
        self.packed_cache.is_some()
    }

    /// Per-feature auto-calibrated bandwidths used by `predict()`.
    pub fn auto_bandwidths(&self) -> &[f64] {
        &self.auto_bandwidths
    }

    /// Reset to initial untrained state.
    pub fn reset(&mut self) {
        for step in &mut self.location_steps {
            step.reset();
        }
        for step in &mut self.scale_steps {
            step.reset();
        }
        self.location_base = 0.0;
        self.scale_base = 0.0;
        self.base_initialized = false;
        self.initial_targets.clear();
        self.samples_seen = 0;
        self.rng_state = self.config.seed;
        self.rolling_sigma_mean = 1.0;
        self.ewma_sq_err = 1.0;
        self.prev_sigma = 0.0;
        self.sigma_velocity = 0.0;
        self.auto_bandwidths.clear();
        self.last_replacement_sum = 0;
        self.ensemble_grad_mean = 0.0;
        self.ensemble_grad_m2 = 0.0;
        self.ensemble_grad_count = 0;
        self.rolling_honest_sigma_mean = 0.0;
        self.packed_cache = None;
        self.samples_since_refresh = 0;
        self.prev_contributions.clear();
        self.prev_prev_contributions.clear();
        self.cached_residual_alignment = 0.0;
        self.cached_reg_sensitivity = 0.0;
        self.cached_depth_sufficiency = 0.0;
        self.cached_effective_dof = 0.0;
        self.contribution_accuracy = vec![0.0; self.location_steps.len()];
    }

    /// Total samples trained.
    #[inline]
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Number of boosting steps (same for location and scale).
    #[inline]
    pub fn n_steps(&self) -> usize {
        self.location_steps.len()
    }

    /// Total trees (location + scale, active + alternates).
    pub fn n_trees(&self) -> usize {
        let loc = self.location_steps.len()
            + self
                .location_steps
                .iter()
                .filter(|s| s.has_alternate())
                .count();
        let scale = self.scale_steps.len()
            + self
                .scale_steps
                .iter()
                .filter(|s| s.has_alternate())
                .count();
        loc + scale
    }

    /// Total leaves across all active trees (location + scale).
    pub fn total_leaves(&self) -> usize {
        let loc: usize = self.location_steps.iter().map(|s| s.n_leaves()).sum();
        let scale: usize = self.scale_steps.iter().map(|s| s.n_leaves()).sum();
        loc + scale
    }

    /// Whether base predictions have been initialized.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.base_initialized
    }

    /// Access the configuration.
    #[inline]
    pub fn config(&self) -> &SGBTConfig {
        &self.config
    }

    /// Learning rate from the model configuration.
    #[inline]
    pub fn learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    /// Set the learning rate for future boosting rounds.
    #[inline]
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.learning_rate = lr;
    }

    /// Set the L2 regularization parameter (lambda) for future boosting rounds.
    #[inline]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.config.lambda = lambda.max(0.0);
    }

    /// Set the maximum tree depth for future replacement trees.
    #[inline]
    pub fn set_max_depth(&mut self, depth: usize) {
        self.config.max_depth = depth.clamp(1, 20);
    }

    /// Adjust the number of boosting steps (trees per chain).
    pub fn set_n_steps(&mut self, n: usize) {
        let n = n.clamp(3, 1000);
        let current = self.location_steps.len();
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
                // Location step
                let mut tc = tree_config.clone();
                tc.seed = self.config.seed ^ (i as u64);
                let detector = self.config.drift_detector.create();
                let step = if shadow_warmup > 0 {
                    BoostingStep::new_with_graduated(tc, detector, mts, shadow_warmup)
                } else {
                    BoostingStep::new_with_max_samples(tc, detector, mts)
                };
                self.location_steps.push(step);

                // Scale step
                let mut tc = tree_config.clone();
                tc.seed = self.config.seed ^ (i as u64) ^ 0x0005_CA1E_0000_0000;
                let detector = self.config.drift_detector.create();
                let step = if shadow_warmup > 0 {
                    BoostingStep::new_with_graduated(tc, detector, mts, shadow_warmup)
                } else {
                    BoostingStep::new_with_max_samples(tc, detector, mts)
                };
                self.scale_steps.push(step);
            }
        } else if n < current {
            self.location_steps.truncate(n);
            self.scale_steps.truncate(n);
        }
        self.contribution_accuracy.resize(n, 0.0);
        self.config.n_steps = n;
    }

    /// Dynamically set the contribution accuracy EWMA half-life.
    pub fn set_prune_half_life(&mut self, hl: usize) {
        self.prune_alpha = 1.0 - (-2.0 / hl.max(1) as f64).exp();
    }

    /// Current rolling σ mean (EWMA of predicted σ).
    #[inline]
    pub fn rolling_sigma_mean(&self) -> f64 {
        self.rolling_sigma_mean
    }

    /// Whether σ-modulated learning rate is active.
    #[inline]
    pub fn is_uncertainty_modulated(&self) -> bool {
        self.uncertainty_modulated_lr
    }

    /// Current rolling honest_sigma mean (EWMA of tree contribution std dev).
    #[inline]
    pub fn rolling_honest_sigma_mean(&self) -> f64 {
        self.rolling_honest_sigma_mean
    }

    /// Base prediction for the location (mean) ensemble.
    #[inline]
    pub fn location_base(&self) -> f64 {
        self.location_base
    }

    /// Access the location boosting steps (for export/inspection).
    pub fn location_steps(&self) -> &[BoostingStep] {
        &self.location_steps
    }

    /// Manually trigger a proactive prune check on the location chain.
    pub fn check_proactive_prune(&mut self) -> bool {
        if self.location_steps.len() <= 1 {
            return false;
        }
        if self.config.accuracy_based_pruning {
            let grace_period = self.config.grace_period as u64;
            let worst = self
                .location_steps
                .iter()
                .enumerate()
                .zip(self.contribution_accuracy.iter())
                .filter(|((_, step), _)| step.slot().n_samples_seen() >= grace_period)
                .min_by(|((_, _), a), ((_, _), b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(((worst_idx, _), &worst_acc)) = worst {
                if worst_acc < 0.0 {
                    self.location_steps[worst_idx].slot_mut().replace_active();
                    self.contribution_accuracy[worst_idx] = 0.0;
                    return true;
                }
            }
            false
        } else {
            let worst_idx = self
                .location_steps
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
            self.location_steps[worst_idx].slot_mut().replace_active();
            true
        }
    }

    /// Total tree replacements across all boosting steps (location + scale).
    pub fn total_replacements(&self) -> u64 {
        self.location_steps
            .iter()
            .chain(self.scale_steps.iter())
            .map(|s| s.slot().replacements())
            .sum()
    }

    /// Snapshots the model into a serializable state.
    #[cfg(any(feature = "serde-json", feature = "serde-bincode"))]
    pub fn to_distributional_state(&self) -> crate::serde_support::DistributionalModelState {
        use crate::serde_support::StepSnapshot;

        let snapshot_step = |s: &BoostingStep| {
            use super::snapshot_tree;
            let slot = s.slot();
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
        };

        crate::serde_support::DistributionalModelState {
            config: self.config.clone(),
            location_steps: self.location_steps.iter().map(snapshot_step).collect(),
            scale_steps: self.scale_steps.iter().map(snapshot_step).collect(),
            location_base: self.location_base,
            scale_base: self.scale_base,
            base_initialized: self.base_initialized,
            initial_targets: self.initial_targets.clone(),
            initial_target_count: self.initial_target_count,
            samples_seen: self.samples_seen,
            rng_state: self.rng_state,
            uncertainty_modulated_lr: self.uncertainty_modulated_lr,
            rolling_sigma_mean: self.rolling_sigma_mean,
            ewma_sq_err: self.ewma_sq_err,
            rolling_honest_sigma_mean: self.rolling_honest_sigma_mean,
        }
    }

    /// Reconstruct a [`DistributionalSGBT`] from a serialized state.
    #[cfg(any(feature = "serde-json", feature = "serde-bincode"))]
    pub fn from_distributional_state(
        state: crate::serde_support::DistributionalModelState,
    ) -> Self {
        use super::rebuild_tree;
        use crate::ensemble::replacement::TreeSlot;
        use crate::serde_support::StepSnapshot;

        let leaf_decay_alpha = state
            .config
            .leaf_half_life
            .map(|hl| (-(2.0_f64.ln()) / hl as f64).exp());
        let max_tree_samples = state.config.max_tree_samples;

        let base_tree_config = crate::ensemble::config::build_tree_config(&state.config)
            .leaf_decay_alpha_opt(leaf_decay_alpha);

        let rebuild_steps = |snaps: &[StepSnapshot], seed_xor: u64| -> Vec<BoostingStep> {
            snaps
                .iter()
                .enumerate()
                .map(|(i, snap)| {
                    let tc = base_tree_config
                        .clone()
                        .seed(state.config.seed ^ (i as u64) ^ seed_xor);

                    let active = rebuild_tree(&snap.tree, tc.clone());
                    let alternate = snap
                        .alternate_tree
                        .as_ref()
                        .map(|s| rebuild_tree(s, tc.clone()));

                    let mut detector = state.config.drift_detector.create();
                    if let Some(ref ds) = snap.drift_state {
                        detector.restore_state(ds);
                    }
                    let mut slot =
                        TreeSlot::from_trees(active, alternate, tc, detector, max_tree_samples);
                    if let Some(ref ads) = snap.alt_drift_state {
                        if let Some(alt_det) = slot.alt_detector_mut() {
                            alt_det.restore_state(ads);
                        }
                    }
                    BoostingStep::from_slot(slot)
                })
                .collect()
        };

        let location_steps = rebuild_steps(&state.location_steps, 0);
        let scale_steps = rebuild_steps(&state.scale_steps, 0x0005_CA1E_0000_0000);

        let scale_mode = state.config.scale_mode;
        let empirical_sigma_alpha = state.config.empirical_sigma_alpha;
        let packed_refresh_interval = state.config.packed_refresh_interval;
        let n_location_steps = location_steps.len();
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
            location_steps,
            scale_steps,
            location_base: state.location_base,
            scale_base: state.scale_base,
            base_initialized: state.base_initialized,
            initial_targets: state.initial_targets,
            initial_target_count: state.initial_target_count,
            samples_seen: state.samples_seen,
            rng_state: state.rng_state,
            uncertainty_modulated_lr: state.uncertainty_modulated_lr,
            rolling_sigma_mean: state.rolling_sigma_mean,
            scale_mode,
            ewma_sq_err: state.ewma_sq_err,
            empirical_sigma_alpha,
            prev_sigma: 0.0,
            sigma_velocity: 0.0,
            auto_bandwidths: Vec::new(),
            last_replacement_sum: 0,
            ensemble_grad_mean: 0.0,
            ensemble_grad_m2: 0.0,
            ensemble_grad_count: 0,
            rolling_honest_sigma_mean: state.rolling_honest_sigma_mean,
            sigma_ring: VecDeque::new(),
            mts_replacement_sum: 0,
            packed_cache: None,
            samples_since_refresh: 0,
            packed_refresh_interval,
            prev_contributions: Vec::new(),
            prev_prev_contributions: Vec::new(),
            cached_residual_alignment: 0.0,
            cached_reg_sensitivity: 0.0,
            cached_depth_sufficiency: 0.0,
            cached_effective_dof: 0.0,
            contribution_accuracy: vec![0.0; n_location_steps],
            prune_alpha,
        }
    }
}

// StreamingLearner impl
use crate::learner::StreamingLearner;

impl StreamingLearner for DistributionalSGBT {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let sample = SampleRef::weighted(features, target, weight);
        DistributionalSGBT::train_one(self, &sample);
    }

    fn predict(&self, features: &[f64]) -> f64 {
        self.predict(features).mu
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples_seen()
    }

    fn reset(&mut self) {
        DistributionalSGBT::reset(self);
    }

    #[allow(deprecated)]
    fn diagnostics_array(&self) -> [f64; 5] {
        <Self as crate::learner::Tunable>::diagnostics_array(self)
    }

    #[allow(deprecated)]
    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64) {
        <Self as crate::learner::Tunable>::adjust_config(self, lr_multiplier, lambda_delta);
    }

    #[allow(deprecated)]
    fn apply_structural_change(&mut self, depth_delta: i32, steps_delta: i32) {
        <Self as crate::learner::Structural>::apply_structural_change(
            self,
            depth_delta,
            steps_delta,
        );
    }

    #[allow(deprecated)]
    fn replacement_count(&self) -> u64 {
        <Self as crate::learner::Structural>::replacement_count(self)
    }
}

impl crate::learner::Tunable for DistributionalSGBT {
    fn diagnostics_array(&self) -> [f64; 5] {
        [
            self.cached_residual_alignment,
            self.cached_reg_sensitivity,
            self.cached_depth_sufficiency,
            self.cached_effective_dof,
            self.rolling_honest_sigma_mean(),
        ]
    }

    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64) {
        self.config.learning_rate = (self.config.learning_rate * lr_multiplier).clamp(1e-4, 1.0);
        self.config.lambda = (self.config.lambda + lambda_delta).max(0.0);
    }
}

impl crate::learner::Structural for DistributionalSGBT {
    fn apply_structural_change(&mut self, _depth_delta: i32, _steps_delta: i32) {
        // Not applicable for distributional model.
    }

    fn replacement_count(&self) -> u64 {
        self.total_replacements()
    }
}

impl crate::automl::DiagnosticSource for DistributionalSGBT {
    fn config_diagnostics(&self) -> Option<crate::automl::auto_builder::ConfigDiagnostics> {
        Some(crate::automl::auto_builder::ConfigDiagnostics {
            residual_alignment: self.cached_residual_alignment,
            regularization_sensitivity: self.cached_reg_sensitivity,
            depth_sufficiency: self.cached_depth_sufficiency,
            effective_dof: self.cached_effective_dof,
            uncertainty: self.rolling_honest_sigma_mean(),
        })
    }
}
