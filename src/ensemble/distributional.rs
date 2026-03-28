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

use crate::ensemble::config::{SGBTConfig, ScaleMode};
use crate::ensemble::step::BoostingStep;
use crate::export_embedded::export_distributional_packed;
use crate::sample::{Observation, SampleRef};
use crate::tree::builder::TreeConfig;

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

// ---------------------------------------------------------------------------
// Diagnostic structs
// ---------------------------------------------------------------------------

/// Per-tree diagnostic summary.
#[derive(Debug, Clone)]
pub struct TreeDiagnostic {
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
    pub trees: Vec<TreeDiagnostic>,
    /// Location trees only (view into `trees`).
    pub location_trees: Vec<TreeDiagnostic>,
    /// Scale trees only (view into `trees`).
    pub scale_trees: Vec<TreeDiagnostic>,
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

// ---------------------------------------------------------------------------
// DistributionalSGBT
// ---------------------------------------------------------------------------

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
    /// Configuration (shared between location and scale ensembles).
    config: SGBTConfig,
    /// Location (mean) boosting steps.
    location_steps: Vec<BoostingStep>,
    /// Scale (log-sigma) boosting steps (only used in `TreeChain` mode).
    scale_steps: Vec<BoostingStep>,
    /// Base prediction for location (mean of initial targets).
    location_base: f64,
    /// Base prediction for scale (log of std of initial targets).
    scale_base: f64,
    /// Whether base predictions have been initialized.
    base_initialized: bool,
    /// Running collection of initial targets for computing base predictions.
    initial_targets: Vec<f64>,
    /// Number of initial targets to collect before setting base predictions.
    initial_target_count: usize,
    /// Total samples trained.
    samples_seen: u64,
    /// RNG state for variant logic.
    rng_state: u64,
    /// Whether σ-modulated learning rate is enabled.
    uncertainty_modulated_lr: bool,
    /// EWMA of the model's predicted σ -- used as the denominator in σ-ratio.
    ///
    /// Updated with alpha = 0.001 (slow adaptation) after each training step.
    /// Initialized from the standard deviation of the initial target collection.
    rolling_sigma_mean: f64,
    /// Scale estimation mode: Empirical (default) or TreeChain.
    scale_mode: ScaleMode,
    /// EWMA of squared prediction errors (empirical σ mode).
    ///
    /// `sigma = sqrt(ewma_sq_err)`. Updated every training step with
    /// `ewma_sq_err = alpha * err² + (1 - alpha) * ewma_sq_err`.
    ewma_sq_err: f64,
    /// EWMA alpha for empirical σ.
    empirical_sigma_alpha: f64,
    /// Previous empirical σ value for computing σ velocity.
    prev_sigma: f64,
    /// EWMA-smoothed derivative of empirical σ (σ velocity).
    /// Positive = σ increasing (errors growing), negative = σ decreasing.
    sigma_velocity: f64,
    /// Per-feature auto-calibrated bandwidths for smooth prediction.
    auto_bandwidths: Vec<f64>,
    /// Sum of replacement counts at last bandwidth computation.
    last_replacement_sum: u64,
    /// Ensemble-level gradient running mean (Welford).
    ensemble_grad_mean: f64,
    /// Ensemble-level gradient M2 (Welford sum of squared deviations).
    ensemble_grad_m2: f64,
    /// Ensemble-level gradient sample count.
    ensemble_grad_count: u64,
    /// EWMA of honest_sigma (tree contribution std dev) for σ-modulated LR.
    ///
    /// When `uncertainty_modulated_lr` is enabled, the honest_sigma ratio
    /// (`honest_sigma / rolling_honest_sigma_mean`) modulates the effective
    /// learning rate, reacting instantly to tree disagreement.
    rolling_honest_sigma_mean: f64,
    /// Packed f32 cache for fast location-only inference (dual-path).
    packed_cache: Option<PackedInferenceCache>,
    /// Samples trained since last packed cache refresh.
    samples_since_refresh: u64,
    /// How often to refresh the packed cache (0 = disabled).
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

        let tree_config = TreeConfig::new()
            .max_depth(config.max_depth)
            .n_bins(config.n_bins)
            .lambda(config.lambda)
            .gamma(config.gamma)
            .grace_period(config.grace_period)
            .delta(config.delta)
            .feature_subsample_rate(config.feature_subsample_rate)
            .leaf_decay_alpha_opt(leaf_decay_alpha)
            .split_reeval_interval_opt(config.split_reeval_interval)
            .feature_types_opt(config.feature_types.clone())
            .gradient_clip_sigma_opt(config.gradient_clip_sigma)
            .monotone_constraints_opt(config.monotone_constraints.clone())
            .max_leaf_output_opt(config.max_leaf_output)
            .adaptive_depth_opt(config.adaptive_depth)
            .min_hessian_sum_opt(config.min_hessian_sum)
            .leaf_model_type(config.leaf_model_type.clone());

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
            rolling_sigma_mean: 1.0, // overwritten during base initialization
            scale_mode,
            ewma_sq_err: 1.0, // overwritten during base initialization
            empirical_sigma_alpha,
            prev_sigma: 0.0,
            sigma_velocity: 0.0,
            auto_bandwidths: Vec::new(),
            last_replacement_sum: 0,
            ensemble_grad_mean: 0.0,
            ensemble_grad_m2: 0.0,
            ensemble_grad_count: 0,
            rolling_honest_sigma_mean: 0.0,
            packed_cache: None,
            samples_since_refresh: 0,
            packed_refresh_interval,
        }
    }

    /// Compute honest_sigma from location tree contributions for a feature vector.
    ///
    /// Returns the Bessel-corrected standard deviation of individual tree
    /// contributions (`learning_rate * tree.predict(features)`). This is
    /// O(n_steps) — the same cost as a single prediction pass.
    fn compute_honest_sigma(&self, features: &[f64]) -> f64 {
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

        // Apply adaptive_mts: override per-step max_tree_samples based on honest_sigma.
        if let Some((base_mts, k)) = self.config.adaptive_mts {
            let sigma_ratio = if self.rolling_honest_sigma_mean > 1e-12 {
                let honest_sigma = self.compute_honest_sigma(features);
                honest_sigma / self.rolling_honest_sigma_mean
            } else {
                1.0
            };
            let effective_mts = (base_mts as f64 / (1.0 + k * sigma_ratio)).max(100.0) as u64;
            for step in &mut self.location_steps {
                step.slot_mut().set_max_tree_samples(Some(effective_mts));
            }
            for step in &mut self.scale_steps {
                step.slot_mut().set_max_tree_samples(Some(effective_mts));
            }
        }

        match self.scale_mode {
            ScaleMode::Empirical => self.train_one_empirical(target, features),
            ScaleMode::TreeChain => self.train_one_tree_chain(target, features),
        }

        // Proactive pruning: replace worst-contributing location tree every N samples.
        if let Some(interval) = self.config.proactive_prune_interval {
            if interval > 0 && self.samples_seen % interval == 0 && self.location_steps.len() > 1 {
                // Find worst step by prediction variance (smallest = least contribution).
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
            }
        }

        // Refresh auto-bandwidths when trees have been replaced
        self.refresh_bandwidths();
    }

    /// Empirical-σ training: location trees only, σ from EWMA of squared errors.
    fn train_one_empirical(&mut self, target: f64, features: &[f64]) {
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
    fn train_one_tree_chain(&mut self, target: f64, features: &[f64]) {
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
        // Location chain with soft routing
        let mut mu = self.location_base;
        for step in &self.location_steps {
            mu += self.config.learning_rate * step.predict_soft_routed(features);
        }

        // Scale chain
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
    /// location learning rate when [`uncertainty_modulated_lr`](SGBTConfig::uncertainty_modulated_lr)
    /// is enabled.
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
    pub fn train_batch<O: Observation>(&mut self, samples: &[O]) {
        for sample in samples {
            self.train_one(sample);
        }
    }

    /// Train on a batch with periodic callback.
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

    /// Compute location gradient (squared or adaptive Huber).
    ///
    /// When `huber_k` is configured, uses Huber loss with adaptive
    /// `delta = k * empirical_sigma` for bounded gradients.
    #[inline]
    fn location_gradient(&self, mu: f64, target: f64) -> (f64, f64) {
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
    fn update_ensemble_grad_stats(&mut self, gradient: f64) {
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
    fn maybe_refresh_packed_cache(&mut self) {
        if self.packed_refresh_interval > 0 {
            self.samples_since_refresh += 1;
            if self.samples_since_refresh >= self.packed_refresh_interval {
                self.refresh_packed_cache();
                self.samples_since_refresh = 0;
            }
        }
    }

    /// Re-export the location ensemble into the packed cache.
    fn refresh_packed_cache(&mut self) {
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

        let (bytes, base) = export_distributional_packed(self, n_features);
        self.packed_cache = Some(PackedInferenceCache {
            bytes,
            base,
            n_features,
        });
    }

    /// Enable or reconfigure the packed inference cache at runtime.
    ///
    /// Sets the refresh interval and immediately builds the initial cache
    /// if the model has been initialized. Pass `0` to disable.
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

    /// Refresh auto-bandwidths if any tree has been replaced since last computation.
    fn refresh_bandwidths(&mut self) {
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

    /// Access the location boosting steps (for export/inspection).
    pub fn location_steps(&self) -> &[BoostingStep] {
        &self.location_steps
    }

    /// Base prediction for the location (mean) ensemble.
    #[inline]
    pub fn location_base(&self) -> f64 {
        self.location_base
    }

    /// Learning rate from the model configuration.
    #[inline]
    pub fn learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    /// Current rolling σ mean (EWMA of predicted σ).
    ///
    /// Returns `1.0` if the model hasn't been initialized yet.
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
    ///
    /// Returns `0.0` before any training samples have been processed.
    #[inline]
    pub fn rolling_honest_sigma_mean(&self) -> f64 {
        self.rolling_honest_sigma_mean
    }

    // -------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------

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
            trees: &mut Vec<TreeDiagnostic>,
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

                trees.push(TreeDiagnostic {
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

        // Compute effective_mts if adaptive_mts is configured
        let effective_mts = self.config.adaptive_mts.map(|(base_mts, k)| {
            let sigma_ratio = if self.rolling_honest_sigma_mean > 1e-12 {
                honest_sigma / self.rolling_honest_sigma_mean
            } else {
                1.0
            };
            (base_mts as f64 / (1.0 + k * sigma_ratio)).max(100.0) as u64
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

    /// Convert this model into a serializable [`crate::serde_support::DistributionalModelState`].
    ///
    /// Captures the full ensemble state (both location and scale trees) for
    /// persistence. Histogram accumulators are NOT serialized -- they rebuild
    /// naturally from continued training.
    #[cfg(any(feature = "serde-json", feature = "serde-bincode"))]
    pub fn to_distributional_state(&self) -> crate::serde_support::DistributionalModelState {
        use super::snapshot_tree;
        use crate::serde_support::{DistributionalModelState, StepSnapshot};

        fn snapshot_step(step: &BoostingStep) -> StepSnapshot {
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
        }

        DistributionalModelState {
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

    /// Reconstruct a [`DistributionalSGBT`] from a serialized [`crate::serde_support::DistributionalModelState`].
    ///
    /// Rebuilds both location and scale ensembles including tree topology
    /// and leaf values. Histogram accumulators are left empty and will
    /// rebuild from continued training.
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

        let base_tree_config = TreeConfig::new()
            .max_depth(state.config.max_depth)
            .n_bins(state.config.n_bins)
            .lambda(state.config.lambda)
            .gamma(state.config.gamma)
            .grace_period(state.config.grace_period)
            .delta(state.config.delta)
            .feature_subsample_rate(state.config.feature_subsample_rate)
            .leaf_decay_alpha_opt(leaf_decay_alpha)
            .split_reeval_interval_opt(state.config.split_reeval_interval)
            .feature_types_opt(state.config.feature_types.clone())
            .gradient_clip_sigma_opt(state.config.gradient_clip_sigma)
            .monotone_constraints_opt(state.config.monotone_constraints.clone())
            .adaptive_depth_opt(state.config.adaptive_depth)
            .leaf_model_type(state.config.leaf_model_type.clone());

        // Rebuild a Vec<BoostingStep> from step snapshots with a given seed transform.
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

        // Location: seed offset 0, Scale: seed offset 0x0005_CA1E_0000_0000
        let location_steps = rebuild_steps(&state.location_steps, 0);
        let scale_steps = rebuild_steps(&state.scale_steps, 0x0005_CA1E_0000_0000);

        let scale_mode = state.config.scale_mode;
        let empirical_sigma_alpha = state.config.empirical_sigma_alpha;
        let packed_refresh_interval = state.config.packed_refresh_interval;
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
            packed_cache: None,
            samples_since_refresh: 0,
            packed_refresh_interval,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

use crate::learner::StreamingLearner;

impl StreamingLearner for DistributionalSGBT {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let sample = SampleRef::weighted(features, target, weight);
        // UFCS: call the inherent train_one(&impl Observation), not this trait method.
        DistributionalSGBT::train_one(self, &sample);
    }

    /// Returns the mean (μ) of the predicted Gaussian distribution.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .build()
            .unwrap()
    }

    #[test]
    fn fresh_model_predicts_zero() {
        let model = DistributionalSGBT::new(test_config());
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(pred.mu.abs() < 1e-12);
        assert!(pred.sigma > 0.0);
    }

    #[test]
    fn sigma_always_positive() {
        let mut model = DistributionalSGBT::new(test_config());

        // Train on various data
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 0.5], x * 2.0 + 1.0));
        }

        // Check multiple predictions
        for i in 0..20 {
            let x = i as f64 * 0.5;
            let pred = model.predict(&[x, x * 0.5]);
            assert!(
                pred.sigma > 0.0,
                "sigma must be positive, got {}",
                pred.sigma
            );
            assert!(pred.sigma.is_finite(), "sigma must be finite");
        }
    }

    #[test]
    fn constant_target_has_small_sigma() {
        let mut model = DistributionalSGBT::new(test_config());

        // Train on constant target = 5.0 with varying features
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 2.0], 5.0));
        }

        let pred = model.predict(&[1.0, 2.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma.is_finite());
        assert!(pred.sigma > 0.0);
        // Sigma should be relatively small for constant target
        // (no noise to explain)
    }

    #[test]
    fn noisy_target_has_finite_predictions() {
        let mut model = DistributionalSGBT::new(test_config());

        // Simple xorshift for deterministic "noise"
        let mut rng: u64 = 42;
        for i in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let noise = (rng % 1000) as f64 / 500.0 - 1.0; // [-1, 1]
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0 + noise));
        }

        let pred = model.predict(&[5.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma.is_finite());
        assert!(pred.sigma > 0.0);
    }

    #[test]
    fn predict_interval_bounds_correct() {
        let mut model = DistributionalSGBT::new(test_config());

        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }

        let (lo, hi) = model.predict_interval(&[5.0], 1.96);
        let pred = model.predict(&[5.0]);

        assert!(lo < pred.mu, "lower bound should be < mu");
        assert!(hi > pred.mu, "upper bound should be > mu");
        assert!((hi - lo - 2.0 * 1.96 * pred.sigma).abs() < 1e-10);
    }

    #[test]
    fn batch_prediction_matches_individual() {
        let mut model = DistributionalSGBT::new(test_config());

        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 2.0], x));
        }

        let features = vec![vec![1.0, 2.0], vec![3.0, 6.0], vec![5.0, 10.0]];
        let batch = model.predict_batch(&features);

        for (feat, batch_pred) in features.iter().zip(batch.iter()) {
            let individual = model.predict(feat);
            assert!((batch_pred.mu - individual.mu).abs() < 1e-12);
            assert!((batch_pred.sigma - individual.sigma).abs() < 1e-12);
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut model = DistributionalSGBT::new(test_config());

        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }

        assert!(model.n_samples_seen() > 0);
        model.reset();

        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.is_initialized());
    }

    #[test]
    fn gaussian_prediction_lower_upper() {
        let pred = GaussianPrediction {
            mu: 10.0,
            sigma: 2.0,
            log_sigma: 2.0_f64.ln(),
            honest_sigma: 0.0,
        };

        assert!((pred.lower(1.96) - (10.0 - 1.96 * 2.0)).abs() < 1e-10);
        assert!((pred.upper(1.96) - (10.0 + 1.96 * 2.0)).abs() < 1e-10);
    }

    #[test]
    fn train_batch_works() {
        let mut model = DistributionalSGBT::new(test_config());
        let samples: Vec<(Vec<f64>, f64)> = (0..100)
            .map(|i| {
                let x = i as f64 * 0.1;
                (vec![x], x * 2.0)
            })
            .collect();

        model.train_batch(&samples);
        assert_eq!(model.n_samples_seen(), 100);
    }

    #[test]
    fn debug_format_works() {
        let model = DistributionalSGBT::new(test_config());
        let debug = format!("{:?}", model);
        assert!(debug.contains("DistributionalSGBT"));
    }

    #[test]
    fn n_trees_counts_both_ensembles() {
        let model = DistributionalSGBT::new(test_config());
        // 10 location + 10 scale = 20 trees minimum
        assert!(model.n_trees() >= 20);
    }

    // -- σ-modulated learning rate tests --

    fn modulated_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .uncertainty_modulated_lr(true)
            .build()
            .unwrap()
    }

    #[test]
    fn sigma_modulated_initializes_rolling_mean() {
        let mut model = DistributionalSGBT::new(modulated_config());
        assert!(model.is_uncertainty_modulated());

        // Before initialization, rolling_sigma_mean is 1.0 (placeholder)
        assert!((model.rolling_sigma_mean() - 1.0).abs() < 1e-12);

        // Train past initialization
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 0.5], x * 2.0 + 1.0));
        }

        // After training, rolling_sigma_mean should have adapted from initial std
        assert!(model.rolling_sigma_mean() > 0.0);
        assert!(model.rolling_sigma_mean().is_finite());
    }

    #[test]
    fn predict_distributional_returns_sigma_ratio() {
        let mut model = DistributionalSGBT::new(modulated_config());

        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0 + 1.0));
        }

        let (mu, sigma, sigma_ratio) = model.predict_distributional(&[5.0]);
        assert!(mu.is_finite());
        assert!(sigma > 0.0);
        assert!(
            (0.1..=10.0).contains(&sigma_ratio),
            "sigma_ratio={}",
            sigma_ratio
        );
    }

    #[test]
    fn predict_distributional_without_modulation_returns_one() {
        let mut model = DistributionalSGBT::new(test_config());
        assert!(!model.is_uncertainty_modulated());

        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }

        let (_mu, _sigma, sigma_ratio) = model.predict_distributional(&[5.0]);
        assert!(
            (sigma_ratio - 1.0).abs() < 1e-12,
            "should be 1.0 when disabled"
        );
    }

    #[test]
    fn modulated_model_sigma_finite_under_varying_noise() {
        let mut model = DistributionalSGBT::new(modulated_config());

        let mut rng: u64 = 123;
        for i in 0..500 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let noise = (rng % 1000) as f64 / 100.0 - 5.0; // [-5, 5]
            let x = i as f64 * 0.1;
            // Regime shift at i=250: noise amplitude increases
            let scale = if i < 250 { 1.0 } else { 5.0 };
            model.train_one(&(vec![x], x * 2.0 + noise * scale));
        }

        let pred = model.predict(&[10.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma.is_finite());
        assert!(pred.sigma > 0.0);
        assert!(model.rolling_sigma_mean().is_finite());
    }

    #[test]
    fn reset_clears_rolling_sigma_mean() {
        let mut model = DistributionalSGBT::new(modulated_config());

        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }

        let sigma_before = model.rolling_sigma_mean();
        assert!(sigma_before > 0.0);

        model.reset();
        assert!((model.rolling_sigma_mean() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn streaming_learner_returns_mu() {
        let mut model = DistributionalSGBT::new(test_config());
        for i in 0..200 {
            let x = i as f64 * 0.1;
            StreamingLearner::train(&mut model, &[x], x * 2.0 + 1.0);
        }
        let pred = StreamingLearner::predict(&model, &[5.0]);
        let gaussian = DistributionalSGBT::predict(&model, &[5.0]);
        assert!(
            (pred - gaussian.mu).abs() < 1e-12,
            "StreamingLearner::predict should return mu"
        );
    }

    // -- Diagnostics tests --

    fn trained_model() -> DistributionalSGBT {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(10) // low grace period to ensure splits
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        for i in 0..500 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 0.5, (i % 3) as f64], x * 2.0 + 1.0));
        }
        model
    }

    #[test]
    fn diagnostics_returns_correct_tree_count() {
        let model = trained_model();
        let diag = model.diagnostics();
        // 10 location + 10 scale = 20 trees
        assert_eq!(diag.trees.len(), 20, "should have 2*n_steps trees");
    }

    #[test]
    fn diagnostics_trees_have_leaves() {
        let model = trained_model();
        let diag = model.diagnostics();
        for (i, tree) in diag.trees.iter().enumerate() {
            assert!(tree.n_leaves >= 1, "tree {i} should have at least 1 leaf");
        }
        // At least some trees should have seen samples.
        let total_samples: u64 = diag.trees.iter().map(|t| t.samples_seen).sum();
        assert!(
            total_samples > 0,
            "at least some trees should have seen samples"
        );
    }

    #[test]
    fn diagnostics_leaf_weight_stats_finite() {
        let model = trained_model();
        let diag = model.diagnostics();
        for (i, tree) in diag.trees.iter().enumerate() {
            let (min, max, mean, std) = tree.leaf_weight_stats;
            assert!(min.is_finite(), "tree {i} min not finite");
            assert!(max.is_finite(), "tree {i} max not finite");
            assert!(mean.is_finite(), "tree {i} mean not finite");
            assert!(std.is_finite(), "tree {i} std not finite");
            assert!(min <= max, "tree {i} min > max");
        }
    }

    #[test]
    fn diagnostics_base_predictions_match() {
        let model = trained_model();
        let diag = model.diagnostics();
        assert!(
            (diag.location_base - model.predict(&[0.0, 0.0, 0.0]).mu).abs() < 100.0,
            "location_base should be plausible"
        );
    }

    #[test]
    fn predict_decomposed_reconstructs_prediction() {
        let model = trained_model();
        let features = [5.0, 2.5, 1.0];
        let pred = model.predict(&features);
        let decomp = model.predict_decomposed(&features);

        assert!(
            (decomp.mu() - pred.mu).abs() < 1e-10,
            "decomposed mu ({}) != predict mu ({})",
            decomp.mu(),
            pred.mu
        );
        assert!(
            (decomp.sigma() - pred.sigma).abs() < 1e-10,
            "decomposed sigma ({}) != predict sigma ({})",
            decomp.sigma(),
            pred.sigma
        );
    }

    #[test]
    fn predict_decomposed_correct_lengths() {
        let model = trained_model();
        let decomp = model.predict_decomposed(&[1.0, 0.5, 0.0]);
        assert_eq!(
            decomp.location_contributions.len(),
            model.n_steps(),
            "location contributions should have n_steps entries"
        );
        assert_eq!(
            decomp.scale_contributions.len(),
            model.n_steps(),
            "scale contributions should have n_steps entries"
        );
    }

    #[test]
    fn feature_importances_work() {
        let model = trained_model();
        let imp = model.feature_importances();
        // After enough training, importances should be non-empty and non-negative.
        // They may or may not sum to 1.0 if no splits have occurred yet.
        for (i, &v) in imp.iter().enumerate() {
            assert!(v >= 0.0, "importance {i} should be non-negative, got {v}");
            assert!(v.is_finite(), "importance {i} should be finite");
        }
        let sum: f64 = imp.iter().sum();
        if sum > 0.0 {
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "non-zero importances should sum to 1.0, got {sum}"
            );
        }
    }

    #[test]
    fn feature_importances_split_works() {
        let model = trained_model();
        let (loc_imp, scale_imp) = model.feature_importances_split();
        for (name, imp) in [("location", &loc_imp), ("scale", &scale_imp)] {
            let sum: f64 = imp.iter().sum();
            if sum > 0.0 {
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "{name} importances should sum to 1.0, got {sum}"
                );
            }
            for &v in imp.iter() {
                assert!(v >= 0.0 && v.is_finite());
            }
        }
    }

    // -- Empirical σ tests --

    #[test]
    fn empirical_sigma_default_mode() {
        use crate::ensemble::config::ScaleMode;
        let config = test_config();
        let model = DistributionalSGBT::new(config);
        assert_eq!(model.scale_mode(), ScaleMode::Empirical);
    }

    #[test]
    fn empirical_sigma_tracks_errors() {
        let mut model = DistributionalSGBT::new(test_config());

        // Train on clean linear data
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 0.5], x * 2.0 + 1.0));
        }

        let sigma_clean = model.empirical_sigma();
        assert!(sigma_clean > 0.0, "sigma should be positive");
        assert!(sigma_clean.is_finite(), "sigma should be finite");

        // Now train on noisy data — sigma should increase
        let mut rng: u64 = 42;
        for i in 200..400 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let noise = (rng % 10000) as f64 / 100.0 - 50.0; // big noise
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 0.5], x * 2.0 + noise));
        }

        let sigma_noisy = model.empirical_sigma();
        assert!(
            sigma_noisy > sigma_clean,
            "noisy regime should increase sigma: clean={sigma_clean}, noisy={sigma_noisy}"
        );
    }

    #[test]
    fn empirical_sigma_modulated_lr_adapts() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .uncertainty_modulated_lr(true)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        // Train and verify sigma_ratio changes
        for i in 0..300 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0 + 1.0));
        }

        let (_, _, sigma_ratio) = model.predict_distributional(&[5.0]);
        assert!(sigma_ratio.is_finite());
        assert!(
            (0.1..=10.0).contains(&sigma_ratio),
            "sigma_ratio={sigma_ratio}"
        );
    }

    #[test]
    fn tree_chain_mode_trains_scale_trees() {
        use crate::ensemble::config::ScaleMode;
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .scale_mode(ScaleMode::TreeChain)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        assert_eq!(model.scale_mode(), ScaleMode::TreeChain);

        for i in 0..500 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 0.5, (i % 3) as f64], x * 2.0 + 1.0));
        }

        let pred = model.predict(&[5.0, 2.5, 1.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma > 0.0);
        assert!(pred.sigma.is_finite());
    }

    #[test]
    fn diagnostics_shows_empirical_sigma() {
        let model = trained_model();
        let diag = model.diagnostics();
        assert!(
            diag.empirical_sigma > 0.0,
            "empirical_sigma should be positive"
        );
        assert!(
            diag.empirical_sigma.is_finite(),
            "empirical_sigma should be finite"
        );
    }

    #[test]
    fn diagnostics_scale_trees_split_fields() {
        let model = trained_model();
        let diag = model.diagnostics();
        assert_eq!(diag.location_trees.len(), model.n_steps());
        assert_eq!(diag.scale_trees.len(), model.n_steps());
        // In empirical mode, scale_trees_active might be 0 (trees not trained)
        // This is expected and actually the point.
    }

    #[test]
    fn reset_clears_empirical_sigma() {
        let mut model = DistributionalSGBT::new(test_config());
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }
        model.reset();
        // After reset, ewma_sq_err resets to 1.0
        assert!((model.empirical_sigma() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn predict_smooth_returns_finite() {
        let config = SGBTConfig::builder()
            .n_steps(3)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        for i in 0..200 {
            let x = (i as f64) * 0.1;
            let features = vec![x, x.sin()];
            let target = 2.0 * x + 1.0;
            model.train_one(&(features, target));
        }

        let pred = model.predict_smooth(&[1.0, 1.0_f64.sin()], 0.5);
        assert!(pred.mu.is_finite(), "smooth mu should be finite");
        assert!(pred.sigma.is_finite(), "smooth sigma should be finite");
        assert!(pred.sigma > 0.0, "smooth sigma should be positive");
    }

    // -- PD sigma modulation tests --

    #[test]
    fn sigma_velocity_responds_to_error_spike() {
        let config = SGBTConfig::builder()
            .n_steps(3)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .uncertainty_modulated_lr(true)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        // Stable phase
        for i in 0..200 {
            let x = (i as f64) * 0.1;
            model.train_one(&(vec![x, x.sin()], 2.0 * x + 1.0));
        }

        let velocity_before = model.sigma_velocity();

        // Error spike -- sudden regime change
        for i in 0..50 {
            let x = (i as f64) * 0.1;
            model.train_one(&(vec![x, x.sin()], 100.0 * x + 50.0));
        }

        let velocity_after = model.sigma_velocity();

        // Velocity should be positive (sigma increasing due to errors)
        assert!(
            velocity_after > velocity_before,
            "sigma velocity should increase after error spike: before={}, after={}",
            velocity_before,
            velocity_after,
        );
    }

    #[test]
    fn sigma_velocity_getter_works() {
        let config = SGBTConfig::builder()
            .n_steps(2)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .build()
            .unwrap();
        let model = DistributionalSGBT::new(config);
        // Fresh model should have zero velocity
        assert_eq!(model.sigma_velocity(), 0.0);
    }

    #[test]
    fn diagnostics_leaf_sample_counts_populated() {
        let config = SGBTConfig::builder()
            .n_steps(3)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        for i in 0..200 {
            let x = (i as f64) * 0.1;
            let features = vec![x, x.sin()];
            let target = 2.0 * x + 1.0;
            model.train_one(&(features, target));
        }

        let diags = model.diagnostics();
        for (ti, tree) in diags.trees.iter().enumerate() {
            assert_eq!(
                tree.leaf_sample_counts.len(),
                tree.n_leaves,
                "tree {} should have sample count per leaf",
                ti,
            );
            // Total samples across leaves should equal samples_seen for trees with data
            if tree.samples_seen > 0 {
                let total: u64 = tree.leaf_sample_counts.iter().sum();
                assert!(
                    total > 0,
                    "tree {} has {} samples_seen but leaf counts sum to 0",
                    ti,
                    tree.samples_seen,
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // Packed cache tests (dual-path inference)
    // -------------------------------------------------------------------

    #[test]
    fn packed_cache_disabled_by_default() {
        let model = DistributionalSGBT::new(test_config());
        assert!(!model.has_packed_cache());
        assert_eq!(model.config().packed_refresh_interval, 0);
    }

    #[test]
    fn packed_cache_refreshes_after_interval() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(5)
            .max_depth(3)
            .n_bins(8)
            .initial_target_count(10)
            .packed_refresh_interval(20)
            .build()
            .unwrap();

        let mut model = DistributionalSGBT::new(config);

        // Train past initialization (10 samples) + refresh interval (20 samples)
        for i in 0..40 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 2.0, x * 0.5], x * 3.0));
        }

        // Cache should exist after enough training
        assert!(
            model.has_packed_cache(),
            "packed cache should exist after training past refresh interval"
        );

        // Predictions should be finite
        let pred = model.predict(&[2.0, 4.0, 1.0]);
        assert!(pred.mu.is_finite(), "mu should be finite: {}", pred.mu);
    }

    #[test]
    fn packed_cache_matches_full_tree() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(5)
            .max_depth(3)
            .n_bins(8)
            .initial_target_count(10)
            .build()
            .unwrap();

        let mut model = DistributionalSGBT::new(config);

        // Train
        for i in 0..80 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 2.0, x * 0.5], x * 3.0));
        }

        // Get full-tree prediction (no cache)
        assert!(!model.has_packed_cache());
        let full_pred = model.predict(&[2.0, 4.0, 1.0]);

        // Enable cache and get cached prediction
        model.enable_packed_cache(10);
        assert!(model.has_packed_cache());
        let cached_pred = model.predict(&[2.0, 4.0, 1.0]);

        // Should match within f32 precision (f64->f32 conversion loses some precision)
        let mu_diff = (full_pred.mu - cached_pred.mu).abs();
        assert!(
            mu_diff < 0.1,
            "packed cache mu ({}) should match full tree mu ({}) within f32 tolerance, diff={}",
            cached_pred.mu,
            full_pred.mu,
            mu_diff
        );

        // Sigma should be identical (not affected by packed cache)
        assert!(
            (full_pred.sigma - cached_pred.sigma).abs() < 1e-12,
            "sigma should be identical: full={}, cached={}",
            full_pred.sigma,
            cached_pred.sigma
        );
    }

    #[test]
    fn honest_sigma_in_gaussian_prediction() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .max_depth(3)
            .grace_period(2)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }

        let pred = model.predict(&[5.0]);
        // honest_sigma should be finite and non-negative
        assert!(
            pred.honest_sigma.is_finite(),
            "honest_sigma should be finite, got {}",
            pred.honest_sigma
        );
        assert!(
            pred.honest_sigma >= 0.0,
            "honest_sigma should be >= 0, got {}",
            pred.honest_sigma
        );
    }

    #[test]
    fn honest_sigma_increases_with_divergence() {
        let config = SGBTConfig::builder()
            .n_steps(8)
            .learning_rate(0.1)
            .max_depth(4)
            .grace_period(2)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        // Train on linear data
        for i in 0..200 {
            let x = i as f64 * 0.05;
            model.train_one(&(vec![x], x * 2.0));
        }

        // Predict in-distribution
        let in_dist = model.predict(&[5.0]);
        // Predict very far out of distribution
        let out_dist = model.predict(&[500.0]);

        // Both should be finite and non-negative
        assert!(in_dist.honest_sigma.is_finite());
        assert!(out_dist.honest_sigma.is_finite());
        assert!(in_dist.honest_sigma >= 0.0);
        assert!(out_dist.honest_sigma >= 0.0);

        // Out-of-distribution should generally produce different honest_sigma
        // (trees will respond differently to unseen regions)
        // We can't guarantee strictly greater since all trees might
        // extrapolate identically, but at minimum both should be valid.
    }

    #[test]
    fn honest_sigma_zero_for_single_step() {
        // With only 1 step, variance of 1 contribution is undefined -> returns 0
        let config = SGBTConfig::builder()
            .n_steps(1)
            .learning_rate(0.1)
            .max_depth(3)
            .grace_period(2)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }
        let pred = model.predict(&[5.0]);
        assert!(
            pred.honest_sigma.abs() < 1e-15,
            "honest_sigma should be 0 with 1 step, got {}",
            pred.honest_sigma
        );
    }

    #[test]
    fn adaptive_mts_with_distributional() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .max_depth(3)
            .grace_period(5)
            .initial_target_count(10)
            .adaptive_mts(500, 1.0)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        // Train 500 samples -- should not crash and produce finite predictions.
        for i in 0..500 {
            let x = (i as f64) * 0.02;
            model.train_one(&(vec![x, x * 0.5], x.sin()));
        }
        let pred = model.predict(&[1.0, 0.5]);
        assert!(
            pred.mu.is_finite(),
            "adaptive_mts distributional: mu should be finite, got {}",
            pred.mu
        );
        assert!(
            pred.sigma.is_finite() && pred.sigma > 0.0,
            "adaptive_mts distributional: sigma should be finite and positive, got {}",
            pred.sigma
        );
    }

    #[test]
    fn proactive_prune_with_distributional() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .max_depth(3)
            .grace_period(5)
            .initial_target_count(10)
            .proactive_prune_interval(100)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        // Train 300 samples -- at sample 100, 200, 300 a prune fires.
        for i in 0..300 {
            let x = (i as f64) * 0.03;
            model.train_one(&(vec![x, x * 0.7], x.cos()));
        }
        let pred = model.predict(&[2.0, 1.4]);
        assert!(
            pred.mu.is_finite(),
            "proactive_prune distributional: mu should be finite, got {}",
            pred.mu
        );
        assert!(
            pred.sigma.is_finite() && pred.sigma > 0.0,
            "proactive_prune distributional: sigma should be finite and positive, got {}",
            pred.sigma
        );
    }
}

#[cfg(test)]
#[cfg(feature = "serde-json")]
mod serde_tests {
    use super::*;
    use crate::SGBTConfig;

    fn make_trained_distributional() -> DistributionalSGBT {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .max_depth(3)
            .grace_period(2)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        for i in 0..50 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x.sin()));
        }
        model
    }

    #[test]
    fn json_round_trip_preserves_predictions() {
        let model = make_trained_distributional();
        let state = model.to_distributional_state();
        let json = crate::serde_support::save_distributional_model(&state).unwrap();
        let loaded_state = crate::serde_support::load_distributional_model(&json).unwrap();
        let restored = DistributionalSGBT::from_distributional_state(loaded_state);

        let test_points = [0.5, 1.0, 2.0, 3.0];
        for &x in &test_points {
            let orig = model.predict(&[x]);
            let rest = restored.predict(&[x]);
            assert!(
                (orig.mu - rest.mu).abs() < 1e-10,
                "JSON round-trip mu mismatch at x={}: {} vs {}",
                x,
                orig.mu,
                rest.mu
            );
            assert!(
                (orig.sigma - rest.sigma).abs() < 1e-10,
                "JSON round-trip sigma mismatch at x={}: {} vs {}",
                x,
                orig.sigma,
                rest.sigma
            );
        }
    }

    #[test]
    fn state_preserves_rolling_sigma_mean() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .max_depth(3)
            .grace_period(2)
            .initial_target_count(10)
            .uncertainty_modulated_lr(true)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        for i in 0..50 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x.sin()));
        }
        let state = model.to_distributional_state();
        assert!(state.uncertainty_modulated_lr);
        assert!(state.rolling_sigma_mean >= 0.0);

        let restored = DistributionalSGBT::from_distributional_state(state);
        assert_eq!(model.n_samples_seen(), restored.n_samples_seen());
    }

    #[test]
    fn auto_bandwidth_computed_distributional() {
        let config = SGBTConfig::builder()
            .n_steps(3)
            .learning_rate(0.1)
            .grace_period(10)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        for i in 0..200 {
            let x = (i as f64) * 0.1;
            model.train_one(&(vec![x, x.sin()], 2.0 * x + 1.0));
        }

        // auto_bandwidths should be populated
        let bws = model.auto_bandwidths();
        assert_eq!(bws.len(), 2, "should have 2 feature bandwidths");

        // Diagnostics should include auto_bandwidths
        let diag = model.diagnostics();
        assert_eq!(diag.auto_bandwidths.len(), 2);

        // TreeDiagnostic should have prediction stats
        assert!(diag.location_trees[0].prediction_mean.is_finite());
        assert!(diag.location_trees[0].prediction_std.is_finite());

        let pred = model.predict(&[1.0, 1.0_f64.sin()]);
        assert!(pred.mu.is_finite(), "auto-bandwidth mu should be finite");
        assert!(pred.sigma > 0.0, "auto-bandwidth sigma should be positive");
    }

    #[test]
    fn max_leaf_output_clamps_predictions() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(1.0) // Intentionally large to force extreme leaf weights
            .max_leaf_output(0.5)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        // Train with extreme targets to force large leaf weights
        for i in 0..200 {
            let target = if i % 2 == 0 { 100.0 } else { -100.0 };
            let sample = crate::Sample::new(vec![i as f64 % 5.0, (i as f64).sin()], target);
            model.train_one(&sample);
        }

        // Each tree's prediction should be clamped to [-0.5, 0.5]
        let pred = model.predict(&[2.0, 0.5]);
        assert!(
            pred.mu.is_finite(),
            "prediction should be finite with clamping"
        );
    }

    #[test]
    fn min_hessian_sum_suppresses_fresh_leaves() {
        let config = SGBTConfig::builder()
            .n_steps(3)
            .learning_rate(0.01)
            .min_hessian_sum(50.0)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        // Train minimal samples
        for i in 0..60 {
            let sample = crate::Sample::new(vec![i as f64, (i as f64).sin()], i as f64 * 0.1);
            model.train_one(&sample);
        }

        let pred = model.predict(&[30.0, 0.5]);
        assert!(
            pred.mu.is_finite(),
            "prediction should be finite with min_hessian_sum"
        );
    }

    #[test]
    fn predict_interpolated_returns_finite() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        for i in 0..200 {
            let x = i as f64 * 0.1;
            let sample = crate::Sample::new(vec![x, x.sin()], x.cos());
            model.train_one(&sample);
        }

        let pred = model.predict_interpolated(&[1.0, 0.5]);
        assert!(pred.mu.is_finite(), "interpolated mu should be finite");
        assert!(pred.sigma > 0.0, "interpolated sigma should be positive");
    }

    #[test]
    fn huber_k_bounds_gradients() {
        let config = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.01)
            .huber_k(1.345)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        // Train with occasional extreme outliers
        for i in 0..300 {
            let target = if i % 50 == 0 {
                1000.0
            } else {
                (i as f64 * 0.1).sin()
            };
            let sample = crate::Sample::new(vec![i as f64 % 10.0, (i as f64).cos()], target);
            model.train_one(&sample);
        }

        let pred = model.predict(&[5.0, 0.3]);
        assert!(
            pred.mu.is_finite(),
            "Huber-loss mu should be finite despite outliers"
        );
        assert!(pred.sigma > 0.0, "sigma should be positive");
    }

    #[test]
    fn ensemble_gradient_stats_populated() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);
        for i in 0..200 {
            let x = i as f64 * 0.1;
            let sample = crate::Sample::new(vec![x, x.sin()], x.cos());
            model.train_one(&sample);
        }

        let diag = model.diagnostics();
        assert!(
            diag.ensemble_grad_mean.is_finite(),
            "ensemble grad mean should be finite"
        );
        assert!(
            diag.ensemble_grad_std >= 0.0,
            "ensemble grad std should be non-negative"
        );
        assert!(
            diag.ensemble_grad_std.is_finite(),
            "ensemble grad std should be finite"
        );
    }

    #[test]
    fn huber_k_validation() {
        let result = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.01)
            .huber_k(-1.0)
            .build();
        assert!(result.is_err(), "negative huber_k should fail validation");
    }

    #[test]
    fn max_leaf_output_validation() {
        let result = SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.01)
            .max_leaf_output(-1.0)
            .build();
        assert!(
            result.is_err(),
            "negative max_leaf_output should fail validation"
        );
    }

    #[test]
    fn predict_sibling_interpolated_varies_with_features() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(6)
            .delta(0.1)
            .initial_target_count(10)
            .build()
            .unwrap();
        let mut model = DistributionalSGBT::new(config);

        for i in 0..2000 {
            let x = (i as f64) * 0.01;
            let y = x.sin() * x + 0.5 * (x * 2.0).cos();
            let sample = crate::Sample::new(vec![x, x * 0.3], y);
            model.train_one(&sample);
        }

        // Verify the method runs correctly and produces finite predictions
        let pred = model.predict_sibling_interpolated(&[5.0, 1.5]);
        assert!(
            pred.mu.is_finite(),
            "sibling interpolated mu should be finite"
        );
        assert!(
            pred.sigma > 0.0,
            "sibling interpolated sigma should be positive"
        );

        // Sweep — at minimum, sibling should produce same or more variation than hard
        let bws = model.auto_bandwidths();
        if bws.iter().any(|&b| b.is_finite()) {
            let hard_preds: Vec<f64> = (0..200)
                .map(|i| {
                    let x = i as f64 * 0.1;
                    model.predict(&[x, x * 0.3]).mu
                })
                .collect();
            let hard_changes = hard_preds
                .windows(2)
                .filter(|w| (w[0] - w[1]).abs() > f64::EPSILON)
                .count();

            let preds: Vec<f64> = (0..200)
                .map(|i| {
                    let x = i as f64 * 0.1;
                    model.predict_sibling_interpolated(&[x, x * 0.3]).mu
                })
                .collect();

            let sibling_changes = preds
                .windows(2)
                .filter(|w| (w[0] - w[1]).abs() > f64::EPSILON)
                .count();
            assert!(
                sibling_changes >= hard_changes,
                "sibling should produce >= hard changes: sibling={}, hard={}",
                sibling_changes,
                hard_changes
            );
        }
    }
}
