//! Auto-builder: diagnostic-driven config adaptation for streaming AutoML.
//!
//! Replaces random config sampling with mathematically principled derivation
//! from data characteristics and model diagnostics.
//!
//! # Architecture
//!
//! 1. **[`FeasibleRegion`]** -- derives config bounds from (n_samples, n_features, variance)
//! 2. **[`WelfordRace`]** -- batch evaluation with center + directional perturbations
//! 3. **[`DiagnosticAdaptor`]** -- streams config adjustments from model diagnostics

use crate::automl::{ConfigSampler, ModelFactory};
use crate::ensemble::config::SGBTConfig;
use crate::learner::SGBTLearner;
use irithyll_core::learner::StreamingLearner;

// ===========================================================================
// ConfigDiagnostics
// ===========================================================================

/// Model diagnostics consumed by the auto-builder.
///
/// Models that implement diagnostic extraction populate this struct.
/// Fields are optional signals -- return 0.0 if not available.
#[derive(Debug, Clone, Default)]
pub struct ConfigDiagnostics {
    /// Gradient alignment between consecutive steps (-1.0 to 1.0).
    /// Positive = model learning efficiently. Negative = overshooting.
    pub residual_alignment: f64,
    /// How much regularization dominates predictions (0.0 = none, high = over-regularized).
    pub regularization_sensitivity: f64,
    /// Within/between variance ratio. High = need more depth. Low = depth sufficient.
    pub depth_sufficiency: f64,
    /// Effective degrees of freedom (model complexity measure).
    pub effective_dof: f64,
    /// Uncertainty measure (honest_sigma or equivalent).
    pub uncertainty: f64,
}

// ===========================================================================
// ConfigBounds
// ===========================================================================

/// Derived bounds for each config parameter.
#[derive(Debug, Clone)]
pub struct ConfigBounds {
    /// (min, max) for max tree depth.
    pub max_depth: (usize, usize),
    /// (min, max) for number of boosting steps.
    pub n_steps: (usize, usize),
    /// (min, max) for grace period.
    pub grace_period: (usize, usize),
    /// (min, max) for learning rate.
    pub learning_rate: (f64, f64),
    /// (min, max) for L2 regularization lambda.
    pub lambda: (f64, f64),
    /// (min, max) for histogram bins.
    pub n_bins: (usize, usize),
    /// (min, max) for feature subsample rate.
    pub feature_subsample: (f64, f64),
}

// ===========================================================================
// FeasibleRegion
// ===========================================================================

/// Data-derived config bounds using SRM complexity budget.
///
/// The complexity budget constrains the config space:
/// `eta * n_steps * 2^max_depth <= n_samples * epsilon^2 / ln(n_features)`
#[derive(Debug, Clone)]
pub struct FeasibleRegion {
    n_samples: usize,
    n_features: usize,
    target_epsilon: f64,
    budget: f64,
}

impl FeasibleRegion {
    /// Derive feasible region from data characteristics.
    ///
    /// `target_variance` is the observed variance of the target variable
    /// (used to set target_epsilon = sqrt(target_variance) * 0.1).
    pub fn from_data(n_samples: usize, n_features: usize, target_variance: f64) -> Self {
        let target_epsilon = target_variance.sqrt().max(1e-8) * 0.1;
        let log_p = (n_features as f64).max(1.0).ln().max(1.0);
        let budget = n_samples as f64 * target_epsilon.powi(2) / log_p;

        Self {
            n_samples,
            n_features,
            target_epsilon,
            budget,
        }
    }

    /// Derive config bounds from the complexity budget.
    pub fn config_bounds(&self) -> ConfigBounds {
        let max_depth_upper =
            ((self.budget.max(1.0).ln() / core::f64::consts::LN_2).floor() as usize).clamp(2, 6);

        let n_steps_upper = ((self.budget / 4.0).ceil() as usize).clamp(3, 50);

        // Hoeffding-derived grace period: R^2 * ln(1/delta) / (2 * margin^2)
        // R=1.0 (normalized), delta=0.05, margin=target_epsilon
        let gp_upper = ((2.0 * (1.0_f64 / 0.05).ln() / self.target_epsilon.powi(2).max(1e-10))
            .ceil() as usize)
            .clamp(3, 200);

        let n_bins_upper = 64usize.min(self.n_samples / 4).max(8);

        ConfigBounds {
            max_depth: (2, max_depth_upper),
            n_steps: (3, n_steps_upper),
            grace_period: (3, gp_upper),
            learning_rate: (0.05, 0.3),
            lambda: (0.1, 5.0),
            n_bins: (8, n_bins_upper),
            feature_subsample: (0.5, 1.0),
        }
    }

    /// Build an SGBTConfig at the CENTER of the feasible region.
    pub fn center_config(&self) -> SGBTConfig {
        let bounds = self.config_bounds();
        SGBTConfig::builder()
            .max_depth((bounds.max_depth.0 + bounds.max_depth.1) / 2)
            .n_steps((bounds.n_steps.0 + bounds.n_steps.1) / 2)
            .grace_period((bounds.grace_period.0 + bounds.grace_period.1) / 2)
            .learning_rate((bounds.learning_rate.0 * bounds.learning_rate.1).sqrt())
            .lambda((bounds.lambda.0 * bounds.lambda.1).sqrt())
            .n_bins((bounds.n_bins.0 + bounds.n_bins.1) / 2)
            .feature_subsample_rate((bounds.feature_subsample.0 + bounds.feature_subsample.1) / 2.0)
            .build()
            .expect("FeasibleRegion::center_config: feasible region produced invalid config")
    }

    /// Generate center + directional perturbation configs.
    ///
    /// Returns 1 center config + up to 14 perturbations (2 per 7 params).
    /// Each perturbation sets one param to its lower or upper bound while
    /// keeping all other params at center values.
    pub fn perturbation_configs(&self) -> Vec<SGBTConfig> {
        let bounds = self.config_bounds();
        let center = self.center_config();
        let mut configs = vec![center.clone()];

        // Helper: center values for building perturbation configs.
        let center_depth = (bounds.max_depth.0 + bounds.max_depth.1) / 2;
        let center_steps = (bounds.n_steps.0 + bounds.n_steps.1) / 2;
        let center_gp = (bounds.grace_period.0 + bounds.grace_period.1) / 2;
        let center_lr = (bounds.learning_rate.0 * bounds.learning_rate.1).sqrt();
        let center_lambda = (bounds.lambda.0 * bounds.lambda.1).sqrt();
        let center_bins = (bounds.n_bins.0 + bounds.n_bins.1) / 2;
        let center_fsr = (bounds.feature_subsample.0 + bounds.feature_subsample.1) / 2.0;

        // Macro to build a config with one parameter overridden from center.
        // Since SGBTConfig fields are pub and it derives Clone, we clone and mutate.
        macro_rules! perturb {
            ($field:ident, $val:expr) => {{
                let mut cfg = center.clone();
                cfg.$field = $val;
                configs.push(cfg);
            }};
        }

        // max_depth perturbations
        if bounds.max_depth.0 != bounds.max_depth.1 {
            perturb!(max_depth, bounds.max_depth.0);
            perturb!(max_depth, bounds.max_depth.1);
        }

        // n_steps perturbations
        if bounds.n_steps.0 != bounds.n_steps.1 {
            perturb!(n_steps, bounds.n_steps.0);
            perturb!(n_steps, bounds.n_steps.1);
        }

        // grace_period perturbations
        if bounds.grace_period.0 != bounds.grace_period.1 {
            perturb!(grace_period, bounds.grace_period.0);
            perturb!(grace_period, bounds.grace_period.1);
        }

        // learning_rate perturbations
        perturb!(learning_rate, bounds.learning_rate.0);
        perturb!(learning_rate, bounds.learning_rate.1);

        // lambda perturbations
        perturb!(lambda, bounds.lambda.0);
        perturb!(lambda, bounds.lambda.1);

        // n_bins perturbations
        if bounds.n_bins.0 != bounds.n_bins.1 {
            perturb!(n_bins, bounds.n_bins.0);
            perturb!(n_bins, bounds.n_bins.1);
        }

        // feature_subsample_rate perturbations
        perturb!(feature_subsample_rate, bounds.feature_subsample.0);
        perturb!(feature_subsample_rate, bounds.feature_subsample.1);

        // Suppress unused variable warnings for the center values used by
        // the macro's field-mutation approach rather than a builder rebuild.
        let _ = (
            center_depth,
            center_steps,
            center_gp,
            center_lr,
            center_lambda,
            center_bins,
            center_fsr,
        );

        configs
    }

    /// Update bounds as more data arrives.
    pub fn update(&mut self, n_samples: usize) {
        self.n_samples = n_samples;
        let log_p = (self.n_features as f64).max(1.0).ln().max(1.0);
        self.budget = n_samples as f64 * self.target_epsilon.powi(2) / log_p;
    }

    /// Current complexity budget.
    pub fn budget(&self) -> f64 {
        self.budget
    }

    /// Current sample count.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Current feature count.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

// ===========================================================================
// WelfordStats
// ===========================================================================

/// Running statistics for race evaluation using Welford's online algorithm.
#[derive(Debug, Clone, Default)]
pub struct WelfordStats {
    /// Number of observations.
    pub n: u64,
    /// Running mean of error.
    pub mean_error: f64,
    /// Running M2 for variance computation.
    pub m2: f64,
}

impl WelfordStats {
    /// Update running statistics with a new error value.
    pub fn update(&mut self, error: f64) {
        self.n += 1;
        let delta = error - self.mean_error;
        self.mean_error += delta / self.n as f64;
        let delta2 = error - self.mean_error;
        self.m2 += delta * delta2;
    }

    /// Sample variance (Bessel-corrected).
    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / (self.n - 1) as f64
        } else {
            0.0
        }
    }

    /// Standard error of the mean.
    pub fn std_error(&self) -> f64 {
        if self.n > 1 {
            (self.variance() / self.n as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }
}

// ===========================================================================
// RaceCandidate (private)
// ===========================================================================

/// A single candidate in a Welford race.
struct RaceCandidate {
    model: Box<dyn StreamingLearner>,
    stats: WelfordStats,
    config_idx: usize,
}

// Manual Debug impl because Box<dyn StreamingLearner> does not impl Debug.
impl core::fmt::Debug for RaceCandidate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RaceCandidate")
            .field("stats", &self.stats)
            .field("config_idx", &self.config_idx)
            .finish_non_exhaustive()
    }
}

// ===========================================================================
// RaceResults
// ===========================================================================

/// Results from a completed Welford race.
#[derive(Debug, Clone)]
pub struct RaceResults {
    /// Index of the winning config.
    pub winner_idx: usize,
    /// Mean error of the winner.
    pub winner_mean_error: f64,
    /// Per-config results: (config_idx, mean_error, std_error, n_samples).
    pub all_results: Vec<(usize, f64, f64, u64)>,
}

// ===========================================================================
// WelfordRace
// ===========================================================================

/// Welford-based batch race: all candidates see all samples.
///
/// Uses center + directional perturbations from the feasible region.
/// No early elimination -- every config is fully evaluated.
pub struct WelfordRace {
    candidates: Vec<RaceCandidate>,
}

// Manual Debug impl to match RaceCandidate.
impl core::fmt::Debug for WelfordRace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WelfordRace")
            .field("n_candidates", &self.candidates.len())
            .field(
                "n_samples",
                &self.candidates.first().map(|c| c.stats.n).unwrap_or(0),
            )
            .finish()
    }
}

impl WelfordRace {
    /// Create a race from feasible region perturbation configs (SGBT-specific).
    ///
    /// Each config produces an [`SGBTLearner`] with squared loss.
    pub fn new(configs: Vec<SGBTConfig>) -> Self {
        let candidates = configs
            .into_iter()
            .enumerate()
            .map(|(i, config)| RaceCandidate {
                model: Box::new(SGBTLearner::from_config(config)),
                stats: WelfordStats::default(),
                config_idx: i,
            })
            .collect();
        Self { candidates }
    }

    /// Create a race from a [`ModelFactory`] (for non-SGBT models).
    ///
    /// Uses `k` random configs from the factory's config space.
    pub fn from_factory(factory: &dyn ModelFactory, k: usize, seed: u64) -> Self {
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, seed);
        let candidates = (0..k)
            .map(|i| {
                let config = sampler.random();
                RaceCandidate {
                    model: factory.create(&config),
                    stats: WelfordStats::default(),
                    config_idx: i,
                }
            })
            .collect();
        Self { candidates }
    }

    /// Feed one sample to ALL candidates (predict-before-train).
    ///
    /// Each candidate predicts, the absolute error is recorded via Welford,
    /// then the candidate trains on the sample.
    pub fn feed(&mut self, features: &[f64], target: f64) {
        for c in &mut self.candidates {
            let pred = c.model.predict(features);
            let error = (target - pred).abs();
            c.stats.update(error);
            c.model.train_one(features, target, 1.0);
        }
    }

    /// Select winner by lowest Welford mean error.
    ///
    /// Consumes the race and returns the winning model along with full results.
    pub fn select_winner(self) -> (Box<dyn StreamingLearner>, RaceResults) {
        let mut results: Vec<(usize, f64, f64, u64)> = self
            .candidates
            .iter()
            .map(|c| {
                (
                    c.config_idx,
                    c.stats.mean_error,
                    c.stats.std_error(),
                    c.stats.n,
                )
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let winner_idx = results[0].0;
        let winner_mean = results[0].1;

        let winner_model = self
            .candidates
            .into_iter()
            .find(|c| c.config_idx == winner_idx)
            .map(|c| c.model)
            .expect("winner must exist in candidates");

        (
            winner_model,
            RaceResults {
                winner_idx,
                winner_mean_error: winner_mean,
                all_results: results,
            },
        )
    }

    /// Number of candidates in the race.
    pub fn n_candidates(&self) -> usize {
        self.candidates.len()
    }

    /// Number of samples fed so far (from the first candidate).
    pub fn n_samples(&self) -> u64 {
        self.candidates.first().map(|c| c.stats.n).unwrap_or(0)
    }
}

// ===========================================================================
// SmoothAdjustments + StructuralChange
// ===========================================================================

/// Smooth parameter adjustments (applied every sample).
#[derive(Debug, Clone, Default)]
pub struct SmoothAdjustments {
    /// Learning rate multiplier (0.5 = halve, 2.0 = double, 1.0 = no change).
    pub lr_multiplier: f64,
    /// Lambda direction: positive = increase, negative = decrease, 0.0 = no change.
    pub lambda_direction: f64,
}

/// Structural changes (queued for next tree replacement).
#[derive(Debug, Clone, Default)]
pub struct StructuralChange {
    /// Suggested depth change (+1, -1, or 0).
    pub depth_delta: i32,
    /// Suggested n_steps change (+2, -2, or 0).
    pub steps_delta: i32,
}

// ===========================================================================
// DiagnosticAdaptor
// ===========================================================================

/// Streams config adjustments from model diagnostics.
///
/// Smooth adjustments (every sample): LR and lambda.
/// Structural adjustments (at tree replacement): depth and n_steps.
///
/// Call-frequency invariant: adjustments are emitted at fixed intervals
/// derived from the feasible region, so models calling `after_train()`
/// at different rates converge to bounded total adjustment.
#[derive(Debug, Clone)]
pub struct DiagnosticAdaptor {
    /// Running EWMA of uncertainty for baseline comparison.
    uncertainty_ewma: f64,
    /// Running EWMA of residual alignment.
    alignment_ewma: f64,
    /// Running EWMA of regularization sensitivity.
    reg_sensitivity_ewma: f64,
    /// Running EWMA of depth sufficiency signal.
    depth_signal_ewma: f64,
    /// Running EWMA of effective DOF.
    dof_ewma: f64,
    /// EWMA decay factor.
    alpha: f64,
    /// Current feasible region.
    region: FeasibleRegion,
    /// Samples seen.
    n_samples: u64,
    /// Whether initialization phase (first 50 samples = pure observation).
    initialized: bool,

    // --- Call-frequency invariance fields ---
    /// Minimum samples between adjustment emissions (from grace_period center).
    adjustment_interval: u64,
    /// Counter since last adjustment emission.
    samples_since_adjustment: u64,
    /// Per-adjustment step magnitude in log-LR space.
    lr_step: f64,
    /// Per-adjustment step magnitude in lambda space.
    lambda_step: f64,
    /// Cumulative LR adjustment in log space (starts 0.0 = no change).
    cumulative_lr_log: f64,
    /// Cumulative lambda adjustment (starts 0.0 = no change from center).
    cumulative_lambda: f64,
    /// Clamping bounds for cumulative_lr_log.
    lr_log_bounds: (f64, f64),
    /// Clamping bounds for cumulative_lambda.
    lambda_bounds: (f64, f64),
    /// Last emitted cumulative_lr_log (for computing delta multiplier).
    last_emitted_lr_log: f64,
    /// Last emitted cumulative_lambda (for computing delta direction).
    last_emitted_lambda: f64,
}

impl DiagnosticAdaptor {
    /// Create a new adaptor backed by a feasible region.
    pub fn new(region: FeasibleRegion) -> Self {
        let bounds = region.config_bounds();

        // Adjustment interval: center of grace_period range, minimum 1.
        let adjustment_interval =
            ((bounds.grace_period.0 + bounds.grace_period.1) / 2).max(1) as u64;

        // Expected number of adjustment points over model lifetime.
        let expected_adjustments =
            (region.n_samples() as f64 / adjustment_interval as f64).max(1.0);

        // LR step calibrated to traverse feasible LR range over model lifetime (log space).
        let lr_step =
            if bounds.learning_rate.1 > bounds.learning_rate.0 && bounds.learning_rate.0 > 0.0 {
                (bounds.learning_rate.1 / bounds.learning_rate.0).ln() / expected_adjustments
            } else {
                0.01
            };

        // Lambda step calibrated to traverse feasible lambda range over model lifetime.
        let lambda_step = if bounds.lambda.1 > bounds.lambda.0 {
            (bounds.lambda.1 - bounds.lambda.0) / expected_adjustments
        } else {
            0.01
        };

        // LR bounds in log space relative to geometric center.
        let center_lr = (bounds.learning_rate.0 * bounds.learning_rate.1).sqrt();
        let lr_log_bounds = if center_lr > 0.0 {
            (
                (bounds.learning_rate.0 / center_lr).ln(),
                (bounds.learning_rate.1 / center_lr).ln(),
            )
        } else {
            (-1.0, 1.0)
        };

        // Lambda bounds relative to arithmetic center.
        let center_lambda = (bounds.lambda.0 + bounds.lambda.1) / 2.0;
        let lambda_bounds = (
            bounds.lambda.0 - center_lambda,
            bounds.lambda.1 - center_lambda,
        );

        Self {
            uncertainty_ewma: 0.0,
            alignment_ewma: 0.0,
            reg_sensitivity_ewma: 0.0,
            depth_signal_ewma: 0.0,
            dof_ewma: 0.0,
            alpha: 0.01, // slow EWMA for stable baselines
            region,
            n_samples: 0,
            initialized: false,
            adjustment_interval,
            samples_since_adjustment: 0,
            lr_step,
            lambda_step,
            cumulative_lr_log: 0.0,
            cumulative_lambda: 0.0,
            lr_log_bounds,
            lambda_bounds,
            last_emitted_lr_log: 0.0,
            last_emitted_lambda: 0.0,
        }
    }

    /// Process diagnostics after each `train_one()`. Returns smooth adjustments.
    ///
    /// During the first 50 samples, only baselines are updated (no adjustments).
    /// After initialization, adjustments are emitted at fixed intervals
    /// (`adjustment_interval` samples apart) to ensure call-frequency invariance.
    pub fn after_train(&mut self, diagnostics: &ConfigDiagnostics) -> SmoothAdjustments {
        self.n_samples += 1;

        // Always update running baselines regardless of adjustment timing.
        let a = self.alpha;
        self.uncertainty_ewma = a * diagnostics.uncertainty + (1.0 - a) * self.uncertainty_ewma;
        self.alignment_ewma = a * diagnostics.residual_alignment + (1.0 - a) * self.alignment_ewma;
        self.reg_sensitivity_ewma =
            a * diagnostics.regularization_sensitivity + (1.0 - a) * self.reg_sensitivity_ewma;
        self.depth_signal_ewma =
            a * diagnostics.depth_sufficiency + (1.0 - a) * self.depth_signal_ewma;
        self.dof_ewma = a * diagnostics.effective_dof + (1.0 - a) * self.dof_ewma;

        // Initialization phase: observe only, don't adjust.
        if self.n_samples < 50 {
            return SmoothAdjustments {
                lr_multiplier: 1.0,
                lambda_direction: 0.0,
            };
        }
        if !self.initialized {
            self.initialized = true;
        }

        // Increment interval counter; emit no-op until interval elapsed.
        self.samples_since_adjustment += 1;
        if self.samples_since_adjustment < self.adjustment_interval {
            return SmoothAdjustments {
                lr_multiplier: 1.0,
                lambda_direction: 0.0,
            };
        }

        // --- Adjustment point reached ---
        self.samples_since_adjustment = 0;

        // Noise floor threshold: signals below this magnitude are noise.
        let threshold = 2.0 * self.alpha.sqrt();

        // LR adjustment from EWMA alignment (not raw signal).
        // Positive alignment = gradients agree = can increase LR.
        // Negative alignment = overshooting = decrease LR.
        if self.alignment_ewma > threshold {
            self.cumulative_lr_log += self.lr_step;
        } else if self.alignment_ewma < -threshold {
            self.cumulative_lr_log -= self.lr_step;
        }
        // Clamp to feasible bounds.
        self.cumulative_lr_log = self
            .cumulative_lr_log
            .clamp(self.lr_log_bounds.0, self.lr_log_bounds.1);

        // Delta multiplier: ratio of new cumulative to last emitted.
        let lr_multiplier = (self.cumulative_lr_log - self.last_emitted_lr_log).exp();
        self.last_emitted_lr_log = self.cumulative_lr_log;

        // Lambda adjustment from regularization sensitivity ratio.
        // High sensitivity relative to baseline = over-regularized -> decrease lambda.
        // Low sensitivity = under-regularized -> increase lambda.
        if self.reg_sensitivity_ewma > 1e-10 {
            let ratio = diagnostics.regularization_sensitivity / self.reg_sensitivity_ewma;
            if ratio > 1.5 {
                self.cumulative_lambda -= self.lambda_step; // over-regularized
            } else if ratio < 0.5 {
                self.cumulative_lambda += self.lambda_step; // under-regularized
            }
        }
        // Clamp to feasible bounds.
        self.cumulative_lambda = self
            .cumulative_lambda
            .clamp(self.lambda_bounds.0, self.lambda_bounds.1);

        // Delta direction: change since last emission.
        let lambda_direction = self.cumulative_lambda - self.last_emitted_lambda;
        self.last_emitted_lambda = self.cumulative_lambda;

        SmoothAdjustments {
            lr_multiplier,
            lambda_direction,
        }
    }

    /// Evaluate structural changes at tree replacement boundary.
    ///
    /// Returns `Some` if the diagnostics suggest depth or step count changes,
    /// `None` if the current structure is adequate.
    pub fn at_replacement(&mut self, diagnostics: &ConfigDiagnostics) -> Option<StructuralChange> {
        if !self.initialized {
            return None;
        }

        // Update feasible region with current sample count
        self.region.update(self.n_samples as usize);
        let bounds = self.region.config_bounds();

        // Depth sufficiency: compare current signal to baseline
        let needs_more_depth = diagnostics.depth_sufficiency > self.depth_signal_ewma * 1.5
            && bounds.max_depth.1 > bounds.max_depth.0; // room to grow

        // DOF ratio: effective DOF relative to data
        let dof_ratio = if self.n_samples > 0 {
            diagnostics.effective_dof / self.n_samples as f64
        } else {
            0.0
        };

        // Target DOF ratio derived from feasible region budget
        let target_dof_ratio = (self.region.budget() / self.n_samples as f64).clamp(0.01, 0.5);
        let needs_more_steps = dof_ratio < target_dof_ratio * 0.5;
        let needs_fewer_steps = dof_ratio > target_dof_ratio * 2.0;

        if needs_more_depth || needs_more_steps || needs_fewer_steps {
            Some(StructuralChange {
                depth_delta: if needs_more_depth { 1 } else { 0 },
                steps_delta: if needs_more_steps {
                    2
                } else if needs_fewer_steps {
                    -2
                } else {
                    0
                },
            })
        } else {
            None
        }
    }

    /// Reset the adaptor state (all EWMA baselines and cumulative adjustments).
    pub fn reset(&mut self) {
        self.uncertainty_ewma = 0.0;
        self.alignment_ewma = 0.0;
        self.reg_sensitivity_ewma = 0.0;
        self.depth_signal_ewma = 0.0;
        self.dof_ewma = 0.0;
        self.n_samples = 0;
        self.initialized = false;
        self.samples_since_adjustment = 0;
        self.cumulative_lr_log = 0.0;
        self.cumulative_lambda = 0.0;
        self.last_emitted_lr_log = 0.0;
        self.last_emitted_lambda = 0.0;
    }

    /// Current feasible region.
    pub fn region(&self) -> &FeasibleRegion {
        &self.region
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feasible_region_sparse_data() {
        let region = FeasibleRegion::from_data(100, 3, 1.0);
        let bounds = region.config_bounds();
        assert!(
            bounds.max_depth.1 <= 4,
            "sparse data (n=100) should have tight depth: got max {}",
            bounds.max_depth.1
        );
        assert!(
            bounds.n_steps.1 <= 15,
            "sparse data (n=100) should have tight n_steps: got max {}",
            bounds.n_steps.1
        );
    }

    #[test]
    fn feasible_region_abundant_data() {
        let region = FeasibleRegion::from_data(10_000, 3, 1.0);
        let bounds = region.config_bounds();
        assert!(
            bounds.max_depth.1 >= 4,
            "abundant data (n=10000) should allow deeper trees: got max {}",
            bounds.max_depth.1
        );
        assert!(
            bounds.n_steps.1 >= 20,
            "abundant data (n=10000) should allow more steps: got max {}",
            bounds.n_steps.1
        );
    }

    #[test]
    fn feasible_region_center_config_valid() {
        let region = FeasibleRegion::from_data(500, 5, 2.0);
        let config = region.center_config();
        // If we got here without panic, the config passed builder validation.
        assert!(config.n_steps > 0, "center n_steps must be > 0");
        assert!(config.max_depth > 0, "center max_depth must be > 0");
        assert!(
            config.learning_rate > 0.0 && config.learning_rate <= 1.0,
            "center learning_rate must be in (0, 1]"
        );
    }

    #[test]
    fn feasible_region_perturbations() {
        let region = FeasibleRegion::from_data(500, 5, 2.0);
        let configs = region.perturbation_configs();
        assert!(
            configs.len() > 1,
            "perturbation_configs should produce > 1 configs, got {}",
            configs.len()
        );
        // All configs must be valid (they passed the builder or were cloned from valid center)
        for (i, cfg) in configs.iter().enumerate() {
            assert!(cfg.n_steps > 0, "config[{i}] n_steps must be > 0");
            assert!(cfg.max_depth > 0, "config[{i}] max_depth must be > 0");
            assert!(
                cfg.learning_rate > 0.0,
                "config[{i}] learning_rate must be > 0"
            );
        }
    }

    #[test]
    fn feasible_region_update_expands() {
        let mut region = FeasibleRegion::from_data(100, 3, 1.0);
        let budget_before = region.budget();
        region.update(10_000);
        assert!(
            region.budget() > budget_before,
            "budget should increase with more data: before={budget_before}, after={}",
            region.budget()
        );
    }

    #[test]
    fn welford_race_all_see_all() {
        let region = FeasibleRegion::from_data(200, 2, 1.0);
        let configs = region.perturbation_configs();
        let n_configs = configs.len();
        let mut race = WelfordRace::new(configs);

        for i in 0..100 {
            let x = i as f64 * 0.1;
            race.feed(&[x, x * 0.5], x * 2.0 + 1.0);
        }

        assert_eq!(race.n_candidates(), n_configs);
        assert_eq!(race.n_samples(), 100);

        let (_winner, results) = race.select_winner();
        for (idx, _mean, _se, n) in &results.all_results {
            assert_eq!(
                *n, 100,
                "config {idx} should have seen 100 samples, got {n}"
            );
        }
    }

    #[test]
    fn welford_race_selects_best() {
        let region = FeasibleRegion::from_data(500, 1, 1.0);
        let configs = region.perturbation_configs();
        let mut race = WelfordRace::new(configs);

        // y = 2x + small noise (deterministic pseudo-noise)
        for i in 0..200 {
            let x = i as f64 * 0.01;
            let noise = ((i * 7 + 3) % 11) as f64 * 0.001 - 0.005;
            race.feed(&[x], 2.0 * x + noise);
        }

        let (_winner, results) = race.select_winner();
        // The winner should have the lowest mean error
        let winner_mean = results.winner_mean_error;
        for (_, mean, _, _) in &results.all_results {
            assert!(
                winner_mean <= *mean + 1e-12,
                "winner mean {winner_mean} should be <= all others, found {mean}"
            );
        }
    }

    #[test]
    fn welford_stats_accuracy() {
        let mut stats = WelfordStats::default();
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        for v in &values {
            stats.update(*v);
        }

        let expected_mean = 5.0;
        let expected_variance = 4.571428571428571; // population-like with Bessel correction
        assert!(
            (stats.mean_error - expected_mean).abs() < 1e-10,
            "mean should be {expected_mean}, got {}",
            stats.mean_error
        );
        assert!(
            (stats.variance() - expected_variance).abs() < 1e-10,
            "variance should be {expected_variance}, got {}",
            stats.variance()
        );
    }

    #[test]
    fn diagnostic_adaptor_initialization() {
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let mut adaptor = DiagnosticAdaptor::new(region);

        // First 49 samples should return no-op adjustments
        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            regularization_sensitivity: 1.0,
            depth_sufficiency: 0.5,
            effective_dof: 10.0,
            uncertainty: 0.1,
        };

        for _ in 0..49 {
            let adj = adaptor.after_train(&diag);
            assert_eq!(
                adj.lr_multiplier, 1.0,
                "during init phase, lr_multiplier should be 1.0"
            );
            assert_eq!(
                adj.lambda_direction, 0.0,
                "during init phase, lambda_direction should be 0.0"
            );
        }
    }

    #[test]
    fn diagnostic_adaptor_lr_adjustment() {
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let bounds = region.config_bounds();
        let interval = ((bounds.grace_period.0 + bounds.grace_period.1) / 2).max(1) as u64;
        let mut adaptor = DiagnosticAdaptor::new(region);

        // Burn through init phase with positive alignment to build EWMA baseline.
        let positive_diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            ..Default::default()
        };
        for _ in 0..50 {
            adaptor.after_train(&positive_diag);
        }

        // Feed enough post-init samples to trigger one adjustment interval.
        // All with strong positive alignment so EWMA stays above threshold.
        let mut found_increase = false;
        for _ in 0..(interval + 1) {
            let adj = adaptor.after_train(&positive_diag);
            if adj.lr_multiplier > 1.0 {
                found_increase = true;
                break;
            }
        }
        assert!(
            found_increase,
            "positive alignment should produce lr_multiplier > 1.0 within one adjustment interval ({interval} samples)"
        );
    }

    #[test]
    fn diagnostic_adaptor_structural_change() {
        // Use a large initial region. The adaptor will update the region
        // with its own sample count in at_replacement(), so we need enough
        // training samples for the budget to support depth headroom.
        let region = FeasibleRegion::from_data(10_000, 3, 10.0);
        let mut adaptor = DiagnosticAdaptor::new(region);

        // Burn through init phase with low depth_sufficiency baseline.
        // Use many samples so region.update(n) keeps a healthy budget.
        let init_diag = ConfigDiagnostics {
            depth_sufficiency: 0.1,
            effective_dof: 5.0,
            ..Default::default()
        };
        for _ in 0..500 {
            adaptor.after_train(&init_diag);
        }

        // Verify the feasible region (after update(500)) has depth headroom.
        // With target_variance=10.0, epsilon = sqrt(10)*0.1 ≈ 0.316,
        // budget = 500 * 0.1 / ln(3) ≈ 45.5 → ln(45.5)/ln(2) ≈ 5.5 → clamp(2,6) = 5
        let mut check_region = adaptor.region().clone();
        check_region.update(500);
        let bounds = check_region.config_bounds();
        assert!(
            bounds.max_depth.1 > bounds.max_depth.0,
            "region must have depth headroom for this test: bounds={:?}",
            bounds.max_depth
        );

        // Now feed high depth_sufficiency -> should suggest increasing depth
        let high_depth_diag = ConfigDiagnostics {
            depth_sufficiency: 1.0, // much higher than baseline ~0.1
            effective_dof: 5.0,
            ..Default::default()
        };
        let change = adaptor.at_replacement(&high_depth_diag);
        assert!(
            change.is_some(),
            "high depth_sufficiency should trigger structural change"
        );
        let change = change.unwrap();
        assert!(
            change.depth_delta > 0,
            "should suggest increasing depth, got delta={}",
            change.depth_delta
        );
    }

    // =======================================================================
    // Call-frequency invariance tests
    // =======================================================================

    /// Helper: run an adaptor for `n_calls` post-init samples with constant diagnostics.
    /// Returns the product of all emitted lr_multipliers.
    fn run_adaptor_total_lr(n_calls: u64, diag: &ConfigDiagnostics) -> f64 {
        let region = FeasibleRegion::from_data(50_000, 5, 1.0);
        let mut adaptor = DiagnosticAdaptor::new(region);

        // Burn through init phase with same diagnostics (builds EWMA baseline).
        for _ in 0..50 {
            adaptor.after_train(diag);
        }

        // Accumulate total LR adjustment as product of emitted multipliers.
        let mut total_lr_log = 0.0_f64;
        for _ in 0..n_calls {
            let adj = adaptor.after_train(diag);
            total_lr_log += adj.lr_multiplier.ln();
        }
        total_lr_log.exp()
    }

    #[test]
    fn call_frequency_invariance_bounded_total_adjustment() {
        // With the old code, 1.01^40000 = ~1.7e173 (infinity for all purposes).
        // With interval gating, the total must stay bounded within feasible region.
        let diag = ConfigDiagnostics {
            residual_alignment: 0.5, // strong positive signal
            ..Default::default()
        };

        let total_100 = run_adaptor_total_lr(100, &diag);
        let total_10000 = run_adaptor_total_lr(10_000, &diag);
        let total_40000 = run_adaptor_total_lr(40_000, &diag);

        // All totals must be finite.
        assert!(
            total_100.is_finite(),
            "total LR after 100 calls must be finite, got {total_100}"
        );
        assert!(
            total_10000.is_finite(),
            "total LR after 10000 calls must be finite, got {total_10000}"
        );
        assert!(
            total_40000.is_finite(),
            "total LR after 40000 calls must be finite, got {total_40000}"
        );

        // The ratio between 40K and 10K calls should be modest (bounded by clamping).
        // With old code this ratio would be 1.01^30000 ≈ 1e130. Now it should be < 10x.
        let ratio = total_40000 / total_10000;
        assert!(
            ratio < 10.0,
            "40K/10K total LR ratio should be bounded, got {ratio:.4} \
             (total_40000={total_40000:.6}, total_10000={total_10000:.6})"
        );
    }

    #[test]
    fn cumulative_lr_never_exceeds_feasible_bounds() {
        let region = FeasibleRegion::from_data(10_000, 5, 1.0);
        let bounds = region.config_bounds();
        let center_lr = (bounds.learning_rate.0 * bounds.learning_rate.1).sqrt();

        let mut adaptor = DiagnosticAdaptor::new(region);

        // Strong positive alignment to push LR upward.
        let diag = ConfigDiagnostics {
            residual_alignment: 0.9,
            ..Default::default()
        };

        // Burn init + feed many samples.
        let mut total_lr_log = 0.0_f64;
        for _ in 0..5_000 {
            let adj = adaptor.after_train(&diag);
            total_lr_log += adj.lr_multiplier.ln();
        }

        // The effective LR (center * exp(total_log)) must stay within feasible bounds.
        let effective_lr = center_lr * total_lr_log.exp();
        assert!(
            effective_lr <= bounds.learning_rate.1 * 1.01, // small epsilon for float
            "effective LR {effective_lr:.6} must not exceed upper bound {:.6}",
            bounds.learning_rate.1
        );
        assert!(
            effective_lr >= bounds.learning_rate.0 * 0.99,
            "effective LR {effective_lr:.6} must not go below lower bound {:.6}",
            bounds.learning_rate.0
        );
    }

    #[test]
    fn init_phase_returns_noop_with_interval_gating() {
        // Ensure the init phase still works correctly with new interval fields.
        let region = FeasibleRegion::from_data(500, 3, 1.0);
        let mut adaptor = DiagnosticAdaptor::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.9,
            regularization_sensitivity: 5.0,
            depth_sufficiency: 1.0,
            effective_dof: 50.0,
            uncertainty: 0.5,
        };

        // All 49 samples during init must return no-op.
        for i in 0..49 {
            let adj = adaptor.after_train(&diag);
            assert_eq!(
                adj.lr_multiplier, 1.0,
                "init phase sample {i}: lr_multiplier should be 1.0, got {}",
                adj.lr_multiplier
            );
            assert_eq!(
                adj.lambda_direction, 0.0,
                "init phase sample {i}: lambda_direction should be 0.0, got {}",
                adj.lambda_direction
            );
        }
    }

    #[test]
    fn reset_clears_cumulative_state() {
        let region = FeasibleRegion::from_data(1_000, 3, 1.0);
        let mut adaptor = DiagnosticAdaptor::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            ..Default::default()
        };

        // Feed enough samples to accumulate some state.
        for _ in 0..200 {
            adaptor.after_train(&diag);
        }

        adaptor.reset();

        // After reset, init phase should be active again (no-op adjustments).
        let adj = adaptor.after_train(&diag);
        assert_eq!(
            adj.lr_multiplier, 1.0,
            "after reset, first sample should be init-phase no-op, got lr={}",
            adj.lr_multiplier
        );
        assert_eq!(
            adj.lambda_direction, 0.0,
            "after reset, first sample should be init-phase no-op, got lambda={}",
            adj.lambda_direction
        );
    }

    #[test]
    fn between_intervals_returns_noop() {
        let region = FeasibleRegion::from_data(1_000, 3, 1.0);
        let bounds = region.config_bounds();
        let interval = ((bounds.grace_period.0 + bounds.grace_period.1) / 2).max(1) as u64;
        let mut adaptor = DiagnosticAdaptor::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            ..Default::default()
        };

        // Burn init phase.
        for _ in 0..50 {
            adaptor.after_train(&diag);
        }

        // The very next sample starts the interval counter at 1.
        // For samples 1..(interval-1), adjustment should be no-op.
        if interval > 2 {
            // Feed one sample to start the counter.
            let adj = adaptor.after_train(&diag);
            assert_eq!(
                adj.lr_multiplier, 1.0,
                "first post-init sample (before interval) should be no-op, got lr={}",
                adj.lr_multiplier
            );
        }
    }
}
