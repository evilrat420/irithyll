//! Auto-builder: diagnostic-driven config adaptation for streaming AutoML.
//!
//! Replaces random config sampling with mathematically principled derivation
//! from data characteristics and model diagnostics. The [`DiagnosticLearner`]
//! uses an RLS meta-learner to DISCOVER the relationship between diagnostic
//! signals and optimal config adjustments from observed performance data,
//! replacing the hardcoded signal-to-adjustment rules of the previous
//! `DiagnosticAdaptor`.
//!
//! # Architecture
//!
//! 1. **[`FeasibleRegion`]** -- derives config bounds from (n_samples, n_features, variance)
//! 2. **[`WelfordRace`]** -- batch evaluation with center + directional perturbations
//! 3. **[`DiagnosticLearner`]** -- RLS meta-learner discovers config adjustments from diagnostics

use crate::automl::{ConfigSampler, ModelFactory};
use crate::ensemble::config::SGBTConfig;
use crate::learner::SGBTLearner;
use crate::learners::rls::RecursiveLeastSquares;
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
// MetaObjective
// ===========================================================================

/// Configurable optimization objective for the meta-learner.
#[derive(Debug, Clone, Copy, Default)]
pub enum MetaObjective {
    /// Minimize root mean squared error (default for regression).
    #[default]
    MinimizeRMSE,
    /// Maximize R-squared (coefficient of determination).
    MaximizeR2,
    /// Maximize directional accuracy (correct sign prediction).
    MaximizeDirection,
    /// Maximize F1 score (harmonic mean of precision and recall).
    MaximizeF1,
    /// Maximize Cohen's kappa (agreement beyond chance).
    MaximizeKappa,
    /// Weighted combination of multiple objectives.
    Composite {
        /// Weight for RMSE component (lower is better, so this is negated internally).
        rmse_weight: f64,
        /// Weight for R-squared component.
        r2_weight: f64,
        /// Weight for directional accuracy component.
        dir_weight: f64,
    },
}

// ===========================================================================
// DiagnosticLearner
// ===========================================================================

/// RLS meta-learner that discovers config adjustments from diagnostic signals.
///
/// Replaces the hardcoded signal-to-adjustment rules of the previous
/// `DiagnosticAdaptor` with a 7-feature RLS that learns the mapping from
/// diagnostic signals to performance changes from observed data.
///
/// # Meta-learner features
///
/// The RLS operates on a 7-dimensional feature vector:
/// 1. `alignment` -- EWMA of residual alignment
/// 2. `reg_sensitivity_ratio` -- current / baseline regularization sensitivity
/// 3. `depth_sufficiency` -- EWMA of depth sufficiency signal
/// 4. `dof_ratio` -- effective DOF / n_samples
/// 5. `uncertainty` -- EWMA of model uncertainty
/// 6. `lr_normalized` -- current cumulative LR in log space, normalized to \[-1, 1\]
/// 7. `lambda_normalized` -- current cumulative lambda, normalized to \[-1, 1\]
///
/// # Observation cycle
///
/// 1. Baselines always update (every sample)
/// 2. Performance trackers always update (RMSE, R², direction, F1, kappa)
/// 3. Init phase (< 50 samples): no adjustments
/// 4. At observation boundaries: build feature vector, train RLS
/// 5. After `min_observations` (14): use counterfactual queries to decide adjustments
#[derive(Debug)]
pub struct DiagnosticLearner {
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

    // --- Meta-learner fields ---
    /// RLS meta-learner (7 features -> performance delta).
    meta_rls: RecursiveLeastSquares,
    /// Number of observations the RLS has been trained on.
    meta_observations: u64,
    /// Minimum observations before the meta-learner emits adjustments.
    min_observations: u64,
    /// Optimization objective.
    objective: MetaObjective,

    // --- Performance trackers ---
    /// EWMA of squared errors (for RMSE).
    squared_error_ewma: f64,
    /// EWMA of target values (for R²).
    target_ewma: f64,
    /// EWMA of squared target deviation from mean (for R²).
    target_var_ewma: f64,
    /// EWMA of correct direction predictions (for directional accuracy).
    direction_ewma: f64,
    /// EWMA of true positives (for F1).
    tp_ewma: f64,
    /// EWMA of false positives (for F1).
    fp_ewma: f64,
    /// EWMA of false negatives (for F1).
    fn_ewma: f64,
    /// EWMA of observed accuracy (for kappa).
    accuracy_ewma: f64,
    /// EWMA of positive rate in targets (for kappa).
    pos_rate_ewma: f64,
    /// EWMA of positive rate in predictions (for kappa).
    pred_pos_rate_ewma: f64,
    /// Previous performance value (for computing delta).
    prev_performance: f64,
}

impl DiagnosticLearner {
    /// Create a new learner backed by a feasible region with default objective.
    pub fn new(region: FeasibleRegion) -> Self {
        Self::with_objective(region, MetaObjective::default())
    }

    /// Create a new learner backed by a feasible region with a specific objective.
    pub fn with_objective(region: FeasibleRegion, objective: MetaObjective) -> Self {
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

        // RLS meta-learner: forgetting factor 0.998, delta 1.0.
        // Lazily initialized on first observation (7 features).
        let meta_rls = RecursiveLeastSquares::with_delta(0.998, 1.0);

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
            meta_rls,
            meta_observations: 0,
            min_observations: 14,
            objective,
            squared_error_ewma: 0.0,
            target_ewma: 0.0,
            target_var_ewma: 0.0,
            direction_ewma: 0.5,
            tp_ewma: 0.0,
            fp_ewma: 0.0,
            fn_ewma: 0.0,
            accuracy_ewma: 0.5,
            pos_rate_ewma: 0.5,
            pred_pos_rate_ewma: 0.5,
            prev_performance: 0.0,
        }
    }

    /// Process diagnostics after each `train_one()`. Returns smooth adjustments.
    ///
    /// The meta-learner observes diagnostic signals alongside prediction
    /// performance and learns the mapping from signals to performance deltas.
    /// During the first 50 samples, only baselines are updated (no adjustments).
    /// After the init phase, adjustments are emitted at fixed intervals.
    /// The RLS requires `min_observations` (14) training points before it
    /// emits non-trivial adjustments via counterfactual queries.
    pub fn after_train(
        &mut self,
        diagnostics: &ConfigDiagnostics,
        prediction: f64,
        target: f64,
    ) -> SmoothAdjustments {
        self.n_samples += 1;
        let a = self.alpha;

        // 1. Always update diagnostic baselines (EWMAs).
        self.uncertainty_ewma = a * diagnostics.uncertainty + (1.0 - a) * self.uncertainty_ewma;
        self.alignment_ewma = a * diagnostics.residual_alignment + (1.0 - a) * self.alignment_ewma;
        self.reg_sensitivity_ewma =
            a * diagnostics.regularization_sensitivity + (1.0 - a) * self.reg_sensitivity_ewma;
        self.depth_signal_ewma =
            a * diagnostics.depth_sufficiency + (1.0 - a) * self.depth_signal_ewma;
        self.dof_ewma = a * diagnostics.effective_dof + (1.0 - a) * self.dof_ewma;

        // 2. Always update performance trackers.
        let error = target - prediction;
        self.squared_error_ewma = a * (error * error) + (1.0 - a) * self.squared_error_ewma;

        let old_target_ewma = self.target_ewma;
        self.target_ewma = a * target + (1.0 - a) * self.target_ewma;
        let dev = target - old_target_ewma;
        self.target_var_ewma = a * (dev * dev) + (1.0 - a) * self.target_var_ewma;

        // Directional accuracy: both have the same sign (or both zero).
        let correct_dir = if (prediction * target) >= 0.0 {
            1.0
        } else {
            0.0
        };
        self.direction_ewma = a * correct_dir + (1.0 - a) * self.direction_ewma;

        // F1 components.
        let predicted_positive = prediction > 0.5;
        let actual_positive = target > 0.5;
        let tp = if predicted_positive && actual_positive {
            1.0
        } else {
            0.0
        };
        let fp = if predicted_positive && !actual_positive {
            1.0
        } else {
            0.0
        };
        let fn_ = if !predicted_positive && actual_positive {
            1.0
        } else {
            0.0
        };
        self.tp_ewma = a * tp + (1.0 - a) * self.tp_ewma;
        self.fp_ewma = a * fp + (1.0 - a) * self.fp_ewma;
        self.fn_ewma = a * fn_ + (1.0 - a) * self.fn_ewma;

        // Kappa components.
        let correct = if (predicted_positive && actual_positive)
            || (!predicted_positive && !actual_positive)
        {
            1.0
        } else {
            0.0
        };
        self.accuracy_ewma = a * correct + (1.0 - a) * self.accuracy_ewma;
        self.pos_rate_ewma =
            a * (if actual_positive { 1.0 } else { 0.0 }) + (1.0 - a) * self.pos_rate_ewma;
        self.pred_pos_rate_ewma =
            a * (if predicted_positive { 1.0 } else { 0.0 }) + (1.0 - a) * self.pred_pos_rate_ewma;

        // 3. Init phase (< 50 samples): return no-op.
        if self.n_samples < 50 {
            return SmoothAdjustments {
                lr_multiplier: 1.0,
                lambda_direction: 0.0,
            };
        }
        if !self.initialized {
            self.initialized = true;
        }

        // 4. Check observation interval: if not at boundary, return no-op.
        self.samples_since_adjustment += 1;
        if self.samples_since_adjustment < self.adjustment_interval {
            return SmoothAdjustments {
                lr_multiplier: 1.0,
                lambda_direction: 0.0,
            };
        }

        // --- Observation boundary reached ---
        self.samples_since_adjustment = 0;

        // 5. Compute performance based on MetaObjective.
        let performance = self.current_performance();
        let perf_delta = performance - self.prev_performance;
        self.prev_performance = performance;

        // 6. Build normalized 7-feature vector.
        let features = self.build_feature_vector(diagnostics);

        // 7. Train RLS (predict-before-train).
        let _rls_pred = self.meta_rls.predict(&features);
        self.meta_rls.train_one(&features, perf_delta, 1.0);
        self.meta_observations += 1;

        // 8. Below min_observations: return no-op (not enough data for counterfactuals).
        if self.meta_observations < self.min_observations {
            return SmoothAdjustments {
                lr_multiplier: 1.0,
                lambda_direction: 0.0,
            };
        }

        // 9. Counterfactual queries: probe LR up/down, lambda up/down.
        let mut features_lr_up = features.clone();
        let mut features_lr_down = features.clone();
        let mut features_lambda_up = features.clone();
        let mut features_lambda_down = features.clone();

        // Feature index 5 = lr_normalized, index 6 = lambda_normalized.
        let lr_probe = 0.2; // probe 20% of normalized range
        features_lr_up[5] = (features[5] + lr_probe).clamp(-1.0, 1.0);
        features_lr_down[5] = (features[5] - lr_probe).clamp(-1.0, 1.0);
        features_lambda_up[6] = (features[6] + lr_probe).clamp(-1.0, 1.0);
        features_lambda_down[6] = (features[6] - lr_probe).clamp(-1.0, 1.0);

        let pred_lr_up = self.meta_rls.predict(&features_lr_up);
        let pred_lr_down = self.meta_rls.predict(&features_lr_down);
        let pred_lambda_up = self.meta_rls.predict(&features_lambda_up);
        let pred_lambda_down = self.meta_rls.predict(&features_lambda_down);

        // 10. Apply calibrated adjustments: move toward the direction that
        // the meta-learner predicts will improve performance.
        // LR: if increasing LR predicts better performance, increase; else decrease.
        let lr_gradient = pred_lr_up - pred_lr_down;
        if lr_gradient > 0.0 {
            self.cumulative_lr_log += self.lr_step;
        } else if lr_gradient < 0.0 {
            self.cumulative_lr_log -= self.lr_step;
        }
        self.cumulative_lr_log = self
            .cumulative_lr_log
            .clamp(self.lr_log_bounds.0, self.lr_log_bounds.1);

        // Lambda: if increasing lambda predicts better performance, increase; else decrease.
        let lambda_gradient = pred_lambda_up - pred_lambda_down;
        if lambda_gradient > 0.0 {
            self.cumulative_lambda += self.lambda_step;
        } else if lambda_gradient < 0.0 {
            self.cumulative_lambda -= self.lambda_step;
        }
        self.cumulative_lambda = self
            .cumulative_lambda
            .clamp(self.lambda_bounds.0, self.lambda_bounds.1);

        // 11. Emit delta since last emission.
        let lr_multiplier = (self.cumulative_lr_log - self.last_emitted_lr_log).exp();
        self.last_emitted_lr_log = self.cumulative_lr_log;

        let lambda_direction = self.cumulative_lambda - self.last_emitted_lambda;
        self.last_emitted_lambda = self.cumulative_lambda;

        SmoothAdjustments {
            lr_multiplier,
            lambda_direction,
        }
    }

    /// Fallback for callers that cannot provide prediction/target.
    ///
    /// Passes zero prediction and zero target, which still updates diagnostic
    /// baselines and interval gating but provides no useful performance signal
    /// to the meta-learner.
    pub fn after_train_diagnostics_only(
        &mut self,
        diagnostics: &ConfigDiagnostics,
    ) -> SmoothAdjustments {
        self.after_train(diagnostics, 0.0, 0.0)
    }

    /// Evaluate structural changes at tree replacement boundary.
    ///
    /// Returns `Some` if the diagnostics suggest depth or step count changes,
    /// `None` if the current structure is adequate.
    ///
    /// Structural changes remain rule-based (not learned) because they are
    /// infrequent, discrete events that the meta-learner cannot observe
    /// often enough to learn from.
    pub fn at_replacement(&mut self, diagnostics: &ConfigDiagnostics) -> Option<StructuralChange> {
        if !self.initialized {
            return None;
        }

        // Update feasible region with current sample count.
        self.region.update(self.n_samples as usize);
        let bounds = self.region.config_bounds();

        // Depth sufficiency: compare current signal to baseline.
        let needs_more_depth = diagnostics.depth_sufficiency > self.depth_signal_ewma * 1.5
            && bounds.max_depth.1 > bounds.max_depth.0; // room to grow

        // DOF ratio: effective DOF relative to data.
        let dof_ratio = if self.n_samples > 0 {
            diagnostics.effective_dof / self.n_samples as f64
        } else {
            0.0
        };

        // Target DOF ratio derived from feasible region budget.
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

    /// Reset the learner state (all EWMA baselines, cumulative adjustments, and meta-learner).
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
        self.meta_rls = RecursiveLeastSquares::with_delta(0.998, 1.0);
        self.meta_observations = 0;
        self.squared_error_ewma = 0.0;
        self.target_ewma = 0.0;
        self.target_var_ewma = 0.0;
        self.direction_ewma = 0.5;
        self.tp_ewma = 0.0;
        self.fp_ewma = 0.0;
        self.fn_ewma = 0.0;
        self.accuracy_ewma = 0.5;
        self.pos_rate_ewma = 0.5;
        self.pred_pos_rate_ewma = 0.5;
        self.prev_performance = 0.0;
    }

    /// Current feasible region.
    pub fn region(&self) -> &FeasibleRegion {
        &self.region
    }

    /// Number of observations the meta-learner has been trained on.
    pub fn meta_observations(&self) -> u64 {
        self.meta_observations
    }

    /// Current optimization objective.
    pub fn objective(&self) -> MetaObjective {
        self.objective
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Compute current performance value based on the configured objective.
    ///
    /// Higher is always better (for "minimize" objectives we negate).
    fn current_performance(&self) -> f64 {
        match self.objective {
            MetaObjective::MinimizeRMSE => -self.squared_error_ewma.sqrt(),
            MetaObjective::MaximizeR2 => {
                if self.target_var_ewma > 1e-15 {
                    1.0 - self.squared_error_ewma / self.target_var_ewma
                } else {
                    0.0
                }
            }
            MetaObjective::MaximizeDirection => self.direction_ewma,
            MetaObjective::MaximizeF1 => {
                let denom = 2.0 * self.tp_ewma + self.fp_ewma + self.fn_ewma;
                if denom > 1e-15 {
                    2.0 * self.tp_ewma / denom
                } else {
                    0.0
                }
            }
            MetaObjective::MaximizeKappa => {
                let expected = self.pos_rate_ewma * self.pred_pos_rate_ewma
                    + (1.0 - self.pos_rate_ewma) * (1.0 - self.pred_pos_rate_ewma);
                if (1.0 - expected).abs() > 1e-15 {
                    (self.accuracy_ewma - expected) / (1.0 - expected)
                } else {
                    0.0
                }
            }
            MetaObjective::Composite {
                rmse_weight,
                r2_weight,
                dir_weight,
            } => {
                let rmse_score = -self.squared_error_ewma.sqrt();
                let r2_score = if self.target_var_ewma > 1e-15 {
                    1.0 - self.squared_error_ewma / self.target_var_ewma
                } else {
                    0.0
                };
                let dir_score = self.direction_ewma;
                rmse_weight * rmse_score + r2_weight * r2_score + dir_weight * dir_score
            }
        }
    }

    /// Build the 7-dimensional normalized feature vector for the meta-learner.
    fn build_feature_vector(&self, diagnostics: &ConfigDiagnostics) -> Vec<f64> {
        // 1. alignment: EWMA of residual alignment (already in [-1, 1]).
        let alignment = self.alignment_ewma.clamp(-1.0, 1.0);

        // 2. reg_sensitivity_ratio: current / baseline, normalized to [-1, 1].
        let reg_ratio = if self.reg_sensitivity_ewma > 1e-10 {
            (diagnostics.regularization_sensitivity / self.reg_sensitivity_ewma - 1.0)
                .clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // 3. depth_sufficiency: EWMA, scaled to roughly [-1, 1].
        let depth_suf = self.depth_signal_ewma.clamp(-1.0, 1.0);

        // 4. dof_ratio: effective DOF / n_samples.
        let dof_ratio = if self.n_samples > 0 {
            (diagnostics.effective_dof / self.n_samples as f64).clamp(0.0, 1.0) * 2.0 - 1.0
        } else {
            0.0
        };

        // 5. uncertainty: EWMA of uncertainty, tanh-squashed.
        let uncertainty = self.uncertainty_ewma.tanh();

        // 6. lr_normalized: cumulative LR in log space, normalized to [-1, 1].
        let lr_range = self.lr_log_bounds.1 - self.lr_log_bounds.0;
        let lr_normalized = if lr_range > 1e-15 {
            ((self.cumulative_lr_log - self.lr_log_bounds.0) / lr_range * 2.0 - 1.0)
                .clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // 7. lambda_normalized: cumulative lambda, normalized to [-1, 1].
        let lambda_range = self.lambda_bounds.1 - self.lambda_bounds.0;
        let lambda_normalized = if lambda_range > 1e-15 {
            ((self.cumulative_lambda - self.lambda_bounds.0) / lambda_range * 2.0 - 1.0)
                .clamp(-1.0, 1.0)
        } else {
            0.0
        };

        vec![
            alignment,
            reg_ratio,
            depth_suf,
            dof_ratio,
            uncertainty,
            lr_normalized,
            lambda_normalized,
        ]
    }
}

/// Backward compatibility alias.
pub type DiagnosticAdaptor = DiagnosticLearner;

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

        for i in 0..200 {
            let x = i as f64 * 0.01;
            let noise = ((i * 7 + 3) % 11) as f64 * 0.001 - 0.005;
            race.feed(&[x], 2.0 * x + noise);
        }

        let (_winner, results) = race.select_winner();
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
        let expected_variance = 4.571428571428571;
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

    // =======================================================================
    // DiagnosticLearner tests
    // =======================================================================

    #[test]
    fn diagnostic_learner_min_observations_gating() {
        // No adjustments before 14 meta-observations (which requires 50 init + 14 intervals).
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let bounds = region.config_bounds();
        let interval = ((bounds.grace_period.0 + bounds.grace_period.1) / 2).max(1) as u64;
        let mut learner = DiagnosticLearner::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            regularization_sensitivity: 1.0,
            depth_sufficiency: 0.5,
            effective_dof: 10.0,
            uncertainty: 0.1,
        };

        // Burn through init phase (49 samples = no-op).
        for _ in 0..49 {
            let adj = learner.after_train(&diag, 0.5, 1.0);
            assert_eq!(
                adj.lr_multiplier, 1.0,
                "during init phase, lr_multiplier should be 1.0"
            );
            assert_eq!(
                adj.lambda_direction, 0.0,
                "during init phase, lambda_direction should be 0.0"
            );
        }

        // After init, feed enough to reach 13 observation intervals (not yet 14).
        // Each interval is `interval` samples. We need 13 * interval samples
        // after init to get 13 observations, which is below min_observations.
        for _ in 0..(13 * interval + interval - 1) {
            let adj = learner.after_train(&diag, 0.5, 1.0);
            // All should be no-op: either between intervals, or below min_observations.
            assert_eq!(
                adj.lr_multiplier,
                1.0,
                "before min_observations, lr_multiplier should be 1.0 (meta_obs={})",
                learner.meta_observations()
            );
            assert_eq!(
                adj.lambda_direction,
                0.0,
                "before min_observations, lambda_direction should be 0.0 (meta_obs={})",
                learner.meta_observations()
            );
        }
    }

    #[test]
    fn diagnostic_learner_cumulative_clamping() {
        // LR and lambda must never exceed feasible bounds even with many observations.
        let region = FeasibleRegion::from_data(10_000, 5, 1.0);
        let bounds = region.config_bounds();
        let center_lr = (bounds.learning_rate.0 * bounds.learning_rate.1).sqrt();

        let mut learner = DiagnosticLearner::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.9,
            regularization_sensitivity: 0.1,
            depth_sufficiency: 0.5,
            effective_dof: 10.0,
            uncertainty: 0.1,
        };

        let mut total_lr_log = 0.0_f64;
        for i in 0..5_000 {
            let pred = i as f64 * 0.01;
            let target = pred + 0.01;
            let adj = learner.after_train(&diag, pred, target);
            total_lr_log += adj.lr_multiplier.ln();
        }

        let effective_lr = center_lr * total_lr_log.exp();
        assert!(
            effective_lr <= bounds.learning_rate.1 * 1.01,
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
    fn diagnostic_learner_backward_compat_alias() {
        // DiagnosticAdaptor alias should work identically to DiagnosticLearner.
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let mut adaptor: DiagnosticAdaptor = DiagnosticAdaptor::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            ..Default::default()
        };

        // Should compile and behave identically.
        let adj = adaptor.after_train(&diag, 0.0, 0.0);
        assert_eq!(
            adj.lr_multiplier, 1.0,
            "backward compat alias: init phase should return no-op"
        );

        // after_train_diagnostics_only should also work via alias.
        let adj2 = adaptor.after_train_diagnostics_only(&diag);
        assert_eq!(
            adj2.lr_multiplier, 1.0,
            "backward compat alias: diagnostics_only should return no-op"
        );
    }

    #[test]
    fn diagnostic_learner_observation_interval_gating() {
        // Between observation intervals, adjustments should be no-op.
        let region = FeasibleRegion::from_data(1_000, 3, 1.0);
        let bounds = region.config_bounds();
        let interval = ((bounds.grace_period.0 + bounds.grace_period.1) / 2).max(1) as u64;
        let mut learner = DiagnosticLearner::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            ..Default::default()
        };

        // Burn init phase.
        for _ in 0..50 {
            learner.after_train(&diag, 0.5, 1.0);
        }

        // Between intervals (first sample after init), should be no-op.
        if interval > 2 {
            let adj = learner.after_train(&diag, 0.5, 1.0);
            assert_eq!(
                adj.lr_multiplier, 1.0,
                "first post-init sample (before interval) should be no-op, got lr={}",
                adj.lr_multiplier
            );
        }
    }

    #[test]
    fn diagnostic_learner_structural_change() {
        let region = FeasibleRegion::from_data(10_000, 3, 10.0);
        let mut learner = DiagnosticLearner::new(region);

        let init_diag = ConfigDiagnostics {
            depth_sufficiency: 0.1,
            effective_dof: 5.0,
            ..Default::default()
        };
        for _ in 0..500 {
            learner.after_train(&init_diag, 0.5, 1.0);
        }

        let mut check_region = learner.region().clone();
        check_region.update(500);
        let bounds = check_region.config_bounds();
        assert!(
            bounds.max_depth.1 > bounds.max_depth.0,
            "region must have depth headroom for this test: bounds={:?}",
            bounds.max_depth
        );

        let high_depth_diag = ConfigDiagnostics {
            depth_sufficiency: 1.0,
            effective_dof: 5.0,
            ..Default::default()
        };
        let change = learner.at_replacement(&high_depth_diag);
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

    #[test]
    fn diagnostic_learner_reset_clears_state() {
        let region = FeasibleRegion::from_data(1_000, 3, 1.0);
        let mut learner = DiagnosticLearner::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            ..Default::default()
        };

        for i in 0..200 {
            learner.after_train(&diag, i as f64 * 0.01, i as f64 * 0.01 + 0.1);
        }

        learner.reset();

        // After reset, meta_observations should be 0.
        assert_eq!(
            learner.meta_observations(),
            0,
            "meta_observations should be 0 after reset"
        );

        // After reset, init phase should be active again (no-op adjustments).
        let adj = learner.after_train(&diag, 0.0, 0.0);
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
    fn diagnostic_learner_meta_objective_default() {
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let learner = DiagnosticLearner::new(region);
        assert!(
            matches!(learner.objective(), MetaObjective::MinimizeRMSE),
            "default objective should be MinimizeRMSE"
        );
    }

    #[test]
    fn diagnostic_learner_with_custom_objective() {
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let learner = DiagnosticLearner::with_objective(region, MetaObjective::MaximizeF1);
        assert!(
            matches!(learner.objective(), MetaObjective::MaximizeF1),
            "objective should be MaximizeF1"
        );
    }

    // =======================================================================
    // Call-frequency invariance tests
    // =======================================================================

    /// Helper: run a learner for `n_calls` post-init samples with constant
    /// diagnostics. Returns the product of all emitted lr_multipliers.
    fn run_learner_total_lr(n_calls: u64, diag: &ConfigDiagnostics) -> f64 {
        let region = FeasibleRegion::from_data(50_000, 5, 1.0);
        let mut learner = DiagnosticLearner::new(region);

        // Burn through init phase.
        for i in 0..50 {
            learner.after_train(diag, i as f64 * 0.01, i as f64 * 0.01 + 0.1);
        }

        let mut total_lr_log = 0.0_f64;
        for i in 0..n_calls {
            let pred = (50 + i) as f64 * 0.01;
            let target = pred + 0.1;
            let adj = learner.after_train(diag, pred, target);
            total_lr_log += adj.lr_multiplier.ln();
        }
        total_lr_log.exp()
    }

    #[test]
    fn call_frequency_invariance_bounded_total_adjustment() {
        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            ..Default::default()
        };

        let total_100 = run_learner_total_lr(100, &diag);
        let total_10000 = run_learner_total_lr(10_000, &diag);
        let total_40000 = run_learner_total_lr(40_000, &diag);

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

        let ratio = total_40000 / total_10000;
        assert!(
            ratio < 10.0,
            "40K/10K total LR ratio should be bounded, got {ratio:.4} \
             (total_40000={total_40000:.6}, total_10000={total_10000:.6})"
        );
    }
}
