//! Auto-builder: diagnostic-driven config adaptation for streaming AutoML.
//!
//! Replaces random config sampling with mathematically principled derivation
//! from data characteristics and model diagnostics. The [`DiagnosticLearner`]
//! uses SPSA (Simultaneous Perturbation Stochastic Approximation) to optimize
//! learning rate and lambda directly from observed performance, replacing
//! the hardcoded signal-to-adjustment rules of the previous `DiagnosticAdaptor`.
//!
//! # Architecture
//!
//! 1. **[`FeasibleRegion`]** -- derives config bounds from (n_samples, n_features, variance)
//! 2. **[`WelfordRace`]** -- batch evaluation with center + directional perturbations
//! 3. **[`DiagnosticLearner`]** -- SPSA optimizer discovers config adjustments from performance

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

        // Lambda bounds: derived from target scale so regularization matches
        // gradient magnitude. High-variance targets need higher lambda, low-variance
        // need lower. Prevents stumpy trees from over-regularization.
        let target_std = self.target_epsilon * 10.0; // undo the 0.1 scaling from from_data()
        let lambda_center = target_std.max(0.01);
        let lambda_bounds = (lambda_center * 0.1, lambda_center * 3.0);

        ConfigBounds {
            max_depth: (2, max_depth_upper),
            n_steps: (3, n_steps_upper),
            grace_period: (3, gp_upper),
            learning_rate: (0.05, 0.3),
            lambda: lambda_bounds,
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
// SPSAPhase
// ===========================================================================

/// Phase of the SPSA optimization cycle.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SPSAPhase {
    /// Collecting initial performance variance (first 50 samples).
    Init,
    /// Evaluating performance under theta + c*delta perturbation.
    PerturbPlus,
    /// Evaluating performance under theta - c*delta perturbation.
    PerturbMinus,
}

// ===========================================================================
// DiagnosticLearner
// ===========================================================================

/// SPSA optimizer that discovers config adjustments from performance signals.
///
/// Replaces the hardcoded signal-to-adjustment rules of the previous
/// `DiagnosticAdaptor` with SPSA (Simultaneous Perturbation Stochastic
/// Approximation) that optimizes learning rate and lambda directly from
/// observed performance, using only 2 function evaluations per iteration
/// regardless of parameter dimensionality.
///
/// # SPSA optimization cycle
///
/// 1. Baselines always update (every sample)
/// 2. Performance trackers always update (RMSE, R², direction, F1, kappa)
/// 3. Init phase (first 50 samples): calibrate perturbation from noise variance
/// 4. PerturbPlus: evaluate performance under theta + c*delta
/// 5. PerturbMinus: evaluate performance under theta - c*delta
/// 6. Gradient estimate + theta update with divergence guard and CUSUM regime detection
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

    // --- Observation interval ---
    /// Minimum samples between phase transitions (from grace_period center).
    observation_interval: u64,

    // --- SPSA config optimization ---
    /// \[lr_log_normalized, lambda_normalized\] in \[0, 1\].
    theta: [f64; 2],
    /// Best theta found so far.
    theta_best: [f64; 2],
    /// Best performance observed.
    best_performance: f64,
    /// Initial step size.
    a_init: f64,
    /// Current step size (may be halved by divergence guard).
    a: f64,
    /// Initial perturbation magnitude (calibrated from noise).
    c_init: f64,
    /// Per-dimension perturbation floor.
    c_floor: [f64; 2],
    /// Local iteration counter (reset on regime change).
    k_local: u64,
    /// Stability constant for gain sequence.
    big_a: f64,
    /// Current SPSA phase.
    phase: SPSAPhase,
    /// Bernoulli +/-1 perturbation vector.
    current_delta: [f64; 2],
    /// Performance recorded during PerturbPlus phase.
    perf_plus: f64,
    /// Performance recorded during PerturbMinus phase.
    perf_minus: f64,
    /// Samples accumulated in the current phase.
    samples_in_phase: u64,
    // --- Regime detection (CUSUM) ---
    /// CUSUM statistic for regime change detection.
    cusum_s: f64,
    /// Baseline performance for CUSUM.
    perf_ewma_baseline: f64,
    /// Running variance of performance.
    perf_variance: f64,
    // --- Config tracking for delta emission ---
    /// Last theta used for adjustment emission.
    last_emitted_theta: [f64; 2],
    /// Total SPSA steps completed.
    total_steps: u64,
    /// xorshift64 RNG state.
    rng_state: u64,

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
}

impl DiagnosticLearner {
    /// Create a new learner backed by a feasible region with default objective.
    pub fn new(region: FeasibleRegion) -> Self {
        Self::with_objective(region, MetaObjective::default())
    }

    /// Create a new learner backed by a feasible region with a specific objective.
    pub fn with_objective(region: FeasibleRegion, objective: MetaObjective) -> Self {
        let bounds = region.config_bounds();

        // Observation interval: center of grace_period range, minimum 1.
        let observation_interval =
            ((bounds.grace_period.0 + bounds.grace_period.1) / 2).max(1) as u64;

        // SPSA gain sequence parameters.
        let big_a = 10.0;
        let a_init = 0.05 * (big_a + 1.0_f64).powf(0.602);
        let c_floor = [0.001; 2];

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
            observation_interval,
            theta: [0.5, 0.5],
            theta_best: [0.5, 0.5],
            best_performance: f64::NEG_INFINITY,
            a_init,
            a: a_init,
            c_init: 0.1,
            c_floor,
            k_local: 0,
            big_a,
            phase: SPSAPhase::Init,
            current_delta: [0.0; 2],
            perf_plus: 0.0,
            perf_minus: 0.0,
            samples_in_phase: 0,
            cusum_s: 0.0,
            perf_ewma_baseline: 0.0,
            perf_variance: 0.0,
            last_emitted_theta: [0.5, 0.5],
            total_steps: 0,
            rng_state: 0xDEAD_BEEF_CAFE_1234,
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
        }
    }

    /// Process diagnostics after each `train_one()`. Returns smooth adjustments.
    ///
    /// The SPSA optimizer cycles through Init -> PerturbPlus -> PerturbMinus
    /// phases, estimating gradients from paired performance evaluations and
    /// updating theta (normalized config parameters) accordingly.
    /// During the first 50 samples, only baselines are updated (no adjustments).
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

        // 3. Increment phase sample counter.
        self.samples_in_phase += 1;

        // 4. SPSA phase state machine.
        let no_op = SmoothAdjustments {
            lr_multiplier: 1.0,
            lambda_direction: 0.0,
        };

        match self.phase {
            SPSAPhase::Init => {
                // Accumulate performance variance for c_init calibration.
                let perf = self.current_performance();
                self.perf_variance =
                    0.01 * (perf - self.perf_ewma_baseline).powi(2) + 0.99 * self.perf_variance;
                self.perf_ewma_baseline = 0.01 * perf + 0.99 * self.perf_ewma_baseline;

                if self.samples_in_phase >= 50 {
                    // Calibrate c_init from observed noise.
                    let noise_std = self.perf_variance.sqrt();
                    self.c_init = (2.0 * noise_std).clamp(0.01, 0.3);
                    self.initialized = true;

                    // Generate first perturbation and transition to PerturbPlus.
                    self.generate_delta();
                    self.phase = SPSAPhase::PerturbPlus;
                    self.samples_in_phase = 0;

                    // Apply theta + c*delta config.
                    let target_theta = self.perturbed_theta(1.0);
                    return self.adjustment_for_theta(&target_theta);
                }
                no_op
            }

            SPSAPhase::PerturbPlus => {
                if self.samples_in_phase >= self.observation_interval {
                    // Record performance under theta + c*delta.
                    self.perf_plus = self.current_performance();

                    // Transition to PerturbMinus: apply theta - c*delta.
                    self.phase = SPSAPhase::PerturbMinus;
                    self.samples_in_phase = 0;

                    let target_theta = self.perturbed_theta(-1.0);
                    return self.adjustment_for_theta(&target_theta);
                }
                no_op
            }

            SPSAPhase::PerturbMinus => {
                if self.samples_in_phase >= self.observation_interval {
                    // Record performance under theta - c*delta.
                    self.perf_minus = self.current_performance();

                    // Do SPSA gradient update.
                    self.do_spsa_update();

                    // Transition back to PerturbPlus with new delta.
                    self.generate_delta();
                    self.phase = SPSAPhase::PerturbPlus;
                    self.samples_in_phase = 0;

                    // Apply new theta + c*delta config.
                    let target_theta = self.perturbed_theta(1.0);
                    return self.adjustment_for_theta(&target_theta);
                }
                no_op
            }
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

    /// Reset the learner state (all EWMA baselines, SPSA state, and performance trackers).
    pub fn reset(&mut self) {
        self.uncertainty_ewma = 0.0;
        self.alignment_ewma = 0.0;
        self.reg_sensitivity_ewma = 0.0;
        self.depth_signal_ewma = 0.0;
        self.dof_ewma = 0.0;
        self.n_samples = 0;
        self.initialized = false;
        // SPSA state
        self.theta = [0.5, 0.5];
        self.theta_best = [0.5, 0.5];
        self.best_performance = f64::NEG_INFINITY;
        self.a = self.a_init;
        self.c_init = 0.1;
        self.k_local = 0;
        self.phase = SPSAPhase::Init;
        self.current_delta = [0.0; 2];
        self.perf_plus = 0.0;
        self.perf_minus = 0.0;
        self.samples_in_phase = 0;
        self.cusum_s = 0.0;
        self.perf_ewma_baseline = 0.0;
        self.perf_variance = 0.0;
        self.last_emitted_theta = [0.5, 0.5];
        self.total_steps = 0;
        self.rng_state = 0xDEAD_BEEF_CAFE_1234;
        // Performance trackers
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
    }

    /// Current feasible region.
    pub fn region(&self) -> &FeasibleRegion {
        &self.region
    }

    /// Total SPSA optimization steps completed.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Current optimization objective.
    pub fn objective(&self) -> MetaObjective {
        self.objective
    }

    /// Current SPSA phase (for testing).
    #[cfg(test)]
    fn phase(&self) -> SPSAPhase {
        self.phase
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

    /// Perform the SPSA gradient update after both perturbation evaluations.
    fn do_spsa_update(&mut self) {
        let a_k = self.a / (self.big_a + self.k_local as f64 + 1.0).powf(0.602);
        let c_k_base = self.c_init / (self.k_local as f64 + 1.0).powf(0.101);

        for i in 0..2 {
            let c_k = c_k_base.max(self.c_floor[i]);
            if self.current_delta[i].abs() > 0.5 {
                let g_hat =
                    (self.perf_plus - self.perf_minus) / (2.0 * c_k * self.current_delta[i]);
                self.theta[i] += a_k * g_hat; // MAXIMIZE
                self.theta[i] = self.theta[i].clamp(0.0, 1.0);
            }
        }

        // Ito-Dhaene divergence guard.
        if self.perf_plus < self.best_performance && self.perf_minus < self.best_performance {
            self.a *= 0.5;
            self.theta = self.theta_best;
        } else {
            let best = self.perf_plus.max(self.perf_minus);
            if best > self.best_performance {
                self.best_performance = best;
                self.theta_best = self.theta;
            }
        }

        // CUSUM regime detection.
        let drift_margin = 0.5 * self.perf_variance.sqrt();
        let drift_threshold = 5.0 * self.perf_variance.sqrt();
        let current = self.current_performance();
        self.cusum_s = (self.cusum_s + (self.perf_ewma_baseline - current) - drift_margin).max(0.0);
        if self.cusum_s > drift_threshold && drift_threshold > 1e-15 {
            self.k_local = 0;
            self.a = self.a_init;
            self.cusum_s = 0.0;
            self.perf_ewma_baseline = current;
        }

        self.k_local += 1;
        self.total_steps += 1;

        // Update variance tracking.
        let perf = self.current_performance();
        self.perf_variance =
            0.01 * (perf - self.perf_ewma_baseline).powi(2) + 0.99 * self.perf_variance;
        self.perf_ewma_baseline = 0.01 * perf + 0.99 * self.perf_ewma_baseline;
    }

    /// Convert normalized theta to actual (lr, lambda) config values.
    fn theta_to_config(&self, theta: &[f64; 2]) -> (f64, f64) {
        let bounds = self.region.config_bounds();
        let lr = bounds.learning_rate.0
            * (bounds.learning_rate.1 / bounds.learning_rate.0.max(1e-15)).powf(theta[0]);
        let lambda = bounds.lambda.0 + theta[1] * (bounds.lambda.1 - bounds.lambda.0);
        (lr.max(1e-10), lambda.max(0.0))
    }

    /// Compute the adjustment to move from last_emitted_theta to the target theta.
    fn adjustment_for_theta(&mut self, target: &[f64; 2]) -> SmoothAdjustments {
        let (target_lr, target_lambda) = self.theta_to_config(target);
        let (last_lr, last_lambda) = self.theta_to_config(&self.last_emitted_theta);
        self.last_emitted_theta = *target;
        SmoothAdjustments {
            lr_multiplier: target_lr / last_lr.max(1e-15),
            lambda_direction: target_lambda - last_lambda,
        }
    }

    /// Compute theta perturbed by `sign * c_k * delta`, clamped to [0, 1].
    fn perturbed_theta(&self, sign: f64) -> [f64; 2] {
        let c_k_base = self.c_init / (self.k_local as f64 + 1.0).powf(0.101);
        let mut result = [0.0; 2];
        for (i, val) in result.iter_mut().enumerate() {
            let c_k = c_k_base.max(self.c_floor[i]);
            *val = (self.theta[i] + sign * c_k * self.current_delta[i]).clamp(0.0, 1.0);
        }
        result
    }

    /// Generate Bernoulli +/-1 perturbation using xorshift64.
    fn generate_delta(&mut self) {
        for d in &mut self.current_delta {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            *d = if self.rng_state % 2 == 0 { 1.0 } else { -1.0 };
        }
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
    // DiagnosticLearner tests (SPSA)
    // =======================================================================

    #[test]
    fn diagnostic_learner_init_phase_no_adjustments() {
        // No adjustments during Init phase (first 50 samples).
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let mut learner = DiagnosticLearner::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            regularization_sensitivity: 1.0,
            depth_sufficiency: 0.5,
            effective_dof: 10.0,
            uncertainty: 0.1,
        };

        // First 49 samples: all should be no-op.
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

        assert_eq!(
            learner.phase(),
            SPSAPhase::Init,
            "should still be in Init phase after 49 samples"
        );
    }

    #[test]
    fn diagnostic_learner_phase_cycling() {
        // Verify Init -> PerturbPlus -> PerturbMinus -> PerturbPlus...
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

        // Init phase: 50 samples.
        for i in 0..50 {
            learner.after_train(&diag, i as f64 * 0.01, i as f64 * 0.01 + 0.1);
        }
        // After 50 samples, should transition to PerturbPlus.
        assert_eq!(
            learner.phase(),
            SPSAPhase::PerturbPlus,
            "should be PerturbPlus after init phase"
        );

        // Feed observation_interval samples to complete PerturbPlus.
        for i in 0..interval {
            let idx = 50 + i;
            learner.after_train(&diag, idx as f64 * 0.01, idx as f64 * 0.01 + 0.1);
        }
        assert_eq!(
            learner.phase(),
            SPSAPhase::PerturbMinus,
            "should be PerturbMinus after completing PerturbPlus"
        );

        // Feed observation_interval samples to complete PerturbMinus.
        for i in 0..interval {
            let idx = 50 + interval + i;
            learner.after_train(&diag, idx as f64 * 0.01, idx as f64 * 0.01 + 0.1);
        }
        assert_eq!(
            learner.phase(),
            SPSAPhase::PerturbPlus,
            "should cycle back to PerturbPlus after PerturbMinus"
        );
    }

    #[test]
    fn diagnostic_learner_theta_bounds_clamping() {
        // Theta must stay in [0, 1] even with many SPSA iterations.
        let region = FeasibleRegion::from_data(10_000, 5, 1.0);
        let mut learner = DiagnosticLearner::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.9,
            regularization_sensitivity: 0.1,
            depth_sufficiency: 0.5,
            effective_dof: 10.0,
            uncertainty: 0.1,
        };

        for i in 0..5_000 {
            let pred = i as f64 * 0.01;
            let target = pred + 0.01;
            learner.after_train(&diag, pred, target);
        }

        // Theta must be within [0, 1].
        assert!(
            learner.theta[0] >= 0.0 && learner.theta[0] <= 1.0,
            "theta[0] must be in [0, 1], got {}",
            learner.theta[0]
        );
        assert!(
            learner.theta[1] >= 0.0 && learner.theta[1] <= 1.0,
            "theta[1] must be in [0, 1], got {}",
            learner.theta[1]
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

        // After reset, total_steps should be 0.
        assert_eq!(
            learner.total_steps(),
            0,
            "total_steps should be 0 after reset"
        );

        // After reset, phase should be Init.
        assert_eq!(
            learner.phase(),
            SPSAPhase::Init,
            "phase should be Init after reset"
        );

        // After reset, theta should be [0.5, 0.5].
        assert_eq!(
            learner.theta,
            [0.5, 0.5],
            "theta should be [0.5, 0.5] after reset"
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
    // SPSA convergence and bounds tests
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
    fn spsa_bounded_total_adjustment() {
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
    }
}
