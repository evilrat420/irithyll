//! Core AutoTuner implementation: train, predict, reset, and lifecycle.

use super::{AutoTunerConfig, Challenger};
use crate::automl::auto_builder;
use crate::automl::budget::BudgetLedger;
use crate::automl::space::ParamMap;
use crate::automl::{ModelFactory, RewardNormalizer};
use crate::bandits::{Bandit, DiscountedThompsonSampling};
use crate::metrics::ewma::EwmaRegressionMetrics;
use irithyll_core::drift::{DriftDetector, DriftSignal};
use irithyll_core::learner::StreamingLearner;

/// Streaming AutoML orchestrator with tournament successive halving.
///
/// The `AutoTuner` implements [`StreamingLearner`] and automatically tunes
/// hyperparameters across one or more model factories. It maintains a champion
/// model that always provides predictions, while candidates compete in
/// successive halving tournaments.
///
/// # Architecture
///
/// 1. **Champion**: The current best model. Always used for `predict()`.
/// 2. **Tournament**: `n_initial` candidates in a successive halving bracket.
/// 3. **Elimination**: After `round_budget` samples, the bottom half is eliminated.
/// 4. **Finals**: The lone finalist is compared to champion; promoted if better.
/// 5. **Multi-factory**: A [`DiscountedThompsonSampling`] bandit selects which
///    factory to sample from for random candidates.
/// 6. **Warm-start**: Half of new candidates are perturbations of the champion's
///    config; the other half are random from bandit-guided factory selection.
///
/// # Example
///
/// ```no_run
/// use irithyll::{auto_tune, automl::Factory, StreamingLearner};
///
/// let mut tuner = auto_tune(Factory::sgbt(5));
/// for i in 0..500 {
///     let x = [i as f64, (i as f64).sin(), (i as f64).cos(), 0.5, -0.3];
///     let y = x[0] * 0.1 + x[1] * 2.0;
///     tuner.train(&x, y);
/// }
/// let pred = tuner.predict(&[500.0, 500.0_f64.sin(), 500.0_f64.cos(), 0.5, -0.3]);
/// assert!(pred.is_finite());
/// ```
pub struct AutoTuner {
    pub(crate) champion: Box<dyn StreamingLearner>,
    pub(crate) champion_ewma: EwmaRegressionMetrics,
    pub(crate) champion_params: ParamMap,
    pub(crate) champion_factory_idx: usize,

    // Tournament state
    pub(crate) candidates: Vec<Challenger>,
    pub(crate) current_round: usize,
    pub(crate) samples_in_round: u64,

    // Multi-factory: per-factory RNG state for SearchSpace sampling.
    // One independent xorshift64 stream per factory keeps factory draws
    // reproducible without coupling factories' random sequences.
    pub(crate) factories: Vec<Box<dyn ModelFactory>>,
    pub(crate) sampler_rngs: Vec<u64>,

    pub(crate) bandit: DiscountedThompsonSampling,
    pub(crate) normalizer: RewardNormalizer,
    pub(crate) config: AutoTunerConfig,

    pub(crate) total_samples: u64,
    pub(crate) promotions: u64,
    pub(crate) tournaments_completed: u64,

    // Adaptive bracket sizing
    pub(crate) effective_n_initial: usize,

    // Optional drift detector for triggering re-racing
    pub(crate) drift_detector: Option<Box<dyn DriftDetector>>,

    /// Optional diagnostic learner (active when auto_builder=true).
    pub(crate) adaptor: Option<auto_builder::DiagnosticLearner>,

    /// Last observed replacement count from champion (for detecting boundaries).
    pub(crate) last_replacement_count: u64,

    /// Samples × complexity budget ledger for the current tournament.
    ///
    /// Tracks per-arm sample counts weighted by complexity so that the
    /// scheduler can prune arms that have exhausted their fair-share budget
    /// using an explicit [`crate::automl::budget::BudgetStatus`] — not a
    /// soft penalty.  Reset at each tournament start.
    pub(crate) budget_ledger: BudgetLedger,
}

// ===========================================================================
// AutoTuner public methods
// ===========================================================================

impl AutoTuner {
    /// Number of times a challenger was promoted to champion.
    pub fn promotions(&self) -> u64 {
        self.promotions
    }

    /// Number of tournaments fully completed.
    pub fn tournaments_completed(&self) -> u64 {
        self.tournaments_completed
    }

    /// Total samples processed.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Names of all registered factories.
    pub fn factory_names(&self) -> Vec<&str> {
        self.factories.iter().map(|f| f.name()).collect()
    }

    /// Number of candidates remaining in the current tournament round.
    pub fn candidates_remaining(&self) -> usize {
        self.candidates.len()
    }

    /// Current tournament round (0-indexed).
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Current adaptive bracket size (may differ from configured `n_initial`).
    pub fn effective_n_initial(&self) -> usize {
        self.effective_n_initial
    }

    /// Delegate proactive prune check to the champion model.
    #[allow(deprecated)]
    pub fn check_proactive_prune(&mut self) -> bool {
        self.champion.check_proactive_prune()
    }

    /// Delegate prune half-life update to the champion model.
    #[allow(deprecated)]
    pub fn set_prune_half_life(&mut self, hl: usize) {
        self.champion.set_prune_half_life(hl);
    }

    /// Take a diagnostic snapshot of the current state.
    ///
    /// Returns a complete picture of the champion, all candidates,
    /// tournament progress, and historical statistics.
    pub fn snapshot(&self) -> super::AutoTunerSnapshot {
        super::scheduler::snapshot_impl(self)
    }

    /// Spawn a new tournament (internal use by builder).
    pub(crate) fn start_tournament(&mut self) {
        super::scheduler::start_tournament(self);
    }
}

// ===========================================================================
// StreamingLearner implementation
// ===========================================================================

impl StreamingLearner for AutoTuner {
    #[allow(deprecated)]
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // 1. Champion: predict -> EWMA -> train.
        let champ_pred = self.champion.predict(features);
        self.champion_ewma.update(target, champ_pred);
        self.champion.train_one(features, target, weight);

        // 1b. Drift detection on champion error.
        let mut drift_restart = false;
        if let Some(ref mut detector) = self.drift_detector {
            let error = (target - champ_pred).abs();
            let signal = detector.update(error);
            if matches!(signal, DriftSignal::Drift) {
                drift_restart = true;
            }
        }

        // 2. All candidates: predict -> EWMA -> train + Welford error stats + budget.
        for c in &mut self.candidates {
            let pred = c.model.predict(features);
            c.ewma.update(target, pred);
            c.model.train_one(features, target, weight);

            // Welford's online update for error statistics.
            let error = (target - pred).abs();
            c.err_count += 1;
            let delta = error - c.err_mean;
            c.err_mean += delta / c.err_count as f64;
            let delta2 = error - c.err_mean;
            c.err_m2 += delta * delta2;

            // AM-4: Record sample in budget ledger for this arm.
            self.budget_ledger.record_sample(c.budget_idx);
        }

        // 3. Increment counters.
        self.samples_in_round += 1;
        self.total_samples += 1;

        // 4. Statistical early elimination check (every round_budget/4 samples).
        let check_interval = (self.config.round_budget / 4).max(1) as u64;
        if self.samples_in_round > 0
            && self.samples_in_round % check_interval == 0
            && self.samples_in_round < self.config.round_budget as u64
        {
            super::racing::try_early_elimination(self);
        }

        // 5. Elimination check.
        if self.samples_in_round >= self.config.round_budget as u64 {
            super::scheduler::eliminate_round(self);
        }

        // 6. Auto-builder diagnostic adaptation.
        if let Some(ref mut adaptor) = self.adaptor {
            // Extract full diagnostics from the champion via StreamingLearner.
            let arr = self.champion.diagnostics_array();
            let diagnostics = auto_builder::ConfigDiagnostics {
                residual_alignment: arr[0],
                regularization_sensitivity: arr[1],
                depth_sufficiency: arr[2],
                effective_dof: arr[3],
                uncertainty: if arr[4] > 0.0 {
                    arr[4]
                } else {
                    super::racing::get_metric(&self.champion_ewma, self.config.metric)
                },
            };
            let adjustments = adaptor.after_train(&diagnostics, champ_pred, target);

            // Apply smooth adjustments to the champion's config.
            if (adjustments.lr_multiplier - 1.0).abs() > 1e-15
                || adjustments.lambda_direction.abs() > 1e-15
            {
                self.champion
                    .adjust_config(adjustments.lr_multiplier, adjustments.lambda_direction);
            }

            // Check for structural boundary: champion's replacement count changed.
            let current_rc = self.champion.replacement_count();
            if current_rc > self.last_replacement_count {
                self.last_replacement_count = current_rc;
                if let Some(change) = adaptor.at_replacement(&diagnostics) {
                    if change.depth_delta != 0 || change.steps_delta != 0 {
                        self.champion
                            .apply_structural_change(change.depth_delta, change.steps_delta);
                    }
                }
            }
        }

        // 7. Drift-triggered re-racing: abort tournament and start fresh.
        if drift_restart {
            self.candidates.clear();
            self.effective_n_initial =
                (self.effective_n_initial * 2).min(self.config.max_n_initial);
            super::scheduler::start_tournament(self);
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        self.champion.predict(features)
    }

    fn n_samples_seen(&self) -> u64 {
        self.total_samples
    }

    fn reset(&mut self) {
        self.champion.reset();
        self.champion_ewma.reset();
        self.candidates.clear();
        self.budget_ledger.reset();
        self.total_samples = 0;
        self.promotions = 0;
        self.tournaments_completed = 0;
        self.samples_in_round = 0;
        self.current_round = 0;
        self.effective_n_initial = self.config.n_initial;
        self.bandit.reset(); // Bandit trait method
        self.normalizer.reset();
        if let Some(ref mut d) = self.drift_detector {
            d.reset();
        }
        if self.config.auto_builder {
            let first_supports = self
                .factories
                .first()
                .map(|f| f.supports_auto_builder())
                .unwrap_or(false);
            if first_supports {
                let n_feat = self
                    .factories
                    .first()
                    .map(|f| f.n_features_hint())
                    .unwrap_or(1)
                    .max(1);
                let region = auto_builder::FeasibleRegion::from_data(100, n_feat, 1.0);
                self.adaptor = Some(auto_builder::DiagnosticLearner::with_objective(
                    region,
                    self.config.meta_objective,
                ));
            } else {
                self.adaptor = None;
            }
        } else {
            self.adaptor = None;
        }
        self.last_replacement_count = 0;
        super::scheduler::start_tournament(self);
    }

    #[allow(deprecated)]
    fn tree_structure(&self) -> Vec<(usize, usize, f64, f64, u64)> {
        self.champion.tree_structure()
    }

    // PR-AM-8: Delegate non-pure-default StreamingLearner methods to the
    // champion so that AutoTuner-as-wrapper preserves full capability.
    // Without these, nesting an AutoTuner inside another wrapper silently
    // loses adjust_config / structural / readout access to the inner model.

    #[allow(deprecated)]
    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64) {
        self.champion.adjust_config(lr_multiplier, lambda_delta);
    }

    #[allow(deprecated)]
    fn apply_structural_change(&mut self, depth_delta: i32, steps_delta: i32) {
        self.champion
            .apply_structural_change(depth_delta, steps_delta);
    }

    #[allow(deprecated)]
    fn replacement_count(&self) -> u64 {
        self.champion.replacement_count()
    }

    #[allow(deprecated)]
    fn readout_weights(&self) -> Option<&[f64]> {
        self.champion.readout_weights()
    }
}

// ===========================================================================
// DiagnosticSource impl
// ===========================================================================

impl crate::automl::DiagnosticSource for AutoTuner {
    #[allow(deprecated)]
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        let arr = self.champion.diagnostics_array();
        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: arr[0],
            regularization_sensitivity: arr[1],
            depth_sufficiency: arr[2],
            effective_dof: arr[3],
            uncertainty: if arr[4] > 0.0 {
                arr[4]
            } else {
                super::racing::get_metric(&self.champion_ewma, self.config.metric)
            },
        })
    }
}
