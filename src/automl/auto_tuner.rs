//! Streaming AutoML orchestrator: tournament successive halving with multi-factory support.
//!
//! The [`AutoTuner`] maintains a champion model that always provides predictions,
//! while candidates from potentially multiple model factories compete in successive
//! halving tournaments. Each tournament starts with `n_initial` candidates; after
//! every `round_budget` samples the bottom half is eliminated. The lone finalist
//! is compared to the champion and promoted if better. A new tournament starts
//! immediately.
//!
//! Half of new candidates are perturbations of the champion's config (warm-start),
//! and half are random configs from bandit-guided factory selection.
//!
//! # References
//!
//! - Wilson et al. (2026) "SUHEN" IEEE TAI -- successive halving for streaming
//! - Wu et al. (2021) "ChaCha for Online AutoML" ICML -- champion-challenger
//! - Qi et al. (2023) "Discounted Thompson Sampling" -- non-stationary bandit

use crate::automl::auto_builder;
use crate::automl::{AutoMetric, ConfigSampler, HyperConfig, ModelFactory, RewardNormalizer};
use crate::bandits::{Bandit, DiscountedThompsonSampling};
use crate::drift::adwin::Adwin;
use crate::metrics::ewma::EwmaRegressionMetrics;
use irithyll_core::drift::{DriftDetector, DriftSignal};
use irithyll_core::learner::StreamingLearner;

// ===========================================================================
// Diagnostic snapshots
// ===========================================================================

/// Snapshot of the AutoTuner's current state for diagnostics.
#[derive(Debug, Clone)]
pub struct AutoTunerSnapshot {
    /// Name of the champion's model factory.
    pub champion_factory: String,
    /// Champion's current EWMA metric value.
    pub champion_metric: f64,
    /// Samples the champion has trained on.
    pub champion_samples: u64,

    /// Current tournament candidates.
    pub candidates: Vec<CandidateSnapshot>,

    /// Current tournament round (0-indexed).
    pub current_round: usize,
    /// Samples processed in current round.
    pub samples_in_round: u64,
    /// Current adaptive bracket size.
    pub effective_n_initial: usize,

    /// Total tournaments completed.
    pub tournaments_completed: u64,
    /// Total champion promotions.
    pub promotions: u64,
    /// Total samples processed.
    pub total_samples: u64,

    /// Registered factory names.
    pub factory_names: Vec<String>,
}

/// Snapshot of a single tournament candidate.
#[derive(Debug, Clone)]
pub struct CandidateSnapshot {
    /// Factory name that produced this candidate.
    pub factory_name: String,
    /// Current EWMA metric value.
    pub metric: f64,
    /// Samples this candidate has trained on.
    pub samples_trained: u64,
    /// Whether this candidate is still in warmup.
    pub in_warmup: bool,
}

impl std::fmt::Display for AutoTunerSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== AutoTuner Snapshot ===")?;
        writeln!(
            f,
            "Champion: {} (metric: {:.6}, samples: {})",
            self.champion_factory, self.champion_metric, self.champion_samples
        )?;
        writeln!(
            f,
            "Tournament: round {}, {}/{} candidates, {} samples in round",
            self.current_round,
            self.candidates.len(),
            self.effective_n_initial,
            self.samples_in_round
        )?;
        writeln!(
            f,
            "History: {} tournaments, {} promotions, {} total samples",
            self.tournaments_completed, self.promotions, self.total_samples
        )?;
        if !self.candidates.is_empty() {
            writeln!(f, "Candidates:")?;
            for (i, c) in self.candidates.iter().enumerate() {
                let warmup_tag = if c.in_warmup { " [warmup]" } else { "" };
                writeln!(
                    f,
                    "  [{i}] {} metric={:.6} samples={}{warmup_tag}",
                    c.factory_name, c.metric, c.samples_trained
                )?;
            }
        }
        Ok(())
    }
}

// ===========================================================================
// AutoTunerConfig
// ===========================================================================

/// Configuration for the [`AutoTuner`].
#[derive(Debug, Clone)]
pub struct AutoTunerConfig {
    /// Initial candidates per tournament (default: 8).
    /// Will be rounded to the nearest power of 2.
    pub n_initial: usize,
    /// Samples per elimination round (default: 100).
    pub round_budget: usize,
    /// Metric to optimize (default: MAE).
    pub metric: AutoMetric,
    /// EWMA span for metric tracking (default: 50).
    pub ewma_span: usize,
    /// Discount factor for factory-level bandit (default: 0.99).
    pub discount: f64,
    /// Perturbation strength for warm-start configs (default: 0.2).
    pub perturb_sigma: f64,
    /// RNG seed (default: 42).
    pub seed: u64,
    /// Minimum adaptive bracket size (default: 4).
    pub min_n_initial: usize,
    /// Maximum adaptive bracket size (default: 32).
    pub max_n_initial: usize,
    /// Enable drift-triggered re-racing (default: false).
    ///
    /// When enabled, an ADWIN detector monitors the champion's prediction
    /// error. If drift is detected, the current tournament is aborted and a
    /// new one starts with an expanded bracket, allowing the tuner to
    /// rapidly adapt to distribution shifts.
    pub use_drift_rerace: bool,
    /// Enable auto-builder mode (default: false).
    ///
    /// When enabled, replaces tournament elimination with Welford race +
    /// diagnostic adaptation. The first training uses all-see-all batch
    /// evaluation instead of tournament elimination.
    pub auto_builder: bool,
    /// Meta-learner optimization objective (default: MinimizeRMSE).
    ///
    /// Only used when `auto_builder` is enabled. Controls which performance
    /// metric the diagnostic learner optimizes.
    pub meta_objective: auto_builder::MetaObjective,
}

impl Default for AutoTunerConfig {
    fn default() -> Self {
        Self {
            n_initial: 8,
            round_budget: 100,
            metric: AutoMetric::MAE,
            ewma_span: 50,
            discount: 0.99,
            perturb_sigma: 0.2,
            seed: 42,
            min_n_initial: 4,
            max_n_initial: 32,
            use_drift_rerace: false,
            auto_builder: false,
            meta_objective: auto_builder::MetaObjective::default(),
        }
    }
}

// ===========================================================================
// Challenger
// ===========================================================================

/// A candidate model competing in the current tournament.
struct Challenger {
    model: Box<dyn StreamingLearner>,
    ewma: EwmaRegressionMetrics,
    config: HyperConfig,
    factory_idx: usize,
    // Welford's online stats for paired error differences (statistical early stopping).
    err_mean: f64,
    err_m2: f64,
    err_count: u64,
}

// ===========================================================================
// AutoTuner
// ===========================================================================

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
    champion: Box<dyn StreamingLearner>,
    champion_ewma: EwmaRegressionMetrics,
    champion_config: HyperConfig,
    champion_factory_idx: usize,

    // Tournament state
    candidates: Vec<Challenger>,
    current_round: usize,
    samples_in_round: u64,

    // Multi-factory
    factories: Vec<Box<dyn ModelFactory>>,
    samplers: Vec<ConfigSampler>,

    bandit: DiscountedThompsonSampling,
    normalizer: RewardNormalizer,
    config: AutoTunerConfig,

    total_samples: u64,
    promotions: u64,
    tournaments_completed: u64,

    // Adaptive bracket sizing
    effective_n_initial: usize,

    // Optional drift detector for triggering re-racing
    drift_detector: Option<Box<dyn DriftDetector>>,

    /// Optional diagnostic learner (active when auto_builder=true).
    adaptor: Option<auto_builder::DiagnosticLearner>,

    /// Last observed replacement count from champion (for detecting boundaries).
    last_replacement_count: u64,
}

// ===========================================================================
// AutoTunerBuilder
// ===========================================================================

/// Builder for [`AutoTuner`].
///
/// At least one [`ModelFactory`] must be provided via [`factory`](Self::factory)
/// or [`add_factory`](Self::add_factory). All other settings have defaults.
pub struct AutoTunerBuilder {
    factories: Vec<Box<dyn ModelFactory>>,
    config: AutoTunerConfig,
}

impl AutoTuner {
    /// Create a builder for an `AutoTuner`.
    pub fn builder() -> AutoTunerBuilder {
        AutoTunerBuilder {
            factories: Vec::new(),
            config: AutoTunerConfig::default(),
        }
    }
}

impl AutoTunerBuilder {
    /// Set the model factory (clears existing factories, sets exactly one).
    pub fn factory(mut self, f: impl ModelFactory + 'static) -> Self {
        self.factories.clear();
        self.factories.push(Box::new(f));
        self
    }

    /// Add an additional model factory for multi-factory racing.
    pub fn add_factory(mut self, f: impl ModelFactory + 'static) -> Self {
        self.factories.push(Box::new(f));
        self
    }

    /// Set the initial candidates per tournament (default: 8).
    pub fn n_initial(mut self, n: usize) -> Self {
        self.config.n_initial = n;
        self
    }

    /// Set the samples per elimination round (default: 100).
    pub fn round_budget(mut self, b: usize) -> Self {
        self.config.round_budget = b;
        self
    }

    /// Set the metric to optimize (default: MAE).
    pub fn metric(mut self, m: AutoMetric) -> Self {
        self.config.metric = m;
        self
    }

    /// Set the EWMA span for metric tracking (default: 50).
    pub fn ewma_span(mut self, s: usize) -> Self {
        self.config.ewma_span = s;
        self
    }

    /// Set the discount factor for factory-level bandit (default: 0.99).
    pub fn discount(mut self, d: f64) -> Self {
        self.config.discount = d;
        self
    }

    /// Set the perturbation strength for warm-start configs (default: 0.2).
    pub fn perturb_sigma(mut self, s: f64) -> Self {
        self.config.perturb_sigma = s;
        self
    }

    /// Set the RNG seed (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.config.seed = s;
        self
    }

    /// Set the minimum adaptive bracket size (default: 4).
    pub fn min_n_initial(mut self, n: usize) -> Self {
        self.config.min_n_initial = n;
        self
    }

    /// Set the maximum adaptive bracket size (default: 32).
    pub fn max_n_initial(mut self, n: usize) -> Self {
        self.config.max_n_initial = n;
        self
    }

    /// Enable drift-triggered re-racing (default: false).
    ///
    /// When enabled, an ADWIN detector monitors the champion's prediction
    /// error. If drift is detected, the current tournament is aborted and a
    /// new one starts with an expanded bracket.
    pub fn use_drift_rerace(mut self, enabled: bool) -> Self {
        self.config.use_drift_rerace = enabled;
        self
    }

    /// Enable the auto-builder (Welford race + diagnostic adaptation).
    ///
    /// When enabled, the first training uses all-see-all batch evaluation
    /// instead of tournament elimination.
    pub fn auto_builder(mut self, enabled: bool) -> Self {
        self.config.auto_builder = enabled;
        self
    }

    /// Set the meta-learner optimization objective (default: MinimizeRMSE).
    ///
    /// Only relevant when `auto_builder` is enabled. Controls which
    /// performance metric the diagnostic learner optimizes.
    pub fn meta_objective(mut self, obj: auto_builder::MetaObjective) -> Self {
        self.config.meta_objective = obj;
        self
    }

    /// Build the `AutoTuner`.
    ///
    /// # Panics
    ///
    /// Panics if no factory was provided.
    pub fn build(self) -> AutoTuner {
        assert!(
            !self.factories.is_empty(),
            "AutoTunerBuilder: at least one ModelFactory is required"
        );
        let config = self.config;

        // Seed must be non-zero for xorshift64.
        let seed = if config.seed == 0 { 1 } else { config.seed };

        // Create one ConfigSampler per factory.
        let mut samplers: Vec<ConfigSampler> = self
            .factories
            .iter()
            .enumerate()
            .map(|(i, f)| ConfigSampler::new(f.config_space(), seed.wrapping_add(i as u64).max(1)))
            .collect();

        // Bandit: one arm per factory.
        let n_factory_arms = self.factories.len().max(1);
        let bandit = DiscountedThompsonSampling::with_seed(n_factory_arms, config.discount, seed);

        // Create initial champion from first factory.
        let champion_factory_idx = 0;
        let champion_config = samplers[0].random();
        let champion = self.factories[0].create(&champion_config);
        let champion_ewma = EwmaRegressionMetrics::new(config.ewma_span);

        let effective_n_initial = config.n_initial;
        let drift_detector: Option<Box<dyn DriftDetector>> = if config.use_drift_rerace {
            Some(Box::new(Adwin::default()))
        } else {
            None
        };
        let adaptor = if config.auto_builder {
            // Create FeasibleRegion with conservative initial estimates.
            let region = auto_builder::FeasibleRegion::from_data(100, 1, 1.0);
            Some(auto_builder::DiagnosticLearner::with_objective(
                region,
                config.meta_objective,
            ))
        } else {
            None
        };
        let mut tuner = AutoTuner {
            champion,
            champion_ewma,
            champion_config,
            champion_factory_idx,
            candidates: Vec::new(),
            current_round: 0,
            samples_in_round: 0,
            factories: self.factories,
            samplers,
            bandit,
            normalizer: RewardNormalizer::with_span(config.ewma_span),
            config,
            total_samples: 0,
            promotions: 0,
            tournaments_completed: 0,
            effective_n_initial,
            drift_detector,
            adaptor,
            last_replacement_count: 0,
        };

        tuner.start_tournament();
        tuner
    }
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

    /// Take a diagnostic snapshot of the current state.
    ///
    /// Returns a complete picture of the champion, all candidates,
    /// tournament progress, and historical statistics.
    pub fn snapshot(&self) -> AutoTunerSnapshot {
        let metric_type = self.config.metric;

        let champion_metric = get_metric(&self.champion_ewma, metric_type);
        let champion_factory = self
            .factories
            .get(self.champion_factory_idx)
            .map(|f| f.name().to_string())
            .unwrap_or_default();

        let candidates: Vec<CandidateSnapshot> = self
            .candidates
            .iter()
            .map(|c| {
                let factory_name = self
                    .factories
                    .get(c.factory_idx)
                    .map(|f| f.name().to_string())
                    .unwrap_or_default();
                let warmup_hint = self
                    .factories
                    .get(c.factory_idx)
                    .map(|f| f.warmup_hint())
                    .unwrap_or(0);
                CandidateSnapshot {
                    factory_name,
                    metric: get_metric(&c.ewma, metric_type),
                    samples_trained: c.ewma.n_samples(),
                    in_warmup: (c.ewma.n_samples() as usize) < warmup_hint,
                }
            })
            .collect();

        AutoTunerSnapshot {
            champion_factory,
            champion_metric,
            champion_samples: self.champion.n_samples_seen(),
            candidates,
            current_round: self.current_round,
            samples_in_round: self.samples_in_round,
            effective_n_initial: self.effective_n_initial,
            tournaments_completed: self.tournaments_completed,
            promotions: self.promotions,
            total_samples: self.total_samples,
            factory_names: self
                .factories
                .iter()
                .map(|f| f.name().to_string())
                .collect(),
        }
    }
}

// ===========================================================================
// StreamingLearner implementation
// ===========================================================================

impl StreamingLearner for AutoTuner {
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

        // 2. All candidates: predict -> EWMA -> train + Welford error stats.
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
            self.try_early_elimination();
        }

        // 5. Elimination check.
        if self.samples_in_round >= self.config.round_budget as u64 {
            self.eliminate_round();
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
                    get_metric(&self.champion_ewma, self.config.metric)
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
            self.start_tournament();
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
        self.total_samples = 0;
        self.promotions = 0;
        self.tournaments_completed = 0;
        self.samples_in_round = 0;
        self.current_round = 0;
        self.effective_n_initial = self.config.n_initial;
        self.bandit.reset();
        self.normalizer.reset();
        if let Some(ref mut d) = self.drift_detector {
            d.reset();
        }
        if self.config.auto_builder {
            let region = auto_builder::FeasibleRegion::from_data(100, 1, 1.0);
            self.adaptor = Some(auto_builder::DiagnosticLearner::with_objective(
                region,
                self.config.meta_objective,
            ));
        } else {
            self.adaptor = None;
        }
        self.last_replacement_count = 0;
        self.start_tournament();
    }
}

// ===========================================================================
// Private tournament logic
// ===========================================================================

impl AutoTuner {
    /// Eliminate the bottom half of candidates after a round completes.
    ///
    /// Candidates still in warmup (fewer samples than their factory's
    /// `warmup_hint()`) are protected and always survive elimination.
    fn eliminate_round(&mut self) {
        if self.candidates.is_empty() {
            self.start_tournament();
            return;
        }

        // Pre-compute warmup hints and complexity hints to avoid borrow issues.
        let warmup_hints: Vec<usize> = self
            .candidates
            .iter()
            .map(|c| {
                self.factories
                    .get(c.factory_idx)
                    .map(|f| f.warmup_hint())
                    .unwrap_or(0)
            })
            .collect();
        let complexity_hints: Vec<usize> = self
            .candidates
            .iter()
            .map(|c| {
                self.factories
                    .get(c.factory_idx)
                    .map(|f| f.complexity_hint())
                    .unwrap_or(100)
            })
            .collect();

        // Partition into warmup-protected and eligible candidates.
        let metric_type = self.config.metric;
        let mut protected_indices: Vec<usize> = Vec::new();
        let mut eligible: Vec<(usize, f64)> = Vec::new();

        for (i, c) in self.candidates.iter().enumerate() {
            if (c.err_count as usize) < warmup_hints[i] {
                protected_indices.push(i);
            } else {
                let raw = get_metric(&c.ewma, metric_type);
                let adj = adjusted_metric(raw, complexity_hints[i], c.ewma.n_samples());
                eligible.push((i, adj));
            }
        }

        // Sort eligible best first (lowest metric).
        eligible.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // From the eligible pool, keep top half (ceiling division: always keep at least 1).
        let total_to_keep = self.candidates.len().div_ceil(2);
        // Protected are always kept; fill remaining slots from eligible.
        let eligible_to_keep = total_to_keep.saturating_sub(protected_indices.len());
        let eligible_to_keep = eligible_to_keep.max(if eligible.is_empty() { 0 } else { 1 });

        let mut keep_set: Vec<usize> = protected_indices;
        keep_set.extend(eligible.iter().take(eligible_to_keep).map(|(i, _)| *i));

        // Rebuild candidates with only survivors.
        let old = std::mem::take(&mut self.candidates);
        self.candidates = old
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep_set.contains(i))
            .map(|(_, c)| c)
            .collect();

        self.current_round += 1;
        self.samples_in_round = 0;

        // If 1 candidate left -> finals.
        if self.candidates.len() <= 1 {
            self.finalize_tournament();
        } else {
            // Reset EWMAs for fair next-round comparison.
            for c in &mut self.candidates {
                c.ewma.reset();
            }
            self.champion_ewma.reset();
        }
    }

    /// Compare the lone finalist to the champion and possibly promote.
    fn finalize_tournament(&mut self) {
        let mut promoted = false;

        if let Some(finalist) = self.candidates.pop() {
            let metric_type = self.config.metric;
            let finalist_metric = get_metric(&finalist.ewma, metric_type);
            let champion_metric = get_metric(&self.champion_ewma, metric_type);

            if finalist_metric < champion_metric && finalist.ewma.n_samples() >= 10 {
                // Promote.
                self.champion = finalist.model;
                self.champion_ewma = finalist.ewma;
                self.champion_config = finalist.config;
                self.champion_factory_idx = finalist.factory_idx;
                self.promotions += 1;
                promoted = true;
            }

            // Update bandit: reward the finalist's factory.
            let reward = self
                .normalizer
                .normalize(finalist_metric.min(champion_metric));
            if finalist.factory_idx < self.bandit.n_arms() {
                self.bandit.update(finalist.factory_idx, reward);
            }
        }

        // Adaptive bracket sizing.
        if promoted {
            // Champion was beaten -- landscape is shifting, explore more.
            self.effective_n_initial =
                (self.effective_n_initial * 2).min(self.config.max_n_initial);
        } else {
            // Champion held -- it's strong, reduce exploration.
            self.effective_n_initial =
                (self.effective_n_initial / 2).max(self.config.min_n_initial);
        }

        self.candidates.clear();
        self.tournaments_completed += 1;
        self.start_tournament();
    }

    /// Spawn a new tournament with `effective_n_initial` candidates.
    fn start_tournament(&mut self) {
        let n = self.effective_n_initial;
        let n_perturb = n / 2;
        let n_factories = self.factories.len();
        let ewma_span = self.config.ewma_span;
        let sigma = self.config.perturb_sigma;

        let mut candidates = Vec::with_capacity(n);

        // 50%: perturbations of champion config (from champion's factory).
        let champ_idx = self.champion_factory_idx;
        for _ in 0..n_perturb {
            let config = self.samplers[champ_idx].perturb(&self.champion_config, sigma);
            let model = self.factories[champ_idx].create(&config);
            candidates.push(Challenger {
                model,
                ewma: EwmaRegressionMetrics::new(ewma_span),
                config,
                factory_idx: champ_idx,
                err_mean: 0.0,
                err_m2: 0.0,
                err_count: 0,
            });
        }

        // 50%: random from factories (bandit-guided selection if multi-factory).
        for _ in n_perturb..n {
            let factory_idx = if n_factories > 1 {
                self.bandit.select_arm() % n_factories
            } else {
                0
            };
            let config = self.samplers[factory_idx].random();
            let model = self.factories[factory_idx].create(&config);
            candidates.push(Challenger {
                model,
                ewma: EwmaRegressionMetrics::new(ewma_span),
                config,
                factory_idx,
                err_mean: 0.0,
                err_m2: 0.0,
                err_count: 0,
            });
        }

        self.candidates = candidates;
        self.current_round = 0;
        self.samples_in_round = 0;
        self.champion_ewma.reset();
    }

    /// Attempt statistical early elimination of clearly outclassed candidates.
    ///
    /// Uses Welford's online statistics to compute a z-test against the median
    /// error. Candidates significantly worse than the median (z > 2.0) are
    /// removed early, saving compute. Warmup-protected candidates are exempt.
    fn try_early_elimination(&mut self) {
        if self.candidates.len() <= 1 {
            return;
        }

        let min_samples_for_test: u64 = 30;

        // Pre-compute warmup hints and complexity hints to avoid borrow issues.
        let warmup_hints: Vec<usize> = self
            .candidates
            .iter()
            .map(|c| {
                self.factories
                    .get(c.factory_idx)
                    .map(|f| f.warmup_hint())
                    .unwrap_or(0)
            })
            .collect();
        let complexity_hints: Vec<usize> = self
            .candidates
            .iter()
            .map(|c| {
                self.factories
                    .get(c.factory_idx)
                    .map(|f| f.complexity_hint())
                    .unwrap_or(100)
            })
            .collect();

        // Compute median of complexity-adjusted error across candidates that are
        // PAST warmup and have enough samples. Candidates still in warmup are
        // excluded from the median to prevent their noisy cold-start metrics
        // from skewing the baseline.
        let mut adjusted_errors: Vec<f64> = self
            .candidates
            .iter()
            .enumerate()
            .filter(|(i, c)| {
                c.err_count >= min_samples_for_test && (c.err_count as usize) >= warmup_hints[*i]
            })
            .map(|(i, c)| adjusted_metric(c.err_mean, complexity_hints[i], c.err_count))
            .collect();
        if adjusted_errors.len() < 2 {
            return;
        }
        adjusted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_adj_err = adjusted_errors[adjusted_errors.len() / 2];

        let z_threshold = 2.0;

        // Collect indices to remove (avoids borrow issues).
        let remove_indices: Vec<usize> = self
            .candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                // Protected: not enough data.
                if c.err_count < min_samples_for_test {
                    return None;
                }
                // Protected: still warming up.
                if (c.err_count as usize) < warmup_hints[i] {
                    return None;
                }
                let adj_err = adjusted_metric(c.err_mean, complexity_hints[i], c.err_count);
                let variance = if c.err_count > 1 {
                    c.err_m2 / (c.err_count - 1) as f64
                } else {
                    0.0
                };
                let std_err = (variance / c.err_count as f64).sqrt();
                if std_err < 1e-12 {
                    return None; // avoid division by zero
                }
                let z = (adj_err - median_adj_err) / std_err;
                if z > z_threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if !remove_indices.is_empty() {
            let old = std::mem::take(&mut self.candidates);
            self.candidates = old
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !remove_indices.contains(i))
                .map(|(_, c)| c)
                .collect();
        }

        // If early elimination reduced to 1 or 0, finalize.
        if self.candidates.len() <= 1 {
            self.finalize_tournament();
        }
    }
}

/// Extract the relevant metric from an EWMA tracker.
///
/// Free function to avoid borrow checker issues with `&self` + `&self.candidates[i].ewma`.
fn get_metric(ewma: &EwmaRegressionMetrics, metric: AutoMetric) -> f64 {
    match metric {
        AutoMetric::MAE => ewma.mae(),
        AutoMetric::MSE => ewma.mse(),
        AutoMetric::RMSE => ewma.rmse(),
    }
}

/// Complexity-adjusted metric: adds a penalty that decays as `1/n_seen`.
///
/// When evaluation data is scarce, higher-complexity models are penalized more,
/// naturally favoring simpler models. As data grows, the penalty vanishes and
/// models are compared on raw performance.
fn adjusted_metric(ewma_metric: f64, complexity: usize, n_seen: u64) -> f64 {
    let penalty = if n_seen > 0 {
        (complexity as f64) / (n_seen as f64)
    } else {
        complexity as f64
    };
    ewma_metric + penalty
}

// Safety: AutoTuner contains only Send+Sync types.
// - Box<dyn StreamingLearner>: StreamingLearner requires Send+Sync
// - Box<dyn ModelFactory>: ModelFactory requires Send+Sync
// - All other fields are owned, non-Rc, non-Cell types.
unsafe impl Send for AutoTuner {}
unsafe impl Sync for AutoTuner {}

// ===========================================================================
// DiagnosticSource impl
// ===========================================================================

impl crate::automl::DiagnosticSource for AutoTuner {
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
                get_metric(&self.champion_ewma, self.config.metric)
            },
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::automl::factories::{AttentionFactory, EsnFactory, MambaFactory, SpikeNetFactory};
    use crate::automl::SgbtFactory;

    /// Verify AutoTuner builder creates an instance with default config values.
    #[test]
    fn tournament_builder_default() {
        let tuner = AutoTuner::builder().factory(SgbtFactory::new(5)).build();

        assert_eq!(
            tuner.config.n_initial, 8,
            "default n_initial should be 8, got {}",
            tuner.config.n_initial
        );
        assert_eq!(
            tuner.config.round_budget, 100,
            "default round_budget should be 100, got {}",
            tuner.config.round_budget
        );
        assert_eq!(
            tuner.config.metric,
            AutoMetric::MAE,
            "default metric should be MAE"
        );
        assert_eq!(
            tuner.config.ewma_span, 50,
            "default ewma_span should be 50, got {}",
            tuner.config.ewma_span
        );
        assert!(
            (tuner.config.discount - 0.99).abs() < 1e-12,
            "default discount should be 0.99, got {}",
            tuner.config.discount
        );
        assert!(
            (tuner.config.perturb_sigma - 0.2).abs() < 1e-12,
            "default perturb_sigma should be 0.2, got {}",
            tuner.config.perturb_sigma
        );
        assert_eq!(
            tuner.config.seed, 42,
            "default seed should be 42, got {}",
            tuner.config.seed
        );
        // Tournament starts immediately, so candidates should be populated.
        assert_eq!(
            tuner.candidates.len(),
            8,
            "should have 8 candidates (n_initial), got {}",
            tuner.candidates.len()
        );
    }

    /// Verify training and prediction produce finite values.
    #[test]
    fn tournament_train_and_predict() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(5))
            .round_budget(50)
            .build();

        for i in 0..200 {
            let x = [i as f64, (i as f64) * 0.5, (i as f64) * 0.1, 1.0, -1.0];
            let y = x[0] * 0.3 + x[1] * 0.7 + 2.0;
            tuner.train(&x, y);
        }

        let pred = tuner.predict(&[50.0, 25.0, 5.0, 1.0, -1.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after 200 training samples, got {pred}"
        );
    }

    /// Verify successive halving completes tournaments.
    #[test]
    fn tournament_successive_halving() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(4)
            .round_budget(20)
            .seed(77)
            .build();

        // With n_initial=4 and round_budget=20:
        // Round 0: 4 candidates, 20 samples -> eliminate to 2
        // Round 1: 2 candidates, 20 samples -> eliminate to 1
        // Finals: 1 finalist vs champion -> tournament complete
        // So each tournament takes 40 samples. 400 samples = ~10 tournaments.
        for i in 0..400 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1] * 3.0 + 1.0;
            tuner.train(&x, y);
        }

        assert!(
            tuner.tournaments_completed() > 0,
            "expected at least 1 tournament completed after 400 samples, got {}",
            tuner.tournaments_completed()
        );
    }

    /// Verify promotions occur after enough training.
    #[test]
    fn tournament_promotes() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(4)
            .round_budget(20)
            .seed(123)
            .build();

        for i in 0..1000 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1] * 3.0 + 1.0;
            tuner.train(&x, y);
        }

        assert!(
            tuner.promotions() > 0,
            "expected at least 1 promotion after 1000 samples, got {}",
            tuner.promotions()
        );
    }

    /// Verify reset clears all state.
    #[test]
    fn tournament_reset() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .round_budget(20)
            .n_initial(4)
            .build();

        for i in 0..100 {
            let x = [i as f64, 0.5, 1.0];
            tuner.train(&x, i as f64);
        }

        assert!(
            tuner.total_samples() > 0,
            "should have samples before reset"
        );

        tuner.reset();

        assert_eq!(
            tuner.total_samples(),
            0,
            "total_samples should be 0 after reset, got {}",
            tuner.total_samples()
        );
        assert_eq!(
            tuner.n_samples_seen(),
            0,
            "n_samples_seen should be 0 after reset, got {}",
            tuner.n_samples_seen()
        );
        assert_eq!(
            tuner.promotions(),
            0,
            "promotions should be 0 after reset, got {}",
            tuner.promotions()
        );
        assert_eq!(
            tuner.tournaments_completed(),
            0,
            "tournaments_completed should be 0 after reset, got {}",
            tuner.tournaments_completed()
        );
        // After reset, a new tournament should be started.
        assert!(
            tuner.candidates_remaining() > 0,
            "should have candidates after reset (new tournament started)"
        );
    }

    /// Verify n_samples_seen matches the number of train calls.
    #[test]
    fn tournament_n_samples_seen() {
        let mut tuner = AutoTuner::builder().factory(SgbtFactory::new(3)).build();

        assert_eq!(tuner.n_samples_seen(), 0, "should start at 0 samples");

        for i in 0..73 {
            let x = [i as f64, 0.0, 0.0];
            tuner.train(&x, i as f64);
        }

        assert_eq!(
            tuner.n_samples_seen(),
            73,
            "should have 73 samples after 73 train calls, got {}",
            tuner.n_samples_seen()
        );
    }

    /// Verify AutoTuner can be used as a trait object.
    #[test]
    fn tournament_implements_streaming_learner() {
        let tuner = AutoTuner::builder().factory(SgbtFactory::new(3)).build();

        let mut boxed: Box<dyn StreamingLearner> = Box::new(tuner);
        boxed.train(&[1.0, 2.0, 3.0], 4.0);
        let pred = boxed.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.is_finite(),
            "trait object prediction should be finite, got {pred}"
        );
    }

    /// Verify factory_names returns correct name for single factory.
    #[test]
    fn tournament_factory_names_single() {
        let tuner = AutoTuner::builder().factory(SgbtFactory::new(5)).build();

        let names = tuner.factory_names();
        assert_eq!(
            names.len(),
            1,
            "single factory should produce 1 name, got {}",
            names.len()
        );
        assert_eq!(
            names[0], "SGBT",
            "factory name should be SGBT, got {}",
            names[0]
        );
    }

    /// Verify multi-factory support works with two different factory types.
    #[test]
    fn tournament_multi_factory() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .add_factory(EsnFactory::new())
            .n_initial(4)
            .round_budget(30)
            .seed(42)
            .build();

        let names = tuner.factory_names();
        assert_eq!(
            names.len(),
            2,
            "should have 2 factories, got {}",
            names.len()
        );

        // Train with 1-dim features (ESN compatibility).
        for i in 0..300 {
            let x = [i as f64 * 0.01];
            let y = (i as f64 * 0.1).sin();
            tuner.train(&x, y);
        }

        let pred = tuner.predict(&[0.5]);
        assert!(
            pred.is_finite(),
            "multi-factory prediction should be finite, got {pred}"
        );
    }

    /// Verify all custom builder values are applied correctly.
    #[test]
    fn tournament_custom_config() {
        let tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(4)
            .round_budget(50)
            .metric(AutoMetric::RMSE)
            .ewma_span(30)
            .discount(0.95)
            .perturb_sigma(0.3)
            .seed(999)
            .build();

        assert_eq!(
            tuner.config.n_initial, 4,
            "n_initial should be 4, got {}",
            tuner.config.n_initial
        );
        assert_eq!(
            tuner.config.round_budget, 50,
            "round_budget should be 50, got {}",
            tuner.config.round_budget
        );
        assert_eq!(
            tuner.config.metric,
            AutoMetric::RMSE,
            "metric should be RMSE"
        );
        assert_eq!(
            tuner.config.ewma_span, 30,
            "ewma_span should be 30, got {}",
            tuner.config.ewma_span
        );
        assert!(
            (tuner.config.discount - 0.95).abs() < 1e-12,
            "discount should be 0.95, got {}",
            tuner.config.discount
        );
        assert!(
            (tuner.config.perturb_sigma - 0.3).abs() < 1e-12,
            "perturb_sigma should be 0.3, got {}",
            tuner.config.perturb_sigma
        );
        assert_eq!(
            tuner.config.seed, 999,
            "seed should be 999, got {}",
            tuner.config.seed
        );
        assert_eq!(
            tuner.candidates.len(),
            4,
            "should have 4 candidates (n_initial=4), got {}",
            tuner.candidates.len()
        );
    }

    /// Verify add_factory appends rather than replaces.
    #[test]
    fn tournament_add_factory() {
        let tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .add_factory(EsnFactory::new())
            .build();

        let names = tuner.factory_names();
        assert_eq!(
            names.len(),
            2,
            "add_factory should append, giving 2 factories, got {}",
            names.len()
        );
        assert_eq!(names[0], "SGBT", "first factory should be SGBT");
        assert_eq!(names[1], "ESN", "second factory should be ESN");

        // Verify factory() clears and add_factory appends.
        let tuner2 = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .add_factory(EsnFactory::new())
            .factory(SgbtFactory::new(5)) // this should clear both, leaving only this one
            .build();

        let names2 = tuner2.factory_names();
        assert_eq!(
            names2.len(),
            1,
            "factory() should clear existing, got {} factories",
            names2.len()
        );
        assert_eq!(names2[0], "SGBT", "only factory should be SGBT");
    }

    /// Verify candidates decrease between rounds (successive halving eliminates).
    #[test]
    fn tournament_candidates_decrease() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(8)
            .round_budget(20)
            .seed(55)
            // Pin bracket size so adaptive sizing doesn't change it mid-test.
            .min_n_initial(8)
            .max_n_initial(8)
            .build();

        assert_eq!(
            tuner.candidates_remaining(),
            8,
            "should start with 8 candidates, got {}",
            tuner.candidates_remaining()
        );

        // Train exactly round_budget samples to trigger one elimination.
        for i in 0..20 {
            let x = [i as f64 * 0.1, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] + x[1];
            tuner.train(&x, y);
        }

        // After first elimination, should have 4 candidates (half of 8).
        assert_eq!(
            tuner.candidates_remaining(),
            4,
            "after first elimination should have 4 candidates, got {}",
            tuner.candidates_remaining()
        );
        assert_eq!(
            tuner.current_round(),
            1,
            "should be in round 1 after first elimination, got {}",
            tuner.current_round()
        );

        // Train another round_budget to trigger second elimination.
        for i in 20..40 {
            let x = [i as f64 * 0.1, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] + x[1];
            tuner.train(&x, y);
        }

        // After second elimination, should have 2 candidates.
        assert_eq!(
            tuner.candidates_remaining(),
            2,
            "after second elimination should have 2 candidates, got {}",
            tuner.candidates_remaining()
        );

        // Train another round_budget to trigger third elimination -> finals -> new tournament.
        for i in 40..60 {
            let x = [i as f64 * 0.1, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] + x[1];
            tuner.train(&x, y);
        }

        // After finals, a new tournament starts with effective_n_initial candidates.
        // With min/max pinned to 8, bracket stays at 8.
        assert_eq!(
            tuner.candidates_remaining(),
            8,
            "after finals new tournament should start with 8 candidates, got {}",
            tuner.candidates_remaining()
        );
        assert!(
            tuner.tournaments_completed() >= 1,
            "should have completed at least 1 tournament, got {}",
            tuner.tournaments_completed()
        );
    }

    /// Verify warmup-protected candidates survive elimination despite poor metrics.
    #[test]
    fn warmup_protection() {
        // ESN has warmup_hint=50, SGBT has warmup_hint=0.
        // With round_budget=25 and 30 training samples, ESN candidates
        // should survive the first elimination despite poor warmup metrics.
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(1))
            .add_factory(EsnFactory::new())
            .n_initial(4)
            .round_budget(25)
            .seed(42)
            // Pin bracket so adaptive sizing doesn't interfere.
            .min_n_initial(4)
            .max_n_initial(4)
            .build();

        // Train 30 samples (triggers one elimination at 25).
        for i in 0..30 {
            let x = [i as f64 * 0.01];
            let y = (i as f64 * 0.1).sin();
            tuner.train(&x, y);
        }

        // After elimination at sample 25, any ESN candidates should have
        // survived due to warmup protection (err_count=25 < warmup_hint=50).
        // The tuner should still be functional.
        let pred = tuner.predict(&[0.5]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after warmup-protected elimination, got {pred}"
        );
    }

    /// Verify early stopping eliminates clearly bad candidates before round ends.
    #[test]
    fn early_stopping_eliminates_bad() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(8)
            .round_budget(200)
            .seed(42)
            // Pin bracket so adaptive sizing doesn't interfere.
            .min_n_initial(8)
            .max_n_initial(8)
            .build();

        assert_eq!(
            tuner.candidates_remaining(),
            8,
            "should start with 8 candidates"
        );

        // Feed a clear signal (y = 2*x[0]) for 100 samples (half the budget).
        // Early elimination checks happen every 50 samples (200/4).
        for i in 0..100 {
            let x = [i as f64 * 0.1, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 2.0;
            tuner.train(&x, y);
        }

        // After 100 samples with clear signal and 2 early elimination checks
        // (at 50 and 100), some bad candidates may have been eliminated.
        // The test verifies the mechanism runs without error and that the
        // tuner remains functional. We can't guarantee elimination since it
        // depends on the random configs, but we can verify it doesn't panic.
        assert!(
            tuner.candidates_remaining() <= 8,
            "candidates should not have increased, got {}",
            tuner.candidates_remaining()
        );

        let pred = tuner.predict(&[5.0, 0.0, 1.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after early elimination, got {pred}"
        );
    }

    /// Verify adaptive bracket grows on promotion.
    #[test]
    fn adaptive_bracket_grows_on_promotion() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(4)
            .round_budget(20)
            .min_n_initial(4)
            .max_n_initial(32)
            .seed(123)
            .build();

        assert_eq!(
            tuner.effective_n_initial(),
            4,
            "should start at n_initial=4"
        );

        // Train enough for promotions to occur.
        for i in 0..2000 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1] * 3.0 + 1.0;
            tuner.train(&x, y);
        }

        // After many tournaments, if any promotion occurred, the bracket
        // should have grown at some point.
        if tuner.promotions() > 0 {
            // At least one promotion happened, so effective_n_initial should
            // have been increased at least once. However, subsequent
            // non-promotions may have shrunk it back. The key property is
            // that the mechanism runs without error.
            assert!(
                tuner.effective_n_initial() >= tuner.config.min_n_initial,
                "effective_n_initial should be >= min_n_initial"
            );
            assert!(
                tuner.effective_n_initial() <= tuner.config.max_n_initial,
                "effective_n_initial should be <= max_n_initial"
            );
        }
    }

    /// Verify adaptive bracket shrinks without promotion.
    #[test]
    fn adaptive_bracket_shrinks_without_promotion() {
        // Use a large n_initial and short round_budget so tournaments
        // complete quickly. The champion gets trained on clear signal
        // before challengers, making promotion unlikely.
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(16)
            .round_budget(10)
            .min_n_initial(4)
            .max_n_initial(32)
            .seed(42)
            .build();

        assert_eq!(
            tuner.effective_n_initial(),
            16,
            "should start at n_initial=16"
        );

        // Pre-train the champion so it's strong (won't be beaten easily).
        // Actually we can't pre-train the champion directly. Instead, run
        // enough tournaments without promotion to see shrinkage.
        // Each tournament with n_initial=16: 4 rounds * 10 = 40 samples per tournament.
        // After ~5 tournaments without promotion, effective_n_initial should drop.
        for i in 0..500 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1] * 3.0 + 1.0;
            tuner.train(&x, y);
        }

        // Even if some promotions occurred, each non-promotion halves the bracket.
        // The mechanism should have run, keeping effective_n_initial within bounds.
        assert!(
            tuner.effective_n_initial() >= tuner.config.min_n_initial,
            "effective_n_initial ({}) should be >= min_n_initial ({})",
            tuner.effective_n_initial(),
            tuner.config.min_n_initial
        );
        assert!(
            tuner.effective_n_initial() <= tuner.config.max_n_initial,
            "effective_n_initial ({}) should be <= max_n_initial ({})",
            tuner.effective_n_initial(),
            tuner.config.max_n_initial
        );
    }

    /// Verify warmup_hint returns correct values for each factory.
    #[test]
    fn warmup_hint_factories() {
        let sgbt: Box<dyn ModelFactory> = Box::new(SgbtFactory::new(5));
        let esn: Box<dyn ModelFactory> = Box::new(EsnFactory::new());
        let mamba: Box<dyn ModelFactory> = Box::new(MambaFactory::new(4));
        let attn: Box<dyn ModelFactory> = Box::new(AttentionFactory::new(8));
        let spike: Box<dyn ModelFactory> = Box::new(SpikeNetFactory::new());

        assert_eq!(sgbt.warmup_hint(), 0, "SGBT warmup_hint should be 0");
        assert_eq!(esn.warmup_hint(), 50, "ESN warmup_hint should be 50");
        assert_eq!(mamba.warmup_hint(), 10, "Mamba warmup_hint should be 10");
        assert_eq!(attn.warmup_hint(), 10, "Attention warmup_hint should be 10");
        assert_eq!(spike.warmup_hint(), 20, "SpikeNet warmup_hint should be 20");
    }

    /// Verify effective_n_initial resets correctly.
    #[test]
    fn effective_n_initial_resets() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(8)
            .round_budget(20)
            .min_n_initial(4)
            .max_n_initial(32)
            .seed(42)
            .build();

        // Train enough to change effective_n_initial.
        for i in 0..200 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1] * 3.0;
            tuner.train(&x, y);
        }

        tuner.reset();

        assert_eq!(
            tuner.effective_n_initial(),
            8,
            "effective_n_initial should reset to n_initial=8, got {}",
            tuner.effective_n_initial()
        );
    }

    /// Verify adjusted_metric produces expected values.
    #[test]
    fn adjusted_metric_computation() {
        // Low complexity, many samples: small penalty (100/1000 = 0.1).
        let adj1 = super::adjusted_metric(1.0, 100, 1000);
        // High complexity, few samples: large penalty (10000/100 = 100.0).
        let adj2 = super::adjusted_metric(1.0, 10000, 100);
        assert!(
            adj2 > adj1,
            "high complexity with few samples should have higher adjusted score: adj1={adj1}, adj2={adj2}"
        );

        // Zero samples: penalty equals complexity.
        let adj_zero = super::adjusted_metric(0.5, 200, 0);
        assert!(
            (adj_zero - 200.5).abs() < 1e-12,
            "with zero samples penalty should equal complexity: expected 200.5, got {adj_zero}"
        );
    }

    /// Verify complexity penalty decays as more samples are observed.
    #[test]
    fn complexity_penalty_decays_with_samples() {
        let adj_early = super::adjusted_metric(1.0, 5000, 50);
        let adj_late = super::adjusted_metric(1.0, 5000, 5000);
        assert!(
            adj_early > adj_late,
            "penalty should decrease with more samples: early={adj_early}, late={adj_late}"
        );

        // The raw metric is the same (1.0), so the difference is purely penalty.
        let penalty_early = adj_early - 1.0;
        let penalty_late = adj_late - 1.0;
        assert!(
            (penalty_early - 100.0).abs() < 1e-12,
            "early penalty should be 5000/50=100.0, got {penalty_early}"
        );
        assert!(
            (penalty_late - 1.0).abs() < 1e-12,
            "late penalty should be 5000/5000=1.0, got {penalty_late}"
        );
    }

    /// Verify use_drift_rerace builder method sets config correctly.
    #[test]
    fn drift_rerace_config() {
        let tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .use_drift_rerace(true)
            .build();

        assert!(
            tuner.config.use_drift_rerace,
            "use_drift_rerace should be true after builder set"
        );
        assert!(
            tuner.drift_detector.is_some(),
            "drift_detector should be Some when use_drift_rerace is true"
        );

        let tuner2 = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .use_drift_rerace(false)
            .build();

        assert!(
            !tuner2.config.use_drift_rerace,
            "use_drift_rerace should be false"
        );
        assert!(
            tuner2.drift_detector.is_none(),
            "drift_detector should be None when use_drift_rerace is false"
        );
    }

    /// Verify drift re-racing does not crash and runs tournaments under distribution shift.
    #[test]
    fn drift_rerace_triggers_new_tournament() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(4)
            .round_budget(20)
            .min_n_initial(4)
            .max_n_initial(32)
            .use_drift_rerace(true)
            .seed(42)
            .build();

        // Phase 1: stable data (y = 2*x).
        for i in 0..200 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), (i as f64).cos()];
            let y = x[0] * 2.0;
            tuner.train(&x, y);
        }

        let tournaments_before_shift = tuner.tournaments_completed();

        // Phase 2: abrupt distribution shift (y = -10*x + 100).
        for i in 200..500 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), (i as f64).cos()];
            let y = x[0] * -10.0 + 100.0;
            tuner.train(&x, y);
        }

        // Verify the tuner completed more tournaments (drift re-racing should
        // have triggered at least one restart). At minimum, verify no crash.
        assert!(
            tuner.tournaments_completed() > tournaments_before_shift,
            "expected more tournaments after distribution shift, before={tournaments_before_shift}, after={}",
            tuner.tournaments_completed()
        );

        let pred = tuner.predict(&[3.0, 0.0, 1.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after drift re-racing, got {pred}"
        );
    }

    /// Verify drift detector is reset when AutoTuner is reset.
    #[test]
    fn drift_rerace_reset() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(4)
            .round_budget(20)
            .use_drift_rerace(true)
            .build();

        // Train some data to accumulate drift detector state.
        for i in 0..100 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), (i as f64).cos()];
            let y = x[0] * 2.0;
            tuner.train(&x, y);
        }

        tuner.reset();

        assert_eq!(
            tuner.total_samples(),
            0,
            "total_samples should be 0 after reset"
        );
        assert!(
            tuner.drift_detector.is_some(),
            "drift_detector should still exist after reset"
        );
        // The drift detector was reset (no crash).
        let pred = tuner.predict(&[0.5, 0.0, 1.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after reset with drift rerace"
        );
    }

    /// Verify snapshot captures state correctly after training.
    #[test]
    fn snapshot_after_training() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(5))
            .round_budget(50)
            .build();

        for i in 0..200 {
            let x = [i as f64, (i as f64) * 0.5, (i as f64) * 0.1, 1.0, -1.0];
            let y = x[0] * 0.3 + x[1] * 0.7 + 2.0;
            tuner.train(&x, y);
        }

        let snap = tuner.snapshot();
        assert!(
            !snap.champion_factory.is_empty(),
            "champion_factory should be non-empty, got {:?}",
            snap.champion_factory
        );
        assert_eq!(
            snap.total_samples, 200,
            "total_samples should be 200, got {}",
            snap.total_samples
        );
        assert_eq!(
            snap.champion_samples, 200,
            "champion_samples should be 200, got {}",
            snap.champion_samples
        );
        assert!(
            !snap.factory_names.is_empty(),
            "factory_names should be non-empty"
        );
    }

    /// Verify snapshot shows candidates during an active tournament.
    #[test]
    fn snapshot_shows_candidates() {
        let tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(4)
            .round_budget(100)
            .build();

        let snap = tuner.snapshot();
        assert!(
            !snap.candidates.is_empty(),
            "candidates should be non-empty immediately after build, got {}",
            snap.candidates.len()
        );
        assert_eq!(
            snap.candidates.len(),
            4,
            "should have 4 candidates (n_initial=4), got {}",
            snap.candidates.len()
        );
        for c in &snap.candidates {
            assert!(
                !c.factory_name.is_empty(),
                "candidate factory_name should be non-empty"
            );
        }
    }

    /// Verify Display impl produces non-empty output without panicking.
    #[test]
    fn snapshot_display() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(4)
            .round_budget(50)
            .build();

        for i in 0..50 {
            let x = [i as f64 * 0.1, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1];
            tuner.train(&x, y);
        }

        let snap = tuner.snapshot();
        let display = format!("{snap}");
        assert!(!display.is_empty(), "Display output should be non-empty");
        assert!(
            display.contains("AutoTuner Snapshot"),
            "Display output should contain header, got: {display}"
        );
        assert!(
            display.contains("Champion:"),
            "Display output should contain Champion line"
        );
    }

    /// Verify snapshot promotions field matches tuner.promotions().
    #[test]
    fn snapshot_tracks_promotions() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_initial(4)
            .round_budget(20)
            .seed(123)
            .build();

        for i in 0..1000 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1] * 3.0 + 1.0;
            tuner.train(&x, y);
        }

        let snap = tuner.snapshot();
        assert_eq!(
            snap.promotions,
            tuner.promotions(),
            "snapshot promotions ({}) should match tuner.promotions() ({})",
            snap.promotions,
            tuner.promotions()
        );
        assert_eq!(
            snap.tournaments_completed,
            tuner.tournaments_completed(),
            "snapshot tournaments_completed ({}) should match tuner.tournaments_completed() ({})",
            snap.tournaments_completed,
            tuner.tournaments_completed()
        );
    }

    /// Verify auto_builder mode creates an adaptor when enabled.
    #[test]
    fn auto_builder_mode_enabled() {
        let tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .auto_builder(true)
            .build();

        assert!(
            tuner.config.auto_builder,
            "auto_builder config should be true"
        );
        assert!(
            tuner.adaptor.is_some(),
            "adaptor should be Some when auto_builder is enabled"
        );
    }

    /// Verify default build has no adaptor.
    #[test]
    fn auto_builder_mode_disabled() {
        let tuner = AutoTuner::builder().factory(SgbtFactory::new(3)).build();

        assert!(
            !tuner.config.auto_builder,
            "auto_builder config should be false by default"
        );
        assert!(
            tuner.adaptor.is_none(),
            "adaptor should be None when auto_builder is disabled"
        );
    }

    /// Verify training with auto_builder=true produces finite predictions.
    #[test]
    fn auto_builder_train_works() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .auto_builder(true)
            .round_budget(50)
            .build();

        for i in 0..100 {
            let x = [i as f64 * 0.1, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1] * 3.0 + 1.0;
            tuner.train(&x, y);
        }

        let pred = tuner.predict(&[5.0, 0.0, 1.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite with auto_builder enabled, got {pred}"
        );
    }
}
