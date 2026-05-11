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
use crate::automl::budget::BudgetLedger;
use crate::automl::space::ParamMap;
use crate::automl::{AutoMetric, ModelFactory, RewardNormalizer};
use crate::bandits::DiscountedThompsonSampling;
use crate::drift::adwin::Adwin;
use crate::metrics::ewma::EwmaRegressionMetrics;
use irithyll_core::drift::DriftDetector;
use irithyll_core::error::ConfigError;
use irithyll_core::learner::StreamingLearner;
use tracing::warn;

mod core;
mod racing;
mod scheduler;

pub use self::core::AutoTuner;
pub use self::racing::CandidateSnapshot;
pub use self::scheduler::AutoTunerSnapshot;

// ===========================================================================
// Diagnostic snapshots (exported from scheduler, re-exported here)
// ===========================================================================

/// Snapshot of the AutoTuner's current state for diagnostics.
#[doc(inline)]
pub use self::scheduler::AutoTunerSnapshot as Snapshot;

/// Snapshot of a single tournament candidate.
#[doc(inline)]
pub use self::racing::CandidateSnapshot as CandidateInfo;

// ===========================================================================
// AutoTunerConfig
// ===========================================================================

/// Configuration for the [`AutoTuner`].
#[derive(Debug, Clone)]
pub struct AutoTunerConfig {
    /// Initial candidates per tournament (default: 8).
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
// Challenger (internal)
// ===========================================================================

/// A candidate model competing in the current tournament.
pub(crate) struct Challenger {
    pub model: Box<dyn StreamingLearner>,
    pub ewma: EwmaRegressionMetrics,
    pub params: ParamMap,
    pub factory_idx: usize,
    // Welford's online stats for paired error differences (statistical early stopping).
    pub err_mean: f64,
    pub err_m2: f64,
    pub err_count: u64,
    /// Index into the tournament's [`BudgetLedger`] for this challenger.
    pub budget_idx: usize,
}

// ===========================================================================
// AutoTunerBuilder
// ===========================================================================

/// Builder for [`AutoTuner`].
///
/// At least one [`ModelFactory`] must be provided via [`factory`](Self::factory)
/// or [`add_factory`](Self::add_factory). All other settings have defaults.
pub struct AutoTunerBuilder {
    pub(crate) factories: Vec<Box<dyn ModelFactory>>,
    pub(crate) config: AutoTunerConfig,
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

    /// Validate and build the `AutoTuner`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if:
    /// - No factory was provided.
    /// - `n_initial == 0`.
    /// - `round_budget == 0`.
    /// - `ewma_span == 0`.
    /// - `discount` is not in (0, 1].
    /// - `perturb_sigma < 0`.
    /// - `min_n_initial > max_n_initial`.
    /// - `n_initial` is not in `[min_n_initial, max_n_initial]`.
    pub fn build(self) -> Result<AutoTuner, ConfigError> {
        if self.factories.is_empty() {
            return Err(ConfigError::invalid(
                "factories",
                "at least one ModelFactory is required",
            ));
        }
        let c = &self.config;
        if c.n_initial == 0 {
            return Err(ConfigError::out_of_range(
                "n_initial",
                "must be >= 1",
                c.n_initial,
            ));
        }
        if c.round_budget == 0 {
            return Err(ConfigError::out_of_range(
                "round_budget",
                "must be >= 1",
                c.round_budget,
            ));
        }
        if c.ewma_span == 0 {
            return Err(ConfigError::out_of_range(
                "ewma_span",
                "must be >= 1",
                c.ewma_span,
            ));
        }
        if c.discount <= 0.0 || c.discount > 1.0 {
            return Err(ConfigError::out_of_range(
                "discount",
                "must be in (0, 1]",
                c.discount,
            ));
        }
        if c.perturb_sigma < 0.0 {
            return Err(ConfigError::out_of_range(
                "perturb_sigma",
                "must be >= 0",
                c.perturb_sigma,
            ));
        }
        if c.min_n_initial > c.max_n_initial {
            return Err(ConfigError::invalid(
                "min_n_initial",
                format!(
                    "must be <= max_n_initial ({}), got {}",
                    c.max_n_initial, c.min_n_initial
                ),
            ));
        }
        if c.n_initial < c.min_n_initial || c.n_initial > c.max_n_initial {
            return Err(ConfigError::invalid(
                "n_initial",
                format!(
                    "must be in [min_n_initial ({}), max_n_initial ({})], got {}",
                    c.min_n_initial, c.max_n_initial, c.n_initial
                ),
            ));
        }
        let config = self.config;

        // Seed must be non-zero for xorshift64.
        let seed = if config.seed == 0 { 1 } else { config.seed };

        // Per-factory RNG state for sampling. Each factory gets its own seed
        // derived from the master seed so the streams stay independent.
        let mut sampler_rngs: Vec<u64> = (0..self.factories.len())
            .map(|i| seed.wrapping_add(i as u64).max(1))
            .collect();

        // Bandit: one arm per factory.
        let n_factory_arms = self.factories.len().max(1);
        let bandit = DiscountedThompsonSampling::with_seed(n_factory_arms, config.discount, seed);

        // Create initial champion from first factory.
        // A factory's own config_space() must always produce valid configs —
        // if create() fails here it is a bug in the factory implementation.
        let champion_factory_idx = 0;
        let champion_space = self.factories[0].config_space();
        let champion_params = champion_space
            .sample(&mut sampler_rngs[0])
            .unwrap_or_else(|e| {
                panic!(
                    "AutoTunerBuilder: initial champion search-space sample failed for factory '{}': {}",
                    self.factories[0].name(),
                    e
                )
            });
        let champion = self.factories[0]
            .create(&champion_params)
            .unwrap_or_else(|e| {
                panic!(
                    "AutoTunerBuilder: initial champion creation failed for factory '{}': {}. \
                     The factory's config_space() must produce configs that create() accepts.",
                    self.factories[0].name(),
                    e
                )
            });
        let champion_ewma = EwmaRegressionMetrics::new(config.ewma_span);

        let effective_n_initial = config.n_initial;
        let drift_detector: Option<Box<dyn DriftDetector>> = if config.use_drift_rerace {
            Some(Box::new(Adwin::default()))
        } else {
            None
        };
        // PR-AM-1: Gate auto_builder to SGBT-family factories only.
        // Non-SGBT models have no-op adjust_config / apply_structural_change, so
        // activating the SPSA adaptor would silently do nothing. Warn once at
        // construction and skip rather than running a no-op adaptation loop.
        let adaptor = if config.auto_builder {
            let first_supports = self.factories[0].supports_auto_builder();
            if !first_supports {
                warn!(
                    factory = self.factories[0].name(),
                    "auto_builder=true has no effect for non-SGBT factories; \
                     the SPSA adaptor requires SGBT-family models. Skipping adaptor."
                );
                None
            } else {
                // Initialize FeasibleRegion using the first factory's feature hint.
                let n_feat = self.factories[0].n_features_hint().max(1);
                let region = auto_builder::FeasibleRegion::from_data(100, n_feat, 1.0);
                Some(auto_builder::DiagnosticLearner::with_objective(
                    region,
                    config.meta_objective,
                ))
            }
        } else {
            None
        };
        let mut tuner = AutoTuner {
            champion,
            champion_ewma,
            champion_params,
            champion_factory_idx,
            candidates: Vec::new(),
            current_round: 0,
            samples_in_round: 0,
            factories: self.factories,
            sampler_rngs,
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
            budget_ledger: BudgetLedger::new(),
        };

        tuner.start_tournament();
        Ok(tuner)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automl::Factory;

    /// Verify AutoTuner builder creates an instance with default config values.
    #[test]
    fn tournament_builder_default() {
        let tuner = AutoTuner::builder()
            .factory(Factory::sgbt(5))
            .build()
            .expect("valid config");

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
    }

    /// Verify that the builder creates a champion and a tournament.
    #[test]
    fn tournament_builder_creates_champion_and_tournament() {
        let tuner = AutoTuner::builder()
            .factory(Factory::sgbt(3))
            .build()
            .expect("valid config");

        assert_eq!(
            tuner.total_samples, 0,
            "initial total_samples should be 0, got {}",
            tuner.total_samples
        );
        assert_eq!(
            tuner.candidates.len(),
            8,
            "first tournament should have 8 candidates, got {}",
            tuner.candidates.len()
        );
    }

    /// Test the snapshot diagnostic method.
    #[test]
    fn tournament_snapshot() {
        let tuner = AutoTuner::builder()
            .factory(Factory::sgbt(3))
            .n_initial(4)
            .build()
            .expect("valid config");

        let snap = tuner.snapshot();
        assert_eq!(snap.champion_factory, "SGBT");
        assert_eq!(snap.candidates.len(), 4);
    }

    /// Test reset clears tournament state.
    #[test]
    fn tournament_reset() {
        let mut tuner = AutoTuner::builder()
            .factory(Factory::sgbt(3))
            .build()
            .expect("valid config");

        // Train a bit.
        for i in 0..10 {
            let x = [i as f64, 0.5, 0.3];
            let y = i as f64 * 0.1;
            tuner.train(&x, y);
        }

        assert!(tuner.total_samples > 0);
        tuner.reset();

        assert_eq!(
            tuner.total_samples, 0,
            "total_samples should be 0 after reset"
        );
        assert_eq!(tuner.promotions, 0, "promotions should be 0 after reset");
        assert_eq!(tuner.candidates.len(), 8, "tournament should be restarted");
    }
}
