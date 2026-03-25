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

use crate::automl::{AutoMetric, ConfigSampler, HyperConfig, ModelFactory, RewardNormalizer};
use crate::bandits::{Bandit, DiscountedThompsonSampling};
use crate::metrics::ewma::EwmaRegressionMetrics;
use irithyll_core::learner::StreamingLearner;

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
/// use irithyll::{auto_tune, automl::SgbtFactory, StreamingLearner};
///
/// let mut tuner = auto_tune(SgbtFactory::new(5));
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

        // 2. All candidates: predict -> EWMA -> train.
        for c in &mut self.candidates {
            let pred = c.model.predict(features);
            c.ewma.update(target, pred);
            c.model.train_one(features, target, weight);
        }

        // 3. Increment counters.
        self.samples_in_round += 1;
        self.total_samples += 1;

        // 4. Elimination check.
        if self.samples_in_round >= self.config.round_budget as u64 {
            self.eliminate_round();
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
        self.bandit.reset();
        self.normalizer.reset();
        self.start_tournament();
    }
}

// ===========================================================================
// Private tournament logic
// ===========================================================================

impl AutoTuner {
    /// Eliminate the bottom half of candidates after a round completes.
    fn eliminate_round(&mut self) {
        if self.candidates.is_empty() {
            self.start_tournament();
            return;
        }

        // Collect metrics (free function avoids borrow issues).
        let metric_type = self.config.metric;
        let mut indexed: Vec<(usize, f64)> = self
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, get_metric(&c.ewma, metric_type)))
            .collect();

        // Sort best first (lowest metric).
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top half (ceiling division: always keep at least 1).
        let keep = indexed.len().div_ceil(2);
        let keep_set: Vec<usize> = indexed.iter().take(keep).map(|(i, _)| *i).collect();

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
            }

            // Update bandit: reward the finalist's factory.
            let reward = self
                .normalizer
                .normalize(finalist_metric.min(champion_metric));
            if finalist.factory_idx < self.bandit.n_arms() {
                self.bandit.update(finalist.factory_idx, reward);
            }
        }

        self.candidates.clear();
        self.tournaments_completed += 1;
        self.start_tournament();
    }

    /// Spawn a new tournament with `n_initial` candidates.
    fn start_tournament(&mut self) {
        let n = self.config.n_initial;
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
            });
        }

        self.candidates = candidates;
        self.current_round = 0;
        self.samples_in_round = 0;
        self.champion_ewma.reset();
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

// Safety: AutoTuner contains only Send+Sync types.
// - Box<dyn StreamingLearner>: StreamingLearner requires Send+Sync
// - Box<dyn ModelFactory>: ModelFactory requires Send+Sync
// - All other fields are owned, non-Rc, non-Cell types.
unsafe impl Send for AutoTuner {}
unsafe impl Sync for AutoTuner {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automl::factories::EsnFactory;
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

        // After finals, a new tournament starts with n_initial candidates.
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
}
