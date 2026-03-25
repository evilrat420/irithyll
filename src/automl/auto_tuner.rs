//! Streaming AutoML orchestrator: champion-challenger racing.
//!
//! The [`AutoTuner`] maintains a champion model that always provides predictions,
//! while challengers with different hyperparameter configurations race in parallel.
//! At the end of each race budget, the best challenger is promoted to champion if
//! it outperforms the current champion. The worst half of challengers are replaced
//! with new random configurations sampled from the search space.

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
    /// Number of challenger models racing against the champion.
    pub n_challengers: usize,
    /// Number of samples per race before evaluation.
    pub race_budget: usize,
    /// Metric to optimize (lower is better).
    pub metric: AutoMetric,
    /// EWMA span for metric tracking.
    pub ewma_span: usize,
    /// Discount factor for the bandit (closer to 1 = slower forgetting).
    pub discount: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for AutoTunerConfig {
    fn default() -> Self {
        Self {
            n_challengers: 4,
            race_budget: 200,
            metric: AutoMetric::MAE,
            ewma_span: 50,
            discount: 0.99,
            seed: 42,
        }
    }
}

// ===========================================================================
// AutoTunerConfigBuilder
// ===========================================================================

/// Builder for [`AutoTunerConfig`].
pub struct AutoTunerConfigBuilder {
    config: AutoTunerConfig,
}

impl AutoTunerConfig {
    /// Create a builder with default configuration values.
    pub fn builder() -> AutoTunerConfigBuilder {
        AutoTunerConfigBuilder {
            config: AutoTunerConfig::default(),
        }
    }
}

impl AutoTunerConfigBuilder {
    /// Set the number of challengers (default: 4).
    pub fn n_challengers(mut self, n: usize) -> Self {
        self.config.n_challengers = n;
        self
    }

    /// Set the race budget in samples (default: 200).
    pub fn race_budget(mut self, b: usize) -> Self {
        self.config.race_budget = b;
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

    /// Set the discount factor for the bandit (default: 0.99).
    pub fn discount(mut self, d: f64) -> Self {
        self.config.discount = d;
        self
    }

    /// Set the RNG seed (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.config.seed = s;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> AutoTunerConfig {
        self.config
    }
}

// ===========================================================================
// Challenger
// ===========================================================================

/// A challenger model racing against the champion.
struct Challenger {
    model: Box<dyn StreamingLearner>,
    ewma: EwmaRegressionMetrics,
    config: HyperConfig,
}

// ===========================================================================
// AutoTuner
// ===========================================================================

/// Streaming AutoML orchestrator with champion-challenger racing.
///
/// The `AutoTuner` implements [`StreamingLearner`] and automatically tunes
/// hyperparameters for a given model type. It maintains a champion model that
/// always provides predictions, while challengers with different configurations
/// race in parallel. After each race budget, the best challenger may be promoted
/// to champion, and the worst half of challengers are replaced with new configs.
///
/// # Architecture
///
/// 1. **Champion**: The current best model. Always used for `predict()`.
/// 2. **Challengers**: `n_challengers` models with different hyperparameter configs.
/// 3. **Bandit**: [`DiscountedThompsonSampling`] for non-stationary config selection.
/// 4. **EWMA metrics**: Track performance of champion and each challenger.
/// 5. **Race budget**: After `race_budget` samples, evaluate and possibly promote.
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
    challengers: Vec<Challenger>,
    factory: Box<dyn ModelFactory>,
    sampler: ConfigSampler,
    bandit: DiscountedThompsonSampling,
    normalizer: RewardNormalizer,
    config: AutoTunerConfig,
    samples_in_race: u64,
    total_samples: u64,
    promotions: u64,
}

// ===========================================================================
// AutoTunerBuilder
// ===========================================================================

/// Builder for [`AutoTuner`].
///
/// The only required field is a [`ModelFactory`]; all other settings have defaults.
pub struct AutoTunerBuilder {
    factory: Option<Box<dyn ModelFactory>>,
    config: AutoTunerConfig,
}

impl AutoTuner {
    /// Create a builder for an `AutoTuner`.
    pub fn builder() -> AutoTunerBuilder {
        AutoTunerBuilder {
            factory: None,
            config: AutoTunerConfig::default(),
        }
    }
}

impl AutoTunerBuilder {
    /// Set the model factory (required).
    pub fn factory(mut self, f: impl ModelFactory + 'static) -> Self {
        self.factory = Some(Box::new(f));
        self
    }

    /// Set the number of challengers (default: 4).
    pub fn n_challengers(mut self, n: usize) -> Self {
        self.config.n_challengers = n;
        self
    }

    /// Set the race budget in samples (default: 200).
    pub fn race_budget(mut self, b: usize) -> Self {
        self.config.race_budget = b;
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

    /// Set the discount factor for the bandit (default: 0.99).
    pub fn discount(mut self, d: f64) -> Self {
        self.config.discount = d;
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
        let factory = self
            .factory
            .expect("AutoTunerBuilder: a ModelFactory is required");
        let config = self.config;

        let space = factory.config_space();

        // Seed must be non-zero for xorshift64.
        let sampler_seed = if config.seed == 0 { 1 } else { config.seed };
        let mut sampler = ConfigSampler::new(space, sampler_seed);

        // Create champion from first random config.
        let champion_config = sampler.random();
        let champion = factory.create(&champion_config);
        let champion_ewma = EwmaRegressionMetrics::new(config.ewma_span);

        // Create challengers.
        let mut challengers = Vec::with_capacity(config.n_challengers);
        for _ in 0..config.n_challengers {
            let cfg = sampler.random();
            let model = factory.create(&cfg);
            challengers.push(Challenger {
                model,
                ewma: EwmaRegressionMetrics::new(config.ewma_span),
                config: cfg,
            });
        }

        // Initialize bandit with one arm per challenger.
        let n_arms = config.n_challengers.max(1);
        let bandit = DiscountedThompsonSampling::with_seed(n_arms, config.discount, sampler_seed);

        let normalizer = RewardNormalizer::with_span(config.ewma_span);

        AutoTuner {
            champion,
            champion_ewma,
            champion_config,
            challengers,
            factory,
            sampler,
            bandit,
            normalizer,
            config,
            samples_in_race: 0,
            total_samples: 0,
            promotions: 0,
        }
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

    /// Name of the underlying model factory (e.g., "SGBT", "ESN").
    pub fn factory_name(&self) -> &str {
        self.factory.name()
    }

    /// Total samples processed.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }
}

// ===========================================================================
// StreamingLearner implementation
// ===========================================================================

impl StreamingLearner for AutoTuner {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // 1. Get champion prediction BEFORE training.
        let champion_pred = self.champion.predict(features);

        // 2. Update champion EWMA with pre-train prediction.
        self.champion_ewma.update(target, champion_pred);

        // 3. Train champion.
        self.champion.train_one(features, target, weight);

        // 4. For each challenger: predict, record EWMA, train.
        for challenger in &mut self.challengers {
            let pred = challenger.model.predict(features);
            challenger.ewma.update(target, pred);
            challenger.model.train_one(features, target, weight);
        }

        // 5. Increment race counter.
        self.samples_in_race += 1;
        self.total_samples += 1;

        // 6. Race evaluation: if race_budget reached.
        if self.samples_in_race >= self.config.race_budget as u64 {
            self.evaluate_race();
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
        for c in &mut self.challengers {
            c.model.reset();
            c.ewma.reset();
        }
        self.samples_in_race = 0;
        self.total_samples = 0;
        self.promotions = 0;
        self.bandit.reset();
        self.normalizer.reset();
    }
}

// ===========================================================================
// Private race evaluation
// ===========================================================================

impl AutoTuner {
    /// Evaluate the current race and possibly promote a challenger.
    fn evaluate_race(&mut self) {
        // 1. Get champion's EWMA metric.
        let champion_metric = self.get_metric(&self.champion_ewma);

        // 2. Find the best challenger (lowest metric).
        let mut best_idx: Option<usize> = None;
        let mut best_metric = champion_metric;

        for (i, challenger) in self.challengers.iter().enumerate() {
            let metric = self.get_metric(&challenger.ewma);
            if metric < best_metric {
                best_metric = metric;
                best_idx = Some(i);
            }
        }

        // 3. If best challenger beats champion, promote it.
        if let Some(idx) = best_idx {
            let promoted = self.challengers.remove(idx);
            let demoted_model = std::mem::replace(&mut self.champion, promoted.model);
            let demoted_ewma = std::mem::replace(&mut self.champion_ewma, promoted.ewma);
            let demoted_config = std::mem::replace(&mut self.champion_config, promoted.config);

            // Put demoted champion back as a challenger.
            self.challengers.push(Challenger {
                model: demoted_model,
                ewma: demoted_ewma,
                config: demoted_config,
            });

            self.promotions += 1;
        }

        // 4. Update bandit rewards for each challenger.
        for (i, challenger) in self.challengers.iter().enumerate() {
            let metric = self.get_metric(&challenger.ewma);
            let reward = self.normalizer.normalize(metric);
            if i < self.bandit.n_arms() {
                self.bandit.update(i, reward);
            }
        }

        // 5. Replace the worst half of challengers with new configs.
        self.replace_worst_challengers();

        // 6. Reset race counter.
        self.samples_in_race = 0;
    }

    /// Replace the worst half of challengers with new random configs.
    fn replace_worst_challengers(&mut self) {
        let n_replace = self.challengers.len() / 2;
        if n_replace == 0 {
            return;
        }

        // Collect metrics for all challengers.
        let mut metrics: Vec<(usize, f64)> = self
            .challengers
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.get_metric(&c.ewma)))
            .collect();

        // Sort worst first (highest metric = worst performance).
        metrics.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Replace worst with new random configs.
        let worst_indices: Vec<usize> = metrics.iter().take(n_replace).map(|(i, _)| *i).collect();
        for &idx in &worst_indices {
            let config = self.sampler.random();
            let model = self.factory.create(&config);
            self.challengers[idx] = Challenger {
                model,
                ewma: EwmaRegressionMetrics::new(self.config.ewma_span),
                config,
            };
        }
    }

    /// Get the relevant metric from an EWMA tracker.
    fn get_metric(&self, ewma: &EwmaRegressionMetrics) -> f64 {
        match self.config.metric {
            AutoMetric::MAE => ewma.mae(),
            AutoMetric::MSE => ewma.mse(),
            AutoMetric::RMSE => ewma.rmse(),
        }
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
    use crate::automl::SgbtFactory;

    /// Verify AutoTuner builder creates an instance with default config values.
    #[test]
    fn auto_tuner_builder_default() {
        let tuner = AutoTuner::builder().factory(SgbtFactory::new(5)).build();

        assert_eq!(
            tuner.config.n_challengers, 4,
            "default n_challengers should be 4, got {}",
            tuner.config.n_challengers
        );
        assert_eq!(
            tuner.config.race_budget, 200,
            "default race_budget should be 200, got {}",
            tuner.config.race_budget
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
        assert_eq!(
            tuner.config.seed, 42,
            "default seed should be 42, got {}",
            tuner.config.seed
        );
        assert_eq!(
            tuner.challengers.len(),
            4,
            "should have 4 challengers, got {}",
            tuner.challengers.len()
        );
    }

    /// Verify training and prediction produce finite values.
    #[test]
    fn auto_tuner_train_and_predict() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(5))
            .race_budget(50)
            .build();

        for i in 0..100 {
            let x = [i as f64, (i as f64) * 0.5, (i as f64) * 0.1, 1.0, -1.0];
            let y = x[0] * 0.3 + x[1] * 0.7 + 2.0;
            tuner.train(&x, y);
        }

        let pred = tuner.predict(&[50.0, 25.0, 5.0, 1.0, -1.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after 100 training samples, got {pred}"
        );
    }

    /// Verify that promotions occur after enough training with varied configs.
    #[test]
    fn auto_tuner_promotes_better_config() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_challengers(4)
            .race_budget(50)
            .seed(123)
            .build();

        // Train enough samples to trigger multiple races.
        for i in 0..500 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1] * 3.0 + 1.0;
            tuner.train(&x, y);
        }

        // With enough races and random config diversity, at least one promotion
        // should have occurred.
        assert!(
            tuner.promotions() > 0,
            "expected at least 1 promotion after 500 samples with race_budget=50, got {}",
            tuner.promotions()
        );
    }

    /// Verify reset clears all state.
    #[test]
    fn auto_tuner_reset() {
        let mut tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .race_budget(20)
            .build();

        for i in 0..50 {
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
    }

    /// Verify n_samples_seen matches the number of train calls.
    #[test]
    fn auto_tuner_n_samples_seen() {
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
    fn auto_tuner_implements_streaming_learner() {
        let tuner = AutoTuner::builder().factory(SgbtFactory::new(3)).build();

        let mut boxed: Box<dyn StreamingLearner> = Box::new(tuner);
        boxed.train(&[1.0, 2.0, 3.0], 4.0);
        let pred = boxed.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.is_finite(),
            "trait object prediction should be finite, got {pred}"
        );
    }

    /// Verify factory_name returns the expected string.
    #[test]
    fn auto_tuner_factory_name() {
        let tuner = AutoTuner::builder().factory(SgbtFactory::new(5)).build();

        assert_eq!(
            tuner.factory_name(),
            "SGBT",
            "factory_name should return SGBT, got {}",
            tuner.factory_name()
        );
    }

    /// Verify custom config values are applied correctly.
    #[test]
    fn auto_tuner_custom_config() {
        let tuner = AutoTuner::builder()
            .factory(SgbtFactory::new(3))
            .n_challengers(8)
            .race_budget(100)
            .metric(AutoMetric::RMSE)
            .ewma_span(30)
            .discount(0.95)
            .seed(999)
            .build();

        assert_eq!(
            tuner.config.n_challengers, 8,
            "n_challengers should be 8, got {}",
            tuner.config.n_challengers
        );
        assert_eq!(
            tuner.config.race_budget, 100,
            "race_budget should be 100, got {}",
            tuner.config.race_budget
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
        assert_eq!(
            tuner.config.seed, 999,
            "seed should be 999, got {}",
            tuner.config.seed
        );
        assert_eq!(
            tuner.challengers.len(),
            8,
            "should have 8 challengers, got {}",
            tuner.challengers.len()
        );
    }
}
