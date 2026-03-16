//! Prequential (predictive sequential) evaluation for streaming models.
//!
//! The gold standard for evaluating online learners (Gama et al., 2009, 2013).
//! For each sample the protocol is: **predict -> record metrics -> train**.
//! The model is always tested on data it has not yet seen.
//!
//! Supports optional warmup (train-only period), step intervals (evaluate
//! every N-th sample), and pluggable metric backends (cumulative, rolling,
//! EWMA).

use crate::learner::StreamingLearner;
use crate::metrics::ewma::EwmaRegressionMetrics;
use crate::metrics::regression::RegressionMetrics;
use crate::metrics::rolling::RollingRegressionMetrics;
use crate::sample::Observation;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`PrequentialEvaluator`].
///
/// Controls the warmup period and evaluation frequency.
///
/// # Defaults
///
/// - `warmup`: 0 (evaluate from the first sample)
/// - `step_interval`: 1 (evaluate every sample)
#[derive(Debug, Clone)]
pub struct PrequentialConfig {
    /// Number of warmup samples -- model trains without being evaluated.
    pub warmup: usize,
    /// Evaluate every N-th sample (1 = every sample).
    pub step_interval: usize,
}

impl Default for PrequentialConfig {
    fn default() -> Self {
        Self {
            warmup: 0,
            step_interval: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// Prequential (test-then-train) evaluator for streaming regression models.
///
/// For each sample:
/// 1. **Predict** -- query the model before it sees this sample.
/// 2. **Record** -- update metrics with (target, prediction) if past warmup
///    and on the evaluation step interval.
/// 3. **Train** -- feed the sample to the model.
///
/// This ensures the model is always evaluated on unseen data, which is the
/// correct methodology for streaming ML (as opposed to train-then-test which
/// leaks information).
///
/// # Metric Backends
///
/// - **Cumulative** ([`RegressionMetrics`]) -- always active, tracks MAE/MSE/RMSE/R-squared over all evaluated samples.
/// - **Rolling** ([`RollingRegressionMetrics`]) -- optional, windowed metrics over the last N evaluated samples.
/// - **EWMA** ([`EwmaRegressionMetrics`]) -- optional, exponentially weighted metrics with configurable span.
///
/// # Example
///
/// ```
/// use irithyll::evaluation::{PrequentialEvaluator, PrequentialConfig};
/// use irithyll::learner::{SGBTLearner, StreamingLearner};
/// use irithyll::SGBTConfig;
///
/// let config = SGBTConfig::builder()
///     .n_steps(5)
///     .learning_rate(0.1)
///     .grace_period(10)
///     .build()
///     .unwrap();
/// let mut model = SGBTLearner::from_config(config);
///
/// let mut eval = PrequentialEvaluator::new()
///     .with_rolling_window(50)
///     .with_ewma(20);
///
/// // Simulate a data stream
/// for i in 0..200 {
///     let x = [i as f64, (i as f64).sin()];
///     let y = i as f64 * 0.5 + 1.0;
///     eval.step(&mut model, &x, y);
/// }
///
/// assert!(eval.regression_metrics().n_samples() > 0);
/// ```
#[derive(Debug, Clone)]
pub struct PrequentialEvaluator {
    config: PrequentialConfig,
    regression: RegressionMetrics,
    rolling: Option<RollingRegressionMetrics>,
    ewma: Option<EwmaRegressionMetrics>,
    samples_seen: u64,
    samples_evaluated: u64,
}

impl PrequentialEvaluator {
    /// Create a new evaluator with default configuration (warmup=0, step_interval=1).
    pub fn new() -> Self {
        Self::with_config(PrequentialConfig::default())
    }

    /// Create a new evaluator with the given configuration.
    pub fn with_config(config: PrequentialConfig) -> Self {
        assert!(config.step_interval > 0, "step_interval must be > 0");
        Self {
            config,
            regression: RegressionMetrics::new(),
            rolling: None,
            ewma: None,
            samples_seen: 0,
            samples_evaluated: 0,
        }
    }

    /// Enable rolling (windowed) regression metrics.
    ///
    /// Metrics will be computed over the last `window_size` evaluated samples.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is zero.
    pub fn with_rolling_window(mut self, window_size: usize) -> Self {
        self.rolling = Some(RollingRegressionMetrics::new(window_size));
        self
    }

    /// Enable EWMA (exponentially weighted) regression metrics.
    ///
    /// The decay factor is `alpha = 2 / (span + 1)`.
    ///
    /// # Panics
    ///
    /// Panics if `span` is zero.
    pub fn with_ewma(mut self, span: usize) -> Self {
        self.ewma = Some(EwmaRegressionMetrics::new(span));
        self
    }

    /// Process a single sample through the test-then-train protocol.
    ///
    /// 1. Predict using the current model state.
    /// 2. If past warmup and on the step interval, record metrics.
    /// 3. Train the model on this sample.
    /// 4. Return the prediction (made before training).
    ///
    /// The model **always** trains on every sample, regardless of warmup
    /// or step interval. Only metric recording is gated.
    pub fn step(&mut self, model: &mut dyn StreamingLearner, features: &[f64], target: f64) -> f64 {
        // 1. Predict BEFORE training (test-then-train)
        let prediction = model.predict(features);

        // 2. Record metrics if past warmup and on step interval
        let past_warmup = self.samples_seen >= self.config.warmup as u64;
        let on_step = self.config.step_interval == 1
            || (self.samples_seen % self.config.step_interval as u64 == 0);

        if past_warmup && on_step {
            self.regression.update(target, prediction);
            if let Some(ref mut rolling) = self.rolling {
                rolling.update(target, prediction);
            }
            if let Some(ref mut ewma) = self.ewma {
                ewma.update(target, prediction);
            }
            self.samples_evaluated += 1;
        }

        // 3. Train on this sample
        model.train(features, target);

        // 4. Increment counter
        self.samples_seen += 1;

        prediction
    }

    /// Run the full test-then-train protocol over a slice of observations.
    ///
    /// Iterates over `data`, calling [`step`](Self::step) for each observation.
    pub fn evaluate<O: Observation>(&mut self, model: &mut dyn StreamingLearner, data: &[O]) {
        for obs in data {
            self.step(model, obs.features(), obs.target());
        }
    }

    /// Reference to the cumulative regression metrics.
    pub fn regression_metrics(&self) -> &RegressionMetrics {
        &self.regression
    }

    /// Reference to the rolling regression metrics, if enabled.
    pub fn rolling_metrics(&self) -> Option<&RollingRegressionMetrics> {
        self.rolling.as_ref()
    }

    /// Reference to the EWMA regression metrics, if enabled.
    pub fn ewma_metrics(&self) -> Option<&EwmaRegressionMetrics> {
        self.ewma.as_ref()
    }

    /// Total number of samples processed (both warmup and evaluated).
    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Number of samples that were actually evaluated (past warmup, on step interval).
    pub fn samples_evaluated(&self) -> u64 {
        self.samples_evaluated
    }

    /// Reset all metrics and counters to the initial state.
    ///
    /// Configuration (warmup, step_interval, rolling window, EWMA span) is preserved.
    pub fn reset(&mut self) {
        self.regression.reset();
        if let Some(ref mut rolling) = self.rolling {
            rolling.reset();
        }
        if let Some(ref mut ewma) = self.ewma {
            ewma.reset();
        }
        self.samples_seen = 0;
        self.samples_evaluated = 0;
    }
}

impl Default for PrequentialEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::StreamingLearner;
    use crate::sample::Sample;

    /// Mock learner that always predicts a fixed constant.
    /// Useful for deterministic tests without importing heavy models.
    struct MockLearner {
        prediction: f64,
        n: u64,
    }

    impl MockLearner {
        fn new(prediction: f64) -> Self {
            Self { prediction, n: 0 }
        }
    }

    impl StreamingLearner for MockLearner {
        fn train_one(&mut self, _features: &[f64], _target: f64, _weight: f64) {
            self.n += 1;
        }

        fn predict(&self, _features: &[f64]) -> f64 {
            self.prediction
        }

        fn n_samples_seen(&self) -> u64 {
            self.n
        }

        fn reset(&mut self) {
            self.n = 0;
        }
    }

    #[test]
    fn basic_prequential_evaluation() {
        let mut model = MockLearner::new(0.0);
        let mut eval = PrequentialEvaluator::new();

        for i in 0..100 {
            eval.step(&mut model, &[i as f64], i as f64);
        }

        assert_eq!(eval.samples_seen(), 100);
        assert_eq!(eval.samples_evaluated(), 100);
        assert_eq!(eval.regression_metrics().n_samples(), 100);

        // Mock always predicts 0.0, targets are 0..99
        // MAE = mean(0, 1, 2, ..., 99) = 49.5
        let mae = eval.regression_metrics().mae();
        assert!((mae - 49.5).abs() < 0.01, "expected MAE ~49.5, got {}", mae);
    }

    #[test]
    fn warmup_skips_early_evaluation() {
        let mut model = MockLearner::new(0.0);
        let config = PrequentialConfig {
            warmup: 10,
            step_interval: 1,
        };
        let mut eval = PrequentialEvaluator::with_config(config);

        for i in 0..50 {
            eval.step(&mut model, &[i as f64], i as f64);
        }

        assert_eq!(eval.samples_seen(), 50);
        // Only samples 10..49 are evaluated = 40 samples
        assert_eq!(eval.samples_evaluated(), 40);
        assert_eq!(eval.regression_metrics().n_samples(), 40);

        // Model still trains on all 50 samples
        assert_eq!(model.n_samples_seen(), 50);
    }

    #[test]
    fn step_interval_evaluates_periodically() {
        let mut model = MockLearner::new(0.0);
        let config = PrequentialConfig {
            warmup: 0,
            step_interval: 3,
        };
        let mut eval = PrequentialEvaluator::with_config(config);

        for i in 0..30 {
            eval.step(&mut model, &[i as f64], i as f64);
        }

        assert_eq!(eval.samples_seen(), 30);
        // Evaluated at samples_seen=0,3,6,9,...,27 => 10 evaluations
        assert_eq!(eval.samples_evaluated(), 10);

        // Model still trains on all 30 samples
        assert_eq!(model.n_samples_seen(), 30);
    }

    #[test]
    fn rolling_window_tracks_metrics() {
        let mut model = MockLearner::new(0.0);
        let mut eval = PrequentialEvaluator::new().with_rolling_window(10);

        for i in 0..100 {
            eval.step(&mut model, &[i as f64], i as f64);
        }

        let rolling = eval.rolling_metrics().expect("rolling should be enabled");
        assert_eq!(rolling.n_samples(), 10);
        assert!(rolling.is_full());
        // Last 10 targets: 90..99, prediction=0 => errors = 90..99
        // MAE = (90+91+...+99)/10 = 94.5
        let rolling_mae = rolling.mae();
        assert!(
            (rolling_mae - 94.5).abs() < 0.01,
            "expected rolling MAE ~94.5, got {}",
            rolling_mae
        );
    }

    #[test]
    fn ewma_tracks_metrics() {
        let mut model = MockLearner::new(0.0);
        let mut eval = PrequentialEvaluator::new().with_ewma(20);

        for i in 0..100 {
            eval.step(&mut model, &[i as f64], i as f64);
        }

        let ewma = eval.ewma_metrics().expect("EWMA should be enabled");
        assert_eq!(ewma.n_samples(), 100);
        // EWMA MAE should be weighted toward recent errors (large values)
        assert!(
            ewma.mae() > 50.0,
            "EWMA MAE should be > 50, got {}",
            ewma.mae()
        );
    }

    #[test]
    fn no_rolling_or_ewma_by_default() {
        let eval = PrequentialEvaluator::new();
        assert!(eval.rolling_metrics().is_none());
        assert!(eval.ewma_metrics().is_none());
    }

    #[test]
    fn step_returns_prediction() {
        let mut model = MockLearner::new(42.0);
        let mut eval = PrequentialEvaluator::new();

        let pred = eval.step(&mut model, &[1.0], 5.0);
        assert!(
            (pred - 42.0).abs() < 1e-12,
            "step should return the model's prediction, got {}",
            pred
        );
    }

    #[test]
    fn evaluate_over_observations() {
        let mut model = MockLearner::new(0.0);
        let mut eval = PrequentialEvaluator::new();

        let data: Vec<Sample> = (0..50)
            .map(|i| Sample::new(vec![i as f64], i as f64))
            .collect();

        eval.evaluate(&mut model, &data);

        assert_eq!(eval.samples_seen(), 50);
        assert_eq!(eval.samples_evaluated(), 50);
        assert_eq!(model.n_samples_seen(), 50);
    }

    #[test]
    fn reset_clears_all_state() {
        let mut model = MockLearner::new(0.0);
        let mut eval = PrequentialEvaluator::new()
            .with_rolling_window(10)
            .with_ewma(20);

        for i in 0..50 {
            eval.step(&mut model, &[i as f64], i as f64);
        }

        assert!(eval.samples_seen() > 0);
        assert!(eval.samples_evaluated() > 0);

        eval.reset();

        assert_eq!(eval.samples_seen(), 0);
        assert_eq!(eval.samples_evaluated(), 0);
        assert_eq!(eval.regression_metrics().n_samples(), 0);
        assert_eq!(eval.rolling_metrics().unwrap().n_samples(), 0,);
        assert_eq!(eval.ewma_metrics().unwrap().n_samples(), 0,);
    }

    #[test]
    fn warmup_and_step_interval_combined() {
        let mut model = MockLearner::new(0.0);
        let config = PrequentialConfig {
            warmup: 5,
            step_interval: 2,
        };
        let mut eval = PrequentialEvaluator::with_config(config);

        // Feed 20 samples (indices 0..19)
        for i in 0..20 {
            eval.step(&mut model, &[i as f64], i as f64);
        }

        assert_eq!(eval.samples_seen(), 20);
        // Past warmup: samples_seen >= 5, so indices 5..19
        // On step interval (step_interval=2): samples_seen % 2 == 0
        // Indices 5..19 where index % 2 == 0: 6, 8, 10, 12, 14, 16, 18 = 7
        // Wait: samples_seen starts at 0 before incrementing.
        // At step(): samples_seen is checked BEFORE increment.
        // sample 0: samples_seen=0, past_warmup=false (0 < 5)
        // sample 1: samples_seen=1, past_warmup=false
        // ...
        // sample 5: samples_seen=5, past_warmup=true, 5%2=1 -> not on step
        // sample 6: samples_seen=6, past_warmup=true, 6%2=0 -> evaluated
        // sample 7: samples_seen=7, past_warmup=true, 7%2=1 -> not on step
        // sample 8: samples_seen=8, past_warmup=true, 8%2=0 -> evaluated
        // ...
        // Evaluated: 6,8,10,12,14,16,18 = 7 samples
        assert_eq!(eval.samples_evaluated(), 7);
        assert_eq!(model.n_samples_seen(), 20);
    }

    #[test]
    #[should_panic(expected = "step_interval must be > 0")]
    fn zero_step_interval_panics() {
        let config = PrequentialConfig {
            warmup: 0,
            step_interval: 0,
        };
        PrequentialEvaluator::with_config(config);
    }

    #[test]
    fn default_evaluator_is_new() {
        let eval = PrequentialEvaluator::default();
        assert_eq!(eval.samples_seen(), 0);
        assert_eq!(eval.samples_evaluated(), 0);
        assert!(eval.rolling_metrics().is_none());
        assert!(eval.ewma_metrics().is_none());
    }
}
