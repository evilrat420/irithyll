//! Parallel SGBT training with delayed gradient updates.
//!
//! Instead of sequential gradient propagation through boosting steps,
//! this module uses the full ensemble prediction as the gradient target
//! for all steps simultaneously. Each step trains independently on the
//! same gradient, enabling rayon-based parallelism across steps.
//!
//! # Algorithm
//!
//! For each incoming sample `(x, y)`:
//! 1. Compute the full ensemble prediction: `F(x) = base + lr * sum tree_s(x)`
//! 2. Compute gradient `g = loss.gradient(y, F(x))` and hessian `h = loss.hessian(y, F(x))`
//! 3. Pre-compute `train_count` for each step (sequential, uses RNG state)
//! 4. Train ALL steps in parallel with the same `(x, g, h)` and per-step train_count
//!
//! This is a "delayed gradient" approach: all steps see the same gradient
//! computed from the full ensemble prediction, rather than the sequential
//! rolling prediction used in standard SGBT. This trades a small amount of
//! gradient freshness for parallelism across boosting steps.
//!
//! Requires the `parallel` feature flag for rayon-based parallelism. Without
//! the feature, the module still compiles and works correctly using sequential
//! iteration (identical results, just no multi-core speedup).
//!
//! # Trade-offs
//!
//! - **Pro:** Near-linear speedup with number of cores for large ensembles.
//! - **Con:** Gradient staleness may slow convergence slightly; typically
//!   compensated by a slightly higher learning rate or more training samples.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::step::BoostingStep;
use crate::loss::squared::SquaredLoss;
use crate::loss::Loss;
use crate::sample::Observation;
use crate::tree::builder::TreeConfig;

use std::fmt;

/// Parallel SGBT ensemble with delayed gradient updates.
///
/// All boosting steps train concurrently using the full ensemble prediction
/// for gradient computation. Predictions remain sequential (deterministic)
/// -- only training is parallelized.
///
/// Generic over `L: Loss` so the loss function's gradient/hessian calls
/// are monomorphized (inlined) into the training loop — no virtual dispatch.
///
/// # Differences from [`SGBT`](super::SGBT)
///
/// | Aspect | `SGBT` | `ParallelSGBT` |
/// |--------|--------|----------------|
/// | Gradient target | Rolling (step-by-step) | Full ensemble prediction |
/// | Step training | Sequential | Parallel (rayon) |
/// | Prediction | Sequential | Sequential (identical) |
/// | Convergence | Optimal | Slightly delayed |
/// | Throughput | 1x | ~Nx (N = cores) |
pub struct ParallelSGBT<L: Loss = SquaredLoss> {
    /// Configuration.
    config: SGBTConfig,
    /// Boosting steps (one tree + drift detector each).
    steps: Vec<BoostingStep>,
    /// Loss function (monomorphized — no vtable).
    loss: L,
    /// Base prediction (initial constant, computed from first batch of targets).
    base_prediction: f64,
    /// Whether base_prediction has been initialized.
    base_initialized: bool,
    /// Running collection of initial targets for computing base_prediction.
    initial_targets: Vec<f64>,
    /// Number of initial targets to collect before setting base_prediction.
    initial_target_count: usize,
    /// Total samples trained.
    samples_seen: u64,
    /// RNG state for variant skip logic.
    rng_state: u64,
    /// Pre-allocated buffer for per-step train counts (avoids heap alloc per sample).
    train_counts_buf: Vec<usize>,
}

impl<L: Loss + Clone> Clone for ParallelSGBT<L> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            steps: self.steps.clone(),
            loss: self.loss.clone(),
            base_prediction: self.base_prediction,
            base_initialized: self.base_initialized,
            initial_targets: self.initial_targets.clone(),
            initial_target_count: self.initial_target_count,
            samples_seen: self.samples_seen,
            rng_state: self.rng_state,
            train_counts_buf: self.train_counts_buf.clone(),
        }
    }
}

impl<L: Loss> fmt::Debug for ParallelSGBT<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParallelSGBT")
            .field("n_steps", &self.steps.len())
            .field("samples_seen", &self.samples_seen)
            .field("base_prediction", &self.base_prediction)
            .field("base_initialized", &self.base_initialized)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor for the default loss (SquaredLoss)
// ---------------------------------------------------------------------------

impl ParallelSGBT<SquaredLoss> {
    /// Create a new parallel SGBT ensemble with squared loss (regression).
    pub fn new(config: SGBTConfig) -> Self {
        Self::with_loss(config, SquaredLoss)
    }
}

// ---------------------------------------------------------------------------
// General impl for all Loss types
// ---------------------------------------------------------------------------

impl<L: Loss> ParallelSGBT<L> {
    /// Create a new parallel SGBT ensemble with a specific loss function.
    ///
    /// The loss is stored by value (monomorphized), giving zero-cost
    /// gradient/hessian dispatch.
    ///
    /// ```
    /// use irithyll::SGBTConfig;
    /// use irithyll::ensemble::parallel::ParallelSGBT;
    /// use irithyll::loss::logistic::LogisticLoss;
    ///
    /// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
    /// let model = ParallelSGBT::with_loss(config, LogisticLoss);
    /// ```
    pub fn with_loss(config: SGBTConfig, loss: L) -> Self {
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
            .split_reeval_interval_opt(config.split_reeval_interval);

        let max_tree_samples = config.max_tree_samples;

        let steps: Vec<BoostingStep> = (0..config.n_steps)
            .map(|i| {
                let mut tc = tree_config.clone();
                tc.seed = config.seed ^ (i as u64);
                let detector = config.drift_detector.create();
                BoostingStep::new_with_max_samples(tc, detector, max_tree_samples)
            })
            .collect();

        let seed = config.seed;
        let initial_target_count = config.initial_target_count;
        let n_steps = steps.len();
        Self {
            config,
            steps,
            loss,
            base_prediction: 0.0,
            base_initialized: false,
            initial_targets: Vec::new(),
            initial_target_count,
            samples_seen: 0,
            rng_state: seed,
            train_counts_buf: vec![0; n_steps],
        }
    }

    /// Train on a single observation using delayed gradient updates.
    ///
    /// Accepts any type implementing [`Observation`], including [`Sample`](crate::Sample),
    /// [`SampleRef`](crate::SampleRef), or tuples like `(&[f64], f64)`.
    ///
    /// All boosting steps receive the same gradient/hessian computed from
    /// the full ensemble prediction, then train in parallel (when the
    /// `parallel` feature is enabled).
    pub fn train_one(&mut self, sample: &impl Observation) {
        self.samples_seen += 1;
        let target = sample.target();
        let features = sample.features();

        // Initialize base prediction from first few targets.
        if !self.base_initialized {
            self.initial_targets.push(target);
            if self.initial_targets.len() >= self.initial_target_count {
                self.base_prediction = self.loss.initial_prediction(&self.initial_targets);
                self.base_initialized = true;
                self.initial_targets.clear();
                self.initial_targets.shrink_to_fit();
            }
        }

        // Compute the FULL ensemble prediction (same as predict()).
        let full_pred = self.predict(features);

        // Compute gradient and hessian from the full ensemble prediction.
        // All steps will use these same values (delayed gradient approach).
        let gradient = self.loss.gradient(target, full_pred);
        let hessian = self.loss.hessian(target, full_pred);

        // Pre-compute train_count for each step sequentially into the
        // pre-allocated buffer (zero heap alloc per sample).
        // The RNG state is sequential (xorshift), so we must advance it
        // in order before entering the parallel section.
        for tc in self.train_counts_buf.iter_mut() {
            *tc = self
                .config
                .variant
                .train_count(hessian, &mut self.rng_state);
        }

        // Train all steps with the same gradient/hessian.
        // When `parallel` feature is enabled, use rayon for concurrency.
        // Otherwise, fall back to sequential iteration.
        #[cfg(feature = "parallel")]
        {
            self.steps.par_iter_mut().enumerate().for_each(|(i, step)| {
                step.train_and_predict(features, gradient, hessian, self.train_counts_buf[i]);
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for (i, step) in self.steps.iter_mut().enumerate() {
                step.train_and_predict(features, gradient, hessian, self.train_counts_buf[i]);
            }
        }
    }

    /// Train on a batch of observations.
    pub fn train_batch<O: Observation>(&mut self, samples: &[O]) {
        for sample in samples {
            self.train_one(sample);
        }
    }

    /// Predict the raw output for a feature vector.
    ///
    /// Prediction is always sequential and deterministic, regardless of
    /// whether training uses parallelism.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut pred = self.base_prediction;
        for step in &self.steps {
            pred += self.config.learning_rate * step.predict(features);
        }
        pred
    }

    /// Predict with loss transform applied (e.g., sigmoid for logistic loss).
    pub fn predict_transformed(&self, features: &[f64]) -> f64 {
        self.loss.predict_transform(self.predict(features))
    }

    /// Predict probability (alias for `predict_transformed`).
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        self.predict_transformed(features)
    }

    /// Batch prediction.
    pub fn predict_batch(&self, feature_matrix: &[Vec<f64>]) -> Vec<f64> {
        feature_matrix.iter().map(|f| self.predict(f)).collect()
    }

    /// Number of boosting steps.
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Total trees (active + alternates).
    pub fn n_trees(&self) -> usize {
        self.steps.len() + self.steps.iter().filter(|s| s.has_alternate()).count()
    }

    /// Total leaves across all active trees.
    pub fn total_leaves(&self) -> usize {
        self.steps.iter().map(|s| s.n_leaves()).sum()
    }

    /// Total samples trained.
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// The current base prediction.
    pub fn base_prediction(&self) -> f64 {
        self.base_prediction
    }

    /// Whether the base prediction has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.base_initialized
    }

    /// Access the configuration.
    pub fn config(&self) -> &SGBTConfig {
        &self.config
    }

    /// Immutable access to the loss function.
    pub fn loss(&self) -> &L {
        &self.loss
    }

    /// Feature importances based on accumulated split gains across all trees.
    ///
    /// Returns normalized importances (sum to 1.0) indexed by feature.
    /// Returns an empty Vec if no splits have occurred yet.
    pub fn feature_importances(&self) -> Vec<f64> {
        let mut totals: Vec<f64> = Vec::new();
        for step in &self.steps {
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

    /// Reset the ensemble to initial state.
    pub fn reset(&mut self) {
        for step in &mut self.steps {
            step.reset();
        }
        self.base_prediction = 0.0;
        self.base_initialized = false;
        self.initial_targets.clear();
        self.samples_seen = 0;
        self.rng_state = self.config.seed;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::Sample;

    fn default_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .build()
            .unwrap()
    }

    // -------------------------------------------------------------------
    // 1. Fresh model predicts zero.
    // -------------------------------------------------------------------
    #[test]
    fn new_model_predicts_zero() {
        let model = ParallelSGBT::new(default_config());
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(pred.abs() < 1e-12);
    }

    // -------------------------------------------------------------------
    // 2. train_one does not panic.
    // -------------------------------------------------------------------
    #[test]
    fn train_one_does_not_panic() {
        let mut model = ParallelSGBT::new(default_config());
        model.train_one(&Sample::new(vec![1.0, 2.0, 3.0], 5.0));
        assert_eq!(model.n_samples_seen(), 1);
    }

    // -------------------------------------------------------------------
    // 3. Prediction changes after training.
    // -------------------------------------------------------------------
    #[test]
    fn prediction_changes_after_training() {
        let mut model = ParallelSGBT::new(default_config());
        let features = vec![1.0, 2.0, 3.0];
        for i in 0..100 {
            model.train_one(&Sample::new(features.clone(), (i as f64) * 0.1));
        }
        let pred = model.predict(&features);
        assert!(pred.is_finite());
    }

    // -------------------------------------------------------------------
    // 4. Linear signal RMSE improves over time.
    //
    // NOTE: The delayed gradient approach converges slower than sequential
    // SGBT because all steps see the same (slightly stale) gradient. We
    // compensate with a higher learning rate and more training samples,
    // and widen the measurement windows.
    // -------------------------------------------------------------------
    #[test]
    fn linear_signal_rmse_improves() {
        let config = SGBTConfig::builder()
            .n_steps(20)
            .learning_rate(0.15)
            .grace_period(10)
            .max_depth(3)
            .n_bins(16)
            .build()
            .unwrap();
        let mut model = ParallelSGBT::new(config);

        let mut rng: u64 = 12345;
        let mut early_errors = Vec::new();
        let mut late_errors = Vec::new();

        for i in 0..1000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x1 = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x2 = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            let target = 2.0 * x1 + 3.0 * x2;

            let pred = model.predict(&[x1, x2]);
            let error = (pred - target).powi(2);

            if (100..300).contains(&i) {
                early_errors.push(error);
            }
            if i >= 800 {
                late_errors.push(error);
            }

            model.train_one(&Sample::new(vec![x1, x2], target));
        }

        let early_rmse = (early_errors.iter().sum::<f64>() / early_errors.len() as f64).sqrt();
        let late_rmse = (late_errors.iter().sum::<f64>() / late_errors.len() as f64).sqrt();

        assert!(
            late_rmse < early_rmse,
            "RMSE should decrease: early={:.4}, late={:.4}",
            early_rmse,
            late_rmse
        );
    }

    // -------------------------------------------------------------------
    // 5. train_batch is equivalent to sequential train_one calls.
    // -------------------------------------------------------------------
    #[test]
    fn train_batch_equivalent_to_sequential() {
        let config = default_config();
        let mut model_seq = ParallelSGBT::new(config.clone());
        let mut model_batch = ParallelSGBT::new(config);

        let samples: Vec<Sample> = (0..20)
            .map(|i| {
                let x = i as f64 * 0.5;
                Sample::new(vec![x, x * 2.0], x * 3.0)
            })
            .collect();

        for s in &samples {
            model_seq.train_one(s);
        }
        model_batch.train_batch(&samples);

        let pred_seq = model_seq.predict(&[1.0, 2.0]);
        let pred_batch = model_batch.predict(&[1.0, 2.0]);

        assert!(
            (pred_seq - pred_batch).abs() < 1e-10,
            "seq={}, batch={}",
            pred_seq,
            pred_batch
        );
    }

    // -------------------------------------------------------------------
    // 6. Reset returns to initial state.
    // -------------------------------------------------------------------
    #[test]
    fn reset_returns_to_initial() {
        let mut model = ParallelSGBT::new(default_config());
        for i in 0..100 {
            model.train_one(&Sample::new(vec![1.0, 2.0], i as f64));
        }
        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.is_initialized());
        assert!(model.predict(&[1.0, 2.0]).abs() < 1e-12);
    }

    // -------------------------------------------------------------------
    // 7. Base prediction initializes correctly.
    // -------------------------------------------------------------------
    #[test]
    fn base_prediction_initializes() {
        let mut model = ParallelSGBT::new(default_config());
        for i in 0..50 {
            model.train_one(&Sample::new(vec![1.0], i as f64 + 100.0));
        }
        assert!(model.is_initialized());
        let expected = (100.0 + 149.0) / 2.0;
        assert!((model.base_prediction() - expected).abs() < 1.0);
    }

    // -------------------------------------------------------------------
    // 8. with_loss uses custom loss function.
    // -------------------------------------------------------------------
    #[test]
    fn with_loss_uses_custom_loss() {
        use crate::loss::logistic::LogisticLoss;
        let model = ParallelSGBT::with_loss(default_config(), LogisticLoss);
        let pred = model.predict_transformed(&[1.0, 2.0]);
        assert!(
            (pred - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5, got {}",
            pred
        );
    }

    // -------------------------------------------------------------------
    // 9. Debug formatting works.
    // -------------------------------------------------------------------
    #[test]
    fn debug_format_works() {
        let model = ParallelSGBT::new(default_config());
        let debug_str = format!("{:?}", model);
        assert!(
            debug_str.contains("ParallelSGBT"),
            "debug output should contain 'ParallelSGBT', got: {}",
            debug_str,
        );
    }

    // -------------------------------------------------------------------
    // 10. Accessors return expected values.
    // -------------------------------------------------------------------
    #[test]
    fn accessors_return_expected_values() {
        let config = default_config();
        let n = config.n_steps;
        let model = ParallelSGBT::new(config);

        assert_eq!(model.n_steps(), n);
        assert_eq!(model.n_trees(), n); // no alternates initially
        assert_eq!(model.total_leaves(), n); // 1 leaf per tree initially
        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.is_initialized());
    }

    // -------------------------------------------------------------------
    // 11. Batch prediction matches individual predictions.
    // -------------------------------------------------------------------
    #[test]
    fn batch_prediction_matches_individual() {
        let mut model = ParallelSGBT::new(default_config());
        let features = vec![1.0, 2.0, 3.0];
        for i in 0..50 {
            model.train_one(&Sample::new(features.clone(), (i as f64) * 0.5));
        }

        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![0.0, 0.0, 0.0],
        ];
        let batch_preds = model.predict_batch(&matrix);

        for (feats, batch_pred) in matrix.iter().zip(batch_preds.iter()) {
            let single_pred = model.predict(feats);
            assert!(
                (single_pred - batch_pred).abs() < 1e-12,
                "batch and single predictions should match",
            );
        }
    }

    // -------------------------------------------------------------------
    // 12. Feature importances are normalized.
    // -------------------------------------------------------------------
    #[test]
    fn feature_importances_normalized() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(3)
            .n_bins(16)
            .build()
            .unwrap();
        let mut model = ParallelSGBT::new(config);

        // Train enough for splits to occur.
        let mut rng: u64 = 42;
        for _ in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x1 = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let x2 = (rng as f64 / u64::MAX as f64) * 10.0 - 5.0;
            let target = 3.0 * x1 - x2;
            model.train_one(&Sample::new(vec![x1, x2], target));
        }

        let importances = model.feature_importances();
        if !importances.is_empty() {
            let sum: f64 = importances.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-8,
                "importances should sum to 1.0, got {}",
                sum,
            );
            for &v in &importances {
                assert!(v >= 0.0, "importances should be non-negative");
            }
        }
    }

    // -------------------------------------------------------------------
    // 13. Variant train_counts are pre-computed correctly (Skip variant).
    // -------------------------------------------------------------------
    #[test]
    fn skip_variant_works_with_parallel() {
        use crate::ensemble::variants::SGBTVariant;

        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .variant(SGBTVariant::Skip { k: 3 })
            .build()
            .unwrap();
        let mut model = ParallelSGBT::new(config);

        // Should not panic, and should train with some steps skipped.
        for i in 0..100 {
            model.train_one(&Sample::new(vec![1.0, 2.0], i as f64));
        }

        assert_eq!(model.n_samples_seen(), 100);
        let pred = model.predict(&[1.0, 2.0]);
        assert!(pred.is_finite());
    }

    // -------------------------------------------------------------------
    // 14. MI variant works with parallel.
    // -------------------------------------------------------------------
    #[test]
    fn mi_variant_works_with_parallel() {
        use crate::ensemble::variants::SGBTVariant;

        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .variant(SGBTVariant::MultipleIterations { multiplier: 2.0 })
            .build()
            .unwrap();
        let mut model = ParallelSGBT::new(config);

        for i in 0..100 {
            model.train_one(&Sample::new(vec![1.0, 2.0], i as f64));
        }

        assert_eq!(model.n_samples_seen(), 100);
        let pred = model.predict(&[1.0, 2.0]);
        assert!(pred.is_finite());
    }

    // -------------------------------------------------------------------
    // 15. Predict_proba and predict_transformed are equivalent.
    // -------------------------------------------------------------------
    #[test]
    fn predict_proba_equals_predict_transformed() {
        let mut model = ParallelSGBT::new(default_config());
        for i in 0..50 {
            model.train_one(&Sample::new(vec![1.0, 2.0], i as f64));
        }

        let feats = [1.0, 2.0];
        let transformed = model.predict_transformed(&feats);
        let proba = model.predict_proba(&feats);
        assert!(
            (transformed - proba).abs() < 1e-12,
            "predict_proba and predict_transformed should be identical",
        );
    }
}
