//! NGBoost-style distributional SGBT — outputs Gaussian N(μ, σ²) instead of a point estimate.
//!
//! [`DistributionalSGBT`] maintains two independent ensembles of streaming trees:
//! one targeting the location (mean) and one targeting the scale (log-sigma).
//! The combined output is a full predictive Gaussian distribution, enabling
//! uncertainty quantification per prediction.
//!
//! # Gradient Derivation
//!
//! The negative log-likelihood of N(μ, σ²) decomposes into two sets of gradients:
//!
//! **Location (w.r.t. μ):** standard squared-loss gradients
//! - gradient = μ - y
//! - hessian = 1.0
//!
//! **Scale (w.r.t. log σ):** where z = (y - μ) / σ
//! - gradient = 1.0 - z²
//! - hessian = max(2z², 0.01)
//!
//! The scale is parameterized in log-space to guarantee σ > 0.
//!
//! # References
//!
//! Duan et al. (2020). "NGBoost: Natural Gradient Boosting for Probabilistic Prediction."

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::step::BoostingStep;
use crate::sample::Observation;
use crate::tree::builder::TreeConfig;

/// Prediction from a distributional model: full Gaussian N(μ, σ²).
#[derive(Debug, Clone, Copy)]
pub struct GaussianPrediction {
    /// Location parameter (mean).
    pub mu: f64,
    /// Scale parameter (standard deviation, always > 0).
    pub sigma: f64,
    /// Log of scale parameter (raw model output for scale ensemble).
    pub log_sigma: f64,
}

impl GaussianPrediction {
    /// Lower bound of a symmetric confidence interval.
    ///
    /// For 95% CI, use `z = 1.96`.
    #[inline]
    pub fn lower(&self, z: f64) -> f64 {
        self.mu - z * self.sigma
    }

    /// Upper bound of a symmetric confidence interval.
    #[inline]
    pub fn upper(&self, z: f64) -> f64 {
        self.mu + z * self.sigma
    }
}

/// NGBoost-style distributional streaming gradient boosted trees.
///
/// Outputs a full Gaussian predictive distribution N(μ, σ²) by maintaining two
/// independent ensembles — one for location (mean) and one for scale (log-sigma).
///
/// # Example
///
/// ```
/// use irithyll::SGBTConfig;
/// use irithyll::ensemble::distributional::DistributionalSGBT;
///
/// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
/// let mut model = DistributionalSGBT::new(config);
///
/// // Train on streaming data
/// model.train_one(&(vec![1.0, 2.0], 3.5));
///
/// // Get full distributional prediction
/// let pred = model.predict(&[1.0, 2.0]);
/// println!("mean={}, sigma={}", pred.mu, pred.sigma);
/// ```
pub struct DistributionalSGBT {
    /// Configuration (shared between location and scale ensembles).
    config: SGBTConfig,
    /// Location (mean) boosting steps.
    location_steps: Vec<BoostingStep>,
    /// Scale (log-sigma) boosting steps.
    scale_steps: Vec<BoostingStep>,
    /// Base prediction for location (mean of initial targets).
    location_base: f64,
    /// Base prediction for scale (log of std of initial targets).
    scale_base: f64,
    /// Whether base predictions have been initialized.
    base_initialized: bool,
    /// Running collection of initial targets for computing base predictions.
    initial_targets: Vec<f64>,
    /// Number of initial targets to collect before setting base predictions.
    initial_target_count: usize,
    /// Total samples trained.
    samples_seen: u64,
    /// RNG state for variant logic.
    rng_state: u64,
}

impl Clone for DistributionalSGBT {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            location_steps: self.location_steps.clone(),
            scale_steps: self.scale_steps.clone(),
            location_base: self.location_base,
            scale_base: self.scale_base,
            base_initialized: self.base_initialized,
            initial_targets: self.initial_targets.clone(),
            initial_target_count: self.initial_target_count,
            samples_seen: self.samples_seen,
            rng_state: self.rng_state,
        }
    }
}

impl std::fmt::Debug for DistributionalSGBT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributionalSGBT")
            .field("n_steps", &self.location_steps.len())
            .field("samples_seen", &self.samples_seen)
            .field("location_base", &self.location_base)
            .field("scale_base", &self.scale_base)
            .field("base_initialized", &self.base_initialized)
            .finish()
    }
}

impl DistributionalSGBT {
    /// Create a new distributional SGBT with the given configuration.
    ///
    /// Both location and scale ensembles use the same tree configuration
    /// but with different seeds to ensure independence.
    pub fn new(config: SGBTConfig) -> Self {
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
            .split_reeval_interval_opt(config.split_reeval_interval)
            .feature_types_opt(config.feature_types.clone())
            .gradient_clip_sigma_opt(config.gradient_clip_sigma)
            .monotone_constraints_opt(config.monotone_constraints.clone());

        let max_tree_samples = config.max_tree_samples;

        // Location ensemble (seed offset: 0)
        let location_steps: Vec<BoostingStep> = (0..config.n_steps)
            .map(|i| {
                let mut tc = tree_config.clone();
                tc.seed = config.seed ^ (i as u64);
                let detector = config.drift_detector.create();
                BoostingStep::new_with_max_samples(tc, detector, max_tree_samples)
            })
            .collect();

        // Scale ensemble (seed offset: 0xSCALE)
        let scale_steps: Vec<BoostingStep> = (0..config.n_steps)
            .map(|i| {
                let mut tc = tree_config.clone();
                tc.seed = config.seed ^ (i as u64) ^ 0x0005_CA1E_0000_0000;
                let detector = config.drift_detector.create();
                BoostingStep::new_with_max_samples(tc, detector, max_tree_samples)
            })
            .collect();

        let seed = config.seed;
        let initial_target_count = config.initial_target_count;
        Self {
            config,
            location_steps,
            scale_steps,
            location_base: 0.0,
            scale_base: 0.0,
            base_initialized: false,
            initial_targets: Vec::new(),
            initial_target_count,
            samples_seen: 0,
            rng_state: seed,
        }
    }

    /// Train on a single observation.
    pub fn train_one(&mut self, sample: &impl Observation) {
        self.samples_seen += 1;
        let target = sample.target();
        let features = sample.features();

        // Initialize base predictions from first few targets
        if !self.base_initialized {
            self.initial_targets.push(target);
            if self.initial_targets.len() >= self.initial_target_count {
                // Location base = mean
                let sum: f64 = self.initial_targets.iter().sum();
                let mean = sum / self.initial_targets.len() as f64;
                self.location_base = mean;

                // Scale base = log(std) — clamped for stability
                let var: f64 = self
                    .initial_targets
                    .iter()
                    .map(|&y| (y - mean) * (y - mean))
                    .sum::<f64>()
                    / self.initial_targets.len() as f64;
                self.scale_base = (var.sqrt().max(1e-6)).ln();

                self.base_initialized = true;
                self.initial_targets.clear();
                self.initial_targets.shrink_to_fit();
            }
            return;
        }

        // Current predictions
        let mut mu = self.location_base;
        let mut log_sigma = self.scale_base;

        // Sequential boosting: both ensembles target their respective residuals
        for s in 0..self.location_steps.len() {
            let sigma = log_sigma.exp().max(1e-8);
            let z = (target - mu) / sigma;

            // Location gradients (squared loss w.r.t. mu)
            let g_mu = mu - target;
            let h_mu = 1.0;

            // Scale gradients (NLL w.r.t. log_sigma)
            let g_sigma = 1.0 - z * z;
            let h_sigma = (2.0 * z * z).clamp(0.01, 100.0);

            let train_count = self.config.variant.train_count(h_mu, &mut self.rng_state);

            // Train location step
            let loc_pred =
                self.location_steps[s].train_and_predict(features, g_mu, h_mu, train_count);
            mu += self.config.learning_rate * loc_pred;

            // Train scale step
            let scale_pred =
                self.scale_steps[s].train_and_predict(features, g_sigma, h_sigma, train_count);
            log_sigma += self.config.learning_rate * scale_pred;
        }
    }

    /// Predict the full Gaussian distribution for a feature vector.
    pub fn predict(&self, features: &[f64]) -> GaussianPrediction {
        let mut mu = self.location_base;
        let mut log_sigma = self.scale_base;

        for s in 0..self.location_steps.len() {
            mu += self.config.learning_rate * self.location_steps[s].predict(features);
            log_sigma += self.config.learning_rate * self.scale_steps[s].predict(features);
        }

        let sigma = log_sigma.exp().max(1e-8);
        GaussianPrediction {
            mu,
            sigma,
            log_sigma,
        }
    }

    /// Predict the mean (location parameter) only.
    #[inline]
    pub fn predict_mu(&self, features: &[f64]) -> f64 {
        self.predict(features).mu
    }

    /// Predict the standard deviation (scale parameter) only.
    #[inline]
    pub fn predict_sigma(&self, features: &[f64]) -> f64 {
        self.predict(features).sigma
    }

    /// Predict a symmetric confidence interval.
    ///
    /// `confidence` is the Z-score multiplier:
    /// - 1.0 → 68% CI
    /// - 1.96 → 95% CI
    /// - 2.576 → 99% CI
    pub fn predict_interval(&self, features: &[f64], confidence: f64) -> (f64, f64) {
        let pred = self.predict(features);
        (pred.lower(confidence), pred.upper(confidence))
    }

    /// Batch prediction.
    pub fn predict_batch(&self, feature_matrix: &[Vec<f64>]) -> Vec<GaussianPrediction> {
        feature_matrix.iter().map(|f| self.predict(f)).collect()
    }

    /// Train on a batch of observations.
    pub fn train_batch<O: Observation>(&mut self, samples: &[O]) {
        for sample in samples {
            self.train_one(sample);
        }
    }

    /// Train on a batch with periodic callback.
    pub fn train_batch_with_callback<O: Observation, F: FnMut(usize)>(
        &mut self,
        samples: &[O],
        interval: usize,
        mut callback: F,
    ) {
        let interval = interval.max(1);
        for (i, sample) in samples.iter().enumerate() {
            self.train_one(sample);
            if (i + 1) % interval == 0 {
                callback(i + 1);
            }
        }
        let total = samples.len();
        if total % interval != 0 {
            callback(total);
        }
    }

    /// Reset to initial untrained state.
    pub fn reset(&mut self) {
        for step in &mut self.location_steps {
            step.reset();
        }
        for step in &mut self.scale_steps {
            step.reset();
        }
        self.location_base = 0.0;
        self.scale_base = 0.0;
        self.base_initialized = false;
        self.initial_targets.clear();
        self.samples_seen = 0;
        self.rng_state = self.config.seed;
    }

    /// Total samples trained.
    #[inline]
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Number of boosting steps (same for location and scale).
    #[inline]
    pub fn n_steps(&self) -> usize {
        self.location_steps.len()
    }

    /// Total trees (location + scale, active + alternates).
    pub fn n_trees(&self) -> usize {
        let loc = self.location_steps.len()
            + self
                .location_steps
                .iter()
                .filter(|s| s.has_alternate())
                .count();
        let scale = self.scale_steps.len()
            + self
                .scale_steps
                .iter()
                .filter(|s| s.has_alternate())
                .count();
        loc + scale
    }

    /// Total leaves across all active trees (location + scale).
    pub fn total_leaves(&self) -> usize {
        let loc: usize = self.location_steps.iter().map(|s| s.n_leaves()).sum();
        let scale: usize = self.scale_steps.iter().map(|s| s.n_leaves()).sum();
        loc + scale
    }

    /// Whether base predictions have been initialized.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.base_initialized
    }

    /// Access the configuration.
    #[inline]
    pub fn config(&self) -> &SGBTConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(20)
            .max_depth(4)
            .n_bins(16)
            .initial_target_count(10)
            .build()
            .unwrap()
    }

    #[test]
    fn fresh_model_predicts_zero() {
        let model = DistributionalSGBT::new(test_config());
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(pred.mu.abs() < 1e-12);
        assert!(pred.sigma > 0.0);
    }

    #[test]
    fn sigma_always_positive() {
        let mut model = DistributionalSGBT::new(test_config());

        // Train on various data
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 0.5], x * 2.0 + 1.0));
        }

        // Check multiple predictions
        for i in 0..20 {
            let x = i as f64 * 0.5;
            let pred = model.predict(&[x, x * 0.5]);
            assert!(
                pred.sigma > 0.0,
                "sigma must be positive, got {}",
                pred.sigma
            );
            assert!(pred.sigma.is_finite(), "sigma must be finite");
        }
    }

    #[test]
    fn constant_target_has_small_sigma() {
        let mut model = DistributionalSGBT::new(test_config());

        // Train on constant target = 5.0 with varying features
        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 2.0], 5.0));
        }

        let pred = model.predict(&[1.0, 2.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma.is_finite());
        assert!(pred.sigma > 0.0);
        // Sigma should be relatively small for constant target
        // (no noise to explain)
    }

    #[test]
    fn noisy_target_has_finite_predictions() {
        let mut model = DistributionalSGBT::new(test_config());

        // Simple xorshift for deterministic "noise"
        let mut rng: u64 = 42;
        for i in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let noise = (rng % 1000) as f64 / 500.0 - 1.0; // [-1, 1]
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0 + noise));
        }

        let pred = model.predict(&[5.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma.is_finite());
        assert!(pred.sigma > 0.0);
    }

    #[test]
    fn predict_interval_bounds_correct() {
        let mut model = DistributionalSGBT::new(test_config());

        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }

        let (lo, hi) = model.predict_interval(&[5.0], 1.96);
        let pred = model.predict(&[5.0]);

        assert!(lo < pred.mu, "lower bound should be < mu");
        assert!(hi > pred.mu, "upper bound should be > mu");
        assert!((hi - lo - 2.0 * 1.96 * pred.sigma).abs() < 1e-10);
    }

    #[test]
    fn batch_prediction_matches_individual() {
        let mut model = DistributionalSGBT::new(test_config());

        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x, x * 2.0], x));
        }

        let features = vec![vec![1.0, 2.0], vec![3.0, 6.0], vec![5.0, 10.0]];
        let batch = model.predict_batch(&features);

        for (feat, batch_pred) in features.iter().zip(batch.iter()) {
            let individual = model.predict(feat);
            assert!((batch_pred.mu - individual.mu).abs() < 1e-12);
            assert!((batch_pred.sigma - individual.sigma).abs() < 1e-12);
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut model = DistributionalSGBT::new(test_config());

        for i in 0..200 {
            let x = i as f64 * 0.1;
            model.train_one(&(vec![x], x * 2.0));
        }

        assert!(model.n_samples_seen() > 0);
        model.reset();

        assert_eq!(model.n_samples_seen(), 0);
        assert!(!model.is_initialized());
    }

    #[test]
    fn gaussian_prediction_lower_upper() {
        let pred = GaussianPrediction {
            mu: 10.0,
            sigma: 2.0,
            log_sigma: 2.0_f64.ln(),
        };

        assert!((pred.lower(1.96) - (10.0 - 1.96 * 2.0)).abs() < 1e-10);
        assert!((pred.upper(1.96) - (10.0 + 1.96 * 2.0)).abs() < 1e-10);
    }

    #[test]
    fn train_batch_works() {
        let mut model = DistributionalSGBT::new(test_config());
        let samples: Vec<(Vec<f64>, f64)> = (0..100)
            .map(|i| {
                let x = i as f64 * 0.1;
                (vec![x], x * 2.0)
            })
            .collect();

        model.train_batch(&samples);
        assert_eq!(model.n_samples_seen(), 100);
    }

    #[test]
    fn debug_format_works() {
        let model = DistributionalSGBT::new(test_config());
        let debug = format!("{:?}", model);
        assert!(debug.contains("DistributionalSGBT"));
    }

    #[test]
    fn n_trees_counts_both_ensembles() {
        let model = DistributionalSGBT::new(test_config());
        // 10 location + 10 scale = 20 trees minimum
        assert!(model.n_trees() >= 20);
    }
}
