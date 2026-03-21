//! Adaptive Random Forest (ARF) for streaming classification.
//!
//! An ensemble of streaming learners with ADWIN-based drift detection and
//! automatic tree replacement. Each member trains on a Poisson(lambda)-weighted
//! bootstrap of the stream with a random feature subspace.
//!
//! # Algorithm
//!
//! For each new sample *(x, y)* and each tree *t*:
//!
//! 1. Draw *k ~ Poisson(lambda)* -- the bootstrap weight.
//! 2. Predict *y_hat = tree_t(x\[mask_t\])* before training (for drift detection).
//! 3. Train *k* times on the masked feature vector.
//! 4. Feed correctness (0.0 = correct, 1.0 = incorrect) to the ADWIN detector.
//! 5. On drift: reset the tree and detector, allowing a fresh learner to adapt
//!    to the new distribution.
//!
//! Final prediction is majority vote across all trees.
//!
//! # Reference
//!
//! Gomes, H. M., et al. (2017). "Adaptive random forests for evolving data
//! stream classification." *Machine Learning*, 106(9-10), 1469-1495.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::drift::adwin::Adwin;
use crate::drift::{DriftDetector, DriftSignal};
use crate::learner::StreamingLearner;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn poisson(lambda: f64, rng: &mut u64) -> u64 {
    let l = crate::math::exp(-lambda);
    let mut k = 0u64;
    let mut p = 1.0f64;
    loop {
        k += 1;
        let u = xorshift64(rng) as f64 / u64::MAX as f64;
        p *= u;
        if p <= l {
            return k - 1;
        }
    }
}

// ---------------------------------------------------------------------------
// ARFConfig
// ---------------------------------------------------------------------------

/// Configuration for [`AdaptiveRandomForest`].
#[derive(Debug, Clone)]
pub struct ARFConfig {
    /// Number of trees in the forest.
    pub n_trees: usize,
    /// Poisson lambda for bootstrap resampling (default 6.0).
    pub lambda: f64,
    /// Fraction of features per tree. 0.0 = auto (sqrt(d)/d).
    pub feature_fraction: f64,
    /// ADWIN delta for drift detection (default 1e-3).
    pub drift_delta: f64,
    /// ADWIN delta for warning detection (default 1e-2).
    pub warning_delta: f64,
    /// Random seed.
    pub seed: u64,
}

/// Builder for [`ARFConfig`].
#[derive(Debug, Clone)]
pub struct ARFConfigBuilder {
    n_trees: usize,
    lambda: f64,
    feature_fraction: f64,
    drift_delta: f64,
    warning_delta: f64,
    seed: u64,
}

impl ARFConfig {
    /// Start building a configuration with the given number of trees.
    pub fn builder(n_trees: usize) -> ARFConfigBuilder {
        ARFConfigBuilder {
            n_trees,
            lambda: 6.0,
            feature_fraction: 0.0,
            drift_delta: 1e-3,
            warning_delta: 1e-2,
            seed: 42,
        }
    }
}

impl ARFConfigBuilder {
    /// Poisson lambda for bootstrap resampling.
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Fraction of features per tree. 0.0 = auto (sqrt).
    pub fn feature_fraction(mut self, f: f64) -> Self {
        self.feature_fraction = f;
        self
    }

    /// ADWIN delta for drift detection.
    pub fn drift_delta(mut self, d: f64) -> Self {
        self.drift_delta = d;
        self
    }

    /// ADWIN delta for warning detection.
    pub fn warning_delta(mut self, d: f64) -> Self {
        self.warning_delta = d;
        self
    }

    /// Random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Build the configuration, validating all parameters.
    pub fn build(self) -> Result<ARFConfig, String> {
        if self.n_trees == 0 {
            return Err("n_trees must be >= 1".into());
        }
        if self.lambda <= 0.0 || !self.lambda.is_finite() {
            return Err("lambda must be positive and finite".into());
        }
        if self.feature_fraction < 0.0 || self.feature_fraction > 1.0 {
            return Err("feature_fraction must be in [0.0, 1.0]".into());
        }
        if self.drift_delta <= 0.0 || self.drift_delta >= 1.0 {
            return Err("drift_delta must be in (0, 1)".into());
        }
        if self.warning_delta <= 0.0 || self.warning_delta >= 1.0 {
            return Err("warning_delta must be in (0, 1)".into());
        }
        Ok(ARFConfig {
            n_trees: self.n_trees,
            lambda: self.lambda,
            feature_fraction: self.feature_fraction,
            drift_delta: self.drift_delta,
            warning_delta: self.warning_delta,
            seed: self.seed,
        })
    }
}

// ---------------------------------------------------------------------------
// ARFMember
// ---------------------------------------------------------------------------

struct ARFMember {
    learner: Box<dyn StreamingLearner>,
    drift_detector: Adwin,
    warning_detector: Adwin,
    feature_mask: Vec<usize>,
    n_correct: u64,
    n_evaluated: u64,
}

// ---------------------------------------------------------------------------
// AdaptiveRandomForest
// ---------------------------------------------------------------------------

/// Adaptive Random Forest for streaming classification.
///
/// An ensemble of `n_trees` streaming learners, each trained on a
/// Poisson-weighted bootstrap with a random feature subspace. ADWIN drift
/// detection automatically resets individual trees when their error rate
/// changes significantly.
///
/// # Example
///
/// ```
/// use irithyll::{AdaptiveRandomForest, StreamingLearner};
/// use irithyll::ensemble::adaptive_forest::ARFConfig;
/// use irithyll::learners::linear::StreamingLinearModel;
///
/// let config = ARFConfig::builder(5).lambda(6.0).build().unwrap();
/// let mut arf = AdaptiveRandomForest::new(config, || {
///     Box::new(StreamingLinearModel::new(0.01))
/// });
///
/// arf.train_one(&[1.0, 0.0], 0.0);
/// let pred = arf.predict(&[1.0, 0.0]);
/// ```
pub struct AdaptiveRandomForest {
    config: ARFConfig,
    trees: Vec<ARFMember>,
    n_features: usize,
    n_samples: u64,
    n_drifts: usize,
    rng_state: u64,
    /// Stored factory for creating replacement learners on drift.
    factory: Box<dyn Fn() -> Box<dyn StreamingLearner> + Send + Sync>,
}

impl AdaptiveRandomForest {
    /// Create a new ARF with the given config and learner factory.
    ///
    /// The `factory` closure is called `n_trees` times to create the initial
    /// ensemble, and again whenever a tree is replaced after drift detection.
    pub fn new<F>(config: ARFConfig, factory: F) -> Self
    where
        F: Fn() -> Box<dyn StreamingLearner> + Send + Sync + 'static,
    {
        let mut rng = config.seed;
        let trees: Vec<ARFMember> = (0..config.n_trees)
            .map(|_| {
                // Feature masks are initialized lazily on first train_one
                let _ = xorshift64(&mut rng);
                ARFMember {
                    learner: factory(),
                    drift_detector: Adwin::with_delta(config.drift_delta),
                    warning_detector: Adwin::with_delta(config.warning_delta),
                    feature_mask: Vec::new(),
                    n_correct: 0,
                    n_evaluated: 0,
                }
            })
            .collect();

        Self {
            config,
            trees,
            n_features: 0,
            n_samples: 0,
            n_drifts: 0,
            rng_state: rng,
            factory: Box::new(factory),
        }
    }

    /// Initialize feature masks for all trees (called lazily on first sample).
    fn init_feature_masks(&mut self) {
        let d = self.n_features;
        let fraction = if self.config.feature_fraction == 0.0 {
            crate::math::sqrt(d as f64) / d as f64
        } else {
            self.config.feature_fraction
        };
        let k = (crate::math::ceil(fraction * d as f64) as usize)
            .max(1)
            .min(d);

        for member in &mut self.trees {
            // Fisher-Yates partial shuffle to select k unique indices
            let mut indices: Vec<usize> = (0..d).collect();
            for i in 0..k {
                let j = i + (xorshift64(&mut self.rng_state) as usize % (d - i));
                indices.swap(i, j);
            }
            indices.truncate(k);
            indices.sort_unstable();
            member.feature_mask = indices;
        }
    }

    /// Extract masked features for a given tree member.
    fn mask_features(&self, features: &[f64], mask: &[usize]) -> Vec<f64> {
        if mask.is_empty() {
            features.to_vec()
        } else {
            mask.iter().map(|&i| features[i]).collect()
        }
    }

    /// Train on a single sample.
    pub fn train_one(&mut self, features: &[f64], target: f64) {
        if self.n_features == 0 {
            self.n_features = features.len();
            self.init_feature_masks();
        }
        self.n_samples += 1;

        for i in 0..self.trees.len() {
            let k = poisson(self.config.lambda, &mut self.rng_state);
            let masked = self.mask_features(features, &self.trees[i].feature_mask);

            // Predict before training (for drift detection).
            let pred = self.trees[i].learner.predict(&masked);
            let correct = crate::math::abs(crate::math::round(pred) - target) < 0.5;
            self.trees[i].n_evaluated += 1;
            if correct {
                self.trees[i].n_correct += 1;
            }

            // Train k times (Poisson-weighted bootstrap).
            for _ in 0..k {
                self.trees[i].learner.train(&masked, target);
            }

            // Feed error signal to drift detectors.
            let error_val = if correct { 0.0 } else { 1.0 };
            let drift_signal = self.trees[i].drift_detector.update(error_val);
            let _warning_signal = self.trees[i].warning_detector.update(error_val);

            // On drift: replace the tree.
            if matches!(drift_signal, DriftSignal::Drift) {
                self.trees[i].learner = (self.factory)();
                self.trees[i].drift_detector = Adwin::with_delta(self.config.drift_delta);
                self.trees[i].warning_detector = Adwin::with_delta(self.config.warning_delta);
                self.trees[i].n_correct = 0;
                self.trees[i].n_evaluated = 0;
                self.n_drifts += 1;

                // Re-init feature mask for the new tree.
                let d = self.n_features;
                let fraction = if self.config.feature_fraction == 0.0 {
                    crate::math::sqrt(d as f64) / d as f64
                } else {
                    self.config.feature_fraction
                };
                let k_features = (crate::math::ceil(fraction * d as f64) as usize)
                    .max(1)
                    .min(d);
                let mut indices: Vec<usize> = (0..d).collect();
                for j in 0..k_features {
                    let swap = j + (xorshift64(&mut self.rng_state) as usize % (d - j));
                    indices.swap(j, swap);
                }
                indices.truncate(k_features);
                indices.sort_unstable();
                self.trees[i].feature_mask = indices;
            }
        }
    }

    /// Predict by majority vote across all trees.
    ///
    /// Each tree casts a vote for a class (prediction rounded to nearest
    /// integer). The class with the most votes wins.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let votes = self.predict_votes(features);
        votes
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(class, _)| class)
            .unwrap_or(0.0)
    }

    /// Vote counts per predicted class.
    pub fn predict_votes(&self, features: &[f64]) -> Vec<(f64, u64)> {
        let mut vote_map: Vec<(f64, u64)> = Vec::new();
        for member in &self.trees {
            let masked = self.mask_features(features, &member.feature_mask);
            let pred = crate::math::round(member.learner.predict(&masked));
            if let Some(entry) = vote_map
                .iter_mut()
                .find(|(c, _)| crate::math::abs(*c - pred) < 0.5)
            {
                entry.1 += 1;
            } else {
                vote_map.push((pred, 1));
            }
        }
        vote_map
    }

    /// Number of trees in the ensemble.
    pub fn n_trees(&self) -> usize {
        self.config.n_trees
    }

    /// Total samples processed.
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Per-tree accuracy (correct / evaluated).
    pub fn tree_accuracies(&self) -> Vec<f64> {
        self.trees
            .iter()
            .map(|m| {
                if m.n_evaluated == 0 {
                    0.0
                } else {
                    m.n_correct as f64 / m.n_evaluated as f64
                }
            })
            .collect()
    }

    /// Total number of drift-triggered tree replacements.
    pub fn n_drifts_detected(&self) -> usize {
        self.n_drifts
    }
}

impl StreamingLearner for AdaptiveRandomForest {
    fn train_one(&mut self, features: &[f64], target: f64, _weight: f64) {
        self.train_one(features, target);
    }

    fn predict(&self, features: &[f64]) -> f64 {
        self.predict(features)
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.n_samples = 0;
        self.n_drifts = 0;
        for member in &mut self.trees {
            member.n_correct = 0;
            member.n_evaluated = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;

    struct MockClassifier {
        prediction: f64,
        n: u64,
    }

    impl MockClassifier {
        fn new(prediction: f64) -> Self {
            Self { prediction, n: 0 }
        }
    }

    impl StreamingLearner for MockClassifier {
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
    fn arf_trains_and_predicts() {
        let config = ARFConfig::builder(3).seed(42).build().unwrap();
        let mut arf = AdaptiveRandomForest::new(config, || Box::new(MockClassifier::new(1.0)));

        arf.train_one(&[1.0, 2.0], 1.0);
        let pred = arf.predict(&[1.0, 2.0]);
        assert_eq!(pred, 1.0);
        assert_eq!(arf.n_samples_seen(), 1);
    }

    #[test]
    fn arf_majority_vote() {
        let config = ARFConfig::builder(5).seed(42).build().unwrap();
        // All trees predict 0.0 → unanimous vote
        let mut arf = AdaptiveRandomForest::new(config, || Box::new(MockClassifier::new(0.0)));
        // Need to init feature masks
        arf.n_features = 2;
        arf.init_feature_masks();

        let votes = arf.predict_votes(&[1.0, 2.0]);
        assert_eq!(votes.len(), 1, "all trees should agree");
        assert_eq!(votes[0], (0.0, 5), "5 votes for class 0");
        assert_eq!(arf.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn arf_poisson_valid() {
        let mut rng = 12345u64;
        let mut total = 0u64;
        let n = 1000;
        for _ in 0..n {
            total += poisson(6.0, &mut rng);
        }
        let mean = total as f64 / n as f64;
        // Poisson(6) should have mean ~6
        assert!(
            (mean - 6.0).abs() < 1.0,
            "Poisson mean should be ~6.0, got {}",
            mean
        );
    }

    #[test]
    fn arf_feature_subspace() {
        let config = ARFConfig::builder(3)
            .feature_fraction(0.5)
            .seed(42)
            .build()
            .unwrap();
        let mut arf = AdaptiveRandomForest::new(config, || Box::new(MockClassifier::new(0.0)));

        arf.train_one(&[1.0, 2.0, 3.0, 4.0], 0.0);

        // Each tree should have ceil(0.5 * 4) = 2 features
        for member in &arf.trees {
            assert_eq!(
                member.feature_mask.len(),
                2,
                "expected 2 features, got {}",
                member.feature_mask.len()
            );
        }
    }

    #[test]
    fn arf_streaming_learner_trait() {
        let config = ARFConfig::builder(3).seed(42).build().unwrap();
        let mut arf = AdaptiveRandomForest::new(config, || Box::new(MockClassifier::new(0.0)));

        let learner: &mut dyn StreamingLearner = &mut arf;
        learner.train(&[1.0, 2.0], 0.0);
        assert_eq!(learner.n_samples_seen(), 1);
        let pred = learner.predict(&[1.0, 2.0]);
        assert_eq!(pred, 0.0);
    }

    #[test]
    fn arf_config_validates() {
        assert!(ARFConfig::builder(0).build().is_err());
        assert!(ARFConfig::builder(3).lambda(0.0).build().is_err());
        assert!(ARFConfig::builder(3).lambda(-1.0).build().is_err());
        assert!(ARFConfig::builder(3)
            .feature_fraction(-0.1)
            .build()
            .is_err());
        assert!(ARFConfig::builder(3).feature_fraction(1.1).build().is_err());
        assert!(ARFConfig::builder(3).drift_delta(0.0).build().is_err());
        assert!(ARFConfig::builder(3).drift_delta(1.0).build().is_err());
        assert!(ARFConfig::builder(3).build().is_ok());
    }

    #[test]
    fn arf_tree_accuracies() {
        let config = ARFConfig::builder(3).seed(42).build().unwrap();
        let mut arf = AdaptiveRandomForest::new(config, || Box::new(MockClassifier::new(1.0)));

        // Train with target=1.0, mock predicts 1.0 → all correct
        for _ in 0..10 {
            arf.train_one(&[1.0, 2.0], 1.0);
        }

        let accs = arf.tree_accuracies();
        assert_eq!(accs.len(), 3);
        for &acc in &accs {
            assert!(
                acc > 0.9,
                "accuracy should be ~1.0 for correct mock, got {}",
                acc
            );
        }
    }
}
