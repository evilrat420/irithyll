//! Gaussian Naive Bayes classifier with incremental online learning.
//!
//! This module provides [`GaussianNB`], a streaming Gaussian Naive Bayes
//! classifier that maintains per-class sufficient statistics (count, mean,
//! variance) using Welford's online algorithm. It supports weighted samples,
//! arbitrary numbers of classes (discovered lazily), and integrates with the
//! [`StreamingLearner`] trait for polymorphic stacking.
//!
//! # Algorithm
//!
//! For each class *k* and feature *j*, the model tracks:
//! - `count_k` -- weighted sample count
//! - `mean_kj` -- running mean via Welford's method
//! - `m2_kj` -- running second moment (variance = m2 / count)
//!
//! Prediction selects the class with the highest log-posterior:
//!
//! ```text
//! argmax_k  log P(C=k) + sum_j log N(x_j; mu_kj, sigma2_kj)
//! ```
//!
//! where `N` is the Gaussian PDF and `P(C=k) = count_k / total_count`.
//!
//! # Variance Smoothing
//!
//! Following the sklearn convention, a smoothing term `var_smoothing * max_var`
//! is added to all feature variances to prevent division by zero and improve
//! numerical stability. The default smoothing value is `1e-9`.

use std::f64::consts::PI;
use std::fmt;

use crate::learner::StreamingLearner;

// ---------------------------------------------------------------------------
// ClassStats -- per-class sufficient statistics
// ---------------------------------------------------------------------------

/// Sufficient statistics for a single class, updated via Welford's algorithm.
struct ClassStats {
    /// Weighted sample count for this class.
    count: f64,
    /// Running mean per feature (Welford).
    mean: Vec<f64>,
    /// Running M2 per feature (Welford). Variance = m2 / count.
    m2: Vec<f64>,
}

impl ClassStats {
    /// Create a new `ClassStats` for the given number of features.
    fn new(n_features: usize) -> Self {
        Self {
            count: 0.0,
            mean: vec![0.0; n_features],
            m2: vec![0.0; n_features],
        }
    }

    /// Update statistics with a single weighted observation using Welford's method.
    #[inline]
    fn update(&mut self, features: &[f64], weight: f64) {
        self.count += weight;
        for (j, &fj) in features.iter().enumerate() {
            let delta = fj - self.mean[j];
            self.mean[j] += delta * weight / self.count;
            let delta2 = fj - self.mean[j];
            self.m2[j] += delta * delta2 * weight;
        }
    }

    /// Compute the variance for feature `j`. Returns `m2[j] / count`.
    #[inline]
    fn variance(&self, j: usize) -> f64 {
        if self.count > 0.0 {
            self.m2[j] / self.count
        } else {
            0.0
        }
    }
}

impl Clone for ClassStats {
    fn clone(&self) -> Self {
        Self {
            count: self.count,
            mean: self.mean.clone(),
            m2: self.m2.clone(),
        }
    }
}

impl fmt::Debug for ClassStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClassStats")
            .field("count", &self.count)
            .field("n_features", &self.mean.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// GaussianNB
// ---------------------------------------------------------------------------

/// Incremental Gaussian Naive Bayes classifier.
///
/// Learns per-class Gaussian distributions over each feature using Welford's
/// online algorithm. Supports weighted samples, lazy class discovery, and
/// the [`StreamingLearner`] trait for polymorphic stacking.
///
/// # Examples
///
/// ```ignore
/// use irithyll::learners::naive_bayes::GaussianNB;
/// use irithyll::learner::StreamingLearner;
///
/// let mut nb = GaussianNB::new();
///
/// // Train on two-class data
/// nb.train(&[1.0, 2.0], 0.0);
/// nb.train(&[5.0, 6.0], 1.0);
///
/// // Predict returns the class label as f64
/// let pred = nb.predict(&[1.5, 2.5]);
/// assert_eq!(pred, 0.0);
/// ```
pub struct GaussianNB {
    /// Per-class statistics, indexed by class label (as usize).
    class_stats: Vec<ClassStats>,
    /// Total samples seen (unweighted count for the trait).
    samples_seen: u64,
    /// Number of features (lazy-initialized on first train call).
    n_features: Option<usize>,
    /// Minimum variance smoothing factor (sklearn convention).
    var_smoothing: f64,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl GaussianNB {
    /// Create a new `GaussianNB` with default variance smoothing (`1e-9`).
    #[inline]
    pub fn new() -> Self {
        Self {
            class_stats: Vec::new(),
            samples_seen: 0,
            n_features: None,
            var_smoothing: 1e-9,
        }
    }

    /// Create a new `GaussianNB` with a custom variance smoothing factor.
    ///
    /// The smoothing term added to all variances is `var_smoothing * max_var`,
    /// following the sklearn convention.
    #[inline]
    pub fn with_var_smoothing(var_smoothing: f64) -> Self {
        Self {
            class_stats: Vec::new(),
            samples_seen: 0,
            n_features: None,
            var_smoothing,
        }
    }
}

// ---------------------------------------------------------------------------
// Public query methods
// ---------------------------------------------------------------------------

impl GaussianNB {
    /// Number of classes observed so far.
    #[inline]
    pub fn n_classes(&self) -> usize {
        self.class_stats.len()
    }

    /// Prior probability for each class: `count_k / total_count`.
    ///
    /// Returns an empty vector if no samples have been seen.
    pub fn class_prior(&self) -> Vec<f64> {
        let total: f64 = self.class_stats.iter().map(|s| s.count).sum();
        if total == 0.0 {
            return vec![0.0; self.class_stats.len()];
        }
        self.class_stats.iter().map(|s| s.count / total).collect()
    }

    /// Compute log-probabilities for each class (unnormalized log-posteriors).
    ///
    /// Returns a vector of length `n_classes()`. Each entry is the log of the
    /// unnormalized posterior: `log P(C=k) + sum_j log N(x_j; mu_kj, var_kj)`.
    ///
    /// Returns an empty vector if no training data has been seen.
    pub fn predict_log_proba(&self, features: &[f64]) -> Vec<f64> {
        if self.class_stats.is_empty() {
            return Vec::new();
        }

        let total_count: f64 = self.class_stats.iter().map(|s| s.count).sum();
        if total_count == 0.0 {
            return vec![0.0; self.class_stats.len()];
        }

        let smoothing = self.compute_smoothing();

        self.class_stats
            .iter()
            .map(|stats| {
                if stats.count == 0.0 {
                    return f64::NEG_INFINITY;
                }
                let log_prior = (stats.count / total_count).ln();
                let log_likelihood: f64 = (0..features.len())
                    .map(|j| {
                        let var = stats.variance(j) + smoothing;
                        gaussian_log_likelihood(features[j], stats.mean[j], var)
                    })
                    .sum();
                log_prior + log_likelihood
            })
            .collect()
    }

    /// Compute normalized class probabilities via log-sum-exp.
    ///
    /// Returns a vector of length `n_classes()` that sums to 1.0. Each entry
    /// is the posterior probability `P(C=k | x)`.
    ///
    /// Returns an empty vector if no training data has been seen.
    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        let log_probs = self.predict_log_proba(features);
        if log_probs.is_empty() {
            return Vec::new();
        }
        log_sum_exp_normalize(&log_probs)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

impl GaussianNB {
    /// Compute the smoothing term: `var_smoothing * max_variance` across all
    /// classes and features.
    fn compute_smoothing(&self) -> f64 {
        let mut max_var: f64 = 0.0;
        for stats in &self.class_stats {
            if stats.count == 0.0 {
                continue;
            }
            for j in 0..stats.mean.len() {
                let v = stats.variance(j);
                if v > max_var {
                    max_var = v;
                }
            }
        }
        self.var_smoothing * max_var.max(1e-12)
    }

    /// Ensure `class_stats` is large enough to hold class `label`.
    fn ensure_class(&mut self, label: usize, n_features: usize) {
        while self.class_stats.len() <= label {
            self.class_stats.push(ClassStats::new(n_features));
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

impl StreamingLearner for GaussianNB {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let n_feat = features.len();

        // Lazy-init feature count.
        if self.n_features.is_none() {
            self.n_features = Some(n_feat);
        }

        let label = target.round() as usize;
        self.ensure_class(label, n_feat);
        self.class_stats[label].update(features, weight);
        self.samples_seen += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        if self.class_stats.is_empty() {
            return 0.0;
        }

        let log_probs = self.predict_log_proba(features);
        let mut best_class = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (k, &lp) in log_probs.iter().enumerate() {
            if lp > best_score {
                best_score = lp;
                best_class = k;
            }
        }
        best_class as f64
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        self.class_stats.clear();
        self.samples_seen = 0;
        self.n_features = None;
    }
}

// ---------------------------------------------------------------------------
// Clone impl
// ---------------------------------------------------------------------------

impl Clone for GaussianNB {
    fn clone(&self) -> Self {
        Self {
            class_stats: self.class_stats.clone(),
            samples_seen: self.samples_seen,
            n_features: self.n_features,
            var_smoothing: self.var_smoothing,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

impl fmt::Debug for GaussianNB {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GaussianNB")
            .field("n_classes", &self.class_stats.len())
            .field("samples_seen", &self.samples_seen)
            .field("n_features", &self.n_features)
            .field("var_smoothing", &self.var_smoothing)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Default impl
// ---------------------------------------------------------------------------

impl Default for GaussianNB {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Free functions -- math utilities
// ---------------------------------------------------------------------------

/// Gaussian log-likelihood: `-0.5 * (log(2 * pi * var) + (x - mean)^2 / var)`.
#[inline]
fn gaussian_log_likelihood(x: f64, mean: f64, var: f64) -> f64 {
    -0.5 * ((2.0 * PI * var).ln() + (x - mean).powi(2) / var)
}

/// Convert log-probabilities to normalized probabilities via log-sum-exp.
///
/// Subtracts the maximum log-prob for numerical stability, exponentiates,
/// then normalizes so the result sums to 1.0.
fn log_sum_exp_normalize(log_probs: &[f64]) -> Vec<f64> {
    let max_lp = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if max_lp == f64::NEG_INFINITY {
        // All classes have -inf log-prob; return uniform.
        let n = log_probs.len();
        return vec![1.0 / n as f64; n];
    }

    let exps: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for GaussianNB {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let nb = GaussianNB::new();
        assert_eq!(nb.n_samples_seen(), 0);
        assert_eq!(nb.n_classes(), 0);
        assert_eq!(nb.n_features, None);
        assert!((nb.var_smoothing - 1e-9).abs() < 1e-15);

        let nb2 = GaussianNB::with_var_smoothing(1e-6);
        assert!((nb2.var_smoothing - 1e-6).abs() < 1e-15);

        let nb3 = GaussianNB::default();
        assert_eq!(nb3.n_samples_seen(), 0);
    }

    #[test]
    fn test_two_class() {
        let mut nb = GaussianNB::new();

        // Class 0: features clustered around (1, 1)
        for _ in 0..50 {
            nb.train(&[1.0, 1.0], 0.0);
        }
        // Class 1: features clustered around (10, 10)
        for _ in 0..50 {
            nb.train(&[10.0, 10.0], 1.0);
        }

        assert_eq!(nb.n_samples_seen(), 100);
        assert_eq!(nb.n_classes(), 2);

        // Point near class 0 should predict 0.0
        assert_eq!(nb.predict(&[1.5, 1.5]), 0.0);

        // Point near class 1 should predict 1.0
        assert_eq!(nb.predict(&[9.5, 9.5]), 1.0);
    }

    #[test]
    fn test_three_class() {
        let mut nb = GaussianNB::new();

        // Class 0: around (0, 0)
        for _ in 0..40 {
            nb.train(&[0.0, 0.0], 0.0);
            nb.train(&[0.1, -0.1], 0.0);
        }

        // Class 1: around (10, 0)
        for _ in 0..40 {
            nb.train(&[10.0, 0.0], 1.0);
            nb.train(&[10.1, -0.1], 1.0);
        }

        // Class 2: around (5, 10)
        for _ in 0..40 {
            nb.train(&[5.0, 10.0], 2.0);
            nb.train(&[5.1, 9.9], 2.0);
        }

        assert_eq!(nb.n_classes(), 3);

        assert_eq!(nb.predict(&[0.0, 0.0]), 0.0);
        assert_eq!(nb.predict(&[10.0, 0.0]), 1.0);
        assert_eq!(nb.predict(&[5.0, 10.0]), 2.0);
    }

    #[test]
    fn test_predict_proba() {
        let mut nb = GaussianNB::new();

        for _ in 0..50 {
            nb.train(&[1.0, 1.0], 0.0);
        }
        for _ in 0..50 {
            nb.train(&[10.0, 10.0], 1.0);
        }

        let proba = nb.predict_proba(&[1.0, 1.0]);
        assert_eq!(proba.len(), 2);

        // Probabilities must sum to 1.0
        let sum: f64 = proba.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "probabilities should sum to 1.0, got {}",
            sum,
        );

        // Class 0 should dominate for a point near (1, 1)
        assert!(
            proba[0] > proba[1],
            "P(class=0) should be greater for point near class 0: {:?}",
            proba,
        );

        // Check the other direction
        let proba2 = nb.predict_proba(&[10.0, 10.0]);
        let sum2: f64 = proba2.iter().sum();
        assert!((sum2 - 1.0).abs() < 1e-10);
        assert!(proba2[1] > proba2[0]);
    }

    #[test]
    fn test_class_prior() {
        let mut nb = GaussianNB::new();

        // 30 samples of class 0, 70 samples of class 1
        for _ in 0..30 {
            nb.train(&[1.0], 0.0);
        }
        for _ in 0..70 {
            nb.train(&[5.0], 1.0);
        }

        let priors = nb.class_prior();
        assert_eq!(priors.len(), 2);
        assert!(
            (priors[0] - 0.3).abs() < 1e-10,
            "prior[0] should be 0.3, got {}",
            priors[0],
        );
        assert!(
            (priors[1] - 0.7).abs() < 1e-10,
            "prior[1] should be 0.7, got {}",
            priors[1],
        );
    }

    #[test]
    fn test_incremental_update() {
        let mut nb = GaussianNB::new();

        // Train with minimal data -- only 1 sample per class, no variance
        nb.train(&[0.0, 0.0], 0.0);
        nb.train(&[10.0, 10.0], 1.0);

        // With only 1 sample per class, prediction at midpoint is ambiguous.
        // Record the class-0 probability at the midpoint.
        let proba_early = nb.predict_proba(&[2.0, 2.0]);
        let p0_early = proba_early[0];

        // Add more class-0 data near origin -- model should become more confident
        for _ in 0..50 {
            nb.train(&[0.5, 0.5], 0.0);
            nb.train(&[-0.5, -0.5], 0.0);
        }

        let proba_later = nb.predict_proba(&[2.0, 2.0]);
        let p0_later = proba_later[0];

        // After seeing many class-0 samples near origin, the model should
        // assign higher probability to class 0 at (2,2) due to higher prior
        // and tighter variance estimate.
        assert!(
            p0_later > p0_early || (p0_early - p0_later).abs() < 0.05,
            "incremental updates should improve or maintain confidence: early={}, later={}",
            p0_early,
            p0_later,
        );
    }

    #[test]
    fn test_reset() {
        let mut nb = GaussianNB::new();

        for _ in 0..50 {
            nb.train(&[1.0, 2.0], 0.0);
        }
        for _ in 0..50 {
            nb.train(&[5.0, 6.0], 1.0);
        }

        assert_eq!(nb.n_samples_seen(), 100);
        assert_eq!(nb.n_classes(), 2);
        assert!(nb.n_features.is_some());

        nb.reset();

        assert_eq!(nb.n_samples_seen(), 0);
        assert_eq!(nb.n_classes(), 0);
        assert_eq!(nb.n_features, None);
        assert!(nb.class_prior().is_empty());
        assert!(nb.predict_proba(&[1.0, 2.0]).is_empty());

        // After reset, predict should return default (0.0)
        assert_eq!(nb.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_predict_batch() {
        let mut nb = GaussianNB::new();

        for _ in 0..50 {
            nb.train(&[1.0, 1.0], 0.0);
        }
        for _ in 0..50 {
            nb.train(&[10.0, 10.0], 1.0);
        }

        let rows: Vec<&[f64]> = vec![&[1.0, 1.0], &[10.0, 10.0], &[5.0, 5.0]];
        let batch = nb.predict_batch(&rows);

        assert_eq!(batch.len(), rows.len());
        for (i, row) in rows.iter().enumerate() {
            let individual = nb.predict(row);
            assert!(
                (batch[i] - individual).abs() < 1e-12,
                "batch[{}]={} != individual={}",
                i,
                batch[i],
                individual,
            );
        }
    }

    #[test]
    fn test_trait_object() {
        let mut nb = GaussianNB::new();
        nb.train(&[1.0], 0.0);
        nb.train(&[10.0], 1.0);

        // Must compile and work as Box<dyn StreamingLearner>.
        let mut boxed: Box<dyn StreamingLearner> = Box::new(nb);

        boxed.train(&[1.0], 0.0);
        assert_eq!(boxed.n_samples_seen(), 3);

        let pred = boxed.predict(&[1.0]);
        assert!(
            pred == 0.0 || pred == 1.0,
            "prediction should be a class label"
        );

        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }

    #[test]
    fn test_weighted_samples() {
        let mut nb_uniform = GaussianNB::new();
        let mut nb_weighted = GaussianNB::new();

        // Uniform: samples with spread for each class.
        for i in 0..50 {
            let x = 1.0 + (i as f64) * 0.1;
            nb_uniform.train(&[x], 0.0);
        }
        for i in 0..50 {
            let x = 10.0 + (i as f64) * 0.1;
            nb_uniform.train(&[x], 1.0);
        }

        // Weighted: same spread, but class 0 gets much higher weight.
        for i in 0..50 {
            let x = 1.0 + (i as f64) * 0.1;
            nb_weighted.train_one(&[x], 0.0, 10.0);
        }
        for i in 0..50 {
            let x = 10.0 + (i as f64) * 0.1;
            nb_weighted.train_one(&[x], 1.0, 1.0);
        }

        // At the midpoint, uniform model should be roughly 50/50.
        let proba_uniform = nb_uniform.predict_proba(&[5.5]);
        // Weighted model should favor class 0 due to higher prior.
        let proba_weighted = nb_weighted.predict_proba(&[5.5]);

        assert!(
            proba_weighted[0] > proba_uniform[0],
            "weighted class 0 should have higher posterior: weighted={}, uniform={}",
            proba_weighted[0],
            proba_uniform[0],
        );
    }

    #[test]
    fn test_clone_independence() {
        let mut nb = GaussianNB::new();
        for _ in 0..30 {
            nb.train(&[1.0, 2.0], 0.0);
        }

        let mut cloned = nb.clone();
        assert_eq!(cloned.n_samples_seen(), nb.n_samples_seen());

        // Training the clone should not affect the original.
        for _ in 0..30 {
            cloned.train(&[10.0, 20.0], 1.0);
        }
        assert_eq!(nb.n_samples_seen(), 30);
        assert_eq!(cloned.n_samples_seen(), 60);
        assert_eq!(nb.n_classes(), 1);
        assert_eq!(cloned.n_classes(), 2);
    }

    #[test]
    fn test_debug_format() {
        let nb = GaussianNB::new();
        let debug_str = format!("{:?}", nb);
        assert!(debug_str.contains("GaussianNB"));
        assert!(debug_str.contains("samples_seen"));
        assert!(debug_str.contains("var_smoothing"));
    }
}
