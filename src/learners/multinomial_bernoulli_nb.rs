//! Multinomial and Bernoulli Naive Bayes classifiers.
//!
//! Two complementary streaming Naive Bayes variants:
//!
//! - [`MultinomialNB`] -- for count/frequency features (e.g., word counts in text
//!   classification). Learns `P(feature | class)` as smoothed relative frequencies.
//! - [`BernoulliNB`] -- for binary/boolean features. Explicitly models both
//!   feature presence and absence, unlike Multinomial which ignores zeros.
//!
//! Both support Laplace smoothing, automatic class discovery, and implement
//! [`StreamingLearner`] for polymorphic composition.

use crate::learner::StreamingLearner;

// ---------------------------------------------------------------------------
// MultinomialNB
// ---------------------------------------------------------------------------

/// Multinomial Naive Bayes classifier for streaming count/frequency data.
///
/// Maintains per-class feature count sums with Laplace smoothing. Prediction
/// selects the class maximizing:
///
/// ```text
/// argmax_c { log P(c) + sum_f { x_f * log P(f|c) } }
/// ```
///
/// where `P(f|c) = (count(f,c) + alpha) / (total(c) + alpha * n_features)`.
///
/// # Example
///
/// ```
/// use irithyll::MultinomialNB;
///
/// let mut nb = MultinomialNB::new();
/// // Class 0: high counts in features 0-1
/// nb.train_one(&[5.0, 3.0, 0.0, 0.0], 0);
/// // Class 1: high counts in features 2-3
/// nb.train_one(&[0.0, 0.0, 4.0, 6.0], 1);
///
/// assert_eq!(nb.predict_class(&[4.0, 2.0, 0.0, 0.0]), 0);
/// ```
#[derive(Debug, Clone)]
pub struct MultinomialNB {
    /// Laplace smoothing parameter.
    alpha: f64,
    /// Per-class sample counts.
    class_counts: Vec<u64>,
    /// Per-class, per-feature sum of feature values. `[class][feature]`
    feature_counts: Vec<Vec<f64>>,
    /// Per-class total of all feature value sums.
    class_total: Vec<f64>,
    /// Number of features (0 = lazy init).
    n_features: usize,
    /// Total samples seen.
    n_samples: u64,
}

impl MultinomialNB {
    /// Create a new Multinomial NB with default Laplace smoothing (alpha = 1.0).
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            class_counts: Vec::new(),
            feature_counts: Vec::new(),
            class_total: Vec::new(),
            n_features: 0,
            n_samples: 0,
        }
    }

    /// Create with a custom smoothing parameter.
    pub fn with_alpha(alpha: f64) -> Self {
        let mut nb = Self::new();
        nb.alpha = alpha;
        nb
    }

    /// Ensure vectors are large enough for the given class label.
    fn ensure_class(&mut self, class: usize) {
        while self.class_counts.len() <= class {
            self.class_counts.push(0);
            self.feature_counts.push(vec![0.0; self.n_features]);
            self.class_total.push(0.0);
        }
    }

    /// Train on a single sample.
    pub fn train_one(&mut self, features: &[f64], class: usize) {
        // Lazy feature init.
        if self.n_features == 0 {
            self.n_features = features.len();
            // Resize existing class vectors (if any).
            for fc in &mut self.feature_counts {
                fc.resize(self.n_features, 0.0);
            }
        }
        self.ensure_class(class);
        self.n_samples += 1;

        self.class_counts[class] += 1;
        let total: f64 = features.iter().sum();
        self.class_total[class] += total;
        for (j, &x) in features.iter().enumerate() {
            self.feature_counts[class][j] += x;
        }
    }

    /// Compute unnormalized log-posterior for each class.
    fn log_posteriors(&self, features: &[f64]) -> Vec<f64> {
        let n_total = self.n_samples as f64;
        let n_feat = self.n_features as f64;
        let alpha = self.alpha;

        self.class_counts
            .iter()
            .enumerate()
            .map(|(c, &count)| {
                let log_prior = (count as f64 / n_total).ln();
                let denom = self.class_total[c] + alpha * n_feat;
                let log_likelihood: f64 = features
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| {
                        let p = (self.feature_counts[c][j] + alpha) / denom;
                        x * p.ln()
                    })
                    .sum();
                log_prior + log_likelihood
            })
            .collect()
    }

    /// Predict the most likely class.
    pub fn predict_class(&self, features: &[f64]) -> usize {
        let logs = self.log_posteriors(features);
        logs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Log-probability for each class (unnormalized).
    pub fn predict_log_proba(&self, features: &[f64]) -> Vec<f64> {
        self.log_posteriors(features)
    }

    /// Normalized probability for each class.
    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        let logs = self.log_posteriors(features);
        let max_log = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logs.iter().map(|&l| (l - max_log).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            vec![1.0 / logs.len() as f64; logs.len()]
        } else {
            exps.iter().map(|&e| e / sum).collect()
        }
    }

    /// Number of discovered classes.
    pub fn n_classes(&self) -> usize {
        self.class_counts.len()
    }

    /// Total samples seen.
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.class_counts.clear();
        self.feature_counts.clear();
        self.class_total.clear();
        self.n_features = 0;
        self.n_samples = 0;
    }
}

impl Default for MultinomialNB {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingLearner for MultinomialNB {
    fn train_one(&mut self, features: &[f64], target: f64, _weight: f64) {
        self.train_one(features, target as usize);
    }

    fn predict(&self, features: &[f64]) -> f64 {
        self.predict_class(features) as f64
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ---------------------------------------------------------------------------
// BernoulliNB
// ---------------------------------------------------------------------------

/// Bernoulli Naive Bayes classifier for streaming binary feature data.
///
/// Each feature is binarized: values above `threshold` are "present" (1),
/// values at or below are "absent" (0). Unlike [`MultinomialNB`], Bernoulli NB
/// explicitly models feature absence:
///
/// ```text
/// argmax_c { log P(c) + sum_f { b_f * log P(f|c) + (1 - b_f) * log(1 - P(f|c)) } }
/// ```
///
/// where `b_f = 1 if x_f > threshold, else 0` and
/// `P(f|c) = (present(f,c) + alpha) / (count(c) + 2 * alpha)`.
///
/// # Example
///
/// ```
/// use irithyll::BernoulliNB;
///
/// let mut nb = BernoulliNB::new();
/// nb.train_one(&[1.0, 1.0, 0.0, 0.0], 0);
/// nb.train_one(&[0.0, 0.0, 1.0, 1.0], 1);
///
/// assert_eq!(nb.predict_class(&[1.0, 1.0, 0.0, 0.0]), 0);
/// ```
#[derive(Debug, Clone)]
pub struct BernoulliNB {
    /// Laplace smoothing parameter.
    alpha: f64,
    /// Binarization threshold.
    threshold: f64,
    /// Per-class sample counts.
    class_counts: Vec<u64>,
    /// Per-class, per-feature presence counts. `[class][feature]`
    feature_present: Vec<Vec<u64>>,
    /// Number of features (0 = lazy init).
    n_features: usize,
    /// Total samples seen.
    n_samples: u64,
}

impl BernoulliNB {
    /// Create with default settings (alpha = 1.0, threshold = 0.0).
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            threshold: 0.0,
            class_counts: Vec::new(),
            feature_present: Vec::new(),
            n_features: 0,
            n_samples: 0,
        }
    }

    /// Create with custom smoothing parameter.
    pub fn with_alpha(alpha: f64) -> Self {
        let mut nb = Self::new();
        nb.alpha = alpha;
        nb
    }

    /// Create with custom binarization threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        let mut nb = Self::new();
        nb.threshold = threshold;
        nb
    }

    /// Create with custom smoothing and threshold.
    pub fn with_alpha_and_threshold(alpha: f64, threshold: f64) -> Self {
        let mut nb = Self::new();
        nb.alpha = alpha;
        nb.threshold = threshold;
        nb
    }

    fn ensure_class(&mut self, class: usize) {
        while self.class_counts.len() <= class {
            self.class_counts.push(0);
            self.feature_present.push(vec![0; self.n_features]);
        }
    }

    /// Train on a single sample.
    pub fn train_one(&mut self, features: &[f64], class: usize) {
        if self.n_features == 0 {
            self.n_features = features.len();
            for fp in &mut self.feature_present {
                fp.resize(self.n_features, 0);
            }
        }
        self.ensure_class(class);
        self.n_samples += 1;
        self.class_counts[class] += 1;

        for (j, &x) in features.iter().enumerate() {
            if x > self.threshold {
                self.feature_present[class][j] += 1;
            }
        }
    }

    fn log_posteriors(&self, features: &[f64]) -> Vec<f64> {
        let n_total = self.n_samples as f64;
        let alpha = self.alpha;

        self.class_counts
            .iter()
            .enumerate()
            .map(|(c, &count)| {
                let log_prior = (count as f64 / n_total).ln();
                let denom = count as f64 + 2.0 * alpha;
                let log_likelihood: f64 = features
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| {
                        let p = (self.feature_present[c][j] as f64 + alpha) / denom;
                        let b = if x > self.threshold { 1.0 } else { 0.0 };
                        b * p.ln() + (1.0 - b) * (1.0 - p).ln()
                    })
                    .sum();
                log_prior + log_likelihood
            })
            .collect()
    }

    /// Predict the most likely class.
    pub fn predict_class(&self, features: &[f64]) -> usize {
        let logs = self.log_posteriors(features);
        logs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Log-probability for each class (unnormalized).
    pub fn predict_log_proba(&self, features: &[f64]) -> Vec<f64> {
        self.log_posteriors(features)
    }

    /// Normalized probability for each class.
    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        let logs = self.log_posteriors(features);
        let max_log = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logs.iter().map(|&l| (l - max_log).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            vec![1.0 / logs.len() as f64; logs.len()]
        } else {
            exps.iter().map(|&e| e / sum).collect()
        }
    }

    /// Number of discovered classes.
    pub fn n_classes(&self) -> usize {
        self.class_counts.len()
    }

    /// Total samples seen.
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.class_counts.clear();
        self.feature_present.clear();
        self.n_features = 0;
        self.n_samples = 0;
    }
}

impl Default for BernoulliNB {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingLearner for BernoulliNB {
    fn train_one(&mut self, features: &[f64], target: f64, _weight: f64) {
        self.train_one(features, target as usize);
    }

    fn predict(&self, features: &[f64]) -> f64 {
        self.predict_class(features) as f64
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -- MultinomialNB tests --

    #[test]
    fn multinomial_predict_simple() {
        let mut nb = MultinomialNB::new();
        // Class 0: heavy on features 0-1
        for _ in 0..20 {
            nb.train_one(&[5.0, 3.0, 0.0, 0.0], 0);
        }
        // Class 1: heavy on features 2-3
        for _ in 0..20 {
            nb.train_one(&[0.0, 0.0, 4.0, 6.0], 1);
        }
        assert_eq!(nb.predict_class(&[4.0, 2.0, 0.0, 0.0]), 0);
        assert_eq!(nb.predict_class(&[0.0, 0.0, 3.0, 5.0]), 1);
    }

    #[test]
    fn multinomial_predict_proba_sums_to_one() {
        let mut nb = MultinomialNB::new();
        nb.train_one(&[1.0, 0.0], 0);
        nb.train_one(&[0.0, 1.0], 1);
        let proba = nb.predict_proba(&[0.5, 0.5]);
        let sum: f64 = proba.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10), "sum = {}", sum);
    }

    #[test]
    fn multinomial_auto_discovers_classes() {
        let mut nb = MultinomialNB::new();
        nb.train_one(&[1.0], 0);
        nb.train_one(&[1.0], 2); // skip class 1
        assert_eq!(nb.n_classes(), 3); // 0, 1 (empty), 2
    }

    #[test]
    fn multinomial_smoothing_prevents_zero_prob() {
        let mut nb = MultinomialNB::new();
        nb.train_one(&[1.0, 0.0], 0);
        nb.train_one(&[0.0, 1.0], 1);
        // Even though class 0 never saw feature 1, smoothing prevents -inf
        let log_proba = nb.predict_log_proba(&[0.0, 10.0]);
        assert!(log_proba[0].is_finite(), "log_proba[0] = {}", log_proba[0]);
    }

    #[test]
    fn multinomial_streaming_learner_trait() {
        let mut nb = MultinomialNB::new();
        let learner: &mut dyn StreamingLearner = &mut nb;
        learner.train(&[5.0, 0.0], 0.0);
        learner.train(&[0.0, 5.0], 1.0);
        let pred = learner.predict(&[4.0, 0.0]);
        assert_eq!(pred, 0.0);
    }

    #[test]
    fn multinomial_reset() {
        let mut nb = MultinomialNB::new();
        nb.train_one(&[1.0, 0.0], 0);
        nb.reset();
        assert_eq!(nb.n_samples_seen(), 0);
        assert_eq!(nb.n_classes(), 0);
    }

    // -- BernoulliNB tests --

    #[test]
    fn bernoulli_predict_simple() {
        let mut nb = BernoulliNB::new();
        for _ in 0..20 {
            nb.train_one(&[1.0, 1.0, 0.0, 0.0], 0);
        }
        for _ in 0..20 {
            nb.train_one(&[0.0, 0.0, 1.0, 1.0], 1);
        }
        assert_eq!(nb.predict_class(&[1.0, 1.0, 0.0, 0.0]), 0);
        assert_eq!(nb.predict_class(&[0.0, 0.0, 1.0, 1.0]), 1);
    }

    #[test]
    fn bernoulli_predict_proba_sums_to_one() {
        let mut nb = BernoulliNB::new();
        nb.train_one(&[1.0, 0.0], 0);
        nb.train_one(&[0.0, 1.0], 1);
        let proba = nb.predict_proba(&[1.0, 0.0]);
        let sum: f64 = proba.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10), "sum = {}", sum);
    }

    #[test]
    fn bernoulli_models_absence() {
        let mut nb = BernoulliNB::new();
        // Class 0: feature 0 always present, feature 1 always absent
        for _ in 0..20 {
            nb.train_one(&[1.0, 0.0], 0);
        }
        // Class 1: feature 0 always absent, feature 1 always present
        for _ in 0..20 {
            nb.train_one(&[0.0, 1.0], 1);
        }
        // Absence of feature 1 should strongly indicate class 0
        assert_eq!(nb.predict_class(&[0.5, 0.0]), 0);
        // Absence of feature 0 should strongly indicate class 1
        assert_eq!(nb.predict_class(&[0.0, 0.5]), 1);
    }

    #[test]
    fn bernoulli_threshold_binarization() {
        let mut nb = BernoulliNB::with_threshold(0.5);
        for _ in 0..20 {
            nb.train_one(&[1.0, 0.0], 0);
        }
        for _ in 0..20 {
            nb.train_one(&[0.0, 1.0], 1);
        }
        // 0.3 is below threshold 0.5 → absent; 0.8 is above → present
        assert_eq!(nb.predict_class(&[0.8, 0.3]), 0);
        assert_eq!(nb.predict_class(&[0.3, 0.8]), 1);
    }

    #[test]
    fn bernoulli_streaming_learner_trait() {
        let mut nb = BernoulliNB::new();
        let learner: &mut dyn StreamingLearner = &mut nb;
        learner.train(&[1.0, 0.0], 0.0);
        learner.train(&[0.0, 1.0], 1.0);
        assert_eq!(learner.n_samples_seen(), 2);
    }

    #[test]
    fn bernoulli_reset() {
        let mut nb = BernoulliNB::new();
        nb.train_one(&[1.0, 0.0], 0);
        nb.reset();
        assert_eq!(nb.n_samples_seen(), 0);
        assert_eq!(nb.n_classes(), 0);
    }
}
