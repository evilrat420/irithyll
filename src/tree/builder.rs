//! Incremental tree construction with histogram-based splitting.
//!
//! [`TreeConfig`] defines all hyperparameters for a single streaming decision tree:
//! depth limits, regularization, binning granularity, and the Hoeffding bound
//! confidence parameter that controls when splits are committed.

use crate::ensemble::config::FeatureType;

/// Configuration for a single streaming decision tree.
///
/// These parameters control tree growth, regularization, and the statistical
/// confidence required before committing a split decision.
///
/// # Defaults
///
/// | Parameter                | Default |
/// |--------------------------|---------|
/// | `max_depth`              | 6       |
/// | `n_bins`                 | 64      |
/// | `lambda`                 | 1.0     |
/// | `gamma`                  | 0.0     |
/// | `grace_period`           | 200     |
/// | `delta`                  | 1e-7    |
/// | `feature_subsample_rate` | 1.0     |
#[derive(Debug, Clone)]
pub struct TreeConfig {
    /// Maximum tree depth. Deeper trees capture more interactions but risk
    /// overfitting on streaming data. Default: 6.
    pub max_depth: usize,

    /// Number of histogram bins per feature. More bins give finer split
    /// resolution at the cost of memory and slower convergence. Default: 64.
    pub n_bins: usize,

    /// L2 regularization parameter (lambda). Penalizes large leaf weights,
    /// helping prevent overfitting. Appears in the denominator of the
    /// leaf weight formula: w = -G / (H + lambda). Default: 1.0.
    pub lambda: f64,

    /// Minimum split gain threshold (gamma). A candidate split must achieve
    /// gain > gamma to be accepted. Higher values produce more conservative
    /// trees. Default: 0.0.
    pub gamma: f64,

    /// Minimum number of samples a leaf must accumulate before evaluating
    /// potential splits. Also controls when bin edges are computed from
    /// observed feature values. Default: 200.
    pub grace_period: usize,

    /// Hoeffding bound confidence parameter (delta). Smaller values require
    /// more statistical evidence before committing a split, producing more
    /// conservative but more reliable trees. The bound guarantees that the
    /// chosen split is within epsilon of optimal with probability 1-delta.
    /// Default: 1e-7.
    pub delta: f64,

    /// Fraction of features to consider at each split evaluation. 1.0 means
    /// all features are evaluated; smaller values introduce randomness
    /// (similar to random forest feature bagging). Default: 1.0.
    pub feature_subsample_rate: f64,

    /// Random seed for feature subsampling. Default: 42.
    ///
    /// Set by the ensemble orchestrator to ensure deterministic, diverse
    /// behavior across trees (typically `config.seed ^ step_index`).
    pub seed: u64,

    /// Per-sample decay factor for leaf statistics.
    ///
    /// Computed from `leaf_half_life` as `exp(-ln(2) / half_life)`.
    /// When `Some(alpha)`, leaf gradient/hessian sums and histogram bins
    /// are decayed by `alpha` before each new accumulation.
    /// `None` (default) means no decay.
    pub leaf_decay_alpha: Option<f64>,

    /// Re-evaluation interval for max-depth leaves.
    ///
    /// When `Some(n)`, leaves at max depth will re-evaluate potential splits
    /// every `n` samples, allowing the tree to adapt its structure over time.
    /// `None` (default) disables re-evaluation.
    pub split_reeval_interval: Option<usize>,

    /// Per-feature type declarations (continuous vs categorical).
    ///
    /// When `Some`, categorical features use one-bin-per-category binning and
    /// Fisher optimal binary partitioning. `None` (default) treats all features
    /// as continuous.
    pub feature_types: Option<Vec<FeatureType>>,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 6,
            n_bins: 64,
            lambda: 1.0,
            gamma: 0.0,
            grace_period: 200,
            delta: 1e-7,
            feature_subsample_rate: 1.0,
            seed: 42,
            leaf_decay_alpha: None,
            split_reeval_interval: None,
            feature_types: None,
        }
    }
}

impl TreeConfig {
    /// Create a new `TreeConfig` with default parameters.
    ///
    /// Equivalent to `TreeConfig::default()`, but provided as a named
    /// constructor for clarity in builder chains.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum tree depth.
    #[inline]
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the number of histogram bins per feature.
    #[inline]
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Set the L2 regularization parameter (lambda).
    #[inline]
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set the minimum split gain threshold (gamma).
    #[inline]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the grace period (minimum samples before evaluating splits).
    #[inline]
    pub fn grace_period(mut self, grace_period: usize) -> Self {
        self.grace_period = grace_period;
        self
    }

    /// Set the Hoeffding bound confidence parameter (delta).
    #[inline]
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Set the feature subsample rate.
    #[inline]
    pub fn feature_subsample_rate(mut self, rate: f64) -> Self {
        self.feature_subsample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set the random seed for feature subsampling.
    #[inline]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the per-sample decay factor for leaf statistics.
    #[inline]
    pub fn leaf_decay_alpha(mut self, alpha: f64) -> Self {
        self.leaf_decay_alpha = Some(alpha);
        self
    }

    /// Optionally set the per-sample decay factor for leaf statistics.
    #[inline]
    pub fn leaf_decay_alpha_opt(mut self, alpha: Option<f64>) -> Self {
        self.leaf_decay_alpha = alpha;
        self
    }

    /// Set the re-evaluation interval for max-depth leaves.
    #[inline]
    pub fn split_reeval_interval(mut self, interval: usize) -> Self {
        self.split_reeval_interval = Some(interval);
        self
    }

    /// Optionally set the re-evaluation interval for max-depth leaves.
    #[inline]
    pub fn split_reeval_interval_opt(mut self, interval: Option<usize>) -> Self {
        self.split_reeval_interval = interval;
        self
    }

    /// Set the per-feature type declarations.
    #[inline]
    pub fn feature_types(mut self, types: Vec<FeatureType>) -> Self {
        self.feature_types = Some(types);
        self
    }

    /// Optionally set the per-feature type declarations.
    #[inline]
    pub fn feature_types_opt(mut self, types: Option<Vec<FeatureType>>) -> Self {
        self.feature_types = types;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let cfg = TreeConfig::default();
        assert_eq!(cfg.max_depth, 6);
        assert_eq!(cfg.n_bins, 64);
        assert!((cfg.lambda - 1.0).abs() < f64::EPSILON);
        assert!((cfg.gamma - 0.0).abs() < f64::EPSILON);
        assert_eq!(cfg.grace_period, 200);
        assert!((cfg.delta - 1e-7).abs() < f64::EPSILON);
        assert!((cfg.feature_subsample_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn new_equals_default() {
        let a = TreeConfig::new();
        let b = TreeConfig::default();
        assert_eq!(a.max_depth, b.max_depth);
        assert_eq!(a.n_bins, b.n_bins);
        assert!((a.lambda - b.lambda).abs() < f64::EPSILON);
        assert!((a.gamma - b.gamma).abs() < f64::EPSILON);
        assert_eq!(a.grace_period, b.grace_period);
        assert!((a.delta - b.delta).abs() < f64::EPSILON);
        assert!((a.feature_subsample_rate - b.feature_subsample_rate).abs() < f64::EPSILON);
    }

    #[test]
    fn builder_chain() {
        let cfg = TreeConfig::new()
            .max_depth(10)
            .n_bins(128)
            .lambda(0.5)
            .gamma(0.1)
            .grace_period(500)
            .delta(1e-3)
            .feature_subsample_rate(0.8);

        assert_eq!(cfg.max_depth, 10);
        assert_eq!(cfg.n_bins, 128);
        assert!((cfg.lambda - 0.5).abs() < f64::EPSILON);
        assert!((cfg.gamma - 0.1).abs() < f64::EPSILON);
        assert_eq!(cfg.grace_period, 500);
        assert!((cfg.delta - 1e-3).abs() < f64::EPSILON);
        assert!((cfg.feature_subsample_rate - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn feature_subsample_rate_clamped() {
        let cfg = TreeConfig::new().feature_subsample_rate(1.5);
        assert!((cfg.feature_subsample_rate - 1.0).abs() < f64::EPSILON);

        let cfg = TreeConfig::new().feature_subsample_rate(-0.3);
        assert!((cfg.feature_subsample_rate - 0.0).abs() < f64::EPSILON);
    }
}
