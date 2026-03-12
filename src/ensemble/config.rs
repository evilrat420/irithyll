//! SGBT configuration with builder pattern and full validation.
//!
//! [`SGBTConfig`] holds all hyperparameters for the Streaming Gradient Boosted
//! Trees ensemble. Use [`SGBTConfig::builder`] for ergonomic construction with
//! validation on [`build()`](SGBTConfigBuilder::build).

use crate::drift::adwin::Adwin;
use crate::drift::ddm::Ddm;
use crate::drift::pht::PageHinkleyTest;
use crate::drift::DriftDetector;
use crate::ensemble::variants::SGBTVariant;
use crate::error::{ConfigError, Result};

// ---------------------------------------------------------------------------
// FeatureType
// ---------------------------------------------------------------------------

/// Declares whether a feature is continuous (default) or categorical.
///
/// Categorical features are handled differently in the tree construction:
/// - **Binning:** One bin per observed category value instead of equal-width bins.
/// - **Split evaluation:** Fisher optimal binary partitioning — categories are sorted
///   by gradient_sum/hessian_sum ratio, then the best contiguous partition is found
///   using the same left-to-right XGBoost gain scan.
/// - **Routing:** Categorical splits use a `u64` bitmask where bit `i` set means
///   category `i` goes left. This supports up to 64 distinct category values per feature.
///
/// # Example
///
/// ```
/// use irithyll::ensemble::config::FeatureType;
/// use irithyll::SGBTConfig;
///
/// let config = SGBTConfig::builder()
///     .feature_types(vec![FeatureType::Continuous, FeatureType::Categorical])
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FeatureType {
    /// Numeric feature split by threshold comparisons (default).
    #[default]
    Continuous,
    /// Categorical feature split by bitmask partitioning.
    Categorical,
}

// ---------------------------------------------------------------------------
// DriftDetectorType
// ---------------------------------------------------------------------------

/// Which drift detector to instantiate for each boosting step.
///
/// Each variant stores the detector's configuration parameters so that fresh
/// instances can be created on demand (e.g. when replacing a drifted tree).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DriftDetectorType {
    /// Page-Hinkley Test with custom delta (magnitude tolerance) and lambda
    /// (detection threshold).
    PageHinkley {
        /// Magnitude tolerance. Default 0.005.
        delta: f64,
        /// Detection threshold. Default 50.0.
        lambda: f64,
    },

    /// ADWIN with custom confidence parameter.
    Adwin {
        /// Confidence (smaller = fewer false positives). Default 0.002.
        delta: f64,
    },

    /// DDM with custom warning/drift levels and minimum warmup instances.
    Ddm {
        /// Warning threshold multiplier. Default 2.0.
        warning_level: f64,
        /// Drift threshold multiplier. Default 3.0.
        drift_level: f64,
        /// Minimum observations before detection activates. Default 30.
        min_instances: u64,
    },
}

impl Default for DriftDetectorType {
    fn default() -> Self {
        DriftDetectorType::PageHinkley {
            delta: 0.005,
            lambda: 50.0,
        }
    }
}

impl DriftDetectorType {
    /// Create a new, fresh drift detector from this configuration.
    pub fn create(&self) -> Box<dyn DriftDetector> {
        match self {
            Self::PageHinkley { delta, lambda } => {
                Box::new(PageHinkleyTest::with_params(*delta, *lambda))
            }
            Self::Adwin { delta } => Box::new(Adwin::with_delta(*delta)),
            Self::Ddm {
                warning_level,
                drift_level,
                min_instances,
            } => Box::new(Ddm::with_params(
                *warning_level,
                *drift_level,
                *min_instances,
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// SGBTConfig
// ---------------------------------------------------------------------------

/// Configuration for the SGBT ensemble.
///
/// All numeric parameters are validated at build time via [`SGBTConfigBuilder`].
///
/// # Defaults
///
/// | Parameter                | Default              |
/// |--------------------------|----------------------|
/// | `n_steps`                | 100                  |
/// | `learning_rate`          | 0.0125               |
/// | `feature_subsample_rate` | 0.75                 |
/// | `max_depth`              | 6                    |
/// | `n_bins`                 | 64                   |
/// | `lambda`                 | 1.0                  |
/// | `gamma`                  | 0.0                  |
/// | `grace_period`           | 200                  |
/// | `delta`                  | 1e-7                 |
/// | `drift_detector`         | PageHinkley(0.005, 50.0) |
/// | `variant`                | Standard             |
/// | `seed`                   | 0xDEAD_BEEF_CAFE_4242 |
/// | `initial_target_count`   | 50                   |
/// | `leaf_half_life`         | None (disabled)      |
/// | `max_tree_samples`       | None (disabled)      |
/// | `split_reeval_interval`  | None (disabled)      |
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SGBTConfig {
    /// Number of boosting steps (trees in the ensemble). Default 100.
    pub n_steps: usize,
    /// Learning rate (shrinkage). Default 0.0125.
    pub learning_rate: f64,
    /// Fraction of features to subsample per tree. Default 0.75.
    pub feature_subsample_rate: f64,
    /// Maximum tree depth. Default 6.
    pub max_depth: usize,
    /// Number of histogram bins. Default 64.
    pub n_bins: usize,
    /// L2 regularization parameter (lambda). Default 1.0.
    pub lambda: f64,
    /// Minimum split gain (gamma). Default 0.0.
    pub gamma: f64,
    /// Grace period: min samples before evaluating splits. Default 200.
    pub grace_period: usize,
    /// Hoeffding bound confidence (delta). Default 1e-7.
    pub delta: f64,
    /// Drift detector type for tree replacement. Default: PageHinkley.
    pub drift_detector: DriftDetectorType,
    /// SGBT computational variant. Default: Standard.
    pub variant: SGBTVariant,
    /// Random seed for deterministic reproducibility. Default: 0xDEAD_BEEF_CAFE_4242.
    ///
    /// Controls feature subsampling and variant skip/MI stochastic decisions.
    /// Two models with the same seed and same data will produce identical results.
    pub seed: u64,
    /// Number of initial targets to collect before computing the base prediction.
    /// Default: 50.
    pub initial_target_count: usize,

    /// Half-life for exponential leaf decay (in samples per leaf).
    ///
    /// After `leaf_half_life` samples, a leaf's accumulated gradient/hessian
    /// statistics have half the weight of the most recent sample. This causes
    /// the model to continuously adapt to changing data distributions rather
    /// than freezing on early observations.
    ///
    /// `None` (default) disables decay — traditional monotonic accumulation.
    #[serde(default)]
    pub leaf_half_life: Option<usize>,

    /// Maximum samples a single tree processes before proactive replacement.
    ///
    /// After this many samples, the tree is replaced with a fresh one regardless
    /// of drift detector state. Prevents stale tree structure from persisting
    /// when the drift detector is not sensitive enough.
    ///
    /// `None` (default) disables time-based replacement.
    #[serde(default)]
    pub max_tree_samples: Option<u64>,

    /// Interval (in samples per leaf) at which max-depth leaves re-evaluate
    /// whether a split would improve them.
    ///
    /// Inspired by EFDT (Manapragada et al. 2018). When a leaf has accumulated
    /// `split_reeval_interval` samples since its last evaluation and has reached
    /// max depth, it re-evaluates whether a split should be performed.
    ///
    /// `None` (default) disables re-evaluation — max-depth leaves are permanent.
    #[serde(default)]
    pub split_reeval_interval: Option<usize>,

    /// Optional human-readable feature names.
    ///
    /// When set, enables [`named_feature_importances`](super::SGBT::named_feature_importances) and
    /// [`train_one_named`](super::SGBT::train_one_named) for production-friendly named access.
    /// Length must match the number of features in training data.
    #[serde(default)]
    pub feature_names: Option<Vec<String>>,

    /// Optional per-feature type declarations.
    ///
    /// When set, declares which features are categorical vs continuous.
    /// Categorical features use one-bin-per-category binning and Fisher
    /// optimal binary partitioning for split evaluation.
    /// Length must match the number of features in training data.
    ///
    /// `None` (default) treats all features as continuous.
    #[serde(default)]
    pub feature_types: Option<Vec<FeatureType>>,

    /// Gradient clipping threshold in standard deviations per leaf.
    ///
    /// When enabled, each leaf tracks an EWMA of gradient mean and variance.
    /// Incoming gradients that exceed `mean ± sigma * gradient_clip_sigma` are
    /// clamped to the boundary. This prevents outlier samples from corrupting
    /// leaf statistics, which is critical in streaming settings where sudden
    /// label floods can destabilize the model.
    ///
    /// Typical value: 3.0 (3-sigma clipping).
    /// `None` (default) disables gradient clipping.
    #[serde(default)]
    pub gradient_clip_sigma: Option<f64>,

    /// Per-feature monotonic constraints.
    ///
    /// Each element specifies the monotonic relationship between a feature and
    /// the prediction:
    /// - `+1`: prediction must be non-decreasing as feature value increases.
    /// - `-1`: prediction must be non-increasing as feature value increases.
    /// - `0`: no constraint (unconstrained).
    ///
    /// During split evaluation, candidate splits that would violate monotonicity
    /// (left child value > right child value for +1 constraints, or vice versa)
    /// are rejected.
    ///
    /// Length must match the number of features in training data.
    /// `None` (default) means no monotonic constraints.
    #[serde(default)]
    pub monotone_constraints: Option<Vec<i8>>,

    /// EWMA smoothing factor for quality-based tree pruning.
    ///
    /// When `Some(alpha)`, each boosting step tracks an exponentially weighted
    /// moving average of its marginal contribution to the ensemble. Trees whose
    /// contribution drops below [`quality_prune_threshold`](Self::quality_prune_threshold)
    /// for [`quality_prune_patience`](Self::quality_prune_patience) consecutive
    /// samples are replaced with a fresh tree that can learn the current regime.
    ///
    /// This prevents "dead wood" — trees from a past regime that no longer
    /// contribute meaningfully to ensemble accuracy.
    ///
    /// `None` (default) disables quality-based pruning.
    /// Suggested value: 0.01.
    #[serde(default)]
    pub quality_prune_alpha: Option<f64>,

    /// Minimum contribution threshold for quality-based pruning.
    ///
    /// A tree's EWMA contribution must stay above this value to avoid being
    /// flagged as dead wood. Only used when `quality_prune_alpha` is `Some`.
    ///
    /// Default: 1e-6.
    #[serde(default = "default_quality_prune_threshold")]
    pub quality_prune_threshold: f64,

    /// Consecutive low-contribution samples before a tree is replaced.
    ///
    /// After this many consecutive samples where a tree's EWMA contribution
    /// is below `quality_prune_threshold`, the tree is reset. Only used when
    /// `quality_prune_alpha` is `Some`.
    ///
    /// Default: 500.
    #[serde(default = "default_quality_prune_patience")]
    pub quality_prune_patience: u64,

    /// EWMA smoothing factor for error-weighted sample importance.
    ///
    /// When `Some(alpha)`, samples the model predicted poorly get higher
    /// effective weight during histogram accumulation. The weight is:
    /// `1.0 + |error| / (rolling_mean_error + epsilon)`, capped at 10x.
    ///
    /// This is a streaming version of AdaBoost's reweighting applied at the
    /// gradient level — learning capacity focuses on hard/novel patterns,
    /// enabling faster adaptation to regime changes.
    ///
    /// `None` (default) disables error weighting.
    /// Suggested value: 0.01.
    #[serde(default)]
    pub error_weight_alpha: Option<f64>,
}

fn default_quality_prune_threshold() -> f64 {
    1e-6
}

fn default_quality_prune_patience() -> u64 {
    500
}

impl Default for SGBTConfig {
    fn default() -> Self {
        Self {
            n_steps: 100,
            learning_rate: 0.0125,
            feature_subsample_rate: 0.75,
            max_depth: 6,
            n_bins: 64,
            lambda: 1.0,
            gamma: 0.0,
            grace_period: 200,
            delta: 1e-7,
            drift_detector: DriftDetectorType::default(),
            variant: SGBTVariant::default(),
            seed: 0xDEAD_BEEF_CAFE_4242,
            initial_target_count: 50,
            leaf_half_life: None,
            max_tree_samples: None,
            split_reeval_interval: None,
            feature_names: None,
            feature_types: None,
            gradient_clip_sigma: None,
            monotone_constraints: None,
            quality_prune_alpha: None,
            quality_prune_threshold: 1e-6,
            quality_prune_patience: 500,
            error_weight_alpha: None,
        }
    }
}

impl SGBTConfig {
    /// Start building a configuration via the builder pattern.
    pub fn builder() -> SGBTConfigBuilder {
        SGBTConfigBuilder::default()
    }
}

// ---------------------------------------------------------------------------
// SGBTConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for [`SGBTConfig`] with validation on [`build()`](Self::build).
///
/// # Example
///
/// ```
/// use irithyll::ensemble::config::{SGBTConfig, DriftDetectorType};
/// use irithyll::ensemble::variants::SGBTVariant;
///
/// let config = SGBTConfig::builder()
///     .n_steps(200)
///     .learning_rate(0.05)
///     .drift_detector(DriftDetectorType::Adwin { delta: 0.01 })
///     .variant(SGBTVariant::Skip { k: 10 })
///     .build()
///     .expect("valid config");
/// ```
#[derive(Debug, Clone, Default)]
pub struct SGBTConfigBuilder {
    config: SGBTConfig,
}

impl SGBTConfigBuilder {
    /// Set the number of boosting steps (trees in the ensemble).
    pub fn n_steps(mut self, n: usize) -> Self {
        self.config.n_steps = n;
        self
    }

    /// Set the learning rate (shrinkage factor).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set the fraction of features to subsample per tree.
    pub fn feature_subsample_rate(mut self, rate: f64) -> Self {
        self.config.feature_subsample_rate = rate;
        self
    }

    /// Set the maximum tree depth.
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = depth;
        self
    }

    /// Set the number of histogram bins per feature.
    pub fn n_bins(mut self, bins: usize) -> Self {
        self.config.n_bins = bins;
        self
    }

    /// Set the L2 regularization parameter (lambda).
    pub fn lambda(mut self, l: f64) -> Self {
        self.config.lambda = l;
        self
    }

    /// Set the minimum split gain (gamma).
    pub fn gamma(mut self, g: f64) -> Self {
        self.config.gamma = g;
        self
    }

    /// Set the grace period (minimum samples before evaluating splits).
    pub fn grace_period(mut self, gp: usize) -> Self {
        self.config.grace_period = gp;
        self
    }

    /// Set the Hoeffding bound confidence parameter (delta).
    pub fn delta(mut self, d: f64) -> Self {
        self.config.delta = d;
        self
    }

    /// Set the drift detector type for tree replacement.
    pub fn drift_detector(mut self, dt: DriftDetectorType) -> Self {
        self.config.drift_detector = dt;
        self
    }

    /// Set the SGBT computational variant.
    pub fn variant(mut self, v: SGBTVariant) -> Self {
        self.config.variant = v;
        self
    }

    /// Set the random seed for deterministic reproducibility.
    ///
    /// Controls feature subsampling and variant skip/MI stochastic decisions.
    /// Two models with the same seed and data sequence will produce identical results.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Set the number of initial targets to collect before computing the base prediction.
    ///
    /// The model collects this many target values before initializing the base
    /// prediction (via `loss.initial_prediction`). Default: 50.
    pub fn initial_target_count(mut self, count: usize) -> Self {
        self.config.initial_target_count = count;
        self
    }

    /// Set the half-life for exponential leaf decay (in samples per leaf).
    ///
    /// After `n` samples, a leaf's accumulated statistics have half the weight
    /// of the most recent sample. Enables continuous adaptation to concept drift.
    pub fn leaf_half_life(mut self, n: usize) -> Self {
        self.config.leaf_half_life = Some(n);
        self
    }

    /// Set the maximum samples a single tree processes before proactive replacement.
    ///
    /// After `n` samples, the tree is replaced regardless of drift detector state.
    pub fn max_tree_samples(mut self, n: u64) -> Self {
        self.config.max_tree_samples = Some(n);
        self
    }

    /// Set the split re-evaluation interval for max-depth leaves.
    ///
    /// Every `n` samples per leaf, max-depth leaves re-evaluate whether a split
    /// would improve them. Inspired by EFDT (Manapragada et al. 2018).
    pub fn split_reeval_interval(mut self, n: usize) -> Self {
        self.config.split_reeval_interval = Some(n);
        self
    }

    /// Set human-readable feature names.
    ///
    /// Enables named feature importances and named training input.
    /// Names must be unique; validated at [`build()`](Self::build).
    pub fn feature_names(mut self, names: Vec<String>) -> Self {
        self.config.feature_names = Some(names);
        self
    }

    /// Set per-feature type declarations.
    ///
    /// Declares which features are categorical vs continuous. Categorical features
    /// use one-bin-per-category binning and Fisher optimal binary partitioning.
    /// Supports up to 64 distinct category values per categorical feature.
    pub fn feature_types(mut self, types: Vec<FeatureType>) -> Self {
        self.config.feature_types = Some(types);
        self
    }

    /// Set per-leaf gradient clipping threshold (in standard deviations).
    ///
    /// Each leaf tracks an EWMA of gradient mean and variance. Gradients
    /// exceeding `mean ± sigma * n` are clamped. Prevents outlier labels
    /// from corrupting streaming model stability.
    ///
    /// Typical value: 3.0 (3-sigma clipping).
    pub fn gradient_clip_sigma(mut self, sigma: f64) -> Self {
        self.config.gradient_clip_sigma = Some(sigma);
        self
    }

    /// Set per-feature monotonic constraints.
    ///
    /// `+1` = non-decreasing, `-1` = non-increasing, `0` = unconstrained.
    /// Candidate splits violating monotonicity are rejected during tree growth.
    pub fn monotone_constraints(mut self, constraints: Vec<i8>) -> Self {
        self.config.monotone_constraints = Some(constraints);
        self
    }

    /// Enable quality-based tree pruning with the given EWMA smoothing factor.
    ///
    /// Trees whose marginal contribution drops below the threshold for
    /// `patience` consecutive samples are replaced with fresh trees.
    /// Suggested alpha: 0.01.
    pub fn quality_prune_alpha(mut self, alpha: f64) -> Self {
        self.config.quality_prune_alpha = Some(alpha);
        self
    }

    /// Set the minimum contribution threshold for quality-based pruning.
    ///
    /// Default: 1e-6. Only relevant when `quality_prune_alpha` is set.
    pub fn quality_prune_threshold(mut self, threshold: f64) -> Self {
        self.config.quality_prune_threshold = threshold;
        self
    }

    /// Set the patience (consecutive low-contribution samples) before pruning.
    ///
    /// Default: 500. Only relevant when `quality_prune_alpha` is set.
    pub fn quality_prune_patience(mut self, patience: u64) -> Self {
        self.config.quality_prune_patience = patience;
        self
    }

    /// Enable error-weighted sample importance with the given EWMA smoothing factor.
    ///
    /// Samples the model predicted poorly get higher effective weight.
    /// Suggested alpha: 0.01.
    pub fn error_weight_alpha(mut self, alpha: f64) -> Self {
        self.config.error_weight_alpha = Some(alpha);
        self
    }

    /// Validate and build the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidConfig`](crate::IrithyllError::InvalidConfig) with a structured
    /// [`ConfigError`] if any parameter is out of its valid range.
    pub fn build(self) -> Result<SGBTConfig> {
        let c = &self.config;

        // -- Ensemble-level parameters --
        if c.n_steps == 0 {
            return Err(ConfigError::out_of_range("n_steps", "must be > 0", c.n_steps).into());
        }
        if c.learning_rate <= 0.0 || c.learning_rate > 1.0 {
            return Err(ConfigError::out_of_range(
                "learning_rate",
                "must be in (0, 1]",
                c.learning_rate,
            )
            .into());
        }
        if c.feature_subsample_rate <= 0.0 || c.feature_subsample_rate > 1.0 {
            return Err(ConfigError::out_of_range(
                "feature_subsample_rate",
                "must be in (0, 1]",
                c.feature_subsample_rate,
            )
            .into());
        }

        // -- Tree-level parameters --
        if c.max_depth == 0 {
            return Err(ConfigError::out_of_range("max_depth", "must be > 0", c.max_depth).into());
        }
        if c.n_bins < 2 {
            return Err(ConfigError::out_of_range("n_bins", "must be >= 2", c.n_bins).into());
        }
        if c.lambda < 0.0 {
            return Err(ConfigError::out_of_range("lambda", "must be >= 0", c.lambda).into());
        }
        if c.gamma < 0.0 {
            return Err(ConfigError::out_of_range("gamma", "must be >= 0", c.gamma).into());
        }
        if c.grace_period == 0 {
            return Err(
                ConfigError::out_of_range("grace_period", "must be > 0", c.grace_period).into(),
            );
        }
        if c.delta <= 0.0 || c.delta >= 1.0 {
            return Err(ConfigError::out_of_range("delta", "must be in (0, 1)", c.delta).into());
        }

        if c.initial_target_count == 0 {
            return Err(ConfigError::out_of_range(
                "initial_target_count",
                "must be > 0",
                c.initial_target_count,
            )
            .into());
        }

        // -- Streaming adaptation parameters --
        if let Some(hl) = c.leaf_half_life {
            if hl == 0 {
                return Err(ConfigError::out_of_range("leaf_half_life", "must be >= 1", hl).into());
            }
        }
        if let Some(max) = c.max_tree_samples {
            if max < 100 {
                return Err(
                    ConfigError::out_of_range("max_tree_samples", "must be >= 100", max).into(),
                );
            }
        }
        if let Some(interval) = c.split_reeval_interval {
            if interval < c.grace_period {
                return Err(ConfigError::invalid(
                    "split_reeval_interval",
                    format!(
                        "must be >= grace_period ({}), got {}",
                        c.grace_period, interval
                    ),
                )
                .into());
            }
        }

        // -- Feature names --
        if let Some(ref names) = c.feature_names {
            let mut seen = std::collections::HashSet::new();
            for name in names {
                if !seen.insert(name.as_str()) {
                    return Err(ConfigError::invalid(
                        "feature_names",
                        format!("duplicate feature name: '{}'", name),
                    )
                    .into());
                }
            }
        }

        // -- Feature types --
        if let Some(ref types) = c.feature_types {
            if let Some(ref names) = c.feature_names {
                if !names.is_empty() && !types.is_empty() && names.len() != types.len() {
                    return Err(ConfigError::invalid(
                        "feature_types",
                        format!(
                            "length ({}) must match feature_names length ({})",
                            types.len(),
                            names.len()
                        ),
                    )
                    .into());
                }
            }
        }

        // -- Gradient clipping --
        if let Some(sigma) = c.gradient_clip_sigma {
            if sigma <= 0.0 {
                return Err(
                    ConfigError::out_of_range("gradient_clip_sigma", "must be > 0", sigma).into(),
                );
            }
        }

        // -- Monotonic constraints --
        if let Some(ref mc) = c.monotone_constraints {
            for (i, &v) in mc.iter().enumerate() {
                if v != -1 && v != 0 && v != 1 {
                    return Err(ConfigError::invalid(
                        "monotone_constraints",
                        format!("feature {}: must be -1, 0, or +1, got {}", i, v),
                    )
                    .into());
                }
            }
        }

        // -- Quality-based pruning parameters --
        if let Some(alpha) = c.quality_prune_alpha {
            if alpha <= 0.0 || alpha >= 1.0 {
                return Err(ConfigError::out_of_range(
                    "quality_prune_alpha",
                    "must be in (0, 1)",
                    alpha,
                )
                .into());
            }
        }
        if c.quality_prune_threshold <= 0.0 {
            return Err(ConfigError::out_of_range(
                "quality_prune_threshold",
                "must be > 0",
                c.quality_prune_threshold,
            )
            .into());
        }
        if c.quality_prune_patience == 0 {
            return Err(ConfigError::out_of_range(
                "quality_prune_patience",
                "must be > 0",
                c.quality_prune_patience,
            )
            .into());
        }

        // -- Error-weighted sample importance --
        if let Some(alpha) = c.error_weight_alpha {
            if alpha <= 0.0 || alpha >= 1.0 {
                return Err(ConfigError::out_of_range(
                    "error_weight_alpha",
                    "must be in (0, 1)",
                    alpha,
                )
                .into());
            }
        }

        // -- Drift detector parameters --
        match &c.drift_detector {
            DriftDetectorType::PageHinkley { delta, lambda } => {
                if *delta <= 0.0 {
                    return Err(ConfigError::out_of_range(
                        "drift_detector.PageHinkley.delta",
                        "must be > 0",
                        delta,
                    )
                    .into());
                }
                if *lambda <= 0.0 {
                    return Err(ConfigError::out_of_range(
                        "drift_detector.PageHinkley.lambda",
                        "must be > 0",
                        lambda,
                    )
                    .into());
                }
            }
            DriftDetectorType::Adwin { delta } => {
                if *delta <= 0.0 || *delta >= 1.0 {
                    return Err(ConfigError::out_of_range(
                        "drift_detector.Adwin.delta",
                        "must be in (0, 1)",
                        delta,
                    )
                    .into());
                }
            }
            DriftDetectorType::Ddm {
                warning_level,
                drift_level,
                min_instances,
            } => {
                if *warning_level <= 0.0 {
                    return Err(ConfigError::out_of_range(
                        "drift_detector.Ddm.warning_level",
                        "must be > 0",
                        warning_level,
                    )
                    .into());
                }
                if *drift_level <= 0.0 {
                    return Err(ConfigError::out_of_range(
                        "drift_detector.Ddm.drift_level",
                        "must be > 0",
                        drift_level,
                    )
                    .into());
                }
                if *drift_level <= *warning_level {
                    return Err(ConfigError::invalid(
                        "drift_detector.Ddm.drift_level",
                        format!(
                            "must be > warning_level ({}), got {}",
                            warning_level, drift_level
                        ),
                    )
                    .into());
                }
                if *min_instances == 0 {
                    return Err(ConfigError::out_of_range(
                        "drift_detector.Ddm.min_instances",
                        "must be > 0",
                        min_instances,
                    )
                    .into());
                }
            }
        }

        // -- Variant parameters --
        match &c.variant {
            SGBTVariant::Standard => {} // no extra validation
            SGBTVariant::Skip { k } => {
                if *k == 0 {
                    return Err(
                        ConfigError::out_of_range("variant.Skip.k", "must be > 0", k).into(),
                    );
                }
            }
            SGBTVariant::MultipleIterations { multiplier } => {
                if *multiplier <= 0.0 {
                    return Err(ConfigError::out_of_range(
                        "variant.MultipleIterations.multiplier",
                        "must be > 0",
                        multiplier,
                    )
                    .into());
                }
            }
        }

        Ok(self.config)
    }
}

// ---------------------------------------------------------------------------
// Display impls
// ---------------------------------------------------------------------------

impl std::fmt::Display for DriftDetectorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PageHinkley { delta, lambda } => {
                write!(f, "PageHinkley(delta={}, lambda={})", delta, lambda)
            }
            Self::Adwin { delta } => write!(f, "Adwin(delta={})", delta),
            Self::Ddm {
                warning_level,
                drift_level,
                min_instances,
            } => write!(
                f,
                "Ddm(warning={}, drift={}, min_instances={})",
                warning_level, drift_level, min_instances
            ),
        }
    }
}

impl std::fmt::Display for SGBTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SGBTConfig {{ steps={}, lr={}, depth={}, bins={}, variant={}, drift={} }}",
            self.n_steps,
            self.learning_rate,
            self.max_depth,
            self.n_bins,
            self.variant,
            self.drift_detector,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // 1. Default config values are correct
    // ------------------------------------------------------------------
    #[test]
    fn default_config_values() {
        let cfg = SGBTConfig::default();
        assert_eq!(cfg.n_steps, 100);
        assert!((cfg.learning_rate - 0.0125).abs() < f64::EPSILON);
        assert!((cfg.feature_subsample_rate - 0.75).abs() < f64::EPSILON);
        assert_eq!(cfg.max_depth, 6);
        assert_eq!(cfg.n_bins, 64);
        assert!((cfg.lambda - 1.0).abs() < f64::EPSILON);
        assert!((cfg.gamma - 0.0).abs() < f64::EPSILON);
        assert_eq!(cfg.grace_period, 200);
        assert!((cfg.delta - 1e-7).abs() < f64::EPSILON);
        assert_eq!(cfg.variant, SGBTVariant::Standard);
    }

    // ------------------------------------------------------------------
    // 2. Builder chain works
    // ------------------------------------------------------------------
    #[test]
    fn builder_chain() {
        let cfg = SGBTConfig::builder()
            .n_steps(50)
            .learning_rate(0.1)
            .feature_subsample_rate(0.5)
            .max_depth(10)
            .n_bins(128)
            .lambda(0.5)
            .gamma(0.1)
            .grace_period(500)
            .delta(1e-3)
            .drift_detector(DriftDetectorType::Adwin { delta: 0.01 })
            .variant(SGBTVariant::Skip { k: 5 })
            .build()
            .expect("valid config");

        assert_eq!(cfg.n_steps, 50);
        assert!((cfg.learning_rate - 0.1).abs() < f64::EPSILON);
        assert!((cfg.feature_subsample_rate - 0.5).abs() < f64::EPSILON);
        assert_eq!(cfg.max_depth, 10);
        assert_eq!(cfg.n_bins, 128);
        assert!((cfg.lambda - 0.5).abs() < f64::EPSILON);
        assert!((cfg.gamma - 0.1).abs() < f64::EPSILON);
        assert_eq!(cfg.grace_period, 500);
        assert!((cfg.delta - 1e-3).abs() < f64::EPSILON);
        assert_eq!(cfg.variant, SGBTVariant::Skip { k: 5 });

        // Verify drift detector type is Adwin.
        match &cfg.drift_detector {
            DriftDetectorType::Adwin { delta } => {
                assert!((delta - 0.01).abs() < f64::EPSILON);
            }
            _ => panic!("expected Adwin drift detector"),
        }
    }

    // ------------------------------------------------------------------
    // 3. Validation rejects invalid values
    // ------------------------------------------------------------------
    #[test]
    fn validation_rejects_n_steps_zero() {
        let result = SGBTConfig::builder().n_steps(0).build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("n_steps"));
    }

    #[test]
    fn validation_rejects_learning_rate_zero() {
        let result = SGBTConfig::builder().learning_rate(0.0).build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("learning_rate"));
    }

    #[test]
    fn validation_rejects_learning_rate_above_one() {
        let result = SGBTConfig::builder().learning_rate(1.5).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_accepts_learning_rate_one() {
        let result = SGBTConfig::builder().learning_rate(1.0).build();
        assert!(result.is_ok());
    }

    #[test]
    fn validation_rejects_negative_learning_rate() {
        let result = SGBTConfig::builder().learning_rate(-0.1).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_feature_subsample_zero() {
        let result = SGBTConfig::builder().feature_subsample_rate(0.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_feature_subsample_above_one() {
        let result = SGBTConfig::builder().feature_subsample_rate(1.01).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_max_depth_zero() {
        let result = SGBTConfig::builder().max_depth(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_n_bins_one() {
        let result = SGBTConfig::builder().n_bins(1).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_negative_lambda() {
        let result = SGBTConfig::builder().lambda(-0.1).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_accepts_zero_lambda() {
        let result = SGBTConfig::builder().lambda(0.0).build();
        assert!(result.is_ok());
    }

    #[test]
    fn validation_rejects_negative_gamma() {
        let result = SGBTConfig::builder().gamma(-0.1).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_grace_period_zero() {
        let result = SGBTConfig::builder().grace_period(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_delta_zero() {
        let result = SGBTConfig::builder().delta(0.0).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_delta_one() {
        let result = SGBTConfig::builder().delta(1.0).build();
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // 3b. Drift detector parameter validation
    // ------------------------------------------------------------------
    #[test]
    fn validation_rejects_pht_negative_delta() {
        let result = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::PageHinkley {
                delta: -1.0,
                lambda: 50.0,
            })
            .build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("PageHinkley"));
    }

    #[test]
    fn validation_rejects_pht_zero_lambda() {
        let result = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::PageHinkley {
                delta: 0.005,
                lambda: 0.0,
            })
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_adwin_delta_out_of_range() {
        let result = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::Adwin { delta: 0.0 })
            .build();
        assert!(result.is_err());

        let result = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::Adwin { delta: 1.0 })
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_ddm_warning_above_drift() {
        let result = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::Ddm {
                warning_level: 3.0,
                drift_level: 2.0,
                min_instances: 30,
            })
            .build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("drift_level"));
        assert!(msg.contains("must be > warning_level"));
    }

    #[test]
    fn validation_rejects_ddm_equal_levels() {
        let result = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::Ddm {
                warning_level: 2.0,
                drift_level: 2.0,
                min_instances: 30,
            })
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_ddm_zero_min_instances() {
        let result = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::Ddm {
                warning_level: 2.0,
                drift_level: 3.0,
                min_instances: 0,
            })
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_ddm_zero_warning_level() {
        let result = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::Ddm {
                warning_level: 0.0,
                drift_level: 3.0,
                min_instances: 30,
            })
            .build();
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // 3c. Variant parameter validation
    // ------------------------------------------------------------------
    #[test]
    fn validation_rejects_skip_k_zero() {
        let result = SGBTConfig::builder()
            .variant(SGBTVariant::Skip { k: 0 })
            .build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("Skip"));
    }

    #[test]
    fn validation_rejects_mi_zero_multiplier() {
        let result = SGBTConfig::builder()
            .variant(SGBTVariant::MultipleIterations { multiplier: 0.0 })
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_mi_negative_multiplier() {
        let result = SGBTConfig::builder()
            .variant(SGBTVariant::MultipleIterations { multiplier: -1.0 })
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_accepts_standard_variant() {
        let result = SGBTConfig::builder().variant(SGBTVariant::Standard).build();
        assert!(result.is_ok());
    }

    // ------------------------------------------------------------------
    // 4. DriftDetectorType creates correct detector types
    // ------------------------------------------------------------------
    #[test]
    fn drift_detector_type_creates_page_hinkley() {
        let dt = DriftDetectorType::PageHinkley {
            delta: 0.01,
            lambda: 100.0,
        };
        let mut detector = dt.create();

        // Should start with zero mean.
        assert_eq!(detector.estimated_mean(), 0.0);

        // Feed a stable value -- should not drift.
        let signal = detector.update(1.0);
        assert_ne!(signal, crate::drift::DriftSignal::Drift);
    }

    #[test]
    fn drift_detector_type_creates_adwin() {
        let dt = DriftDetectorType::Adwin { delta: 0.05 };
        let mut detector = dt.create();

        assert_eq!(detector.estimated_mean(), 0.0);
        let signal = detector.update(1.0);
        assert_ne!(signal, crate::drift::DriftSignal::Drift);
    }

    #[test]
    fn drift_detector_type_creates_ddm() {
        let dt = DriftDetectorType::Ddm {
            warning_level: 2.0,
            drift_level: 3.0,
            min_instances: 30,
        };
        let mut detector = dt.create();

        assert_eq!(detector.estimated_mean(), 0.0);
        let signal = detector.update(0.1);
        assert_eq!(signal, crate::drift::DriftSignal::Stable);
    }

    // ------------------------------------------------------------------
    // 5. Default drift detector is PageHinkley
    // ------------------------------------------------------------------
    #[test]
    fn default_drift_detector_is_page_hinkley() {
        let dt = DriftDetectorType::default();
        match dt {
            DriftDetectorType::PageHinkley { delta, lambda } => {
                assert!((delta - 0.005).abs() < f64::EPSILON);
                assert!((lambda - 50.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected default DriftDetectorType to be PageHinkley"),
        }
    }

    // ------------------------------------------------------------------
    // 6. Default config builds without error
    // ------------------------------------------------------------------
    #[test]
    fn default_config_builds() {
        let result = SGBTConfig::builder().build();
        assert!(result.is_ok());
    }

    // ------------------------------------------------------------------
    // 7. Config clone preserves all fields
    // ------------------------------------------------------------------
    #[test]
    fn config_clone_preserves_fields() {
        let original = SGBTConfig::builder()
            .n_steps(50)
            .learning_rate(0.05)
            .variant(SGBTVariant::MultipleIterations { multiplier: 5.0 })
            .build()
            .unwrap();

        let cloned = original.clone();

        assert_eq!(cloned.n_steps, 50);
        assert!((cloned.learning_rate - 0.05).abs() < f64::EPSILON);
        assert_eq!(
            cloned.variant,
            SGBTVariant::MultipleIterations { multiplier: 5.0 }
        );
    }

    // ------------------------------------------------------------------
    // 8. All three DDM valid configs accepted
    // ------------------------------------------------------------------
    #[test]
    fn valid_ddm_config_accepted() {
        let result = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::Ddm {
                warning_level: 1.5,
                drift_level: 2.5,
                min_instances: 10,
            })
            .build();
        assert!(result.is_ok());
    }

    // ------------------------------------------------------------------
    // 9. Created detectors are functional (round-trip test)
    // ------------------------------------------------------------------
    #[test]
    fn created_detectors_are_functional() {
        // Create a detector, feed it data, verify it can detect drift.
        let dt = DriftDetectorType::PageHinkley {
            delta: 0.005,
            lambda: 50.0,
        };
        let mut detector = dt.create();

        // Feed 500 stable values.
        for _ in 0..500 {
            detector.update(1.0);
        }

        // Feed shifted values -- should eventually drift.
        let mut drifted = false;
        for _ in 0..500 {
            if detector.update(10.0) == crate::drift::DriftSignal::Drift {
                drifted = true;
                break;
            }
        }
        assert!(
            drifted,
            "detector created from DriftDetectorType should be functional"
        );
    }

    // ------------------------------------------------------------------
    // 10. Boundary acceptance: n_bins=2 is the minimum valid
    // ------------------------------------------------------------------
    #[test]
    fn boundary_n_bins_two_accepted() {
        let result = SGBTConfig::builder().n_bins(2).build();
        assert!(result.is_ok());
    }

    // ------------------------------------------------------------------
    // 11. Boundary acceptance: grace_period=1 is valid
    // ------------------------------------------------------------------
    #[test]
    fn boundary_grace_period_one_accepted() {
        let result = SGBTConfig::builder().grace_period(1).build();
        assert!(result.is_ok());
    }

    // ------------------------------------------------------------------
    // 12. Feature names — valid config with names
    // ------------------------------------------------------------------
    #[test]
    fn feature_names_accepted() {
        let cfg = SGBTConfig::builder()
            .feature_names(vec!["price".into(), "volume".into(), "spread".into()])
            .build()
            .unwrap();
        assert_eq!(
            cfg.feature_names.as_ref().unwrap(),
            &["price", "volume", "spread"]
        );
    }

    // ------------------------------------------------------------------
    // 13. Feature names — duplicate names rejected
    // ------------------------------------------------------------------
    #[test]
    fn feature_names_rejects_duplicates() {
        let result = SGBTConfig::builder()
            .feature_names(vec!["price".into(), "volume".into(), "price".into()])
            .build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("duplicate"));
    }

    // ------------------------------------------------------------------
    // 14. Feature names — serde backward compat (missing field)
    // ------------------------------------------------------------------
    #[test]
    fn feature_names_serde_backward_compat() {
        // JSON without feature_names should deserialize fine (defaults to None).
        let cfg = SGBTConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: SGBTConfig = serde_json::from_str(&json).unwrap();
        assert!(restored.feature_names.is_none());
    }

    // ------------------------------------------------------------------
    // 15. Feature names — empty vec is valid
    // ------------------------------------------------------------------
    #[test]
    fn feature_names_empty_vec_accepted() {
        let cfg = SGBTConfig::builder().feature_names(vec![]).build().unwrap();
        assert!(cfg.feature_names.unwrap().is_empty());
    }
}
