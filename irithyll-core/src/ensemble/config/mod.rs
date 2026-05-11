//! SGBT configuration with builder pattern and full validation.
//!
//! [`SGBTConfig`] holds all hyperparameters for the Streaming Gradient Boosted
//! Trees ensemble. Use [`SGBTConfig::builder`] for ergonomic construction with
//! validation on [`build()`](SGBTConfigBuilder::build).

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::drift::adwin::Adwin;
use crate::drift::ddm::Ddm;
use crate::drift::pht::PageHinkleyTest;
use crate::drift::DriftDetector;
use crate::ensemble::variants::SGBTVariant;
use crate::error::Result;
use crate::tree::leaf_model::LeafModelType;

mod display;
mod tree_config_helper;
mod validation;

pub(crate) use tree_config_helper::build_tree_config;

pub use crate::feature::FeatureType;

/// How [`DistributionalSGBT`](super::distributional::DistributionalSGBT)
/// estimates uncertainty (σ).
///
/// - **`Empirical`** (default): tracks an EWMA of squared prediction errors.
///   `σ = sqrt(ewma_sq_err)`.  Always calibrated, zero tuning, O(1) compute.
///   Use this when σ drives learning-rate modulation (σ high → learn faster).
///
/// - **`TreeChain`**: trains a full second ensemble of Hoeffding trees to predict
///   log(σ) from features (NGBoost-style dual chain).  Gives *feature-conditional*
///   uncertainty but requires strong signal in the scale gradients.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "_serde_support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[non_exhaustive]
pub enum ScaleMode {
    #[default]
    /// Empirical sigma: EWMA of squared prediction errors (default).
    Empirical,
    /// Tree-chain: a second boosting chain learns log(sigma) from features.
    TreeChain,
}

/// Which drift detector to instantiate for each boosting step.
///
/// Each variant stores the detector's configuration parameters so that fresh
/// instances can be created on demand (e.g. when replacing a drifted tree).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "_serde_support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[non_exhaustive]
pub enum DriftDetectorType {
    /// Page-Hinkley Test (default detector).
    PageHinkley {
        /// Significance level controlling sensitivity vs. false-alarm rate.
        delta: f64,
        /// Minimum cumulative sum threshold before drift is signalled.
        lambda: f64,
    },
    /// ADWIN adaptive windowing detector.
    Adwin {
        /// Confidence parameter (smaller = more sensitive).
        delta: f64,
    },
    /// Drift Detection Method (DDM) Welford detector.
    Ddm {
        /// Standard-deviation multiplier for the warning level.
        warning_level: f64,
        /// Standard-deviation multiplier for the drift level.
        drift_level: f64,
        /// Minimum samples required before drift signalling is active.
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

/// Configuration for the SGBT ensemble.
///
/// All numeric parameters are validated at build time via [`SGBTConfigBuilder`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "_serde_support",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct SGBTConfig {
    /// Number of sequential boosting steps (trees) in the ensemble.
    pub n_steps: usize,
    /// Shrinkage applied to each tree's contribution.
    pub learning_rate: f64,
    /// Fraction of features sampled per split candidate evaluation.
    pub feature_subsample_rate: f64,
    /// Maximum tree depth (split decisions per root-to-leaf path).
    pub max_depth: usize,
    /// Number of histogram bins per feature for split evaluation.
    pub n_bins: usize,
    /// L2 regularization on leaf values (lambda in XGBoost objective).
    pub lambda: f64,
    /// Minimum gain required to accept a split (gamma in XGBoost).
    pub gamma: f64,
    /// Hoeffding bound grace period: minimum samples before any split is considered.
    pub grace_period: usize,
    /// Hoeffding bound confidence parameter (smaller = more conservative splits).
    pub delta: f64,
    /// Drift detector configuration used for each boosting step.
    pub drift_detector: DriftDetectorType,
    /// Which SGBT algorithm variant to use.
    pub variant: SGBTVariant,
    /// Random seed for feature subsampling and tie-breaking.
    pub seed: u64,
    /// Number of target samples to buffer before fixing the base prediction.
    pub initial_target_count: usize,

    /// Leaf value exponential half-life (samples). `None` disables decay.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub leaf_half_life: Option<usize>,

    /// Maximum training samples per tree before the tree is replaced. `None` = unlimited.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub max_tree_samples: Option<u64>,

    /// Adaptive max-tree-samples: `(warmup_samples, percentile)`. Derives the
    /// threshold from the empirical sample distribution rather than a fixed value.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub adaptive_mts: Option<(u64, f64)>,

    /// Floor on the adaptive MTS threshold (prevents degenerate collapses).
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub adaptive_mts_floor: f64,

    /// Proactive pruning interval in samples. `None` disables proactive pruning.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub proactive_prune_interval: Option<u64>,

    /// Split re-evaluation interval (samples). `None` disables periodic re-evaluation.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub split_reeval_interval: Option<usize>,

    /// Human-readable names for each feature column (used in diagnostics/explainability).
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub feature_names: Option<Vec<String>>,

    /// Per-feature type hints (continuous vs. categorical) for the binning strategy.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub feature_types: Option<Vec<FeatureType>>,

    /// Gradient clipping: clip to `sigma * gradient_clip_sigma`. `None` disables.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub gradient_clip_sigma: Option<f64>,

    /// Per-feature monotonicity constraints: `1` = increasing, `-1` = decreasing, `0` = none.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub monotone_constraints: Option<Vec<i8>>,

    /// Quality pruning significance level (alpha). `None` disables quality pruning.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub quality_prune_alpha: Option<f64>,

    /// Minimum EWMA contribution magnitude for a step to survive quality pruning.
    #[cfg_attr(
        feature = "_serde_support",
        serde(default = "default_quality_prune_threshold")
    )]
    pub quality_prune_threshold: f64,

    /// Consecutive low-contribution rounds before a step is pruned.
    #[cfg_attr(
        feature = "_serde_support",
        serde(default = "default_quality_prune_patience")
    )]
    pub quality_prune_patience: u64,

    /// Error-weighted sample importance EWMA alpha. `None` disables weighting.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub error_weight_alpha: Option<f64>,

    /// Whether to modulate the learning rate by the model's estimated uncertainty.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub uncertainty_modulated_lr: bool,

    /// Strategy for computing the scale (uncertainty) head in distributional mode.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub scale_mode: ScaleMode,

    /// EWMA smoothing factor for the empirical sigma estimate.
    #[cfg_attr(
        feature = "_serde_support",
        serde(default = "default_empirical_sigma_alpha")
    )]
    pub empirical_sigma_alpha: f64,

    /// Maximum absolute leaf output value (clamp). `None` = no clamp.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub max_leaf_output: Option<f64>,

    /// Adaptive leaf output bound derived from the rolling leaf magnitude. `None` disables.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub adaptive_leaf_bound: Option<f64>,

    /// Adaptive depth fractional limit derived from leaf sample counts. `None` disables.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub adaptive_depth: Option<f64>,

    /// Minimum hessian sum required to accept a split. `None` = no minimum.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub min_hessian_sum: Option<f64>,

    /// Huber loss delta override (used when loss is `Huber`). `None` uses the default.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub huber_k: Option<f64>,

    /// Shadow warmup: alternate trees train for this many samples before replacing. `None` = immediate.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub shadow_warmup: Option<usize>,

    /// Leaf model type (constant, linear, MLP). Default is constant value.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub leaf_model_type: LeafModelType,

    /// Interval (samples) between packed-node cache refreshes for fast inference.
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub packed_refresh_interval: u64,

    /// Override for the Hoeffding bound range R. `None` uses the default (1.0).
    #[cfg_attr(feature = "_serde_support", serde(default))]
    pub hoeffding_r: Option<f64>,
}

#[cfg(feature = "_serde_support")]
fn default_empirical_sigma_alpha() -> f64 {
    0.01
}

#[cfg(feature = "_serde_support")]
fn default_quality_prune_threshold() -> f64 {
    1e-6
}

#[cfg(feature = "_serde_support")]
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
            adaptive_mts: None,
            adaptive_mts_floor: 0.0,
            proactive_prune_interval: None,
            split_reeval_interval: None,
            feature_names: None,
            feature_types: None,
            gradient_clip_sigma: None,
            monotone_constraints: None,
            quality_prune_alpha: None,
            quality_prune_threshold: 1e-6,
            quality_prune_patience: 500,
            error_weight_alpha: None,
            uncertainty_modulated_lr: false,
            scale_mode: ScaleMode::default(),
            empirical_sigma_alpha: 0.01,
            max_leaf_output: None,
            adaptive_leaf_bound: None,
            adaptive_depth: None,
            min_hessian_sum: None,
            huber_k: None,
            shadow_warmup: None,
            leaf_model_type: LeafModelType::default(),
            packed_refresh_interval: 0,
            hoeffding_r: None,
        }
    }
}

impl SGBTConfig {
    /// Create a new builder for [`SGBTConfig`].
    pub fn builder() -> SGBTConfigBuilder {
        SGBTConfigBuilder::default()
    }
}

/// Builder for [`SGBTConfig`] with validation on [`build()`](Self::build).
#[derive(Debug, Clone, Default)]
pub struct SGBTConfigBuilder {
    config: SGBTConfig,
}

impl SGBTConfigBuilder {
    /// Set the number of boosting steps (trees).
    pub fn n_steps(mut self, n: usize) -> Self {
        self.config.n_steps = n;
        self
    }

    /// Set the learning rate (shrinkage).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set the feature subsampling rate (`0.0..=1.0`).
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

    /// Set the L2 regularization coefficient on leaf values.
    pub fn lambda(mut self, l: f64) -> Self {
        self.config.lambda = l;
        self
    }

    /// Set the minimum split gain (gamma).
    pub fn gamma(mut self, g: f64) -> Self {
        self.config.gamma = g;
        self
    }

    /// Set the Hoeffding bound grace period (minimum samples before splits).
    pub fn grace_period(mut self, gp: usize) -> Self {
        self.config.grace_period = gp;
        self
    }

    /// Set the Hoeffding bound confidence parameter.
    pub fn delta(mut self, d: f64) -> Self {
        self.config.delta = d;
        self
    }

    /// Set the drift detector configuration.
    pub fn drift_detector(mut self, dt: DriftDetectorType) -> Self {
        self.config.drift_detector = dt;
        self
    }

    /// Set the algorithm variant.
    pub fn variant(mut self, v: SGBTVariant) -> Self {
        self.config.variant = v;
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Set the number of targets to buffer before fixing the base prediction.
    pub fn initial_target_count(mut self, count: usize) -> Self {
        self.config.initial_target_count = count;
        self
    }

    /// Set the leaf value exponential half-life in samples.
    pub fn leaf_half_life(mut self, n: usize) -> Self {
        self.config.leaf_half_life = Some(n);
        self
    }

    /// Set the maximum samples per tree before replacement.
    pub fn max_tree_samples(mut self, n: u64) -> Self {
        self.config.max_tree_samples = Some(n);
        self
    }

    /// Set adaptive MTS parameters: warmup samples and percentile.
    pub fn adaptive_mts(mut self, base_mts: u64, k: f64) -> Self {
        self.config.adaptive_mts = Some((base_mts, k));
        self
    }

    /// Set the floor on the adaptive MTS threshold.
    pub fn adaptive_mts_floor(mut self, fraction: f64) -> Self {
        self.config.adaptive_mts_floor = fraction;
        self
    }

    /// Set the proactive pruning interval in samples.
    pub fn proactive_prune_interval(mut self, interval: u64) -> Self {
        self.config.proactive_prune_interval = Some(interval);
        self
    }

    /// Set the split re-evaluation interval in samples.
    pub fn split_reeval_interval(mut self, n: usize) -> Self {
        self.config.split_reeval_interval = Some(n);
        self
    }

    /// Set human-readable feature names (used in diagnostics).
    pub fn feature_names(mut self, names: Vec<String>) -> Self {
        self.config.feature_names = Some(names);
        self
    }

    /// Set per-feature type hints (continuous vs. categorical).
    pub fn feature_types(mut self, types: Vec<FeatureType>) -> Self {
        self.config.feature_types = Some(types);
        self
    }

    /// Set gradient clipping sigma multiplier.
    pub fn gradient_clip_sigma(mut self, sigma: f64) -> Self {
        self.config.gradient_clip_sigma = Some(sigma);
        self
    }

    /// Set per-feature monotonicity constraints (`1`, `-1`, or `0`).
    pub fn monotone_constraints(mut self, constraints: Vec<i8>) -> Self {
        self.config.monotone_constraints = Some(constraints);
        self
    }

    /// Set quality pruning significance level (alpha).
    pub fn quality_prune_alpha(mut self, alpha: f64) -> Self {
        self.config.quality_prune_alpha = Some(alpha);
        self
    }

    /// Set minimum EWMA contribution for quality pruning survival.
    pub fn quality_prune_threshold(mut self, threshold: f64) -> Self {
        self.config.quality_prune_threshold = threshold;
        self
    }

    /// Set consecutive low-contribution patience before quality pruning.
    pub fn quality_prune_patience(mut self, patience: u64) -> Self {
        self.config.quality_prune_patience = patience;
        self
    }

    /// Set error-weighted sample importance EWMA alpha.
    pub fn error_weight_alpha(mut self, alpha: f64) -> Self {
        self.config.error_weight_alpha = Some(alpha);
        self
    }

    /// Enable or disable uncertainty-modulated learning rate.
    pub fn uncertainty_modulated_lr(mut self, enabled: bool) -> Self {
        self.config.uncertainty_modulated_lr = enabled;
        self
    }

    /// Set the scale mode for distributional SGBT.
    pub fn scale_mode(mut self, mode: ScaleMode) -> Self {
        self.config.scale_mode = mode;
        self
    }

    /// Set the EWMA alpha for empirical sigma estimation.
    pub fn empirical_sigma_alpha(mut self, alpha: f64) -> Self {
        self.config.empirical_sigma_alpha = alpha;
        self
    }

    /// Set the maximum absolute leaf output (clamp).
    pub fn max_leaf_output(mut self, max: f64) -> Self {
        self.config.max_leaf_output = Some(max);
        self
    }

    /// Set the adaptive leaf output bound multiplier.
    pub fn adaptive_leaf_bound(mut self, k: f64) -> Self {
        self.config.adaptive_leaf_bound = Some(k);
        self
    }

    /// Set the adaptive depth fractional limit.
    pub fn adaptive_depth(mut self, factor: f64) -> Self {
        self.config.adaptive_depth = Some(factor);
        self
    }

    /// Set the minimum hessian sum required to accept a split.
    pub fn min_hessian_sum(mut self, min_h: f64) -> Self {
        self.config.min_hessian_sum = Some(min_h);
        self
    }

    /// Set the Huber loss delta override.
    pub fn huber_k(mut self, k: f64) -> Self {
        self.config.huber_k = Some(k);
        self
    }

    /// Set the shadow warmup samples before tree replacement.
    pub fn shadow_warmup(mut self, warmup: usize) -> Self {
        self.config.shadow_warmup = Some(warmup);
        self
    }

    /// Set the leaf model type (constant, linear, MLP).
    pub fn leaf_model_type(mut self, lmt: LeafModelType) -> Self {
        self.config.leaf_model_type = lmt;
        self
    }

    /// Set the packed-node cache refresh interval in samples.
    pub fn packed_refresh_interval(mut self, interval: u64) -> Self {
        self.config.packed_refresh_interval = interval;
        self
    }

    /// Override the Hoeffding bound range R (default 1.0).
    pub fn hoeffding_r(mut self, r: f64) -> Self {
        self.config.hoeffding_r = Some(r);
        self
    }

    /// Validate and build the configuration.
    pub fn build(self) -> Result<SGBTConfig> {
        validation::validate_and_build(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn default_config_valid() {
        let cfg = SGBTConfig::default();
        assert_eq!(cfg.n_steps, 100);
        assert_eq!(cfg.learning_rate, 0.0125);
    }

    #[test]
    fn builder_basic() {
        let cfg = SGBTConfig::builder()
            .n_steps(50)
            .learning_rate(0.05)
            .build()
            .unwrap();
        assert_eq!(cfg.n_steps, 50);
        assert_eq!(cfg.learning_rate, 0.05);
    }

    #[test]
    fn validation_rejects_zero_n_steps() {
        let result = SGBTConfig::builder().n_steps(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn validation_accepts_valid_learning_rate() {
        let result = SGBTConfig::builder().learning_rate(0.1).build();
        assert!(result.is_ok());
    }

    #[test]
    fn validation_rejects_zero_learning_rate() {
        let result = SGBTConfig::builder().learning_rate(0.0).build();
        assert!(result.is_err());
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
    fn drift_detector_type_create() {
        let dt = DriftDetectorType::PageHinkley {
            delta: 0.005,
            lambda: 50.0,
        };
        let mut detector = dt.create();
        for _ in 0..500 {
            detector.update(1.0);
        }
        let mut drifted = false;
        for _ in 0..500 {
            if detector.update(10.0) == crate::drift::DriftSignal::Drift {
                drifted = true;
                break;
            }
        }
        assert!(drifted);
    }

    #[test]
    fn boundary_n_bins_two_accepted() {
        let result = SGBTConfig::builder().n_bins(2).build();
        assert!(result.is_ok());
    }

    #[test]
    fn boundary_grace_period_one_accepted() {
        let result = SGBTConfig::builder().grace_period(1).build();
        assert!(result.is_ok());
    }

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

    #[test]
    fn feature_names_rejects_duplicates() {
        let result = SGBTConfig::builder()
            .feature_names(vec!["price".into(), "volume".into(), "price".into()])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn feature_names_empty_vec_accepted() {
        let cfg = SGBTConfig::builder().feature_names(vec![]).build().unwrap();
        assert!(cfg.feature_names.unwrap().is_empty());
    }

    #[test]
    fn builder_adaptive_leaf_bound() {
        let cfg = SGBTConfig::builder()
            .adaptive_leaf_bound(3.0)
            .build()
            .unwrap();
        assert_eq!(cfg.adaptive_leaf_bound, Some(3.0));
    }

    #[test]
    fn validation_rejects_zero_adaptive_leaf_bound() {
        let result = SGBTConfig::builder().adaptive_leaf_bound(0.0).build();
        assert!(result.is_err());
    }
}
