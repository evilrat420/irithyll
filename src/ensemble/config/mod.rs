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
use crate::error::Result;
use crate::tree::leaf_model::LeafModelType;

// Re-export submodules internally.
mod display;
mod tree_config_helper;
mod validation;

// Re-export key items for internal use only.
pub(crate) use tree_config_helper::build_tree_config;

// ---------------------------------------------------------------------------
// FeatureType -- re-exported from irithyll-core
// ---------------------------------------------------------------------------

pub use irithyll_core::feature::FeatureType;

// ---------------------------------------------------------------------------
// ScaleMode
// ---------------------------------------------------------------------------

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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ScaleMode {
    /// EWMA of squared prediction errors — always calibrated, O(1).
    #[default]
    Empirical,
    /// Full Hoeffding-tree ensemble for feature-conditional log(σ) prediction.
    TreeChain,
}

// ---------------------------------------------------------------------------
// DriftDetectorType
// ---------------------------------------------------------------------------

/// Which drift detector to instantiate for each boosting step.
///
/// Each variant stores the detector's configuration parameters so that fresh
/// instances can be created on demand (e.g. when replacing a drifted tree).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
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
/// | `adaptive_mts`           | None (disabled)      |
/// | `proactive_prune_interval` | None (disabled)    |
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
    /// `None` (default) disables decay -- traditional monotonic accumulation.
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
    /// `None` (default) disables re-evaluation -- max-depth leaves are permanent.
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
    /// This prevents "dead wood" -- trees from a past regime that no longer
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
    /// gradient level -- learning capacity focuses on hard/novel patterns,
    /// enabling faster adaptation to regime changes.
    ///
    /// `None` (default) disables error weighting.
    /// Suggested value: 0.01.
    #[serde(default)]
    pub error_weight_alpha: Option<f64>,

    /// Enable σ-modulated learning rate for [`DistributionalSGBT`](super::distributional::DistributionalSGBT).
    ///
    /// When `true`, the **location** (μ) ensemble's learning rate is scaled by
    /// `sigma_ratio = current_sigma / rolling_sigma_mean`, where `rolling_sigma_mean`
    /// is an EWMA of the model's predicted σ (alpha = 0.001).
    ///
    /// This means the model learns μ **faster** when σ is elevated (high uncertainty)
    /// and **slower** when σ is low (confident regime). The scale (σ) ensemble always
    /// trains at the unmodulated base rate to prevent positive feedback loops.
    ///
    /// Default: `false`.
    #[serde(default)]
    pub uncertainty_modulated_lr: bool,

    /// How the scale (σ) is estimated in [`DistributionalSGBT`](super::distributional::DistributionalSGBT).
    ///
    /// - [`Empirical`](ScaleMode::Empirical) (default): EWMA of squared prediction
    ///   errors.  `σ = sqrt(ewma_sq_err)`.  Always calibrated, zero tuning, O(1).
    /// - [`TreeChain`](ScaleMode::TreeChain): full dual-chain NGBoost with a
    ///   separate tree ensemble predicting log(σ) from features.
    ///
    /// For σ-modulated learning (`uncertainty_modulated_lr = true`), `Empirical`
    /// is strongly recommended — scale tree gradients are inherently weak and
    /// the trees often fail to split.
    #[serde(default)]
    pub scale_mode: ScaleMode,

    /// EWMA smoothing factor for empirical σ estimation.
    ///
    /// Controls the adaptation speed of `σ = sqrt(ewma_sq_err)` when
    /// [`scale_mode`](Self::scale_mode) is [`Empirical`](ScaleMode::Empirical).
    /// Higher values react faster to regime changes but are noisier.
    ///
    /// Default: `0.01` (~100-sample effective window).
    #[serde(default = "default_empirical_sigma_alpha")]
    pub empirical_sigma_alpha: f64,

    /// Maximum absolute leaf output value.
    ///
    /// When `Some(max)`, leaf predictions are clamped to `[-max, max]`.
    /// Bounds leaf weight magnitude to maintain prediction stability
    /// in feedback loops. `None` (default) means no clamping.
    #[serde(default)]
    pub max_leaf_output: Option<f64>,

    /// Per-leaf adaptive output bound (sigma multiplier).
    ///
    /// When `Some(k)`, each leaf tracks an EWMA of its own output weight and
    /// clamps predictions to `|output_mean| + k * output_std`. The EWMA uses
    /// `leaf_decay_alpha` when `leaf_half_life` is set, otherwise Welford online.
    ///
    /// This is strictly superior to `max_leaf_output` for streaming — the bound
    /// is per-leaf, self-calibrating, and regime-synchronized. A leaf that usually
    /// outputs 0.3 can't suddenly output 2.9 just because it fits in the global clamp.
    ///
    /// Typical value: 3.0 (3-sigma bound).
    /// `None` (default) disables adaptive bounds (falls back to `max_leaf_output`).
    #[serde(default)]
    pub adaptive_leaf_bound: Option<f64>,

    /// Per-split information criterion (Lunde-Kleppe-Skaug 2020).
    ///
    /// When `Some(cir_factor)`, replaces `max_depth` with a per-split
    /// generalization test. Each candidate split must have
    /// `gain > cir_factor * sigma^2_g / n * n_features`.
    /// `max_depth * 2` becomes a hard safety ceiling.
    ///
    /// Typical: 7.5 (<=10 features), 9.0 (<=50), 11.0 (<=200).
    /// `None` (default) uses static `max_depth` only.
    #[serde(default)]
    pub adaptive_depth: Option<f64>,

    /// Minimum hessian sum before a leaf produces non-zero output.
    ///
    /// When `Some(min_h)`, leaves with `hess_sum < min_h` return 0.0.
    /// Prevents post-replacement spikes from fresh leaves with insufficient
    /// samples. `None` (default) means all leaves contribute immediately.
    #[serde(default)]
    pub min_hessian_sum: Option<f64>,

    /// Huber loss delta multiplier for [`DistributionalSGBT`](super::distributional::DistributionalSGBT).
    ///
    /// When `Some(k)`, the distributional location gradient uses Huber loss
    /// with adaptive `delta = k * empirical_sigma`. This bounds gradients by
    /// construction. Standard value: `1.345` (95% efficiency at Gaussian).
    /// `None` (default) uses squared loss.
    #[serde(default)]
    pub huber_k: Option<f64>,

    /// Shadow warmup for graduated tree handoff.
    ///
    /// When `Some(n)`, an always-on shadow (alternate) tree is spawned immediately
    /// alongside every active tree. The shadow trains on the same gradient stream
    /// but does not contribute to predictions until it has seen `n` samples.
    ///
    /// As the active tree ages past 80% of `max_tree_samples`, its prediction
    /// weight linearly decays to 0 at 120%. The shadow's weight ramps from 0 to 1
    /// over `n` samples after warmup. When the active weight reaches 0, the shadow
    /// is promoted and a new shadow is spawned — no cold-start prediction dip.
    ///
    /// Requires `max_tree_samples` to be set for time-based graduated handoff.
    /// Drift-based replacement still uses hard swap (shadow is already warm).
    ///
    /// `None` (default) disables graduated handoff — uses traditional hard swap.
    #[serde(default)]
    pub shadow_warmup: Option<usize>,

    /// Leaf prediction model type.
    ///
    /// Controls how each leaf computes its prediction:
    /// - [`ClosedForm`](LeafModelType::ClosedForm) (default): constant leaf weight.
    /// - [`Linear`](LeafModelType::Linear): per-leaf online ridge regression with
    ///   AdaGrad optimization. Optional `decay` for concept drift. Recommended for
    ///   low-depth trees (depth 2--4).
    /// - [`MLP`](LeafModelType::MLP): per-leaf single-hidden-layer neural network.
    ///   Optional `decay` for concept drift.
    /// - [`Adaptive`](LeafModelType::Adaptive): starts as closed-form, auto-promotes
    ///   when the Hoeffding bound confirms a more complex model is better.
    ///
    /// Default: [`ClosedForm`](LeafModelType::ClosedForm).
    #[serde(default)]
    pub leaf_model_type: LeafModelType,

    /// Packed cache refresh interval for [`DistributionalSGBT`](super::distributional::DistributionalSGBT).
    ///
    /// When non-zero, the distributional model maintains a packed f32 cache of
    /// its location ensemble that is re-exported every `packed_refresh_interval`
    /// training samples. Predictions use the cache for O(1)-per-tree inference
    /// via contiguous memory traversal, falling back to full tree traversal when
    /// the cache is absent or produces non-finite results.
    ///
    /// `0` (default) disables the packed cache.
    #[serde(default)]
    pub packed_refresh_interval: u64,

    /// Sigma-modulated adaptive tree replacement speed.
    ///
    /// When `Some((base_mts, k))`, trees are replaced faster during high
    /// uncertainty and slower during stability. The effective max-tree-samples
    /// is computed as `base_mts / (1 + k * sigma_ratio)`, where `sigma_ratio`
    /// is derived from the model's contribution variance (base SGBT) or
    /// honest_sigma (distributional SGBT).
    ///
    /// Overrides `max_tree_samples` when set. `k` controls sensitivity
    /// (typical: 1.0).
    ///
    /// `None` (default) disables adaptive replacement speed.
    #[serde(default)]
    pub adaptive_mts: Option<(u64, f64)>,

    /// Minimum effective MTS as a fraction of `base_mts`.
    ///
    /// When adaptive MTS is active, the effective tree lifetime can shrink
    /// aggressively under high uncertainty. This floor prevents runaway
    /// replacement by clamping `effective_mts >= base_mts * fraction`.
    ///
    /// The hard floor of 100 samples still applies beneath this.
    ///
    /// Default: `0.0` (only the hard floor of 100 applies).
    #[serde(default)]
    pub adaptive_mts_floor: f64,

    /// Proactive pruning interval.
    ///
    /// Every `interval` samples, the worst-contributing tree is identified
    /// and replaced. This enables continuous ensemble hygiene beyond
    /// drift-detector-triggered replacement.
    ///
    /// Enables contribution tracking automatically.
    ///
    /// `None` (default) disables proactive pruning.
    #[serde(default)]
    pub proactive_prune_interval: Option<u64>,

    /// Use accuracy-based pruning instead of variance-based.
    ///
    /// When `true`, proactive pruning tracks each tree's signed contribution
    /// alignment with the ensemble residual. Trees that consistently push
    /// predictions in the wrong direction get pruned first. Young trees
    /// (below grace_period samples) are protected from pruning.
    ///
    /// When `false` (default), the original variance-based pruning selects
    /// the tree with the smallest prediction variance for replacement.
    ///
    /// Only relevant when `proactive_prune_interval` is `Some`.
    #[serde(default)]
    pub accuracy_based_pruning: bool,

    /// Half-life (in samples) for the contribution accuracy EWMA.
    ///
    /// Controls how many recent samples influence the pruning decision.
    /// `None` (default): auto-derived from tree lifetime — uses `adaptive_mts`
    /// base if set, else `max_tree_samples` if set, else `grace_period`.
    /// `Some(n)`: explicit half-life for fine-grained control.
    #[serde(default)]
    pub prune_half_life: Option<usize>,

    /// Hoeffding bound range parameter (R) for split decisions.
    ///
    /// The Hoeffding bound is `ε = sqrt(R² · ln(1/δ) / (2n))`.
    /// R is an upper bound on the range of the gain function.
    ///
    /// The default R=1.0 is conservative. For targets with known variance σ²,
    /// setting R=sqrt(σ²) gives data-proportional split thresholds:
    /// tighter bounds when targets are low-variance, looser when high-variance.
    ///
    /// `None` (default) uses R=1.0 (conservative, safe for unknown target scale).
    #[serde(default)]
    pub hoeffding_r: Option<f64>,
}

fn default_empirical_sigma_alpha() -> f64 {
    0.01
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
            adaptive_mts: None,
            adaptive_mts_floor: 0.0,
            proactive_prune_interval: None,
            accuracy_based_pruning: false,
            prune_half_life: None,
            hoeffding_r: None,
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

    /// Enable σ-modulated learning rate for distributional models.
    ///
    /// Scales the location (μ) learning rate by `current_sigma / rolling_sigma_mean`,
    /// so the model adapts faster during high-uncertainty regimes and conserves
    /// during stable periods. Only affects [`DistributionalSGBT`](super::distributional::DistributionalSGBT).
    ///
    /// By default uses empirical σ (EWMA of squared errors).  Set
    /// [`scale_mode(ScaleMode::TreeChain)`](Self::scale_mode) for feature-conditional σ.
    pub fn uncertainty_modulated_lr(mut self, enabled: bool) -> Self {
        self.config.uncertainty_modulated_lr = enabled;
        self
    }

    /// Set the scale estimation mode for [`DistributionalSGBT`](super::distributional::DistributionalSGBT).
    ///
    /// - [`Empirical`](ScaleMode::Empirical): EWMA of squared prediction errors (default, recommended).
    /// - [`TreeChain`](ScaleMode::TreeChain): dual-chain NGBoost with scale tree ensemble.
    pub fn scale_mode(mut self, mode: ScaleMode) -> Self {
        self.config.scale_mode = mode;
        self
    }

    /// EWMA alpha for empirical σ. Controls adaptation speed. Default `0.01`.
    ///
    /// Only used when `scale_mode` is [`Empirical`](ScaleMode::Empirical).
    pub fn empirical_sigma_alpha(mut self, alpha: f64) -> Self {
        self.config.empirical_sigma_alpha = alpha;
        self
    }

    /// Set the maximum absolute leaf output value.
    ///
    /// Clamps leaf predictions to `[-max, max]`, breaking feedback loops
    /// that cause unbounded prediction growth.
    pub fn max_leaf_output(mut self, max: f64) -> Self {
        self.config.max_leaf_output = Some(max);
        self
    }

    /// Set per-leaf adaptive output bound (sigma multiplier).
    ///
    /// Each leaf tracks EWMA of its own output weight and clamps to
    /// `|output_mean| + k * output_std`. Self-calibrating per-leaf.
    /// Recommended: use with `leaf_half_life` for streaming scenarios.
    pub fn adaptive_leaf_bound(mut self, k: f64) -> Self {
        self.config.adaptive_leaf_bound = Some(k);
        self
    }

    /// Set the per-split information criterion factor (Lunde-Kleppe-Skaug 2020).
    ///
    /// Replaces static `max_depth` with a per-split generalization test.
    /// Typical: 7.5 (<=10 features), 9.0 (<=50), 11.0 (<=200).
    pub fn adaptive_depth(mut self, factor: f64) -> Self {
        self.config.adaptive_depth = Some(factor);
        self
    }

    /// Set the minimum hessian sum for leaf output.
    ///
    /// Fresh leaves with `hess_sum < min_h` return 0.0, preventing
    /// post-replacement spikes.
    pub fn min_hessian_sum(mut self, min_h: f64) -> Self {
        self.config.min_hessian_sum = Some(min_h);
        self
    }

    /// Set the Huber loss delta multiplier for [`DistributionalSGBT`](super::distributional::DistributionalSGBT).
    ///
    /// When set, location gradients use Huber loss with adaptive
    /// `delta = k * empirical_sigma`. Standard value: `1.345` (95% Gaussian efficiency).
    pub fn huber_k(mut self, k: f64) -> Self {
        self.config.huber_k = Some(k);
        self
    }

    /// Enable graduated tree handoff with the given shadow warmup samples.
    ///
    /// Spawns an always-on shadow tree that trains alongside the active tree.
    /// After `warmup` samples, the shadow begins contributing to predictions
    /// via graduated blending. Eliminates prediction dips during tree replacement.
    pub fn shadow_warmup(mut self, warmup: usize) -> Self {
        self.config.shadow_warmup = Some(warmup);
        self
    }

    /// Set the leaf prediction model type.
    ///
    /// [`LeafModelType::Linear`] is recommended for low-depth configurations
    /// (depth 2--4) where per-leaf linear models reduce approximation error.
    ///
    /// [`LeafModelType::Adaptive`] automatically selects between closed-form and
    /// a trainable model per leaf, using the Hoeffding bound for promotion.
    pub fn leaf_model_type(mut self, lmt: LeafModelType) -> Self {
        self.config.leaf_model_type = lmt;
        self
    }

    /// Set the packed cache refresh interval for distributional models.
    ///
    /// When non-zero, [`DistributionalSGBT`](super::distributional::DistributionalSGBT)
    /// maintains a packed f32 cache refreshed every `interval` training samples.
    /// `0` (default) disables the cache.
    pub fn packed_refresh_interval(mut self, interval: u64) -> Self {
        self.config.packed_refresh_interval = interval;
        self
    }

    /// Set sigma-modulated tree replacement speed.
    ///
    /// `base_mts` is the tree lifetime under normal conditions.
    /// `k` controls sensitivity (typical: 1.0).
    ///
    /// When set, trees are replaced faster during high uncertainty
    /// and slower during stability. Overrides `max_tree_samples`.
    pub fn adaptive_mts(mut self, base_mts: u64, k: f64) -> Self {
        self.config.adaptive_mts = Some((base_mts, k));
        self
    }

    /// Set a minimum effective MTS as a fraction of `base_mts`.
    ///
    /// Prevents adaptive MTS from shrinking tree lifetime below
    /// `base_mts * fraction`. For example, `0.25` ensures effective MTS
    /// never drops below 25% of `base_mts`.
    ///
    /// Only meaningful when `adaptive_mts` is also set.
    pub fn adaptive_mts_floor(mut self, fraction: f64) -> Self {
        self.config.adaptive_mts_floor = fraction;
        self
    }

    /// Set proactive pruning interval.
    ///
    /// Every `interval` samples, the worst-contributing tree is replaced.
    /// Enables contribution tracking automatically.
    pub fn proactive_prune_interval(mut self, interval: u64) -> Self {
        self.config.proactive_prune_interval = Some(interval);
        self
    }

    /// Enable accuracy-based proactive pruning.
    ///
    /// When enabled, trees are pruned based on signed contribution alignment
    /// with the ensemble residual instead of prediction variance.
    pub fn accuracy_based_pruning(mut self, enabled: bool) -> Self {
        self.config.accuracy_based_pruning = enabled;
        self
    }

    /// Set the half-life for the contribution accuracy EWMA used by proactive pruning.
    ///
    /// Overrides the automatic derivation (adaptive_mts base → max_tree_samples → grace_period).
    pub fn prune_half_life(mut self, n: usize) -> Self {
        self.config.prune_half_life = Some(n);
        self
    }

    /// Set the Hoeffding bound range parameter (R) for split decisions.
    ///
    /// `ε = sqrt(R² · ln(1/δ) / (2n))`. Set to `sqrt(target_variance)` for
    /// data-proportional split thresholds. Must be finite and positive.
    ///
    /// Default: `None` (uses R=1.0, conservative for unknown target scale).
    pub fn hoeffding_r(mut self, r: f64) -> Self {
        self.config.hoeffding_r = Some(r);
        self
    }

    /// Validate and build the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidConfig`](crate::IrithyllError::InvalidConfig) with a structured
    /// error if any parameter is out of its valid range.
    pub fn build(self) -> Result<SGBTConfig> {
        validation::validate_and_build(self.config)
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
            .delta(1e-5)
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
        assert!((cfg.delta - 1e-5).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // 3. Validation: n_steps must be > 0
    // ------------------------------------------------------------------
    #[test]
    fn validation_n_steps_zero() {
        let cfg = SGBTConfig::builder().n_steps(0).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 4. Validation: learning_rate in (0, 1]
    // ------------------------------------------------------------------
    #[test]
    fn validation_learning_rate_zero() {
        let cfg = SGBTConfig::builder().learning_rate(0.0).build();
        assert!(cfg.is_err());
    }

    #[test]
    fn validation_learning_rate_too_high() {
        let cfg = SGBTConfig::builder().learning_rate(1.1).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 5. Validation: feature_subsample_rate in (0, 1]
    // ------------------------------------------------------------------
    #[test]
    fn validation_feature_subsample_rate_zero() {
        let cfg = SGBTConfig::builder().feature_subsample_rate(0.0).build();
        assert!(cfg.is_err());
    }

    #[test]
    fn validation_feature_subsample_rate_too_high() {
        let cfg = SGBTConfig::builder().feature_subsample_rate(1.1).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 6. Validation: max_depth > 0
    // ------------------------------------------------------------------
    #[test]
    fn validation_max_depth_zero() {
        let cfg = SGBTConfig::builder().max_depth(0).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 7. Validation: n_bins >= 2
    // ------------------------------------------------------------------
    #[test]
    fn validation_n_bins_too_small() {
        let cfg = SGBTConfig::builder().n_bins(1).build();
        assert!(cfg.is_err());
    }

    #[test]
    fn validation_n_bins_two_ok() {
        let cfg = SGBTConfig::builder().n_bins(2).build();
        assert!(cfg.is_ok());
    }

    // ------------------------------------------------------------------
    // 8. Validation: lambda >= 0
    // ------------------------------------------------------------------
    #[test]
    fn validation_lambda_negative() {
        let cfg = SGBTConfig::builder().lambda(-0.1).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 9. Validation: gamma >= 0
    // ------------------------------------------------------------------
    #[test]
    fn validation_gamma_negative() {
        let cfg = SGBTConfig::builder().gamma(-0.1).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 10. Validation: grace_period > 0
    // ------------------------------------------------------------------
    #[test]
    fn validation_grace_period_zero() {
        let cfg = SGBTConfig::builder().grace_period(0).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 11. Validation: delta in (0, 1)
    // ------------------------------------------------------------------
    #[test]
    fn validation_delta_zero() {
        let cfg = SGBTConfig::builder().delta(0.0).build();
        assert!(cfg.is_err());
    }

    #[test]
    fn validation_delta_one() {
        let cfg = SGBTConfig::builder().delta(1.0).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 12. Validation: initial_target_count > 0
    // ------------------------------------------------------------------
    #[test]
    fn validation_initial_target_count_zero() {
        let cfg = SGBTConfig::builder().initial_target_count(0).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 13. Feature names must be unique
    // ------------------------------------------------------------------
    #[test]
    fn validation_duplicate_feature_names() {
        let cfg = SGBTConfig::builder()
            .feature_names(vec!["a".into(), "a".into()])
            .build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 14. Feature names and types must match in length
    // ------------------------------------------------------------------
    #[test]
    fn validation_feature_names_types_mismatch() {
        let cfg = SGBTConfig::builder()
            .feature_names(vec!["a".into(), "b".into()])
            .feature_types(vec![FeatureType::Continuous])
            .build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 15. Monotone constraints must be -1, 0, or 1
    // ------------------------------------------------------------------
    #[test]
    fn validation_bad_monotone_constraint() {
        let cfg = SGBTConfig::builder().monotone_constraints(vec![2]).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 16. Quality prune alpha in (0, 1)
    // ------------------------------------------------------------------
    #[test]
    fn validation_quality_prune_alpha_zero() {
        let cfg = SGBTConfig::builder().quality_prune_alpha(0.0).build();
        assert!(cfg.is_err());
    }

    #[test]
    fn validation_quality_prune_alpha_one() {
        let cfg = SGBTConfig::builder().quality_prune_alpha(1.0).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 17. Error weight alpha in (0, 1)
    // ------------------------------------------------------------------
    #[test]
    fn validation_error_weight_alpha_zero() {
        let cfg = SGBTConfig::builder().error_weight_alpha(0.0).build();
        assert!(cfg.is_err());
    }

    #[test]
    fn validation_error_weight_alpha_one() {
        let cfg = SGBTConfig::builder().error_weight_alpha(1.0).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 18. Empirical sigma alpha in [0.0, 1.0]
    // ------------------------------------------------------------------
    #[test]
    fn validation_empirical_sigma_alpha_too_high() {
        let cfg = SGBTConfig::builder().empirical_sigma_alpha(1.1).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 19. Adaptive MTS floor in [0.0, 1.0]
    // ------------------------------------------------------------------
    #[test]
    fn validation_adaptive_mts_floor_too_high() {
        let cfg = SGBTConfig::builder().adaptive_mts_floor(1.1).build();
        assert!(cfg.is_err());
    }

    // ------------------------------------------------------------------
    // 20. Drift detector validation
    // ------------------------------------------------------------------
    #[test]
    fn validation_drift_detector_pht_bad_delta() {
        let cfg = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::PageHinkley {
                delta: 0.0,
                lambda: 1.0,
            })
            .build();
        assert!(cfg.is_err());
    }

    #[test]
    fn validation_drift_detector_adwin_bad_delta() {
        let cfg = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::Adwin { delta: 1.1 })
            .build();
        assert!(cfg.is_err());
    }

    #[test]
    fn validation_drift_detector_ddm_bad_levels() {
        let cfg = SGBTConfig::builder()
            .drift_detector(DriftDetectorType::Ddm {
                warning_level: 3.0,
                drift_level: 2.0,
                min_instances: 30,
            })
            .build();
        assert!(cfg.is_err());
    }
}
