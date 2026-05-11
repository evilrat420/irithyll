#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
//! Streaming machine learning in Rust.
//!
//! irithyll is a streaming ML library for the case where data arrives in order
//! and never stops. There is no training set. There is no batch loop. Every
//! sample updates the model and is then released -- no buffer, no replay.
//!
//! All models implement [`StreamingLearner`], a two-method contract:
//! `train_one(features, target, weight)` and `predict(features) -> f64`. A
//! `Box<dyn StreamingLearner>` is a fully typed model. Anything that
//! implements the trait slots into a pipeline, an MoE expert, an AutoML
//! candidate, a projection wrapper, or a classification head.
//!
//! # Model Survey
//!
//! **Gradient-boosted trees** -- [`SGBT`] is the flagship: sequential gradient
//! boosting over streaming Hoeffding trees with automatic drift replacement
//! ([Gunasekara et al., 2024](https://doi.org/10.1007/s10994-024-06517-y)).
//! Variants: [`BaggedSGBT`], [`DistributionalSGBT`], [`MulticlassSGBT`],
//! [`QuantileRegressorSGBT`], [`MultiTargetSGBT`], [`AdaptiveRandomForest`],
//! [`ParallelSGBT`] (requires `parallel` feature).
//!
//! **Linear and kernel models** -- [`RecursiveLeastSquares`] with prediction
//! intervals, [`StreamingLinearModel`] with SGD, [`KRLS`] with RBF /
//! polynomial / linear kernels and ALD sparsification,
//! [`LocallyWeightedRegression`], [`MondrianForest`].
//!
//! **Neural streaming architectures** -- [`StreamingMamba`] (selective SSM,
//! BD-LRU block-diagonal variant), [`StreamingTTT`] (test-time training with
//! Titans-style momentum), [`StreamingKAN`] (Kolmogorov-Arnold networks with
//! B-spline basis), [`StreamingsLSTM`] (exponential-gated stabilized LSTM),
//! [`StreamingMGrade`] (minimal recurrent gating with delay convolutions),
//! [`SpikeNet`] (spiking neural network with e-prop), [`EchoStateNetwork`],
//! [`NextGenRC`], [`LogLinearAttention`] (hierarchical Fenwick state,
//! Han Guo et al., ICLR 2026), [`StreamingAttentionModel`] (GLA,
//! DeltaNet, Hawk, RetNet, RWKV-7 variants), [`NeuralMoE`].
//!
//! **AutoML** -- [`AutoTuner`] races model families under champion-challenger
//! promotion using empirical Bernstein bounds (Maurer & Pontil, 2009) as the
//! statistical gate -- no fixed elimination thresholds. [`AdaptationBus`]
//! composes per-arm adaptation policies (drift reracing, plasticity, meta
//! adaptation) in a Lipschitz-product framework.
//!
//! **Preprocessing and pipelines** -- [`IncrementalNormalizer`], [`CCIPCA`]
//! (O(kd) streaming PCA), [`MinMaxScaler`], [`OneHotEncoder`],
//! [`PolynomialFeatures`], [`FeatureHasher`], [`OnlineFeatureSelector`].
//! Chain with [`Pipeline::builder()`] or the [`pipe`] factory.
//!
//! **Evaluation and drift** -- [`PrequentialEvaluator`],
//! [`AdaptiveConformalInterval`], [`StreamingAUC`], [`EwmaRegressionMetrics`],
//! drift detectors (Page-Hinkley, ADWIN, DDM) via [`DriftDetector`].
//!
//! **Bandits** -- [`EpsilonGreedy`], [`UCB1`], [`UCBTuned`],
//! [`ThompsonSampling`], [`LinUCB`], [`DiscountedThompsonSampling`].
//!
//! **Clustering** -- [`StreamingKMeans`], [`CluStream`], [`DBStream`].
//!
//! **Anomaly detection** -- [`HalfSpaceTree`].
//!
//! **Projection** -- [`ProjectedLearner`] via PAST subspace tracking
//! (Yang, 1995), supervised or PCA mode.
//!
//! # Embedded deployment
//!
//! The companion crate `irithyll-core` is `#![no_std]` and exports trained
//! trees as 12-byte packed nodes ([`PackedNode`]) that traverse branch-free on
//! Cortex-M0+. Train in the cloud, export with [`export_embedded`], deploy on
//! bare metal. The boundary is hard and tested against `thumbv6m-none-eabi`.
//!
//! # Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `serde-json` | Yes | JSON model checkpoint / restore |
//! | `serde-bincode` | No | Bincode serialization (compact, fast) |
//! | `parallel` | No | Rayon-parallel tree training via [`ParallelSGBT`] |
//! | `simd` | No | AVX2 histogram acceleration |
//! | `simd-avx2` | No | Explicit AVX2 SIMD intrinsics |
//! | `kmeans-binning` | No | K-means histogram binning strategy |
//! | `arrow` | No | Apache Arrow `RecordBatch` integration |
//! | `parquet` | No | Parquet file I/O |
//! | `onnx` | No | ONNX model export |
//! | `neural-leaves` | No | Experimental MLP leaf models |
//! | `distill` | No | Knowledge distillation for [`AutoTuner`] racing |
//! | `full` | No | All of the above |
//!
//! # Quick Start
//!
//! The smallest useful pipeline -- normalize, boost, predict:
//!
//! ```no_run
//! use irithyll::{pipe, normalizer, sgbt, StreamingLearner};
//!
//! let mut model = pipe(normalizer()).learner(sgbt(50, 0.01));
//! model.train(&[100.0, 0.5, 42.0], 3.14);
//! let pred = model.predict(&[100.0, 0.5, 42.0]);
//! ```
//!
//! Race three model families against each other -- let the data choose:
//!
//! ```no_run
//! use irithyll::{automl::{AutoTuner, Factory}, StreamingLearner};
//!
//! let mut tuner = AutoTuner::builder()
//!     .add_factory(Factory::sgbt(5))
//!     .add_factory(Factory::mamba(5))
//!     .add_factory(Factory::esn())
//!     .build()
//!     .unwrap();
//!
//! tuner.train(&[1.0, 2.0, 3.0, 4.0, 5.0], 6.0);
//! let pred = tuner.predict(&[1.0, 2.0, 3.0, 4.0, 5.0]);
//! ```
//!
//! Wrap any regressor for binary classification:
//!
//! ```no_run
//! use irithyll::{sgbt, binary_classifier, StreamingLearner};
//!
//! let mut clf = binary_classifier(sgbt(50, 0.05));
//! clf.train(&[1.5, -0.3, 2.1], 1.0);
//! let label = clf.predict(&[1.5, -0.3, 2.1]);
//! ```
//!
//! For the extended ergonomics guide -- pipeline composition, AutoML
//! tournaments, drift wiring, embedded deployment -- see
//! [`docs/USAGE.md`](https://github.com/evilrat420/irithyll/blob/main/docs/USAGE.md)
//! and [`MODELS.md`](https://github.com/evilrat420/irithyll/blob/main/MODELS.md).
//!
//! # Design Principles
//!
//! **One sample at a time, every time.** No mini-batches hidden inside
//! `train_one`. Architectures that originally required offline training (TTT,
//! KAN, Mamba) are reimplemented with online updates that converge
//! sample-by-sample -- and tested for it.
//!
//! **O(1) memory per model.** State size is a function of the model, not the
//! data seen. Drift detectors are bounded ring buffers; histograms have fixed
//! bin counts; subspace trackers carry rank-k projections, not covariance
//! matrices.
//!
//! **Bounded readouts before linear heads.** Every neural model that feeds a
//! recursive least squares head bounds its features first -- `tanh`, `sigmoid`,
//! L2-normalize, clamp. Unbounded features explode the RLS weights silently.
//!
//! **Every threshold derives from a paper or the data.** Bernstein bounds over
//! fixed elimination thresholds. Information-decay matching over grid-searched
//! half-lives. Magic numbers are technical debt.

pub mod common;
pub mod error;
pub mod sample;

pub mod drift;
pub mod ensemble;
pub mod generators;
pub mod histogram;
pub mod kan;
pub mod loss;
pub mod metrics;
pub mod moe;
pub mod stream;
pub mod tree;

pub mod anomaly;
pub mod attention;
pub mod automl;
pub mod bandits;

pub mod clustering;
pub mod continual;
pub mod evaluation;
pub mod explain;
pub mod learner;
pub mod learners;
pub mod pipeline;
pub mod preprocessing;
pub mod projection;
pub mod reservoir;
pub mod serde_support;
pub mod snn;
pub mod ssm;
pub mod time_series;
pub mod ttt;

pub mod lstm;
pub mod mgrade;

#[cfg(feature = "arrow")]
pub mod arrow_support;

#[cfg(feature = "onnx")]
pub mod onnx_export;

pub mod export_embedded;

/// River-style ergonomic pipeline construction macro.
///
/// Expands a sequence of preprocessors and a terminal learner into the nested
/// `PipelineBuilder::new().pipe(...).pipe(...).learner(...)` form.
///
/// # Syntax
///
/// ```text
/// make_pipeline!(preprocessor => ... => preprocessor => learner)
/// make_pipeline!(learner)   // degenerate: no preprocessors
/// ```
///
/// The final (rightmost) argument is the learner; all preceding arguments
/// are preprocessors chained via `.pipe()`.
///
/// # Examples
///
/// ```rust
/// use irithyll::{make_pipeline, normalizer, ccipca, sgbt, StreamingLearner};
///
/// let mut model = make_pipeline!(normalizer() => ccipca(3) => sgbt(50, 0.01));
/// model.train(&[1.0, 2.0, 3.0, 4.0, 5.0], 42.0);
/// let pred = model.predict(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert!(pred.is_finite());
/// ```
///
/// Degenerate case (single learner, no preprocessors):
///
/// ```rust
/// use irithyll::{make_pipeline, sgbt, StreamingLearner};
///
/// let mut model = make_pipeline!(sgbt(50, 0.01));
/// model.train(&[1.0, 2.0], 3.0);
/// let pred = model.predict(&[1.0, 2.0]);
/// assert!(pred.is_finite());
/// ```
#[macro_export]
macro_rules! make_pipeline {
    // Degenerate case: a single learner with no preprocessors.
    // Produces a Pipeline with an empty preprocessor list.
    ($learner:expr) => {
        $crate::pipeline::Pipeline::builder().learner($learner)
    };

    // Base case: one preprocessor followed by the terminal learner.
    ($pre:expr => $learner:expr) => {
        $crate::pipeline::Pipeline::builder()
            .pipe($pre)
            .learner($learner)
    };

    // Recursive case: strip the leftmost preprocessor, recurse on the rest,
    // then prepend via .pipe(). We achieve this by expanding the inner tail
    // into a PipelineBuilder (using a builder-internal helper arm), then
    // terminating with the rightmost learner.
    //
    // We implement using an internal accumulator pattern:
    // make_pipeline!(@acc builder; rest...; learner)
    ($first:expr => $($rest:expr)=>+) => {{
        let builder = $crate::pipeline::Pipeline::builder().pipe($first);
        $crate::make_pipeline!(@acc builder; $($rest)=>+)
    }};

    // Accumulator: more than one token left — next token is a preprocessor.
    (@acc $builder:expr; $pre:expr => $($rest:expr)=>+) => {{
        let builder = $builder.pipe($pre);
        $crate::make_pipeline!(@acc builder; $($rest)=>+)
    }};

    // Accumulator base: only the terminal learner remains.
    (@acc $builder:expr; $learner:expr) => {
        $builder.learner($learner)
    };
}

// Re-exports -- irithyll-core packed inference
pub use irithyll_core;
#[doc(inline)]
pub use irithyll_core::{
    EnsembleView, FormatError, PackedNode, PackedNodeI16, QuantizedEnsembleHeader,
    QuantizedEnsembleView,
};

// Re-exports -- TurboQuant quantization (8-bit, 3.5-bit, 2.5-bit)
#[doc(inline)]
pub use irithyll_core::turbo_quant::{
    quantize, quantize_f32, quantize_i16, quantize_weights, quantize_weights_with_seed, QuantMode,
    TurboQuantized, TurboQuantizedView,
};

// Re-exports -- SIMD activations
#[doc(inline)]
pub use irithyll_core::simd::{simd_exp, simd_sigmoid, simd_silu, simd_tanh};

// Re-exports -- core types
pub use common::{PlasticityConfig, PlasticityConfigBuilder};
pub use drift::{DriftDetector, DriftSignal};
pub use ensemble::adaptive::AdaptiveSGBT;
pub use ensemble::bagged::BaggedSGBT;
pub use ensemble::config::{SGBTConfig, ScaleMode};
pub use ensemble::diagnostics::{DistributionalDiagnostics, EnsembleDiagnostics, TreeDiagnostics};
pub use ensemble::distributional::{
    DecomposedPrediction, DistributionalSGBT, DistributionalTreeDiagnostic, GaussianPrediction,
    ModelDiagnostics,
};
pub use ensemble::moe_distributional::MoEDistributionalSGBT;
pub use ensemble::multi_target::MultiTargetSGBT;
pub use ensemble::multiclass::MulticlassSGBT;
pub use ensemble::quantile_regressor::QuantileRegressorSGBT;
pub use ensemble::{DynSGBT, SGBT};
pub use error::{ConfigError, IrithyllError};
pub use histogram::{BinnerKind, BinningStrategy};
#[doc(inline)]
pub use irithyll_core::feature::FeatureType;
pub use loss::{Loss, LossType};
pub use sample::{Observation, Sample, SampleRef};
pub use tree::leaf_model::LeafModelType;
pub use tree::StreamingTree;

// Re-exports -- explainability
pub use explain::importance_drift::ImportanceDriftMonitor;
pub use explain::streaming::StreamingShap;
pub use explain::treeshap::ShapValues;

// Re-exports -- parallel (feature-gated)
#[cfg(feature = "parallel")]
pub use ensemble::parallel::ParallelSGBT;

// Re-exports -- async streaming
pub use stream::{AsyncSGBT, Prediction, PredictionStream, Predictor, SampleSender};

// Re-exports -- metrics
pub use metrics::auc::StreamingAUC;
pub use metrics::conformal::AdaptiveConformalInterval;
pub use metrics::ewma::{EwmaClassificationMetrics, EwmaRegressionMetrics};
pub use metrics::kappa::{CohenKappa, KappaM, KappaT};
pub use metrics::rolling::{RollingClassificationMetrics, RollingRegressionMetrics};
pub use metrics::{
    ClassificationMetrics, FeatureImportance, MetricSet, OnlineTemperatureScaling,
    RegressionMetrics,
};
// Re-exports -- StreamingMetric trait and composable instances (Wave 7-2)
pub use metrics::{Accuracy, LogLoss, MetricUnion, Pinball, StreamingMetric, MAE, MSE, R2, RMSE};

// Re-exports -- evaluation
pub use evaluation::{
    HoldoutStrategy, PrequentialConfig, PrequentialEvaluator, ProgressiveValidator,
};

// Re-exports -- clustering
pub use clustering::{
    CluStream, CluStreamConfig, ClusterFeature, DBStream, DBStreamConfig, MicroCluster,
    StreamingKMeans, StreamingKMeansConfig,
};

// Re-exports -- classification
pub use ensemble::adaptive_forest::AdaptiveRandomForest;
pub use learners::{BernoulliNB, ClassificationMode, ClassificationWrapper, MultinomialNB};
pub use tree::hoeffding_classifier::HoeffdingTreeClassifier;

// Re-exports -- anomaly detection
pub use anomaly::hst::{AnomalyScore, HSTConfig, HalfSpaceTree};

// Re-exports -- streaming learner trait and capability traits
pub use learner::{HasReadout, SGBTLearner, StreamingLearner, Structural, Tunable};

// Re-exports -- continual learning
pub use continual::ContinualLearner;

// Re-exports -- preprocessing & pipeline
pub use pipeline::{Pipeline, PipelineBuilder, StreamingPreprocessor};
pub use preprocessing::{
    FeatureHasher, IncrementalNormalizer, MinMaxScaler, OneHotEncoder, OnlineFeatureSelector,
    PolynomialFeatures, TargetEncoder, CCIPCA,
};
// === Wave 7-3 target preprocessor ===
pub use preprocessing::{
    StreamingTargetPreprocessor, TargetEncoderPreprocessor, TargetLog1pTransform, TargetScaler,
};

// Re-exports -- learning rate scheduling
pub use ensemble::lr_schedule::LRScheduler;

// Re-exports -- streaming learners
pub use learners::{
    GaussianNB, Kernel, LinearKernel, LocallyWeightedRegression, MondrianForest, PolynomialKernel,
    RBFKernel, RecursiveLeastSquares, StreamingLinearModel, StreamingPolynomialRegression, KRLS,
};

// Re-exports -- time series
pub use time_series::{
    DecomposedPoint, DecompositionConfig, HoltWinters, HoltWintersConfig, SNARIMAXCoefficients,
    SNARIMAXConfig, Seasonality, StreamingDecomposition, SNARIMAX,
};

// Re-exports -- bandits
pub use bandits::{
    Bandit, ContextualBandit, DiscountedThompsonSampling, EpsilonGreedy, LinUCB, ThompsonSampling,
    UCBTuned, UCB1,
};

// Re-exports -- reservoir computing
pub use reservoir::{
    ESNConfig, ESNConfigBuilder, ESNPreprocessor, EchoStateNetwork, NGRCConfig, NGRCConfigBuilder,
    NextGenRC, StreamingESN,
};

// Re-exports -- state space models
pub use ssm::{MambaConfig, MambaConfigBuilder, MambaPreprocessor, StreamingMamba};

// Re-exports -- spiking neural networks
pub use snn::{
    SpikeNet, SpikeNetConfig, SpikeNetConfigBuilder, SpikePreprocessor, StreamingSpikeNet,
};

// Re-exports -- test-time training
pub use ttt::{StreamingTTT, TTTConfig, TTTConfigBuilder};

// Re-exports -- sLSTM (stabilized LSTM with exponential gating)
pub use lstm::{SLSTMConfig, SLSTMConfigBuilder, StreamingLSTM, StreamingsLSTM};

// Re-exports -- mGRADE (minimal recurrent gating with delay convolutions)
pub use mgrade::{MGradeConfig, MGradeConfigBuilder, StreamingMGrade};

// Re-exports -- Kolmogorov-Arnold Networks
pub use kan::{KANConfig, KANConfigBuilder, StreamingKAN};

// Re-exports -- streaming linear attention
pub use attention::{
    default_lambda_init, AttentionPreprocessor, LogLinearAttention, LogLinearState,
    StreamingAttentionConfig, StreamingAttentionConfigBuilder, StreamingAttentionModel,
    DEFAULT_MAX_LEVELS, DEFAULT_TAU,
};

// Re-exports -- neural moe
pub use moe::{NeuralMoE, NeuralMoEBuilder, NeuralMoEConfig};

// Re-exports -- projection
pub use projection::{ProjectedLearner, ProjectionConfig, ProjectionConfigBuilder};

// Re-exports -- automl
pub use automl::RewardNormalizer;
pub use automl::{Algorithm, Factory, FactoryError};
pub use automl::{AutoMetric, AutoTuner, AutoTunerBuilder, AutoTunerConfig, ModelFactory};
pub use automl::{
    ConfigDiagnostics, DiagnosticAdaptor, DiagnosticLearner, DiagnosticSource, FeasibleRegion,
    MetaObjective, TerminateAfter, WelfordRace,
};
// AM-15 Knowledge distillation for WelfordRace (default OFF, enable via `distill` feature).
#[cfg(feature = "distill")]
pub use automl::{DistillationConfig, DistillationStats};
// AdaptationBus — compose-safe per-arm adaptation coordination (AM-12).
pub use automl::{
    AdaptContext, AdaptationBus, BusError, CriticalGuard, DriftRateAdapter, MetaAdapter,
    NoOpAdapter, PlasticityAdapter, ThetaDelta,
};
// Typed search-space API (v10).
pub use automl::{
    categorical, int_range, linear_range, log_range, when, Category, Condition, Constraint,
    ParamDef, ParamMap, ParamValue, SamplerError, Scale, SearchSpace, SpaceError,
};
// Empirical Bernstein racing — statistical promotion gate (AM-2).
pub use automl::{
    bernstein_compare, bernstein_halfwidth, bernstein_promotion_test, empirical_bernstein_ci,
    ewma_bernstein_ci, ArmStats, EwmaWelfordTracker, PromotionVerdict, WelfordTracker,
    BERNSTEIN_DELTA, MIN_SAMPLES_FOR_BERNSTEIN,
};
// Samples × complexity budget accounting (AM-4).
pub use automl::{ArmBudget, BudgetLedger, BudgetStatus};
// Top-K champion cohort (AM-5).
pub use automl::{ChampionCohort, CohortMember, CohortMemberSnapshot, CohortWeight, COHORT_K};
// Cross-model MetaLearner trait (AM-11) — per-family declaration consumed by
// the AdaptationBus (AM-12) for Lipschitz-product compose-safety.
pub use automl::{
    ComplexityClass, FactoryMetaLearner, MetaLearner, MetaScore, MetaSearch, NoOpMetaLearner,
    Objective, SgbtClassificationMetaLearner, SgbtMetaLearner,
};
// Legacy positional config-space API (deprecated; remove in v11).
#[allow(deprecated)]
pub use automl::{ConfigSampler, ConfigSpace, HyperConfig, HyperParam};

// ---------------------------------------------------------------------------
// Convenience factory functions
// ---------------------------------------------------------------------------

/// Create an SGBT learner with squared loss from minimal parameters.
///
/// For full control, use [`SGBTConfig::builder()`] directly.
///
/// ```
/// use irithyll::{sgbt, StreamingLearner};
///
/// let mut model = sgbt(50, 0.01);
/// model.train(&[1.0, 2.0], 3.0);
/// let pred = model.predict(&[1.0, 2.0]);
/// ```
pub fn sgbt(n_steps: usize, learning_rate: f64) -> SGBTLearner {
    let config = SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(learning_rate)
        .build()
        .expect("sgbt() factory: invalid parameters");
    SGBTLearner::from_config(config)
}

/// Create a streaming linear model with the given learning rate.
///
/// ```
/// use irithyll::{linear, StreamingLearner};
///
/// let mut model = linear(0.01);
/// model.train(&[1.0, 2.0], 3.0);
/// ```
pub fn linear(learning_rate: f64) -> StreamingLinearModel {
    StreamingLinearModel::new(learning_rate)
}

/// Create a recursive least squares model with the given forgetting factor.
///
/// ```
/// use irithyll::{rls, StreamingLearner};
///
/// let mut model = rls(0.99);
/// model.train(&[1.0, 2.0], 3.0);
/// ```
pub fn rls(forgetting_factor: f64) -> RecursiveLeastSquares {
    RecursiveLeastSquares::new(forgetting_factor)
}

/// Create a Gaussian Naive Bayes classifier.
///
/// ```
/// use irithyll::{gaussian_nb, StreamingLearner};
///
/// let mut model = gaussian_nb();
/// model.train(&[1.0, 2.0], 0.0);
/// ```
pub fn gaussian_nb() -> GaussianNB {
    GaussianNB::new()
}

/// Create a Mondrian forest with the given number of trees.
///
/// ```
/// use irithyll::{mondrian, StreamingLearner};
///
/// let mut model = mondrian(10);
/// model.train(&[1.0, 2.0], 3.0);
/// ```
pub fn mondrian(n_trees: usize) -> MondrianForest {
    let config = learners::mondrian::MondrianForestConfig::builder()
        .n_trees(n_trees)
        .build()
        .expect("mondrian() factory: invalid parameters");
    MondrianForest::new(config)
}

/// Create an incremental normalizer for streaming standardization.
///
/// ```
/// use irithyll::{normalizer, StreamingPreprocessor};
///
/// let mut norm = normalizer();
/// let z = norm.update_and_transform(&[10.0, 200.0]);
/// ```
pub fn normalizer() -> IncrementalNormalizer {
    IncrementalNormalizer::new()
}

/// Start building a pipeline with the first preprocessor.
///
/// Shorthand for `Pipeline::builder().pipe(preprocessor)`.
///
/// ```
/// use irithyll::{pipe, normalizer, sgbt, StreamingLearner};
///
/// let mut pipeline = pipe(normalizer()).learner(sgbt(10, 0.01));
/// pipeline.train(&[100.0, 0.5], 42.0);
/// let pred = pipeline.predict(&[100.0, 0.5]);
/// ```
pub fn pipe(preprocessor: impl StreamingPreprocessor + 'static) -> PipelineBuilder {
    PipelineBuilder::new().pipe(preprocessor)
}

/// Create a kernel recursive least squares model with an RBF kernel.
///
/// ```
/// use irithyll::{krls, StreamingLearner};
///
/// let mut model = krls(1.0, 100, 1e-4);
/// model.train(&[1.0], 1.0_f64.sin());
/// ```
pub fn krls(gamma: f64, budget: usize, ald_threshold: f64) -> KRLS {
    KRLS::new(Box::new(RBFKernel::new(gamma)), budget, ald_threshold)
}

/// Create a CCIPCA preprocessor for streaming dimensionality reduction.
///
/// ```
/// use irithyll::{ccipca, StreamingPreprocessor};
///
/// let mut pca = ccipca(3);
/// let reduced = pca.update_and_transform(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert_eq!(reduced.len(), 3);
/// ```
pub fn ccipca(n_components: usize) -> CCIPCA {
    CCIPCA::new(n_components)
}

/// Create a feature hasher for fixed-size dimensionality reduction.
///
/// ```
/// use irithyll::{feature_hasher, StreamingPreprocessor};
///
/// let mut h = feature_hasher(32);
/// let hashed = h.update_and_transform(&[1.0, 2.0, 3.0]);
/// assert_eq!(hashed.len(), 32);
/// ```
pub fn feature_hasher(n_buckets: usize) -> FeatureHasher {
    FeatureHasher::new(n_buckets)
}

/// Create a min-max scaler that normalizes features to `[0, 1]`.
///
/// ```
/// use irithyll::{min_max_scaler, StreamingPreprocessor};
///
/// let mut scaler = min_max_scaler();
/// let _ = scaler.update_and_transform(&[10.0, 200.0]);
/// ```
pub fn min_max_scaler() -> MinMaxScaler {
    MinMaxScaler::new()
}

/// Create a one-hot encoder for the given categorical feature indices.
///
/// ```
/// use irithyll::{one_hot, StreamingPreprocessor};
///
/// let mut enc = one_hot(vec![0]); // feature 0 is categorical
/// let encoded = enc.update_and_transform(&[2.0, 3.5]);
/// ```
pub fn one_hot(categorical_indices: Vec<usize>) -> OneHotEncoder {
    OneHotEncoder::new(categorical_indices)
}

/// Create a degree-2 polynomial feature generator (interactions + squares).
///
/// ```
/// use irithyll::{polynomial_features, StreamingPreprocessor};
///
/// let poly = polynomial_features();
/// let expanded = poly.transform(&[1.0, 2.0]);
/// assert_eq!(expanded.len(), 5); // [x0, x1, x0*x0, x0*x1, x1*x1]
/// ```
pub fn polynomial_features() -> PolynomialFeatures {
    PolynomialFeatures::new()
}

/// Create a target encoder with Bayesian smoothing for categorical features.
///
/// Note: [`TargetEncoder`] does not implement [`StreamingPreprocessor`] because
/// it requires the target value. Use its methods directly.
///
/// ```
/// use irithyll::target_encoder;
///
/// let mut enc = target_encoder(vec![0]); // feature 0 is categorical
/// enc.update(&[1.0, 3.5], 10.0);
/// let encoded = enc.transform(&[1.0, 3.5]);
/// ```
pub fn target_encoder(categorical_indices: Vec<usize>) -> TargetEncoder {
    TargetEncoder::new(categorical_indices)
}

/// Create an adaptive SGBT with a learning rate scheduler.
///
/// ```
/// use irithyll::{adaptive_sgbt, StreamingLearner};
/// use irithyll::ensemble::lr_schedule::ExponentialDecayLR;
///
/// let mut model = adaptive_sgbt(50, 0.1, ExponentialDecayLR::new(0.1, 0.999));
/// model.train(&[1.0, 2.0], 3.0);
/// ```
pub fn adaptive_sgbt(
    n_steps: usize,
    learning_rate: f64,
    scheduler: impl ensemble::lr_schedule::LRScheduler + 'static,
) -> AdaptiveSGBT {
    let config = SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(learning_rate)
        .build()
        .expect("adaptive_sgbt() factory: invalid parameters");
    AdaptiveSGBT::new(config, scheduler)
}

/// Create an epsilon-greedy bandit with the given number of arms and exploration rate.
///
/// ```
/// use irithyll::{epsilon_greedy, Bandit};
///
/// let mut bandit = epsilon_greedy(3, 0.1);
/// let arm = bandit.select_arm();
/// bandit.update(arm, 1.0);
/// ```
pub fn epsilon_greedy(n_arms: usize, epsilon: f64) -> EpsilonGreedy {
    EpsilonGreedy::new(n_arms, epsilon)
}

/// Create a UCB1 bandit with the given number of arms.
///
/// ```
/// use irithyll::{ucb1, Bandit};
///
/// let mut bandit = ucb1(3);
/// let arm = bandit.select_arm();
/// bandit.update(arm, 1.0);
/// ```
pub fn ucb1(n_arms: usize) -> UCB1 {
    UCB1::new(n_arms)
}

/// Create a UCB-Tuned bandit with the given number of arms.
///
/// ```
/// use irithyll::{ucb_tuned, Bandit};
///
/// let mut bandit = ucb_tuned(3);
/// let arm = bandit.select_arm();
/// bandit.update(arm, 1.0);
/// ```
pub fn ucb_tuned(n_arms: usize) -> UCBTuned {
    UCBTuned::new(n_arms)
}

/// Create a Thompson Sampling bandit with Beta(1,1) prior.
///
/// Rewards should be in `[0, 1]` (Bernoulli setting).
///
/// ```
/// use irithyll::{thompson, Bandit};
///
/// let mut bandit = thompson(3);
/// let arm = bandit.select_arm();
/// bandit.update(arm, 1.0);
/// ```
pub fn thompson(n_arms: usize) -> ThompsonSampling {
    ThompsonSampling::new(n_arms)
}

/// Create a LinUCB contextual bandit.
///
/// ```
/// use irithyll::{lin_ucb, ContextualBandit};
///
/// let mut bandit = lin_ucb(3, 5, 1.0);
/// let ctx = vec![0.1, 0.2, 0.3, 0.4, 0.5];
/// let arm = bandit.select_arm(&ctx);
/// bandit.update(arm, &ctx, 1.0);
/// ```
pub fn lin_ucb(n_arms: usize, n_features: usize, alpha: f64) -> LinUCB {
    LinUCB::new(n_arms, n_features, alpha)
}

/// Create a Next Generation Reservoir Computer.
///
/// ```
/// use irithyll::{ngrc, StreamingLearner};
///
/// let mut model = ngrc(2, 1, 2);
/// model.train(&[1.0], 2.0);
/// model.train(&[2.0], 3.0);
/// model.train(&[3.0], 4.0);
/// let pred = model.predict(&[4.0]);
/// ```
pub fn ngrc(k: usize, s: usize, degree: usize) -> reservoir::NextGenRC {
    reservoir::NextGenRC::new(
        reservoir::NGRCConfig::builder()
            .k(k)
            .s(s)
            .degree(degree)
            .build()
            .expect("ngrc() factory: invalid parameters"),
    )
}

/// Create an Echo State Network with cycle topology.
///
/// ```
/// use irithyll::{esn, StreamingLearner};
///
/// let mut model = esn(50, 0.9);
/// for i in 0..60 {
///     model.train(&[i as f64 * 0.1], 0.0);
/// }
/// let pred = model.predict(&[1.0]);
/// ```
pub fn esn(n_reservoir: usize, spectral_radius: f64) -> reservoir::EchoStateNetwork {
    reservoir::EchoStateNetwork::new(
        reservoir::ESNConfig::builder()
            .n_reservoir(n_reservoir)
            .spectral_radius(spectral_radius)
            .build()
            .expect("esn() factory: invalid parameters"),
    )
}

/// Create an ESN preprocessor for pipeline composition.
///
/// ```
/// use irithyll::{esn_preprocessor, pipe, rls, StreamingLearner};
///
/// let mut pipeline = pipe(esn_preprocessor(30, 0.9)).learner(rls(0.998));
/// pipeline.train(&[1.0], 2.0);
/// let pred = pipeline.predict(&[1.5]);
/// ```
pub fn esn_preprocessor(n_reservoir: usize, spectral_radius: f64) -> reservoir::ESNPreprocessor {
    reservoir::ESNPreprocessor::new(
        reservoir::ESNConfig::builder()
            .n_reservoir(n_reservoir)
            .spectral_radius(spectral_radius)
            .warmup(0)
            .build()
            .expect("esn_preprocessor() factory: invalid parameters"),
    )
}

/// Create a streaming Mamba (selective SSM) model.
///
/// ```
/// use irithyll::{mamba, StreamingLearner};
///
/// let mut model = mamba(3, 16);
/// model.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = model.predict(&[1.0, 2.0, 3.0]);
/// ```
pub fn mamba(d_in: usize, n_state: usize) -> ssm::StreamingMamba {
    ssm::StreamingMamba::new(
        ssm::MambaConfig::builder()
            .d_in(d_in)
            .n_state(n_state)
            .build()
            .expect("mamba() factory: invalid parameters"),
    )
}

/// Create a Mamba preprocessor for pipeline composition.
///
/// ```
/// use irithyll::{mamba_preprocessor, pipe, rls, StreamingLearner};
///
/// let mut pipeline = pipe(mamba_preprocessor(3, 8)).learner(rls(0.99));
/// pipeline.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = pipeline.predict(&[1.0, 2.0, 3.0]);
/// ```
pub fn mamba_preprocessor(d_in: usize, n_state: usize) -> ssm::MambaPreprocessor {
    ssm::MambaPreprocessor::new(d_in, n_state, 42)
}

/// Create a streaming Mamba with BD-LRU block-diagonal recurrence.
///
/// Channels are grouped into blocks of `block_size` with dense cross-channel
/// mixing within each block.
///
/// ```
/// use irithyll::{mamba_bd, StreamingLearner};
///
/// let mut model = mamba_bd(8, 16, 4);
/// model.train(&[0.1; 8], 1.0);
/// let pred = model.predict(&[0.1; 8]);
/// ```
pub fn mamba_bd(d_in: usize, n_state: usize, block_size: usize) -> ssm::StreamingMamba {
    ssm::StreamingMamba::new(
        ssm::MambaConfig::builder()
            .d_in(d_in)
            .n_state(n_state)
            .version(ssm::MambaVersion::BlockDiagonal { block_size })
            .block_size(block_size)
            .build()
            .expect("mamba_bd() factory: invalid parameters"),
    )
}

/// Create a spiking neural network with e-prop learning.
///
/// ```
/// use irithyll::{spikenet, StreamingLearner};
///
/// let mut model = spikenet(32);
/// model.train(&[0.5, -0.3], 1.0);
/// let pred = model.predict(&[0.5, -0.3]);
/// ```
pub fn spikenet(n_hidden: usize) -> snn::SpikeNet {
    snn::SpikeNet::new(
        snn::SpikeNetConfig::builder()
            .n_hidden(n_hidden)
            .build()
            .expect("spikenet() factory: invalid parameters"),
    )
}

/// Create a streaming TTT (Test-Time Training) model.
///
/// The hidden state is a linear model updated by gradient descent at every
/// step. Optional Titans-style momentum and weight decay.
///
/// ```no_run
/// use irithyll::{streaming_ttt, StreamingLearner};
///
/// let mut model = streaming_ttt(16, 0.1);
/// model.train(&[1.0, 2.0], 3.0);
/// let pred = model.predict(&[1.0, 2.0]);
/// ```
pub fn streaming_ttt(d_model: usize, learning_rate: f64) -> ttt::StreamingTTT {
    ttt::StreamingTTT::new(
        ttt::TTTConfig::builder()
            .d_model(d_model)
            .learning_rate(learning_rate)
            .build()
            .expect("streaming_ttt() factory: invalid parameters"),
    )
}

/// Create a streaming sLSTM (stabilized LSTM with exponential gating).
///
/// Uses exponential gates with log-domain stabilization for numerically
/// stable long-range memory. RLS readout maps hidden state to predictions.
///
/// ```no_run
/// use irithyll::{streaming_slstm, StreamingLearner};
///
/// let mut model = streaming_slstm(16);
/// model.train(&[1.0, 2.0], 3.0);
/// let pred = model.predict(&[1.0, 2.0]);
/// ```
pub fn streaming_slstm(d_model: usize) -> lstm::StreamingLSTM {
    lstm::StreamingLSTM::new(
        lstm::SLSTMConfig::builder()
            .d_model(d_model)
            .build()
            .expect("streaming_slstm() factory: invalid parameters"),
    )
}

/// Create a streaming mGRADE (minimal recurrent gating with delay convolutions).
///
/// ```no_run
/// use irithyll::{mgrade, StreamingLearner};
///
/// let mut model = mgrade(3, 16);
/// model.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = model.predict(&[1.0, 2.0, 3.0]);
/// ```
pub fn mgrade(d_in: usize, d_hidden: usize) -> mgrade::StreamingMGrade {
    mgrade::StreamingMGrade::new(
        mgrade::MGradeConfig::builder()
            .d_in(d_in)
            .d_hidden(d_hidden)
            .build()
            .expect("mgrade() factory: invalid parameters"),
    )
}

/// Create a streaming KAN with the given layer sizes and learning rate.
///
/// ```no_run
/// use irithyll::{streaming_kan, StreamingLearner};
///
/// let mut model = streaming_kan(&[3, 10, 1], 0.01);
/// model.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = model.predict(&[1.0, 2.0, 3.0]);
/// ```
pub fn streaming_kan(layer_sizes: &[usize], learning_rate: f64) -> kan::StreamingKAN {
    kan::StreamingKAN::new(
        kan::KANConfig::builder()
            .layer_sizes(layer_sizes.to_vec())
            .learning_rate(learning_rate)
            .build()
            .expect("streaming_kan() factory: invalid layer sizes"),
    )
}

/// Create a Gated Linear Attention model (SOTA streaming attention).
pub fn gla(d_model: usize, n_heads: usize) -> attention::StreamingAttentionModel {
    attention::gla(d_model, n_heads)
}

/// Create a Gated DeltaNet model (strongest retrieval, NVIDIA 2024).
pub fn delta_net(d_model: usize, n_heads: usize) -> attention::StreamingAttentionModel {
    attention::delta_net(d_model, n_heads)
}

/// Create a Hawk model (lightest streaming attention, vector state).
pub fn hawk(d_model: usize) -> attention::StreamingAttentionModel {
    attention::hawk(d_model)
}

/// Create a RetNet model (simplest, fixed decay).
pub fn ret_net(d_model: usize, gamma: f64) -> attention::StreamingAttentionModel {
    attention::ret_net(d_model, gamma)
}

/// Create a streaming attention model with any mode.
pub fn streaming_attention(
    d_model: usize,
    mode: irithyll_core::attention::AttentionMode,
) -> attention::StreamingAttentionModel {
    attention::streaming_attention(d_model, mode)
}

/// Create a Log-Linear Attention model (Han Guo et al., ICLR 2026 — v10 headline).
///
/// Wraps any inner linear-attention rule with an O(log T) hierarchical
/// Fenwick state, bridging linear-attention efficiency and softmax
/// expressivity. State memory is `max_levels * d_k * d_v * n_heads`
/// per layer.
pub fn log_linear(
    d_model: usize,
    n_heads: usize,
    inner: irithyll_core::attention::AttentionMode,
    max_levels: usize,
) -> attention::StreamingAttentionModel {
    attention::log_linear(d_model, n_heads, inner, max_levels)
}

/// Create an auto-tuning streaming learner with default settings.
///
/// Uses champion-challenger racing to automatically tune hyperparameters
/// for the given model factory. The champion always provides predictions
/// while challengers with different configs are evaluated in parallel.
///
/// For full control, use [`AutoTuner::builder()`].
///
/// ```no_run
/// use irithyll::{auto_tune, automl::Factory, StreamingLearner};
///
/// let mut tuner = auto_tune(Factory::sgbt(5));
/// tuner.train(&[1.0, 2.0, 3.0, 4.0, 5.0], 10.0);
/// let pred = tuner.predict(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// ```
pub fn auto_tune(factory: impl automl::ModelFactory + 'static) -> automl::AutoTuner {
    automl::AutoTuner::builder()
        .factory(factory)
        .build()
        .expect("auto_tune() factory: invalid parameters")
}

/// Wrap any streaming learner for binary classification.
///
/// The inner model is trained with {0, 1} targets. At prediction time,
/// sigmoid is applied to the raw output and thresholded at 0.5.
///
/// ```
/// use irithyll::{binary_classifier, rls, StreamingLearner};
///
/// let mut clf = binary_classifier(rls(0.99));
/// clf.train(&[1.0, 2.0], 1.0);
/// clf.train(&[-1.0, -2.0], 0.0);
/// let pred = clf.predict(&[1.0, 2.0]);
/// assert!(pred == 0.0 || pred == 1.0);
/// ```
pub fn binary_classifier(model: impl StreamingLearner + 'static) -> ClassificationWrapper {
    ClassificationWrapper::binary(Box::new(model))
}

/// Wrap any streaming learner for multiclass classification.
///
/// Creates K independent scalar heads (the inner model as head 0, plus
/// K-1 additional RLS heads). Predictions are softmax-normalized across
/// all heads and the argmax class index is returned.
///
/// # Panics
///
/// Panics if `n_classes < 2`.
///
/// ```
/// use irithyll::{multiclass_classifier, rls, StreamingLearner};
///
/// let mut clf = multiclass_classifier(rls(0.99), 3);
/// for i in 0..60 {
///     clf.train(&[(i % 3) as f64, 1.0], (i % 3) as f64);
/// }
/// let pred = clf.predict(&[1.0, 1.0]);
/// assert!(pred >= 0.0 && pred < 3.0);
/// ```
pub fn multiclass_classifier(
    model: impl StreamingLearner + 'static,
    n_classes: usize,
) -> ClassificationWrapper {
    ClassificationWrapper::multiclass(Box::new(model), n_classes)
}

// === Wave 7-1 wrapper factories ===
//
// Ergonomic factory functions for every wrapper type.  Each fn:
//   - accepts `impl StreamingLearner + 'static` — no explicit Box::new at call site.
//   - delegates directly to the wrapper's own constructor/builder — no re-implementation.
//   - returns the concrete wrapper type so callers retain access to its methods.
//
// Naming convention: verb + domain, matching the existing factory catalog
// (`sgbt`, `rls`, `normalizer`, ...).

/// Wrap any streaming learner with drift-detected continual adaptation.
///
/// Uses a Page-Hinkley test (Gama et al., 2013 — prequential evaluation
/// protocol) as the default drift detector. For a different detector, call
/// [`ContinualLearner::new`] + [`ContinualLearner::with_drift_detector`]
/// directly.
///
/// The detector is fed `|prediction - target|` (absolute prequential error) on
/// every sample; when it signals `Drift`, the inner model is reset.
///
/// ```
/// use irithyll::{drift_aware, rls, StreamingLearner};
///
/// let mut model = drift_aware(rls(0.99));
/// for i in 0..100 {
///     model.train(&[i as f64], i as f64 * 2.0);
/// }
/// let pred = model.predict(&[50.0]);
/// assert!(pred.is_finite());
/// ```
pub fn drift_aware(learner: impl StreamingLearner + 'static) -> ContinualLearner {
    use irithyll_core::drift::pht::PageHinkleyTest;
    ContinualLearner::new(learner).with_drift_detector(PageHinkleyTest::new())
}

/// Wrap any streaming learner with online projection learning (PAST algorithm).
///
/// Applies Welford normalization, then projects `d_in`-dimensional inputs to
/// `rank` dimensions via the PAST subspace tracker (Yang, 1995 — "Projection
/// approximation subspace tracking"). The forgetting factor `lambda` controls
/// the PAST half-life: `lambda = 0.9999` ≈ 6 931-sample half-life.
///
/// When the inner model exposes RLS readout weights ([`HasReadout`]), the
/// projection updates toward prediction-relevant directions (supervised mode);
/// otherwise it tracks variance-maximising directions (PCA mode).
///
/// # Panics
///
/// Panics if `rank > d_in` (delegated to `SubspaceTracker::new`).
///
/// ```
/// use irithyll::{projected, rls, StreamingLearner};
///
/// let mut model = projected(rls(0.99), 10, 4, 0.9999);
/// for i in 0..20 {
///     model.train(&[i as f64; 10], i as f64);
/// }
/// let pred = model.predict(&[1.0; 10]);
/// assert!(pred.is_finite());
/// ```
pub fn projected(
    learner: impl StreamingLearner + 'static,
    d_in: usize,
    rank: usize,
    lambda: f64,
) -> ProjectedLearner {
    let config = ProjectionConfig::builder()
        .rank(rank)
        .lambda(lambda)
        .build()
        .expect("projected() factory: invalid parameters");
    ProjectedLearner::from_learner(learner, d_in, config)
}

/// Wrap any streaming learner for multiclass classification.
///
/// Creates `n_classes` independent scalar heads (the inner model as head 0,
/// plus `n_classes - 1` additional RLS heads).  Predictions are
/// softmax-normalised across all heads and the argmax class index is returned.
///
/// # Panics
///
/// Panics if `n_classes < 2`.
///
/// ```
/// use irithyll::{multiclass, rls, StreamingLearner};
///
/// let mut clf = multiclass(rls(0.99), 3);
/// for i in 0..60 {
///     clf.train(&[(i % 3) as f64, 1.0], (i % 3) as f64);
/// }
/// let pred = clf.predict(&[1.0, 1.0]);
/// assert!(pred >= 0.0 && pred < 3.0);
/// ```
pub fn multiclass(
    learner: impl StreamingLearner + 'static,
    n_classes: usize,
) -> ClassificationWrapper {
    ClassificationWrapper::multiclass(Box::new(learner), n_classes)
}

// NOTE: `binary_classifier` already exists in the factory catalog above.
// The existing fn is the canonical definition; no duplicate needed here.

/// Wrap any streaming learner with champion-challenger auto-tuning.
///
/// Equivalent to `auto_tune(factory)` but named symmetrically with the other
/// wrapper factories.  Uses the default [`AutoTunerConfig`] (8 initial
/// candidates, 100-sample rounds, MAE metric).
///
/// ```no_run
/// use irithyll::{auto_tuner, automl::Factory, StreamingLearner};
///
/// let mut tuner = auto_tuner(Factory::sgbt(5));
/// tuner.train(&[1.0, 2.0, 3.0, 4.0, 5.0], 10.0);
/// let pred = tuner.predict(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert!(pred.is_finite());
/// ```
pub fn auto_tuner(factory: impl automl::ModelFactory + 'static) -> automl::AutoTuner {
    automl::AutoTuner::builder()
        .factory(factory)
        .build()
        .expect("auto_tuner() factory: invalid parameters")
}

// === Wave 7-1 presets ===
//
// High-level "one-liner" presets that assemble commonly-needed compositions.
// Every constant below is derived from published values or theory — no magic
// numbers.  Citations inline.

/// Production-default streaming regressor.
///
/// Assembles `IncrementalNormalizer → RecursiveLeastSquares(λ=0.99)` in a
/// single pipeline.
///
/// **Default rationale:**
/// - Normalizer: zero-mean / unit-variance Welford online standardization
///   (Welford, 1962 — "Note on a method for calculating corrected sums").
///   Required before RLS to prevent magnitude-driven weight explosion.
/// - RLS forgetting factor λ = 0.99 ≈ 100-sample effective window
///   (half-life = −1/log(λ) ≈ 99.5 samples, information-decay matching for
///   datasets without strong long-range dependence).
///
/// ```
/// use irithyll::{online_regressor, StreamingLearner};
///
/// let mut model = online_regressor();
/// for i in 0..20 {
///     model.train(&[i as f64, 1.0], i as f64 * 3.0 + 1.0);
/// }
/// let pred = model.predict(&[10.0, 1.0]);
/// assert!(pred.is_finite());
/// ```
pub fn online_regressor() -> Box<dyn StreamingLearner> {
    Box::new(pipe(normalizer()).learner(rls(0.99)))
}

/// Auto-tuned SGBT preset.
///
/// Creates an [`AutoTuner`] wrapping `Factory::sgbt(d_features)`.
///
/// The search space spans `learning_rate` ∈ [0.001, 0.3] (log-scaled per
/// Beygelzimer et al., 2015 "Online Gradient Boosting"), `n_steps` ∈ [10, 500],
/// `max_depth` ∈ [3, 10] (LightGBM/XGBoost canonical range), and
/// `lambda` ∈ [0.01, 10.0] (Friedman 2001 regularisation analysis).
/// Racing uses empirical Bernstein bounds (Maurer & Pontil, 2009) so no
/// fixed elimination thresholds are needed — the gate is data-derived.
///
/// For full control, call `AutoTuner::builder()` directly.
///
/// ```no_run
/// use irithyll::{tuned_sgbt, StreamingLearner};
///
/// let mut tuner = tuned_sgbt(5);
/// for i in 0..200 {
///     tuner.train(&[i as f64; 5], (i as f64).sin());
/// }
/// let pred = tuner.predict(&[1.0; 5]);
/// assert!(pred.is_finite());
/// ```
pub fn tuned_sgbt(d_features: usize) -> automl::AutoTuner {
    automl::AutoTuner::builder()
        .factory(automl::Factory::sgbt(d_features))
        .build()
        .expect("tuned_sgbt() preset: invalid parameters")
}

/// Across-family auto-regressor preset.
///
/// Races SGBT, SpikeNet, and streaming KAN (small default config) in a
/// multi-factory [`AutoTuner`].  The champion-challenger protocol (Bernstein
/// racing, Maurer & Pontil 2009) picks the best-performing family per stream
/// without a fixed elimination threshold.
///
/// The `n_features` argument is used to dimension the SGBT factory; KAN and
/// SpikeNet factories do not require it.  For streams where a single family
/// is known to dominate, prefer the targeted preset (`tuned_sgbt`,
/// `auto_tune(Factory::spikenet(...))`).
///
/// ```no_run
/// use irithyll::{auto_regressor, StreamingLearner};
///
/// let mut model = auto_regressor(5);
/// for i in 0..200 {
///     model.train(&[i as f64; 5], (i as f64).powi(2));
/// }
/// let pred = model.predict(&[3.0; 5]);
/// assert!(pred.is_finite());
/// ```
pub fn auto_regressor(n_features: usize) -> automl::AutoTuner {
    automl::AutoTuner::builder()
        .factory(automl::Factory::sgbt(n_features))
        .add_factory(automl::Factory::spike_net())
        .add_factory(automl::Factory::kan(n_features))
        .build()
        .expect("auto_regressor() preset: invalid parameters")
}

// ---------------------------------------------------------------------------
// Wave 7-1 tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod wave7_1_tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Wrapper factory tests
    // -----------------------------------------------------------------------

    #[test]
    fn drift_aware_wraps_arbitrary_learner() {
        let mut model = drift_aware(rls(0.99));
        for i in 0..50 {
            model.train(&[i as f64], i as f64 * 2.0);
        }
        let pred = model.predict(&[25.0]);
        assert!(
            pred.is_finite(),
            "drift_aware should produce finite predictions"
        );
    }

    #[test]
    fn projected_wraps_arbitrary_learner() {
        let mut model = projected(rls(0.99), 6, 3, 0.9999);
        for i in 0..30 {
            model.train(&[i as f64; 6], i as f64);
        }
        let pred = model.predict(&[1.0; 6]);
        assert!(
            pred.is_finite(),
            "projected should produce finite predictions"
        );
    }

    #[test]
    fn multiclass_routes_classes_correctly() {
        let mut clf = multiclass(rls(0.99), 3);
        for i in 0..90 {
            clf.train(&[(i % 3) as f64, 1.0], (i % 3) as f64);
        }
        let pred = clf.predict(&[1.0, 1.0]);
        assert!(
            (0.0..3.0).contains(&pred),
            "multiclass should return class index in [0, n_classes), got {pred}"
        );
    }

    #[test]
    fn binary_classifier_predicts_in_unit_interval() {
        // binary_classifier is defined in the original factory section above;
        // verify the wrapper contract here.
        let mut clf = binary_classifier(rls(0.99));
        for i in 0..30 {
            let sign = if i % 2 == 0 { 1.0_f64 } else { 0.0_f64 };
            clf.train(&[sign * 2.0 - 1.0, 1.0], sign);
        }
        let pred = clf.predict(&[1.0, 1.0]);
        assert!(
            pred == 0.0 || pred == 1.0,
            "binary_classifier should output {{0, 1}}, got {pred}"
        );
    }

    #[test]
    fn auto_tune_uses_supplied_factory() {
        let mut tuner = auto_tuner(automl::Factory::sgbt(3));
        for i in 0..50 {
            tuner.train(&[i as f64; 3], i as f64);
        }
        let pred = tuner.predict(&[1.0; 3]);
        assert!(
            pred.is_finite(),
            "auto_tuner should produce finite predictions"
        );
    }

    // -----------------------------------------------------------------------
    // Preset tests
    // -----------------------------------------------------------------------

    #[test]
    fn online_regressor_preset_compiles_and_predicts() {
        let mut model = online_regressor();
        for i in 0..30 {
            model.train(&[i as f64, 1.0], i as f64 * 3.0 + 1.0);
        }
        let pred = model.predict(&[10.0, 1.0]);
        assert!(
            pred.is_finite(),
            "online_regressor preset should produce finite predictions"
        );
    }

    #[test]
    fn tuned_sgbt_preset_runs_within_model_autotuner() {
        let mut tuner = tuned_sgbt(4);
        for i in 0..100 {
            tuner.train(&[i as f64; 4], (i as f64).sin());
        }
        let pred = tuner.predict(&[1.0; 4]);
        assert!(
            pred.is_finite(),
            "tuned_sgbt preset should produce finite predictions"
        );
    }
}
