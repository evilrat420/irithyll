//! # Irithyll
//!
//! Streaming machine learning in Rust -- gradient boosted trees, kernel methods,
//! linear models, and composable pipelines, all learning one sample at a time.
//!
//! Irithyll provides 12+ streaming algorithms under one unified
//! [`StreamingLearner`] trait. The core is SGBT
//! ([Gunasekara et al., 2024](https://doi.org/10.1007/s10994-024-06517-y)),
//! but the library extends to kernel regression, RLS with confidence intervals,
//! Naive Bayes, Mondrian forests, streaming PCA, and composable pipelines.
//! Every algorithm processes samples one at a time with O(1) memory per model.
//!
//! ## Key Capabilities
//!
//! - **12+ streaming algorithms** -- SGBT, KRLS, RLS, linear SGD, Gaussian NB, Mondrian forests, and more
//! - **Composable pipelines** -- chain preprocessors and learners: `pipe(normalizer()).learner(sgbt(50, 0.01))`
//! - **Concept drift adaptation** -- automatic tree replacement via Page-Hinkley, ADWIN, or DDM
//! - **Kernel methods** -- [`KRLS`] with RBF, polynomial, and linear kernels + ALD sparsification
//! - **Confidence intervals** -- [`RecursiveLeastSquares::predict_interval`] for prediction uncertainty
//! - **Streaming PCA** -- [`CCIPCA`] for O(kd) dimensionality reduction without covariance matrices
//! - **Async streaming** -- tokio-native [`AsyncSGBT`] with bounded channels and concurrent prediction
//! - **Pluggable losses** -- squared, logistic, softmax, Huber, or custom via the [`Loss`] trait
//! - **Serialization** -- checkpoint/restore via JSON or bincode for zero-downtime deployments
//! - **Production-grade** -- SIMD acceleration, parallel training, Arrow/Parquet I/O, ONNX export
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `serde-json` | Yes | JSON model serialization |
//! | `serde-bincode` | No | Bincode serialization (compact, fast) |
//! | `parallel` | No | Rayon-based parallel tree training (`ParallelSGBT`) |
//! | `simd` | No | AVX2 histogram acceleration |
//! | `kmeans-binning` | No | K-means histogram binning strategy |
//! | `arrow` | No | Apache Arrow RecordBatch integration |
//! | `parquet` | No | Parquet file I/O |
//! | `onnx` | No | ONNX model export |
//! | `neural-leaves` | No | Experimental MLP leaf models |
//! | `full` | No | Enable all features |
//!
//! ## Quick Start
//!
//! ```no_run
//! use irithyll::{SGBTConfig, SGBT, Sample};
//!
//! let config = SGBTConfig::builder()
//!     .n_steps(100)
//!     .learning_rate(0.0125)
//!     .build()
//!     .unwrap();
//!
//! let mut model = SGBT::new(config);
//!
//! // Stream samples one at a time
//! let sample = Sample::new(vec![1.0, 2.0, 3.0], 0.5);
//! model.train_one(&sample);
//! let prediction = model.predict(&sample.features);
//! ```
//!
//! Or use factory functions for quick construction:
//!
//! ```no_run
//! use irithyll::{pipe, normalizer, sgbt, StreamingLearner};
//!
//! let mut model = pipe(normalizer()).learner(sgbt(50, 0.01));
//! model.train(&[100.0, 0.5], 42.0);
//! let pred = model.predict(&[100.0, 0.5]);
//! ```
//!
//! ## Algorithm
//!
//! The ensemble maintains `n_steps` boosting stages, each owning a streaming
//! Hoeffding tree and a drift detector. For each sample *(x, y)*:
//!
//! 1. Compute the ensemble prediction *F(x) = base + lr * sum(tree_s(x))*
//! 2. For each boosting step, compute gradient/hessian of the loss at the residual
//! 3. Update the tree's histogram accumulators and evaluate splits via Hoeffding bound
//! 4. Feed the standardized error to the drift detector
//! 5. If drift is detected, replace the tree with a fresh alternate
//!
//! This enables continuous learning without storing past data, with statistically
//! sound split decisions and automatic adaptation to distribution shifts.

pub mod error;
pub mod sample;

pub mod drift;
pub mod ensemble;
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
pub mod reservoir;
pub mod serde_support;
pub mod snn;
pub mod ssm;
pub mod time_series;
pub mod ttt;

#[cfg(feature = "arrow")]
pub mod arrow_support;

#[cfg(feature = "onnx")]
pub mod onnx_export;

pub mod export_embedded;

// Re-exports -- irithyll-core packed inference
pub use irithyll_core;
pub use irithyll_core::{EnsembleView, FormatError, PackedNode};
pub use irithyll_core::{PackedNodeI16, QuantizedEnsembleHeader, QuantizedEnsembleView};

// Re-exports -- core types
pub use drift::{DriftDetector, DriftSignal};
pub use ensemble::adaptive::AdaptiveSGBT;
pub use ensemble::bagged::BaggedSGBT;
pub use ensemble::config::{FeatureType, SGBTConfig, ScaleMode};
pub use ensemble::diagnostics::{DistributionalDiagnostics, EnsembleDiagnostics, TreeDiagnostics};
pub use ensemble::distributional::{
    DecomposedPrediction, DistributionalSGBT, GaussianPrediction, ModelDiagnostics, TreeDiagnostic,
};
pub use ensemble::moe_distributional::MoEDistributionalSGBT;
pub use ensemble::multi_target::MultiTargetSGBT;
pub use ensemble::multiclass::MulticlassSGBT;
pub use ensemble::quantile_regressor::QuantileRegressorSGBT;
pub use ensemble::{DynSGBT, SGBT};
pub use error::{ConfigError, IrithyllError};
pub use histogram::{BinnerKind, BinningStrategy};
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
pub use metrics::{ClassificationMetrics, FeatureImportance, MetricSet, RegressionMetrics};

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
pub use learners::{BernoulliNB, MultinomialNB};
pub use tree::hoeffding_classifier::HoeffdingTreeClassifier;

// Re-exports -- anomaly detection
pub use anomaly::hst::{AnomalyScore, HSTConfig, HalfSpaceTree};

// Re-exports -- streaming learner trait
pub use learner::{SGBTLearner, StreamingLearner};

// Re-exports -- preprocessing & pipeline
pub use pipeline::{Pipeline, PipelineBuilder, StreamingPreprocessor};
pub use preprocessing::{
    FeatureHasher, IncrementalNormalizer, MinMaxScaler, OneHotEncoder, OnlineFeatureSelector,
    PolynomialFeatures, TargetEncoder, CCIPCA,
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
    NextGenRC,
};

// Re-exports -- state space models
pub use ssm::{MambaConfig, MambaConfigBuilder, MambaPreprocessor, StreamingMamba};

// Re-exports -- spiking neural networks
pub use snn::{SpikeNet, SpikeNetConfig, SpikeNetConfigBuilder, SpikePreprocessor};

// Re-exports -- test-time training
pub use ttt::{StreamingTTT, TTTConfig, TTTConfigBuilder};

// Re-exports -- Kolmogorov-Arnold Networks
pub use kan::{KANConfig, KANConfigBuilder, StreamingKAN};

// Re-exports -- streaming linear attention
pub use attention::{
    AttentionPreprocessor, StreamingAttentionConfig, StreamingAttentionConfigBuilder,
    StreamingAttentionModel,
};

// Re-exports -- neural moe
pub use moe::{NeuralMoE, NeuralMoEBuilder, NeuralMoEConfig};

// Re-exports -- automl
#[allow(deprecated)]
pub use automl::{
    Algorithm, AttentionFactory, EsnFactory, Factory, MambaFactory, SgbtFactory, SpikeNetFactory,
};
pub use automl::{AutoMetric, AutoTuner, AutoTunerBuilder, AutoTunerConfig, ModelFactory};
pub use automl::{
    ConfigDiagnostics, DiagnosticAdaptor, DiagnosticSource, FeasibleRegion, WelfordRace,
};
pub use automl::{ConfigSampler, ConfigSpace, HyperConfig, HyperParam, RewardNormalizer};

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
pub fn sgbt(n_steps: usize, lr: f64) -> SGBTLearner {
    let config = SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(lr)
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
pub fn linear(lr: f64) -> StreamingLinearModel {
    StreamingLinearModel::new(lr)
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
        .build();
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
    lr: f64,
    scheduler: impl ensemble::lr_schedule::LRScheduler + 'static,
) -> AdaptiveSGBT {
    let config = SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(lr)
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
/// let mut pipeline = pipe(esn_preprocessor(30, 0.9)).learner(rls(0.999));
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
/// let mut model = streaming_ttt(16, 0.01);
/// model.train(&[1.0, 2.0], 3.0);
/// let pred = model.predict(&[1.0, 2.0]);
/// ```
pub fn streaming_ttt(d_model: usize, eta: f64) -> ttt::StreamingTTT {
    ttt::StreamingTTT::new(
        ttt::TTTConfig::builder()
            .d_model(d_model)
            .eta(eta)
            .build()
            .expect("streaming_ttt() factory: invalid parameters"),
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
pub fn streaming_kan(layer_sizes: &[usize], lr: f64) -> kan::StreamingKAN {
    kan::StreamingKAN::new(
        kan::KANConfig::builder()
            .layer_sizes(layer_sizes.to_vec())
            .lr(lr)
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
    automl::AutoTuner::builder().factory(factory).build()
}
