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
pub mod loss;
pub mod metrics;
pub mod stream;
pub mod tree;

pub mod anomaly;
pub mod explain;
pub mod learner;
pub mod learners;
pub mod pipeline;
pub mod preprocessing;
pub mod serde_support;

#[cfg(feature = "arrow")]
pub mod arrow_support;

#[cfg(feature = "onnx")]
pub mod onnx_export;

// Re-exports -- core types
pub use drift::{DriftDetector, DriftSignal};
pub use ensemble::adaptive::AdaptiveSGBT;
pub use ensemble::bagged::BaggedSGBT;
pub use ensemble::config::{FeatureType, SGBTConfig};
pub use ensemble::distributional::{
    DecomposedPrediction, DistributionalSGBT, GaussianPrediction, ModelDiagnostics, TreeDiagnostic,
};
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
pub use metrics::conformal::AdaptiveConformalInterval;
pub use metrics::ewma::{EwmaClassificationMetrics, EwmaRegressionMetrics};
pub use metrics::rolling::{RollingClassificationMetrics, RollingRegressionMetrics};
pub use metrics::{ClassificationMetrics, FeatureImportance, MetricSet, RegressionMetrics};

// Re-exports -- anomaly detection
pub use anomaly::hst::{AnomalyScore, HSTConfig, HalfSpaceTree};

// Re-exports -- streaming learner trait
pub use learner::{SGBTLearner, StreamingLearner};

// Re-exports -- preprocessing & pipeline
pub use pipeline::{Pipeline, PipelineBuilder, StreamingPreprocessor};
pub use preprocessing::{IncrementalNormalizer, OnlineFeatureSelector, CCIPCA};

// Re-exports -- learning rate scheduling
pub use ensemble::lr_schedule::LRScheduler;

// Re-exports -- streaming learners
pub use learners::{
    GaussianNB, Kernel, LinearKernel, LocallyWeightedRegression, MondrianForest, PolynomialKernel,
    RBFKernel, RecursiveLeastSquares, StreamingLinearModel, StreamingPolynomialRegression, KRLS,
};

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
