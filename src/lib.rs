//! # Irithyll
//!
//! Streaming Gradient Boosted Trees for evolving data streams.
//!
//! Irithyll implements the SGBT algorithm
//! ([Gunasekara et al., 2024](https://doi.org/10.1007/s10994-024-06517-y))
//! in pure Rust, providing incremental gradient boosted tree ensembles that learn
//! one sample at a time. Trees are built using Hoeffding-bound split decisions and
//! automatically replaced when concept drift is detected, making the model suitable
//! for non-stationary environments where the data distribution shifts over time.
//!
//! ## Key Capabilities
//!
//! - **True online learning** -- `train_one()` processes samples individually, no batching
//! - **Concept drift adaptation** -- automatic tree replacement via Page-Hinkley, ADWIN, or DDM
//! - **Async streaming** -- tokio-native [`AsyncSGBT`] with bounded channels and concurrent prediction
//! - **Pluggable losses** -- squared, logistic, softmax, Huber, or custom via the [`Loss`] trait
//! - **Multi-class** -- one-vs-rest committees with softmax normalization ([`MulticlassSGBT`])
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

pub mod loss;
pub mod histogram;
pub mod tree;
pub mod drift;
pub mod ensemble;
pub mod stream;
pub mod metrics;

pub mod serde_support;

#[cfg(feature = "arrow")]
pub mod arrow_support;

#[cfg(feature = "onnx")]
pub mod onnx_export;

// Re-exports — core types
pub use error::IrithyllError;
pub use sample::Sample;
pub use ensemble::config::SGBTConfig;
pub use ensemble::SGBT;
pub use ensemble::multiclass::MulticlassSGBT;
pub use loss::Loss;
pub use drift::{DriftDetector, DriftSignal};
pub use tree::StreamingTree;
pub use histogram::BinningStrategy;

// Re-exports — parallel (feature-gated)
#[cfg(feature = "parallel")]
pub use ensemble::parallel::ParallelSGBT;

// Re-exports — async streaming
pub use stream::{AsyncSGBT, Predictor, SampleSender, Prediction, PredictionStream};

// Re-exports — metrics
pub use metrics::{RegressionMetrics, ClassificationMetrics, FeatureImportance, MetricSet};
