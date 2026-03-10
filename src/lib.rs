//! # Irithyll
//!
//! Streaming Gradient Boosted Trees for evolving data streams.
//!
//! Irithyll implements the SGBT algorithm (Gunasekara et al., 2024) in pure Rust,
//! providing incremental gradient boosted tree ensembles that learn one sample at a time.
//! Trees are built using Hoeffding-bound split decisions and automatically replaced
//! when concept drift is detected.
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
