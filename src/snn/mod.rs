//! Spiking Neural Networks for streaming machine learning.
//!
//! This module wraps the `irithyll-core` fixed-point SNN with f64 interfaces,
//! automatic input scaling, and implementations of [`StreamingLearner`](crate::learner::StreamingLearner) and
//! [`StreamingPreprocessor`](crate::pipeline::StreamingPreprocessor) for seamless integration with irithyll pipelines.
//!
//! # Components
//!
//! - [`SpikeNet`] -- f64-interfaced SNN implementing [`StreamingLearner`](crate::learner::StreamingLearner)
//! - [`SpikeNetConfig`] / [`SpikeNetConfigBuilder`] -- configuration with validation
//! - [`SpikePreprocessor`] -- SNN as a feature transformer implementing [`StreamingPreprocessor`](crate::pipeline::StreamingPreprocessor)
//!
//! # Example
//!
//! ```
//! use irithyll::snn::{SpikeNet, SpikeNetConfig};
//! use irithyll::StreamingLearner;
//!
//! let config = SpikeNetConfig::builder()
//!     .n_hidden(32)
//!     .learning_rate(0.005)
//!     .build()
//!     .unwrap();
//!
//! let mut model = SpikeNet::new(config);
//! model.train(&[0.5, -0.3, 0.8], 1.0);
//! let pred = model.predict(&[0.5, -0.3, 0.8]);
//! ```
//!
//! # As a Pipeline Preprocessor
//!
//! ```
//! use irithyll::snn::SpikePreprocessor;
//! use irithyll::pipeline::{Pipeline, StreamingPreprocessor};
//! use irithyll::learners::StreamingLinearModel;
//! use irithyll::StreamingLearner;
//!
//! let mut pipeline = Pipeline::builder()
//!     .pipe(SpikePreprocessor::new(16, 42))
//!     .learner(StreamingLinearModel::new(0.01));
//!
//! pipeline.train(&[1.0, 2.0, 3.0], 5.0);
//! let pred = pipeline.predict(&[1.0, 2.0, 3.0]);
//! ```

pub mod spike_preprocessor;
pub mod spikenet;
pub mod spikenet_config;

pub use spike_preprocessor::SpikePreprocessor;
pub use spikenet::SpikeNet;
pub use spikenet_config::{SpikeNetConfig, SpikeNetConfigBuilder};
