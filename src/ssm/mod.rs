//! Streaming Mamba (selective state space model) for temporal ML pipelines.
//!
//! This module provides [`StreamingMamba`], a streaming machine learning model
//! that uses a Mamba-style selective SSM as its temporal feature extractor,
//! feeding into a Recursive Least Squares (RLS) readout layer. It integrates
//! with irithyll's [`StreamingLearner`](crate::learner::StreamingLearner) trait
//! and [`StreamingPreprocessor`](crate::pipeline::StreamingPreprocessor) trait.
//!
//! # Architecture
//!
//! ```text
//! input features ──→ [SelectiveSSM] ──→ temporal features ──→ [RLS] ──→ prediction
//!   (d_in)              (d_in state)        (d_in)            (1)
//! ```
//!
//! The SSM processes each feature vector as a timestep, maintaining per-channel
//! hidden state that captures temporal dynamics. The RLS readout learns a linear
//! mapping from the SSM's output to the target variable.
//!
//! # Components
//!
//! - [`StreamingMamba`] -- full model implementing `StreamingLearner`
//! - [`MambaPreprocessor`] -- SSM-only preprocessor implementing `StreamingPreprocessor`
//! - [`MambaConfig`] / [`MambaConfigBuilder`] -- validated configuration
//!
//! # Example
//!
//! ```
//! use irithyll::ssm::{StreamingMamba, MambaConfig};
//! use irithyll::learner::StreamingLearner;
//!
//! let config = MambaConfig::builder()
//!     .d_in(4)
//!     .n_state(16)
//!     .build()
//!     .unwrap();
//!
//! let mut model = StreamingMamba::new(config);
//! model.train(&[1.0, 2.0, 3.0, 4.0], 5.0);
//! let pred = model.predict(&[1.0, 2.0, 3.0, 4.0]);
//! assert!(pred.is_finite());
//! ```

pub mod mamba;
pub mod mamba_config;
pub mod mamba_preprocessor;

pub use mamba::StreamingMamba;
pub use mamba_config::{MambaConfig, MambaConfigBuilder};
pub use mamba_preprocessor::MambaPreprocessor;
