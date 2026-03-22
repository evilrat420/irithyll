//! Reservoir computing models for streaming temporal learning.
//!
//! This module provides two reservoir computing architectures that implement
//! [`StreamingLearner`](crate::learner::StreamingLearner):
//!
//! - **[`NextGenRC`]** (NG-RC) -- a reservoir-free approach that uses time-delay
//!   embeddings and polynomial features to create a nonlinear feature space,
//!   trained online via RLS. No random weights, fully deterministic.
//!
//! - **[`EchoStateNetwork`]** (ESN) -- a classic echo state network with a cycle
//!   reservoir topology and RLS readout, trained online one sample at a time.
//!
//! Both models support the standard `train`/`predict` streaming interface and
//! can be used in pipelines and stacking ensembles via `Box<dyn StreamingLearner>`.
//!
//! # NG-RC vs ESN
//!
//! | Property | NG-RC | ESN |
//! |----------|-------|-----|
//! | Random weights | No | Yes (reservoir) |
//! | Hyperparameters | k, s, degree | n_reservoir, spectral_radius, leak_rate |
//! | Warmup | k * s samples | configurable |
//! | Deterministic | Yes | Yes (given seed) |
//! | Best for | Short memory, polynomial nonlinearity | Long memory, complex dynamics |

pub mod esn;
pub mod esn_config;
pub mod esn_preprocessor;
pub mod ngrc;
pub mod ngrc_config;

pub use esn::EchoStateNetwork;
pub use esn_config::{ESNConfig, ESNConfigBuilder};
pub use esn_preprocessor::ESNPreprocessor;
pub use ngrc::NextGenRC;
pub use ngrc_config::{NGRCConfig, NGRCConfigBuilder};
