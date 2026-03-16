//! Streaming evaluation protocols for online machine learning.
//!
//! This module provides evaluation frameworks that implement the standard
//! streaming ML evaluation paradigm: **test-then-train** (prequential evaluation).
//!
//! # Protocols
//!
//! | Evaluator | Description |
//! |-----------|-------------|
//! | [`PrequentialEvaluator`] | Gold standard: predict, record, train on every sample (Gama et al., 2009) |
//! | [`ProgressiveValidator`] | Configurable holdout: some samples are test-only (never trained on) |
//!
//! # Example
//!
//! ```
//! use irithyll::evaluation::{PrequentialEvaluator, PrequentialConfig};
//! use irithyll::learner::StreamingLearner;
//!
//! // See PrequentialEvaluator and ProgressiveValidator docs for full examples.
//! ```

mod prequential;
mod progressive;

pub use prequential::{PrequentialConfig, PrequentialEvaluator};
pub use progressive::{HoldoutStrategy, ProgressiveValidator};
