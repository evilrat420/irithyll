//! Continual learning wrappers for streaming neural models.
//!
//! Wraps any [`StreamingLearner`] with drift-aware continual learning strategies
//! that prevent catastrophic forgetting and maintain plasticity.
//!
//! The primary type is [`ContinualLearner`], which wraps an opaque streaming model
//! and uses prediction error to drive drift detection. When drift is detected, the
//! inner model is reset so it can adapt to the new regime.
//!
//! # Example
//!
//! ```
//! use irithyll::continual::{ContinualLearner, continual};
//! use irithyll::{linear, StreamingLearner};
//! use irithyll_core::drift::pht::PageHinkleyTest;
//!
//! let mut cl = continual(linear(0.01))
//!     .with_drift_detector(PageHinkleyTest::new());
//!
//! for i in 0..100 {
//!     cl.train(&[i as f64], i as f64 * 2.0);
//! }
//! let pred = cl.predict(&[50.0]);
//! assert!(pred.is_finite());
//! ```

pub mod continual_wrapper;

pub use continual_wrapper::{continual, ContinualLearner};

// Re-export core continual types for convenience.
pub use irithyll_core::continual::{ContinualStrategy, DriftMask, StreamingEWC};
