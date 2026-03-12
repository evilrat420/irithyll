//! Streaming anomaly detection algorithms.
//!
//! This module provides lightweight, single-pass anomaly detectors
//! suitable for data streams where the full dataset is never available.

pub mod hst;

pub use hst::{HSTConfig, HalfSpaceTree};
