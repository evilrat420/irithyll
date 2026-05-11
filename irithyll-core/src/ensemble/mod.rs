//! SGBT ensemble orchestrator -- the core boosting loop.
//!
//! Implements Streaming Gradient Boosted Trees (Gunasekara et al., 2024):
//! a sequence of boosting steps, each owning a streaming tree and drift detector,
//! with automatic tree replacement when concept drift is detected.
//!
//! # Algorithm
//!
//! For each incoming sample `(x, y)`:
//! 1. Compute the current ensemble prediction: `F(x) = base + lr * Σ tree_s(x)`
//! 2. For each boosting step `s = 1..N`:
//!    - Compute gradient `g = loss.gradient(y, current_pred)`
//!    - Compute hessian `h = loss.hessian(y, current_pred)`
//!    - Feed `(x, g, h)` to tree `s` (which internally uses weighted squared loss)
//!    - Update `current_pred += lr * tree_s.predict(x)`
//! 3. The ensemble adapts incrementally, with each tree targeting the residual
//!    of all preceding trees.

pub mod adaptive;
pub mod adaptive_forest;
pub mod bagged;
pub mod config;
pub mod core;
pub mod diagnostics;
pub mod distributional;
pub mod lr_schedule;
pub mod moe;
pub mod moe_distributional;
pub mod multi_target;
pub mod multiclass;
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub mod parallel;
pub mod quantile_regressor;
pub mod replacement;
pub mod stacked;
pub mod step;
pub mod variants;

use alloc::boxed::Box;

use crate::loss::Loss;
#[allow(unused_imports)] // Used in doc links + tests
use crate::sample::Sample;
#[cfg(feature = "_serde_support")]
use crate::tree::builder::TreeConfig;

// Re-export core types from the core module
pub use core::SGBT;

/// Type alias for an SGBT model using dynamic (boxed) loss dispatch.
///
/// Use this when the loss function is determined at runtime (e.g., when
/// deserializing a model from JSON where the loss type is stored as a tag).
///
/// For compile-time loss dispatch (preferred for performance), use
/// `SGBT<LogisticLoss>`, `SGBT<HuberLoss>`, etc.
pub type DynSGBT = SGBT<Box<dyn Loss>>;

// Stub for methods to be implemented.
// This file serves as the hub for the ensemble module.
// Actual implementation methods will be added across multiple files
// in a future wave (accessors, inference, inspection, train, etc.).
