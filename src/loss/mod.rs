//! Loss functions for gradient boosting.
//!
//! Each loss provides gradient and hessian computations used by the boosting
//! loop to compute pseudo-residuals for tree fitting.

pub mod expectile;
pub mod huber;
pub mod logistic;
pub mod quantile;
pub mod softmax;
pub mod squared;

// Re-export core loss types
pub use irithyll_core::loss::{Loss, LossType};
