//! Online projection learning for streaming models.
//!
//! This module provides PAST-based subspace tracking that learns optimal
//! input projections from prediction residuals in a single streaming pass.
//!
//! The main entry point is [`ProjectedLearner`], which wraps any
//! [`StreamingLearner`](crate::learner::StreamingLearner) with automatic
//! input normalization and adaptive dimensionality reduction via the PAST
//! algorithm (Yang, 1995).

mod projected_learner;
mod subspace_tracker;

pub use projected_learner::{ProjectedLearner, ProjectionConfig, ProjectionConfigBuilder};
pub use subspace_tracker::SubspaceTracker;
