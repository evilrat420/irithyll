//! Streaming Test-Time Training (TTT) layers with prediction-directed fast weights.
//!
//! The hidden state is a linear model with weights W, updated by gradient
//! descent at every time step. After warmup, the fast weight update is
//! directed by the readout's prediction error — making W directly minimize
//! prediction loss rather than task-agnostic reconstruction. During warmup,
//! falls back to self-supervised reconstruction.
//!
//! This prediction-directed approach enables fast adaptation on non-stationary
//! data: when the target function shifts, the prediction error immediately
//! redirects the fast weight updates toward the new relationship.
//!
//! Optionally includes Titans-style momentum and weight decay for
//! non-stationary streaming environments.
//!
//! # Architecture
//!
//! ```text
//! x_t → [TTT Layer: project → reconstruct → update W → query] → z_t → [RLS Readout] → ŷ_t
//!                          ↑ prediction_feedback ←────────────────────────── pred_error
//! ```
//!
//! # References
//!
//! - Sun et al. (2024) "Learning to (Learn at Test Time)" ICML
//! - Liu et al. (2026) "TTT with 1 step + frozen projections = RLS" (canonical streaming form is single-head)
//! - Behrouz et al. (2025) "Titans: Learning to Memorize at Test Time"

mod config;
mod layer;
mod model;

pub use config::{TTTConfig, TTTConfigBuilder};
pub use model::StreamingTTT;
