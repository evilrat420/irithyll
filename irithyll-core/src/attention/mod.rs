//! Unified streaming linear attention engine.
//!
//! Covers RetNet, Hawk, GLA, DeltaNet, GatedDeltaNet, RWKV, and mLSTM
//! as configuration variants of one core engine. Based on the State Space
//! Duality (Dao & Gu, ICML 2024) proving these are all equivalent structured
//! linear attention with different parameterizations.
//!
//! # Architecture
//!
//! All seven architectures share a common recurrence:
//!
//! ```text
//! S_t = decay_t * S_{t-1} + update_t    (state update)
//! o_t = query_fn(x_t, S_t)              (output)
//! ```
//!
//! The difference between architectures lies only in how decay, update, and
//! query are computed -- which this module captures via [`AttentionMode`].
//!
//! # Modules
//!
//! - [`config`] -- Configuration types and mode enum
//! - [`state`] -- Vector and matrix state containers
//! - [`gating`] -- Gate computation functions (fixed, sigmoid, exponential, LSTM)
//! - [`update_rules`] -- State update functions for each architecture variant
//! - [`multi_head`] -- Multi-head attention composing heads with output projection

pub mod config;
pub mod gating;
pub mod multi_head;
pub mod state;
pub mod update_rules;

pub use config::{AttentionConfig, AttentionMode};
pub use multi_head::MultiHeadAttention;
pub use state::AttentionState;

use alloc::vec::Vec;

/// Trait for streaming attention layers.
///
/// Implementors maintain internal state that evolves with each call to
/// [`forward`](AttentionLayer::forward). The state captures temporal patterns
/// from the input sequence without requiring storage of past observations.
///
/// # Thread Safety
///
/// All attention layers are `Send + Sync`, enabling use in async pipelines
/// and parallel prediction contexts.
pub trait AttentionLayer: Send + Sync {
    /// Process one input timestep and return the output vector.
    ///
    /// This advances the internal state by one step. The output dimension
    /// equals `d_model` as configured.
    ///
    /// # Arguments
    ///
    /// * `input` -- feature vector for this timestep (length `d_model`)
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;

    /// Get a flat view of the current state (all heads concatenated).
    fn state(&self) -> &[f64];

    /// Output dimension of this attention layer.
    fn output_dim(&self) -> usize;

    /// Reset all head states to zeros, as if no data has been seen.
    fn reset(&mut self);
}
