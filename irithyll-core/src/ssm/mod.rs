//! State Space Models for streaming temporal feature extraction.
//!
//! This module implements diagonal and selective (Mamba-style) state space models
//! for processing sequential data in a streaming fashion. SSMs maintain hidden
//! state that evolves with each input timestep, capturing temporal dependencies
//! without storing past observations.
//!
//! # Architecture
//!
//! The continuous-time SSM is defined by:
//!
//! ```text
//! h'(t) = A * h(t) + B * x(t)     (state evolution)
//! y(t)  = C * h(t) + D * x(t)     (output equation)
//! ```
//!
//! For discrete-time processing, we discretize via Zero-Order Hold (ZOH) or
//! bilinear transform. The **selective** variant (Mamba) makes B, C, and the
//! discretization step Delta input-dependent, enabling content-aware filtering.
//!
//! # Modules
//!
//! - [`diagonal`] -- Non-selective diagonal SSM with fixed parameters
//! - [`selective`] -- Mamba-style selective SSM with input-dependent projections
//! - [`discretize`] -- ZOH and bilinear discretization methods
//! - [`init`] -- A-matrix initialization strategies (Mamba, S4D)
//! - [`projection`] -- Linear algebra helpers and PRNG for weight initialization

pub mod diagonal;
pub mod discretize;
pub mod init;
pub mod projection;
pub mod selective;

pub use diagonal::DiagonalSSM;
pub use selective::SelectiveSSM;

use alloc::vec::Vec;

/// Trait for SSM layers that process sequential data one timestep at a time.
///
/// Implementors maintain internal hidden state that evolves with each call to
/// [`forward`](SSMLayer::forward). The hidden state captures temporal patterns
/// from the input sequence without requiring storage of past observations.
///
/// # Thread Safety
///
/// All SSM layers are `Send + Sync`, enabling use in async pipelines and
/// parallel prediction contexts.
pub trait SSMLayer: Send + Sync {
    /// Process one input timestep and return the output vector.
    ///
    /// This advances the internal hidden state by one step. The output
    /// dimension equals the input dimension for selective SSMs, or 1 for
    /// scalar diagonal SSMs.
    ///
    /// # Arguments
    ///
    /// * `input` -- feature vector for this timestep
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;

    /// Get a reference to the current hidden state.
    fn state(&self) -> &[f64];

    /// Output dimension of this SSM layer.
    fn output_dim(&self) -> usize;

    /// Reset hidden state to zeros, as if no data has been seen.
    fn reset(&mut self);
}
