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
//! For discrete-time processing, we discretize via Zero-Order Hold (ZOH),
//! bilinear (Tustin), or exponential-trapezoidal transform. The **selective**
//! variant (Mamba) makes B, C, and Delta input-dependent.
//!
//! # Modules
//!
//! - [`diagonal`] -- Non-selective diagonal SSM with fixed parameters
//! - [`selective`] -- Mamba-style selective SSM with input-dependent projections
//! - [`selective_bd`] -- Block-diagonal SSM with dense per-block A matrices
//! - [`selective_v3`] -- Mamba-3 variants:
//!   - [`SelectiveSSMv3`] -- Tustin complex (original)
//!   - [`SelectiveSSMv3Exp`] -- exp-trapezoidal 3-term + data-dependent λ_t (paper spec)
//!   - [`SelectiveSSMv3Mimo`] -- true rank-R MIMO with matrix-valued state H ∈ R^{N×P}
//! - [`complex_diag`] -- Standalone complex diagonal SSM primitive (reusable)
//! - [`norm`] -- BCNorm: RMSNorm for B/C stabilization in Mamba-3
//! - [`discretize`] -- ZOH, bilinear (Tustin), and exp-trapezoidal discretization
//! - [`init`] -- A-matrix initialization strategies (Mamba, S4D real/complex)
//! - [`projection`] -- Linear algebra helpers and PRNG for weight initialization

pub mod complex_diag;
pub mod diagonal;
pub mod discretize;
pub mod init;
pub mod norm;
pub mod projection;
pub mod selective;
pub mod selective_bd;
pub mod selective_v3;

pub use complex_diag::{ComplexDiagonalSSM, DiscretizeMethod};
pub use diagonal::DiagonalSSM;
pub use norm::BCNorm;
pub use selective::SelectiveSSM;
pub use selective_bd::SelectiveSSMBD;
pub use selective_v3::{SelectiveSSMv3, SelectiveSSMv3Exp, SelectiveSSMv3Mimo};

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
