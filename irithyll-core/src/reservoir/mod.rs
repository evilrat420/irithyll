//! Reservoir computing primitives for streaming temporal models.
//!
//! This module provides the core building blocks for reservoir computing
//! architectures (Echo State Networks, Next Generation Reservoir Computing)
//! that run in `no_std` environments with only `alloc`.
//!
//! # Components
//!
//! - [`Xorshift64Rng`] -- deterministic PRNG for reproducible reservoir initialization
//! - [`DelayBuffer`] -- circular buffer for time-delay embeddings
//! - [`HighDegreePolynomial`] -- polynomial monomial generator for nonlinear features
//! - [`CycleReservoir`] -- cycle/ring topology reservoir with leaky integration

pub mod cycle;
pub mod delay_buffer;
pub mod polynomial;
pub mod prng;

pub use cycle::CycleReservoir;
pub use delay_buffer::DelayBuffer;
pub use polynomial::HighDegreePolynomial;
pub use prng::Xorshift64Rng;
