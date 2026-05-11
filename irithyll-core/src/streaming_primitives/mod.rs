//! Shared streaming primitives for irithyll-core model building blocks.
//!
//! This module collects ~6 reusable low-level operations that appear across
//! multiple streaming ML models. They are free functions and lightweight structs
//! — not a trait hierarchy — so each model can import exactly what it needs
//! without generic-bound tax or dyn-dispatch overhead.
//!
//! # Primitives
//!
//! | Module | Type / Function | Purpose |
//! |--------|-----------------|---------|
//! | [`welford`] | [`WelfordNormalizer`] | Online per-feature mean/variance, Bessel-corrected |
//! | [`gate_head`] | [`GateHead`] | Learned sigmoid gate `g = sigmoid(W·x + b)`, online SGD |
//! | [`rank1_update`] | [`rank1_outer_update`] / [`rank1_outer_update_inplace`] | In-place `M += α · x · yᵀ` |
//! | [`bounded_mix`] | [`softplus_softmax_mix`] | Smooth probability distribution for Log-Linear Attention |
//! | [`tanh_inplace`](mod@self::tanh_inplace) | [`tanh_inplace`](fn@self::tanh_inplace) | Element-wise tanh bounding before RLS readout |
//!
//! # Placement of `BoundedRLS`
//!
//! `RecursiveLeastSquares` lives in the main `irithyll` crate (std-only).
//! The `BoundedRLS` readout wrapper therefore lives in `irithyll::learners::bounded_rls`
//! rather than here. See that module for the `predict_clipped` helper.
//!
//! # Design rationale (from S1b synthesis)
//!
//! Shared primitives that are 5–20 LOC operations belong in a module of free
//! functions, not a trait. Using a trait would:
//! - Add generic-bound tax at every call site.
//! - Force `no_std`-incompatible types out of the public API surface.
//! - Accrue dyn-dispatch complexity on top of the existing `StreamingLearner` +
//!   `DiagnosticSource` + `AttentionLayer` trait stack.
//!
//! Rule: if X is a primitive, make it a module function. Only make X a trait
//! if it's a protocol with state and dispatch.
//!
//! # `no_std` / `alloc` gating
//!
//! - Free functions (`tanh_inplace`, `rank1_outer_update`, `softplus_softmax_mix`)
//!   have zero allocation and are available with or without `alloc`.
//! - Struct types (`WelfordNormalizer`, `GateHead`) require heap allocation and
//!   are gated behind `#[cfg(feature = "alloc")]`.

pub mod bounded_mix;
pub mod gate_head;
pub mod rank1_update;
pub mod tanh_inplace;
pub mod welford;

// Free functions — available without alloc.
pub use bounded_mix::softplus_softmax_mix;
pub use rank1_update::{rank1_outer_update, rank1_outer_update_inplace};
pub use tanh_inplace::tanh_inplace;

// Struct types — require alloc.
#[cfg(feature = "alloc")]
pub use gate_head::GateHead;

#[cfg(feature = "alloc")]
pub use welford::WelfordNormalizer;
