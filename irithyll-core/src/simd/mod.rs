//! SIMD-accelerated math primitives for neural forward passes.
//!
//! Provides AVX2-accelerated dot product, matrix-vector multiply, and
//! activation functions with automatic runtime feature detection and
//! scalar fallback.
//!
//! On `x86_64` targets with the `std` feature enabled, these functions
//! use runtime `is_x86_feature_detected!("avx2")` to select AVX2 when
//! available. On all other targets (or without `std`), they fall back
//! to scalar implementations.

pub mod ops;

pub use ops::{simd_dot, simd_exp, simd_mat_vec, simd_sigmoid, simd_silu, simd_tanh};
