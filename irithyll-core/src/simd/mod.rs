//! SIMD-accelerated math primitives for neural forward passes.
//!
//! Provides AVX2-accelerated dot product and matrix-vector multiply
//! with automatic runtime feature detection and scalar fallback.
//!
//! On `x86_64` targets with the `std` feature enabled, these functions
//! use runtime `is_x86_feature_detected!("avx2")` to select AVX2 when
//! available. On all other targets (or without `std`), they fall back
//! to scalar implementations.

pub mod ops;

pub use ops::{simd_dot, simd_mat_vec};
