//! Linear algebra helpers and PRNG for SSM weight initialization.
//!
//! This module provides the core math operations needed by SSM layers:
//!
//! - **Matrix-vector multiplication** for input projections (W_B * x, W_C * x, etc.)
//! - **Dot product** for output computation (C^T * h)
//! - **Activation functions** (softplus, sigmoid) for the selective mechanism
//! - **Xorshift64 PRNG** for deterministic weight initialization without external deps
//!
//! All operations are implemented in pure Rust with no external dependencies
//! beyond `libm` (via `crate::math`), making them suitable for `no_std` targets.

use crate::math;

/// Row-major matrix-vector multiply: out = W * x.
///
/// `w` is a `rows x cols` row-major matrix, `x` is a `cols`-vector,
/// `out` is a `rows`-vector (must be pre-allocated).
///
/// Delegates to [`crate::simd::simd_mat_vec`] for AVX2 acceleration
/// when available.
///
/// # Panics
///
/// Debug-asserts that `w.len() == rows * cols`, `x.len() == cols`,
/// and `out.len() == rows`.
#[inline]
pub fn mat_vec(w: &[f64], x: &[f64], rows: usize, cols: usize, out: &mut [f64]) {
    debug_assert_eq!(w.len(), rows * cols, "w must be rows*cols");
    debug_assert_eq!(x.len(), cols, "x must have cols elements");
    debug_assert_eq!(out.len(), rows, "out must have rows elements");
    crate::simd::simd_mat_vec(w, x, rows, cols, out);
}

/// Dot product of two equal-length slices.
///
/// Delegates to [`crate::simd::simd_dot`] for AVX2 acceleration
/// when available.
///
/// # Panics
///
/// Debug-asserts that `a.len() == b.len()`.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dot product requires equal lengths");
    crate::simd::simd_dot(a, b)
}

/// Numerically stable softplus: `ln(1 + exp(x))`.
///
/// Delegates to [`crate::math::softplus`].
#[inline]
pub fn softplus(x: f64) -> f64 {
    math::softplus(x)
}

/// Numerically stable sigmoid: `1 / (1 + exp(-x))`.
///
/// Delegates to [`crate::math::sigmoid`].
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    math::sigmoid(x)
}

/// Xorshift64 pseudo-random number generator.
///
/// A fast, deterministic PRNG suitable for weight initialization. Not
/// cryptographically secure, but provides good statistical properties
/// for ML weight sampling.
///
/// # Example
///
/// ```
/// use irithyll_core::ssm::projection::Xorshift64;
///
/// let mut rng = Xorshift64(12345);
/// let val = rng.next_f64();   // uniform in [0, 1)
/// let normal = rng.next_normal(); // standard normal via Box-Muller
/// ```
pub struct Xorshift64(pub u64);

impl Xorshift64 {
    /// Generate the next random u64 value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Generate a uniform random f64 in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Use upper 53 bits for full mantissa precision
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Generate a standard normal random value via Box-Muller transform.
    ///
    /// Generates two uniform samples and converts to a standard normal.
    /// Discards the second value for simplicity.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        // Box-Muller: need two independent uniforms in (0, 1)
        let u1 = loop {
            let u = self.next_f64();
            if u > 0.0 {
                break u;
            }
        };
        let u2 = self.next_f64();
        math::sqrt(-2.0 * math::ln(u1)) * math::cos(2.0 * core::f64::consts::PI * u2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn mat_vec_identity() {
        // 2x2 identity matrix times [3, 4] = [3, 4]
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![3.0, 4.0];
        let mut out = vec![0.0; 2];
        mat_vec(&w, &x, 2, 2, &mut out);
        assert!(
            math::abs(out[0] - 3.0) < 1e-12,
            "expected 3.0, got {}",
            out[0]
        );
        assert!(
            math::abs(out[1] - 4.0) < 1e-12,
            "expected 4.0, got {}",
            out[1]
        );
    }

    #[test]
    fn mat_vec_rectangular() {
        // 3x2 matrix times 2-vector
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0];
        let mut out = vec![0.0; 3];
        mat_vec(&w, &x, 3, 2, &mut out);
        assert!(math::abs(out[0] - 3.0) < 1e-12, "row 0: 1+2=3");
        assert!(math::abs(out[1] - 7.0) < 1e-12, "row 1: 3+4=7");
        assert!(math::abs(out[2] - 11.0) < 1e-12, "row 2: 5+6=11");
    }

    #[test]
    fn dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot(&a, &b);
        assert!(
            math::abs(result - 32.0) < 1e-12,
            "expected 32.0, got {}",
            result
        );
    }

    #[test]
    fn dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = dot(&a, &b);
        assert!(
            math::abs(result) < 1e-12,
            "orthogonal vectors should have dot=0"
        );
    }

    #[test]
    fn softplus_large_positive() {
        // For x >> 0, softplus(x) ~ x
        let result = softplus(50.0);
        assert!(
            math::abs(result - 50.0) < 1e-10,
            "softplus(50) should be ~50, got {}",
            result
        );
    }

    #[test]
    fn softplus_large_negative() {
        // For x << 0, softplus(x) ~ exp(x) ~ 0
        let result = softplus(-50.0);
        assert!(
            (0.0..1e-20).contains(&result),
            "softplus(-50) should be ~0, got {}",
            result
        );
    }

    #[test]
    fn softplus_zero() {
        let result = softplus(0.0);
        let expected = math::ln(2.0);
        assert!(
            math::abs(result - expected) < 1e-12,
            "softplus(0) should be ln(2)={}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn softplus_always_positive() {
        let values = [-10.0, -1.0, 0.0, 1.0, 10.0];
        for &x in &values {
            let result = softplus(x);
            assert!(
                result > 0.0,
                "softplus({}) should be > 0, got {}",
                x,
                result
            );
        }
    }

    #[test]
    fn sigmoid_range() {
        // Moderate values: strict (0, 1) check
        let moderate = [-10.0, -1.0, 0.0, 1.0, 10.0];
        for &x in &moderate {
            let result = sigmoid(x);
            assert!(
                result > 0.0 && result < 1.0,
                "sigmoid({}) should be in (0,1), got {}",
                x,
                result
            );
        }
        // Extreme values: allow saturation to 0.0 or 1.0 due to f64 precision
        let extreme = [-100.0, 100.0];
        for &x in &extreme {
            let result = sigmoid(x);
            assert!(
                (0.0..=1.0).contains(&result),
                "sigmoid({}) should be in [0,1], got {}",
                x,
                result
            );
        }
    }

    #[test]
    fn sigmoid_zero() {
        let result = sigmoid(0.0);
        assert!(
            math::abs(result - 0.5) < 1e-12,
            "sigmoid(0) should be 0.5, got {}",
            result
        );
    }

    #[test]
    fn sigmoid_symmetry() {
        let x = 3.0;
        let s_pos = sigmoid(x);
        let s_neg = sigmoid(-x);
        assert!(
            math::abs(s_pos + s_neg - 1.0) < 1e-12,
            "sigmoid(x) + sigmoid(-x) should be 1.0"
        );
    }

    #[test]
    fn xorshift_deterministic() {
        let mut rng1 = Xorshift64(42);
        let mut rng2 = Xorshift64(42);
        for _ in 0..100 {
            assert_eq!(
                rng1.next_u64(),
                rng2.next_u64(),
                "same seed should produce same sequence"
            );
        }
    }

    #[test]
    fn xorshift_f64_in_unit_interval() {
        let mut rng = Xorshift64(12345);
        for i in 0..1000 {
            let val = rng.next_f64();
            assert!(
                (0.0..1.0).contains(&val),
                "next_f64() sample {} = {} not in [0,1)",
                i,
                val
            );
        }
    }

    #[test]
    fn xorshift_normal_distribution() {
        // Sample many normals, check mean and variance are approximately 0 and 1
        let mut rng = Xorshift64(9999);
        let n = 10000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let x = rng.next_normal();
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        assert!(
            math::abs(mean) < 0.05,
            "normal mean should be ~0, got {}",
            mean
        );
        assert!(
            math::abs(variance - 1.0) < 0.1,
            "normal variance should be ~1, got {}",
            variance
        );
    }

    #[test]
    fn mat_vec_single_element() {
        let w = vec![7.0];
        let x = vec![3.0];
        let mut out = vec![0.0];
        mat_vec(&w, &x, 1, 1, &mut out);
        assert!(math::abs(out[0] - 21.0) < 1e-12, "7*3=21");
    }

    #[test]
    fn softplus_moderate_values() {
        // For moderate x, test against direct computation
        let x = 5.0;
        let expected = math::ln(1.0 + math::exp(5.0));
        let result = softplus(x);
        assert!(
            math::abs(result - expected) < 1e-10,
            "softplus(5) expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn xorshift_different_seeds_differ() {
        let mut rng1 = Xorshift64(1);
        let mut rng2 = Xorshift64(2);
        let seq1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let seq2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_ne!(
            seq1, seq2,
            "different seeds should produce different sequences"
        );
    }
}
