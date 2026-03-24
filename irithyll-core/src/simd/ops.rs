//! Core SIMD-accelerated operations: dot product and matrix-vector multiply.
//!
//! These are the two hottest primitives across SSM, ESN, and attention forward
//! passes. AVX2 processes 4 `f64` values per cycle, giving up to ~4x throughput
//! on aligned inner loops.
//!
//! # Architecture
//!
//! ```text
//! Public API (safe)           Internal dispatch
//! ─────────────────           ─────────────────
//! simd_dot(a, b)       ──►   avx2::dot_avx2     (x86_64 + AVX2 detected)
//!                      └──►  dot_scalar          (fallback)
//!
//! simd_mat_vec(w,x,..) ──►   avx2::mat_vec_avx2 (x86_64 + AVX2 detected)
//!                      └──►  mat_vec_scalar      (fallback)
//! ```

// Runtime detection macro — only available with std.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
use std::is_x86_feature_detected;

// ---------------------------------------------------------------------------
// Scalar fallbacks (always available, no_std compatible)
// ---------------------------------------------------------------------------

/// Scalar dot product of two slices.
///
/// Computes `sum(a[i] * b[i])` for `i` in `0..min(a.len(), b.len())`.
#[inline]
fn dot_scalar(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar matrix-vector multiply: `out[i] = dot(w[i*cols..], x)`.
///
/// `w` is a `rows x cols` row-major matrix, `x` is a `cols`-vector,
/// `out` is a `rows`-vector (must be pre-allocated).
#[inline]
fn mat_vec_scalar(w: &[f64], x: &[f64], _rows: usize, cols: usize, out: &mut [f64]) {
    for (row, out_i) in out.iter_mut().enumerate() {
        let start = row * cols;
        let mut sum = 0.0;
        for j in 0..cols {
            sum += w[start + j] * x[j];
        }
        *out_i = sum;
    }
}

// ---------------------------------------------------------------------------
// AVX2 implementations (x86_64 + std only)
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", feature = "std"))]
mod avx2 {
    /// AVX2-accelerated dot product: processes 4 f64 values per iteration.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available at runtime (checked via
    /// `is_x86_feature_detected!("avx2")`).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_avx2(a: &[f64], b: &[f64]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let n = a.len().min(b.len());
        let chunks = n / 4;
        let remainder = n % 4;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        // SAFETY: AVX2 availability verified by caller. All pointer arithmetic
        // stays within slice bounds (chunks * 4 <= n).
        unsafe {
            let mut acc = _mm256_setzero_pd();

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm256_loadu_pd(a_ptr.add(offset));
                let vb = _mm256_loadu_pd(b_ptr.add(offset));
                acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
            }

            // Horizontal sum of 4 f64 lanes: [a0, a1, a2, a3]
            let hi128 = _mm256_extractf128_pd(acc, 1); // [a2, a3]
            let lo128 = _mm256_castpd256_pd128(acc); // [a0, a1]
            let pair = _mm_add_pd(lo128, hi128); // [a0+a2, a1+a3]
            let high64 = _mm_unpackhi_pd(pair, pair); // [a1+a3, a1+a3]
            let total = _mm_add_sd(pair, high64); // low lane = a0+a1+a2+a3
            let mut scalar_sum = _mm_cvtsd_f64(total);

            // Handle remainder with scalar tail.
            let base = chunks * 4;
            for i in 0..remainder {
                scalar_sum += *a_ptr.add(base + i) * *b_ptr.add(base + i);
            }

            scalar_sum
        }
    }

    /// AVX2-accelerated matrix-vector multiply.
    ///
    /// Each row is computed as a SIMD dot product of `w[row*cols..]` with `x`.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - AVX2 is available at runtime
    /// - `w.len() >= rows * cols`, `x.len() >= cols`, `out.len() >= rows`
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn mat_vec_avx2(
        w: &[f64],
        x: &[f64],
        _rows: usize,
        cols: usize,
        out: &mut [f64],
    ) {
        for (row, out_i) in out.iter_mut().enumerate() {
            let row_start = row * cols;
            // SAFETY: caller ensures w has at least rows*cols elements.
            // dot_avx2 uses min(a.len(), b.len()) so slicing is safe.
            unsafe {
                *out_i = dot_avx2(&w[row_start..row_start + cols], &x[..cols]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public safe dispatch functions
// ---------------------------------------------------------------------------

/// SIMD-accelerated dot product with runtime feature detection.
///
/// Uses AVX2 on x86_64 (with `std` feature) when available, falls back to
/// scalar otherwise.
///
/// Returns the dot product of `a` and `b`, processing up to the shorter
/// slice's length.
///
/// # Examples
///
/// ```
/// use irithyll_core::simd::simd_dot;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// assert!((simd_dot(&a, &b) - 32.0).abs() < 1e-12);
/// ```
pub fn simd_dot(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: we just checked for AVX2 support.
            return unsafe { avx2::dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// SIMD-accelerated matrix-vector multiply with runtime feature detection.
///
/// Computes `out[i] = sum_j w[i*cols + j] * x[j]` for each row.
/// Uses AVX2 on x86_64 (with `std` feature) when available, falls back to
/// scalar otherwise.
///
/// # Panics
///
/// Panics if `w.len() < rows * cols`, `out.len() < rows`, or `x.len() < cols`.
///
/// # Examples
///
/// ```
/// use irithyll_core::simd::simd_mat_vec;
///
/// // 2x3 matrix times 3-vector
/// let w = [1.0, 2.0, 3.0,  4.0, 5.0, 6.0];
/// let x = [1.0, 1.0, 1.0];
/// let mut out = [0.0; 2];
/// simd_mat_vec(&w, &x, 2, 3, &mut out);
/// assert!((out[0] - 6.0).abs() < 1e-12);   // 1+2+3
/// assert!((out[1] - 15.0).abs() < 1e-12);  // 4+5+6
/// ```
pub fn simd_mat_vec(w: &[f64], x: &[f64], rows: usize, cols: usize, out: &mut [f64]) {
    assert!(
        w.len() >= rows * cols,
        "simd_mat_vec: w.len()={} < rows*cols={}",
        w.len(),
        rows * cols
    );
    assert!(
        out.len() >= rows,
        "simd_mat_vec: out.len()={} < rows={}",
        out.len(),
        rows
    );
    assert!(
        x.len() >= cols,
        "simd_mat_vec: x.len()={} < cols={}",
        x.len(),
        cols
    );

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: bounds checked above, AVX2 detected.
            unsafe {
                avx2::mat_vec_avx2(w, x, rows, cols, out);
            }
            return;
        }
    }
    mat_vec_scalar(w, x, rows, cols, out);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    // Simple deterministic PRNG for test data generation.
    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn next_f64(&mut self) -> f64 {
            // Map to [-1, 1) range for interesting test values
            (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64) * 2.0 - 1.0
        }

        fn fill_vec(&mut self, n: usize) -> Vec<f64> {
            (0..n).map(|_| self.next_f64()).collect()
        }
    }

    // -------------------------------------------------------------------
    // Dot product tests
    // -------------------------------------------------------------------

    #[test]
    fn dot_empty_returns_zero() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        assert_eq!(simd_dot(&a, &b), 0.0, "dot of empty slices should be 0");
    }

    #[test]
    fn dot_single_element() {
        let a = [3.0];
        let b = [4.0];
        assert!(
            (simd_dot(&a, &b) - 12.0).abs() < 1e-12,
            "dot([3], [4]) should be 12, got {}",
            simd_dot(&a, &b)
        );
    }

    #[test]
    fn dot_known_result() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = simd_dot(&a, &b);
        assert!(
            (result - 32.0).abs() < 1e-12,
            "dot([1,2,3], [4,5,6]) should be 32, got {}",
            result
        );
    }

    #[test]
    fn dot_large_matches_scalar() {
        let mut rng = TestRng::new(42);
        let a = rng.fill_vec(1000);
        let b = rng.fill_vec(1000);

        let simd_result = simd_dot(&a, &b);
        let scalar_result = dot_scalar(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-9,
            "1000-element dot: SIMD={} vs scalar={}, diff={}",
            simd_result,
            scalar_result,
            (simd_result - scalar_result).abs()
        );
    }

    #[test]
    fn dot_mismatched_lengths() {
        // Should use the shorter length
        let a = [1.0, 2.0, 3.0, 999.0];
        let b = [4.0, 5.0, 6.0];
        let result = simd_dot(&a, &b);
        assert!(
            (result - 32.0).abs() < 1e-12,
            "mismatched lengths should use min, expected 32, got {}",
            result
        );
    }

    #[test]
    fn dot_non_aligned_length() {
        // 7 elements: 1 full AVX2 chunk (4) + 3 remainder
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = simd_dot(&a, &b);
        assert!(
            (result - 28.0).abs() < 1e-12,
            "dot of [1..7] with [1..1] should be 28, got {}",
            result
        );
    }

    #[test]
    fn dot_negative_values() {
        let a = [-1.0, -2.0, -3.0, -4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        // -4 + -6 + -6 + -4 = -20
        let result = simd_dot(&a, &b);
        assert!(
            (result - (-20.0)).abs() < 1e-12,
            "expected -20, got {}",
            result
        );
    }

    #[test]
    fn dot_orthogonal_vectors() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let result = simd_dot(&a, &b);
        assert!(
            result.abs() < 1e-12,
            "orthogonal vectors should have dot=0, got {}",
            result
        );
    }

    // -------------------------------------------------------------------
    // Matrix-vector multiply tests
    // -------------------------------------------------------------------

    #[test]
    fn mat_vec_identity_like() {
        // 3x3 identity matrix times [1, 2, 3] = [1, 2, 3]
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let x = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        simd_mat_vec(&w, &x, 3, 3, &mut out);
        assert!(
            (out[0] - 1.0).abs() < 1e-12,
            "identity row 0: expected 1, got {}",
            out[0]
        );
        assert!(
            (out[1] - 2.0).abs() < 1e-12,
            "identity row 1: expected 2, got {}",
            out[1]
        );
        assert!(
            (out[2] - 3.0).abs() < 1e-12,
            "identity row 2: expected 3, got {}",
            out[2]
        );
    }

    #[test]
    fn mat_vec_known_result() {
        // 2x3 matrix:
        // [1 2 3]   [1]   [1+4+9]   [14]
        // [4 5 6] * [2] = [4+10+18] = [32]
        //           [3]
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 2.0, 3.0];
        let mut out = [0.0; 2];
        simd_mat_vec(&w, &x, 2, 3, &mut out);
        assert!(
            (out[0] - 14.0).abs() < 1e-12,
            "row 0: expected 14, got {}",
            out[0]
        );
        assert!(
            (out[1] - 32.0).abs() < 1e-12,
            "row 1: expected 32, got {}",
            out[1]
        );
    }

    #[test]
    fn mat_vec_large_matches_scalar() {
        let mut rng = TestRng::new(7777);
        let rows = 100;
        let cols = 100;
        let w = rng.fill_vec(rows * cols);
        let x = rng.fill_vec(cols);
        let mut out_simd = vec![0.0; rows];
        let mut out_scalar = vec![0.0; rows];

        simd_mat_vec(&w, &x, rows, cols, &mut out_simd);
        mat_vec_scalar(&w, &x, rows, cols, &mut out_scalar);

        for i in 0..rows {
            assert!(
                (out_simd[i] - out_scalar[i]).abs() < 1e-9,
                "row {}: SIMD={} vs scalar={}, diff={}",
                i,
                out_simd[i],
                out_scalar[i],
                (out_simd[i] - out_scalar[i]).abs()
            );
        }
    }

    #[test]
    fn mat_vec_single_row() {
        // 1xN is just a dot product
        let w = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x = [2.0, 2.0, 2.0, 2.0, 2.0];
        let mut out = [0.0; 1];
        simd_mat_vec(&w, &x, 1, 5, &mut out);
        // 2+4+6+8+10 = 30
        assert!(
            (out[0] - 30.0).abs() < 1e-12,
            "single-row mat_vec should be dot product, expected 30, got {}",
            out[0]
        );
    }

    #[test]
    fn mat_vec_single_element() {
        let w = [7.0];
        let x = [3.0];
        let mut out = [0.0; 1];
        simd_mat_vec(&w, &x, 1, 1, &mut out);
        assert!(
            (out[0] - 21.0).abs() < 1e-12,
            "1x1 mat_vec: 7*3=21, got {}",
            out[0]
        );
    }

    // -------------------------------------------------------------------
    // Panic tests
    // -------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "simd_mat_vec: w.len()")]
    fn mat_vec_panics_w_too_short() {
        let w = [1.0, 2.0]; // need 2*3=6
        let x = [1.0, 2.0, 3.0];
        let mut out = [0.0; 2];
        simd_mat_vec(&w, &x, 2, 3, &mut out);
    }

    #[test]
    #[should_panic(expected = "simd_mat_vec: out.len()")]
    fn mat_vec_panics_out_too_short() {
        let w = [1.0; 6];
        let x = [1.0; 3];
        let mut out = [0.0; 1]; // need 2
        simd_mat_vec(&w, &x, 2, 3, &mut out);
    }

    #[test]
    #[should_panic(expected = "simd_mat_vec: x.len()")]
    fn mat_vec_panics_x_too_short() {
        let w = [1.0; 6];
        let x = [1.0; 2]; // need 3
        let mut out = [0.0; 2];
        simd_mat_vec(&w, &x, 2, 3, &mut out);
    }

    // -------------------------------------------------------------------
    // Platform-specific test
    // -------------------------------------------------------------------

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    #[test]
    fn simd_available_on_x86() {
        // On modern x86_64, AVX2 should be available.
        // This test verifies the runtime detection path doesn't panic.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let result = simd_dot(&a, &b);
        // 8+14+18+20+20+18+14+8 = 120
        assert!(
            (result - 120.0).abs() < 1e-12,
            "8-element dot product should be 120, got {}",
            result
        );

        // Also verify AVX2 is actually detected (informational).
        if is_x86_feature_detected!("avx2") {
            // AVX2 path was used — good.
        }
    }
}
