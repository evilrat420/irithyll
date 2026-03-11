//! SIMD-accelerated operations for histogram computation.
//!
//! Provides element-wise vector subtraction and horizontal sum using AVX2
//! intrinsics on x86_64, with scalar fallbacks for other architectures or
//! when AVX2 is unavailable at runtime.
//!
//! All public functions are safe; `unsafe` is confined to internal intrinsic
//! calls behind `#[target_feature]` gates.

// ---------------------------------------------------------------------------
// subtract_f64
// ---------------------------------------------------------------------------

/// Element-wise `out[i] = a[i] - b[i]` for f64 slices.
///
/// Used by the histogram subtraction trick to derive sibling histograms.
///
/// # Panics
/// Panics if `a`, `b`, and `out` do not all have the same length.
pub fn subtract_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    assert_eq!(
        a.len(),
        b.len(),
        "subtract_f64: a and b must have the same length"
    );
    assert_eq!(
        a.len(),
        out.len(),
        "subtract_f64: a and out must have the same length"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: we checked that AVX2 is available and all slices have
            // equal length, so the inner function's pointer arithmetic is valid.
            unsafe {
                subtract_f64_avx2(a, b, out);
            }
            return;
        }
    }

    subtract_f64_scalar(a, b, out);
}

/// Scalar fallback for `subtract_f64`.
#[inline]
fn subtract_f64_scalar(a: &[f64], b: &[f64], out: &mut [f64]) {
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

/// AVX2-accelerated path: processes 4 f64 values per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn subtract_f64_avx2(a: &[f64], b: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm256_loadu_pd(a_ptr.add(offset));
        let vb = _mm256_loadu_pd(b_ptr.add(offset));
        let vr = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(out_ptr.add(offset), vr);
    }

    // Scalar tail for remaining elements.
    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        *out_ptr.add(idx) = *a_ptr.add(idx) - *b_ptr.add(idx);
    }
}

// ---------------------------------------------------------------------------
// sum_f64
// ---------------------------------------------------------------------------

/// Horizontal sum of all elements in `slice`.
///
/// Used by `total_gradient()` and `total_hessian()`.
pub fn sum_f64(slice: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is available and we only read within bounds.
            return unsafe { sum_f64_avx2(slice) };
        }
    }

    sum_f64_scalar(slice)
}

/// Scalar fallback for `sum_f64`.
#[inline]
fn sum_f64_scalar(slice: &[f64]) -> f64 {
    slice.iter().sum()
}

/// AVX2-accelerated horizontal sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_f64_avx2(slice: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let len = slice.len();
    let chunks = len / 4;
    let remainder = len % 4;
    let ptr = slice.as_ptr();

    let mut acc = _mm256_setzero_pd();

    for i in 0..chunks {
        let offset = i * 4;
        let v = _mm256_loadu_pd(ptr.add(offset));
        acc = _mm256_add_pd(acc, v);
    }

    // Horizontal reduce: acc = [a0, a1, a2, a3]
    // Extract high 128 bits and add to low 128 bits.
    let hi128 = _mm256_extractf128_pd(acc, 1); // [a2, a3]
    let lo128 = _mm256_castpd256_pd128(acc); // [a0, a1]
    let sum128 = _mm_add_pd(lo128, hi128); // [a0+a2, a1+a3]

    // Final horizontal add of the two f64 lanes.
    let shuf = _mm_unpackhi_pd(sum128, sum128); // [a1+a3, a1+a3]
    let total = _mm_add_sd(sum128, shuf); // low lane = a0+a1+a2+a3
    let mut result: f64 = _mm_cvtsd_f64(total);

    // Scalar tail.
    let tail_start = chunks * 4;
    for i in 0..remainder {
        result += *ptr.add(tail_start + i);
    }

    result
}

// ---------------------------------------------------------------------------
// subtract_u64
// ---------------------------------------------------------------------------

/// Element-wise `out[i] = a[i].saturating_sub(b[i])` for u64 slices.
///
/// AVX2 lacks native u64 saturating subtraction, so this is implemented as
/// a straightforward scalar loop. The function signature keeps the API
/// consistent with the f64 variants.
///
/// # Panics
/// Panics if `a`, `b`, and `out` do not all have the same length.
pub fn subtract_u64(a: &[u64], b: &[u64], out: &mut [u64]) {
    assert_eq!(
        a.len(),
        b.len(),
        "subtract_u64: a and b must have the same length"
    );
    assert_eq!(
        a.len(),
        out.len(),
        "subtract_u64: a and out must have the same length"
    );

    for i in 0..a.len() {
        out[i] = a[i].saturating_sub(b[i]);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // subtract_f64 tests
    // -----------------------------------------------------------------------

    #[test]
    fn subtract_f64_basic() {
        let a = [10.0, 20.0, 30.0, 40.0, 50.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0; 5];
        subtract_f64(&a, &b, &mut out);
        assert_eq!(out, vec![9.0, 18.0, 27.0, 36.0, 45.0]);
    }

    #[test]
    fn subtract_f64_empty() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        let mut out: Vec<f64> = vec![];
        subtract_f64(&a, &b, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn subtract_f64_single_element() {
        let a = [42.5];
        let b = [17.3];
        let mut out = vec![0.0; 1];
        subtract_f64(&a, &b, &mut out);
        assert!((out[0] - 25.2).abs() < 1e-12);
    }

    #[test]
    fn subtract_f64_length_7_non_aligned() {
        let a: Vec<f64> = (1..=7).map(|x| x as f64 * 10.0).collect();
        let b: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        let mut out = vec![0.0; 7];
        subtract_f64(&a, &b, &mut out);
        for i in 0..7 {
            let expected = (i + 1) as f64 * 10.0 - (i + 1) as f64;
            assert!((out[i] - expected).abs() < 1e-12, "mismatch at index {i}");
        }
    }

    #[test]
    fn subtract_f64_length_13_non_aligned() {
        let a: Vec<f64> = (0..13).map(|x| x as f64 * 3.0).collect();
        let b: Vec<f64> = (0..13).map(|x| x as f64 * 1.5).collect();
        let mut out = vec![0.0; 13];
        subtract_f64(&a, &b, &mut out);
        for i in 0..13 {
            let expected = i as f64 * 1.5;
            assert!((out[i] - expected).abs() < 1e-12, "mismatch at index {i}");
        }
    }

    #[test]
    fn subtract_f64_large_256_matches_scalar() {
        let a: Vec<f64> = (0..256).map(|x| (x as f64) * 1.1 + 0.7).collect();
        let b: Vec<f64> = (0..256).map(|x| (x as f64) * 0.3 + 0.2).collect();
        let mut out_simd = vec![0.0; 256];
        let mut out_scalar = vec![0.0; 256];

        subtract_f64(&a, &b, &mut out_simd);
        subtract_f64_scalar(&a, &b, &mut out_scalar);

        for i in 0..256 {
            assert!(
                (out_simd[i] - out_scalar[i]).abs() < 1e-12,
                "mismatch at index {i}: simd={} scalar={}",
                out_simd[i],
                out_scalar[i]
            );
        }
    }

    #[test]
    fn subtract_f64_negative_results() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 4];
        subtract_f64(&a, &b, &mut out);
        assert_eq!(out, vec![-4.0, -4.0, -4.0, -4.0]);
    }

    // -----------------------------------------------------------------------
    // sum_f64 tests
    // -----------------------------------------------------------------------

    #[test]
    fn sum_f64_basic() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((sum_f64(&v) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn sum_f64_empty() {
        let v: [f64; 0] = [];
        assert_eq!(sum_f64(&v), 0.0);
    }

    #[test]
    fn sum_f64_single_element() {
        let v = [42.0];
        assert!((sum_f64(&v) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn sum_f64_length_7_non_aligned() {
        let v: Vec<f64> = (1..=7).map(|x| x as f64).collect();
        // 1+2+3+4+5+6+7 = 28
        assert!((sum_f64(&v) - 28.0).abs() < 1e-12);
    }

    #[test]
    fn sum_f64_length_13_non_aligned() {
        let v: Vec<f64> = (1..=13).map(|x| x as f64).collect();
        // sum 1..=13 = 91
        assert!((sum_f64(&v) - 91.0).abs() < 1e-12);
    }

    #[test]
    fn sum_f64_large_256_matches_scalar() {
        let v: Vec<f64> = (0..256).map(|x| (x as f64) * 0.7 + 0.1).collect();
        let simd_result = sum_f64(&v);
        let scalar_result = sum_f64_scalar(&v);
        assert!(
            (simd_result - scalar_result).abs() < 1e-9,
            "simd={simd_result} scalar={scalar_result}"
        );
    }

    #[test]
    fn sum_f64_negative_values() {
        let v = [-1.0, -2.0, -3.0, -4.0, -5.0];
        assert!((sum_f64(&v) - (-15.0)).abs() < 1e-12);
    }

    #[test]
    fn sum_f64_mixed_positive_negative() {
        let v = [10.0, -3.0, 5.0, -7.0, 2.0, -1.0];
        // 10 - 3 + 5 - 7 + 2 - 1 = 6
        assert!((sum_f64(&v) - 6.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // subtract_u64 tests
    // -----------------------------------------------------------------------

    #[test]
    fn subtract_u64_basic() {
        let a = [10u64, 20, 30, 40, 50];
        let b = [1u64, 2, 3, 4, 5];
        let mut out = vec![0u64; 5];
        subtract_u64(&a, &b, &mut out);
        assert_eq!(out, vec![9, 18, 27, 36, 45]);
    }

    #[test]
    fn subtract_u64_empty() {
        let a: [u64; 0] = [];
        let b: [u64; 0] = [];
        let mut out: Vec<u64> = vec![];
        subtract_u64(&a, &b, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn subtract_u64_single_element() {
        let a = [100u64];
        let b = [42u64];
        let mut out = vec![0u64; 1];
        subtract_u64(&a, &b, &mut out);
        assert_eq!(out[0], 58);
    }

    #[test]
    fn subtract_u64_saturation() {
        // b > a should saturate to 0, not underflow.
        let a = [5u64, 0, 3, 10];
        let b = [10u64, 1, 100, 10];
        let mut out = vec![0u64; 4];
        subtract_u64(&a, &b, &mut out);
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    #[test]
    fn subtract_u64_length_7_non_aligned() {
        let a: Vec<u64> = (10..17).collect();
        let b: Vec<u64> = (1..8).collect();
        let mut out = vec![0u64; 7];
        subtract_u64(&a, &b, &mut out);
        // 10-1=9, 11-2=9, ..., 16-7=9
        for val in &out {
            assert_eq!(*val, 9);
        }
    }

    #[test]
    fn subtract_u64_large_256_matches() {
        let a: Vec<u64> = (0..256).map(|x| x * 3 + 100).collect();
        let b: Vec<u64> = (0..256).map(|x| x * 2 + 50).collect();
        let mut out = vec![0u64; 256];
        subtract_u64(&a, &b, &mut out);
        for i in 0..256 {
            let expected = (i as u64 * 3 + 100).saturating_sub(i as u64 * 2 + 50);
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn subtract_u64_mixed_saturation() {
        // Some elements saturate, some don't.
        let a = [100u64, 5, 200, 0, 50];
        let b = [50u64, 10, 100, 0, 51];
        let mut out = vec![0u64; 5];
        subtract_u64(&a, &b, &mut out);
        assert_eq!(out, vec![50, 0, 100, 0, 0]);
    }

    #[test]
    fn subtract_u64_max_values() {
        let a = [u64::MAX, u64::MAX];
        let b = [u64::MAX, 0];
        let mut out = vec![0u64; 2];
        subtract_u64(&a, &b, &mut out);
        assert_eq!(out, vec![0, u64::MAX]);
    }

    // -----------------------------------------------------------------------
    // Panic tests (length mismatches)
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "subtract_f64: a and b must have the same length")]
    fn subtract_f64_panics_on_length_mismatch_ab() {
        let a = [1.0, 2.0];
        let b = [1.0];
        let mut out = vec![0.0; 2];
        subtract_f64(&a, &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "subtract_f64: a and out must have the same length")]
    fn subtract_f64_panics_on_length_mismatch_out() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0];
        let mut out = vec![0.0; 3];
        subtract_f64(&a, &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "subtract_u64: a and b must have the same length")]
    fn subtract_u64_panics_on_length_mismatch() {
        let a = [1u64, 2];
        let b = [1u64];
        let mut out = vec![0u64; 2];
        subtract_u64(&a, &b, &mut out);
    }
}
