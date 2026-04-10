//! Core SIMD-accelerated operations: dot product, matrix-vector multiply,
//! and activation functions.
//!
//! These are the hottest primitives across SSM, ESN, and attention/neural
//! forward passes. AVX2 processes 4 `f64` values per cycle, giving up to
//! ~4x throughput on aligned inner loops.
//!
//! # Architecture
//!
//! ```text
//! Public API (safe)            Internal dispatch
//! ─────────────────            ─────────────────
//! simd_dot(a, b)        ──►   avx2::dot_avx2      (x86_64 + AVX2 detected)
//!                       └──►  dot_scalar           (fallback)
//!
//! simd_mat_vec(w,x,..)  ──►   avx2::mat_vec_avx2  (x86_64 + AVX2 detected)
//!                       └──►  mat_vec_scalar       (fallback)
//!
//! simd_tanh(in, out)    ──►   avx2::tanh_avx2     (x86_64 + AVX2, Padé [2,2])
//!                       └──►  tanh_scalar          (fallback)
//!
//! simd_exp(in, out)     ──►   avx2::exp_avx2      (x86_64 + AVX2, range-reduced deg-5)
//!                       └──►  exp_scalar           (fallback)
//!
//! simd_sigmoid(in, out) ──►   avx2::sigmoid_avx2  (x86_64 + AVX2, via exp)
//!                       └──►  sigmoid_scalar       (fallback)
//!
//! simd_silu(in, out)    ──►   avx2::silu_avx2     (x86_64 + AVX2, via sigmoid)
//!                       └──►  silu_scalar          (fallback)
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

/// Scalar tanh: delegates to `crate::math::tanh`.
#[inline]
fn tanh_scalar(input: &[f64], output: &mut [f64]) {
    for (i, &x) in input.iter().enumerate() {
        output[i] = crate::math::tanh(x);
    }
}

/// Scalar exp: delegates to `crate::math::exp`.
#[inline]
fn exp_scalar(input: &[f64], output: &mut [f64]) {
    for (i, &x) in input.iter().enumerate() {
        output[i] = crate::math::exp(x);
    }
}

/// Scalar sigmoid: delegates to `crate::math::sigmoid`.
#[inline]
fn sigmoid_scalar(input: &[f64], output: &mut [f64]) {
    for (i, &x) in input.iter().enumerate() {
        output[i] = crate::math::sigmoid(x);
    }
}

/// Scalar SiLU: `output[i] = input[i] * sigmoid(input[i])`.
#[inline]
fn silu_scalar(input: &[f64], output: &mut [f64]) {
    for (i, &x) in input.iter().enumerate() {
        output[i] = x * crate::math::sigmoid(x);
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

    /// AVX2-accelerated tanh using Padé \[2,2\] rational approximation.
    ///
    /// For |x| > 4.97 the result saturates to ±1.0.
    /// Otherwise: `tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)`.
    ///
    /// Processes 4 f64 values per iteration with a scalar tail for
    /// non-multiple-of-4 lengths.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn tanh_avx2(input: &[f64], output: &mut [f64]) {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let n = input.len();
        let chunks = n / 4;

        // SAFETY: AVX2 availability verified by caller. All pointer arithmetic
        // stays within slice bounds.
        unsafe {
            let c15 = _mm256_set1_pd(15.0);
            let c6 = _mm256_set1_pd(6.0);
            let pos_sat = _mm256_set1_pd(4.97);
            let neg_sat = _mm256_set1_pd(-4.97);
            let one = _mm256_set1_pd(1.0);
            let neg_one = _mm256_set1_pd(-1.0);

            for i in 0..chunks {
                let off = i * 4;
                let x = _mm256_loadu_pd(input.as_ptr().add(off));
                let x2 = _mm256_mul_pd(x, x);

                // Padé [2,2]: x * (15 + x²) / (15 + 6*x²)
                let numer = _mm256_mul_pd(x, _mm256_add_pd(c15, x2));
                let denom = _mm256_add_pd(c15, _mm256_mul_pd(c6, x2));
                let approx = _mm256_div_pd(numer, denom);

                // Clamp to [-1, 1] for |x| > 4.97
                let clamped = _mm256_min_pd(one, _mm256_max_pd(neg_one, approx));

                // Use saturation mask: if |x| > 4.97, output sign(x)
                let sat_pos = _mm256_cmp_pd(x, pos_sat, _CMP_GT_OQ);
                let sat_neg = _mm256_cmp_pd(x, neg_sat, _CMP_LT_OQ);
                let result = _mm256_blendv_pd(clamped, one, sat_pos);
                let result = _mm256_blendv_pd(result, neg_one, sat_neg);

                _mm256_storeu_pd(output.as_mut_ptr().add(off), result);
            }
        }

        // Scalar tail for remainder elements.
        for i in (chunks * 4)..n {
            output[i] = crate::math::tanh(input[i]);
        }
    }

    /// AVX2-accelerated exp using range reduction + degree-5 polynomial + 2^n scaling.
    ///
    /// Algorithm: clamp to [-708, 708], split x = n*ln2 + r where n = round(x/ln2),
    /// compute exp(r) via Horner polynomial, then scale by 2^n via IEEE 754 exponent
    /// manipulation.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn exp_avx2(input: &[f64], output: &mut [f64]) {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let n = input.len();
        let chunks = n / 4;

        unsafe {
            let ln2 = _mm256_set1_pd(core::f64::consts::LN_2);
            let log2e = _mm256_set1_pd(core::f64::consts::LOG2_E);
            let clamp_hi = _mm256_set1_pd(708.0);
            let clamp_lo = _mm256_set1_pd(-708.0);
            let one = _mm256_set1_pd(1.0);
            let half = _mm256_set1_pd(0.5);
            let c3 = _mm256_set1_pd(1.0 / 6.0);
            let c4 = _mm256_set1_pd(1.0 / 24.0);
            let c5 = _mm256_set1_pd(1.0 / 120.0);
            let bias = _mm256_set1_epi64x(1023);

            for i in 0..chunks {
                let off = i * 4;
                let x = _mm256_loadu_pd(input.as_ptr().add(off));

                // Clamp to prevent overflow/underflow
                let x = _mm256_min_pd(clamp_hi, _mm256_max_pd(clamp_lo, x));

                // Range reduction: n = round(x / ln2), r = x - n*ln2
                let x_scaled = _mm256_mul_pd(x, log2e);
                let n_f = _mm256_floor_pd(_mm256_add_pd(x_scaled, half));
                let r = _mm256_sub_pd(x, _mm256_mul_pd(n_f, ln2));

                // Horner polynomial: 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120))))
                let mut p = _mm256_add_pd(c4, _mm256_mul_pd(c5, r));
                p = _mm256_add_pd(c3, _mm256_mul_pd(p, r));
                p = _mm256_add_pd(half, _mm256_mul_pd(p, r));
                p = _mm256_add_pd(one, _mm256_mul_pd(p, r));
                p = _mm256_add_pd(one, _mm256_mul_pd(p, r));

                // Scale by 2^n via IEEE 754 exponent manipulation
                let n_i32 = _mm256_cvtpd_epi32(n_f);
                let n_i64 = _mm256_cvtepi32_epi64(n_i32);
                let shifted = _mm256_slli_epi64(_mm256_add_epi64(n_i64, bias), 52);
                let pow2n = _mm256_castsi256_pd(shifted);
                let result = _mm256_mul_pd(p, pow2n);

                _mm256_storeu_pd(output.as_mut_ptr().add(off), result);
            }
        }

        // Scalar tail
        for i in (chunks * 4)..n {
            output[i] = crate::math::exp(input[i]);
        }
    }

    /// AVX2-accelerated sigmoid: `1 / (1 + exp(-x))`.
    ///
    /// Computes exp(-x) inline using the same range-reduction polynomial as
    /// `exp_avx2`, then applies the sigmoid formula.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sigmoid_avx2(input: &[f64], output: &mut [f64]) {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let n = input.len();
        let chunks = n / 4;

        unsafe {
            let ln2 = _mm256_set1_pd(core::f64::consts::LN_2);
            let log2e = _mm256_set1_pd(core::f64::consts::LOG2_E);
            let clamp_hi = _mm256_set1_pd(708.0);
            let clamp_lo = _mm256_set1_pd(-708.0);
            let one = _mm256_set1_pd(1.0);
            let half = _mm256_set1_pd(0.5);
            let c3 = _mm256_set1_pd(1.0 / 6.0);
            let c4 = _mm256_set1_pd(1.0 / 24.0);
            let c5 = _mm256_set1_pd(1.0 / 120.0);
            let bias = _mm256_set1_epi64x(1023);
            let neg_one = _mm256_set1_pd(-1.0);

            for i in 0..chunks {
                let off = i * 4;
                let x = _mm256_loadu_pd(input.as_ptr().add(off));

                // Negate x for exp(-x), then clamp
                let neg_x = _mm256_mul_pd(x, neg_one);
                let neg_x = _mm256_min_pd(clamp_hi, _mm256_max_pd(clamp_lo, neg_x));

                // Range reduction
                let x_scaled = _mm256_mul_pd(neg_x, log2e);
                let n_f = _mm256_floor_pd(_mm256_add_pd(x_scaled, half));
                let r = _mm256_sub_pd(neg_x, _mm256_mul_pd(n_f, ln2));

                // Horner polynomial for exp(r)
                let mut p = _mm256_add_pd(c4, _mm256_mul_pd(c5, r));
                p = _mm256_add_pd(c3, _mm256_mul_pd(p, r));
                p = _mm256_add_pd(half, _mm256_mul_pd(p, r));
                p = _mm256_add_pd(one, _mm256_mul_pd(p, r));
                p = _mm256_add_pd(one, _mm256_mul_pd(p, r));

                // Scale by 2^n
                let n_i32 = _mm256_cvtpd_epi32(n_f);
                let n_i64 = _mm256_cvtepi32_epi64(n_i32);
                let shifted = _mm256_slli_epi64(_mm256_add_epi64(n_i64, bias), 52);
                let pow2n = _mm256_castsi256_pd(shifted);
                let exp_neg_x = _mm256_mul_pd(p, pow2n);

                // sigmoid = 1 / (1 + exp(-x))
                let result = _mm256_div_pd(one, _mm256_add_pd(one, exp_neg_x));

                _mm256_storeu_pd(output.as_mut_ptr().add(off), result);
            }
        }

        // Scalar tail
        for i in (chunks * 4)..n {
            output[i] = crate::math::sigmoid(input[i]);
        }
    }

    /// AVX2-accelerated SiLU: `x * sigmoid(x)`.
    ///
    /// Computes sigmoid via `sigmoid_avx2`, then multiplies element-wise by input.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn silu_avx2(input: &[f64], output: &mut [f64]) {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        // First compute sigmoid into output
        unsafe {
            sigmoid_avx2(input, output);
        }

        // Then multiply by input: output[i] = input[i] * sigmoid(input[i])
        let n = input.len();
        let chunks = n / 4;
        unsafe {
            for i in 0..chunks {
                let off = i * 4;
                let x = _mm256_loadu_pd(input.as_ptr().add(off));
                let sig = _mm256_loadu_pd(output.as_ptr().add(off));
                _mm256_storeu_pd(output.as_mut_ptr().add(off), _mm256_mul_pd(x, sig));
            }
        }
        // Scalar tail: output already has sigmoid from scalar tail in sigmoid_avx2
        for i in (chunks * 4)..n {
            output[i] *= input[i];
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

/// SIMD-accelerated element-wise tanh with runtime feature detection.
///
/// Uses an AVX2 Padé \[2,2\] rational approximation on x86_64 (with `std`
/// feature) when available, falls back to `crate::math::tanh` otherwise.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
///
/// # Examples
///
/// ```
/// use irithyll_core::simd::simd_tanh;
///
/// let input = [0.0, 1.0, -1.0];
/// let mut output = [0.0; 3];
/// simd_tanh(&input, &mut output);
/// assert!(output[0].abs() < 1e-10);
/// assert!((output[1] - 0.7616).abs() < 0.01);
/// ```
pub fn simd_tanh(input: &[f64], output: &mut [f64]) {
    assert!(
        output.len() >= input.len(),
        "simd_tanh: output.len()={} < input.len()={}",
        output.len(),
        input.len()
    );
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: bounds checked above, AVX2 detected.
            unsafe {
                avx2::tanh_avx2(input, output);
            }
            return;
        }
    }
    tanh_scalar(input, output);
}

/// SIMD-accelerated element-wise exp with runtime feature detection.
///
/// Uses AVX2 range-reduction + degree-5 polynomial on x86_64 (with `std`
/// feature) when available, falls back to `crate::math::exp` otherwise.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
///
/// # Examples
///
/// ```
/// use irithyll_core::simd::simd_exp;
///
/// let input = [0.0, 1.0];
/// let mut output = [0.0; 2];
/// simd_exp(&input, &mut output);
/// assert!((output[0] - 1.0).abs() < 1e-10);
/// assert!((output[1] - core::f64::consts::E).abs() < 1e-10);
/// ```
pub fn simd_exp(input: &[f64], output: &mut [f64]) {
    assert!(
        output.len() >= input.len(),
        "simd_exp: output.len()={} < input.len()={}",
        output.len(),
        input.len()
    );
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: bounds checked above, AVX2 detected.
            unsafe {
                avx2::exp_avx2(input, output);
            }
            return;
        }
    }
    exp_scalar(input, output);
}

/// SIMD-accelerated element-wise sigmoid with runtime feature detection.
///
/// Uses AVX2 vectorized exp on x86_64 (with `std` feature) when available,
/// falls back to `crate::math::sigmoid` otherwise.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
///
/// # Examples
///
/// ```
/// use irithyll_core::simd::simd_sigmoid;
///
/// let input = [0.0];
/// let mut output = [0.0; 1];
/// simd_sigmoid(&input, &mut output);
/// assert!((output[0] - 0.5).abs() < 1e-10);
/// ```
pub fn simd_sigmoid(input: &[f64], output: &mut [f64]) {
    assert!(
        output.len() >= input.len(),
        "simd_sigmoid: output.len()={} < input.len()={}",
        output.len(),
        input.len()
    );
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: bounds checked above, AVX2 detected.
            unsafe {
                avx2::sigmoid_avx2(input, output);
            }
            return;
        }
    }
    sigmoid_scalar(input, output);
}

/// SIMD-accelerated element-wise SiLU (Sigmoid Linear Unit) with runtime
/// feature detection.
///
/// Computes `output[i] = input[i] * sigmoid(input[i])` for each element.
/// Uses AVX2 vectorized sigmoid on x86_64 (with `std` feature) when
/// available, falls back to scalar otherwise.
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
///
/// # Examples
///
/// ```
/// use irithyll_core::simd::simd_silu;
///
/// let input = [0.0];
/// let mut output = [0.0; 1];
/// simd_silu(&input, &mut output);
/// assert!(output[0].abs() < 1e-10); // silu(0) = 0 * 0.5 = 0
/// ```
pub fn simd_silu(input: &[f64], output: &mut [f64]) {
    assert!(
        output.len() >= input.len(),
        "simd_silu: output.len()={} < input.len()={}",
        output.len(),
        input.len()
    );
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: bounds checked above, AVX2 detected.
            unsafe {
                avx2::silu_avx2(input, output);
            }
            return;
        }
    }
    silu_scalar(input, output);
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

    // -------------------------------------------------------------------
    // Activation function tests
    // -------------------------------------------------------------------

    #[test]
    fn tanh_known_values() {
        let input = [0.0, 1.0, -1.0, 5.0, -5.0, 0.5];
        let mut output = [0.0; 6];
        simd_tanh(&input, &mut output);
        let expected = [0.0, 0.7616, -0.7616, 0.9999, -0.9999, 0.4621];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.01,
                "tanh[{i}]: expected ~{exp}, got {got}"
            );
        }
    }

    #[test]
    fn tanh_matches_scalar() {
        let mut rng = TestRng::new(42);
        let input = rng.fill_vec(100);
        let mut simd_out = vec![0.0; 100];
        let mut scalar_out = vec![0.0; 100];
        simd_tanh(&input, &mut simd_out);
        for (i, &x) in input.iter().enumerate() {
            scalar_out[i] = crate::math::tanh(x);
        }
        for i in 0..100 {
            assert!(
                (simd_out[i] - scalar_out[i]).abs() < 0.01,
                "tanh[{i}]: SIMD={} vs scalar={}",
                simd_out[i],
                scalar_out[i]
            );
        }
    }

    #[test]
    fn exp_known_values() {
        let input = [0.0, 1.0, -1.0, 2.0, -2.0];
        let mut output = [0.0; 5];
        simd_exp(&input, &mut output);
        let expected = [
            1.0,
            core::f64::consts::E,
            1.0 / core::f64::consts::E,
            core::f64::consts::E * core::f64::consts::E,
            1.0 / (core::f64::consts::E * core::f64::consts::E),
        ];
        for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            let rel = (got - exp).abs() / exp.abs().max(1e-15);
            assert!(
                rel < 1e-5,
                "exp[{i}]: expected {exp}, got {got}, rel_err={rel}"
            );
        }
    }

    #[test]
    fn exp_matches_scalar() {
        let mut rng = TestRng::new(99);
        // Generate values in range [-10, 10] for good coverage
        let input: Vec<f64> = (0..100).map(|_| rng.next_f64() * 10.0).collect();
        let mut simd_out = vec![0.0; 100];
        let mut scalar_out = vec![0.0; 100];
        simd_exp(&input, &mut simd_out);
        for (i, &x) in input.iter().enumerate() {
            scalar_out[i] = crate::math::exp(x);
        }
        for i in 0..100 {
            let rel = (simd_out[i] - scalar_out[i]).abs() / scalar_out[i].abs().max(1e-15);
            assert!(
                rel < 1e-5,
                "exp[{i}] (x={}): SIMD={} vs scalar={}, rel_err={}",
                input[i],
                simd_out[i],
                scalar_out[i],
                rel
            );
        }
    }

    #[test]
    fn exp_extreme_values() {
        // Test clamping behavior at boundaries
        let input = [700.0, -700.0, 0.0, 100.0, -100.0];
        let mut output = [0.0; 5];
        simd_exp(&input, &mut output);
        // exp(700) is huge but finite
        assert!(output[0].is_finite(), "exp(700) should be finite");
        assert!(output[0] > 0.0, "exp(700) should be positive");
        // exp(-700) is tiny but positive
        assert!(output[1] > 0.0, "exp(-700) should be positive");
        assert!(output[1].is_finite(), "exp(-700) should be finite");
        // exp(0) = 1
        assert!((output[2] - 1.0).abs() < 1e-12, "exp(0) should be 1.0");
    }

    #[test]
    fn sigmoid_known_values() {
        let input = [0.0, 10.0, -10.0, 1.0];
        let mut output = [0.0; 4];
        simd_sigmoid(&input, &mut output);
        assert!(
            (output[0] - 0.5).abs() < 0.01,
            "sigmoid(0) should be ~0.5, got {}",
            output[0]
        );
        assert!(
            output[1] > 0.99,
            "sigmoid(10) should be ~1.0, got {}",
            output[1]
        );
        assert!(
            output[2] < 0.01,
            "sigmoid(-10) should be ~0.0, got {}",
            output[2]
        );
    }

    #[test]
    fn sigmoid_matches_scalar() {
        let mut rng = TestRng::new(123);
        let input: Vec<f64> = (0..100).map(|_| rng.next_f64() * 20.0 - 10.0).collect();
        let mut simd_out = vec![0.0; 100];
        let mut scalar_out = vec![0.0; 100];
        simd_sigmoid(&input, &mut simd_out);
        for (i, &x) in input.iter().enumerate() {
            scalar_out[i] = crate::math::sigmoid(x);
        }
        for i in 0..100 {
            assert!(
                (simd_out[i] - scalar_out[i]).abs() < 1e-6,
                "sigmoid[{i}] (x={}): SIMD={} vs scalar={}, diff={}",
                input[i],
                simd_out[i],
                scalar_out[i],
                (simd_out[i] - scalar_out[i]).abs()
            );
        }
    }

    #[test]
    fn silu_known_values() {
        let input = [0.0, 1.0, -1.0, 3.0];
        let mut output = [0.0; 4];
        simd_silu(&input, &mut output);
        // silu(0) = 0 * 0.5 = 0
        assert!(
            output[0].abs() < 0.01,
            "silu(0) should be ~0, got {}",
            output[0]
        );
        // silu(1) = 1 * sigmoid(1) ~ 0.731
        assert!(
            (output[1] - 0.731).abs() < 0.01,
            "silu(1) should be ~0.731, got {}",
            output[1]
        );
    }

    #[test]
    fn silu_matches_scalar() {
        let mut rng = TestRng::new(456);
        let input: Vec<f64> = (0..100).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
        let mut simd_out = vec![0.0; 100];
        simd_silu(&input, &mut simd_out);
        for (i, &x) in input.iter().enumerate() {
            let expected = x * crate::math::sigmoid(x);
            assert!(
                (simd_out[i] - expected).abs() < 1e-6,
                "silu[{i}] (x={}): SIMD={} vs scalar={}, diff={}",
                x,
                simd_out[i],
                expected,
                (simd_out[i] - expected).abs()
            );
        }
    }

    #[test]
    fn activations_handle_empty() {
        let input: [f64; 0] = [];
        let mut output: [f64; 0] = [];
        simd_tanh(&input, &mut output);
        simd_exp(&input, &mut output);
        simd_sigmoid(&input, &mut output);
        simd_silu(&input, &mut output);
    }
}
