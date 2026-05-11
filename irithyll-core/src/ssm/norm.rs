//! BCNorm — RMSNorm stabilization for Mamba-3 B and C projections.
//!
//! [`BCNorm`] implements the normalization applied to input projections
//! B_t and C_t in Mamba-3 (Lahoti et al., arXiv:2603.15569, ICLR 2026, §3.2).
//!
//! ## Why BCNorm?
//!
//! In Mamba-2, B_t and C_t are raw linear projections from the input. With
//! exp-trapezoidal discretization (3-term recurrence), the prior endpoint
//! `β_t · B_{t-1} · x_{t-1}` can accumulate scale from the input, causing
//! instability when inputs have large magnitude.
//!
//! BCNorm stabilizes B_t and C_t via Root Mean Square normalization before
//! they enter the state recurrence. This removes the need for a separate
//! output gate RMSNorm (the post-gate RMSNorm present in Mamba-2 is eliminated
//! in Mamba-3 by BCNorm — see §4.1, Table 3).
//!
//! ## What it does NOT do
//!
//! BCNorm is applied **after** the input projection (`W_B · x`) but **before**
//! the SSM recurrence. It does not normalize the state `h_t` or the output `y_t`.
//! The normalization is scale-invariant: it removes the global L2 norm of the
//! B or C vector while preserving direction.
//!
//! ## Learnable scale
//!
//! A learnable scalar `gamma` (default: 1.0) re-introduces scale after
//! normalization. This is the standard RMSNorm formulation (Zhang & Sennrich,
//! NeurIPS 2019). In a streaming context without backprop, `gamma` can be
//! treated as a fixed hyperparameter or updated via an outer learning loop.
//!
//! ## Formula
//!
//! For a vector `v` of length N:
//!
//! ```text
//! v_normed[n] = gamma · v[n] / sqrt(mean(v[i]²) + eps)
//! ```
//!
//! where `eps` is a small stabilizing constant (paper uses eps=1e-6, §4.1).
//!
//! ## References
//!
//! - Lahoti et al. "Mamba-3: Improved Sequence Modeling using State Space
//!   Principles." arXiv:2603.15569, ICLR 2026. §3.2, §4.1.
//! - Zhang & Sennrich. "Root Mean Square Layer Normalization." NeurIPS 2019.

use alloc::vec::Vec;

use crate::math;

/// BCNorm: RMSNorm for B and C projections in Mamba-3.
///
/// Applies Root Mean Square normalization to the B_t and C_t vectors
/// produced by the input projection. This stabilizes the 3-term exp-trapezoidal
/// recurrence by bounding the magnitude of input contributions to the state.
///
/// ## Usage pattern
///
/// ```
/// use irithyll_core::ssm::norm::BCNorm;
///
/// let norm = BCNorm::new(8);  // n_state = 8
/// let raw_b = vec![1.0, -2.0, 3.0, -1.5, 0.5, 2.5, -0.8, 1.2];
/// let normed = norm.normalize(&raw_b);
/// // normed has unit RMS * gamma, safe to feed into SSM recurrence
/// assert_eq!(normed.len(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct BCNorm {
    /// RMS normalizer epsilon for numerical stability.
    /// Paper value: 1e-6 (Lahoti et al., §4.1).
    eps: f64,
    /// Learnable scale parameter (default: 1.0).
    /// In streaming mode, this is a fixed hyperparameter.
    gamma: f64,
}

impl BCNorm {
    /// Create a BCNorm with paper-default epsilon (1e-6) and gamma=1.0.
    ///
    /// # Arguments
    ///
    /// * `_n_state` -- number of state dimensions (retained for future
    ///   per-dimension gamma extension; currently gamma is shared)
    pub fn new(_n_state: usize) -> Self {
        Self {
            eps: 1e-6, // Lahoti et al., §4.1
            gamma: 1.0,
        }
    }

    /// Create a BCNorm with custom epsilon and gamma.
    ///
    /// # Arguments
    ///
    /// * `eps` -- numerical stability constant (must be > 0; paper uses 1e-6)
    /// * `gamma` -- global scale parameter (must be > 0; paper default 1.0)
    ///
    /// # Panics
    ///
    /// Panics if `eps <= 0.0` or `gamma <= 0.0`.
    pub fn with_params(eps: f64, gamma: f64) -> Self {
        assert!(eps > 0.0, "BCNorm eps must be > 0, got {}", eps);
        assert!(gamma > 0.0, "BCNorm gamma must be > 0, got {}", gamma);
        Self { eps, gamma }
    }

    /// Apply RMS normalization to a B or C vector.
    ///
    /// Computes `gamma · v / sqrt(mean(v²) + eps)`.
    ///
    /// The output has the same direction as `v` and an L2 magnitude of
    /// approximately `gamma · sqrt(N)` (if v was unit RMS before normalization)
    /// — bounded regardless of the input scale.
    ///
    /// # Scale invariance
    ///
    /// `normalize(α·v) == normalize(v)` for any scalar α > 0. This is the
    /// key property: BCNorm removes scale from B_t and C_t before they enter
    /// the state recurrence, preventing input magnitude from destabilizing the
    /// 3-term exp-trapezoidal update.
    ///
    /// # Arguments
    ///
    /// * `v` -- input vector (B_t or C_t projection, any length)
    ///
    /// # Returns
    ///
    /// Normalized vector of the same length as `v`.
    pub fn normalize(&self, v: &[f64]) -> Vec<f64> {
        if v.is_empty() {
            return Vec::new();
        }
        let n = v.len() as f64;
        let mean_sq: f64 = v.iter().map(|&vi| vi * vi).sum::<f64>() / n;
        let rms = math::sqrt(mean_sq + self.eps);
        let scale = self.gamma / rms;
        v.iter().map(|&vi| vi * scale).collect()
    }

    /// Normalize in-place, writing to a pre-allocated output buffer.
    ///
    /// Avoids allocation when `out` is already sized correctly.
    /// Panics (debug) if `v.len() != out.len()`.
    pub fn normalize_into(&self, v: &[f64], out: &mut [f64]) {
        debug_assert_eq!(
            v.len(),
            out.len(),
            "BCNorm: input and output slices must have equal length"
        );
        if v.is_empty() {
            return;
        }
        let n = v.len() as f64;
        let mean_sq: f64 = v.iter().map(|&vi| vi * vi).sum::<f64>() / n;
        let rms = math::sqrt(mean_sq + self.eps);
        let scale = self.gamma / rms;
        for (o, &vi) in out.iter_mut().zip(v.iter()) {
            *o = vi * scale;
        }
    }

    /// Get the epsilon value.
    #[inline]
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Get the gamma scale.
    #[inline]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Scale invariance: BCNorm must produce the same output for v and 2*v.
    /// This is the core property that makes BCNorm useful for stabilizing
    /// input projections with varying magnitudes.
    #[test]
    fn bcnorm_scale_invariance() {
        let norm = BCNorm::new(4);
        let v = vec![1.0, -2.0, 3.0, -1.0];
        let v_scaled: Vec<f64> = v.iter().map(|&x| x * 100.0).collect();

        let normed = norm.normalize(&v);
        let normed_scaled = norm.normalize(&v_scaled);

        // BCNorm is *approximately* scale-invariant: norm(α·v) ≈ norm(v).
        // With eps=1e-6, the relative residual is O(eps / (α²·RMS²)).
        // For α=100, RMS²≈3.75: residual ≈ 1e-6/(1e4·3.75) ≈ 2.7e-11 → absolute diff < 1e-6.
        for (i, (&a, &b)) in normed.iter().zip(normed_scaled.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "BCNorm must be scale-invariant: norm(v)[{}]={} vs norm(100v)[{}]={}",
                i,
                a,
                i,
                b
            );
        }
    }

    #[test]
    fn bcnorm_output_has_unit_rms_times_gamma() {
        let norm = BCNorm::new(4);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let normed = norm.normalize(&v);

        // rms of normalized output should be gamma = 1.0
        let rms: f64 = math::sqrt(normed.iter().map(|&x| x * x).sum::<f64>() / normed.len() as f64);
        assert!(
            (rms - norm.gamma()).abs() < 1e-6,
            "RMS of BCNorm output should be gamma={}, got {}",
            norm.gamma(),
            rms
        );
    }

    #[test]
    fn bcnorm_custom_gamma_scales_output() {
        let norm2 = BCNorm::with_params(1e-6, 2.0);
        let norm1 = BCNorm::new(4);
        let v = vec![1.0, -1.0, 2.0, -2.0];

        let out1 = norm1.normalize(&v);
        let out2 = norm2.normalize(&v);

        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (b - 2.0 * a).abs() < 1e-10,
                "gamma=2 should double the output at index {}: a={}, b={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn bcnorm_zero_vector_is_finite() {
        let norm = BCNorm::new(4);
        let v = vec![0.0, 0.0, 0.0, 0.0];
        let normed = norm.normalize(&v);
        for &x in &normed {
            assert!(
                x.is_finite(),
                "BCNorm of zero vector must be finite: got {}",
                x
            );
            assert_eq!(x, 0.0, "BCNorm of zero vector must be zero, got {}", x);
        }
    }

    #[test]
    fn bcnorm_empty_vector() {
        let norm = BCNorm::new(0);
        let normed = norm.normalize(&[]);
        assert!(
            normed.is_empty(),
            "BCNorm of empty slice must return empty vec"
        );
    }

    #[test]
    fn bcnorm_normalize_into_matches_normalize() {
        let norm = BCNorm::new(4);
        let v = vec![1.0, -2.0, 3.0, -4.0];
        let normed = norm.normalize(&v);
        let mut out = vec![0.0; 4];
        norm.normalize_into(&v, &mut out);
        for (i, (&a, &b)) in normed.iter().zip(out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "normalize_into must match normalize at index {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn bcnorm_single_element() {
        let norm = BCNorm::new(1);
        let v = vec![5.0];
        let normed = norm.normalize(&v);
        // rms = sqrt(25 / 1 + 1e-6) ≈ 5.0; output ≈ 1.0 * 5.0 / 5.0 = 1.0
        assert!(
            normed.len() == 1 && normed[0].is_finite(),
            "single-element BCNorm must be finite"
        );
        assert!(
            (normed[0] - norm.gamma()).abs() < 1e-4,
            "BCNorm of [5.0] should be ~gamma=1.0, got {}",
            normed[0]
        );
    }
}
