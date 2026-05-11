//! Discretization methods for continuous-time state space models.
//!
//! Converts continuous-time dynamics `h'(t) = A*h(t) + B*x(t)` into discrete
//! recurrences `h[k] = A_bar * h[k-1] + B_bar * x[k]`.
//!
//! ## Methods
//!
//! | Method | Recurrence terms | Paper |
//! |---|---|---|
//! | **ZOH** | 2-term | Mamba-1 (formal), Gu & Dao 2023 |
//! | **Exp-Euler** | 2-term | Mamba-1/-2 actual impl (footnote 3, Gu & Dao 2023) |
//! | **Bilinear (Tustin)** | 2-term | S4, Gu et al. 2022 |
//! | **Exp-Trapezoidal** | 3-term with data-dependent λ_t | Mamba-3, Lahoti et al. ICLR 2026 |
//!
//! All operate on scalar (diagonal) A values, since all SSMs in this module
//! use diagonal state matrices.
//!
//! ## Exp-Trapezoidal vs Bilinear (Tustin)
//!
//! Both are O(Δ³) accurate and preserve stability, but they are **mathematically
//! distinct**:
//!
//! - **Bilinear (Tustin)**: Möbius map `(I + Δ/2·A)(I - Δ/2·A)⁻¹`. Produces a
//!   2-term recurrence. Used in S4 for real-valued A.
//!
//! - **Exp-Trapezoidal (Mamba-3)**: Derived from the exponential-adjusted ODE
//!   `e^{-At}x(t)`, approximating the state-input integral with a convex
//!   combination of endpoints weighted by the data-dependent λ_t ∈ \[0,1\].
//!   Produces a **3-term recurrence** that implicitly replaces the short causal
//!   conv required in Mamba-1/-2 (Lahoti et al., arXiv:2603.15569, §3.1).
//!
//! The 3-term recurrence is:
//! ```text
//! h_t = α_t · h_{t-1} + β_t · B_{t-1} · x_{t-1} + γ_t · B_t · x_t
//! α_t = exp(Δ_t · A)                              (state transition)
//! β_t = (1 - λ_t) · Δ_t · exp(Δ_t · A)           (prior-endpoint contribution)
//! γ_t = λ_t · Δ_t                                 (current-endpoint contribution)
//! ```
//! where λ_t = 1 recovers Mamba-2 exp-Euler; λ_t = 1/2 recovers classical
//! trapezoidal (Lahoti et al., arXiv:2603.15569, Table 1).

use crate::math;

/// Zero-Order Hold discretization for a diagonal A element.
///
/// Given continuous-time scalar dynamics `h'(t) = a * h(t) + b * x(t)`, the
/// ZOH discretization with step size `delta` produces:
///
/// ```text
/// a_bar = exp(delta * a)
/// b_bar_factor = (exp(delta * a) - 1) / a
/// ```
///
/// The actual discrete B is `b_bar_factor * b_t`, but since B is often
/// input-dependent, we return only the factor.
///
/// # Numerical Stability
///
/// When `|a| < 1e-12`, we use the Taylor expansion `b_bar_factor = delta`
/// (L'Hopital's rule: `lim_{a->0} (exp(da)-1)/a = d`).
///
/// # Arguments
///
/// * `a` -- continuous-time diagonal A element (should be negative for stability)
/// * `delta` -- discretization step size (positive)
///
/// # Returns
///
/// `(a_bar, b_bar_factor)` -- the discretized state transition and input scaling factor
#[inline]
pub fn zoh_discretize(a: f64, delta: f64) -> (f64, f64) {
    let a_bar = math::exp(delta * a);
    let b_bar_factor = if math::abs(a) < 1e-12 {
        // L'Hopital: lim_{a->0} (exp(delta*a) - 1) / a = delta
        delta
    } else {
        (a_bar - 1.0) / a
    };
    (a_bar, b_bar_factor)
}

/// Bilinear (Tustin) discretization for a diagonal A element.
///
/// Uses the trapezoidal approximation:
///
/// ```text
/// a_bar = (1 + delta*a/2) / (1 - delta*a/2)
/// b_bar_factor = delta / (1 - delta*a/2)
/// ```
///
/// The bilinear transform maps the left-half s-plane to the unit disk exactly,
/// preserving stability. It has better frequency-domain behavior than ZOH for
/// oscillatory systems but introduces frequency warping.
///
/// # Arguments
///
/// * `a` -- continuous-time diagonal A element (should be negative for stability)
/// * `delta` -- discretization step size (positive)
///
/// # Returns
///
/// `(a_bar, b_bar_factor)` -- the discretized state transition and input scaling factor
#[inline]
pub fn bilinear_discretize(a: f64, delta: f64) -> (f64, f64) {
    let half_da = 0.5 * delta * a;
    let denom = 1.0 - half_da;
    let a_bar = (1.0 + half_da) / denom;
    let b_bar_factor = delta / denom;
    (a_bar, b_bar_factor)
}

/// Trapezoidal discretization for a complex diagonal A element.
///
/// Extends the bilinear (Tustin) method to complex-valued state dynamics,
/// as used in Mamba-3 (Gu & Dao, ICLR 2026). For complex A = a_re + j*a_im:
///
/// ```text
/// A_bar = (1 + delta*A/2) / (1 - delta*A/2)
/// B_bar_factor = delta / (1 - delta*A/2)
/// ```
///
/// Both are computed via complex arithmetic. The trapezoidal transform maps
/// the left-half complex plane to the unit disk, preserving stability: if
/// `a_re < 0`, then `|A_bar| < 1`.
///
/// # Arguments
///
/// * `a_re` -- real part of continuous-time A (should be negative for stability)
/// * `a_im` -- imaginary part of continuous-time A
/// * `delta` -- discretization step size (positive)
///
/// # Returns
///
/// `(a_bar_re, a_bar_im, b_factor_re, b_factor_im)` -- complex discretized
/// state transition and input scaling factor
#[inline]
pub fn trapezoidal_complex(a_re: f64, a_im: f64, delta: f64) -> (f64, f64, f64, f64) {
    // Numerator: 1 + delta*A/2 = (1 + delta*a_re/2) + j*(delta*a_im/2)
    let num_re = 1.0 + 0.5 * delta * a_re;
    let num_im = 0.5 * delta * a_im;

    // Denominator: 1 - delta*A/2 = (1 - delta*a_re/2) + j*(-delta*a_im/2)
    let den_re = 1.0 - 0.5 * delta * a_re;
    let den_im = -0.5 * delta * a_im;

    // |denominator|^2
    let denom_sq = den_re * den_re + den_im * den_im;

    // A_bar = num / den = (num * conj(den)) / |den|^2
    let a_bar_re = (num_re * den_re + num_im * den_im) / denom_sq;
    let a_bar_im = (num_im * den_re - num_re * den_im) / denom_sq;

    // B_bar_factor = delta / den = delta * conj(den) / |den|^2
    let b_factor_re = delta * den_re / denom_sq;
    let b_factor_im = delta * (-den_im) / denom_sq;

    (a_bar_re, a_bar_im, b_factor_re, b_factor_im)
}

/// Exponential-trapezoidal discretization for a complex diagonal A element.
///
/// This is the **Mamba-3 discretization** from Lahoti et al. (arXiv:2603.15569,
/// ICLR 2026, §3.1). It is **distinct** from the bilinear (Tustin) method used
/// in S4 — both are O(Δ³) accurate but produce different state trajectories and
/// different recurrence structures.
///
/// ## Mathematical derivation (§3.1, Table 1)
///
/// Starting from the exponential-adjusted ODE `d/dt[e^{-At}x(t)]`, the exact
/// integral is approximated by a convex combination of the endpoints:
///
/// ```text
/// α_t = exp(Δ_t · A)                                (state transition)
/// β_t = (1 − λ_t) · Δ_t · exp(Δ_t · A)             (prior endpoint weight)
/// γ_t = λ_t · Δ_t                                   (current endpoint weight)
/// ```
///
/// The resulting 3-term recurrence is:
///
/// ```text
/// h_t = α_t · h_{t-1} + β_t · B_{t-1} · x_{t-1} + γ_t · B_t · x_t
/// ```
///
/// This 3-term form implicitly encodes a 2-wide causal convolution on the
/// state-input `B_t·x_t`, replacing the explicit short conv in Mamba-1/-2.
///
/// ## λ_t semantics
///
/// - λ_t = 1 → β_t = 0 → reduces to Mamba-2 exp-Euler (2-term)
/// - λ_t = 0 → γ_t = 0 → backward-only endpoint rule
/// - λ_t = 1/2 → symmetric trapezoidal (minimum error O(Δ³) theoretically)
/// - λ_t learned freely in \[0,1\] via sigmoid (paper §4.3: "does better without constraint")
///
/// ## Stability
///
/// `|α_t|² = |exp(Δ·A)|² = exp(2·Δ·Re(A))`. Since Re(A) < 0 is enforced
/// by `A_re = -exp(log|re|)`, stability holds for **any positive Δ_t** without
/// additional constraints — stronger than Tustin which requires `|1 - Δ·A/2| > 0`.
///
/// # Arguments
///
/// * `a_re` -- real part of continuous-time A (must be negative for stability)
/// * `a_im` -- imaginary part of continuous-time A (oscillatory component)
/// * `delta` -- discretization step size (positive, data-dependent in Mamba-3)
/// * `lambda` -- convex combination parameter in [0, 1] (data-dependent, learned)
///
/// # Returns
///
/// `(α_re, α_im, β_re, β_im, γ_re, γ_im)` -- the 3-term discretized coefficients.
/// β and γ are scalars here (real × complex product), returned as complex for
/// uniform interface with the `step_complex` path.
///
/// # References
///
/// Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu. "Mamba-3: Improved Sequence
/// Modeling using State Space Principles." arXiv:2603.15569, ICLR 2026. Table 1.
#[inline]
pub fn exp_trapezoidal_complex(
    a_re: f64,
    a_im: f64,
    delta: f64,
    lambda: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    // α_t = exp(Δ · A) in complex arithmetic
    // exp((delta * a_re) + j*(delta * a_im)) = exp(delta*a_re) * (cos(delta*a_im) + j*sin(delta*a_im))
    let exp_re_part = math::exp(delta * a_re);
    let angle = delta * a_im;
    let alpha_re = exp_re_part * math::cos(angle);
    let alpha_im = exp_re_part * math::sin(angle);

    // β_t = (1 - λ) · Δ · α_t  (complex: scalar × complex)
    let beta_scalar = (1.0 - lambda) * delta;
    let beta_re = beta_scalar * alpha_re;
    let beta_im = beta_scalar * alpha_im;

    // γ_t = λ · Δ  (real scalar — no imaginary component)
    let gamma_re = lambda * delta;
    let gamma_im = 0.0;

    (alpha_re, alpha_im, beta_re, beta_im, gamma_re, gamma_im)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoh_negative_a_produces_decaying_state() {
        let a = -1.0;
        let delta = 0.1;
        let (a_bar, b_bar_factor) = zoh_discretize(a, delta);
        // a_bar = exp(-0.1) ~ 0.9048
        assert!(
            a_bar > 0.0 && a_bar < 1.0,
            "a_bar should be in (0,1) for negative a"
        );
        assert!(b_bar_factor > 0.0, "b_bar_factor should be positive");
        let expected_a_bar = math::exp(-0.1);
        assert!(
            math::abs(a_bar - expected_a_bar) < 1e-12,
            "expected a_bar={}, got {}",
            expected_a_bar,
            a_bar
        );
    }

    #[test]
    fn zoh_a_near_zero_uses_lhopital() {
        let a = 1e-15;
        let delta = 0.5;
        let (a_bar, b_bar_factor) = zoh_discretize(a, delta);
        // a_bar ~ 1.0, b_bar_factor ~ delta = 0.5
        assert!(
            math::abs(a_bar - 1.0) < 1e-10,
            "a_bar should be ~1.0 when a~0"
        );
        assert!(
            math::abs(b_bar_factor - delta) < 1e-10,
            "b_bar_factor should be ~delta when a~0, got {}",
            b_bar_factor
        );
    }

    #[test]
    fn zoh_large_negative_a_decays_quickly() {
        let a = -100.0;
        let delta = 1.0;
        let (a_bar, _b_bar_factor) = zoh_discretize(a, delta);
        // exp(-100) ~ 0
        assert!(
            a_bar < 1e-40,
            "a_bar should be ~0 for very negative a*delta"
        );
    }

    #[test]
    fn bilinear_negative_a_stable() {
        let a = -2.0;
        let delta = 0.1;
        let (a_bar, b_bar_factor) = bilinear_discretize(a, delta);
        // a_bar = (1 - 0.1) / (1 + 0.1) = 0.9/1.1 ~ 0.818
        assert!(
            math::abs(a_bar) < 1.0,
            "bilinear should preserve stability, got a_bar={}",
            a_bar
        );
        assert!(b_bar_factor > 0.0, "b_bar_factor should be positive");
        let expected_a_bar = (1.0 + 0.5 * 0.1 * (-2.0)) / (1.0 - 0.5 * 0.1 * (-2.0));
        assert!(
            math::abs(a_bar - expected_a_bar) < 1e-12,
            "expected {}, got {}",
            expected_a_bar,
            a_bar
        );
    }

    #[test]
    fn bilinear_a_zero_identity() {
        let a = 0.0;
        let delta = 0.5;
        let (a_bar, b_bar_factor) = bilinear_discretize(a, delta);
        assert!(
            math::abs(a_bar - 1.0) < 1e-12,
            "a_bar should be 1.0 when a=0"
        );
        assert!(
            math::abs(b_bar_factor - delta) < 1e-12,
            "b_bar_factor should be delta when a=0"
        );
    }

    #[test]
    fn zoh_and_bilinear_agree_for_small_delta() {
        // For small delta, both methods should produce similar results
        let a = -1.0;
        let delta = 0.001;
        let (zoh_a, zoh_b) = zoh_discretize(a, delta);
        let (bil_a, bil_b) = bilinear_discretize(a, delta);
        assert!(
            math::abs(zoh_a - bil_a) < 1e-5,
            "ZOH and bilinear should agree for small delta: zoh={}, bil={}",
            zoh_a,
            bil_a
        );
        assert!(
            math::abs(zoh_b - bil_b) < 1e-5,
            "ZOH and bilinear b_bar should agree: zoh={}, bil={}",
            zoh_b,
            bil_b
        );
    }

    #[test]
    fn zoh_delta_zero_gives_identity() {
        let a = -5.0;
        let delta = 0.0;
        let (a_bar, b_bar_factor) = zoh_discretize(a, delta);
        assert!(
            math::abs(a_bar - 1.0) < 1e-12,
            "a_bar should be 1.0 when delta=0"
        );
        assert!(
            math::abs(b_bar_factor) < 1e-12,
            "b_bar_factor should be ~0 when delta=0"
        );
    }

    #[test]
    fn trapezoidal_complex_real_matches_bilinear() {
        // When imaginary part is zero, trapezoidal_complex should match bilinear_discretize
        let a = -2.0;
        let delta = 0.1;
        let (bil_a, bil_b) = bilinear_discretize(a, delta);
        let (trap_a_re, trap_a_im, trap_b_re, trap_b_im) = trapezoidal_complex(a, 0.0, delta);
        assert!(
            math::abs(trap_a_re - bil_a) < 1e-12,
            "trap a_bar_re should match bilinear a_bar: trap={}, bil={}",
            trap_a_re,
            bil_a
        );
        assert!(
            math::abs(trap_a_im) < 1e-12,
            "trap a_bar_im should be 0 for real A, got {}",
            trap_a_im
        );
        assert!(
            math::abs(trap_b_re - bil_b) < 1e-12,
            "trap b_factor_re should match bilinear b_factor: trap={}, bil={}",
            trap_b_re,
            bil_b
        );
        assert!(
            math::abs(trap_b_im) < 1e-12,
            "trap b_factor_im should be 0 for real A, got {}",
            trap_b_im
        );
    }

    #[test]
    fn trapezoidal_complex_stable_eigenvalue() {
        // Complex A with negative real part should produce |A_bar| < 1
        let a_re = -1.0;
        let a_im = 2.0;
        let delta = 0.1;
        let (a_bar_re, a_bar_im, _, _) = trapezoidal_complex(a_re, a_im, delta);
        let mag_sq = a_bar_re * a_bar_re + a_bar_im * a_bar_im;
        assert!(
            mag_sq < 1.0,
            "|A_bar|^2 should be < 1 for stable A, got {}",
            mag_sq
        );
    }

    // --- exp_trapezoidal_complex tests ---

    /// Paper spec: Table 1 — λ=1 reduces exp-trapezoidal to exp-Euler (Mamba-2).
    /// At λ=1: β=0 (prior endpoint weight = 0), γ=Δ (current only).
    /// The α matches exp-Euler exactly: α = exp(Δ·A).
    #[test]
    fn exp_trapezoidal_matches_paper_spec() {
        let a_re = -1.0;
        let a_im = 0.5;
        let delta = 0.1;

        // λ=1 → should match Mamba-2 exp-Euler: α=exp(Δ·A), β=0, γ=Δ
        let (alpha_re, alpha_im, beta_re, beta_im, gamma_re, gamma_im) =
            exp_trapezoidal_complex(a_re, a_im, delta, 1.0);

        // α = exp(Δ·a_re) * (cos(Δ·a_im), sin(Δ·a_im))
        let exp_factor = math::exp(delta * a_re);
        let angle = delta * a_im;
        let expected_alpha_re = exp_factor * math::cos(angle);
        let expected_alpha_im = exp_factor * math::sin(angle);

        assert!(
            math::abs(alpha_re - expected_alpha_re) < 1e-12,
            "alpha_re should match exp-Euler: expected={}, got={}",
            expected_alpha_re,
            alpha_re
        );
        assert!(
            math::abs(alpha_im - expected_alpha_im) < 1e-12,
            "alpha_im should match exp-Euler: expected={}, got={}",
            expected_alpha_im,
            alpha_im
        );
        // β must be zero at λ=1 (prior endpoint unused)
        assert!(
            math::abs(beta_re) < 1e-12,
            "beta_re must be 0 at lambda=1, got {}",
            beta_re
        );
        assert!(
            math::abs(beta_im) < 1e-12,
            "beta_im must be 0 at lambda=1, got {}",
            beta_im
        );
        // γ = 1.0 * Δ = delta at λ=1
        assert!(
            math::abs(gamma_re - delta) < 1e-12,
            "gamma_re must be delta at lambda=1: expected={}, got={}",
            delta,
            gamma_re
        );
        assert!(
            math::abs(gamma_im) < 1e-12,
            "gamma_im must be 0, got {}",
            gamma_im
        );
    }

    /// λ=0 → γ=0 (current endpoint unused), β=(1-0)·Δ·α = Δ·α (prior endpoint only).
    #[test]
    fn exp_trapezoidal_lambda_zero_backward_only() {
        let a_re = -1.0;
        let a_im = 0.5;
        let delta = 0.2;

        let (alpha_re, alpha_im, beta_re, beta_im, gamma_re, gamma_im) =
            exp_trapezoidal_complex(a_re, a_im, delta, 0.0);

        // γ = 0 at λ=0
        assert!(
            math::abs(gamma_re) < 1e-12,
            "gamma_re must be 0 at lambda=0, got {}",
            gamma_re
        );
        assert!(
            math::abs(gamma_im) < 1e-12,
            "gamma_im must be 0 at lambda=0, got {}",
            gamma_im
        );
        // β = Δ·α at λ=0
        assert!(
            math::abs(beta_re - delta * alpha_re) < 1e-12,
            "beta_re must be delta*alpha_re at lambda=0: expected={}, got={}",
            delta * alpha_re,
            beta_re
        );
        assert!(
            math::abs(beta_im - delta * alpha_im) < 1e-12,
            "beta_im must be delta*alpha_im at lambda=0: expected={}, got={}",
            delta * alpha_im,
            beta_im
        );
    }

    /// Stability: |α_t|² = exp(2·Δ·Re(A)) < 1 whenever Re(A) < 0.
    /// This is stronger than Tustin: no Δ constraint needed.
    #[test]
    fn exp_trapezoidal_stable_for_negative_real_a() {
        let test_cases = [
            (-0.5, 1.0, 0.1, 0.5),
            (-2.0, 3.15, 1.0, 0.3), // large delta — exp-trap stays stable
            (-0.1, 0.0, 5.0, 0.7),  // very large delta
            (-10.0, 2.0, 0.5, 1.0), // large negative a_re
        ];
        for (a_re, a_im, delta, lambda) in test_cases {
            let (alpha_re, alpha_im, _, _, _, _) =
                exp_trapezoidal_complex(a_re, a_im, delta, lambda);
            let mag_sq = alpha_re * alpha_re + alpha_im * alpha_im;
            assert!(
                mag_sq < 1.0,
                "|alpha|² must be < 1 for Re(A)<0: a_re={}, a_im={}, delta={}, lambda={}, got mag_sq={}",
                a_re, a_im, delta, lambda, mag_sq
            );
        }
    }

    /// At small delta, exp-trapezoidal and Tustin should converge (both O(Δ³) accurate).
    #[test]
    fn exp_trapezoidal_and_tustin_agree_at_small_delta() {
        let a_re = -1.0;
        let a_im = 0.5;
        let delta = 0.001; // very small
        let lambda = 0.5; // symmetric

        let (et_a_re, et_a_im, _, _, _, _) = exp_trapezoidal_complex(a_re, a_im, delta, lambda);
        let (tu_a_re, tu_a_im, _, _) = trapezoidal_complex(a_re, a_im, delta);

        assert!(
            math::abs(et_a_re - tu_a_re) < 1e-5,
            "exp-trap and Tustin alpha_re should agree at small delta: et={}, tu={}",
            et_a_re,
            tu_a_re
        );
        assert!(
            math::abs(et_a_im - tu_a_im) < 1e-5,
            "exp-trap and Tustin alpha_im should agree at small delta: et={}, tu={}",
            et_a_im,
            tu_a_im
        );
    }

    /// Distinct from Tustin at moderate delta: exp-trap uses exp(Δ·A) while
    /// Tustin uses Möbius map. They should differ at Δ=0.5.
    #[test]
    fn exp_trapezoidal_distinct_from_tustin_at_moderate_delta() {
        let a_re = -1.0;
        let a_im = 1.0;
        let delta = 0.5;
        let lambda = 0.5;

        let (et_a_re, et_a_im, _, _, _, _) = exp_trapezoidal_complex(a_re, a_im, delta, lambda);
        let (tu_a_re, tu_a_im, _, _) = trapezoidal_complex(a_re, a_im, delta);

        let diff = (et_a_re - tu_a_re).powi(2) + (et_a_im - tu_a_im).powi(2);
        assert!(
            diff > 1e-8,
            "exp-trap and Tustin should differ at delta=0.5: et=({},{}) tu=({},{})",
            et_a_re,
            et_a_im,
            tu_a_re,
            tu_a_im
        );
    }

    /// Beta + gamma coefficients must sum to delta (total input weight at λ=0.5).
    /// At λ=0.5 and real A=0: β=0.5·Δ, γ=0.5·Δ → β_re + γ_re = Δ.
    #[test]
    fn exp_trapezoidal_input_weights_sum_to_delta_at_zero_a() {
        // With A=0: exp(0)=1, so α=1, β=(1-λ)·Δ·1, γ=λ·Δ → β+γ = Δ
        let a_re = 0.0;
        let a_im = 0.0;
        let delta = 0.3;
        let lambda = 0.5;
        let (_, _, beta_re, _, gamma_re, _) = exp_trapezoidal_complex(a_re, a_im, delta, lambda);
        assert!(
            math::abs(beta_re + gamma_re - delta) < 1e-12,
            "beta_re + gamma_re should equal delta when A=0: expected={}, got={}",
            delta,
            beta_re + gamma_re
        );
    }
}
