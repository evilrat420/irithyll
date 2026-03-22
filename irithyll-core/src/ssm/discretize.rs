//! Discretization methods for continuous-time state space models.
//!
//! Converts continuous-time dynamics `h'(t) = A*h(t) + B*x(t)` into discrete
//! recurrences `h[k] = A_bar * h[k-1] + B_bar * x[k]`.
//!
//! Two methods are provided:
//!
//! - **Zero-Order Hold (ZOH)** -- assumes the input is piecewise constant between
//!   timesteps. This is the standard discretization used in Mamba/S4.
//!
//! - **Bilinear (Tustin)** -- uses trapezoidal integration, preserving stability
//!   of the continuous system. Better frequency-domain properties for oscillatory
//!   dynamics.
//!
//! Both operate on scalar (diagonal) A values, since all SSMs in this module
//! use diagonal state matrices.

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
}
