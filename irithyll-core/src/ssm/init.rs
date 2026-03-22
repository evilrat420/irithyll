//! A-matrix initialization strategies for diagonal state space models.
//!
//! The A matrix governs state decay dynamics. All SSMs in this module use
//! diagonal A matrices, so initialization reduces to choosing N real values
//! stored as their natural logarithms (`log_A`). The actual A values are
//! recovered as `A_n = -exp(log_A_n)`, ensuring they are always negative
//! (stable).
//!
//! # Initialization Strategies
//!
//! - **Mamba** -- `A_n = -(n+1)`, log_A_n = ln(n+1). Places eigenvalues at
//!   increasing negative values, giving a spectrum of decay rates from slow
//!   (n=0, A=-1) to fast (n=N-1, A=-N). This is the default in Mamba.
//!
//! - **S4D-Lin (Real)** -- `A_n = -0.5` for all n. Uniform decay rate.
//!   Simplest initialization, used as a baseline.
//!
//! - **S4D-Inv (Real)** -- `A_n = -0.5 + N/(n+1)`. Inverse-frequency spacing
//!   inspired by the HiPPO-LegS initialization. Places eigenvalues at
//!   increasing negative real values with harmonic spacing.

use alloc::vec::Vec;

use crate::math;

/// Mamba-style A initialization: `A_n = -(n+1)`, stored as `log_A_n = ln(n+1)`.
///
/// This produces a spectrum of decay rates where state dimension 0 decays
/// slowest (A = -1) and dimension N-1 decays fastest (A = -N). The logarithmic
/// parameterization ensures A remains negative after exponentiation.
///
/// # Arguments
///
/// * `n_state` -- number of state dimensions (N)
///
/// # Returns
///
/// Vec of length `n_state` containing `log_A` values.
///
/// # Example
///
/// ```
/// use irithyll_core::ssm::init::mamba_init;
///
/// let log_a = mamba_init(4);
/// assert_eq!(log_a.len(), 4);
/// // A_0 = -exp(ln(1)) = -1, A_1 = -exp(ln(2)) = -2, etc.
/// ```
pub fn mamba_init(n_state: usize) -> Vec<f64> {
    let mut log_a = Vec::with_capacity(n_state);
    for n in 0..n_state {
        log_a.push(math::ln((n + 1) as f64));
    }
    log_a
}

/// S4D-Lin initialization: `A_n = -0.5` for all n, stored as `log_A_n = ln(0.5)`.
///
/// Uniform decay rate across all state dimensions. This is the simplest
/// initialization and serves as a reasonable default when the decay spectrum
/// doesn't matter (e.g., when the model can learn to adjust via input-dependent
/// projections).
///
/// # Arguments
///
/// * `n_state` -- number of state dimensions (N)
///
/// # Returns
///
/// Vec of length `n_state` containing `log_A` values (all equal to `ln(0.5)`).
pub fn s4d_lin_real(n_state: usize) -> Vec<f64> {
    let val = math::ln(0.5);
    let mut log_a = Vec::with_capacity(n_state);
    for _ in 0..n_state {
        log_a.push(val);
    }
    log_a
}

/// S4D-Inv initialization: `A_n = -0.5 + N/(n+1)` (real part only).
///
/// Inverse-frequency spacing inspired by HiPPO-LegS. The eigenvalues are:
///
/// ```text
/// A_0 = -0.5 + N/1 = N - 0.5    (positive for N >= 1, but stored as log of |A_n|)
/// A_1 = -0.5 + N/2
/// ...
/// A_{N-1} = -0.5 + N/N = 0.5
/// ```
///
/// Note: For use in diagonal SSMs, we take `A_n = -(0.5 + n/(N))` as a
/// stable approximation (all negative), since the original S4D-Inv uses
/// complex eigenvalues with negative real parts. We store `log_A_n = ln(0.5 + n/N)`.
///
/// # Arguments
///
/// * `n_state` -- number of state dimensions (N)
///
/// # Returns
///
/// Vec of length `n_state` containing `log_A` values.
pub fn s4d_inv_real(n_state: usize) -> Vec<f64> {
    let n = n_state as f64;
    let mut log_a = Vec::with_capacity(n_state);
    for i in 0..n_state {
        // A_i = -(0.5 + i/N), ensure always negative
        let a_mag = 0.5 + (i as f64) / n;
        log_a.push(math::ln(a_mag));
    }
    log_a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mamba_init_length_and_values() {
        let log_a = mamba_init(4);
        assert_eq!(log_a.len(), 4);
        // log_A_0 = ln(1) = 0
        assert!(math::abs(log_a[0]) < 1e-12, "log_A[0] should be ln(1)=0");
        // log_A_1 = ln(2)
        assert!(
            math::abs(log_a[1] - math::ln(2.0)) < 1e-12,
            "log_A[1] should be ln(2)"
        );
        // log_A_3 = ln(4)
        assert!(
            math::abs(log_a[3] - math::ln(4.0)) < 1e-12,
            "log_A[3] should be ln(4)"
        );
    }

    #[test]
    fn mamba_init_produces_negative_a() {
        let log_a = mamba_init(8);
        for (n, &la) in log_a.iter().enumerate() {
            let a = -math::exp(la);
            assert!(
                a < 0.0,
                "A[{}] = {} should be negative (log_A={})",
                n,
                a,
                la
            );
            let expected = -((n + 1) as f64);
            assert!(
                math::abs(a - expected) < 1e-10,
                "A[{}] expected {}, got {}",
                n,
                expected,
                a
            );
        }
    }

    #[test]
    fn s4d_lin_all_equal() {
        let log_a = s4d_lin_real(5);
        assert_eq!(log_a.len(), 5);
        let expected = math::ln(0.5);
        for (i, &la) in log_a.iter().enumerate() {
            assert!(
                math::abs(la - expected) < 1e-12,
                "log_A[{}] should be ln(0.5), got {}",
                i,
                la
            );
        }
    }

    #[test]
    fn s4d_lin_produces_negative_a() {
        let log_a = s4d_lin_real(3);
        for &la in &log_a {
            let a = -math::exp(la);
            assert!(a < 0.0, "A should be negative, got {}", a);
            assert!(math::abs(a - (-0.5)) < 1e-12, "A should be -0.5, got {}", a);
        }
    }

    #[test]
    fn s4d_inv_increasing_magnitude() {
        let log_a = s4d_inv_real(8);
        assert_eq!(log_a.len(), 8);
        // Each subsequent A should have larger magnitude
        for i in 1..log_a.len() {
            assert!(
                log_a[i] > log_a[i - 1],
                "log_A[{}]={} should be > log_A[{}]={}",
                i,
                log_a[i],
                i - 1,
                log_a[i - 1]
            );
        }
    }

    #[test]
    fn s4d_inv_all_negative_a() {
        let log_a = s4d_inv_real(16);
        for (i, &la) in log_a.iter().enumerate() {
            let a = -math::exp(la);
            assert!(a < 0.0, "A[{}] should be negative, got {}", i, a);
        }
    }

    #[test]
    fn mamba_init_single_state() {
        let log_a = mamba_init(1);
        assert_eq!(log_a.len(), 1);
        assert!(
            math::abs(log_a[0]) < 1e-12,
            "single state log_A should be ln(1)=0"
        );
    }

    #[test]
    fn s4d_inv_first_element() {
        let log_a = s4d_inv_real(4);
        // A_0 = -(0.5 + 0/4) = -0.5, log_A_0 = ln(0.5)
        let expected = math::ln(0.5);
        assert!(
            math::abs(log_a[0] - expected) < 1e-12,
            "s4d_inv log_A[0] should be ln(0.5), got {}",
            log_a[0]
        );
    }
}
