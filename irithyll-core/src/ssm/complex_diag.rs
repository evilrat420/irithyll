//! Complex Diagonal SSM — standalone reusable streaming primitive.
//!
//! [`ComplexDiagonalSSM`] implements the complex-valued diagonal recurrence:
//!
//! ```text
//! A = Diag(-exp(log|re|) + j · im)    (stable complex eigenvalues)
//! h_t = discretize(A, Δ_t) · h_{t-1} + input_contribution(B, x_t, Δ_t)
//! y_t = Re(C^T · h_t)                  (real output)
//! ```
//!
//! This is the mathematical core shared by all Mamba-3 variants. Extracting it
//! as a standalone primitive enables:
//!
//! 1. **Reservoir computing**: complex echo state networks with oscillatory dynamics.
//! 2. **Signal processing**: market-data phase tracking via complex rotation.
//! 3. **Composability**: `StreamingMamba V3Exp` and `V3Mimo` both use this cell.
//! 4. **Testability**: unit-test the recurrence math independently of the
//!    full Mamba plumbing.
//!
//! ## Parameterization
//!
//! Complex A is stored as `log_a_complex: Vec<f64>` with interleaved layout
//! `[log|re_0|, im_0, log|re_1|, im_1, ...]`. Actual eigenvalues:
//!
//! ```text
//! A_n = -exp(log_a_complex[2n]) + j · log_a_complex[2n+1]
//! ```
//!
//! Stability: Re(A_n) < 0 is structurally enforced by the negated exp, so
//! `|α_n| = exp(Δ · Re(A_n)) < 1` for any Δ > 0.
//!
//! ## Discretization methods
//!
//! Two methods are supported via [`DiscretizeMethod`]:
//!
//! - **Tustin** (default, `trapezoidal_complex`): S4-style bilinear transform.
//! - **ExpTrapezoidal** (Mamba-3 spec, `exp_trapezoidal_complex`): 3-term
//!   recurrence with data-dependent λ_t.
//!
//! ## References
//!
//! - Lahoti et al. "Mamba-3: Improved Sequence Modeling using State Space
//!   Principles." arXiv:2603.15569, ICLR 2026. §2-3 (complex SSM, exp-trap).
//! - Gu et al. "On the Parameterization and Initialization of Diagonal State
//!   Space Models." NeurIPS 2022. (S4D, s4d_inv_complex init).
//! - Proposition 2 (Mamba-3 paper): complex SSM of dim N/2 ≡ real SSM of dim N
//!   with block-diagonal 2×2 rotation matrices.

use alloc::vec;
use alloc::vec::Vec;

use crate::math;
use crate::ssm::discretize::{exp_trapezoidal_complex, trapezoidal_complex};
use crate::ssm::init::s4d_inv_complex;

/// Discretization method for [`ComplexDiagonalSSM`].
///
/// Selects how the continuous-time complex eigenvalue A is mapped to a
/// discrete-time transition coefficient α.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscretizeMethod {
    /// Bilinear (Tustin) transform: S4-style, 2-term recurrence.
    ///
    /// `α = (I + Δ/2·A)(I - Δ/2·A)⁻¹`. Maps left-half s-plane to unit disk
    /// exactly. Good for S4-style oscillatory SSMs.
    Tustin,
    /// Exponential-trapezoidal: Mamba-3 spec, 3-term recurrence.
    ///
    /// `α = exp(Δ·A)`. Stronger stability (no Δ constraint). Requires
    /// `lambda` per step — either fixed or data-dependent from the model.
    ///
    /// See `exp_trapezoidal_complex` for the full 3-term derivation.
    /// (Lahoti et al., arXiv:2603.15569, ICLR 2026, Table 1.)
    ExpTrapezoidal,
}

/// Complex Diagonal SSM — standalone streaming primitive.
///
/// Maintains a complex hidden state `h ∈ C^N` (stored as 2N real values,
/// interleaved re/im) evolving via a stable complex diagonal recurrence.
/// The real-valued scalar output `y = Re(C^T · h)` projects the complex
/// state back to a real observation.
///
/// ## State layout
///
/// `h` is a flat `Vec<f64>` of length `2 * n_state`:
/// `[re_0, im_0, re_1, im_1, ..., re_{N-1}, im_{N-1}]`.
///
/// ## Stability guarantee
///
/// The A parameterization `A_n = -exp(log_a_complex[2n]) + j·log_a_complex[2n+1]`
/// ensures `Re(A_n) < 0` structurally. Combined with either Tustin or
/// exp-trapezoidal discretization, the spectral radius of the state transition
/// is guaranteed < 1 for any valid Δ > 0.
///
/// ## Initialization
///
/// Default: S4D-Inv complex (`s4d_inv_complex`), which gives harmonically-spaced
/// eigenvalues with oscillatory imaginary parts. This is the Mamba-3 default.
///
/// # Example
///
/// ```
/// use irithyll_core::ssm::complex_diag::{ComplexDiagonalSSM, DiscretizeMethod};
///
/// let mut cell = ComplexDiagonalSSM::new(8, DiscretizeMethod::Tustin);
/// let b = vec![1.0; 8];
/// let c = vec![1.0; 8];
/// let y = cell.step(0.1, &b, &c, 1.0, 0.5);
/// assert!(y.is_finite(), "output must be finite");
/// assert_eq!(cell.state().len(), 16, "state is 2*n_state complex values");
/// ```
pub struct ComplexDiagonalSSM {
    /// Log-magnitude of A's real part and direct imaginary values.
    /// Layout: `[log|re_0|, im_0, log|re_1|, im_1, ...]` (length = 2*n_state).
    /// Actual: `A_n = -exp(log_a[2n]) + j·log_a[2n+1]`.
    log_a_complex: Vec<f64>,
    /// Complex hidden state (length = 2 * n_state, interleaved re/im).
    h: Vec<f64>,
    /// Number of complex state dimensions.
    n_state: usize,
    /// Previous B·x contribution for 3-term recurrence (exp-trapezoidal only).
    /// Layout: `[re_0, im_0, re_1, im_1, ...]` (length = 2 * n_state).
    prev_bx_re: Vec<f64>,
    prev_bx_im: Vec<f64>,
    /// Discretization method.
    method: DiscretizeMethod,
}

impl ComplexDiagonalSSM {
    /// Create a new complex diagonal SSM with S4D-Inv complex initialization.
    ///
    /// Uses `s4d_inv_complex` for harmonically-spaced eigenvalues with
    /// oscillatory imaginary parts (Gu et al., NeurIPS 2022).
    ///
    /// # Arguments
    ///
    /// * `n_state` -- number of complex state dimensions (total state dim = 2*N)
    /// * `method` -- discretization method (Tustin or ExpTrapezoidal)
    pub fn new(n_state: usize, method: DiscretizeMethod) -> Self {
        let log_a_complex = s4d_inv_complex(n_state);
        debug_assert!(
            log_a_complex
                .iter()
                .enumerate()
                .step_by(2)
                .all(|(_i, &v)| v < 20.0),
            "log|re| values from s4d_inv_complex must not overflow exp (< 20.0), \
             but some exceed threshold. Max state dim where ln(0.5+N/1) > 20 is N > e^20 ≈ 5e8."
        );
        Self {
            h: vec![0.0; 2 * n_state],
            prev_bx_re: vec![0.0; n_state],
            prev_bx_im: vec![0.0; n_state],
            n_state,
            log_a_complex,
            method,
        }
    }

    /// Create a ComplexDiagonalSSM with custom A-matrix log-parameters.
    ///
    /// # Arguments
    ///
    /// * `log_a_complex` -- 2*n_state values in `[log|re_0|, im_0, ...]` layout.
    ///   Real parts: `A_re = -exp(log_a[2n])` (must satisfy `log_a[2n] > 0` for |A_re| > 1).
    /// * `method` -- discretization method
    ///
    /// # Panics
    ///
    /// Panics if `log_a_complex.len()` is not even.
    pub fn with_init(log_a_complex: Vec<f64>, method: DiscretizeMethod) -> Self {
        assert!(
            log_a_complex.len() % 2 == 0,
            "log_a_complex must have even length (interleaved re/im), got {}",
            log_a_complex.len()
        );
        let n_state = log_a_complex.len() / 2;
        Self {
            h: vec![0.0; 2 * n_state],
            prev_bx_re: vec![0.0; n_state],
            prev_bx_im: vec![0.0; n_state],
            n_state,
            log_a_complex,
            method,
        }
    }

    /// Advance state by one timestep and return the real-valued scalar output.
    ///
    /// Implements the SISO (single-input single-output) forward pass:
    ///
    /// ```text
    /// A_n = -exp(log_a[2n]) + j·log_a[2n+1]
    /// (α, β, γ) = discretize(A_n, delta, lambda)
    /// h_n ← α·h_n + β·prev_bx_n + γ·(B[n]·x)
    /// y += Re(C[n]·h_n)  = C[n] · Re(h_n)  (real C case)
    /// ```
    ///
    /// For [`DiscretizeMethod::Tustin`], the 3-term β·prev_bx term is zero
    /// (β=0 by construction — `prev_bx` is not used).
    ///
    /// # Arguments
    ///
    /// * `delta` -- step size (positive, data-dependent in Mamba-3)
    /// * `b` -- real input projection vector (length = n_state)
    /// * `c` -- real output projection vector (length = n_state)
    /// * `x` -- scalar input at this timestep
    /// * `lambda` -- exp-trapezoidal mixing parameter ∈ \[0,1\] (ignored for Tustin)
    ///
    /// # Returns
    ///
    /// Real scalar output `y = Re(C^T · h)`.
    pub fn step(&mut self, delta: f64, b: &[f64], c: &[f64], x: f64, lambda: f64) -> f64 {
        debug_assert_eq!(b.len(), self.n_state, "b must have n_state elements");
        debug_assert_eq!(c.len(), self.n_state, "c must have n_state elements");

        let mut y = 0.0;

        for n in 0..self.n_state {
            let a_re = -math::exp(self.log_a_complex[2 * n]);
            let a_im = self.log_a_complex[2 * n + 1];

            // Current B·x contribution (real B, scalar x → real scalar)
            let bx = b[n] * x;

            // Compute state update based on discretization method
            let (h_re_new, h_im_new) = match self.method {
                DiscretizeMethod::Tustin => {
                    let (a_bar_re, a_bar_im, b_fac_re, b_fac_im) =
                        trapezoidal_complex(a_re, a_im, delta);
                    // 2-term: h = α·h + b_fac·bx
                    let h_re_old = self.h[2 * n];
                    let h_im_old = self.h[2 * n + 1];
                    let h_re = a_bar_re * h_re_old - a_bar_im * h_im_old + b_fac_re * bx;
                    let h_im = a_bar_re * h_im_old + a_bar_im * h_re_old + b_fac_im * bx;
                    (h_re, h_im)
                }
                DiscretizeMethod::ExpTrapezoidal => {
                    let (alpha_re, alpha_im, beta_re, beta_im, gamma_re, gamma_im) =
                        exp_trapezoidal_complex(a_re, a_im, delta, lambda);

                    let h_re_old = self.h[2 * n];
                    let h_im_old = self.h[2 * n + 1];

                    // 3-term: h = α·h + β·prev_bx + γ·bx
                    // α·h (complex × complex):
                    let ah_re = alpha_re * h_re_old - alpha_im * h_im_old;
                    let ah_im = alpha_re * h_im_old + alpha_im * h_re_old;

                    // β·prev_bx (complex β, real prev_bx stored as [re, im] of complex state):
                    let pbx_re = self.prev_bx_re[n];
                    let pbx_im = self.prev_bx_im[n];
                    let b_prev_re = beta_re * pbx_re - beta_im * pbx_im;
                    let b_prev_im = beta_re * pbx_im + beta_im * pbx_re;

                    // γ·bx (real γ_re from paper: γ = λ·Δ is real, γ_im=0):
                    let b_curr_re = gamma_re * bx;
                    let b_curr_im = gamma_im * bx;

                    let h_re = ah_re + b_prev_re + b_curr_re;
                    let h_im = ah_im + b_prev_im + b_curr_im;
                    (h_re, h_im)
                }
            };

            self.h[2 * n] = h_re_new;
            self.h[2 * n + 1] = h_im_new;

            // Cache B·x as complex for next step's β term (only meaningful for ExpTrapezoidal)
            // For Tustin this is a no-op store that adds no cost.
            self.prev_bx_re[n] = bx;
            self.prev_bx_im[n] = 0.0; // real B·x has no imaginary part

            // Output: y += Re(C[n] · h_n) = C[n] · Re(h_n) (real C)
            y += c[n] * h_re_new;
        }

        y
    }

    /// Advance state with complex B and C projections.
    ///
    /// Returns `Re(C^* · h)` where `C^*` is the complex conjugate of C.
    /// This is the full complex SSM output per Mamba-3 Proposition 2:
    /// `y = Re((C_re + j·C_im)^* · h) = C_re·Re(h) + C_im·Im(h)`.
    ///
    /// # Arguments
    ///
    /// * `delta` -- step size
    /// * `b_re` -- real part of B vector (length = n_state)
    /// * `b_im` -- imaginary part of B vector (length = n_state)
    /// * `c_re` -- real part of C vector (length = n_state)
    /// * `c_im` -- imaginary part of C vector (length = n_state)
    /// * `x` -- scalar input
    /// * `lambda` -- exp-trapezoidal λ_t (ignored for Tustin)
    ///
    /// # Returns
    ///
    /// Real scalar output `Re(C^* · h)`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_complex(
        &mut self,
        delta: f64,
        b_re: &[f64],
        b_im: &[f64],
        c_re: &[f64],
        c_im: &[f64],
        x: f64,
        lambda: f64,
    ) -> f64 {
        debug_assert_eq!(b_re.len(), self.n_state);
        debug_assert_eq!(b_im.len(), self.n_state);
        debug_assert_eq!(c_re.len(), self.n_state);
        debug_assert_eq!(c_im.len(), self.n_state);

        let mut y = 0.0;

        for n in 0..self.n_state {
            let a_re = -math::exp(self.log_a_complex[2 * n]);
            let a_im = self.log_a_complex[2 * n + 1];

            // Complex B·x: bx = (B_re[n] + j·B_im[n]) * x
            let bx_re = b_re[n] * x;
            let bx_im = b_im[n] * x;

            let (h_re_new, h_im_new) = match self.method {
                DiscretizeMethod::Tustin => {
                    let (a_bar_re, a_bar_im, b_fac_re, b_fac_im) =
                        trapezoidal_complex(a_re, a_im, delta);
                    let h_re_old = self.h[2 * n];
                    let h_im_old = self.h[2 * n + 1];
                    // α·h
                    let ah_re = a_bar_re * h_re_old - a_bar_im * h_im_old;
                    let ah_im = a_bar_re * h_im_old + a_bar_im * h_re_old;
                    // b_fac · bx (complex × complex):
                    let b_contrib_re = b_fac_re * bx_re - b_fac_im * bx_im;
                    let b_contrib_im = b_fac_re * bx_im + b_fac_im * bx_re;
                    (ah_re + b_contrib_re, ah_im + b_contrib_im)
                }
                DiscretizeMethod::ExpTrapezoidal => {
                    let (alpha_re, alpha_im, beta_re, beta_im, gamma_re, gamma_im) =
                        exp_trapezoidal_complex(a_re, a_im, delta, lambda);

                    let h_re_old = self.h[2 * n];
                    let h_im_old = self.h[2 * n + 1];

                    // α·h
                    let ah_re = alpha_re * h_re_old - alpha_im * h_im_old;
                    let ah_im = alpha_re * h_im_old + alpha_im * h_re_old;

                    // β·prev_bx (both complex)
                    let pbx_re = self.prev_bx_re[n];
                    let pbx_im = self.prev_bx_im[n];
                    let b_prev_re = beta_re * pbx_re - beta_im * pbx_im;
                    let b_prev_im = beta_re * pbx_im + beta_im * pbx_re;

                    // γ·bx (γ is real: gamma_im=0)
                    let b_curr_re = gamma_re * bx_re - gamma_im * bx_im;
                    let b_curr_im = gamma_re * bx_im + gamma_im * bx_re;

                    (ah_re + b_prev_re + b_curr_re, ah_im + b_prev_im + b_curr_im)
                }
            };

            self.h[2 * n] = h_re_new;
            self.h[2 * n + 1] = h_im_new;

            // Cache complex B·x for 3-term recurrence
            self.prev_bx_re[n] = bx_re;
            self.prev_bx_im[n] = bx_im;

            // Re(C^* · h) = C_re · Re(h) + C_im · Im(h)
            y += c_re[n] * h_re_new + c_im[n] * h_im_new;
        }

        y
    }

    /// Per-state-dimension L2 energy for plasticity and diagnostics.
    ///
    /// Returns a vec of length `n_state` where each element is
    /// `sqrt(Re(h_n)² + Im(h_n)²)` — the magnitude of the n-th complex state.
    ///
    /// Bounded by stability: `|h_n| ≤ Σ_{t} |α|^{T-t} · |input_t|`. Under a
    /// stable recurrence, this converges to a finite value.
    pub fn state_energies(&self) -> Vec<f64> {
        (0..self.n_state)
            .map(|n| {
                let re = self.h[2 * n];
                let im = self.h[2 * n + 1];
                math::sqrt(re * re + im * im)
            })
            .collect()
    }

    /// Get the complex hidden state as interleaved re/im f64 slice.
    ///
    /// Layout: `[re_0, im_0, re_1, im_1, ...]`, length = `2 * n_state`.
    #[inline]
    pub fn state(&self) -> &[f64] {
        &self.h
    }

    /// Number of complex state dimensions.
    #[inline]
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// Current discretization method.
    #[inline]
    pub fn method(&self) -> DiscretizeMethod {
        self.method
    }

    /// Reset the hidden state and previous B·x cache to zero.
    ///
    /// Clears all temporal memory without changing A-matrix parameters.
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.prev_bx_re.fill(0.0);
        self.prev_bx_im.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stability over 10^6 steps: complex state magnitude must stay bounded.
    /// Validates the structural stability guarantee from the A parameterization.
    #[test]
    fn complex_diag_million_step_finite() {
        let mut cell = ComplexDiagonalSSM::new(8, DiscretizeMethod::ExpTrapezoidal);
        let b: Vec<f64> = (0..8).map(|n| 0.1 * (n as f64 + 1.0)).collect();
        let c: Vec<f64> = (0..8).map(|n| 0.1 * (n as f64 + 1.0)).collect();

        let mut max_abs_output = 0.0_f64;
        for step in 0..1_000_000u64 {
            let x = if step % 2 == 0 { 1.0 } else { -1.0 };
            let lambda = 0.5;
            let delta = 0.1;
            let y = cell.step(delta, &b, &c, x, lambda);
            assert!(
                y.is_finite(),
                "output must be finite at step {}: got {}",
                step,
                y
            );
            max_abs_output = max_abs_output.max(y.abs());
        }

        // State must remain finite
        for (n, &s) in cell.state().iter().enumerate() {
            assert!(
                s.is_finite(),
                "state[{}] must be finite after 10^6 steps: got {}",
                n,
                s
            );
        }

        // State must be bounded (not growing without bound)
        let state_norm: f64 = cell.state().iter().map(|s| s * s).sum::<f64>().sqrt();
        assert!(
            state_norm < 1e6,
            "state Frobenius norm must be bounded after 10^6 steps: got {}",
            state_norm
        );
    }

    #[test]
    fn complex_diag_tustin_stable() {
        let mut cell = ComplexDiagonalSSM::new(4, DiscretizeMethod::Tustin);
        let b = vec![0.1; 4];
        let c = vec![0.1; 4];
        for step in 0..1000 {
            let y = cell.step(0.1, &b, &c, 1.0, 0.5);
            assert!(
                y.is_finite(),
                "Tustin output must be finite at step {}",
                step
            );
        }
        for &s in cell.state() {
            assert!(s.is_finite(), "Tustin state must remain finite");
        }
    }

    #[test]
    fn complex_diag_reset_clears_state() {
        let mut cell = ComplexDiagonalSSM::new(4, DiscretizeMethod::Tustin);
        let b = vec![1.0; 4];
        let c = vec![1.0; 4];
        let _ = cell.step(0.1, &b, &c, 1.0, 0.5);

        let energy_before: f64 = cell.state().iter().map(|s| s * s).sum();
        assert!(energy_before > 0.0, "state must be non-zero after step");

        cell.reset();
        for &s in cell.state() {
            assert!(s.abs() < 1e-15, "state must be zero after reset, got {}", s);
        }
        for &s in &cell.prev_bx_re {
            assert!(s.abs() < 1e-15, "prev_bx_re must be zero after reset");
        }
    }

    #[test]
    fn complex_diag_zero_input_zero_output_from_zero_state() {
        let mut cell = ComplexDiagonalSSM::new(4, DiscretizeMethod::ExpTrapezoidal);
        let b = vec![1.0; 4];
        let c = vec![1.0; 4];
        let y = cell.step(0.1, &b, &c, 0.0, 0.5);
        assert!(
            y.abs() < 1e-15,
            "zero input from zero state must give zero output, got {}",
            y
        );
    }

    #[test]
    fn complex_diag_state_energies_bounded() {
        let mut cell = ComplexDiagonalSSM::new(8, DiscretizeMethod::ExpTrapezoidal);
        let b = vec![0.5; 8];
        let c = vec![0.5; 8];
        for _ in 0..1000 {
            let _ = cell.step(0.1, &b, &c, 1.0, 0.5);
        }
        let energies = cell.state_energies();
        assert_eq!(
            energies.len(),
            8,
            "state_energies must have n_state entries"
        );
        for (n, &e) in energies.iter().enumerate() {
            assert!(
                e.is_finite() && e >= 0.0,
                "energy[{}] must be finite non-negative, got {}",
                n,
                e
            );
        }
    }

    #[test]
    fn complex_diag_with_init_custom_params() {
        // Custom init with 2 complex state dims
        let log_a = vec![
            0.5, 1.0, // n=0: A_re=-exp(0.5)≈-1.65, A_im=1.0
            1.0, 2.0, // n=1: A_re=-exp(1.0)≈-2.72, A_im=2.0
        ];
        let mut cell = ComplexDiagonalSSM::with_init(log_a, DiscretizeMethod::Tustin);
        assert_eq!(cell.n_state(), 2);
        let b = vec![0.5, 0.5];
        let c = vec![0.5, 0.5];
        let y = cell.step(0.1, &b, &c, 1.0, 0.5);
        assert!(y.is_finite());
    }

    #[test]
    fn complex_diag_step_complex_produces_finite_output() {
        let mut cell = ComplexDiagonalSSM::new(4, DiscretizeMethod::ExpTrapezoidal);
        let b_re = vec![0.3; 4];
        let b_im = vec![0.1; 4];
        let c_re = vec![0.3; 4];
        let c_im = vec![0.1; 4];
        for _ in 0..100 {
            let y = cell.step_complex(0.1, &b_re, &b_im, &c_re, &c_im, 1.0, 0.5);
            assert!(
                y.is_finite(),
                "complex step output must be finite: got {}",
                y
            );
        }
    }

    /// The 3-term recurrence (exp-trap) and 2-term (Tustin) must agree at Δ→0.
    /// This validates that `prev_bx` cache does not corrupt when beta→0.
    #[test]
    fn complex_diag_exp_trap_and_tustin_agree_small_delta() {
        let n = 4;
        let log_a = vec![0.5, 0.3, 1.0, 0.5, 1.5, 0.8, 2.0, 1.0];
        let mut cell_et =
            ComplexDiagonalSSM::with_init(log_a.clone(), DiscretizeMethod::ExpTrapezoidal);
        let mut cell_tu = ComplexDiagonalSSM::with_init(log_a, DiscretizeMethod::Tustin);
        let b = vec![0.2_f64; n];
        let c = vec![0.2_f64; n];
        let delta = 0.0001; // very small delta
        let lambda = 1.0; // λ=1 → β=0 → exp-trap collapses to 2-term

        let mut y_et = 0.0;
        let mut y_tu = 0.0;
        for _ in 0..10 {
            y_et = cell_et.step(delta, &b, &c, 1.0, lambda);
            y_tu = cell_tu.step(delta, &b, &c, 1.0, 0.5);
        }
        // At λ=1 and tiny Δ, exp-trap ≈ Tustin
        assert!(
            (y_et - y_tu).abs() < 1e-4,
            "at small delta and lambda=1, exp-trap should approximate Tustin: et={}, tu={}",
            y_et,
            y_tu
        );
    }

    #[test]
    fn complex_diag_n_state_accessor() {
        let cell = ComplexDiagonalSSM::new(16, DiscretizeMethod::Tustin);
        assert_eq!(cell.n_state(), 16);
        assert_eq!(cell.state().len(), 32);
    }
}
