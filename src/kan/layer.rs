//! KAN layer: learnable B-spline activations on edges.
//!
//! Each edge (j, i) computes:
//! `φ_{j,i}(x) = w_b · SiLU(x) + w_s · Σ_g c_g · B_g(x)`
//!
//! Layer output:
//! `output_j = Σ_i φ_{j,i}(x_i)`
//!
//! The spline coefficient update is *sparse*: only `k+1` basis functions
//! are non-zero at any point, so only `k+1` coefficients per edge receive
//! gradient updates per step.
//!
//! Reference: Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks"

use super::bspline;

// ---------------------------------------------------------------------------
// RNG helpers (xorshift64, same as the rest of irithyll)
// ---------------------------------------------------------------------------

/// Advance xorshift64 state and return raw u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Uniform f64 in [0, 1) from xorshift64.
#[inline]
fn xorshift64_f64(state: &mut u64) -> f64 {
    // Use 53-bit mantissa for full f64 precision.
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Standard normal via Box-Muller transform (returns one sample).
fn standard_normal(state: &mut u64) -> f64 {
    loop {
        let u1 = xorshift64_f64(state);
        let u2 = xorshift64_f64(state);
        if u1 > 0.0 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * core::f64::consts::PI * u2;
            return r * theta.cos();
        }
    }
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

/// Sigmoid with overflow protection (clamped to [-500, 500]).
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-500.0, 500.0)).exp())
}

/// SiLU (Sigmoid Linear Unit): x * sigmoid(x).
#[inline]
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// SiLU derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
#[inline]
fn silu_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s + x * s * (1.0 - s)
}

// ---------------------------------------------------------------------------
// KANLayer
// ---------------------------------------------------------------------------

/// Kolmogorov-Arnold Network layer with learnable B-spline edge activations.
///
/// Each edge `(j, i)` from input `i` to output `j` has:
/// - `n_coeffs` B-spline coefficients (the learnable activation shape)
/// - `w_b`: weight for the residual SiLU(x) bypass
/// - `w_s`: scale for the spline component
///
/// The spline grid is shared across all edges (uniform, clamped).
#[allow(dead_code)] // n_in/n_out exposed for future use by StreamingKAN
pub(crate) struct KANLayer {
    /// B-spline coefficients, flattened as `[n_out * n_in * n_coeffs]`.
    /// Layout: edge (j, i) has coefficients at offset `(j * n_in + i) * n_coeffs`.
    coefficients: Vec<f64>,
    /// Momentum velocity for each coefficient, same layout as `coefficients`.
    /// Accumulates gradient direction across samples for faster convergence
    /// on sparse B-spline updates.
    velocity: Vec<f64>,
    /// Residual bypass weights, `[n_out * n_in]`.
    w_b: Vec<f64>,
    /// Spline scale weights, `[n_out * n_in]`.
    w_s: Vec<f64>,
    /// Shared knot vector (clamped uniform).
    grid: Vec<f64>,
    n_in: usize,
    n_out: usize,
    /// Spline order (3 = cubic).
    k: usize,
    /// Number of grid intervals.
    g: usize,
    /// Number of B-spline coefficients per edge (= g + k).
    n_coeffs: usize,
    /// Momentum factor for SGD (0.0 = no momentum).
    momentum: f64,
}

impl KANLayer {
    /// Create a new KAN layer.
    ///
    /// - `n_in`: input dimension
    /// - `n_out`: output dimension
    /// - `k`: spline order (3 for cubic)
    /// - `g`: number of grid intervals
    /// - `momentum`: SGD momentum factor (0.0 = disabled, 0.9 = standard)
    /// - `seed`: mutable RNG state
    pub fn new(
        n_in: usize,
        n_out: usize,
        k: usize,
        g: usize,
        momentum: f64,
        seed: &mut u64,
    ) -> Self {
        let n_coeffs = g + k;
        let n_edges = n_out * n_in;
        let grid = bspline::make_grid(-1.0, 1.0, g, k);

        // Initialize coefficients: small random (Gaussian * 0.1)
        let total_coeffs = n_edges * n_coeffs;
        let mut coefficients = Vec::with_capacity(total_coeffs);
        for _ in 0..total_coeffs {
            coefficients.push(standard_normal(seed) * 0.1);
        }

        // Velocity: zero-initialized (same layout as coefficients)
        let velocity = vec![0.0; total_coeffs];

        // Residual bypass: 1/n_in (scaled initialization)
        let w_b_val = 1.0 / n_in as f64;
        let w_b = vec![w_b_val; n_edges];

        // Spline scale: 1.0
        let w_s = vec![1.0; n_edges];

        Self {
            coefficients,
            velocity,
            w_b,
            w_s,
            grid,
            n_in,
            n_out,
            k,
            g,
            n_coeffs,
            momentum,
        }
    }

    /// Forward pass. Returns output of length `n_out`.
    ///
    /// For each output j:
    /// `output_j = Σ_i [w_b_{j,i} * SiLU(x_i) + w_s_{j,i} * Σ_g c_{j,i,g} * B_g(x_i)]`
    ///
    /// Inputs are clamped to the grid domain for spline evaluation safety.
    /// Outputs are clamped to `[-1e6, 1e6]` to prevent blowup in multi-layer
    /// stacks (intermediate activations feed into subsequent layers).
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        debug_assert_eq!(input.len(), self.n_in, "input length mismatch");
        let mut output = vec![0.0; self.n_out];

        for (j, out_j) in output.iter_mut().enumerate() {
            for (i, &x_raw) in input.iter().enumerate() {
                // Clamp to grid domain for stable spline evaluation
                let x = x_raw.clamp(-1.0, 1.0);
                let edge = j * self.n_in + i;
                let coeff_base = edge * self.n_coeffs;

                // Evaluate B-spline bases at x
                let (span, bases) = bspline::evaluate_basis(x, &self.grid, self.g, self.k);

                // Spline value: Σ c_g * B_g(x) (only k+1 non-zero terms)
                let basis_start = span - self.k; // safe: span >= k
                let mut spline_val = 0.0;
                for (b, &basis_val) in bases.iter().enumerate() {
                    let coeff_idx = basis_start + b;
                    if coeff_idx < self.n_coeffs {
                        spline_val += self.coefficients[coeff_base + coeff_idx] * basis_val;
                    }
                }

                // φ_{j,i}(x) = w_b * SiLU(x) + w_s * spline_val
                let phi = self.w_b[edge] * silu(x) + self.w_s[edge] * spline_val;
                *out_j += phi;
            }
        }

        // Clamp outputs to prevent blowup in multi-layer stacks
        for val in &mut output {
            *val = val.clamp(-1e6, 1e6);
        }

        output
    }

    /// Backward pass: update parameters and return input gradient of length `n_in`.
    ///
    /// Sparse update: only `k+1` coefficients per edge are touched per step.
    /// Gradients are clipped to `[-1e3, 1e3]` for numerical stability in
    /// multi-layer stacks.
    pub fn backward(&mut self, input: &[f64], output_grad: &[f64], lr: f64) -> Vec<f64> {
        debug_assert_eq!(input.len(), self.n_in, "input length mismatch");
        debug_assert_eq!(output_grad.len(), self.n_out, "output_grad length mismatch");

        let mut input_grad = vec![0.0; self.n_in];
        const GRAD_CLIP: f64 = 1e3;

        for (j, &delta_j_raw) in output_grad.iter().enumerate() {
            // Clip incoming gradient for stability
            let delta_j = delta_j_raw.clamp(-GRAD_CLIP, GRAD_CLIP);
            if !delta_j.is_finite() {
                continue;
            }

            for (i, &x_raw) in input.iter().enumerate() {
                // Match forward clamping
                let x = x_raw.clamp(-1.0, 1.0);
                let edge = j * self.n_in + i;
                let coeff_base = edge * self.n_coeffs;

                // Re-evaluate bases and derivatives at x
                let (span, bases) = bspline::evaluate_basis(x, &self.grid, self.g, self.k);
                let (_, derivs) =
                    bspline::evaluate_basis_derivatives(x, &self.grid, self.g, self.k);

                let basis_start = span - self.k;

                // Compute spline_val and spline_deriv for gradient computation
                let mut spline_val = 0.0;
                let mut spline_deriv = 0.0;
                for (b, (&basis_val, &deriv_val)) in bases.iter().zip(derivs.iter()).enumerate() {
                    let coeff_idx = basis_start + b;
                    if coeff_idx < self.n_coeffs {
                        let c = self.coefficients[coeff_base + coeff_idx];
                        spline_val += c * basis_val;
                        spline_deriv += c * deriv_val;
                    }
                }

                // --- Update coefficients (SPARSE: only k+1 per edge) ---
                // NaN/Inf guard: skip coefficient updates if any gradient is non-finite.
                // Per-edge momentum: velocity accumulates gradient direction across
                // samples, making sparse B-spline updates converge much faster.
                let coeff_grad_base = delta_j * self.w_s[edge];
                if coeff_grad_base.is_finite() {
                    for (b, &basis_val) in bases.iter().enumerate() {
                        let coeff_idx = basis_start + b;
                        if coeff_idx < self.n_coeffs {
                            let grad = coeff_grad_base * basis_val;
                            if grad.is_finite() {
                                // dL/dc_g = delta_j * w_s * B_g(x)
                                let vi = coeff_base + coeff_idx;
                                self.velocity[vi] = self.momentum * self.velocity[vi] + lr * grad;
                                self.coefficients[vi] -= self.velocity[vi];
                            }
                        }
                    }
                }

                // --- Update w_b ---
                // dL/dw_b = delta_j * SiLU(x)
                let wb_grad = delta_j * silu(x);
                if wb_grad.is_finite() {
                    self.w_b[edge] -= lr * wb_grad;
                }

                // --- Update w_s ---
                // dL/dw_s = delta_j * spline_val
                let ws_grad = delta_j * spline_val;
                if ws_grad.is_finite() {
                    self.w_s[edge] -= lr * ws_grad;
                }

                // --- Input gradient ---
                // dφ/dx = w_b * SiLU'(x) + w_s * Σ c_g * B'_g(x)
                let dphi_dx = self.w_b[edge] * silu_derivative(x) + self.w_s[edge] * spline_deriv;
                input_grad[i] += delta_j * dphi_dx;
            }
        }

        // Clip output gradients for downstream layers
        for g in &mut input_grad {
            *g = g.clamp(-GRAD_CLIP, GRAD_CLIP);
        }

        input_grad
    }

    /// Reset all learnable parameters to fresh random values.
    pub fn reset(&mut self, seed: &mut u64) {
        for c in &mut self.coefficients {
            *c = standard_normal(seed) * 0.1;
        }
        for v in &mut self.velocity {
            *v = 0.0;
        }
        let w_b_val = 1.0 / self.n_in as f64;
        for w in &mut self.w_b {
            *w = w_b_val;
        }
        for w in &mut self.w_s {
            *w = 1.0;
        }
    }

    /// Input dimension.
    #[allow(dead_code)]
    pub fn n_in(&self) -> usize {
        self.n_in
    }

    /// Output dimension.
    #[allow(dead_code)]
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// Total number of learnable parameters.
    ///
    /// `n_out * n_in * (n_coeffs + 2)` — coefficients plus w_b and w_s per edge.
    pub fn n_params(&self) -> usize {
        self.n_out * self.n_in * (self.n_coeffs + 2)
    }

    /// Read-only access to all B-spline coefficients.
    #[allow(dead_code)]
    #[inline]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Mutable access to all B-spline coefficients.
    ///
    /// Used by [`StreamingKAN`](super::StreamingKAN) for coefficient decay
    /// (forgetting mechanism for concept-drift adaptation).
    #[inline]
    pub fn coefficients_mut(&mut self) -> &mut [f64] {
        &mut self.coefficients
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seed() -> u64 {
        0xDEAD_BEEF_CAFE_BABEu64
    }

    #[test]
    fn forward_dimensions() {
        let mut seed = make_seed();
        let layer = KANLayer::new(4, 3, 3, 5, 0.9, &mut seed);
        let input = vec![0.1, -0.5, 0.3, 0.8];
        let output = layer.forward(&input);
        assert_eq!(
            output.len(),
            3,
            "output should have n_out=3 elements, got {}",
            output.len()
        );
    }

    #[test]
    fn forward_finite() {
        let mut seed = make_seed();
        let layer = KANLayer::new(4, 3, 3, 5, 0.9, &mut seed);
        let input = vec![0.1, -0.5, 0.3, 0.8];
        let output = layer.forward(&input);
        for (idx, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "output[{}] is not finite: {}", idx, val);
        }
    }

    #[test]
    fn backward_updates_coefficients() {
        let mut seed = make_seed();
        let mut layer = KANLayer::new(2, 2, 3, 5, 0.9, &mut seed);
        let input = vec![0.3, -0.7];
        let output_grad = vec![1.0, -0.5];

        // Snapshot coefficients before backward
        let coeffs_before: Vec<f64> = layer.coefficients.clone();

        let _input_grad = layer.backward(&input, &output_grad, 0.01);

        // At least some coefficients should have changed
        let changed = coeffs_before
            .iter()
            .zip(layer.coefficients.iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-15)
            .count();

        assert!(
            changed > 0,
            "no coefficients were updated during backward pass"
        );
    }

    #[test]
    fn backward_sparse() {
        let n_in = 3;
        let n_out = 2;
        let k = 3;
        let g = 5;
        let mut seed = make_seed();
        let mut layer = KANLayer::new(n_in, n_out, k, g, 0.9, &mut seed);
        let input = vec![0.2, -0.4, 0.6];
        let output_grad = vec![1.0, 1.0];

        let coeffs_before: Vec<f64> = layer.coefficients.clone();
        let _input_grad = layer.backward(&input, &output_grad, 0.01);

        let changed = coeffs_before
            .iter()
            .zip(layer.coefficients.iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-15)
            .count();

        // Maximum: n_edges * (k+1) = 6 * 4 = 24
        let n_edges = n_out * n_in;
        let max_changed = n_edges * (k + 1);
        assert!(
            changed <= max_changed,
            "too many coefficients changed: {} > max {} (n_edges={} * (k+1)={})",
            changed,
            max_changed,
            n_edges,
            k + 1
        );
        // Should also have changed at least some
        assert!(changed > 0, "no coefficients changed — sparsity check moot");
    }

    #[test]
    fn n_params_formula() {
        let n_in = 4;
        let n_out = 3;
        let k = 3;
        let g = 5;
        let mut seed = make_seed();
        let layer = KANLayer::new(n_in, n_out, k, g, 0.9, &mut seed);
        let n_coeffs = g + k;
        let expected = n_out * n_in * (n_coeffs + 2);
        assert_eq!(
            layer.n_params(),
            expected,
            "n_params should be {} but got {}",
            expected,
            layer.n_params()
        );
    }

    #[test]
    fn learning_y_equals_x_squared() {
        // Train a single-edge KAN layer to approximate y = x^2
        let mut seed = make_seed();
        let mut layer = KANLayer::new(1, 1, 3, 5, 0.9, &mut seed);
        let lr = 0.01;

        // Generate training data: x in [-1, 1]
        let n_samples = 20;
        let mut xs: Vec<f64> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            xs.push(-1.0 + 2.0 * i as f64 / (n_samples - 1) as f64);
        }

        // Compute initial error
        let compute_mse = |layer: &KANLayer| -> f64 {
            let mut mse = 0.0;
            for &x in &xs {
                let y_pred = layer.forward(&[x])[0];
                let y_true = x * x;
                mse += (y_pred - y_true).powi(2);
            }
            mse / xs.len() as f64
        };

        let initial_error = compute_mse(&layer);

        // Train for 200 iterations
        for _ in 0..200 {
            // Shuffle-free: iterate over all samples each epoch
            for &x in &xs {
                let y_true = x * x;
                let y_pred = layer.forward(&[x])[0];
                let grad = 2.0 * (y_pred - y_true); // dL/dy
                let _input_grad = layer.backward(&[x], &[grad], lr);
            }
        }

        let final_error = compute_mse(&layer);

        assert!(
            final_error < initial_error,
            "error should decrease: initial={:.6}, final={:.6}",
            initial_error,
            final_error
        );
        // Should achieve reasonable fit (error well below initial)
        assert!(
            final_error < initial_error * 0.5,
            "error should decrease substantially: initial={:.6}, final={:.6} (ratio={:.4})",
            initial_error,
            final_error,
            final_error / initial_error
        );
    }
}
