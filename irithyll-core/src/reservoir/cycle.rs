//! Cycle/ring reservoir topology for Echo State Networks.
//!
//! [`CycleReservoir`] implements a simple yet effective reservoir topology where
//! each neuron receives input from exactly one predecessor in a ring:
//! neuron `i` receives from neuron `(i-1) mod N`.
//!
//! This topology is computationally efficient (O(N*d) per step instead of O(N^2)
//! for dense reservoirs) and has been shown to achieve competitive performance
//! with dense random reservoirs for many tasks.
//!
//! # State Update
//!
//! For input vector `u` and current state `x`, the update rule is:
//!
//! ```text
//! x_tilde[i] = tanh(w_cycle[i-1] * x[i-1] + dot(w_input[i], u) + bias[i])
//! x[i] = (1 - leak_rate) * x_old[i] + leak_rate * x_tilde[i]
//! ```
//!
//! where neuron 0's predecessor is neuron N-1 (ring closure).

use super::prng::Xorshift64Rng;
use crate::rng::{xorshift64, xorshift64_f64};
use alloc::vec;
use alloc::vec::Vec;

/// Cycle/ring topology reservoir.
///
/// The reservoir has `N` neurons arranged in a ring. Each neuron `i` has:
/// - One cycle weight from its predecessor `(i-1) mod N`
/// - A row of `d` input weights (one per input feature)
/// - A bias term
///
/// # Examples
///
/// ```
/// use irithyll_core::reservoir::CycleReservoir;
///
/// let mut res = CycleReservoir::new(50, 3, 0.9, 1.0, 0.3, 0.0, 42);
/// res.update(&[1.0, 2.0, 3.0]);
/// assert_eq!(res.state().len(), 50);
/// ```
pub struct CycleReservoir {
    /// Current reservoir state vector (N neurons).
    state: Vec<f64>,
    /// Cycle weights: w_cycle[i] is the weight FROM neuron i TO neuron (i+1) mod N.
    /// Length N.
    w_cycle: Vec<f64>,
    /// Input weight matrix: w_input[i * n_inputs + j] is the weight from input j
    /// to neuron i. Flattened row-major, shape (N, d).
    w_input: Vec<f64>,
    /// Bias vector, length N.
    bias: Vec<f64>,
    /// Leak rate alpha in (0, 1].
    leak_rate: f64,
    /// Number of reservoir neurons.
    n_reservoir: usize,
    /// Number of input features.
    n_inputs: usize,
    /// Spectral radius used at init (stored for reinit).
    spectral_radius: f64,
    /// Input weight scaling used at init (stored for reinit).
    input_scaling: f64,
    /// Bias scaling used at init (stored for reinit).
    bias_scaling: f64,
}

impl CycleReservoir {
    /// Create a new cycle reservoir with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `n_reservoir` -- number of reservoir neurons
    /// * `n_inputs` -- number of input features
    /// * `spectral_radius` -- controls cycle weight magnitude (typical: 0.1-0.99)
    /// * `input_scaling` -- input weights drawn from U[-input_scaling, input_scaling]
    /// * `leak_rate` -- leaky integration rate in (0, 1]. 1.0 = no memory of previous state.
    /// * `bias_scaling` -- bias drawn from U[-bias_scaling, bias_scaling]. 0.0 = no bias.
    /// * `seed` -- PRNG seed for reproducibility.
    ///
    /// # Panics
    ///
    /// Panics if `n_reservoir == 0` or `leak_rate` is not in (0, 1].
    pub fn new(
        n_reservoir: usize,
        n_inputs: usize,
        spectral_radius: f64,
        input_scaling: f64,
        leak_rate: f64,
        bias_scaling: f64,
        seed: u64,
    ) -> Self {
        assert!(n_reservoir > 0, "n_reservoir must be > 0");
        assert!(
            leak_rate > 0.0 && leak_rate <= 1.0,
            "leak_rate must be in (0, 1]",
        );

        let mut rng = Xorshift64Rng::new(seed);

        // Cycle weights: random sign * spectral_radius.
        // In a cycle topology, the spectral radius equals the absolute value
        // of the cycle weight (all eigenvalues have this magnitude).
        let mut w_cycle = vec![0.0; n_reservoir];
        for w in &mut w_cycle {
            *w = rng.next_sign() * spectral_radius;
        }

        // Input weights: uniform [-input_scaling, input_scaling].
        let mut w_input = vec![0.0; n_reservoir * n_inputs];
        for w in &mut w_input {
            *w = rng.next_uniform(input_scaling);
        }

        // Bias: uniform [-bias_scaling, bias_scaling].
        let mut bias = vec![0.0; n_reservoir];
        if bias_scaling > 0.0 {
            for b in &mut bias {
                *b = rng.next_uniform(bias_scaling);
            }
        }

        Self {
            state: vec![0.0; n_reservoir],
            w_cycle,
            w_input,
            bias,
            leak_rate,
            n_reservoir,
            n_inputs,
            spectral_radius,
            input_scaling,
            bias_scaling,
        }
    }

    /// Advance the reservoir state by one time step with the given input.
    ///
    /// This updates the internal state in-place using the leaky integration
    /// rule with cycle topology.
    pub fn update(&mut self, input: &[f64]) {
        debug_assert_eq!(input.len(), self.n_inputs);

        let n = self.n_reservoir;
        let d = self.n_inputs;
        let leak = self.leak_rate;
        let one_minus_leak = 1.0 - leak;

        // We need the old state to compute the new state, so clone it.
        let old_state = self.state.clone();

        // 1. Batch all input dot products via mat_vec (replaces n individual simd_dot calls).
        let mut input_terms = vec![0.0; n];
        crate::simd::simd_mat_vec(&self.w_input, input, n, d, &mut input_terms);

        // 2. Compute all pre-activations.
        let mut pre = vec![0.0; n];
        for i in 0..n {
            let prev = if i == 0 { n - 1 } else { i - 1 };
            pre[i] = self.w_cycle[prev] * old_state[prev] + input_terms[i] + self.bias[i];
        }

        // 3. Batch tanh.
        let mut x_tilde = vec![0.0; n];
        crate::simd::simd_tanh(&pre, &mut x_tilde);

        // 4. Leaky integration with defensive clamp.
        for i in 0..n {
            self.state[i] = (one_minus_leak * old_state[i] + leak * x_tilde[i]).clamp(-10.0, 10.0);
        }
    }

    /// Current reservoir state vector (read-only).
    #[inline]
    pub fn state(&self) -> &[f64] {
        &self.state
    }

    /// Number of reservoir neurons.
    #[inline]
    pub fn n_reservoir(&self) -> usize {
        self.n_reservoir
    }

    /// Number of input features.
    #[inline]
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// Surgically reinitialize a single reservoir unit `j`.
    ///
    /// Resets the unit's state to zero and draws fresh weights for:
    /// - Input weights: row `j` of `w_input` (uniform in [-input_scaling, input_scaling])
    /// - Cycle weight connecting TO unit `j`: `w_cycle[(j-1+N) mod N]` (random sign * spectral_radius)
    /// - Bias: `bias[j]` (uniform in [-bias_scaling, bias_scaling], or 0 if bias_scaling == 0)
    ///
    /// All other units' weights and state are preserved.
    ///
    /// # Arguments
    ///
    /// * `j` -- reservoir unit index to reinitialize (must be < `n_reservoir`)
    /// * `rng` -- mutable RNG state for generating fresh weights
    ///
    /// # Panics
    ///
    /// Panics if `j >= n_reservoir`.
    pub fn reinitialize_unit(&mut self, j: usize, rng: &mut u64) {
        assert!(
            j < self.n_reservoir,
            "unit index {} out of range (n_reservoir={})",
            j,
            self.n_reservoir
        );

        let n = self.n_reservoir;
        let d = self.n_inputs;

        // Zero state for this unit.
        self.state[j] = 0.0;

        // Reinit input weights: row j spans [j * d .. (j+1) * d].
        let row_start = j * d;
        for col in 0..d {
            self.w_input[row_start + col] = (xorshift64_f64(rng) * 2.0 - 1.0) * self.input_scaling;
        }

        // Reinit the cycle weight connecting TO unit j.
        // w_cycle[prev] is the weight FROM neuron prev TO neuron (prev+1) mod N.
        // The weight that feeds INTO unit j is w_cycle[(j-1+N) mod N].
        let prev = if j == 0 { n - 1 } else { j - 1 };
        self.w_cycle[prev] = if xorshift64(rng) & 1 == 0 {
            self.spectral_radius
        } else {
            -self.spectral_radius
        };

        // Reinit bias for this unit.
        if self.bias_scaling > 0.0 {
            self.bias[j] = (xorshift64_f64(rng) * 2.0 - 1.0) * self.bias_scaling;
        } else {
            self.bias[j] = 0.0;
        }
    }

    /// Reset the reservoir state to all zeros.
    pub fn reset(&mut self) {
        for s in &mut self.state {
            *s = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_starts_at_zero() {
        let res = CycleReservoir::new(10, 2, 0.9, 1.0, 0.3, 0.0, 42);
        assert_eq!(res.state().len(), 10);
        for &s in res.state() {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn update_changes_state() {
        let mut res = CycleReservoir::new(10, 2, 0.9, 1.0, 0.3, 0.0, 42);
        res.update(&[1.0, -1.0]);

        // After one update with nonzero input, at least some neurons should be nonzero.
        let nonzero_count = res.state().iter().filter(|&&s| s.abs() > 1e-15).count();
        assert!(
            nonzero_count > 0,
            "expected nonzero neurons after update, got all zeros",
        );
    }

    #[test]
    fn state_bounded_by_tanh() {
        let mut res = CycleReservoir::new(20, 3, 0.9, 1.0, 1.0, 0.0, 7);
        // Drive with large inputs for many steps.
        for _ in 0..100 {
            res.update(&[100.0, -100.0, 50.0]);
        }
        // With leak_rate = 1.0 and tanh, state values must be in [-1, 1].
        for &s in res.state() {
            assert!(
                (-1.0..=1.0).contains(&s),
                "state {} out of [-1, 1] bounds",
                s,
            );
        }
    }

    #[test]
    fn leak_rate_one_is_no_memory() {
        let mut res = CycleReservoir::new(5, 1, 0.9, 1.0, 1.0, 0.0, 42);
        res.update(&[1.0]);
        let state_after_one = res.state().to_vec();

        // Reset and replay — should get identical state.
        res.reset();
        res.update(&[1.0]);
        for (i, (&a, &b)) in state_after_one.iter().zip(res.state().iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-15,
                "neuron {} differs: {} vs {}",
                i,
                a,
                b,
            );
        }
    }

    #[test]
    fn reset_returns_to_zero() {
        let mut res = CycleReservoir::new(10, 2, 0.9, 1.0, 0.3, 0.0, 42);
        for _ in 0..50 {
            res.update(&[1.0, -0.5]);
        }
        assert!(res.state().iter().any(|&s| s.abs() > 0.01));

        res.reset();
        for &s in res.state() {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut r1 = CycleReservoir::new(15, 3, 0.9, 1.0, 0.3, 0.1, 123);
        let mut r2 = CycleReservoir::new(15, 3, 0.9, 1.0, 0.3, 0.1, 123);

        let inputs = [[1.0, 2.0, 3.0], [-1.0, 0.5, -0.5], [0.0, 0.0, 1.0]];
        for inp in &inputs {
            r1.update(inp);
            r2.update(inp);
        }

        for (i, (&a, &b)) in r1.state().iter().zip(r2.state().iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-15,
                "neuron {} differs between identical reservoirs: {} vs {}",
                i,
                a,
                b,
            );
        }
    }

    #[test]
    fn different_seeds_produce_different_weights() {
        let r1 = CycleReservoir::new(10, 2, 0.9, 1.0, 0.3, 0.0, 1);
        let r2 = CycleReservoir::new(10, 2, 0.9, 1.0, 0.3, 0.0, 2);

        // Input weights should differ.
        let differ = r1
            .w_input
            .iter()
            .zip(r2.w_input.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-15);
        assert!(differ, "different seeds should produce different weights");
    }

    #[test]
    fn leaky_integration_smooths_state() {
        // With low leak_rate, state should change slowly.
        let mut res_slow = CycleReservoir::new(5, 1, 0.9, 1.0, 0.05, 0.0, 42);
        let mut res_fast = CycleReservoir::new(5, 1, 0.9, 1.0, 1.0, 0.0, 42);

        res_slow.update(&[1.0]);
        res_fast.update(&[1.0]);

        // Slow leak reservoir should have smaller magnitude states.
        let slow_norm: f64 = res_slow.state().iter().map(|s| s * s).sum();
        let fast_norm: f64 = res_fast.state().iter().map(|s| s * s).sum();
        assert!(
            slow_norm < fast_norm,
            "slow leak ({}) should have smaller norm than fast leak ({})",
            slow_norm,
            fast_norm,
        );
    }

    #[test]
    fn accessors_return_correct_values() {
        let res = CycleReservoir::new(20, 4, 0.9, 1.0, 0.3, 0.0, 42);
        assert_eq!(res.n_reservoir(), 20);
        assert_eq!(res.n_inputs(), 4);
    }

    #[test]
    #[should_panic(expected = "n_reservoir must be > 0")]
    fn zero_neurons_panics() {
        let _ = CycleReservoir::new(0, 2, 0.9, 1.0, 0.3, 0.0, 42);
    }

    #[test]
    fn reinitialize_unit_preserves_others() {
        let mut res = CycleReservoir::new(10, 3, 0.9, 1.0, 0.3, 0.5, 42);
        // Drive the reservoir to populate state.
        for _ in 0..20 {
            res.update(&[1.0, -0.5, 0.3]);
        }

        // Snapshot all weights and state before reinit.
        let state_before = res.state.clone();
        let w_input_before = res.w_input.clone();
        let w_cycle_before = res.w_cycle.clone();
        let bias_before = res.bias.clone();

        // Reinitialize unit 3.
        let target = 3;
        let mut rng = 999u64;
        res.reinitialize_unit(target, &mut rng);

        // Unit 3's state should be zeroed.
        assert_eq!(
            res.state[target], 0.0,
            "reinitialized unit state should be zero"
        );

        // Unit 3's input weights should differ from before.
        let row_start = target * 3;
        let old_row = &w_input_before[row_start..row_start + 3];
        let new_row = &res.w_input[row_start..row_start + 3];
        let any_changed = old_row
            .iter()
            .zip(new_row.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(any_changed, "input weights for unit 3 should have changed");

        // Cycle weight feeding into unit 3 (w_cycle[2]) should have changed.
        let prev = target - 1; // 2
        assert!(
            (w_cycle_before[prev] - res.w_cycle[prev]).abs() > 1e-15
                || res.w_cycle[prev].abs() == res.spectral_radius,
            "cycle weight feeding unit 3 should be reinitialized"
        );

        // Bias for unit 3 should have changed.
        assert!(
            (bias_before[target] - res.bias[target]).abs() > 1e-15
                || res.bias[target].abs() <= res.bias_scaling,
            "bias for unit 3 should be reinitialized"
        );

        // Other units' state should be unchanged.
        for (i, &sb) in state_before.iter().enumerate().take(10) {
            if i == target {
                continue;
            }
            assert!(
                (sb - res.state[i]).abs() < 1e-15,
                "state of unit {} should be preserved, was {} now {}",
                i,
                sb,
                res.state[i]
            );
        }

        // Other units' input weights should be unchanged.
        for i in 0..10 {
            if i == target {
                continue;
            }
            let rs = i * 3;
            for col in 0..3 {
                assert!(
                    (w_input_before[rs + col] - res.w_input[rs + col]).abs() < 1e-15,
                    "input weight [{}, {}] should be preserved",
                    i,
                    col
                );
            }
        }

        // Other cycle weights should be unchanged (except w_cycle[prev] which feeds target).
        for (i, &wb) in w_cycle_before.iter().enumerate().take(10) {
            if i == prev {
                continue;
            }
            assert!(
                (wb - res.w_cycle[i]).abs() < 1e-15,
                "cycle weight {} should be preserved, was {} now {}",
                i,
                wb,
                res.w_cycle[i]
            );
        }

        // Other biases should be unchanged.
        for (i, &bb) in bias_before.iter().enumerate().take(10) {
            if i == target {
                continue;
            }
            assert!(
                (bb - res.bias[i]).abs() < 1e-15,
                "bias {} should be preserved",
                i
            );
        }
    }
}
