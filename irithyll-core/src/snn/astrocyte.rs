//! Astrocyte-gated synaptic modulation for spiking neural networks.
//!
//! Astrocytes track slow-timescale spike rates per neuron and produce
//! a modulatory signal that scales effective synaptic weights:
//! `w_eff = w * (1 + g_astro)`.
//!
//! High spike rates strengthen connections (reinforce active pathways),
//! low rates weaken them (prune inactive pathways).
//!
//! Reference: "Astrocyte-Gated Modulated Plasticity" Frontiers Neurosci 2025

use alloc::vec;
use alloc::vec::Vec;

use crate::math::sigmoid;

/// Default target spike rate (10% -- biologically typical for cortical neurons).
const DEFAULT_TARGET_RATE: f64 = 0.1;

/// Q1.14 unit value.
const Q14_ONE: i32 = 16384;

/// Astrocyte gate for modulating synaptic weights based on spike rates.
///
/// Tracks an EWMA of spike rates per hidden neuron and computes a modulatory
/// signal in `[-1, 1]`. The modulation scales effective input weights via
/// `w_eff = w * (1 + g_astro)`, strengthening active pathways and pruning
/// inactive ones.
pub struct AstrocyteGate {
    /// EWMA spike rate per hidden neuron, range [0, 1].
    spike_rates: Vec<f64>,
    /// Modulatory signal per neuron, range [-1, 1].
    modulation: Vec<f64>,
    /// Time constant (higher = slower, smoother). Default: 1000.
    tau: f64,
    /// Target spike rate. Default: 0.1.
    target_rate: f64,
    /// Number of hidden neurons.
    n_hidden: usize,
}

impl AstrocyteGate {
    /// Create a new astrocyte gate for `n_hidden` neurons.
    ///
    /// Spike rates are initialized to the target rate (0.1), so initial
    /// modulation is near zero (no effect on weights).
    ///
    /// # Arguments
    ///
    /// * `n_hidden` -- number of hidden neurons to modulate
    /// * `tau` -- EWMA time constant (higher = slower adaptation). Must be > 0.
    pub fn new(n_hidden: usize, tau: f64) -> Self {
        let mut gate = Self {
            spike_rates: vec![DEFAULT_TARGET_RATE; n_hidden],
            modulation: vec![0.0; n_hidden],
            tau,
            target_rate: DEFAULT_TARGET_RATE,
            n_hidden,
        };
        // Compute initial modulation (should be ~0 since rates == target)
        gate.recompute_modulation();
        gate
    }

    /// Update spike rates and modulation from the current spike vector.
    ///
    /// EWMA update: `rate[j] = (1 - 1/tau) * rate[j] + (1/tau) * spike[j]`
    ///
    /// # Arguments
    ///
    /// * `spikes` -- binary spike vector, length must equal `n_hidden`
    pub fn update(&mut self, spikes: &[u8]) {
        debug_assert_eq!(spikes.len(), self.n_hidden);
        let alpha = 1.0 / self.tau;
        let decay = 1.0 - alpha;

        for (j, &spike) in spikes.iter().enumerate().take(self.n_hidden) {
            let spike_val = if spike != 0 { 1.0 } else { 0.0 };
            self.spike_rates[j] = decay * self.spike_rates[j] + alpha * spike_val;
        }

        self.recompute_modulation();
    }

    /// Recompute modulation signals from current spike rates.
    ///
    /// `g_astro[j] = 2 * sigmoid(rate[j] - target_rate) - 1`
    fn recompute_modulation(&mut self) {
        for j in 0..self.n_hidden {
            self.modulation[j] = 2.0 * sigmoid(self.spike_rates[j] - self.target_rate) - 1.0;
        }
    }

    /// Modulate a base weight for neuron `neuron_j` using Q1.14 arithmetic.
    ///
    /// Returns `w * (1 + g_astro[j])` computed in fixed-point:
    /// `(base_weight * (Q14_ONE + modulation_q14)) >> 14`, clamped to i16 range.
    ///
    /// # Arguments
    ///
    /// * `neuron_j` -- index of the hidden neuron
    /// * `base_weight` -- original Q1.14 weight
    #[inline]
    pub fn modulate_weight(&self, neuron_j: usize, base_weight: i16) -> i16 {
        // Convert modulation [-1, 1] to Q1.14 scaled by 0.5 (half range for stability)
        // modulation_q14 in [-8192, 8192] (half of Q14_ONE)
        let mod_q14 = (self.modulation[neuron_j] * 8192.0) as i32;
        let scale = Q14_ONE + mod_q14; // range [8192, 24576] = [0.5, 1.5] in Q1.14
        let result = (base_weight as i32 * scale) >> 14;
        result.clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }

    /// Read-only access to modulation signals.
    pub fn modulation(&self) -> &[f64] {
        &self.modulation
    }

    /// Read-only access to spike rates for diagnostics.
    pub fn spike_rates(&self) -> &[f64] {
        &self.spike_rates
    }

    /// Reset rates to target and modulation to zero.
    pub fn reset(&mut self) {
        for r in self.spike_rates.iter_mut() {
            *r = self.target_rate;
        }
        self.recompute_modulation();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn astrocyte_new_initializes_correctly() {
        let gate = AstrocyteGate::new(8, 1000.0);
        assert_eq!(gate.spike_rates().len(), 8);
        assert_eq!(gate.modulation().len(), 8);
        // Initial rates should be at target (0.1)
        for &r in gate.spike_rates() {
            assert!((r - 0.1).abs() < 1e-10);
        }
        // Initial modulation should be near zero (rates at target)
        for &m in gate.modulation() {
            assert!(m.abs() < 1e-3);
        }
    }

    #[test]
    fn astrocyte_high_spike_rate_strengthens() {
        let mut gate = AstrocyteGate::new(4, 10.0); // fast tau for testing
                                                    // Spike neuron 0 every step for 50 steps
        for _ in 0..50 {
            gate.update(&[1, 0, 0, 0]);
        }
        // Neuron 0 should have positive modulation (strengthen)
        // After 50 steps at tau=10, rate converges to ~1.0, modulation ~0.73
        assert!(
            gate.modulation()[0] > 0.1,
            "high-rate neuron should have positive modulation, got {}",
            gate.modulation()[0]
        );
        // Neuron 1 should have negative modulation (weaken)
        // After 50 steps at tau=10, rate decays from 0.1 toward 0.
        // sigmoid(-0.1) ~ 0.475, so modulation ~ -0.05. Check it's negative.
        assert!(
            gate.modulation()[1] < 0.0,
            "low-rate neuron should have negative modulation, got {}",
            gate.modulation()[1]
        );
    }

    #[test]
    fn astrocyte_modulate_weight_bounded() {
        let mut gate = AstrocyteGate::new(2, 10.0);
        // Drive high rate
        for _ in 0..100 {
            gate.update(&[1, 0]);
        }
        // Modulated weight should be different from original but bounded
        let original: i16 = 1000;
        let modulated = gate.modulate_weight(0, original);
        assert!(
            modulated > original,
            "high-rate modulation should increase weight"
        );
        assert!(modulated < i16::MAX, "modulated weight should not overflow");

        let modulated_low = gate.modulate_weight(1, original);
        assert!(
            modulated_low < original,
            "low-rate modulation should decrease weight"
        );
    }

    #[test]
    fn astrocyte_reset() {
        let mut gate = AstrocyteGate::new(4, 10.0);
        for _ in 0..50 {
            gate.update(&[1, 1, 1, 1]);
        }
        gate.reset();
        for &r in gate.spike_rates() {
            assert!((r - 0.1).abs() < 1e-10);
        }
    }

    #[test]
    fn astrocyte_modulate_zero_weight() {
        let gate = AstrocyteGate::new(2, 1000.0);
        // Zero weight stays zero regardless of modulation
        assert_eq!(gate.modulate_weight(0, 0), 0);
    }
}
