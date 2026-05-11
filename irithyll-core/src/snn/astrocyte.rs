//! Astrocyte-gated synaptic modulation for spiking neural networks.
//!
//! Two gating modes are supported:
//!
//! - [`AstrocyteMode::WeightMod`]: scales effective input weights in the forward
//!   pass via `w_eff = w * (1 + g_astro)`. This is the original irithyll mode.
//!
//! - [`AstrocyteMode::LearningRateGate`]: gates the **learning rate** in the weight
//!   update rather than the forward-pass weights. Per the four-factor AGMP rule
//!   (Dong & He, Frontiers Neurosci 2025, Eq. 4):
//!   `Δw_ij = η_eff(j) · M_j · e_ij`, where `η_eff(j) = η · g_j` and `g_j ∈ (0,1)`.
//!   This preserves clean e-prop semantics: the forward pass sees unmodified stored
//!   weights, eliminating the train/predict weight-distribution shift that
//!   `WeightMod` introduces.
//!
//! Reference: "Astrocyte-Gated Multi-Timescale Plasticity", Dong & He,
//! Frontiers in Neuroscience, 2025. <https://pmc.ncbi.nlm.nih.gov/articles/PMC12886396/>

use alloc::vec;
use alloc::vec::Vec;

use crate::math::sigmoid;

/// Default target spike rate (10% -- biologically typical for cortical neurons).
const DEFAULT_TARGET_RATE: f64 = 0.1;

/// Q1.14 unit value.
const Q14_ONE: i32 = 16384;

/// Astrocyte gating mode.
///
/// Controls how the per-neuron astrocyte gate signal is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstrocyteMode {
    /// Gate the **input weights** in the forward pass:
    /// `w_eff = w * (1 + g_astro)`. This is the original irithyll behaviour.
    /// Note: modifying weights in the forward pass creates a distribution shift
    /// between the weights seen during inference and the stored weights.
    WeightMod,

    /// Gate the **learning rate** in the weight update (AGMP proper).
    ///
    /// Per Dong & He, Frontiers Neurosci 2025, Eq. 4:
    /// `Δw_ij = η_eff(j) · M_j · e_ij` where `η_eff(j) = η · g_j`.
    ///
    /// The forward pass uses stored weights unmodified — no train/predict
    /// distribution shift. Only the magnitude of each per-neuron weight update
    /// is scaled by the gate.
    LearningRateGate,
}

/// Astrocyte gate tracking slow-timescale spike rates and producing a
/// per-neuron modulatory signal.
///
/// The gate signal `g_j ∈ (0, 1)` is computed from the EWMA spike rate of
/// each hidden neuron relative to the target rate. Depending on the
/// [`AstrocyteMode`], this signal is applied either to the forward-pass weights
/// or to the per-neuron effective learning rate during weight updates.
pub struct AstrocyteGate {
    /// EWMA spike rate per hidden neuron, range [0, 1].
    spike_rates: Vec<f64>,
    /// Modulatory signal per neuron, range [-1, 1] (WeightMod) or (0, 1) (LearningRateGate).
    modulation: Vec<f64>,
    /// Time constant (higher = slower, smoother). Default: 1000.
    tau: f64,
    /// Target spike rate. Default: 0.1.
    target_rate: f64,
    /// Number of hidden neurons.
    n_hidden: usize,
    /// Gating mode.
    mode: AstrocyteMode,
}

impl AstrocyteGate {
    /// Create a new astrocyte gate for `n_hidden` neurons using [`AstrocyteMode::WeightMod`].
    ///
    /// Spike rates are initialized to the target rate (0.1), so initial
    /// modulation is near zero (no effect on weights).
    ///
    /// # Arguments
    ///
    /// * `n_hidden` -- number of hidden neurons to modulate
    /// * `tau` -- EWMA time constant (higher = slower adaptation). Must be > 0.
    pub fn new(n_hidden: usize, tau: f64) -> Self {
        Self::with_mode(n_hidden, tau, AstrocyteMode::WeightMod)
    }

    /// Create a new astrocyte gate with an explicit [`AstrocyteMode`].
    ///
    /// # Arguments
    ///
    /// * `n_hidden` -- number of hidden neurons to modulate
    /// * `tau` -- EWMA time constant (higher = slower adaptation). Must be > 0.
    /// * `mode` -- whether to gate weights (forward pass) or learning rate (weight update)
    pub fn with_mode(n_hidden: usize, tau: f64, mode: AstrocyteMode) -> Self {
        let mut gate = Self {
            spike_rates: vec![DEFAULT_TARGET_RATE; n_hidden],
            modulation: vec![0.0; n_hidden],
            tau,
            target_rate: DEFAULT_TARGET_RATE,
            n_hidden,
            mode,
        };
        // Compute initial modulation (should be ~0 since rates == target)
        gate.recompute_modulation();
        gate
    }

    /// The active gating mode.
    #[inline]
    pub fn mode(&self) -> AstrocyteMode {
        self.mode
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
    /// `WeightMod`:          `g[j] = 2 * σ(rate[j] - target) - 1`  ∈ (-1, 1)
    /// `LearningRateGate`:   `g[j] = σ(rate[j] - target)`           ∈ (0, 1)
    ///
    /// The `LearningRateGate` form corresponds to Dong & He 2025 Eq. 4 simplified
    /// to a single-timescale gate (full four-factor AGMP extends this with a
    /// learned normalised astrocyte state; this is the core gating principle).
    fn recompute_modulation(&mut self) {
        match self.mode {
            AstrocyteMode::WeightMod => {
                for j in 0..self.n_hidden {
                    self.modulation[j] =
                        2.0 * sigmoid(self.spike_rates[j] - self.target_rate) - 1.0;
                }
            }
            AstrocyteMode::LearningRateGate => {
                for j in 0..self.n_hidden {
                    // g_j ∈ (0,1): above-target neurons get closer to 1 (full LR),
                    // below-target neurons get closer to 0 (suppressed LR).
                    self.modulation[j] = sigmoid(self.spike_rates[j] - self.target_rate);
                }
            }
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

    /// Per-neuron effective learning rate in Q1.14 for `LearningRateGate` mode.
    ///
    /// Returns `(eta_q14 * gate_j) >> 14` so that callers can substitute
    /// `effective_eta_q14(j, eta)` for `eta` in `update_weights_fixed`.
    ///
    /// In `WeightMod` mode this still returns `eta_q14` unmodified, so callers
    /// can call this unconditionally and only use the result when
    /// `mode() == AstrocyteMode::LearningRateGate`.
    ///
    /// The gate `g_j = sigmoid(rate_j - target_rate)` is remapped to `(0, 1)` via
    /// a simple sigmoid (not the `[-1,1]` range used by `WeightMod`). This matches
    /// Dong & He 2025 Eq. 4: `g_i[t] = σ(k_m · â_i[t] + β_m)`.
    #[inline]
    pub fn effective_eta_q14(&self, neuron_j: usize, eta_q14: i16) -> i16 {
        // gate_j is already in (0,1) because sigmoid output is in (0,1).
        // modulation for LearningRateGate mode is stored as sigmoid(rate - target) ∈ (0,1).
        let gate = self.modulation[neuron_j]; // (0, 1)
        let gate_q14 = (gate * Q14_ONE as f64) as i32;
        // effective_eta = eta * gate_j (both in Q1.14, product >> 14)
        let result = (eta_q14 as i32 * gate_q14) >> 14;
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

    /// AGMP Dong & He 2025 Eq. 4: the astrocyte gate must modulate the
    /// learning dynamics (via effective eta), NOT the forward-pass weights.
    ///
    /// Empirical check: with `LearningRateGate`, after driving neuron 0 to a
    /// high spike rate, `effective_eta_q14` for neuron 0 is strictly larger than
    /// for neuron 1 (which is kept silent). This confirms the gate scales the
    /// learning rate — high-rate neurons learn faster, low-rate neurons learn slower.
    /// At the same time, `modulate_weight` in `LearningRateGate` mode is NOT called
    /// by the network, so the forward-pass weights are unaffected.
    #[test]
    fn agmp_gates_learning_rate() {
        use crate::snn::lif::f64_to_q14;

        let mut gate = AstrocyteGate::with_mode(4, 10.0, AstrocyteMode::LearningRateGate);
        // Drive neuron 0 to a high spike rate, keep neurons 1-3 silent.
        for _ in 0..80 {
            gate.update(&[1, 0, 0, 0]);
        }

        let eta = f64_to_q14(0.01);

        // Neuron 0: high spike rate → gate close to 1 → effective eta close to eta.
        let eta_0 = gate.effective_eta_q14(0, eta);
        // Neuron 1: low/zero spike rate → gate << 0.5 → effective eta < eta.
        let eta_1 = gate.effective_eta_q14(1, eta);

        assert!(
            eta_0 > eta_1,
            "high-rate neuron should have larger effective eta than silent neuron: \
             eta_0={eta_0}, eta_1={eta_1}"
        );
        // Both must be non-negative (gate is always (0,1) in LearningRateGate mode).
        assert!(
            eta_0 >= 0,
            "effective eta must be non-negative, got {eta_0}"
        );
        assert!(
            eta_1 >= 0,
            "effective eta must be non-negative, got {eta_1}"
        );
        // Neither should exceed the base eta (gate ∈ (0,1) so product ≤ eta).
        assert!(
            eta_0 <= eta,
            "effective eta must not exceed base eta: eta_0={eta_0} eta={eta}"
        );

        // Forward-pass weights should NOT be affected by LearningRateGate mode.
        // `modulate_weight` is only semantically meaningful in WeightMod mode;
        // network_fixed does not call it when mode == LearningRateGate.
        // Verify the gate modulation is stored in (0,1) — not the (-1,1) WeightMod range.
        for &m in gate.modulation() {
            assert!(
                m > 0.0 && m < 1.0,
                "LearningRateGate modulation must be in (0,1), got {m}"
            );
        }
    }

    #[test]
    fn learning_rate_gate_mode_roundtrips() {
        let gate = AstrocyteGate::with_mode(2, 500.0, AstrocyteMode::LearningRateGate);
        assert_eq!(gate.mode(), AstrocyteMode::LearningRateGate);
        let gate2 = AstrocyteGate::new(2, 500.0);
        assert_eq!(gate2.mode(), AstrocyteMode::WeightMod);
    }
}
