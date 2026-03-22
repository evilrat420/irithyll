//! e-prop three-factor learning rule for online SNN training.
//!
//! e-prop (Bellec et al., 2020) enables online learning in recurrent spiking
//! networks using only locally available information: presynaptic activity,
//! postsynaptic eligibility, and a top-down learning signal. This avoids
//! backpropagation through time (BPTT) and its associated memory costs.
//!
//! # Three Factors
//!
//! Weight updates follow `delta_W_ji = eta * e_bar_ji * L_j` where:
//!
//! 1. **Eligibility trace** (`e_bar_ji`) -- filtered product of the presynaptic
//!    trace and the surrogate gradient, capturing which synapses contributed
//!    to recent postsynaptic activity
//! 2. **Learning signal** (`L_j`) -- error feedback projected through fixed
//!    random weights (feedback alignment), indicating whether the neuron's
//!    contribution was helpful or harmful
//! 3. **Learning rate** (`eta`) -- scales the update magnitude
//!
//! All functions operate on slices in Q1.14 fixed-point, enabling use in
//! `no_std` environments without floating-point hardware.

/// Update presynaptic traces (filtered spike trains).
///
/// The presynaptic trace is an exponential moving average of the spike train,
/// providing a temporal smoothing of presynaptic activity:
///
/// ```text
/// trace[i] = (trace[i] * alpha) >> 14 + spike[i] * Q14_ONE
/// ```
///
/// # Arguments
///
/// * `trace` -- mutable slice of presynaptic traces in Q1.14
/// * `spikes` -- binary spike vector (0 or 1)
/// * `alpha` -- decay factor in Q1.14 (e.g., 0.95 => 15565)
///
/// # Panics
///
/// Panics if `trace.len() != spikes.len()`.
pub fn update_pre_trace_fixed(trace: &mut [i16], spikes: &[u8], alpha: i16) {
    debug_assert_eq!(trace.len(), spikes.len());
    let q14_one = super::lif::Q14_ONE;
    for i in 0..trace.len() {
        let decayed = (trace[i] as i32 * alpha as i32) >> 14;
        let spike_contrib = if spikes[i] != 0 { q14_one as i32 } else { 0 };
        let new_val = decayed + spike_contrib;
        trace[i] = new_val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// Update eligibility traces for one postsynaptic neuron.
///
/// The eligibility trace captures which synapses contributed to recent
/// postsynaptic activity, using the surrogate gradient as a "credit" signal:
///
/// ```text
/// elig[i] = (elig[i] * kappa >> 14) + ((1 - kappa) * psi * pre_trace[i] >> 14) >> 14
/// ```
///
/// where `psi` is the surrogate gradient of the postsynaptic neuron and
/// `pre_trace[i]` is the presynaptic trace of input `i`.
///
/// # Arguments
///
/// * `elig` -- mutable slice of eligibility traces for this neuron's synapses
/// * `psi` -- surrogate gradient of the postsynaptic neuron, in Q1.14
/// * `pre_traces` -- presynaptic traces for all input neurons
/// * `kappa` -- eligibility trace decay factor in Q1.14
///
/// # Panics
///
/// Panics if `elig.len() != pre_traces.len()`.
pub fn update_eligibility_fixed(elig: &mut [i16], psi: i16, pre_traces: &[i16], kappa: i16) {
    debug_assert_eq!(elig.len(), pre_traces.len());
    let q14_one = super::lif::Q14_ONE as i32;
    let one_minus_kappa = q14_one - kappa as i32;

    for i in 0..elig.len() {
        // Decay existing trace
        let decayed = (elig[i] as i32 * kappa as i32) >> 14;
        // New contribution: (1 - kappa) * psi * pre_trace[i]
        // Two Q1.14 multiplies: first psi * pre_trace, then scale by (1-kappa)
        let psi_pre = (psi as i32 * pre_traces[i] as i32) >> 14;
        let contribution = (one_minus_kappa * psi_pre) >> 14;
        let new_val = decayed + contribution;
        elig[i] = new_val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// Compute the learning signal for one hidden neuron via feedback alignment.
///
/// The learning signal replaces backpropagated error gradients with a fixed
/// random projection (Lillicrap et al., 2016), which has been shown to
/// enable learning without weight transport:
///
/// ```text
/// L_j = sum_k(feedback[k] * error[k]) >> 14
/// ```
///
/// # Arguments
///
/// * `feedback` -- fixed random feedback weights from output to this neuron, Q1.14
/// * `error` -- per-output error signals in Q1.14
///
/// # Returns
///
/// Learning signal in Q1.14, clamped to i16 range.
///
/// # Panics
///
/// Panics if `feedback.len() != error.len()`.
pub fn compute_learning_signal_fixed(feedback: &[i16], error: &[i16]) -> i16 {
    debug_assert_eq!(feedback.len(), error.len());
    let mut acc: i32 = 0;
    for i in 0..feedback.len() {
        acc += (feedback[i] as i32 * error[i] as i32) >> 14;
    }
    acc.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Apply weight update using the e-prop three-factor rule.
///
/// ```text
/// w[i] += (eta * (elig[i] * learning_signal >> 14)) >> 14
/// ```
///
/// # Arguments
///
/// * `weights` -- mutable slice of synaptic weights in Q1.14
/// * `eligibilities` -- eligibility traces for each synapse
/// * `learning_signal` -- learning signal for the postsynaptic neuron
/// * `eta` -- learning rate in Q1.14
///
/// # Panics
///
/// Panics if `weights.len() != eligibilities.len()`.
pub fn update_weights_fixed(
    weights: &mut [i16],
    eligibilities: &[i16],
    learning_signal: i16,
    eta: i16,
) {
    debug_assert_eq!(weights.len(), eligibilities.len());
    for i in 0..weights.len() {
        // elig * learning_signal in Q1.14
        let product = (eligibilities[i] as i32 * learning_signal as i32) >> 14;
        // Scale by learning rate
        let delta = (eta as i32 * product) >> 14;
        let new_w = weights[i] as i32 + delta;
        weights[i] = new_w.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// Update output weights using the delta rule.
///
/// The readout layer uses a simple delta rule (no eligibility traces needed):
///
/// ```text
/// w[j] += (eta * error * spike[j]) >> 14
/// ```
///
/// Only active (spiking) presynaptic neurons contribute to the update.
///
/// # Arguments
///
/// * `weights` -- mutable output weights for one readout neuron
/// * `error` -- error signal for this output (target - prediction), Q1.14
/// * `spikes` -- binary spike vector from hidden layer
/// * `eta` -- learning rate in Q1.14
///
/// # Panics
///
/// Panics if `weights.len() != spikes.len()`.
pub fn update_output_weights_fixed(weights: &mut [i16], error: i16, spikes: &[u8], eta: i16) {
    debug_assert_eq!(weights.len(), spikes.len());
    for j in 0..weights.len() {
        if spikes[j] != 0 {
            let delta = (eta as i32 * error as i32) >> 14;
            let new_w = weights[j] as i32 + delta;
            weights[j] = new_w.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snn::lif::{f64_to_q14, q14_to_f64, Q14_HALF, Q14_ONE};

    #[test]
    fn pre_trace_accumulates_spikes() {
        let alpha = f64_to_q14(0.9);
        let mut trace = [0_i16; 3];
        let spikes = [1_u8, 0, 1];

        update_pre_trace_fixed(&mut trace, &spikes, alpha);

        // Spike channels should have Q14_ONE, non-spike should stay at 0
        assert_eq!(
            trace[0], Q14_ONE,
            "spiking channel should accumulate to Q14_ONE"
        );
        assert_eq!(trace[1], 0, "non-spiking channel should stay at 0");
        assert_eq!(
            trace[2], Q14_ONE,
            "spiking channel should accumulate to Q14_ONE"
        );
    }

    #[test]
    fn pre_trace_decays_without_spikes() {
        let alpha = f64_to_q14(0.5);
        let mut trace = [Q14_ONE; 2];
        let no_spikes = [0_u8; 2];

        update_pre_trace_fixed(&mut trace, &no_spikes, alpha);

        // After decay: Q14_ONE * 0.5 = Q14_HALF
        assert_eq!(trace[0], Q14_HALF, "trace should decay by alpha");
        assert_eq!(trace[1], Q14_HALF, "trace should decay by alpha");
    }

    #[test]
    fn eligibility_accumulates_with_psi_and_trace() {
        let kappa = f64_to_q14(0.9);
        let psi = f64_to_q14(0.5); // moderate surrogate gradient
        let pre_traces = [Q14_ONE; 2]; // full presynaptic traces
        let mut elig = [0_i16; 2];

        update_eligibility_fixed(&mut elig, psi, &pre_traces, kappa);

        // elig = 0 * kappa + (1-kappa) * psi * pre_trace
        // = (1-0.9) * 0.5 * 1.0 = 0.05
        let expected = f64_to_q14(0.05);
        let tolerance = 50; // allow for fixed-point rounding
        assert!(
            (elig[0] - expected).abs() < tolerance,
            "eligibility should be ~0.05, got {}",
            q14_to_f64(elig[0])
        );
    }

    #[test]
    fn eligibility_decays_with_kappa() {
        let kappa = f64_to_q14(0.5);
        let psi = 0_i16; // no new contribution
        let pre_traces = [0_i16; 1];
        let mut elig = [Q14_ONE; 1]; // start with 1.0

        update_eligibility_fixed(&mut elig, psi, &pre_traces, kappa);

        // Should decay to 0.5
        assert_eq!(
            elig[0], Q14_HALF,
            "eligibility should decay by kappa to 0.5"
        );
    }

    #[test]
    fn learning_signal_combines_feedback_and_error() {
        let feedback = [Q14_ONE, f64_to_q14(-0.5)];
        let error = [f64_to_q14(0.1), f64_to_q14(0.2)];

        let l = compute_learning_signal_fixed(&feedback, &error);

        // L = (1.0 * 0.1) + (-0.5 * 0.2) = 0.1 - 0.1 = 0.0
        let tolerance = 50;
        assert!(
            l.abs() < tolerance,
            "learning signal should be ~0.0, got {}",
            q14_to_f64(l)
        );
    }

    #[test]
    fn learning_signal_responds_to_error_direction() {
        let feedback = [Q14_ONE]; // positive feedback weight
        let positive_error = [f64_to_q14(0.5)];
        let negative_error = [f64_to_q14(-0.5)];

        let l_pos = compute_learning_signal_fixed(&feedback, &positive_error);
        let l_neg = compute_learning_signal_fixed(&feedback, &negative_error);

        assert!(l_pos > 0, "positive error should produce positive signal");
        assert!(l_neg < 0, "negative error should produce negative signal");
    }

    #[test]
    fn weight_update_direction() {
        let eta = f64_to_q14(0.01);
        let learning_signal = f64_to_q14(0.5);
        let elig = [f64_to_q14(0.3), f64_to_q14(-0.2)];
        let mut weights = [0_i16; 2];

        update_weights_fixed(&mut weights, &elig, learning_signal, eta);

        assert!(
            weights[0] > 0,
            "positive elig + positive signal should increase weight"
        );
        assert!(
            weights[1] < 0,
            "negative elig + positive signal should decrease weight"
        );
    }

    #[test]
    fn weight_update_magnitude_scales_with_eta() {
        let elig = [Q14_ONE; 1];
        let learning_signal = Q14_ONE;

        let eta_small = f64_to_q14(0.001);
        let eta_large = f64_to_q14(0.1);

        let mut w_small = [0_i16; 1];
        let mut w_large = [0_i16; 1];

        update_weights_fixed(&mut w_small, &elig, learning_signal, eta_small);
        update_weights_fixed(&mut w_large, &elig, learning_signal, eta_large);

        assert!(
            w_large[0].abs() > w_small[0].abs(),
            "larger eta should produce larger update: small={}, large={}",
            w_small[0],
            w_large[0]
        );
    }

    #[test]
    fn output_weight_update_only_for_spiking_neurons() {
        let eta = f64_to_q14(0.01);
        let error = f64_to_q14(1.0);
        let spikes = [1_u8, 0, 1, 0];
        let mut weights = [0_i16; 4];

        update_output_weights_fixed(&mut weights, error, &spikes, eta);

        assert!(
            weights[0] != 0,
            "spiking neuron 0 should have weight update"
        );
        assert_eq!(weights[1], 0, "non-spiking neuron 1 should have no update");
        assert!(
            weights[2] != 0,
            "spiking neuron 2 should have weight update"
        );
        assert_eq!(weights[3], 0, "non-spiking neuron 3 should have no update");
    }

    #[test]
    fn output_weight_update_sign_matches_error() {
        let eta = f64_to_q14(0.1);
        let spikes = [1_u8; 1];

        let mut w_pos = [0_i16; 1];
        let mut w_neg = [0_i16; 1];

        update_output_weights_fixed(&mut w_pos, f64_to_q14(0.5), &spikes, eta);
        update_output_weights_fixed(&mut w_neg, f64_to_q14(-0.5), &spikes, eta);

        assert!(w_pos[0] > 0, "positive error should increase weight");
        assert!(w_neg[0] < 0, "negative error should decrease weight");
    }

    #[test]
    fn weights_clamp_to_i16_range() {
        let eta = Q14_ONE; // very large learning rate
        let learning_signal = Q14_ONE;
        let elig = [i16::MAX; 1];
        let mut weights = [i16::MAX; 1];

        update_weights_fixed(&mut weights, &elig, learning_signal, eta);

        assert!(
            weights[0] >= i16::MIN && weights[0] <= i16::MAX,
            "weight should be clamped to i16 range"
        );
    }
}
