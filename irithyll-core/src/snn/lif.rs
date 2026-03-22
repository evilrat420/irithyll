//! Leaky Integrate-and-Fire neuron step in Q1.14 fixed-point arithmetic.
//!
//! The LIF neuron is the fundamental unit of the spiking network. At each
//! timestep, the membrane potential decays toward rest, input current is
//! integrated, and a spike is emitted if the potential exceeds a threshold.
//! After spiking, the membrane is reset by subtracting the threshold
//! (subtract-reset model).
//!
//! # Q1.14 Fixed-Point
//!
//! Values are stored as `i16` where `16384` represents `1.0`. This gives
//! a representable range of approximately `[-2.0, +2.0)` with ~0.00006
//! resolution. All multiply operations use `i32` intermediates:
//!
//! ```text
//! (a as i32 * b as i32) >> 14
//! ```

/// 1.0 in Q1.14 fixed-point representation.
pub const Q14_ONE: i16 = 16384;

/// 0.5 in Q1.14 fixed-point representation.
pub const Q14_HALF: i16 = 8192;

/// 0.25 in Q1.14 fixed-point representation.
pub const Q14_QUARTER: i16 = 4096;

/// Single LIF neuron update step (subtract-reset model).
///
/// Computes the new membrane potential after exponential decay and input
/// current integration, then checks for threshold crossing. If a spike
/// occurs, the threshold is subtracted from the membrane potential.
///
/// # Arguments
///
/// * `membrane` -- current membrane potential in Q1.14
/// * `alpha` -- decay factor in Q1.14 (e.g., 0.95 * 16384 = 15565)
/// * `input_current` -- total input current as i32 (pre-accumulated weighted spikes)
/// * `v_thr` -- firing threshold in Q1.14
///
/// # Returns
///
/// `(new_membrane, did_spike)` -- the updated membrane potential (clamped to i16 range)
/// and whether a spike was emitted.
#[inline]
pub fn lif_step(membrane: i16, alpha: i16, input_current: i32, v_thr: i16) -> (i16, bool) {
    // Exponential decay: V_new = alpha * V_old (Q1.14 * Q1.14 -> shift right 14)
    let decay = (membrane as i32 * alpha as i32) >> 14;
    // Integrate input current
    let v_new = decay + input_current;
    // Check threshold crossing
    let spike = v_new >= v_thr as i32;
    // Subtract-reset: if spiked, subtract threshold
    let v_reset = if spike { v_new - v_thr as i32 } else { v_new };
    // Clamp to i16 range to prevent overflow on subsequent steps
    (
        v_reset.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
        spike,
    )
}

/// Piecewise linear surrogate gradient for backpropagation through spikes.
///
/// The true gradient of the Heaviside spike function is zero almost everywhere
/// and infinite at the threshold. The surrogate gradient approximates this
/// with a triangular function centered on the threshold:
///
/// ```text
/// psi(V) = gamma * max(0, 1 - |V - V_thr| / V_thr)
/// ```
///
/// This produces a nonzero gradient in the window `[0, 2 * V_thr]`, enabling
/// gradient-based learning through spike events.
///
/// # Arguments
///
/// * `membrane` -- current membrane potential in Q1.14
/// * `v_thr` -- firing threshold in Q1.14 (must be > 0)
/// * `gamma` -- dampening factor in Q1.14 (controls gradient magnitude)
///
/// # Returns
///
/// Surrogate gradient value in Q1.14.
#[inline]
pub fn surrogate_gradient_pwl(membrane: i16, v_thr: i16, gamma: i16) -> i16 {
    debug_assert!(v_thr > 0, "v_thr must be positive for surrogate gradient");
    // Compute |membrane - v_thr| safely, handling potential overflow
    let diff = membrane as i32 - v_thr as i32;
    let abs_diff = if diff < 0 { -diff } else { diff };

    let v_thr_i32 = v_thr as i32;

    if abs_diff < v_thr_i32 {
        // Inside the triangular window: gamma * (v_thr - |diff|) / v_thr
        // All in Q1.14: (gamma * (v_thr - abs_diff)) / v_thr
        let numerator = gamma as i32 * (v_thr_i32 - abs_diff);
        (numerator / v_thr_i32) as i16
    } else {
        0
    }
}

/// Convert an f64 value to Q1.14 fixed-point representation.
///
/// Clamps to the representable range `[-2.0, ~+2.0)`.
#[inline]
pub fn f64_to_q14(value: f64) -> i16 {
    let scaled = value * Q14_ONE as f64;
    scaled.clamp(i16::MIN as f64, i16::MAX as f64) as i16
}

/// Convert a Q1.14 fixed-point value back to f64.
#[inline]
pub fn q14_to_f64(value: i16) -> f64 {
    value as f64 / Q14_ONE as f64
}

/// Fixed-point multiply: (a * b) >> 14, using i32 intermediate.
#[inline]
pub fn q14_mul(a: i16, b: i16) -> i16 {
    ((a as i32 * b as i32) >> 14) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subthreshold_input_no_spike() {
        // Membrane at 0, small input current below threshold
        let v_thr = Q14_HALF; // 0.5
        let alpha = f64_to_q14(0.95);
        let input = 1000_i32; // ~0.06 in Q1.14 — well below 0.5 threshold
        let (v_new, spike) = lif_step(0, alpha, input, v_thr);
        assert!(!spike, "should not spike with small input");
        assert_eq!(v_new, 1000, "membrane should be 0*alpha + 1000 = 1000");
    }

    #[test]
    fn suprathreshold_input_causes_spike_and_reset() {
        let v_thr = Q14_HALF; // 0.5
        let alpha = f64_to_q14(0.95);
        // Large input that pushes membrane above threshold
        let input = 10000_i32; // ~0.61
        let (v_new, spike) = lif_step(0, alpha, input, v_thr);
        assert!(spike, "should spike when input exceeds threshold");
        // After spike: 10000 - 8192 = 1808
        assert_eq!(v_new, 1808, "membrane should be reset by subtracting v_thr");
    }

    #[test]
    fn decay_reduces_membrane_without_input() {
        let alpha = f64_to_q14(0.5); // decay by half
        let v_thr = Q14_ONE;
        let membrane = Q14_HALF; // 0.5

        let (v_new, spike) = lif_step(membrane, alpha, 0, v_thr);
        assert!(!spike, "should not spike without input");
        // 0.5 * 0.5 = 0.25 in Q1.14 = 4096
        assert_eq!(v_new, Q14_QUARTER, "membrane should decay to 0.25");
    }

    #[test]
    fn accumulated_input_crosses_threshold() {
        let v_thr = Q14_HALF;
        let alpha = f64_to_q14(0.9);

        // Build up membrane over several timesteps
        let mut membrane: i16 = 0;
        let input = 2000_i32; // ~0.12 each step
        let mut total_spikes = 0;
        for _ in 0..20 {
            let (v, spike) = lif_step(membrane, alpha, input, v_thr);
            membrane = v;
            if spike {
                total_spikes += 1;
            }
        }
        assert!(
            total_spikes > 0,
            "should eventually spike after accumulating input over 20 steps"
        );
    }

    #[test]
    fn membrane_clamps_to_i16_range() {
        let alpha = Q14_ONE;
        let v_thr = Q14_ONE;
        // Huge input that would overflow i16
        let input = 60000_i32;
        let (v_new, _spike) = lif_step(i16::MAX, alpha, input, v_thr);
        // After spike reset, still clamped
        assert!(
            v_new <= i16::MAX && v_new >= i16::MIN,
            "membrane must be within i16 range"
        );
    }

    #[test]
    fn surrogate_gradient_peak_at_threshold() {
        let v_thr = Q14_HALF;
        let gamma = Q14_ONE; // gamma = 1.0

        // At membrane = v_thr, dist = 0, so psi = gamma * 1.0 = gamma
        let psi = surrogate_gradient_pwl(v_thr, v_thr, gamma);
        assert_eq!(
            psi, gamma,
            "surrogate gradient should peak at threshold: got {}",
            psi
        );
    }

    #[test]
    fn surrogate_gradient_zero_far_from_threshold() {
        let v_thr = Q14_HALF;
        let gamma = Q14_ONE;

        // Far above threshold
        let psi_high = surrogate_gradient_pwl(Q14_ONE + Q14_HALF, v_thr, gamma);
        assert_eq!(psi_high, 0, "should be zero far above threshold");

        // Far below threshold (negative membrane)
        let psi_low = surrogate_gradient_pwl(-Q14_HALF, v_thr, gamma);
        assert_eq!(psi_low, 0, "should be zero far below threshold");
    }

    #[test]
    fn surrogate_gradient_symmetric_around_threshold() {
        let v_thr = Q14_HALF;
        let gamma = Q14_ONE;
        let offset = 1000_i16;

        let psi_above = surrogate_gradient_pwl(v_thr + offset, v_thr, gamma);
        let psi_below = surrogate_gradient_pwl(v_thr - offset, v_thr, gamma);
        assert_eq!(
            psi_above, psi_below,
            "surrogate gradient should be symmetric around v_thr"
        );
    }

    #[test]
    fn q14_conversion_roundtrip() {
        let values = [0.0, 0.5, 1.0, -1.0, 0.95, -0.3];
        for &v in &values {
            let q = f64_to_q14(v);
            let back = q14_to_f64(q);
            assert!(
                (back - v).abs() < 0.001,
                "roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    #[test]
    fn q14_mul_correctness() {
        // 0.5 * 0.5 = 0.25
        let result = q14_mul(Q14_HALF, Q14_HALF);
        assert_eq!(result, Q14_QUARTER, "0.5 * 0.5 should be 0.25");

        // 1.0 * 0.5 = 0.5
        let result2 = q14_mul(Q14_ONE, Q14_HALF);
        assert_eq!(result2, Q14_HALF, "1.0 * 0.5 should be 0.5");
    }

    #[test]
    fn negative_membrane_decay() {
        let alpha = f64_to_q14(0.9);
        let v_thr = Q14_HALF;
        let membrane: i16 = -4000; // negative membrane
        let (v_new, spike) = lif_step(membrane, alpha, 0, v_thr);
        assert!(!spike, "negative membrane should not spike");
        // -4000 * 0.9 ~ -3600 (exact: (-4000 * 14746) >> 14 = -3600)
        let expected = ((-4000_i32 * alpha as i32) >> 14) as i16;
        assert_eq!(v_new, expected, "negative membrane should decay toward 0");
    }
}
