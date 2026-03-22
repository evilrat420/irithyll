//! Non-spiking leaky integrator readout neuron.
//!
//! The readout neuron accumulates weighted spike inputs from the hidden layer
//! without producing spikes itself. Its membrane potential serves as the
//! continuous output prediction of the network.
//!
//! ```text
//! y[t] = kappa_out * y[t-1] + sum(W_kj * z_j[t])
//! ```
//!
//! The membrane uses i32 precision to avoid accumulation overflow from many
//! input spikes over long sequences.

/// Non-spiking leaky integrator readout neuron.
///
/// The readout accumulates weighted spike inputs with exponential decay.
/// Its membrane potential (i32 for extra precision) is the network's
/// continuous output.
///
/// # Precision
///
/// While hidden layer neurons use i16 membranes, the readout uses i32 to
/// prevent overflow during long sequences. The decay factor `kappa` is
/// still Q1.14 (i16), but the membrane has ~18 bits of headroom above
/// the Q1.14 range.
pub struct ReadoutNeuron {
    /// Membrane potential (higher precision accumulator).
    /// Stored in Q1.14-compatible scale but with i32 range.
    pub membrane: i32,
    /// Decay factor in Q1.14 (controls how fast past inputs are forgotten).
    pub kappa: i16,
}

impl ReadoutNeuron {
    /// Create a new readout neuron with the given decay factor.
    ///
    /// # Arguments
    ///
    /// * `kappa` -- decay factor in Q1.14 (0 = no memory, Q14_ONE = perfect memory)
    pub fn new(kappa: i16) -> Self {
        Self { membrane: 0, kappa }
    }

    /// Advance the readout by one timestep.
    ///
    /// Decays the membrane potential and integrates new weighted input:
    ///
    /// ```text
    /// membrane = (membrane * kappa) >> 14 + weighted_input
    /// ```
    ///
    /// # Arguments
    ///
    /// * `weighted_input` -- sum of `W_kj * z_j[t]` for active spikes, as i32
    #[inline]
    pub fn step(&mut self, weighted_input: i32) {
        let decayed = (self.membrane as i64 * self.kappa as i64) >> 14;
        self.membrane = (decayed as i32).saturating_add(weighted_input);
    }

    /// Get the raw membrane potential as i32.
    #[inline]
    pub fn output_i32(&self) -> i32 {
        self.membrane
    }

    /// Dequantize the membrane potential to f64.
    ///
    /// # Arguments
    ///
    /// * `scale` -- output scaling factor (typically `1.0 / Q14_ONE as f64`)
    #[inline]
    pub fn output_f64(&self, scale: f64) -> f64 {
        self.membrane as f64 * scale
    }

    /// Reset the membrane potential to zero.
    pub fn reset(&mut self) {
        self.membrane = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snn::lif::{f64_to_q14, Q14_HALF, Q14_ONE};

    #[test]
    fn new_readout_has_zero_membrane() {
        let r = ReadoutNeuron::new(Q14_HALF);
        assert_eq!(r.membrane, 0);
        assert_eq!(r.output_i32(), 0);
    }

    #[test]
    fn step_accumulates_input() {
        let mut r = ReadoutNeuron::new(Q14_ONE); // no decay
        r.step(1000);
        assert_eq!(r.output_i32(), 1000);
        r.step(500);
        assert_eq!(r.output_i32(), 1500, "should accumulate with kappa=1.0");
    }

    #[test]
    fn step_decays_membrane() {
        let mut r = ReadoutNeuron::new(Q14_HALF); // 0.5 decay
        r.step(1000);
        assert_eq!(r.output_i32(), 1000);

        // Next step with zero input: 1000 * 0.5 = 500
        r.step(0);
        assert_eq!(r.output_i32(), 500, "membrane should decay by kappa");

        // Another step: 500 * 0.5 = 250
        r.step(0);
        assert_eq!(r.output_i32(), 250, "membrane should continue decaying");
    }

    #[test]
    fn output_f64_dequantizes_correctly() {
        let mut r = ReadoutNeuron::new(Q14_ONE);
        r.membrane = Q14_ONE as i32; // set to 1.0 in Q1.14
        let scale = 1.0 / Q14_ONE as f64;
        let out = r.output_f64(scale);
        assert!(
            (out - 1.0).abs() < 0.001,
            "output_f64 should dequantize 16384 to ~1.0, got {}",
            out
        );
    }

    #[test]
    fn reset_clears_membrane() {
        let mut r = ReadoutNeuron::new(Q14_HALF);
        r.step(5000);
        assert!(r.output_i32() != 0);
        r.reset();
        assert_eq!(r.output_i32(), 0, "reset should zero the membrane");
    }

    #[test]
    fn no_overflow_with_large_accumulation() {
        let mut r = ReadoutNeuron::new(f64_to_q14(0.99));
        // Feed many large inputs to test i32 accumulation
        for _ in 0..1000 {
            r.step(10000);
        }
        // Should not panic or wrap around
        assert!(
            r.output_i32() > 0,
            "membrane should be positive after positive inputs"
        );
    }

    #[test]
    fn decay_with_negative_membrane() {
        let mut r = ReadoutNeuron::new(Q14_HALF);
        r.step(-2000);
        assert_eq!(r.output_i32(), -2000);

        r.step(0);
        assert_eq!(
            r.output_i32(),
            -1000,
            "negative membrane should decay toward 0"
        );
    }

    #[test]
    fn zero_kappa_forgets_immediately() {
        let mut r = ReadoutNeuron::new(0); // zero decay
        r.step(5000);
        assert_eq!(r.output_i32(), 5000);

        r.step(0);
        assert_eq!(
            r.output_i32(),
            0,
            "zero kappa should forget membrane immediately"
        );
    }
}
