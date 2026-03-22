//! Complete SNN combining LIF neurons, delta encoding, e-prop learning, and readout.
//!
//! [`SpikeNetFixed`] is the core spiking neural network implementation. It
//! manages all neuron state, weights, eligibility traces, and spike encoding
//! in a single struct. The "Fixed" name refers to fixed-point arithmetic
//! (Q1.14), not fixed array sizes -- the network uses `Vec` for runtime-sized
//! buffers but maintains a constant memory footprint after construction.
//!
//! # Architecture
//!
//! ```text
//! Raw input (i16)
//!   |
//!   v
//! Delta Encoder -> spike_buf (2*N_IN binary spikes)
//!   |
//!   v
//! Hidden Layer (N_HID LIF neurons, recurrently connected)
//!   |  - w_input: spike_buf -> hidden
//!   |  - w_recurrent: hidden -> hidden (from previous timestep)
//!   |
//!   v
//! Readout Layer (N_OUT leaky integrators)
//!   |  - w_output: hidden spikes -> readout
//!   |
//!   v
//! Output (i32 membrane potentials, dequantized to f64)
//! ```
//!
//! # Memory Layout
//!
//! All weight matrices are stored as flat `Vec<i16>` in row-major order:
//! - `w_input[j * n_enc + i]` = weight from encoded input `i` to hidden neuron `j`
//! - `w_recurrent[j * n_hid + i]` = weight from hidden `i` to hidden `j`
//! - `w_output[k * n_hid + j]` = weight from hidden `j` to readout `k`
//! - `feedback[j * n_out + k]` = fixed random feedback from output `k` to hidden `j`

use alloc::vec;
use alloc::vec::Vec;

use super::eprop::{
    compute_learning_signal_fixed, update_eligibility_fixed, update_output_weights_fixed,
    update_pre_trace_fixed, update_weights_fixed,
};
use super::lif::{lif_step, surrogate_gradient_pwl};
use super::readout::ReadoutNeuron;
use super::spike_encoding::DeltaEncoderFixed;

/// Configuration for a `SpikeNetFixed` network.
///
/// All Q1.14 parameters should be in the range `[-2.0, +2.0)` when converted
/// from f64 via `f64_to_q14()`. Typical defaults:
///
/// | Parameter | f64 | Q1.14 |
/// |-----------|-----|-------|
/// | alpha | 0.95 | 15565 |
/// | kappa | 0.99 | 16220 |
/// | kappa_out | 0.90 | 14746 |
/// | eta | 0.001 | 16 |
/// | v_thr | 0.50 | 8192 |
/// | gamma | 0.30 | 4915 |
/// | spike_threshold | 0.05 | 819 |
#[derive(Debug, Clone)]
pub struct SpikeNetFixedConfig {
    /// Number of raw input features (encoded to 2x for spike channels).
    pub n_input: usize,
    /// Number of hidden LIF neurons.
    pub n_hidden: usize,
    /// Number of output readout neurons.
    pub n_output: usize,
    /// Membrane decay factor in Q1.14.
    pub alpha: i16,
    /// Eligibility trace decay factor in Q1.14.
    pub kappa: i16,
    /// Readout membrane decay factor in Q1.14.
    pub kappa_out: i16,
    /// Learning rate in Q1.14.
    pub eta: i16,
    /// Firing threshold in Q1.14.
    pub v_thr: i16,
    /// Surrogate gradient dampening factor in Q1.14.
    pub gamma: i16,
    /// Delta encoding threshold in Q1.14.
    pub spike_threshold: i16,
    /// PRNG seed for reproducible weight initialization.
    pub seed: u64,
    /// Weight initialization range: weights sampled from `[-range, +range]` in Q1.14.
    pub weight_init_range: i16,
}

impl Default for SpikeNetFixedConfig {
    fn default() -> Self {
        Self {
            n_input: 1,
            n_hidden: 64,
            n_output: 1,
            alpha: 15565,         // 0.95
            kappa: 16220,         // 0.99
            kappa_out: 14746,     // 0.90
            eta: 16,              // 0.001
            v_thr: 8192,          // 0.50
            gamma: 4915,          // 0.30
            spike_threshold: 819, // 0.05
            seed: 42,
            weight_init_range: 1638, // 0.10
        }
    }
}

/// Inline xorshift64 PRNG for weight initialization.
///
/// Returns the next pseudo-random u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a random i16 in `[-range, +range]` from the PRNG.
///
/// `range` should be positive. If zero, returns 0.
#[inline]
fn xorshift64_i16(state: &mut u64, range: i16) -> i16 {
    let raw = xorshift64(state);
    let abs_range = if range < 0 { -range } else { range };
    if abs_range == 0 {
        return 0;
    }
    let abs_u64 = abs_range as u64;
    let modulus = 2 * abs_u64 + 1;
    ((raw % modulus) as i16) - abs_range
}

/// Complete spiking neural network with e-prop online learning.
///
/// Manages all neuron state, synaptic weights, eligibility traces, and spike
/// encoding. After construction, memory is fixed -- no further allocations
/// occur during `forward` or `train_step`.
///
/// # Thread Safety
///
/// `SpikeNetFixed` is `Send + Sync` because it contains only `Vec<T>` and
/// primitive fields with no interior mutability.
pub struct SpikeNetFixed {
    config: SpikeNetFixedConfig,
    n_input_encoded: usize, // 2 * n_input

    // --- Neuron state ---
    membrane: Vec<i16>,   // [n_hidden]
    spikes: Vec<u8>,      // [n_hidden]
    prev_spikes: Vec<u8>, // [n_hidden] previous timestep for recurrent

    // --- Presynaptic traces ---
    pre_trace_in: Vec<i16>,  // [n_input_encoded]
    pre_trace_hid: Vec<i16>, // [n_hidden]

    // --- Weights (row-major) ---
    w_input: Vec<i16>,     // [n_hidden * n_input_encoded]
    w_recurrent: Vec<i16>, // [n_hidden * n_hidden]
    w_output: Vec<i16>,    // [n_output * n_hidden]
    feedback: Vec<i16>,    // [n_hidden * n_output] (fixed random, never updated)

    // --- Eligibility traces ---
    elig_in: Vec<i16>,  // [n_hidden * n_input_encoded]
    elig_rec: Vec<i16>, // [n_hidden * n_hidden]

    // --- Readout ---
    readout: Vec<ReadoutNeuron>, // [n_output]

    // --- Encoder ---
    encoder: DeltaEncoderFixed,

    // --- Spike buffer ---
    spike_buf: Vec<u8>, // [n_input_encoded]

    // --- Error buffer (reusable) ---
    error_buf: Vec<i16>, // [n_output]

    // --- Counters ---
    n_samples: u64,
}

// Safety: SpikeNetFixed contains only Vec, primitives, and other Send+Sync types
unsafe impl Send for SpikeNetFixed {}
unsafe impl Sync for SpikeNetFixed {}

impl SpikeNetFixed {
    /// Create a new SpikeNetFixed with the given configuration.
    ///
    /// Allocates all internal buffers and initializes weights from the PRNG.
    /// No further allocations occur during operation.
    pub fn new(config: SpikeNetFixedConfig) -> Self {
        let n_in = config.n_input;
        let n_hid = config.n_hidden;
        let n_out = config.n_output;
        let n_enc = 2 * n_in;

        let mut rng_state = if config.seed == 0 { 1 } else { config.seed };
        let range = config.weight_init_range;

        // Initialize input weights
        let w_input: Vec<i16> = (0..n_hid * n_enc)
            .map(|_| xorshift64_i16(&mut rng_state, range))
            .collect();

        // Initialize recurrent weights
        let w_recurrent: Vec<i16> = (0..n_hid * n_hid)
            .map(|_| xorshift64_i16(&mut rng_state, range))
            .collect();

        // Initialize output weights
        let w_output: Vec<i16> = (0..n_out * n_hid)
            .map(|_| xorshift64_i16(&mut rng_state, range))
            .collect();

        // Initialize fixed random feedback weights
        let feedback: Vec<i16> = (0..n_hid * n_out)
            .map(|_| xorshift64_i16(&mut rng_state, range))
            .collect();

        let readout: Vec<ReadoutNeuron> = (0..n_out)
            .map(|_| ReadoutNeuron::new(config.kappa_out))
            .collect();

        let encoder = DeltaEncoderFixed::new(n_in, config.spike_threshold);

        Self {
            n_input_encoded: n_enc,
            membrane: vec![0; n_hid],
            spikes: vec![0; n_hid],
            prev_spikes: vec![0; n_hid],
            pre_trace_in: vec![0; n_enc],
            pre_trace_hid: vec![0; n_hid],
            w_input,
            w_recurrent,
            w_output,
            feedback,
            elig_in: vec![0; n_hid * n_enc],
            elig_rec: vec![0; n_hid * n_hid],
            readout,
            encoder,
            spike_buf: vec![0; n_enc],
            error_buf: vec![0; n_out],
            n_samples: 0,
            config,
        }
    }

    /// Run one forward timestep without learning.
    ///
    /// Encodes input into spikes, updates hidden neuron states, and advances
    /// the readout. Does NOT update weights or eligibility traces.
    ///
    /// # Arguments
    ///
    /// * `input_i16` -- raw input features in Q1.14, length must equal `config.n_input`
    pub fn forward(&mut self, input_i16: &[i16]) {
        let n_hid = self.config.n_hidden;
        let n_enc = self.n_input_encoded;

        // 1. Delta-encode input into spike buffer
        self.encoder.encode(input_i16, &mut self.spike_buf);

        // 2. Store previous spikes for recurrent computation
        self.prev_spikes.copy_from_slice(&self.spikes);

        // 3. Update hidden layer LIF neurons
        for j in 0..n_hid {
            // Compute input current from encoded spikes
            let mut current: i32 = 0;
            let w_in_offset = j * n_enc;
            for i in 0..n_enc {
                if self.spike_buf[i] != 0 {
                    current += self.w_input[w_in_offset + i] as i32;
                }
            }

            // Add recurrent current from previous hidden spikes
            let w_rec_offset = j * n_hid;
            for i in 0..n_hid {
                if self.prev_spikes[i] != 0 {
                    current += self.w_recurrent[w_rec_offset + i] as i32;
                }
            }

            // LIF step
            let (v_new, spike) = lif_step(
                self.membrane[j],
                self.config.alpha,
                current,
                self.config.v_thr,
            );
            self.membrane[j] = v_new;
            self.spikes[j] = spike as u8;
        }

        // 4. Update readout neurons
        let n_out = self.config.n_output;
        for k in 0..n_out {
            let w_out_offset = k * n_hid;
            let mut weighted_input: i32 = 0;
            for j in 0..n_hid {
                if self.spikes[j] != 0 {
                    weighted_input += self.w_output[w_out_offset + j] as i32;
                }
            }
            self.readout[k].step(weighted_input);
        }
    }

    /// Run one forward + learning timestep (e-prop three-factor rule).
    ///
    /// Performs a forward pass, then computes error signals and updates
    /// all weights using the e-prop learning rule.
    ///
    /// # Arguments
    ///
    /// * `input_i16` -- raw input features in Q1.14
    /// * `target_i16` -- target values in Q1.14, length must equal `config.n_output`
    pub fn train_step(&mut self, input_i16: &[i16], target_i16: &[i16]) {
        let n_hid = self.config.n_hidden;
        let n_enc = self.n_input_encoded;
        let n_out = self.config.n_output;

        // 1. Forward pass
        self.forward(input_i16);

        // 2. Compute error signals: error = target - readout
        for (k, &target_k) in target_i16.iter().enumerate().take(n_out) {
            let readout_clamped = self.readout[k]
                .output_i32()
                .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            self.error_buf[k] = target_k.saturating_sub(readout_clamped);
        }

        // 3. Update presynaptic traces
        update_pre_trace_fixed(&mut self.pre_trace_in, &self.spike_buf, self.config.alpha);
        update_pre_trace_fixed(&mut self.pre_trace_hid, &self.spikes, self.config.alpha);

        // 4. For each hidden neuron: update eligibility, compute learning signal, update weights
        for j in 0..n_hid {
            // Surrogate gradient
            let psi =
                surrogate_gradient_pwl(self.membrane[j], self.config.v_thr, self.config.gamma);

            // Update input eligibility traces for neuron j
            let elig_in_start = j * n_enc;
            let elig_in_end = elig_in_start + n_enc;
            update_eligibility_fixed(
                &mut self.elig_in[elig_in_start..elig_in_end],
                psi,
                &self.pre_trace_in,
                self.config.kappa,
            );

            // Update recurrent eligibility traces for neuron j
            let elig_rec_start = j * n_hid;
            let elig_rec_end = elig_rec_start + n_hid;
            update_eligibility_fixed(
                &mut self.elig_rec[elig_rec_start..elig_rec_end],
                psi,
                &self.pre_trace_hid,
                self.config.kappa,
            );

            // Compute learning signal via feedback alignment
            let fb_start = j * n_out;
            let fb_end = fb_start + n_out;
            let learning_signal = compute_learning_signal_fixed(
                &self.feedback[fb_start..fb_end],
                &self.error_buf[..n_out],
            );

            // Update input weights for neuron j
            let w_in_start = j * n_enc;
            let w_in_end = w_in_start + n_enc;
            update_weights_fixed(
                &mut self.w_input[w_in_start..w_in_end],
                &self.elig_in[elig_in_start..elig_in_end],
                learning_signal,
                self.config.eta,
            );

            // Update recurrent weights for neuron j
            let w_rec_start = j * n_hid;
            let w_rec_end = w_rec_start + n_hid;
            update_weights_fixed(
                &mut self.w_recurrent[w_rec_start..w_rec_end],
                &self.elig_rec[elig_rec_start..elig_rec_end],
                learning_signal,
                self.config.eta,
            );
        }

        // 5. Update output weights via delta rule
        for k in 0..n_out {
            let w_out_start = k * n_hid;
            let w_out_end = w_out_start + n_hid;
            update_output_weights_fixed(
                &mut self.w_output[w_out_start..w_out_end],
                self.error_buf[k],
                &self.spikes,
                self.config.eta,
            );
        }

        self.n_samples += 1;
    }

    /// Get raw readout membrane potentials as i32.
    ///
    /// Returns a reference to an internal buffer that is updated on each
    /// `forward` or `train_step` call.
    pub fn predict_raw(&self) -> Vec<i32> {
        self.readout.iter().map(|r| r.output_i32()).collect()
    }

    /// Get the first readout's membrane potential, dequantized to f64.
    ///
    /// # Arguments
    ///
    /// * `output_scale` -- scaling factor (typically `1.0 / Q14_ONE as f64`)
    pub fn predict_f64(&self, output_scale: f64) -> f64 {
        if self.readout.is_empty() {
            return 0.0;
        }
        self.readout[0].output_f64(output_scale)
    }

    /// Get all readout membrane potentials, dequantized to f64.
    ///
    /// # Arguments
    ///
    /// * `output_scale` -- scaling factor per output
    pub fn predict_all_f64(&self, output_scale: f64) -> Vec<f64> {
        self.readout
            .iter()
            .map(|r| r.output_f64(output_scale))
            .collect()
    }

    /// Number of training samples seen.
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Reference to the network configuration.
    pub fn config(&self) -> &SpikeNetFixedConfig {
        &self.config
    }

    /// Number of hidden neurons.
    pub fn n_hidden(&self) -> usize {
        self.config.n_hidden
    }

    /// Number of encoded input channels (2 * n_input).
    pub fn n_input_encoded(&self) -> usize {
        self.n_input_encoded
    }

    /// Get current hidden spike vector.
    pub fn hidden_spikes(&self) -> &[u8] {
        &self.spikes
    }

    /// Get current hidden membrane potentials.
    pub fn hidden_membrane(&self) -> &[i16] {
        &self.membrane
    }

    /// Compute total memory usage in bytes.
    ///
    /// Counts all Vec contents plus struct overhead.
    pub fn memory_bytes(&self) -> usize {
        let n_hid = self.config.n_hidden;
        let n_enc = self.n_input_encoded;
        let n_out = self.config.n_output;
        let n_in = self.config.n_input;

        let size_of_i16 = core::mem::size_of::<i16>();
        let size_of_u8 = core::mem::size_of::<u8>();

        // Neuron state
        let membrane = n_hid * size_of_i16;
        let spikes = n_hid * size_of_u8;
        let prev_spikes = n_hid * size_of_u8;

        // Presynaptic traces
        let pre_trace_in = n_enc * size_of_i16;
        let pre_trace_hid = n_hid * size_of_i16;

        // Weights
        let w_input = n_hid * n_enc * size_of_i16;
        let w_recurrent = n_hid * n_hid * size_of_i16;
        let w_output = n_out * n_hid * size_of_i16;
        let feedback = n_hid * n_out * size_of_i16;

        // Eligibility traces
        let elig_in = n_hid * n_enc * size_of_i16;
        let elig_rec = n_hid * n_hid * size_of_i16;

        // Readout (membrane i32 + kappa i16 + padding)
        let readout_size = n_out * core::mem::size_of::<ReadoutNeuron>();

        // Encoder state
        let encoder_prev = n_in * size_of_i16;
        let encoder_thr = n_in * size_of_i16;

        // Spike buffer
        let spike_buf = n_enc * size_of_u8;

        // Error buffer
        let error_buf = n_out * size_of_i16;

        // Struct overhead (approximate)
        let struct_overhead = core::mem::size_of::<Self>();

        // Total Vec contents
        let vec_contents = membrane
            + spikes
            + prev_spikes
            + pre_trace_in
            + pre_trace_hid
            + w_input
            + w_recurrent
            + w_output
            + feedback
            + elig_in
            + elig_rec
            + readout_size
            + encoder_prev
            + encoder_thr
            + spike_buf
            + error_buf;

        struct_overhead + vec_contents
    }

    /// Reset all network state (neuron potentials, traces, readout) to zero.
    ///
    /// Weights are re-initialized from the original seed. The network behaves
    /// as if freshly constructed after calling reset.
    pub fn reset(&mut self) {
        // Zero neuron state
        for v in self.membrane.iter_mut() {
            *v = 0;
        }
        for s in self.spikes.iter_mut() {
            *s = 0;
        }
        for s in self.prev_spikes.iter_mut() {
            *s = 0;
        }

        // Zero traces
        for t in self.pre_trace_in.iter_mut() {
            *t = 0;
        }
        for t in self.pre_trace_hid.iter_mut() {
            *t = 0;
        }
        for e in self.elig_in.iter_mut() {
            *e = 0;
        }
        for e in self.elig_rec.iter_mut() {
            *e = 0;
        }

        // Reset readout
        for r in self.readout.iter_mut() {
            r.reset();
        }

        // Reset encoder
        self.encoder.reset();

        // Zero spike buffer
        for s in self.spike_buf.iter_mut() {
            *s = 0;
        }

        // Zero error buffer
        for e in self.error_buf.iter_mut() {
            *e = 0;
        }

        // Re-initialize weights from seed
        let mut rng_state = if self.config.seed == 0 {
            1
        } else {
            self.config.seed
        };
        let range = self.config.weight_init_range;

        for w in self.w_input.iter_mut() {
            *w = xorshift64_i16(&mut rng_state, range);
        }
        for w in self.w_recurrent.iter_mut() {
            *w = xorshift64_i16(&mut rng_state, range);
        }
        for w in self.w_output.iter_mut() {
            *w = xorshift64_i16(&mut rng_state, range);
        }
        for w in self.feedback.iter_mut() {
            *w = xorshift64_i16(&mut rng_state, range);
        }

        self.n_samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snn::lif::{f64_to_q14, Q14_ONE};

    fn default_small_config() -> SpikeNetFixedConfig {
        SpikeNetFixedConfig {
            n_input: 2,
            n_hidden: 8,
            n_output: 1,
            alpha: f64_to_q14(0.95),
            kappa: f64_to_q14(0.99),
            kappa_out: f64_to_q14(0.9),
            eta: f64_to_q14(0.01),
            v_thr: f64_to_q14(0.5),
            gamma: f64_to_q14(0.3),
            spike_threshold: f64_to_q14(0.05),
            seed: 42,
            weight_init_range: f64_to_q14(0.1),
        }
    }

    #[test]
    fn construction_initializes_all_buffers() {
        let config = default_small_config();
        let net = SpikeNetFixed::new(config);

        assert_eq!(net.membrane.len(), 8);
        assert_eq!(net.spikes.len(), 8);
        assert_eq!(net.n_input_encoded(), 4);
        assert_eq!(net.w_input.len(), 8 * 4);
        assert_eq!(net.w_recurrent.len(), 8 * 8);
        assert_eq!(net.w_output.len(), 8);
        assert_eq!(net.feedback.len(), 8);
        assert_eq!(net.elig_in.len(), 8 * 4);
        assert_eq!(net.elig_rec.len(), 8 * 8);
        assert_eq!(net.readout.len(), 1);
        assert_eq!(net.n_samples_seen(), 0);
    }

    #[test]
    fn forward_does_not_crash() {
        let config = default_small_config();
        let mut net = SpikeNetFixed::new(config);

        // First call (encoder warmup)
        net.forward(&[f64_to_q14(0.5), f64_to_q14(-0.3)]);
        // Second call (actual spikes possible)
        net.forward(&[f64_to_q14(0.8), f64_to_q14(0.2)]);

        // Should produce some output
        let raw = net.predict_raw();
        assert_eq!(raw.len(), 1, "should have one readout output");
    }

    #[test]
    fn train_step_increments_counter() {
        let config = default_small_config();
        let mut net = SpikeNetFixed::new(config);

        let input = [f64_to_q14(0.5), f64_to_q14(-0.3)];
        let target = [f64_to_q14(0.7)];

        net.train_step(&input, &target);
        assert_eq!(net.n_samples_seen(), 1);

        net.train_step(&input, &target);
        assert_eq!(net.n_samples_seen(), 2);
    }

    #[test]
    fn predictions_change_after_training() {
        let config = SpikeNetFixedConfig {
            n_input: 2,
            n_hidden: 16,
            n_output: 1,
            alpha: f64_to_q14(0.9),
            kappa: f64_to_q14(0.95),
            kappa_out: f64_to_q14(0.85),
            eta: f64_to_q14(0.05),  // larger learning rate for visible change
            v_thr: f64_to_q14(0.3), // lower threshold for more spiking
            gamma: f64_to_q14(0.5),
            spike_threshold: f64_to_q14(0.01), // very sensitive encoding
            seed: 12345,
            weight_init_range: f64_to_q14(0.2),
        };

        let mut net = SpikeNetFixed::new(config);
        let scale = 1.0 / Q14_ONE as f64;

        // Warm up encoder
        net.forward(&[0, 0]);
        let pred_before = net.predict_f64(scale);

        // Train on a pattern for many steps
        for step in 0..200 {
            let x = if step % 2 == 0 {
                [f64_to_q14(0.8), f64_to_q14(-0.5)]
            } else {
                [f64_to_q14(-0.3), f64_to_q14(0.6)]
            };
            let target = if step % 2 == 0 {
                [f64_to_q14(1.0)]
            } else {
                [f64_to_q14(-1.0)]
            };
            net.train_step(&x, &target);
        }

        let pred_after = net.predict_f64(scale);

        assert!(
            (pred_after - pred_before).abs() > 1e-10,
            "prediction should change after training: before={}, after={}",
            pred_before,
            pred_after
        );
    }

    #[test]
    fn reset_restores_initial_state() {
        let config = default_small_config();
        let mut net = SpikeNetFixed::new(config.clone());
        let fresh = SpikeNetFixed::new(config);

        // Train a few steps
        net.train_step(&[1000, -500], &[2000]);
        net.train_step(&[-1000, 500], &[-2000]);
        assert!(net.n_samples_seen() > 0);

        // Reset
        net.reset();

        // Compare with fresh network
        assert_eq!(net.n_samples_seen(), 0);
        assert_eq!(net.membrane, fresh.membrane);
        assert_eq!(net.spikes, fresh.spikes);
        assert_eq!(
            net.w_input, fresh.w_input,
            "weights should be re-initialized from seed"
        );
        assert_eq!(net.w_recurrent, fresh.w_recurrent);
        assert_eq!(net.w_output, fresh.w_output);
        assert_eq!(net.feedback, fresh.feedback);
    }

    #[test]
    fn memory_bytes_is_reasonable() {
        let config = SpikeNetFixedConfig {
            n_input: 10,
            n_hidden: 64,
            n_output: 1,
            ..SpikeNetFixedConfig::default()
        };
        let net = SpikeNetFixed::new(config);
        let mem = net.memory_bytes();

        // Dominant terms: w_recurrent = 64*64*2 = 8192 bytes
        // w_input = 64*20*2 = 2560 bytes
        // elig_rec = 64*64*2 = 8192 bytes
        // elig_in = 64*20*2 = 2560 bytes
        // Total Vec contents should be ~22KB + struct overhead
        assert!(
            mem > 20_000,
            "memory should be at least 20KB for 10-in/64-hid/1-out, got {}",
            mem
        );
        assert!(
            mem < 100_000,
            "memory should be under 100KB for small network, got {}",
            mem
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let config = default_small_config();
        let mut net1 = SpikeNetFixed::new(config.clone());
        let mut net2 = SpikeNetFixed::new(config);

        let input = [f64_to_q14(0.3), f64_to_q14(-0.7)];
        let target = [f64_to_q14(0.5)];

        for _ in 0..10 {
            net1.train_step(&input, &target);
            net2.train_step(&input, &target);
        }

        let scale = 1.0 / Q14_ONE as f64;
        let p1 = net1.predict_f64(scale);
        let p2 = net2.predict_f64(scale);
        assert_eq!(p1, p2, "same seed should produce identical predictions");
    }

    #[test]
    fn multi_output_network() {
        let config = SpikeNetFixedConfig {
            n_input: 3,
            n_hidden: 8,
            n_output: 3,
            ..SpikeNetFixedConfig::default()
        };
        let mut net = SpikeNetFixed::new(config);

        net.forward(&[1000, -500, 200]);
        net.forward(&[1500, 0, -300]);

        let raw = net.predict_raw();
        assert_eq!(raw.len(), 3, "should have 3 readout outputs");

        let scale = 1.0 / Q14_ONE as f64;
        let all = net.predict_all_f64(scale);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn train_step_with_multi_output() {
        let config = SpikeNetFixedConfig {
            n_input: 2,
            n_hidden: 8,
            n_output: 2,
            ..SpikeNetFixedConfig::default()
        };
        let mut net = SpikeNetFixed::new(config);

        // Should not panic
        net.train_step(&[1000, -500], &[2000, -1000]);
        assert_eq!(net.n_samples_seen(), 1);
    }

    #[test]
    fn hidden_spikes_accessible() {
        let config = default_small_config();
        let mut net = SpikeNetFixed::new(config);

        net.forward(&[0, 0]);
        net.forward(&[Q14_ONE, -Q14_ONE]); // big change to trigger spikes

        let spikes = net.hidden_spikes();
        assert_eq!(spikes.len(), 8);
        // Spikes are binary
        for &s in spikes {
            assert!(s == 0 || s == 1, "spike should be 0 or 1, got {}", s);
        }
    }

    #[test]
    fn config_default_is_sensible() {
        let config = SpikeNetFixedConfig::default();
        assert!(config.alpha > 0, "alpha should be positive");
        assert!(config.v_thr > 0, "v_thr should be positive");
        assert!(config.eta > 0, "eta should be positive");
        assert!(config.n_hidden > 0, "n_hidden should be positive");
    }
}
