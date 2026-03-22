//! SNN-based streaming feature preprocessor.
//!
//! [`SpikePreprocessor`] uses a spiking neural network as a feature transformer,
//! implementing [`StreamingPreprocessor`] for use in irithyll pipelines. The SNN
//! processes input features and outputs a vector of readout membrane potentials
//! and hidden layer spike counts, creating temporally-aware feature representations.
//!
//! # Output Features
//!
//! The output vector has `n_hidden + n_outputs` elements:
//! - First `n_hidden` elements: spike indicators from hidden layer (0.0 or 1.0)
//! - Last `n_outputs` elements: readout membrane potentials (dequantized f64)
//!
//! This provides both discrete spike features and continuous readout features
//! to downstream learners.

use crate::pipeline::StreamingPreprocessor;
use irithyll_core::snn::lif::{f64_to_q14, Q14_HALF, Q14_ONE};
use irithyll_core::snn::network_fixed::{SpikeNetFixed, SpikeNetFixedConfig};

/// SNN-based streaming feature preprocessor.
///
/// Processes input features through a spiking neural network and outputs
/// spike counts and readout potentials as transformed features.
///
/// # Example
///
/// ```
/// use irithyll::snn::SpikePreprocessor;
/// use irithyll::pipeline::{Pipeline, StreamingPreprocessor};
/// use irithyll::learners::StreamingLinearModel;
/// use irithyll::StreamingLearner;
///
/// let mut pipeline = Pipeline::builder()
///     .pipe(SpikePreprocessor::new(16, 42))
///     .learner(StreamingLinearModel::new(0.01));
///
/// pipeline.train(&[1.0, 2.0, 3.0], 5.0);
/// let pred = pipeline.predict(&[1.0, 2.0, 3.0]);
/// ```
pub struct SpikePreprocessor {
    n_hidden: usize,
    n_outputs: usize,
    seed: u64,
    inner: Option<SpikeNetFixed>,
    input_scale: Vec<f64>,
    input_max_abs: Vec<f64>,
    quantized_buf: Vec<i16>,
    n_input: usize,
    n_samples: u64,
}

// SpikePreprocessor is Send + Sync because SpikeNetFixed is Send + Sync
unsafe impl Send for SpikePreprocessor {}
unsafe impl Sync for SpikePreprocessor {}

impl SpikePreprocessor {
    /// Create a new SpikePreprocessor with default SNN parameters.
    ///
    /// # Arguments
    ///
    /// * `n_hidden` -- number of hidden LIF neurons
    /// * `seed` -- PRNG seed for reproducible initialization
    pub fn new(n_hidden: usize, seed: u64) -> Self {
        Self {
            n_hidden,
            n_outputs: 1,
            seed,
            inner: None,
            input_scale: Vec::new(),
            input_max_abs: Vec::new(),
            quantized_buf: Vec::new(),
            n_input: 0,
            n_samples: 0,
        }
    }

    /// Create with a specified number of readout outputs.
    ///
    /// The total output dimension will be `n_hidden + n_outputs`.
    pub fn with_outputs(n_hidden: usize, n_outputs: usize, seed: u64) -> Self {
        Self {
            n_hidden,
            n_outputs,
            seed,
            inner: None,
            input_scale: Vec::new(),
            input_max_abs: Vec::new(),
            quantized_buf: Vec::new(),
            n_input: 0,
            n_samples: 0,
        }
    }

    fn initialize(&mut self, n_input: usize) {
        self.n_input = n_input;

        let config = SpikeNetFixedConfig {
            n_input,
            n_hidden: self.n_hidden,
            n_output: self.n_outputs,
            alpha: f64_to_q14(0.95),
            kappa: f64_to_q14(0.99),
            kappa_out: f64_to_q14(0.9),
            eta: 0, // no learning in preprocessor mode
            v_thr: f64_to_q14(0.5),
            gamma: f64_to_q14(0.3),
            spike_threshold: f64_to_q14(0.05),
            seed: self.seed,
            weight_init_range: f64_to_q14(0.1),
        };

        self.inner = Some(SpikeNetFixed::new(config));
        self.input_scale = vec![Q14_HALF as f64; n_input];
        self.input_max_abs = vec![1.0; n_input];
        self.quantized_buf = vec![0i16; n_input];
    }

    fn update_scaling(&mut self, features: &[f64]) {
        for (i, &feat) in features.iter().enumerate().take(self.n_input) {
            let abs_val = feat.abs();
            if abs_val > self.input_max_abs[i] {
                self.input_max_abs[i] = abs_val;
            }
        }
        for i in 0..self.n_input {
            if self.input_max_abs[i] > 1e-10 {
                self.input_scale[i] = Q14_HALF as f64 / self.input_max_abs[i];
            }
        }
    }

    fn quantize_input(&mut self, features: &[f64]) {
        for (i, &feat) in features.iter().enumerate().take(self.n_input) {
            let scaled = feat * self.input_scale[i];
            self.quantized_buf[i] = scaled.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        }
    }

    fn extract_features(&self) -> Vec<f64> {
        let inner = self.inner.as_ref().unwrap();
        let mut out = Vec::with_capacity(self.n_hidden + self.n_outputs);

        // Hidden spike indicators
        for &s in inner.hidden_spikes() {
            out.push(s as f64);
        }

        // Readout potentials
        let scale = 1.0 / Q14_ONE as f64;
        for v in inner.predict_all_f64(scale) {
            out.push(v);
        }

        out
    }
}

impl StreamingPreprocessor for SpikePreprocessor {
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        if self.inner.is_none() {
            self.initialize(features.len());
        }

        self.update_scaling(features);
        self.quantize_input(features);

        // Forward the network (updates state)
        let quantized: Vec<i16> = self.quantized_buf.clone();
        self.inner.as_mut().unwrap().forward(&quantized);
        self.n_samples += 1;

        self.extract_features()
    }

    fn transform(&self, _features: &[f64]) -> Vec<f64> {
        if self.inner.is_none() {
            // Not initialized yet -- return zeros of expected size
            return vec![0.0; self.n_hidden + self.n_outputs];
        }

        // Return current state without advancing the network
        self.extract_features()
    }

    fn output_dim(&self) -> Option<usize> {
        if self.inner.is_some() {
            Some(self.n_hidden + self.n_outputs)
        } else {
            None
        }
    }

    fn reset(&mut self) {
        if let Some(ref mut net) = self.inner {
            net.reset();
        }
        for v in self.input_max_abs.iter_mut() {
            *v = 1.0;
        }
        for v in self.input_scale.iter_mut() {
            *v = Q14_HALF as f64;
        }
        self.n_samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_initialization_on_first_update() {
        let mut pp = SpikePreprocessor::new(8, 42);
        assert!(pp.inner.is_none());
        assert_eq!(pp.output_dim(), None);

        let out = pp.update_and_transform(&[1.0, 2.0, 3.0]);
        assert!(pp.inner.is_some());
        assert_eq!(pp.output_dim(), Some(9)); // 8 hidden + 1 output
        assert_eq!(out.len(), 9);
    }

    #[test]
    fn transform_before_init_returns_zeros() {
        let pp = SpikePreprocessor::new(4, 42);
        let out = pp.transform(&[1.0, 2.0]);
        assert_eq!(out.len(), 5); // 4 hidden + 1 output
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn output_dim_matches_hidden_plus_output() {
        let mut pp = SpikePreprocessor::with_outputs(16, 3, 42);
        pp.update_and_transform(&[1.0, 2.0]);
        assert_eq!(pp.output_dim(), Some(19)); // 16 hidden + 3 outputs
    }

    #[test]
    fn spike_features_are_binary() {
        let mut pp = SpikePreprocessor::new(8, 42);

        // Need at least 2 calls for delta encoding to produce spikes
        pp.update_and_transform(&[0.0, 0.0]);
        let out = pp.update_and_transform(&[10.0, -10.0]);

        // First 8 elements are spike indicators (0.0 or 1.0)
        for i in 0..8 {
            assert!(
                out[i] == 0.0 || out[i] == 1.0,
                "spike feature {} should be 0.0 or 1.0, got {}",
                i,
                out[i]
            );
        }
    }

    #[test]
    fn reset_clears_preprocessor_state() {
        let mut pp = SpikePreprocessor::new(4, 42);
        pp.update_and_transform(&[1.0, 2.0]);
        pp.update_and_transform(&[3.0, 4.0]);
        assert!(pp.n_samples > 0);

        pp.reset();
        assert_eq!(pp.n_samples, 0);
    }

    #[test]
    fn multiple_timesteps_produce_different_features() {
        let mut pp = SpikePreprocessor::new(16, 42);

        // Drive the network with many steps of oscillating input
        // to give the SNN time to build up membrane potentials and produce spikes.
        let mut outputs = Vec::new();
        for step in 0..20 {
            let t = step as f64 * 0.5;
            let input = [t.sin() * 10.0, t.cos() * 10.0, (t * 2.0).sin() * 5.0];
            outputs.push(pp.update_and_transform(&input));
        }

        // At least one pair (not necessarily consecutive) should differ.
        let any_diff = outputs
            .windows(2)
            .any(|w| w[0].iter().zip(&w[1]).any(|(a, b)| (a - b).abs() > 1e-15));
        assert!(
            any_diff,
            "features should change across timesteps with varying oscillating inputs"
        );
    }

    #[test]
    fn transform_does_not_advance_state() {
        let mut pp = SpikePreprocessor::new(8, 42);
        pp.update_and_transform(&[1.0, 2.0]);
        pp.update_and_transform(&[3.0, 4.0]);

        // Multiple transform calls should return the same result
        let t1 = pp.transform(&[5.0, 6.0]);
        let t2 = pp.transform(&[5.0, 6.0]);
        assert_eq!(t1, t2, "transform should not change state");
    }

    #[test]
    fn works_in_pipeline() {
        use crate::learner::StreamingLearner;
        use crate::learners::StreamingLinearModel;
        use crate::pipeline::Pipeline;

        let mut pipeline = Pipeline::builder()
            .pipe(SpikePreprocessor::new(8, 42))
            .learner(StreamingLinearModel::new(0.01));

        pipeline.train(&[1.0, 2.0], 3.0);
        pipeline.train(&[4.0, 5.0], 6.0);

        let pred = pipeline.predict(&[2.5, 3.5]);
        assert!(pred.is_finite(), "pipeline prediction should be finite");
    }
}
