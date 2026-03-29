//! f64-interfaced SNN wrapping `SpikeNetFixed` with automatic scaling.
//!
//! [`SpikeNet`] bridges the gap between irithyll's f64 [`StreamingLearner`]
//! interface and the Q1.14 fixed-point core. It handles:
//!
//! - Lazy initialization (input dimension discovered from first sample)
//! - Adaptive input scaling (EWMA max tracker to map features into Q1.14 range)
//! - Target quantization and output dequantization
//! - Sample weight support via learning rate modulation
//!
//! # Scaling Strategy
//!
//! During the first 50 samples, a running maximum absolute value is tracked
//! per feature. After warmup, `input_scale[i] = Q14_HALF / max_abs[i]`,
//! ensuring features map to roughly `[-0.5, +0.5]` in Q1.14. An EWMA
//! with decay 0.99 allows the scale to adapt to non-stationary distributions.

use crate::learner::StreamingLearner;
use irithyll_core::snn::lif::{f64_to_q14, Q14_HALF, Q14_ONE};
use irithyll_core::snn::network_fixed::{SpikeNetFixed, SpikeNetFixedConfig};

use super::spikenet_config::SpikeNetConfig;

/// Warmup period: number of samples before input scaling stabilizes.
const WARMUP_SAMPLES: u64 = 50;

/// EWMA decay for the running max tracker.
const MAX_EWMA_DECAY: f64 = 0.99;

/// Streaming SpikeNet implementing [`StreamingLearner`].
///
/// Wraps a `SpikeNetFixed` (Q1.14 integer SNN) with f64 input/output
/// conversion and adaptive scaling.
///
/// # Lazy Initialization
///
/// The input dimension is unknown until the first `train_one` or `predict`
/// call, at which point the underlying `SpikeNetFixed` is constructed.
///
/// # Example
///
/// ```
/// use irithyll::snn::{SpikeNet, SpikeNetConfig};
/// use irithyll::StreamingLearner;
///
/// let config = SpikeNetConfig::builder()
///     .n_hidden(32)
///     .learning_rate(0.005)
///     .build()
///     .unwrap();
///
/// let mut model = SpikeNet::new(config);
///
/// // First call discovers input dimension (3 features)
/// model.train(&[0.5, -0.3, 0.8], 1.0);
/// let pred = model.predict(&[0.5, -0.3, 0.8]);
/// ```
pub struct SpikeNet {
    config: SpikeNetConfig,
    inner: Option<SpikeNetFixed>,

    // f64 <-> i16 scaling
    input_scale: Vec<f64>,
    input_max_abs: Vec<f64>,
    output_scale: f64,
    target_scale: f64,

    // Reusable quantization buffer
    quantized_input: Vec<i16>,
    quantized_target: Vec<i16>,

    n_samples: u64,
    n_input: usize,
}

// SpikeNet is Send + Sync because SpikeNetFixed is Send + Sync and all
// other fields are plain Vec/f64.
unsafe impl Send for SpikeNet {}
unsafe impl Sync for SpikeNet {}

impl SpikeNet {
    /// Create a new SpikeNet with the given configuration.
    ///
    /// The underlying network is not allocated until the first sample arrives
    /// (lazy initialization), because the input dimension is unknown.
    pub fn new(config: SpikeNetConfig) -> Self {
        Self {
            config,
            inner: None,
            input_scale: Vec::new(),
            input_max_abs: Vec::new(),
            output_scale: 1.0 / Q14_ONE as f64,
            target_scale: Q14_HALF as f64, // map target 1.0 -> Q14_HALF
            quantized_input: Vec::new(),
            quantized_target: Vec::new(),
            n_samples: 0,
            n_input: 0,
        }
    }

    /// Create with a known input dimension (avoids lazy init).
    ///
    /// Useful when the feature count is known ahead of time.
    pub fn with_n_input(config: SpikeNetConfig, n_input: usize) -> Self {
        let mut net = Self::new(config);
        net.initialize(n_input);
        net
    }

    /// Initialize the underlying network for the given input dimension.
    fn initialize(&mut self, n_input: usize) {
        self.n_input = n_input;

        let fixed_config = SpikeNetFixedConfig {
            n_input,
            n_hidden: self.config.n_hidden,
            n_output: self.config.n_outputs,
            alpha: f64_to_q14(self.config.alpha),
            kappa: f64_to_q14(self.config.kappa),
            kappa_out: f64_to_q14(self.config.kappa_out),
            eta: f64_to_q14(self.config.eta),
            v_thr: f64_to_q14(self.config.v_thr),
            gamma: f64_to_q14(self.config.gamma),
            spike_threshold: f64_to_q14(self.config.spike_threshold),
            seed: self.config.seed,
            weight_init_range: f64_to_q14(self.config.weight_init_range),
        };

        self.inner = Some(SpikeNetFixed::new(fixed_config));

        // Initialize scaling: start with unit scale, adapt during warmup
        self.input_scale = vec![Q14_HALF as f64; n_input];
        self.input_max_abs = vec![1.0; n_input]; // avoid division by zero

        self.quantized_input = vec![0i16; n_input];
        self.quantized_target = vec![0i16; self.config.n_outputs];
    }

    /// Update the running max absolute value per feature and recompute scales.
    fn update_input_scaling(&mut self, features: &[f64]) {
        for (i, &feat) in features.iter().enumerate().take(self.n_input) {
            let abs_val = feat.abs();
            if abs_val > self.input_max_abs[i] {
                // During warmup: take the max directly
                self.input_max_abs[i] = abs_val;
            } else if self.n_samples >= WARMUP_SAMPLES {
                // After warmup: EWMA for gradual adaptation
                self.input_max_abs[i] =
                    MAX_EWMA_DECAY * self.input_max_abs[i] + (1.0 - MAX_EWMA_DECAY) * abs_val;
            }
        }

        // Recompute scales: map [-max_abs, +max_abs] -> [-Q14_HALF, +Q14_HALF]
        for i in 0..self.n_input {
            if self.input_max_abs[i] > 1e-10 {
                self.input_scale[i] = Q14_HALF as f64 / self.input_max_abs[i];
            }
        }
    }

    /// Quantize f64 features to i16 using current scales.
    fn quantize_input(&mut self, features: &[f64]) {
        for (i, &feat) in features.iter().enumerate().take(self.n_input) {
            let scaled = feat * self.input_scale[i];
            self.quantized_input[i] = scaled.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        }
    }

    /// Quantize f64 target to i16.
    fn quantize_target(&mut self, target: f64) {
        let scaled = target * self.target_scale;
        self.quantized_target[0] = scaled.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
    }

    /// Whether the network has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.inner.is_some()
    }

    /// Number of input features (0 if not yet initialized).
    pub fn n_input(&self) -> usize {
        self.n_input
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &SpikeNetConfig {
        &self.config
    }

    /// Total memory usage in bytes (0 if not initialized).
    pub fn memory_bytes(&self) -> usize {
        match &self.inner {
            Some(net) => net.memory_bytes(),
            None => 0,
        }
    }

    /// Access the underlying fixed-point network (if initialized).
    pub fn inner(&self) -> Option<&SpikeNetFixed> {
        self.inner.as_ref()
    }
}

impl StreamingLearner for SpikeNet {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // Lazy initialization on first call
        if self.inner.is_none() {
            self.initialize(features.len());
        }

        assert_eq!(
            features.len(),
            self.n_input,
            "feature dimension mismatch: expected {}, got {}",
            self.n_input,
            features.len()
        );

        // Update input scaling
        self.update_input_scaling(features);

        // Quantize inputs
        self.quantize_input(features);

        // Quantize target
        self.quantize_target(target);

        // Apply sample weight: for SNNs with fixed-point internals, weight
        // modulation is approximated. Weight ~0 skips the update, weight >0
        // trains once with the base learning rate. The fixed-point config
        // does not support per-sample eta changes without mutable config access.
        if let Some(ref mut net) = self.inner {
            if weight > 1e-10 {
                net.train_step(&self.quantized_input, &self.quantized_target);
            }
            // weight ~0 means skip training for this sample
        }

        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let inner = match &self.inner {
            Some(net) => net,
            None => return 0.0,
        };

        if features.len() != self.n_input {
            return 0.0;
        }

        // Use current readout state without advancing the network.
        // This is side-effect-free: we just read the current readout membrane.
        inner.predict_f64(self.output_scale)
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        if let Some(ref mut net) = self.inner {
            net.reset();
        }
        // Reset scaling
        for v in self.input_max_abs.iter_mut() {
            *v = 1.0;
        }
        for v in self.input_scale.iter_mut() {
            *v = Q14_HALF as f64;
        }
        self.n_samples = 0;
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for SpikeNet {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        // Membrane potential variance as uncertainty proxy: high variance in
        // membrane states indicates the network is being driven hard / unstable.
        let uncertainty = match &self.inner {
            Some(net) => {
                let membrane = net.hidden_membrane();
                if membrane.is_empty() {
                    0.0
                } else {
                    // Mean absolute membrane potential as fraction of Q14_ONE.
                    let sum: f64 = membrane.iter().map(|&v| (v as f64).abs()).sum();
                    let mean_abs = sum / membrane.len() as f64;
                    mean_abs / irithyll_core::snn::lif::Q14_ONE as f64
                }
            }
            None => 0.0, // Not yet initialized.
        };

        // Weight matrix size: n_hidden * n_hidden (recurrent) is the dominant DOF.
        Some(crate::automl::ConfigDiagnostics {
            effective_dof: (self.config.n_hidden * self.config.n_hidden) as f64,
            // Learning rate controls update magnitude.
            regularization_sensitivity: self.config.eta,
            uncertainty,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SpikeNetConfig {
        SpikeNetConfig::builder()
            .n_hidden(16)
            .n_outputs(1)
            .learning_rate(0.01)
            .alpha(0.9)
            .v_thr(0.3)
            .gamma(0.5)
            .spike_threshold(0.01)
            .seed(42)
            .weight_init_range(0.2)
            .build()
            .unwrap()
    }

    #[test]
    fn lazy_initialization_on_first_train() {
        let config = test_config();
        let mut model = SpikeNet::new(config);

        assert!(!model.is_initialized());
        model.train(&[0.5, -0.3], 1.0);
        assert!(model.is_initialized());
        assert_eq!(model.n_input(), 2);
    }

    #[test]
    fn predict_before_training_returns_zero() {
        let config = test_config();
        let model = SpikeNet::new(config);
        let pred = model.predict(&[1.0, 2.0]);
        assert_eq!(pred, 0.0, "uninitialized model should predict 0.0");
    }

    #[test]
    fn predictions_change_after_training() {
        let config = SpikeNetConfig::builder()
            .n_hidden(32)
            .learning_rate(0.05)
            .alpha(0.85)
            .v_thr(0.2)
            .gamma(0.5)
            .spike_threshold(0.005)
            .seed(12345)
            .weight_init_range(0.3)
            .build()
            .unwrap();

        let mut model = SpikeNet::new(config);

        // Warm up
        model.train(&[0.0, 0.0], 0.0);
        let pred_initial = model.predict(&[0.0, 0.0]);

        // Train with a consistent pattern
        for step in 0..300 {
            let x = if step % 2 == 0 {
                [1.0, -0.5]
            } else {
                [-0.5, 1.0]
            };
            let y = if step % 2 == 0 { 1.0 } else { -1.0 };
            model.train(&x, y);
        }

        let pred_after = model.predict(&[1.0, -0.5]);
        assert!(
            (pred_after - pred_initial).abs() > 1e-10,
            "prediction should change after 300 training steps: initial={}, after={}",
            pred_initial,
            pred_after
        );
    }

    #[test]
    fn n_samples_tracks_correctly() {
        let config = test_config();
        let mut model = SpikeNet::new(config);

        assert_eq!(model.n_samples_seen(), 0);
        model.train(&[1.0], 0.5);
        assert_eq!(model.n_samples_seen(), 1);
        model.train(&[2.0], 1.0);
        assert_eq!(model.n_samples_seen(), 2);
    }

    #[test]
    fn reset_clears_state() {
        let config = test_config();
        let mut model = SpikeNet::new(config);

        model.train(&[1.0, 2.0], 3.0);
        model.train(&[4.0, 5.0], 6.0);
        assert_eq!(model.n_samples_seen(), 2);

        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
    }

    #[test]
    fn with_n_input_initializes_immediately() {
        let config = test_config();
        let model = SpikeNet::with_n_input(config, 5);

        assert!(model.is_initialized());
        assert_eq!(model.n_input(), 5);
    }

    #[test]
    fn memory_bytes_positive_after_init() {
        let config = test_config();
        let model = SpikeNet::with_n_input(config, 4);
        assert!(
            model.memory_bytes() > 0,
            "memory_bytes should be > 0 after initialization"
        );
    }

    #[test]
    fn input_scaling_adapts() {
        let config = test_config();
        let mut model = SpikeNet::new(config);

        // Train with small values first
        for _ in 0..10 {
            model.train(&[0.01, 0.02], 0.0);
        }

        // Then train with larger values -- scaling should adapt
        for _ in 0..10 {
            model.train(&[10.0, 20.0], 0.0);
        }

        // Should not panic or produce NaN
        let pred = model.predict(&[5.0, 10.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );
    }

    #[test]
    fn weighted_training_does_not_crash() {
        let config = test_config();
        let mut model = SpikeNet::new(config);

        // Various weights
        model.train_one(&[1.0, 2.0], 3.0, 0.5);
        model.train_one(&[1.0, 2.0], 3.0, 2.0);
        model.train_one(&[1.0, 2.0], 3.0, 0.0);

        assert_eq!(model.n_samples_seen(), 3);
    }

    #[test]
    fn predict_is_deterministic_without_train() {
        let config = test_config();
        let mut model = SpikeNet::new(config);

        model.train(&[1.0, 2.0], 3.0);
        model.train(&[4.0, 5.0], 6.0);

        // Multiple predict calls should return the same value
        let p1 = model.predict(&[1.0, 2.0]);
        let p2 = model.predict(&[1.0, 2.0]);
        assert_eq!(p1, p2, "predict should be deterministic: {} vs {}", p1, p2);
    }
}
