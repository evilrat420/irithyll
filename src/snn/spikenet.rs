//! f64-interfaced SNN wrapping `SpikeNetFixed` with automatic scaling.
//!
//! [`SpikeNet`] bridges the gap between irithyll's f64 [`StreamingLearner`]
//! interface and the Q1.14 fixed-point core. It handles:
//!
//! - Lazy initialization (input dimension discovered from first sample)
//! - Adaptive input scaling (EWMA max tracker to map features into Q1.14 range)
//! - Target quantization and e-prop learning
//! - An online RLS readout layer trained on cached hidden spike state + input
//!   features, enabling side-effect-free `predict()` that actually uses the
//!   current input features (unlike the raw SNN readout membrane).
//! - Sample weight support via learning rate modulation
//!
//! # Scaling Strategy
//!
//! During the first 50 samples, a running maximum absolute value is tracked
//! per feature. After warmup, `input_scale[i] = Q14_HALF / max_abs[i]`,
//! ensuring features map to roughly `[-0.5, +0.5]` in Q1.14. An EWMA
//! with decay 0.99 allows the scale to adapt to non-stationary distributions.
//!
//! # Prediction
//!
//! The SNN's internal e-prop readout membrane is stateful and updates only
//! during `train_one`. Calling `predict(features)` on the raw membrane would
//! ignore the new features entirely, producing stale predictions that are
//! disconnected from the input. Instead, SpikeNet trains a small RLS readout
//! on `[hidden_spike_bits; input_features]` after each training step and uses
//! it for inference. This follows the ESN/Mamba pattern: the recurrent core
//! (SNN) provides temporal state; the linear readout (RLS) maps that state
//! plus the current input to the target prediction in a side-effect-free way.

use super::spikenet_config::LearningRule;
use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;
use irithyll_core::snn::astrocyte::AstrocyteMode;
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
    target_scale: f64,

    // Reusable quantization buffer (used during train_one only)
    quantized_input: Vec<i16>,
    quantized_target: Vec<i16>,

    // RLS readout: trained on [hidden_spikes_f64; input_features] -> target.
    // This enables side-effect-free predict() that uses the current input,
    // matching the ESN/Mamba pattern and avoiding the stale-membrane problem.
    readout_rls: RecursiveLeastSquares,
    /// Cached hidden spike vector from the last training step (f64 0.0/1.0).
    last_spike_state: Vec<f64>,

    n_samples: u64,
    n_input: usize,

    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// Change from two steps ago, for acceleration-based alignment.
    prev_prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
    /// EWMA of spike rate (spikes_this_step / total_neurons) for utilization.
    spike_rate_ewma: f64,
}

// SpikeNet is Send + Sync by composition — all fields are Send+Sync types.
// (No unsafe impl needed; Rust auto-derives these.)

impl SpikeNet {
    /// Create a new SpikeNet with the given configuration.
    ///
    /// The underlying network is not allocated until the first sample arrives
    /// (lazy initialization), because the input dimension is unknown.
    pub fn new(config: SpikeNetConfig) -> Self {
        // Forgetting factor 0.995: mild non-stationarity adaptation.
        let readout_rls = RecursiveLeastSquares::new(0.995);
        Self {
            config,
            inner: None,
            input_scale: Vec::new(),
            input_max_abs: Vec::new(),
            target_scale: Q14_ONE as f64, // map target 1.0 -> Q14_ONE (full range)
            quantized_input: Vec::new(),
            quantized_target: Vec::new(),
            readout_rls,
            last_spike_state: Vec::new(),
            n_samples: 0,
            n_input: 0,
            prev_prediction: 0.0,
            prev_change: 0.0,
            prev_prev_change: 0.0,
            alignment_ewma: 0.0,
            spike_rate_ewma: 0.0,
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

        // PpProp is a forward-compatible placeholder: it falls back to e-prop
        // (Stdp) until a PP-prop kernel lands in SpikeNetFixed.
        // Ref: Kaiser et al., NeurIPS 2022.
        if self.config.learning_rule == LearningRule::PpProp {
            tracing::warn!(
                "SpikeNet: LearningRule::PpProp is not yet implemented in the \
                 fixed-point kernel; falling back to e-prop (Stdp). \
                 Ref: Kaiser et al., NeurIPS 2022."
            );
        }

        let fixed_config = SpikeNetFixedConfig {
            n_input,
            n_hidden: self.config.n_hidden,
            n_output: self.config.n_outputs,
            alpha: f64_to_q14(self.config.alpha),
            kappa: f64_to_q14(self.config.kappa),
            kappa_out: f64_to_q14(self.config.kappa_out),
            eta: f64_to_q14(self.config.learning_rate),
            v_thr: f64_to_q14(self.config.v_thr),
            gamma: f64_to_q14(self.config.gamma),
            spike_threshold: f64_to_q14(self.config.spike_threshold),
            seed: self.config.seed,
            weight_init_range: f64_to_q14(self.config.weight_init_range),
            use_astrocyte: self.config.astrocyte,
            astrocyte_tau: self.config.astrocyte_tau,
            // SpikeNetConfig doesn't yet expose astrocyte_mode; default to the
            // conservative WeightMod path (prior behaviour before AGMP addition).
            astrocyte_mode: AstrocyteMode::WeightMod,
        };

        self.inner = Some(SpikeNetFixed::new(fixed_config));

        // Initialize scaling: start with unit scale, adapt during warmup
        self.input_scale = vec![Q14_HALF as f64; n_input];
        self.input_max_abs = vec![1.0; n_input]; // avoid division by zero

        self.quantized_input = vec![0i16; n_input];
        self.quantized_target = vec![0i16; self.config.n_outputs];

        // Initialize the cached spike state to all-zeros (network not yet run).
        self.last_spike_state = vec![0.0f64; self.config.n_hidden];
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

        if features.len() != self.n_input {
            return;
        }

        // Update input scaling
        self.update_input_scaling(features);

        // Quantize inputs and target
        self.quantize_input(features);
        self.quantize_target(target);

        // Apply sample weight: for SNNs with fixed-point internals, weight
        // modulation is approximated. Weight ~0 skips the update, weight >0
        // trains once with the base learning rate. The fixed-point config
        // does not support per-sample eta changes without mutable config access.
        if let Some(ref mut net) = self.inner {
            if weight > 1e-10 {
                net.train_step(&self.quantized_input, &self.quantized_target);
            } else {
                // Even if we skip weight update, run forward to keep state current.
                net.forward(&self.quantized_input);
            }

            // Cache hidden spike state as f64 for the RLS readout.
            let spikes = net.hidden_spikes();
            let n_total = spikes.len();
            for (dst, &s) in self.last_spike_state.iter_mut().zip(spikes.iter()) {
                *dst = s as f64;
            }

            // Track spike rate EWMA.
            if n_total > 0 {
                let n_spiking = spikes.iter().filter(|&&s| s > 0).count();
                let rate = n_spiking as f64 / n_total as f64;
                const SPIKE_ALPHA: f64 = 0.01;
                if self.n_samples == 0 {
                    self.spike_rate_ewma = rate;
                } else {
                    self.spike_rate_ewma =
                        (1.0 - SPIKE_ALPHA) * self.spike_rate_ewma + SPIKE_ALPHA * rate;
                }
            }
        }

        // Train the RLS readout on [hidden_spikes; input_features] -> target.
        // This is the prediction surface used by predict(), ensuring that
        // predict(features) actually uses the current input rather than
        // just echoing the stale SNN membrane.
        //
        // Option D — eligibility trace natural alignment:
        //
        // SpikeNet naturally follows Option D for the RLS readout. The e-prop
        // learning rule inside `net.train_step()` works as follows:
        //
        //   1. Run the current input through the SNN (forward pass), producing spikes.
        //   2. Compute eligibility traces from the pre-update membrane potential and
        //      presynaptic activity — these represent the pre-spike-advance state.
        //   3. Update SNN weights using eligibility × learning signal (weight update).
        //
        // The spikes cached in `last_spike_state` are from step 1 (the forward pass),
        // which runs with the SNN weights BEFORE the e-prop update in step 3. This
        // means `last_spike_state` encodes the pre-weight-update network response.
        //
        // At predict time, `last_spike_state` from the previous train_one call is
        // used — which was computed with pre-update weights at that step. The RLS
        // is therefore trained on the same class of features (pre-update spike
        // responses) that predict() will observe, eliminating the train/predict
        // feature-distribution mismatch.
        //
        // SpikeNet is the canonical reference implementation of the Option D pattern
        // for recurrent learners with an RLS readout. Other recurrent models (ESN,
        // TTT) should follow the same ordering: compute readout features from the
        // pre-advance state, train RLS, then advance the recurrent state.
        //
        // Reference: R8 — GLA predict() quality: a mathematically principled fix.
        if weight > 1e-10 {
            let mut readout_features = Vec::with_capacity(self.config.n_hidden + self.n_input);
            readout_features.extend_from_slice(&self.last_spike_state);
            readout_features.extend_from_slice(features);
            self.readout_rls
                .train_one(&readout_features, target, weight);
        }

        // Update residual alignment tracking (acceleration-based) using
        // the RLS prediction (which uses the current features, not stale membrane).
        let current_pred = self.predict(features);
        let current_change = current_pred - self.prev_prediction;
        if self.n_samples > 0 {
            let acceleration = current_change - self.prev_change;
            let prev_acceleration = self.prev_change - self.prev_prev_change;
            let agreement = if acceleration.abs() > 1e-15 && prev_acceleration.abs() > 1e-15 {
                if (acceleration > 0.0) == (prev_acceleration > 0.0) {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            };
            const ALIGN_ALPHA: f64 = 0.05;
            if self.n_samples == 1 {
                self.alignment_ewma = agreement;
            } else {
                self.alignment_ewma =
                    (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
            }
        }
        self.prev_prev_change = self.prev_change;
        self.prev_change = current_change;
        self.prev_prediction = current_pred;

        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        if self.inner.is_none() {
            return 0.0;
        }

        if features.len() != self.n_input {
            return 0.0;
        }

        // Use the RLS readout: [cached_hidden_spikes; input_features] -> target.
        //
        // Design rationale (matches ESN/Mamba pattern):
        // - The SNN's e-prop readout membrane reflects state from the last
        //   training step and ignores the current `features` argument entirely.
        // - The RLS readout is trained on [spikes_from_step_t-1; features_t]
        //   → target_t during train_one, then used here for side-effect-free
        //   inference that actually depends on the current input.
        // - Before any training, the RLS returns 0.0 (uninitialised weights).
        let mut readout_features = Vec::with_capacity(self.config.n_hidden + self.n_input);
        readout_features.extend_from_slice(&self.last_spike_state);
        readout_features.extend_from_slice(features);
        self.readout_rls.predict(&readout_features)
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
        // Reset RLS readout and cached spike state.
        self.readout_rls.reset();
        for v in self.last_spike_state.iter_mut() {
            *v = 0.0;
        }
        self.n_samples = 0;
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.prev_prev_change = 0.0;
        self.alignment_ewma = 0.0;
        self.spike_rate_ewma = 0.0;
    }

    #[allow(deprecated)]
    fn diagnostics_array(&self) -> [f64; 5] {
        <Self as crate::learner::Tunable>::diagnostics_array(self)
    }
}

impl crate::learner::Tunable for SpikeNet {
    fn diagnostics_array(&self) -> [f64; 5] {
        use crate::automl::DiagnosticSource;
        match self.config_diagnostics() {
            Some(d) => [
                d.residual_alignment,
                d.regularization_sensitivity,
                d.depth_sufficiency,
                d.effective_dof,
                d.uncertainty,
            ],
            None => [0.0; 5],
        }
    }

    fn adjust_config(&mut self, _lr_multiplier: f64, _lambda_delta: f64) {
        // SpikeNet does not expose a tunable LR/lambda; no-op.
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

        // Spike rate EWMA as depth_sufficiency: healthy SNN activity indicates
        // the network is being utilized. Range [0, 1], healthy is 0.1-0.3.
        let depth_sufficiency = self.spike_rate_ewma.clamp(0.0, 1.0);

        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.alignment_ewma,
            regularization_sensitivity: self.config.learning_rate,
            depth_sufficiency,
            effective_dof: self.config.n_hidden as f64,
            uncertainty,
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

    #[test]
    fn test_spikenet_dimension_mismatch_no_panic() {
        // After initialization with 2 features, train with wrong dimension — should silently return.
        let config = test_config();
        let mut model = SpikeNet::new(config);
        model.train(&[0.5, -0.3], 1.0); // initializes with n_input=2
        assert_eq!(model.n_input(), 2);
        // This used to assert_eq! (panic). Now should be a graceful no-op.
        model.train(&[1.0, 2.0, 3.0], 0.5); // 3 features, expected 2
                                            // Samples seen only incremented on successful train
        assert_eq!(
            model.n_samples_seen(),
            1,
            "mismatched-dimension sample should not be counted"
        );
    }

    /// SpikeNet must never perform worse than random on a simple linearly separable
    /// binary classification task. Pre-fix, the stale SNN readout membrane caused
    /// anti-correlated predictions (33% on Agrawal) because predict() ignored the
    /// input features entirely. With the RLS readout, predict(features) uses the
    /// current input and must exceed 50% chance baseline.
    #[test]
    fn test_spikenet_binary_classification_above_chance() {
        // Simple 2-class problem: class 1 = feature > 0, class 0 = feature < 0.
        // This is trivially linearly separable, so any correct learner should
        // reach well above 50% after a few hundred samples.
        let config = SpikeNetConfig::builder()
            .n_hidden(32)
            .learning_rate(0.02)
            .alpha(0.9)
            .v_thr(0.3)
            .gamma(0.5)
            .spike_threshold(0.01)
            .seed(99)
            .weight_init_range(0.2)
            .build()
            .unwrap();

        let mut model = SpikeNet::new(config);

        // Deterministic alternating samples: class 1 at x=+1, class 0 at x=-1.
        // Use prequential (test-then-train) protocol matching the benchmark.
        let n_samples = 500;
        let mut correct = 0usize;
        let mut total = 0usize;

        // Warmup phase: train only, don't count.
        for i in 0..50usize {
            let x = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            let y = if i % 2 == 0 { 1.0_f64 } else { 0.0_f64 };
            model.train(&[x, x * 0.5], y);
        }

        // Evaluation phase: prequential (predict then train).
        for i in 0..n_samples {
            let x = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            let y = if i % 2 == 0 { 1.0_f64 } else { 0.0_f64 };

            let pred = model.predict(&[x, x * 0.5]);
            let pred_class = if pred >= 0.5 { 1.0_f64 } else { 0.0_f64 };
            if (pred_class - y).abs() < 0.1 {
                correct += 1;
            }
            total += 1;

            model.train(&[x, x * 0.5], y);
        }

        let accuracy = correct as f64 / total as f64;
        assert!(
            accuracy > 0.5,
            "SpikeNet must exceed 50% chance baseline on a simple binary task, \
             got accuracy = {:.3} ({}/{} correct). \
             This indicates predict() is ignoring input features (stale membrane bug).",
            accuracy,
            correct,
            total
        );
    }

    #[test]
    fn spikenet_predict_reads_current_input() {
        // Option D empirical confirmation: predict(x_a) != predict(x_b) for distinct
        // inputs, confirming the RLS readout uses the current-input component of its
        // feature vector [last_spike_state; input_features].
        //
        // SpikeNet is the canonical Option D implementation: last_spike_state encodes
        // pre-weight-update spikes (eligibility trace alignment), and input_features
        // provides direct current-input dependence in the readout. This test verifies
        // both that predictions are finite and that they discriminate distinct inputs.
        let config = SpikeNetConfig::builder()
            .n_hidden(32)
            .learning_rate(0.02)
            .alpha(0.9)
            .v_thr(0.3)
            .gamma(0.5)
            .spike_threshold(0.01)
            .seed(42)
            .weight_init_range(0.2)
            .build()
            .unwrap();

        let mut model = SpikeNet::new(config);

        // Train enough to build meaningful RLS readout weights.
        for i in 0..100 {
            let x = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            let y = if i % 2 == 0 { 1.0_f64 } else { 0.0_f64 };
            model.train(&[x, x * 0.5], y);
        }

        // Two distinct inputs should produce distinct predictions because
        // the input_features component of [last_spike_state; input_features]
        // differs between them.
        let pred_a = model.predict(&[1.0, 0.5]);
        let pred_b = model.predict(&[-1.0, -0.5]);

        assert!(
            pred_a.is_finite(),
            "predict(+1.0, +0.5) should be finite, got {pred_a}"
        );
        assert!(
            pred_b.is_finite(),
            "predict(-1.0, -0.5) should be finite, got {pred_b}"
        );
        assert_ne!(
            pred_a.to_bits(),
            pred_b.to_bits(),
            "SpikeNet predict must reflect current input: predict(+1,+0.5)={pred_a} == predict(-1,-0.5)={pred_b}"
        );
    }
}
