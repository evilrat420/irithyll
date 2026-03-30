//! Echo State Network (ESN) streaming learner.
//!
//! [`EchoStateNetwork`] implements a classic ESN with a cycle/ring reservoir
//! topology and an online RLS readout layer. The reservoir provides a rich
//! nonlinear temporal feature space, while RLS trains the readout weights
//! incrementally.
//!
//! # Algorithm
//!
//! For each input vector `u(t)`:
//!
//! 1. **Reservoir update**: Drive the cycle reservoir with `u(t)`, producing
//!    state `x(t)` via leaky integration and tanh activation.
//!
//! 2. **Feature construction**: Build readout features as `[x(t); u(t)]` if
//!    `passthrough_input` is enabled, otherwise just `x(t)`.
//!
//! 3. **Online readout**: Train an RLS model on these features to predict
//!    the target.
//!
//! The first `warmup` samples drive the reservoir without training the readout,
//! allowing the reservoir state to develop meaningful dynamics before learning.
//!
//! # References
//!
//! Jaeger, H. (2001). "The echo state approach to analysing and training
//! recurrent neural networks." GMD Report 148.

use std::fmt;

use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;
use irithyll_core::reservoir::CycleReservoir;

use super::esn_config::ESNConfig;

/// Echo State Network with cycle reservoir topology and online RLS readout.
///
/// The reservoir is initialized once at construction time with deterministic
/// random weights (given the seed). During streaming, each `train_one` call
/// advances the reservoir by one step and (after warmup) trains the RLS readout.
///
/// `predict` is side-effect-free: it uses the current reservoir state to build
/// features and returns the RLS prediction without advancing the reservoir.
///
/// # Examples
///
/// ```
/// use irithyll::reservoir::{EchoStateNetwork, ESNConfig};
/// use irithyll::learner::StreamingLearner;
///
/// let config = ESNConfig::builder()
///     .n_reservoir(50)
///     .spectral_radius(0.9)
///     .leak_rate(0.3)
///     .warmup(20)
///     .build()
///     .unwrap();
///
/// let mut esn = EchoStateNetwork::new(config);
///
/// // Feed a sine wave: train on x(t) to predict x(t+1).
/// for i in 0..200 {
///     let t = i as f64 * 0.1;
///     let x = t.sin();
///     let target = (t + 0.1).sin();
///     esn.train(&[x], target);
/// }
///
/// let pred = esn.predict(&[0.5_f64.sin()]);
/// assert!(pred.is_finite());
/// ```
pub struct EchoStateNetwork {
    /// Configuration.
    config: ESNConfig,
    /// Cycle reservoir.
    reservoir: CycleReservoir,
    /// RLS readout layer.
    rls: RecursiveLeastSquares,
    /// Total observations seen (including warmup).
    total_seen: u64,
    /// Observations trained on (post-warmup).
    samples_trained: u64,
    /// Input dimension (set lazily on first call).
    n_inputs: Option<usize>,
    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// Change from two steps ago, for acceleration-based alignment.
    prev_prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
}

impl EchoStateNetwork {
    /// Create a new ESN from the given configuration.
    ///
    /// The reservoir is initialized lazily on the first call to `train_one`,
    /// when the input dimension becomes known.
    pub fn new(config: ESNConfig) -> Self {
        let rls = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta);

        Self {
            reservoir: CycleReservoir::new(
                config.n_reservoir,
                1, // placeholder — will be re-created on first train_one
                config.spectral_radius,
                config.input_scaling,
                config.leak_rate,
                config.bias_scaling,
                config.seed,
            ),
            rls,
            total_seen: 0,
            samples_trained: 0,
            n_inputs: None,
            config,
            prev_prediction: 0.0,
            prev_change: 0.0,
            prev_prev_change: 0.0,
            alignment_ewma: 0.0,
        }
    }

    /// Whether the warmup period has passed and the readout is being trained.
    #[inline]
    pub fn past_warmup(&self) -> bool {
        self.total_seen > self.config.warmup as u64
    }

    /// Build the readout feature vector from the current reservoir state.
    ///
    /// If `passthrough_input` is enabled, the feature vector is `[state; input]`.
    /// Otherwise, it is just `[state]`.
    fn build_readout_features(&self, input: &[f64]) -> Vec<f64> {
        let state = self.reservoir.state();
        if self.config.passthrough_input {
            let mut features = Vec::with_capacity(state.len() + input.len());
            features.extend_from_slice(state);
            features.extend_from_slice(input);
            features
        } else {
            state.to_vec()
        }
    }

    /// Access the underlying configuration.
    pub fn config(&self) -> &ESNConfig {
        &self.config
    }

    /// Total observations seen (including warmup).
    pub fn total_seen(&self) -> u64 {
        self.total_seen
    }

    /// Forward-looking prediction uncertainty from the RLS readout.
    ///
    /// Returns the estimated prediction standard deviation, computed as the
    /// square root of the RLS noise variance (EWMA of squared residuals).
    /// This is a model-level uncertainty signal that does not require
    /// transformed features.
    ///
    /// Returns 0.0 before any training has occurred.
    #[inline]
    pub fn prediction_uncertainty(&self) -> f64 {
        self.rls.noise_variance().sqrt()
    }

    /// Access the current reservoir state.
    pub fn reservoir_state(&self) -> &[f64] {
        self.reservoir.state()
    }

    /// Initialize or re-initialize the reservoir with the correct input dimension.
    fn ensure_reservoir(&mut self, n_inputs: usize) {
        if self.n_inputs.is_none() || self.n_inputs != Some(n_inputs) {
            self.n_inputs = Some(n_inputs);
            self.reservoir = CycleReservoir::new(
                self.config.n_reservoir,
                n_inputs,
                self.config.spectral_radius,
                self.config.input_scaling,
                self.config.leak_rate,
                self.config.bias_scaling,
                self.config.seed,
            );
        }
    }
}

impl StreamingLearner for EchoStateNetwork {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // Initialize reservoir on first call.
        self.ensure_reservoir(features.len());

        // Drive the reservoir forward one step.
        self.reservoir.update(features);
        self.total_seen += 1;

        // After warmup, train the RLS readout.
        if self.past_warmup() {
            let readout_features = self.build_readout_features(features);
            let current_pred = self.rls.predict(&readout_features);

            // Update residual alignment tracking (acceleration-based).
            let current_change = current_pred - self.prev_prediction;
            if self.samples_trained > 0 {
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
                self.alignment_ewma =
                    (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
            }
            self.prev_prev_change = self.prev_change;
            self.prev_change = current_change;
            self.prev_prediction = current_pred;

            self.rls.train_one(&readout_features, target, weight);
            self.samples_trained += 1;
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        // Side-effect-free: does NOT update reservoir state.
        // Uses current reservoir state + the given features to build readout features.
        if !self.past_warmup() || self.n_inputs.is_none() {
            return 0.0;
        }

        let readout_features = self.build_readout_features(features);
        self.rls.predict(&readout_features)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_trained
    }

    fn reset(&mut self) {
        self.reservoir.reset();
        self.rls.reset();
        self.total_seen = 0;
        self.samples_trained = 0;
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.prev_prev_change = 0.0;
        self.alignment_ewma = 0.0;
        // Keep n_inputs and reservoir weights — reset only resets learned state,
        // not the architecture. If the user wants a fresh reservoir, they should
        // construct a new ESN.
    }

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
}

impl fmt::Debug for EchoStateNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EchoStateNetwork")
            .field("n_reservoir", &self.config.n_reservoir)
            .field("spectral_radius", &self.config.spectral_radius)
            .field("leak_rate", &self.config.leak_rate)
            .field("warmup", &self.config.warmup)
            .field("total_seen", &self.total_seen)
            .field("samples_trained", &self.samples_trained)
            .field("past_warmup", &self.past_warmup())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for EchoStateNetwork {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.alignment_ewma,
            regularization_sensitivity: 1.0 - self.config.forgetting_factor,
            depth_sufficiency: 0.0, // Single-layer readout.
            effective_dof: self.rls.weights().len() as f64,
            uncertainty: self.prediction_uncertainty(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_esn() -> EchoStateNetwork {
        let config = ESNConfig::builder()
            .n_reservoir(50)
            .warmup(10)
            .build()
            .unwrap();
        EchoStateNetwork::new(config)
    }

    #[test]
    fn cold_start_returns_zero() {
        let esn = default_esn();
        assert_eq!(esn.predict(&[1.0]), 0.0);
        assert_eq!(esn.n_samples_seen(), 0);
        assert!(!esn.past_warmup());
    }

    #[test]
    fn warmup_period_no_training() {
        let mut esn = default_esn();
        // warmup=10: the first 10 calls only drive the reservoir, no RLS training.
        for i in 0..10 {
            esn.train(&[i as f64 * 0.1], 0.0);
        }
        assert!(
            !esn.past_warmup(),
            "10th sample is last warmup sample, not yet past warmup"
        );
        assert_eq!(esn.n_samples_seen(), 0);

        // The 11th call transitions past warmup and trains.
        esn.train(&[1.0], 0.0);
        assert!(esn.past_warmup(), "11th sample should be past warmup");
        assert_eq!(esn.n_samples_seen(), 1);
    }

    #[test]
    fn trains_after_warmup() {
        let mut esn = default_esn();
        for i in 0..15 {
            esn.train(&[i as f64 * 0.1], 0.0);
        }
        // 10 warmup + 5 trained = 15 total, 5 trained.
        assert_eq!(esn.n_samples_seen(), 5);
        assert_eq!(esn.total_seen(), 15);
    }

    #[test]
    fn predict_is_side_effect_free() {
        let mut esn = default_esn();
        for i in 0..20 {
            esn.train(&[i as f64 * 0.1], i as f64);
        }

        let total_before = esn.total_seen();
        let samples_before = esn.n_samples_seen();

        let _ = esn.predict(&[99.0]);

        assert_eq!(
            esn.total_seen(),
            total_before,
            "predict should not increment total_seen",
        );
        assert_eq!(
            esn.n_samples_seen(),
            samples_before,
            "predict should not increment samples_trained",
        );
    }

    #[test]
    fn reset_clears_learned_state() {
        let mut esn = default_esn();
        for i in 0..30 {
            esn.train(&[i as f64 * 0.1], i as f64);
        }
        assert!(esn.n_samples_seen() > 0);

        esn.reset();
        assert_eq!(esn.n_samples_seen(), 0);
        assert_eq!(esn.total_seen(), 0);
        assert!(!esn.past_warmup());

        // Reservoir state should be zero after reset.
        for &s in esn.reservoir_state() {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let config1 = ESNConfig::builder()
            .n_reservoir(30)
            .warmup(5)
            .seed(42)
            .build()
            .unwrap();
        let config2 = config1.clone();

        let mut esn1 = EchoStateNetwork::new(config1);
        let mut esn2 = EchoStateNetwork::new(config2);

        for i in 0..20 {
            let x = (i as f64 * 0.3).sin();
            let y = (i as f64 * 0.3 + 0.3).sin();
            esn1.train(&[x], y);
            esn2.train(&[x], y);
        }

        let pred1 = esn1.predict(&[0.5]);
        let pred2 = esn2.predict(&[0.5]);
        assert!(
            (pred1 - pred2).abs() < 1e-12,
            "same seed should produce identical predictions: {} vs {}",
            pred1,
            pred2,
        );
    }

    #[test]
    fn sine_wave_regression() {
        let config = ESNConfig::builder()
            .n_reservoir(100)
            .spectral_radius(0.9)
            .leak_rate(0.3)
            .input_scaling(1.0)
            .warmup(50)
            .forgetting_factor(0.999)
            .seed(42)
            .build()
            .unwrap();
        let mut esn = EchoStateNetwork::new(config);

        let dt = 0.1;
        let n_train = 500;

        // Train: predict sin(t + dt) from sin(t).
        for i in 0..n_train {
            let t = i as f64 * dt;
            let x = t.sin();
            let target = (t + dt).sin();
            esn.train(&[x], target);
        }

        // Evaluate on continuation.
        let mut total_error = 0.0;
        let n_test = 50;
        for i in n_train..(n_train + n_test) {
            let t = i as f64 * dt;
            let x = t.sin();
            let target = (t + dt).sin();
            esn.train(&[x], target);
            let pred = esn.predict(&[x]);
            total_error += (pred - target).abs();
        }
        let mae = total_error / n_test as f64;
        assert!(mae < 0.5, "ESN sine wave MAE should be < 0.5, got {}", mae,);
    }

    #[test]
    fn trait_object_compatibility() {
        let config = ESNConfig::builder()
            .n_reservoir(20)
            .warmup(5)
            .build()
            .unwrap();
        let esn = EchoStateNetwork::new(config);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(esn);

        for i in 0..20 {
            boxed.train(&[i as f64 * 0.1], i as f64);
        }
        let pred = boxed.predict(&[1.0]);
        assert!(pred.is_finite());
    }

    #[test]
    fn no_passthrough_input() {
        let config = ESNConfig::builder()
            .n_reservoir(30)
            .warmup(5)
            .passthrough_input(false)
            .build()
            .unwrap();
        let mut esn = EchoStateNetwork::new(config);

        for i in 0..30 {
            esn.train(&[i as f64 * 0.1], i as f64);
        }
        let pred = esn.predict(&[1.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite without passthrough"
        );
    }

    #[test]
    fn reservoir_state_evolves() {
        let mut esn = default_esn();
        esn.train(&[1.0], 0.0);

        // After one input, the reservoir state should have changed from zero.
        let nonzero = esn
            .reservoir_state()
            .iter()
            .filter(|&&s| s.abs() > 1e-15)
            .count();
        assert!(nonzero > 0, "reservoir state should be nonzero after input",);
    }

    #[test]
    fn esn_prediction_uncertainty() {
        let config = ESNConfig::builder()
            .n_reservoir(50)
            .warmup(10)
            .build()
            .unwrap();
        let mut esn = EchoStateNetwork::new(config);

        // Before training, uncertainty is 0.0
        assert!(
            esn.prediction_uncertainty().abs() < 1e-15,
            "uncertainty should be 0.0 before training, got {}",
            esn.prediction_uncertainty()
        );

        // Train on 100 samples
        for i in 0..100 {
            let t = i as f64 * 0.1;
            let x = t.sin();
            let target = (t + 0.1).sin();
            esn.train(&[x], target);
        }

        let unc = esn.prediction_uncertainty();
        assert!(
            unc > 0.0,
            "prediction_uncertainty should be > 0 after training, got {}",
            unc
        );
        assert!(
            unc.is_finite(),
            "prediction_uncertainty should be finite, got {}",
            unc
        );
    }
}
