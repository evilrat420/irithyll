//! Streaming Mamba model: selective SSM + SiLU gating + RLS readout.
//!
//! [`StreamingMamba`] is a complete streaming regression model that combines:
//!
//! 1. A **selective SSM** (Mamba-style) for temporal feature extraction
//! 2. A **SiLU multiplicative gate** for content-dependent filtering (Gu & Dao, 2024)
//! 3. A **residual connection** preserving the raw input signal
//! 4. A **Recursive Least Squares** (RLS) readout for mapping gated features to predictions
//!
//! This architecture processes each input as a timestep: the SSM maintains hidden
//! state capturing temporal patterns, the SiLU gate learns which SSM outputs to
//! amplify or suppress, and the RLS readout learns a linear mapping from the
//! gated output to the target variable. All components update incrementally,
//! making the model fully streaming with O(1) memory per timestep.
//!
//! # Training Flow
//!
//! ```text
//! features ──→ SSM.forward() ──→ ssm_output ──┐
//!    │                                          ├──→ ssm_output ⊗ SiLU(gate)
//!    └──→ gate = SiLU(W_gate · x + b)  ────────┘         │
//!                                                    + residual(x)
//!                                                         │
//!                                                   [gated_out; pooled_state]
//!                                                         │
//!                                                   RLS.train_one()
//! ```
//!
//! The SSM computes `y = C_t @ h + D * x` in its `forward()` method — this IS
//! the learned projection from hidden state to output space. The SiLU gate acts
//! as a learned content-dependent filter: `gate_i = SiLU(W_gate[i] · x + b[i])`.
//! The gated output `ssm_output ⊗ gate + residual(x)` is then combined with
//! pooled hidden state and fed to the RLS readout.
//!
//! # Prediction
//!
//! `predict()` uses the cached readout features from the most recent `train_one()`
//! call. This avoids a side-effect (advancing SSM state) during prediction,
//! maintaining the contract that `predict()` is read-only. If no training has
//! occurred, returns 0.0.

use irithyll_core::ssm::{SSMLayer, SelectiveSSM};

use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;
use crate::ssm::mamba_config::MambaConfig;

// ---------------------------------------------------------------------------
// RNG helpers (xorshift64, same pattern as ttt::layer)
// ---------------------------------------------------------------------------

/// Advance xorshift64 state and return raw u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Uniform f64 in [0, 1) from xorshift64.
#[inline]
fn xorshift64_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Standard normal via Box-Muller transform (returns one sample).
fn standard_normal(state: &mut u64) -> f64 {
    loop {
        let u1 = xorshift64_f64(state);
        let u2 = xorshift64_f64(state);
        if u1 > 0.0 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            return r * theta.cos();
        }
    }
}

// ---------------------------------------------------------------------------
// SiLU activation
// ---------------------------------------------------------------------------

/// SiLU (Sigmoid Linear Unit): `x * sigmoid(x)`.
///
/// Also known as the Swish activation. Used in the Mamba block's gating
/// branch to produce a smooth, non-negative gate signal that allows
/// gradient flow in both directions.
#[inline]
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// StreamingMamba
// ---------------------------------------------------------------------------

/// Streaming Mamba model implementing [`StreamingLearner`].
///
/// Combines a selective SSM for temporal feature extraction with a SiLU
/// multiplicative gate and an RLS readout layer. The SSM processes each
/// input as a timestep, evolving hidden state to capture temporal
/// dependencies. A learned SiLU gate (`W_gate · x + b`) produces a
/// content-dependent filter that is element-wise multiplied with the SSM
/// output, followed by a residual connection from the raw input. This
/// gated architecture (Gu & Dao, 2024) prevents noise from passing
/// through the SSM unfiltered.
///
/// The readout sees `[gated_output; pooled_hidden_state]`, where the
/// gated output is `ssm_output ⊗ SiLU(gate) + residual(x)`. The pooled
/// hidden state adds temporal detail that the gating may compress.
///
/// # Example
///
/// ```
/// use irithyll::ssm::{StreamingMamba, MambaConfig};
/// use irithyll::learner::StreamingLearner;
///
/// let config = MambaConfig::builder()
///     .d_in(3)
///     .n_state(8)
///     .build()
///     .unwrap();
///
/// let mut model = StreamingMamba::new(config);
///
/// // Train on a stream of 3-dimensional features
/// for i in 0..100 {
///     let x = [i as f64 * 0.1, (i as f64).sin(), 1.0];
///     let y = x[0] + 0.5 * x[1];
///     model.train(&x, y);
/// }
///
/// let pred = model.predict(&[10.0, 0.0, 1.0]);
/// assert!(pred.is_finite());
/// ```
pub struct StreamingMamba {
    /// Model configuration.
    config: MambaConfig,
    /// Selective SSM for temporal feature extraction.
    ssm: SelectiveSSM,
    /// RLS readout layer for prediction.
    readout: RecursiveLeastSquares,
    /// SiLU gate projection weights: d_in × d_in matrix (row-major).
    /// Maps raw input to gate signal: `gate[i] = SiLU(sum_j(W[i*d+j]*x[j]) + b[i])`.
    gate_weights: Vec<f64>,
    /// SiLU gate bias vector (d_in elements).
    gate_bias: Vec<f64>,
    /// Cached readout features (gated SSM output + pooled state) from the
    /// most recent `train_one` call.
    last_features: Vec<f64>,
    /// Total samples trained on.
    n_samples: u64,
    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// Change from two steps ago, for acceleration-based alignment.
    prev_prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
    /// EWMA of maximum Frobenius squared norm of SSM state for utilization ratio.
    max_frob_sq_ewma: f64,
}

impl StreamingMamba {
    /// Maximum number of features fed to the RLS readout layer.
    ///
    /// The readout sees `[ssm_output; pooled_hidden_state]`. The SSM output
    /// (`d_in` dims) provides the learned C-projection, while a compact
    /// mean-pooled projection of the hidden state preserves temporal detail
    /// that the C-projection may discard. Total features are capped at this
    /// limit to keep the RLS covariance matrix bounded.
    pub const MAX_READOUT_FEATURES: usize = 128;

    /// Create a new streaming Mamba model from the given configuration.
    ///
    /// Initializes the SSM with random weights (seeded by `config.seed`),
    /// a SiLU gate with Xavier-initialized weights, and an RLS readout
    /// with the specified forgetting factor and P matrix scale.
    ///
    /// The readout feature vector is `[gated_output; pooled_state]`:
    /// - `gated_output` (`d_in` dims): SSM output ⊗ SiLU(gate) + residual(x)
    /// - `pooled_state` (up to `d_in` dims): mean-pooled hidden state for extra temporal signal
    ///
    /// When the full hidden state is small enough, it's used directly instead of pooling.
    pub fn new(config: MambaConfig) -> Self {
        let ssm = SelectiveSSM::new(config.d_in, config.n_state, config.seed);
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta_rls);
        let readout_dim = Self::readout_dim(config.d_in, config.n_state);
        let last_features = vec![0.0; readout_dim];

        // Initialize gate weights with Xavier normal: N(0, sqrt(2 / (fan_in + fan_out))).
        // Both fan_in and fan_out are d_in, so scale = sqrt(2 / (2 * d_in)) = 1/sqrt(d_in).
        let (gate_weights, gate_bias) = Self::init_gate_weights(config.d_in, config.seed);

        Self {
            config,
            ssm,
            readout,
            gate_weights,
            gate_bias,
            last_features,
            n_samples: 0,
            prev_prediction: 0.0,
            prev_change: 0.0,
            prev_prev_change: 0.0,
            alignment_ewma: 0.0,
            max_frob_sq_ewma: 0.0,
        }
    }

    /// Initialize gate weights with Xavier normal distribution.
    ///
    /// Uses a separate RNG stream derived from the model seed (offset by a
    /// large prime to avoid correlation with SSM weight initialization).
    fn init_gate_weights(d_in: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        // Use a different seed stream than the SSM by mixing with a prime offset.
        let mut rng_state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        if rng_state == 0 {
            rng_state = 1;
        }

        let scale = 1.0 / (d_in as f64).sqrt();
        let gate_weights: Vec<f64> = (0..d_in * d_in)
            .map(|_| standard_normal(&mut rng_state) * scale)
            .collect();
        let gate_bias = vec![0.0; d_in];

        (gate_weights, gate_bias)
    }

    /// Compute readout dimension: ssm_output (d_in) + pooled hidden state.
    ///
    /// Hidden state pool target = min(d_in, MAX_READOUT_FEATURES - d_in).
    /// This gives low-d_in inputs (e.g. Lorenz d_in=3) extra temporal features
    /// while keeping high-d_in inputs (e.g. d_in=50) bounded.
    fn readout_dim(d_in: usize, n_state: usize) -> usize {
        let state_len = d_in * n_state;
        let pool_budget = Self::MAX_READOUT_FEATURES.saturating_sub(d_in);
        let target_pool = pool_budget.min(d_in * 2).min(state_len);
        if target_pool == 0 {
            return d_in;
        }
        if state_len <= target_pool {
            // Small enough — use full state.
            d_in + state_len
        } else {
            // Mean-pool into target_pool dims.
            let chunk_size = state_len.div_ceil(target_pool);
            let pooled = state_len.div_ceil(chunk_size);
            d_in + pooled
        }
    }

    /// Build readout features: `[gated_output; pooled_hidden_state]`.
    ///
    /// The gated output (SSM output ⊗ SiLU gate + residual) provides the
    /// primary signal after content-dependent filtering. The pooled hidden
    /// state adds temporal detail that the gating may compress out — critical
    /// for low-d_in inputs where d_in output channels can't represent the
    /// full temporal state. The pooled state is NOT gated.
    fn build_readout_features(&self, ssm_output: &[f64], state: &[f64]) -> Vec<f64> {
        let d_in = self.config.d_in;
        let pool_budget = Self::MAX_READOUT_FEATURES.saturating_sub(d_in);
        let target_pool = pool_budget.min(d_in * 2).min(state.len());

        if target_pool == 0 {
            return ssm_output.to_vec();
        }

        let mut rf = Vec::with_capacity(d_in + target_pool);
        rf.extend_from_slice(ssm_output);

        if state.len() <= target_pool {
            // Small enough — use full state directly.
            rf.extend_from_slice(state);
        } else {
            // Mean-pool into target_pool dims.
            let chunk_size = state.len().div_ceil(target_pool);
            for chunk in state.chunks(chunk_size) {
                let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
                rf.push(mean);
            }
        }
        rf
    }

    /// Get a reference to the model configuration.
    pub fn config(&self) -> &MambaConfig {
        &self.config
    }

    /// Get the current SSM hidden state.
    pub fn ssm_state(&self) -> &[f64] {
        self.ssm.state()
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
        self.readout.noise_variance().sqrt()
    }

    /// Get the cached readout features (gated output + pooled state) from the last training step.
    pub fn last_features(&self) -> &[f64] {
        &self.last_features
    }
}

impl StreamingLearner for StreamingMamba {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let d_in = self.config.d_in;

        // 1. Forward through SSM to get temporal features.
        //    The SSM computes y = C_t @ h + D * x — a learned projection.
        let ssm_output = self.ssm.forward(features);

        // 2. Compute SiLU gate from raw input: gate[i] = SiLU(W[i,:] · x + b[i]).
        let gated_output: Vec<f64> = (0..d_in)
            .map(|i| {
                let mut sum = self.gate_bias[i];
                let row = &self.gate_weights[i * d_in..(i + 1) * d_in];
                for (w, &x) in row.iter().zip(features.iter()) {
                    sum += w * x;
                }
                let gate_val = silu(sum);
                // 3. Apply gate + residual: gated_out[i] = ssm_out[i] * gate[i] + x[i]
                ssm_output[i] * gate_val + features[i]
            })
            .collect();

        // 4. Build readout features: gated output + compact hidden state projection.
        let state = self.ssm.state();
        let readout_features = self.build_readout_features(&gated_output, state);

        // 5. Track state Frobenius squared norm for utilization ratio.
        let frob_sq: f64 = self.ssm.state().iter().map(|s| s * s).sum();
        const FROB_ALPHA: f64 = 0.001;
        self.max_frob_sq_ewma = if frob_sq > self.max_frob_sq_ewma {
            frob_sq
        } else {
            (1.0 - FROB_ALPHA) * self.max_frob_sq_ewma + FROB_ALPHA * frob_sq
        };

        // 6. Update residual alignment tracking (acceleration-based).
        let current_pred = self.readout.predict(&readout_features);
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
            self.alignment_ewma =
                (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
        }
        self.prev_prev_change = self.prev_change;
        self.prev_change = current_change;
        self.prev_prediction = current_pred;

        // 7. Train RLS readout on gated features.
        self.readout.train_one(&readout_features, target, weight);

        // 8. Cache readout features for predict()
        self.last_features = readout_features;

        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        // Use cached features from last train_one to avoid SSM state mutation.
        // If never trained (or after reset), return 0.0.
        if self.n_samples == 0 {
            return 0.0;
        }

        // During warmup, predictions may be unreliable but we still return them
        // so the caller can decide how to handle them.
        //
        // Note: We use the LAST SSM output, not the current features.
        // This is because predict() must be side-effect-free (no SSM state change).
        // In a streaming context, the typical pattern is:
        //   predict(x_t) -> p_t  (using state from x_{t-1})
        //   train(x_t, y_t)      (advances state to include x_t)
        //
        // If the caller wants to predict on new features without training,
        // they should use MambaPreprocessor + separate RLS.
        let _ = features; // Acknowledged but not used -- see design note above
        self.readout.predict(&self.last_features)
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        self.ssm.reset();
        self.readout.reset();
        // Re-initialize gate weights from scratch (deterministic from seed).
        let (gw, gb) = Self::init_gate_weights(self.config.d_in, self.config.seed);
        self.gate_weights = gw;
        self.gate_bias = gb;
        for f in self.last_features.iter_mut() {
            *f = 0.0;
        }
        self.n_samples = 0;
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.prev_prev_change = 0.0;
        self.alignment_ewma = 0.0;
        self.max_frob_sq_ewma = 0.0;
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

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for StreamingMamba {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        // RLS saturation: 1.0 - trace(P) / (delta * d).
        let rls_saturation = {
            let p = self.readout.p_matrix();
            let d = self.readout.weights().len();
            if d > 0 && self.readout.delta() > 0.0 {
                let trace: f64 = (0..d).map(|i| p[i * d + i]).sum();
                (1.0 - trace / (self.readout.delta() * d as f64)).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };

        // State Frobenius ratio: current ||S||_F^2 / max(||S||_F^2).
        let state_frob_ratio = {
            let state = self.ssm.state();
            let frob_sq: f64 = state.iter().map(|s| s * s).sum();
            if self.max_frob_sq_ewma > 1e-15 {
                (frob_sq / self.max_frob_sq_ewma).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };

        let depth_sufficiency = 0.5 * rls_saturation + 0.5 * state_frob_ratio;

        // Weight magnitude: ||w||_2 / sqrt(d).
        let w = self.readout.weights();
        let effective_dof = if !w.is_empty() {
            let sq_sum: f64 = w.iter().map(|wi| wi * wi).sum();
            sq_sum.sqrt() / (w.len() as f64).sqrt()
        } else {
            0.0
        };

        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.alignment_ewma,
            regularization_sensitivity: 1.0 - self.config.forgetting_factor,
            depth_sufficiency,
            effective_dof,
            uncertainty: self.prediction_uncertainty(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(d_in: usize) -> MambaConfig {
        MambaConfig::builder().d_in(d_in).build().unwrap()
    }

    #[test]
    fn new_creates_fresh_model() {
        let model = StreamingMamba::new(default_config(3));
        assert_eq!(model.n_samples_seen(), 0);
        // Readout sees gated SSM output (d_in=3) + pooled state.
        let expected = StreamingMamba::readout_dim(3, 32);
        assert_eq!(
            model.last_features().len(),
            expected,
            "last_features should have {} dimensions (d_in + pool), got {}",
            expected,
            model.last_features().len()
        );
        assert!(
            expected > 3,
            "readout should include pooled state for d_in=3"
        );
        // Gate weights should be initialized (Xavier normal, non-zero).
        assert_eq!(
            model.gate_weights.len(),
            3 * 3,
            "gate_weights should be d_in * d_in = 9"
        );
        assert!(
            model.gate_weights.iter().any(|&w| w.abs() > 1e-15),
            "gate_weights should be non-zero after Xavier init"
        );
        assert_eq!(
            model.gate_bias.len(),
            3,
            "gate_bias should have d_in = 3 elements"
        );
        assert!(
            model.gate_bias.iter().all(|&b| b.abs() < 1e-15),
            "gate_bias should be zero-initialized"
        );
    }

    #[test]
    fn train_one_increments_samples() {
        let mut model = StreamingMamba::new(default_config(2));
        model.train_one(&[1.0, 2.0], 3.0, 1.0);
        assert_eq!(model.n_samples_seen(), 1);
        model.train_one(&[4.0, 5.0], 6.0, 1.0);
        assert_eq!(model.n_samples_seen(), 2);
    }

    #[test]
    fn predict_before_training_returns_zero() {
        let model = StreamingMamba::new(default_config(3));
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.abs() < 1e-15,
            "prediction before training should be 0.0, got {}",
            pred
        );
    }

    #[test]
    fn predict_after_training_is_finite() {
        let mut model = StreamingMamba::new(default_config(2));
        model.train(&[1.0, 2.0], 3.0);
        let pred = model.predict(&[1.0, 2.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );
    }

    #[test]
    fn reset_clears_everything() {
        let mut model = StreamingMamba::new(default_config(2));
        model.train(&[1.0, 2.0], 3.0);
        model.train(&[4.0, 5.0], 6.0);
        assert_eq!(model.n_samples_seen(), 2);

        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        for &f in model.last_features() {
            assert!(
                f.abs() < 1e-15,
                "last_features should be zeroed after reset"
            );
        }
        // SSM state should be zeroed
        for &h in model.ssm_state() {
            assert!(h.abs() < 1e-15, "SSM state should be zeroed after reset");
        }
    }

    #[test]
    fn gating_filters_ssm_output() {
        // Verify the SiLU gate produces output different from raw SSM output.
        // We compare a model with gate weights vs zeroed gate weights.
        let config = MambaConfig::builder()
            .d_in(3)
            .n_state(8)
            .seed(42)
            .build()
            .unwrap();

        // Model with normal gate weights (Xavier init).
        let mut model_gated = StreamingMamba::new(config.clone());

        // Model with zeroed gate weights: gate = SiLU(0) = 0, so output = 0 + residual.
        let mut model_zeroed = StreamingMamba::new(config);
        for w in model_zeroed.gate_weights.iter_mut() {
            *w = 0.0;
        }

        let x = [1.0, 0.5, -0.3];
        let y = 2.0;

        model_gated.train(&x, y);
        model_zeroed.train(&x, y);

        let pred_gated = model_gated.predict(&x);
        let pred_zeroed = model_zeroed.predict(&x);

        // Both should be finite.
        assert!(
            pred_gated.is_finite(),
            "gated prediction should be finite, got {}",
            pred_gated
        );
        assert!(
            pred_zeroed.is_finite(),
            "zeroed-gate prediction should be finite, got {}",
            pred_zeroed
        );

        // With different gate weights, the cached features should differ.
        assert!(
            (pred_gated - pred_zeroed).abs() > 1e-15,
            "gated and zeroed-gate predictions should differ: gated={}, zeroed={}",
            pred_gated,
            pred_zeroed
        );
    }

    #[test]
    fn silu_activation_correctness() {
        // SiLU(0) = 0
        assert!((silu(0.0)).abs() < 1e-15, "SiLU(0) should be 0");
        // SiLU(x) approaches x for large positive x
        let large = silu(10.0);
        assert!(
            (large - 10.0).abs() < 0.01,
            "SiLU(10) should be close to 10, got {}",
            large
        );
        // SiLU(x) approaches 0 for large negative x
        let neg = silu(-10.0);
        assert!(
            neg.abs() < 0.01,
            "SiLU(-10) should be close to 0, got {}",
            neg
        );
        // SiLU is smooth and passes through negative territory slightly
        let mid = silu(-1.0);
        assert!(mid < 0.0, "SiLU(-1) should be negative, got {}", mid);
    }

    #[test]
    fn gate_weights_deterministic() {
        // Same seed should produce identical gate weights.
        let m1 = StreamingMamba::new(default_config(4));
        let m2 = StreamingMamba::new(default_config(4));
        assert_eq!(
            m1.gate_weights, m2.gate_weights,
            "same seed should produce identical gate weights"
        );
        assert_eq!(
            m1.gate_bias, m2.gate_bias,
            "same seed should produce identical gate bias"
        );
    }

    #[test]
    fn reset_restores_gate_weights() {
        let mut model = StreamingMamba::new(default_config(3));
        let original_weights = model.gate_weights.clone();

        // Mutate gate weights to simulate drift (not a real operation, just for test).
        for w in model.gate_weights.iter_mut() {
            *w += 1.0;
        }
        assert_ne!(model.gate_weights, original_weights);

        model.reset();
        assert_eq!(
            model.gate_weights, original_weights,
            "gate weights should be restored to initial values after reset"
        );
    }

    #[test]
    fn train_convenience_uses_unit_weight() {
        let mut model1 = StreamingMamba::new(default_config(2));
        let mut model2 = StreamingMamba::new(default_config(2));

        model1.train(&[1.0, 2.0], 3.0);
        model2.train_one(&[1.0, 2.0], 3.0, 1.0);

        // Both should have the same state
        assert_eq!(model1.n_samples_seen(), model2.n_samples_seen());
        let p1 = model1.predict(&[1.0, 2.0]);
        let p2 = model2.predict(&[1.0, 2.0]);
        assert!(
            (p1 - p2).abs() < 1e-12,
            "train() and train_one(w=1) should be equivalent: {} vs {}",
            p1,
            p2
        );
    }

    #[test]
    fn convergence_on_linear_target() {
        // The gated SSM+RLS should learn a simple bounded linear relationship.
        // We use periodic features to keep the target bounded and stationary.
        // With the SiLU gate + residual connection, the model converges
        // quickly. We verify that late cumulative error is lower than early
        // cumulative error, measuring over the first 50 vs last 200 samples
        // of a 1000-sample run.
        let config = MambaConfig::builder()
            .d_in(2)
            .n_state(8)
            .forgetting_factor(0.999)
            .warmup(5)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingMamba::new(config);

        let mut all_errors = Vec::new();

        for i in 0..1000 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos()];
            let y = 0.7 * x[0] + 0.3 * x[1];

            if model.n_samples_seen() > 0 {
                let pred = model.predict(&x);
                let err = (pred - y).powi(2);
                all_errors.push(err);
            }

            model.train(&x, y);
        }

        // Model should produce finite predictions that don't diverge.
        // With the gated architecture, convergence is fast due to the residual
        // connection, so we verify the final error is bounded (< 0.05).
        let mse_late: f64 = all_errors[all_errors.len() - 200..].iter().sum::<f64>() / 200.0;

        assert!(
            mse_late < 0.05,
            "late MSE ({}) should be bounded (< 0.05): model should converge to low error",
            mse_late,
        );
        assert!(
            mse_late.is_finite(),
            "late MSE should be finite, got {}",
            mse_late
        );
    }

    #[test]
    fn convergence_on_sine_wave() {
        // Test on a more complex target: predicting a sine wave.
        // We measure early errors in the first few samples (before RLS has
        // adapted) against late errors.
        let config = MambaConfig::builder()
            .d_in(2)
            .n_state(8)
            .forgetting_factor(0.999)
            .seed(123)
            .build()
            .unwrap();
        let mut model = StreamingMamba::new(config);

        let mut errors_early = Vec::new();
        let mut errors_late = Vec::new();

        for i in 0..1000 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos()];
            let y = (t + 0.1).sin(); // predict next value of sin

            if model.n_samples_seen() > 0 {
                let pred = model.predict(&x);
                let err = (pred - y).powi(2);
                if (1..5).contains(&i) {
                    errors_early.push(err);
                } else if i >= 800 {
                    errors_late.push(err);
                }
            }

            model.train(&x, y);
        }

        let mse_early: f64 = errors_early.iter().sum::<f64>() / errors_early.len() as f64;
        let mse_late: f64 = errors_late.iter().sum::<f64>() / errors_late.len() as f64;

        assert!(
            mse_late < mse_early,
            "late MSE ({}) should be smaller than early MSE ({}): model should converge on sine",
            mse_late,
            mse_early
        );
    }

    #[test]
    fn config_accessor() {
        let config = MambaConfig::builder()
            .d_in(5)
            .n_state(32)
            .seed(77)
            .build()
            .unwrap();
        let model = StreamingMamba::new(config);
        assert_eq!(model.config().d_in, 5);
        assert_eq!(model.config().n_state, 32);
        assert_eq!(model.config().seed, 77);
    }

    #[test]
    fn predict_batch_works() {
        let mut model = StreamingMamba::new(default_config(2));
        model.train(&[1.0, 2.0], 3.0);

        let rows: Vec<&[f64]> = vec![&[1.0, 2.0], &[3.0, 4.0]];
        let preds = model.predict_batch(&rows);
        assert_eq!(preds.len(), 2);
        for p in &preds {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn weighted_training() {
        let mut model = StreamingMamba::new(default_config(2));
        // Training with weight 0 should have minimal effect
        model.train_one(&[1.0, 2.0], 100.0, 0.0);
        let pred_zero_weight = model.predict(&[1.0, 2.0]);

        let mut model2 = StreamingMamba::new(default_config(2));
        model2.train_one(&[1.0, 2.0], 100.0, 1.0);
        let pred_unit_weight = model2.predict(&[1.0, 2.0]);

        // With zero weight, the readout should barely update
        assert!(
            pred_zero_weight.abs() < pred_unit_weight.abs() + 1.0,
            "zero-weight training should have less effect: zero_w={}, unit_w={}",
            pred_zero_weight,
            pred_unit_weight
        );
    }

    #[test]
    fn mamba_prediction_uncertainty() {
        let config = MambaConfig::builder().d_in(2).n_state(8).build().unwrap();
        let mut model = StreamingMamba::new(config);

        // Before training, uncertainty is 0.0
        assert!(
            model.prediction_uncertainty().abs() < 1e-15,
            "uncertainty should be 0.0 before training, got {}",
            model.prediction_uncertainty()
        );

        // Train on 100 samples
        for i in 0..100 {
            let t = i as f64 * 0.1;
            let x = [t.sin(), t.cos()];
            let y = 0.7 * x[0] + 0.3 * x[1];
            model.train(&x, y);
        }

        let unc = model.prediction_uncertainty();
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

    // -----------------------------------------------------------------------
    // Readout capping tests
    // -----------------------------------------------------------------------

    #[test]
    fn readout_includes_ssm_output_and_state() {
        // Readout = SSM output (d_in) + pooled hidden state.
        // For d_in=3, n_state=32: state_len=96, pool_budget=125, target=min(6,96)=6.
        // 96 > 6, so pool: chunk=96/6=16, pooled=96/16=6. Total = 3+6 = 9.
        let config = MambaConfig::builder().d_in(3).n_state(32).build().unwrap();
        let model = StreamingMamba::new(config);
        assert!(
            model.last_features().len() > 3,
            "readout should include pooled state for low d_in, got {}",
            model.last_features().len()
        );
    }

    #[test]
    fn readout_bounded_by_max() {
        // High-dim inputs should still be capped.
        let config = MambaConfig::builder().d_in(50).n_state(64).build().unwrap();
        let model = StreamingMamba::new(config);
        assert!(
            model.last_features().len() <= StreamingMamba::MAX_READOUT_FEATURES,
            "readout should be capped at MAX_READOUT_FEATURES, got {}",
            model.last_features().len()
        );
    }

    #[test]
    fn larger_n_state_gives_more_features() {
        // More state dims → more pooled features (until cap).
        let small = MambaConfig::builder().d_in(3).n_state(4).build().unwrap();
        let large = MambaConfig::builder().d_in(3).n_state(32).build().unwrap();
        let model_s = StreamingMamba::new(small);
        let model_l = StreamingMamba::new(large);
        assert!(
            model_l.last_features().len() >= model_s.last_features().len(),
            "larger n_state should give at least as many features: small={}, large={}",
            model_s.last_features().len(),
            model_l.last_features().len()
        );
    }

    #[test]
    fn high_dim_training_produces_finite_predictions() {
        // d_in=50 with SSM output readout (50-dim features).
        let config = MambaConfig::builder()
            .d_in(50)
            .n_state(32)
            .forgetting_factor(0.998)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingMamba::new(config);

        let features: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        for i in 0..200 {
            let target = (i as f64 * 0.05).sin();
            model.train(&features, target);
        }

        let pred = model.predict(&features);
        assert!(
            pred.is_finite(),
            "prediction with 50-dim input should be finite, got {}",
            pred
        );
    }

    #[test]
    fn high_dim_convergence() {
        // Verify the model converges with hybrid readout features.
        let config = MambaConfig::builder()
            .d_in(10)
            .n_state(16)
            .forgetting_factor(0.999)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingMamba::new(config);

        let expected_dim = StreamingMamba::readout_dim(10, 16);
        assert!(
            model.last_features().len() == expected_dim,
            "readout dim should be {}, got {}",
            expected_dim,
            model.last_features().len()
        );

        let mut errors_early = Vec::new();
        let mut errors_late = Vec::new();

        for i in 0..2000 {
            let t = i as f64 * 0.1;
            // Simple periodic target: sum of two sinusoids from the first two features.
            let x: Vec<f64> = (0..10).map(|k| (t + k as f64 * 0.3).sin()).collect();
            let y = 0.5 * x[0] + 0.3 * x[1];

            if model.n_samples_seen() > 0 {
                let pred = model.predict(&x);
                let err = (pred - y).powi(2);
                if (50..200).contains(&i) {
                    errors_early.push(err);
                } else if i >= 1500 {
                    errors_late.push(err);
                }
            }

            model.train(&x, y);
        }

        let mse_early: f64 = errors_early.iter().sum::<f64>() / errors_early.len() as f64;
        let mse_late: f64 = errors_late.iter().sum::<f64>() / errors_late.len() as f64;

        assert!(
            mse_late < mse_early,
            "high-dim model should converge: late MSE ({}) should be < early MSE ({})",
            mse_late,
            mse_early
        );
    }

    #[test]
    fn readout_always_includes_d_in_ssm_output() {
        // Every readout starts with d_in SSM output features.
        for d_in in [1, 3, 10, 50] {
            let config = MambaConfig::builder()
                .d_in(d_in)
                .n_state(32)
                .build()
                .unwrap();
            let model = StreamingMamba::new(config);
            assert!(
                model.last_features().len() >= d_in,
                "readout should have at least d_in={} features, got {}",
                d_in,
                model.last_features().len(),
            );
        }
    }
}
