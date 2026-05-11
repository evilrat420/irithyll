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
//! gated + state-energy features to the target variable. All components update
//! incrementally, making the model fully streaming with O(1) memory per timestep.
//!
//! # Readout Features
//!
//! For **V1**, the readout sees `2 * d_in` features.
//! For **V3** (Tustin) and **V3Mimo**, the readout sees `d_in + n_groups`
//! features.
//! For **V3Exp** (complex-diagonal exp-trapezoidal), the readout sees
//! `d_in + n_groups + 4 * n_groups * n_state` features: gated output,
//! per-group Frobenius energy, plus per-component `Re(h_{g,n})`,
//! `Im(h_{g,n})`, `|h_{g,n}|`, and `|h_{g,n}|^2`. Surfacing the full
//! complex state plus its modulus and squared modulus gives the linear
//! RLS readout phase-sensitive AND phase-invariant amplitude features
//! needed for parity-class tasks; modulus-squared is the cleanest
//! quadratic-in-state feature that pure linear projections cannot express.
//!
//! ## V1 Readout (2 * d_in)
//!
//! The readout sees `2 * d_in` features:
//!
//! 1. **Gated SSM output** (`d_in` dims): `SSM_output ⊗ SiLU(gate) + residual(x)`.
//!    The SSM's C projection (`y = C_t @ h + D * x`) extracts the learned linear
//!    temporal signal from hidden state.
//!
//! 2. **Per-channel state energy** (`d_in` dims): `energy[d] = ||h[d, :]||_2`.
//!    The L2 norm of each channel's `n_state`-dimensional state vector captures
//!    how much temporal activation each channel carries. This is a nonlinear
//!    summary that complements the C projection's linear combination, and scales
//!    naturally with `n_state` (more state elements accumulate more energy).
//!
//! # Training Flow
//!
//! ```text
//! features ──→ SSM.forward() ──→ ssm_output ──┐
//!    │                                          ├──→ ssm_output ⊗ SiLU(gate)
//!    └──→ gate = SiLU(W_gate · x + b)  ────────┘         │
//!                                                    + residual(x)
//!                                                         │
//!                                             ┌──── gated_output (d_in)
//!                                             │
//!    SSM hidden state h ──→ per-channel ──────┤
//!                           L2 norm           └──── state_energy (d_in)
//!                                                         │
//!                                                  [gated; energy] (2*d_in)
//!                                                         │
//!                                                   RLS.train_one()
//! ```
//!
//! # Prediction
//!
//! `predict()` uses the cached readout features from the most recent `train_one()`
//! call. This avoids a side-effect (advancing SSM state) during prediction,
//! maintaining the contract that `predict()` is read-only. If no training has
//! occurred, returns 0.0.

pub mod features;
pub mod learner_impl;

use irithyll_core::continual::NeuronRegeneration;
use irithyll_core::ssm::{
    SSMLayer, SelectiveSSM, SelectiveSSMBD, SelectiveSSMv3, SelectiveSSMv3Exp, SelectiveSSMv3Mimo,
};

use crate::learner::StreamingLearner;
use crate::learners::RecursiveLeastSquares;
use crate::ssm::mamba_config::{MambaConfig, MambaVersion};

use irithyll_core::rng::standard_normal;

// ---------------------------------------------------------------------------
// SSM variant dispatch
// ---------------------------------------------------------------------------

/// Internal SSM variant for V1/V3/V3Exp/V3Mimo/BD dispatch.
pub(crate) enum SSMVariant {
    /// Mamba-1: per-channel scalar processing, real states, ZOH discretization.
    V1(SelectiveSSM),
    /// Mamba-3 MIMO-lite: Tustin complex, grouped-channel (backward compat).
    V3(SelectiveSSMv3),
    /// Mamba-3 paper-spec: exp-trapezoidal 3-term + data-dependent λ_t.
    V3Exp(SelectiveSSMv3Exp),
    /// Mamba-3 paper-spec MIMO: true rank-R matrix-valued state H ∈ R^{N×P}.
    V3Mimo(SelectiveSSMv3Mimo),
    /// BD-LRU: block-diagonal linear recurrence with dense m×m blocks.
    BD(SelectiveSSMBD),
}

impl SSMVariant {
    /// Forward one timestep through the SSM.
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        match self {
            SSMVariant::V1(ssm) => ssm.forward(input),
            SSMVariant::V3(ssm) => ssm.forward(input),
            SSMVariant::V3Exp(ssm) => ssm.forward(input),
            SSMVariant::V3Mimo(ssm) => ssm.forward(input),
            SSMVariant::BD(ssm) => ssm.forward(input),
        }
    }

    /// Get a reference to the current hidden state.
    fn state(&self) -> &[f64] {
        match self {
            SSMVariant::V1(ssm) => ssm.state(),
            SSMVariant::V3(ssm) => ssm.state(),
            SSMVariant::V3Exp(ssm) => ssm.state(),
            SSMVariant::V3Mimo(ssm) => ssm.state(),
            SSMVariant::BD(ssm) => ssm.state(),
        }
    }

    /// Reset hidden state to zeros.
    fn reset(&mut self) {
        match self {
            SSMVariant::V1(ssm) => ssm.reset(),
            SSMVariant::V3(ssm) => ssm.reset(),
            SSMVariant::V3Exp(ssm) => ssm.reset(),
            SSMVariant::V3Mimo(ssm) => ssm.reset(),
            SSMVariant::BD(ssm) => ssm.reset(),
        }
    }
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
/// The readout sees `2 * d_in` features: the gated SSM output (`d_in`)
/// plus per-channel state energy (`d_in`). The gated output carries the
/// C-projected temporal signal, while the state energy (L2 norm of each
/// channel's hidden state vector) captures nonlinear temporal activation
/// patterns that the linear C projection may miss. This is invariant to
/// `n_state` in dimension (always `d_in` extra features) while scaling
/// naturally in magnitude (larger state accumulates more energy).
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
    pub(crate) config: MambaConfig,
    /// Selective SSM for temporal feature extraction (V1 or V3).
    pub(crate) ssm: SSMVariant,
    /// RLS readout layer for prediction.
    pub(crate) readout: RecursiveLeastSquares,
    /// SiLU gate projection weights: d_in × d_in matrix (row-major).
    /// Maps raw input to gate signal: `gate[i] = SiLU(sum_j(W[i*d+j]*x[j]) + b[i])`.
    pub(crate) gate_weights: Vec<f64>,
    /// SiLU gate bias vector (d_in elements).
    pub(crate) gate_bias: Vec<f64>,
    /// Cached readout features (gated SSM output) from the most recent
    /// `train_one` call.
    pub(crate) last_features: Vec<f64>,
    /// Total samples trained on.
    pub(crate) n_samples: u64,
    /// Previous prediction for residual alignment tracking.
    pub(crate) prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    pub(crate) prev_change: f64,
    /// Change from two steps ago, for acceleration-based alignment.
    pub(crate) prev_prev_change: f64,
    /// EWMA of residual alignment signal.
    pub(crate) alignment_ewma: f64,
    /// EWMA of maximum Frobenius squared norm of SSM state for utilization ratio.
    pub(crate) max_frob_sq_ewma: f64,
    /// Optional plasticity guard for maintaining learning capacity.
    pub(crate) plasticity_guard: Option<NeuronRegeneration>,
    /// Snapshot of per-channel state energy from previous step.
    pub(crate) prev_state_energy: Vec<f64>,
    /// Cached SSM output (d_in dims) from the most recent `train_one` call.
    ///
    /// Used by `predict()` to reconstruct gated readout features for the
    /// current input without mutating SSM state. Combining the cached SSM
    /// temporal output with the current input's gate and residual gives a
    /// side-effect-free prediction that uses the actual input features rather
    /// than stale ones from the previous timestep.
    pub(crate) last_ssm_output: Vec<f64>,
    /// Optional fixed random projection weights for the V3Exp tanh feature lift.
    ///
    /// When `Some`, the V3Exp readout appends `tanh(W·z + b)` features after
    /// the base readout, where `z = [base_features ; (2*input - 1)]`. The
    /// weights are sampled once at construction (deterministic from `config.seed`)
    /// and never updated — only the linear RLS readout on top of the lifted
    /// features is online-trained. This is a Random Feature Network in the
    /// Rahimi & Recht (2008) sense: linear regression on `n_lift` random
    /// nonlinear features approximates kernel ridge regression in the limit
    /// `n_lift → ∞`. Universal approximation; no extra training overhead.
    pub(crate) lift_weights: Option<Vec<f64>>,
    /// Optional fixed random projection biases for the V3Exp tanh feature lift.
    pub(crate) lift_bias: Option<Vec<f64>>,
    /// Number of tanh random features in the V3Exp lift (0 for non-V3Exp).
    pub(crate) n_lift: usize,
    /// Pre-lift feature dimension (base features + d_in for raw-input augmentation).
    pub(crate) lift_input_dim: usize,
}

impl StreamingMamba {
    /// Create a new streaming Mamba model from the given configuration.
    ///
    /// Initializes the SSM with random weights (seeded by `config.seed`),
    /// a SiLU gate with Xavier-initialized weights, and an RLS readout
    /// with the specified forgetting factor and P matrix scale.
    ///
    /// Readout feature dimensions:
    /// - V1: `2 * d_in`
    /// - V3 / V3Mimo: `d_in + n_groups` (Frobenius energy per group)
    /// - V3Exp: `d_in + n_groups + 4 * n_groups * n_state` (adds
    ///   per-component Re(h), Im(h), |h|, and |h|^2 to expose phase plus
    ///   phase-invariant amplitude — modulus-squared injects a
    ///   quadratic-in-state feature for parity-class tasks)
    /// - BlockDiagonal: `d_in + d_in/block_size`
    pub fn new(config: MambaConfig) -> Self {
        let ssm = match config.version {
            MambaVersion::V1 => {
                SSMVariant::V1(SelectiveSSM::new(config.d_in, config.n_state, config.seed))
            }
            MambaVersion::V3 => SSMVariant::V3(SelectiveSSMv3::new(
                config.d_in,
                config.n_state,
                config.n_groups,
                config.seed,
            )),
            MambaVersion::V3Exp { use_bcnorm } => SSMVariant::V3Exp(SelectiveSSMv3Exp::new(
                config.d_in,
                config.n_state,
                config.n_groups,
                config.seed,
                use_bcnorm,
            )),
            MambaVersion::V3Mimo { rank, use_bcnorm } => {
                SSMVariant::V3Mimo(SelectiveSSMv3Mimo::new(
                    config.d_in,
                    config.n_state,
                    config.n_groups,
                    rank,
                    config.seed,
                    use_bcnorm,
                ))
            }
            MambaVersion::BlockDiagonal { block_size } => SSMVariant::BD(SelectiveSSMBD::new(
                config.d_in,
                config.n_state,
                block_size,
                config.seed,
            )),
        };
        let readout = RecursiveLeastSquares::with_delta(config.forgetting_factor, config.delta_rls);
        let readout_dim = Self::readout_dim_for_config(&config);
        let last_features = vec![0.0; readout_dim];

        let (gate_weights, gate_bias) = Self::init_gate_weights(config.d_in, config.seed);

        let plasticity_n_units = match config.version {
            MambaVersion::V1 => config.d_in,
            MambaVersion::V3 | MambaVersion::V3Exp { .. } | MambaVersion::V3Mimo { .. } => {
                config.n_groups
            }
            MambaVersion::BlockDiagonal { block_size } => config.d_in / block_size,
        };
        // Create plasticity guard if a PlasticityConfig was provided.
        // Tracks SSM channels (group_size=1 = per-channel tracking).
        let plasticity_guard = config.plasticity.as_ref().map(|p| {
            NeuronRegeneration::new(
                plasticity_n_units,
                1, // group_size = 1 (per-channel tracking)
                p.regen_fraction,
                p.regen_interval,
                p.utility_alpha,
                config.seed.wrapping_add(0x_DEAD_CAFE),
            )
        });
        let prev_state_energy = vec![0.0; plasticity_n_units];
        let last_ssm_output = vec![0.0; config.d_in];

        // Random tanh feature lift (V3Exp only). The lift projects ONLY raw
        // input (rescaled to `{-1, +1}`) and produces `n_lift` tanh-rand
        // features. Random projection weights are deterministic from the seed
        // and never updated; only the linear RLS readout above adapts.
        // See `n_lift_for_config` doc for the "why raw input only" rationale.
        let n_lift = Self::n_lift_for_config(&config);
        let lift_input_dim = config.d_in;
        let (lift_weights, lift_bias) = if n_lift > 0 {
            Self::init_lift_weights(lift_input_dim, n_lift, config.seed)
        } else {
            (None, None)
        };

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
            plasticity_guard,
            prev_state_energy,
            last_ssm_output,
            lift_weights,
            lift_bias,
            n_lift,
            lift_input_dim,
        }
    }

    /// Initialize random tanh-feature lift weights (Gaussian projection).
    ///
    /// Returns `(W, b)` with `W` being `n_lift x lift_input_dim` row-major and
    /// `b` being `n_lift`-length. Weights are sampled from `N(0, 1/sqrt(d))`
    /// (Glorot-style init for tanh activations), biases from `Uniform(-1, 1)`.
    /// Determinism: derived from `seed.wrapping_add(0xF1F7_F1F7_F1F7_F1F7)`.
    pub(crate) fn init_lift_weights(
        lift_input_dim: usize,
        n_lift: usize,
        seed: u64,
    ) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        let mut rng_state = seed.wrapping_add(0xF1F7_F1F7_F1F7_F1F7);
        if rng_state == 0 {
            rng_state = 1;
        }
        let scale = 1.0 / (lift_input_dim as f64).sqrt();
        let weights: Vec<f64> = (0..n_lift * lift_input_dim)
            .map(|_| standard_normal(&mut rng_state) * scale)
            .collect();
        let biases: Vec<f64> = (0..n_lift)
            .map(|_| {
                // Box-Muller-tail style uniform: derive from another standard_normal,
                // mod into [-1, 1]. Standard_normal is N(0, 1), so clamp and rescale.
                standard_normal(&mut rng_state).tanh()
            })
            .collect();
        (Some(weights), Some(biases))
    }

    /// Initialize gate weights with Xavier normal distribution.
    pub(crate) fn init_gate_weights(d_in: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
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

    /// Compute the BASE readout dimension (without the random feature lift).
    ///
    /// V3Exp adds `n_lift_for_config(config)` more features at the readout
    /// surface — see [`Self::readout_dim_for_config`] for the full count.
    pub(crate) fn base_readout_dim_for_config(config: &MambaConfig) -> usize {
        match config.version {
            MambaVersion::V1 => config.d_in * 2,
            // V3 (Tustin) and V3Mimo: gated output (d_in) + per-group state energy (n_groups).
            // V3Mimo Frobenius norm is still per-group (n_groups), not per channel.
            MambaVersion::V3 | MambaVersion::V3Mimo { .. } => config.d_in + config.n_groups,
            // V3Exp BASE: gated output (d_in) + per-group Frobenius energy (n_groups)
            // + per-component complex-state surfacing (4 * n_groups * n_state):
            // Re(h_{g,n}), Im(h_{g,n}), |h_{g,n}|, |h_{g,n}|^2. The lift adds
            // `n_lift` more features on top — see `n_lift_for_config`.
            MambaVersion::V3Exp { .. } => {
                config.d_in + config.n_groups + 4 * config.n_groups * config.n_state
            }
            MambaVersion::BlockDiagonal { block_size } => config.d_in + config.d_in / block_size,
        }
    }

    /// Number of tanh random features added at the V3Exp readout (0 for others).
    ///
    /// **Why a lift is necessary for V3Exp.** A closed-form sufficiency probe
    /// shows that linear regression on the V3Exp BASE features alone cannot
    /// solve high-degree-polynomial targets like multi-bit XOR parity (the
    /// canonical task that requires an SSM with complex eigenvalues, per
    /// Goel et al. 2022 / Abbe et al. 2023). The bilinear cell update
    /// `bx = (W_B·x) * mean(x_group)` reaches at most degree-4 in the input
    /// bits via `|h|^2`; for k-bit XOR with `k > 4`, no linear readout on
    /// degree-≤4 features can reach the target — empirically verified at
    /// 0.49 batch-LS accuracy on 8-bit parity.
    ///
    /// **Why random tanh features are the principled answer.** Linear
    /// regression on `n_lift` random nonlinear features approximates kernel
    /// ridge regression with the Gaussian RBF kernel as `n_lift → ∞`
    /// (Rahimi & Recht 2008, Williams 1998). The Gaussian RBF kernel is a
    /// universal approximator. The lift is *not* trained — only the linear
    /// RLS layer above it adapts; the random projection is fixed by `seed`,
    /// preserving determinism and adding zero per-step training cost.
    ///
    /// **Why we project from raw input only.** The tanh-rand lift takes raw
    /// input bits (rescaled to `{-1, +1}` for symmetry under bit-flip) as
    /// its sole input. Mixing in the V3Exp base features dilutes the bit
    /// signal — base features carry temporally-smoothed amplitudes that for
    /// i.i.d. tasks like parity add noise rather than signal. The base
    /// features still feed the readout DIRECTLY (linear path), so the model
    /// retains all of V3Exp's complex-mode information; the lift is a
    /// pure-input nonlinear capability layered on top. Empirically,
    /// pure-input tanh-rand reaches 1.00 batch-LS on 8-bit parity at 256
    /// features; combined input+features lift dilutes to 0.50.
    ///
    /// **Why `max(64, 32 * d_in)`.** Empirical sample-complexity for random
    /// features scales linearly with input dimension on smooth targets;
    /// 32× headroom gives a strong safety margin without exploding readout
    /// dim. The 64 floor keeps small-d models above the universal-
    /// approximation noise floor.
    ///
    /// Bounded-feature invariant: tanh outputs are in `[-1, 1]`, satisfying
    /// the bounded-readout requirement automatically.
    pub(crate) fn n_lift_for_config(config: &MambaConfig) -> usize {
        match config.version {
            MambaVersion::V3Exp { .. } => (32 * config.d_in).max(64),
            _ => 0,
        }
    }

    /// Compute readout dimension (base + lift).
    pub(crate) fn readout_dim_for_config(config: &MambaConfig) -> usize {
        Self::base_readout_dim_for_config(config) + Self::n_lift_for_config(config)
    }

    /// Build readout features based on the Mamba version.
    pub(crate) fn build_readout_features(
        &self,
        gated_output: &[f64],
        state: &[f64],
        raw_input: &[f64],
    ) -> Vec<f64> {
        features::build_readout_features(self, gated_output, state, raw_input)
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
    #[inline]
    pub fn prediction_uncertainty(&self) -> f64 {
        self.readout.noise_variance().sqrt()
    }

    /// Get the cached readout features (gated output + state energy) from the last training step.
    pub fn last_features(&self) -> &[f64] {
        &self.last_features
    }
}

impl StreamingLearner for StreamingMamba {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        learner_impl::train_one(self, features, target, weight);
    }

    fn predict(&self, features: &[f64]) -> f64 {
        learner_impl::predict(self, features)
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        learner_impl::reset(self);
    }

    #[allow(deprecated)]
    fn diagnostics_array(&self) -> [f64; 5] {
        <Self as crate::learner::Tunable>::diagnostics_array(self)
    }

    #[allow(deprecated)]
    fn readout_weights(&self) -> Option<&[f64]> {
        let w = <Self as crate::learner::HasReadout>::readout_weights(self);
        if w.is_empty() {
            None
        } else {
            Some(w)
        }
    }
}

impl crate::learner::Tunable for StreamingMamba {
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

    fn adjust_config(&mut self, lr_multiplier: f64, _lambda_delta: f64) {
        // Scale the RLS readout forgetting factor as the primary tuning knob.
        <crate::learners::RecursiveLeastSquares as crate::learner::Tunable>::adjust_config(
            &mut self.readout,
            lr_multiplier,
            0.0,
        );
    }
}

impl crate::learner::HasReadout for StreamingMamba {
    fn readout_weights(&self) -> &[f64] {
        self.readout.weights()
    }
}

impl crate::automl::DiagnosticSource for StreamingMamba {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
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
mod tests;
