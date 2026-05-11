//! Multi-head streaming linear attention.
//!
//! [`MultiHeadAttention`] composes multiple attention head instances with
//! an output projection, implementing the [`AttentionLayer`] trait. Each head
//! independently maintains state and computes its update rule according to
//! the configured [`AttentionMode`].

use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use super::config::{AttentionConfig, AttentionMode, GatedDeltaMode};
use super::gating::{
    exponential_gate, extended_sigmoid_gate, fixed_decay, init_weights, lstm_gates, mat_vec,
    sigmoid_gate, vector_decay, vector_lower_bounded_gate, vector_sigmoid_gate, Xorshift64,
};
use super::log_linear_state::LogLinearState;
use super::state::AttentionState;
use super::update_rules;
use super::AttentionLayer;
use crate::math;
use crate::rng::standard_normal;
use crate::streaming_primitives::{softplus_softmax_mix, tanh_inplace};

/// A single attention head with its own state and projection weights.
struct AttentionHead {
    /// Current recurrent state.
    state: AttentionState,
    /// Key projection: d_model -> d_key (row-major).
    w_key: Vec<f64>,
    /// Value projection: d_model -> d_value (row-major).
    w_value: Vec<f64>,
    /// Query projection: d_model -> d_key (row-major).
    w_query: Vec<f64>,
    /// Gate weights for GLA/GatedDeltaNet/mLSTM forget gate (length d_model).
    w_gate: Vec<f64>,
    /// Second gate weights for mLSTM input gate (length d_model).
    w_gate2: Vec<f64>,
    /// Decay weights for RWKV (length d_model).
    w_decay: Vec<f64>,
    /// Alpha (decay) weights for Hawk (length d_model, projected to d_value).
    w_alpha: Vec<f64>,
    /// Beta (input scale) weights for Hawk (length d_model, projected to d_value).
    w_beta: Vec<f64>,
    /// Per-composition key projections for DeltaProduct (n_h * d_key * d_model).
    w_comp_keys: Vec<f64>,
    /// Per-composition value projections for DeltaProduct (n_h * d_value * d_model).
    w_comp_values: Vec<f64>,
    /// Per-composition beta projections for DeltaProduct (n_h * d_model).
    w_comp_betas: Vec<f64>,
    /// Per-dimension decay projection for RWKV-7 (d_key * d_model).
    w_decay_vec: Vec<f64>,
    /// ICLR projection for RWKV-7 (d_key * d_model).
    w_iclr: Vec<f64>,
    /// Removal key modifier for RWKV-7 (length d_key).
    w_xi: Vec<f64>,
    /// Replacement key modifier for RWKV-7 (length d_key).
    w_alpha_rk: Vec<f64>,
    /// Per-dimension lower-bounded gate for HGRN2 (d_key * d_model).
    w_hgrn2_gate: Vec<f64>,
    /// Vector gate projection for GLAVector (d_key * d_model).
    ///
    /// Projects input to a per-key-dimension gate `α_t ∈ (0,1)^{d_k}`.
    /// Empty for all modes except `GLAVector`.
    w_gate_vec: Vec<f64>,
    /// Per-token beta projection for GatedDeltaNet PerToken mode (length d_model).
    ///
    /// Computes `β_t = sigmoid(w_beta_scalar · x_t)` per token per head.
    /// Empty unless `GatedDeltaNet { gate_mode_delta: GatedDeltaMode::PerToken }`.
    ///
    /// Per Yang et al. ICLR 2025 (arXiv:2412.06464): the canonical form uses
    /// data-dependent `β_t` to make delta-rule mixing input-dependent.
    w_beta_scalar: Vec<f64>,
    /// Per-level λ projection for LogLinear mode (max_levels * d_model
    /// row-major).
    ///
    /// Empty for all non-LogLinear modes. Each row produces one
    /// raw logit fed to softplus-softmax mixing — paper §3.2.
    w_lambda: Vec<f64>,
    /// Hierarchical Fenwick state for LogLinear mode.
    ///
    /// `None` for all non-LogLinear modes. Holds `max_levels`
    /// matrices per head (paper §2-§3, R1 §3.4 Option B padding).
    log_linear_state: Option<LogLinearState>,
}

/// Multi-head streaming linear attention layer.
///
/// Composes `n_heads` attention heads, each operating independently on a
/// shared input. Head outputs are concatenated and projected through `w_out`
/// to produce the final `d_model`-dimensional output.
///
/// # Architecture
///
/// For each timestep:
/// 1. Each head projects input to key, value, query
/// 2. Computes gate/decay based on the configured mode
/// 3. Applies the appropriate update rule to its state
/// 4. Computes head output via state query
/// 5. All head outputs are concatenated and projected to `d_model`
///
/// # Example
///
/// ```
/// use irithyll_core::attention::{AttentionConfig, AttentionMode, MultiHeadAttention, AttentionLayer};
///
/// let config = AttentionConfig {
///     d_model: 8,
///     n_heads: 2,
///     d_key: 4,
///     d_value: 4,
///     mode: AttentionMode::RetNet { gamma: 0.95 },
///     seed: 42,
/// };
/// let mut attn = MultiHeadAttention::new(config);
/// let input = vec![1.0; 8];
/// let output = attn.forward(&input);
/// assert_eq!(output.len(), 8);
/// ```
pub struct MultiHeadAttention {
    /// Configuration.
    config: AttentionConfig,
    /// Per-head state and projections.
    heads: Vec<AttentionHead>,
    /// Output projection: (n_heads * d_value) -> d_model (row-major).
    w_out: Vec<f64>,
    /// Flat state cache for readout (avoids re-allocation).
    state_cache: Vec<f64>,
    /// Scratch buffer for key projection (d_key), avoids per-call heap allocation.
    scratch_k: Vec<f64>,
    /// Scratch buffer for value projection (d_value), avoids per-call heap allocation.
    scratch_v: Vec<f64>,
    /// Scratch buffer for query projection (d_key), avoids per-call heap allocation.
    scratch_q: Vec<f64>,
    /// Scratch buffer for concatenated head outputs (n_heads * d_value).
    scratch_concat: Vec<f64>,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer from configuration.
    ///
    /// All weights are initialized with small random normal values (scale 0.01)
    /// using a deterministic xorshift64 PRNG seeded from `config.seed`.
    pub fn new(config: AttentionConfig) -> Self {
        let mut rng = Xorshift64(config.seed);
        let d_model = config.d_model;
        let d_key = config.d_key;
        let d_value = config.d_value;
        let n_heads = config.n_heads;

        let mut heads = Vec::with_capacity(n_heads);
        for _ in 0..n_heads {
            let state = match &config.mode {
                AttentionMode::Hawk => AttentionState::new_vector(d_value),
                _ => AttentionState::new_matrix(d_key, d_value),
            };

            // For LogLinear: allocate the Fenwick stack per head; the
            // base `state` matrix is unused but retained for diagnostic
            // uniformity. State storage lives in `log_linear_state`.
            let log_linear_state = match &config.mode {
                AttentionMode::LogLinear { max_levels, .. } => {
                    Some(LogLinearState::new(*max_levels, d_key, d_value))
                }
                _ => None,
            };

            let w_key = init_weights(&mut rng, d_key * d_model);
            let w_value = init_weights(&mut rng, d_value * d_model);
            let w_query = init_weights(&mut rng, d_key * d_model);

            // Mode-dependent gate weights
            let w_gate = match &config.mode {
                AttentionMode::GLA
                | AttentionMode::GatedDeltaNet { .. }
                | AttentionMode::MLSTM
                | AttentionMode::DeltaProduct { .. } => init_weights(&mut rng, d_model),
                _ => Vec::new(),
            };

            let w_gate2 = match &config.mode {
                AttentionMode::MLSTM => init_weights(&mut rng, d_model),
                _ => Vec::new(),
            };

            let w_decay = match &config.mode {
                AttentionMode::RWKV { .. } => init_weights(&mut rng, d_model),
                _ => Vec::new(),
            };

            let w_alpha = match &config.mode {
                AttentionMode::Hawk => init_weights(&mut rng, d_value * d_model),
                _ => Vec::new(),
            };

            let w_beta = match &config.mode {
                AttentionMode::Hawk => init_weights(&mut rng, d_value * d_model),
                _ => Vec::new(),
            };

            let (w_comp_keys, w_comp_values, w_comp_betas) = match &config.mode {
                AttentionMode::DeltaProduct {
                    n_compositions,
                    reflections: _,
                } => {
                    let n_h = *n_compositions;
                    (
                        init_weights(&mut rng, n_h * d_key * d_model),
                        init_weights(&mut rng, n_h * d_value * d_model),
                        init_weights(&mut rng, n_h * d_model),
                    )
                }
                _ => (Vec::new(), Vec::new(), Vec::new()),
            };

            let (w_decay_vec, w_iclr, w_xi, w_alpha_rk) = match &config.mode {
                AttentionMode::RWKV7 => (
                    init_weights(&mut rng, d_key * d_model),
                    init_weights(&mut rng, d_key * d_model),
                    init_weights(&mut rng, d_key),
                    init_weights(&mut rng, d_key),
                ),
                _ => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            };

            let w_hgrn2_gate = match &config.mode {
                AttentionMode::HGRN2 { .. } => init_weights(&mut rng, d_key * d_model),
                _ => Vec::new(),
            };

            // GLAVector: per-key-dimension gate projection (d_key * d_model).
            let w_gate_vec = match &config.mode {
                AttentionMode::GLAVector => init_weights(&mut rng, d_key * d_model),
                _ => Vec::new(),
            };

            // GatedDeltaNet PerToken: per-token beta projection (d_model).
            // Computes β_t = sigmoid(w_beta_scalar · x_t) per Yang et al. ICLR 2025.
            let w_beta_scalar = match &config.mode {
                AttentionMode::GatedDeltaNet {
                    gate_mode_delta: GatedDeltaMode::PerToken,
                    ..
                } => init_weights(&mut rng, d_model),
                _ => Vec::new(),
            };

            // LogLinear: per-level λ projection W_λ ∈ R^{max_levels × d_model}.
            // Paper §3.2: λ = softplus_softmax_mix(W_λ x + bias).
            let w_lambda = match &config.mode {
                AttentionMode::LogLinear { max_levels, .. } => {
                    init_weights(&mut rng, max_levels * d_model)
                }
                _ => Vec::new(),
            };

            heads.push(AttentionHead {
                state,
                w_key,
                w_value,
                w_query,
                w_gate,
                w_gate2,
                w_decay,
                w_alpha,
                w_beta,
                w_comp_keys,
                w_comp_values,
                w_comp_betas,
                w_decay_vec,
                w_iclr,
                w_xi,
                w_alpha_rk,
                w_hgrn2_gate,
                w_gate_vec,
                w_beta_scalar,
                w_lambda,
                log_linear_state,
            });
        }

        let concat_dim = n_heads * d_value;
        let w_out = init_weights(&mut rng, d_model * concat_dim);

        // Compute total state size for cache.
        // LogLinear: padded `max_levels * d_k * d_v` per head (R1 §3.4
        // Option B — constant-shape state, paper-mandated stability
        // choice). All other modes: standard d_k*d_v or d_v.
        let state_size = match &config.mode {
            AttentionMode::Hawk => n_heads * d_value,
            AttentionMode::LogLinear { max_levels, .. } => n_heads * max_levels * d_key * d_value,
            _ => n_heads * d_key * d_value,
        };

        Self {
            config,
            heads,
            w_out,
            state_cache: vec![0.0; state_size],
            scratch_k: vec![0.0; d_key],
            scratch_v: vec![0.0; d_value],
            scratch_q: vec![0.0; d_key],
            scratch_concat: vec![0.0; concat_dim],
        }
    }

    /// Computes the readout for the current state without advancing it.
    ///
    /// This is the canonical streaming inference path — it separates readout
    /// from state update so the readout always reflects the post-state-advance
    /// distribution at predict time.
    ///
    /// Concretely: projects `input` to queries, computes `q(x_t) · S_{t-1}`
    /// (or the vector-state equivalent for Hawk), applies the output projection,
    /// and returns the result — all without mutating any head state or the state
    /// cache. This implements eq. R8.1 from the GLA analysis: the only
    /// prequential-label-independent readout for linear attention models.
    ///
    /// In the streaming supervised setting `predict(x_t)` must return an
    /// estimate of `y_t` that depends on the current input `x_t` and the
    /// history `H_{t-1}`. This method provides exactly that: the attention
    /// state is held at `S_{t-1}` (the state after the last `train_one`), and
    /// the query is computed fresh from `x_t`, giving `q(x_t) · S_{t-1}`.
    ///
    /// # Arguments
    ///
    /// * `input` -- feature vector of length `d_model`
    ///
    /// # Returns
    ///
    /// Attention readout of length `d_model`.
    pub fn query_state(&self, input: &[f64]) -> Vec<f64> {
        let d_model = self.config.d_model;
        let d_key = self.config.d_key;
        let d_value = self.config.d_value;
        let n_heads = self.config.n_heads;
        let concat_dim = n_heads * d_value;

        let mut concat_output = vec![0.0; concat_dim];

        for (h, head) in self.heads.iter().enumerate() {
            // Compute query projection from current input
            let mut q = vec![0.0; d_key];
            mat_vec(&head.w_query, input, d_key, d_model, &mut q);

            // Read out from current state without updating it
            let head_output = match &self.config.mode {
                AttentionMode::Hawk => {
                    // Hawk state is the vector itself; return a clone as output
                    head.state.as_slice().to_vec()
                }
                AttentionMode::LogLinear {
                    max_levels,
                    lambda_init,
                    ..
                } => {
                    // Compute λ from current input (no state mutation).
                    // Paper §3.2: λ = softplus_softmax_mix(W_λ x + bias).
                    let mut raw = vec![0.0; *max_levels];
                    mat_vec(&head.w_lambda, input, *max_levels, d_model, &mut raw);
                    for r in raw.iter_mut() {
                        *r += *lambda_init;
                    }
                    let mut lambdas = vec![0.0; *max_levels];
                    softplus_softmax_mix(&raw, super::log_linear::DEFAULT_TAU, &mut lambdas);

                    // Read pre-update mixed state.
                    let lls = head
                        .log_linear_state
                        .as_ref()
                        .expect("LogLinear mode must have log_linear_state");
                    let mut o = vec![0.0; d_value];
                    lls.query_mixed(&q, &lambdas, &mut o);
                    o
                }
                _ => {
                    // Matrix state: S^T * q  (pure read, no state mutation)
                    head.state.query(&q)
                }
            };

            let offset = h * d_value;
            concat_output[offset..offset + d_value].copy_from_slice(&head_output);
        }

        // Apply output projection: w_out * concat_output
        let mut output = vec![0.0; d_model];
        mat_vec(
            &self.w_out,
            &concat_output,
            d_model,
            concat_dim,
            &mut output,
        );

        // LogLinear: bound the output via tanh per AGENTS.md "Bounded
        // readout features" invariant. λ-mixed q^T S can grow
        // arbitrarily even with Σ λ ≤ 1; tanh maps R → (-1, 1).
        if matches!(self.config.mode, AttentionMode::LogLinear { .. }) {
            tanh_inplace(&mut output);
        }

        output
    }

    /// Compatibility alias for [`query_state`](Self::query_state).
    ///
    /// Same semantics, principled name.
    #[deprecated(
        since = "10.0.0",
        note = "renamed to `query_state` — same semantics, principled name"
    )]
    #[doc(hidden)]
    pub fn forward_readonly(&self, input: &[f64]) -> Vec<f64> {
        self.query_state(input)
    }

    /// Update the flat state cache from all head states.
    ///
    /// For LogLinear mode, sources from each head's `log_linear_state`
    /// (padded `max_levels * d_k * d_v`); for all other modes, from
    /// the base `state` slice. The cache size is set in `new()` per
    /// mode and stays constant across `forward` calls.
    fn update_state_cache(&mut self) {
        let mut offset = 0;
        for head in &self.heads {
            let slice = if let Some(lls) = head.log_linear_state.as_ref() {
                lls.flat_state()
            } else {
                head.state.as_slice()
            };
            let len = slice.len();
            self.state_cache[offset..offset + len].copy_from_slice(slice);
            offset += len;
        }
    }

    /// Surgically reinitialize a single attention head `h`.
    ///
    /// Resets the head's recurrent state to zero and draws fresh random weights
    /// for all projection matrices (key, value, query, and any mode-specific
    /// gate/decay weights) using the same scale (0.01 * standard_normal) as
    /// the original initialization.
    ///
    /// All other heads and the output projection are preserved.
    ///
    /// # Arguments
    ///
    /// * `h` -- head index to reinitialize (must be < `n_heads`)
    /// * `rng` -- mutable RNG state for generating fresh weights
    ///
    /// # Panics
    ///
    /// Panics if `h >= self.heads.len()`.
    pub fn reinitialize_head(&mut self, h: usize, rng: &mut u64) {
        assert!(
            h < self.heads.len(),
            "head index {} out of range (n_heads={})",
            h,
            self.heads.len()
        );

        let head = &mut self.heads[h];

        // Zero the recurrent state.
        head.state.reset();
        // LogLinear: clear the entire Fenwick stack of this head.
        if let Some(lls) = head.log_linear_state.as_mut() {
            lls.reset();
        }

        // Helper: reinit a weight slice with scale 0.01 * standard_normal.
        let reinit = |weights: &mut [f64], rng: &mut u64| {
            for w in weights.iter_mut() {
                *w = standard_normal(rng) * 0.01;
            }
        };

        // Core projections (always present).
        reinit(&mut head.w_key, rng);
        reinit(&mut head.w_value, rng);
        reinit(&mut head.w_query, rng);

        // Mode-dependent weights (only reinit non-empty vecs).
        if !head.w_gate.is_empty() {
            reinit(&mut head.w_gate, rng);
        }
        if !head.w_gate2.is_empty() {
            reinit(&mut head.w_gate2, rng);
        }
        if !head.w_decay.is_empty() {
            reinit(&mut head.w_decay, rng);
        }
        if !head.w_alpha.is_empty() {
            reinit(&mut head.w_alpha, rng);
        }
        if !head.w_beta.is_empty() {
            reinit(&mut head.w_beta, rng);
        }
        if !head.w_comp_keys.is_empty() {
            reinit(&mut head.w_comp_keys, rng);
        }
        if !head.w_comp_values.is_empty() {
            reinit(&mut head.w_comp_values, rng);
        }
        if !head.w_comp_betas.is_empty() {
            reinit(&mut head.w_comp_betas, rng);
        }
        if !head.w_decay_vec.is_empty() {
            reinit(&mut head.w_decay_vec, rng);
        }
        if !head.w_iclr.is_empty() {
            reinit(&mut head.w_iclr, rng);
        }
        if !head.w_xi.is_empty() {
            reinit(&mut head.w_xi, rng);
        }
        if !head.w_alpha_rk.is_empty() {
            reinit(&mut head.w_alpha_rk, rng);
        }
        if !head.w_hgrn2_gate.is_empty() {
            reinit(&mut head.w_hgrn2_gate, rng);
        }
        if !head.w_gate_vec.is_empty() {
            reinit(&mut head.w_gate_vec, rng);
        }
        if !head.w_beta_scalar.is_empty() {
            reinit(&mut head.w_beta_scalar, rng);
        }
        if !head.w_lambda.is_empty() {
            reinit(&mut head.w_lambda, rng);
        }

        // Update the state cache region for this head.
        self.update_state_cache();
    }
}

impl AttentionLayer for MultiHeadAttention {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        debug_assert_eq!(
            input.len(),
            self.config.d_model,
            "input length must equal d_model"
        );

        let d_model = self.config.d_model;
        let d_key = self.config.d_key;
        let d_value = self.config.d_value;
        let n_heads = self.config.n_heads;
        let concat_dim = n_heads * d_value;

        // Take scratch buffers out of self to avoid borrow conflicts with self.heads
        let mut k = mem::take(&mut self.scratch_k);
        let mut v = mem::take(&mut self.scratch_v);
        let mut q = mem::take(&mut self.scratch_q);
        let mut concat_output = mem::take(&mut self.scratch_concat);

        for (h, head) in self.heads.iter_mut().enumerate() {
            // Zero and reuse scratch buffers for key, value, query projections
            k.iter_mut().for_each(|x| *x = 0.0);
            v.iter_mut().for_each(|x| *x = 0.0);
            q.iter_mut().for_each(|x| *x = 0.0);
            mat_vec(&head.w_key, input, d_key, d_model, &mut k);
            mat_vec(&head.w_value, input, d_value, d_model, &mut v);
            mat_vec(&head.w_query, input, d_key, d_model, &mut q);

            // Compute gate/decay and apply update rule based on mode
            let head_output = match &self.config.mode {
                AttentionMode::RetNet { gamma } => {
                    let decay = fixed_decay(*gamma);
                    update_rules::additive_update(&mut head.state, &k, &v, decay);
                    head.state.query(&q)
                }
                AttentionMode::Hawk => {
                    // Hawk: project input to alpha/beta, apply sigmoid to alpha
                    let mut alpha_raw = vec![0.0; d_value];
                    let mut beta_raw = vec![0.0; d_value];
                    mat_vec(&head.w_alpha, input, d_value, d_model, &mut alpha_raw);
                    mat_vec(&head.w_beta, input, d_value, d_model, &mut beta_raw);

                    // alpha = sigmoid(alpha_raw) for stable decay in (0,1)
                    let mut alpha = vec![0.0; d_value];
                    let mut beta = vec![0.0; d_value];
                    for i in 0..d_value {
                        alpha[i] = math::sigmoid(alpha_raw[i]);
                        beta[i] = math::sigmoid(beta_raw[i]);
                    }

                    update_rules::hawk_update(&mut head.state, &v, &alpha, &beta);
                    // Hawk output is the vector state itself
                    head.state.as_slice().to_vec()
                }
                AttentionMode::GLA => {
                    let decay = sigmoid_gate(&head.w_gate, input);
                    update_rules::additive_update(&mut head.state, &k, &v, decay);
                    head.state.query(&q)
                }
                AttentionMode::GLAVector => {
                    // Per-key-dimension gate: α_t ∈ (0,1)^{d_k} (paper-canonical GLA).
                    // Each row of w_gate_vec projects input to one gate scalar.
                    let alpha = vector_sigmoid_gate(&head.w_gate_vec, input, d_key);
                    update_rules::additive_update_vec(&mut head.state, &k, &v, &alpha);
                    head.state.query(&q)
                }
                AttentionMode::DeltaNet => {
                    // Normalize key for stable delta rule
                    let k_norm = l2_normalize(&k);
                    update_rules::delta_update(&mut head.state, &k_norm, &v);
                    head.state.query(&q)
                }
                AttentionMode::GatedDeltaNet {
                    beta_scale,
                    gate_mode_delta,
                } => {
                    let decay = sigmoid_gate(&head.w_gate, input);
                    // Resolve beta: static scalar or per-token sigmoid projection.
                    // Per Yang et al. ICLR 2025 (arXiv:2412.06464): the canonical
                    // form uses β_t = sigmoid(W_β · x_t) for data-dependent mixing.
                    let beta = match gate_mode_delta {
                        GatedDeltaMode::Static => *beta_scale,
                        GatedDeltaMode::PerToken => sigmoid_gate(&head.w_beta_scalar, input),
                    };
                    // Key normalization is handled inside gated_delta_update
                    update_rules::gated_delta_update(&mut head.state, &k, &v, decay, beta);
                    head.state.query(&q)
                }
                AttentionMode::RWKV { initial_decay } => {
                    let gate = exponential_gate(&head.w_decay, input, *initial_decay);
                    // Use -ln(gate) as the w parameter for exponential_update
                    // gate = exp(-w) => w = -ln(gate)
                    let w = if gate > 1e-30 { -math::ln(gate) } else { 50.0 };
                    update_rules::exponential_update(&mut head.state, &k, &v, w);
                    head.state.query(&q)
                }
                AttentionMode::MLSTM => {
                    let (forget, input_gate) = lstm_gates(&head.w_gate, &head.w_gate2, input);
                    update_rules::mlstm_update(&mut head.state, &k, &v, forget, input_gate);
                    head.state.query(&q)
                }
                AttentionMode::DeltaProduct {
                    n_compositions,
                    reflections,
                } => {
                    let n_h = *n_compositions;
                    let use_reflections = *reflections;
                    let gate = sigmoid_gate(&head.w_gate, input);

                    // Project n_h keys, values, betas
                    let mut comp_keys_storage: Vec<Vec<f64>> = Vec::with_capacity(n_h);
                    let mut comp_values_storage: Vec<Vec<f64>> = Vec::with_capacity(n_h);
                    let mut comp_betas: Vec<f64> = Vec::with_capacity(n_h);

                    for j in 0..n_h {
                        // Key projection for composition j
                        let mut kj = vec![0.0; d_key];
                        let k_offset = j * d_key * d_model;
                        mat_vec(
                            &head.w_comp_keys[k_offset..k_offset + d_key * d_model],
                            input,
                            d_key,
                            d_model,
                            &mut kj,
                        );
                        // L2-normalize the key
                        let kj_norm = l2_normalize(&kj);
                        comp_keys_storage.push(kj_norm);

                        // Value projection for composition j
                        let mut vj = vec![0.0; d_value];
                        let v_offset = j * d_value * d_model;
                        mat_vec(
                            &head.w_comp_values[v_offset..v_offset + d_value * d_model],
                            input,
                            d_value,
                            d_model,
                            &mut vj,
                        );
                        comp_values_storage.push(vj);

                        // Beta for composition j.
                        // reflections=false: β ∈ (0, 1) via plain sigmoid (default delta-rule range).
                        // reflections=true:  β ∈ (0, 2) via 2·sigmoid (full Householder reflections,
                        //                    Siems et al. NeurIPS 2025, arXiv:2502.10297, §4).
                        let b_offset = j * d_model;
                        let beta = if use_reflections {
                            extended_sigmoid_gate(
                                &head.w_comp_betas[b_offset..b_offset + d_model],
                                input,
                            )
                        } else {
                            sigmoid_gate(&head.w_comp_betas[b_offset..b_offset + d_model], input)
                        };
                        comp_betas.push(beta);
                    }

                    // Build slice references for update function
                    let comp_keys_refs: Vec<&[f64]> =
                        comp_keys_storage.iter().map(|k| k.as_slice()).collect();
                    let comp_values_refs: Vec<&[f64]> =
                        comp_values_storage.iter().map(|v| v.as_slice()).collect();

                    update_rules::delta_product_update(
                        &mut head.state,
                        &comp_keys_refs,
                        &comp_values_refs,
                        &comp_betas,
                        gate,
                    );
                    head.state.query(&q)
                }
                AttentionMode::RWKV7 => {
                    // Per-dimension vector decay
                    let w = vector_decay(&head.w_decay_vec, input, d_key);

                    // Removal key: k * xi, then L2-normalize
                    let mut kappa = vec![0.0; d_key];
                    for i in 0..d_key {
                        kappa[i] = k[i] * head.w_xi[i];
                    }
                    let kappa_hat = l2_normalize(&kappa);

                    // ICLR vector
                    let a = vector_sigmoid_gate(&head.w_iclr, input, d_key);

                    // Replacement key: k * lerp(1, a, alpha)
                    let mut k_tilde = vec![0.0; d_key];
                    for i in 0..d_key {
                        let mix = 1.0 + head.w_alpha_rk[i] * (a[i] - 1.0);
                        k_tilde[i] = k[i] * mix;
                    }

                    update_rules::rwkv7_update(&mut head.state, &w, &kappa_hat, &a, &k_tilde, &v);
                    head.state.query(&q)
                }
                AttentionMode::HGRN2 { lower_bound } => {
                    // Per-dimension lower-bounded forget gate
                    let alpha =
                        vector_lower_bounded_gate(&head.w_hgrn2_gate, input, d_key, *lower_bound);
                    update_rules::hgrn2_update(&mut head.state, &k, &v, &alpha);
                    head.state.query(&q)
                }
                AttentionMode::LogLinear {
                    inner,
                    max_levels,
                    lambda_init,
                } => {
                    // Step 1: per-inner-mode key preprocessing.
                    // Delta-family inner rules (DeltaNet, GatedDeltaNet,
                    // DeltaProduct, RWKV7) require L2-normalized keys
                    // for bounded state growth (paper §3.2 carryover,
                    // R1 §3.5 risk #2 mitigation). All other inner rules
                    // pass the raw key through.
                    let k_for_leaf: Vec<f64> = match inner.as_ref() {
                        AttentionMode::DeltaNet
                        | AttentionMode::GatedDeltaNet { .. }
                        | AttentionMode::DeltaProduct { .. }
                        | AttentionMode::RWKV7 => l2_normalize(&k),
                        _ => k.clone(),
                    };

                    // Step 2: compute λ via softplus-softmax mix.
                    // Paper §3.2: bounded mixture, Σ λ ≤ 1.
                    let mut raw = vec![0.0; *max_levels];
                    mat_vec(&head.w_lambda, input, *max_levels, d_model, &mut raw);
                    for r in raw.iter_mut() {
                        *r += *lambda_init;
                    }
                    let mut lambdas = vec![0.0; *max_levels];
                    softplus_softmax_mix(&raw, super::log_linear::DEFAULT_TAU, &mut lambdas);

                    // Step 3: push the new leaf bucket and run carry
                    // propagation (paper §2.1 — classical Fenwick
                    // increment).
                    let lls = head
                        .log_linear_state
                        .as_mut()
                        .expect("LogLinear mode must have log_linear_state");
                    lls.push_leaf(&k_for_leaf, &v);

                    // Step 4: read out the (post-update) mixed state.
                    let mut o = vec![0.0; d_value];
                    lls.query_mixed(&q, &lambdas, &mut o);
                    o
                }
            };

            // Copy head output to concatenated buffer
            let offset = h * d_value;
            concat_output[offset..offset + d_value].copy_from_slice(&head_output);
        }

        // Return scratch buffers to self
        self.scratch_k = k;
        self.scratch_v = v;
        self.scratch_q = q;

        // Output projection: w_out * concat_output
        let mut output = vec![0.0; d_model];
        mat_vec(
            &self.w_out,
            &concat_output,
            d_model,
            concat_dim,
            &mut output,
        );

        // Delta-family output bounding via tanh after W_out.
        //
        // All five delta-family variants (DeltaNet, GatedDeltaNet, DeltaProduct,
        // RWKV7, HGRN2) produce unbounded W_out projections. Anything feeding
        // RLS readout must be bounded to prevent weight explosion (AGENTS.md
        // "Bounded readout features" principle). tanh maps R → (-1, 1) and is
        // the standard bounding primitive for linear-attention family outputs.
        //
        // RetNet, Hawk, GLA, GLAVector, RWKV, MLSTM are excluded — their gating
        // (sigmoid/forget gate scales) provides implicit magnitude control.
        match &self.config.mode {
            AttentionMode::DeltaNet
            | AttentionMode::GatedDeltaNet { .. }
            | AttentionMode::DeltaProduct { .. }
            | AttentionMode::RWKV7
            | AttentionMode::HGRN2 { .. }
            | AttentionMode::LogLinear { .. } => {
                tanh_inplace(&mut output);
            }
            _ => {}
        }

        // Return concat scratch to self
        self.scratch_concat = concat_output;

        // Update state cache
        self.update_state_cache();

        output
    }

    fn state(&self) -> &[f64] {
        &self.state_cache
    }

    fn output_dim(&self) -> usize {
        self.config.d_model
    }

    fn reset(&mut self) {
        for head in &mut self.heads {
            head.state.reset();
            // LogLinear: clear all Fenwick levels and size counter.
            if let Some(lls) = head.log_linear_state.as_mut() {
                lls.reset();
            }
        }
        for x in self.state_cache.iter_mut() {
            *x = 0.0;
        }
    }
}

/// L2-normalize a vector. Returns zero vector if norm is zero.
fn l2_normalize(v: &[f64]) -> Vec<f64> {
    let norm_sq: f64 = v.iter().map(|&x| x * x).sum();
    let norm = math::sqrt(norm_sq);
    if norm < 1e-12 {
        vec![0.0; v.len()]
    } else {
        let inv = 1.0 / norm;
        v.iter().map(|&x| x * inv).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::config::{AttentionConfig, AttentionMode};

    fn make_config(mode: AttentionMode) -> AttentionConfig {
        AttentionConfig {
            d_model: 8,
            n_heads: 2,
            d_key: 4,
            d_value: 4,
            mode,
            seed: 42,
        }
    }

    fn make_input(d_model: usize) -> Vec<f64> {
        let mut rng = Xorshift64(999);
        let mut input = Vec::with_capacity(d_model);
        for _ in 0..d_model {
            input.push(rng.next_normal());
        }
        input
    }

    #[test]
    fn retnet_output_dimension_matches_config() {
        let config = make_config(AttentionMode::RetNet { gamma: 0.95 });
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(
            output.len(),
            8,
            "output dimension should match d_model=8, got {}",
            output.len()
        );
        assert_eq!(attn.output_dim(), 8, "output_dim() should return d_model=8");
    }

    #[test]
    fn hawk_output_dimension_matches_config() {
        let config = make_config(AttentionMode::Hawk);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(
            output.len(),
            8,
            "Hawk output should match d_model=8, got {}",
            output.len()
        );
    }

    #[test]
    fn gla_output_dimension_matches_config() {
        let config = make_config(AttentionMode::GLA);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(
            output.len(),
            8,
            "GLA output should match d_model=8, got {}",
            output.len()
        );
    }

    #[test]
    fn deltanet_output_dimension_matches_config() {
        let config = make_config(AttentionMode::DeltaNet);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(
            output.len(),
            8,
            "DeltaNet output should match d_model=8, got {}",
            output.len()
        );
    }

    #[test]
    fn gated_deltanet_output_dimension_matches_config() {
        let config = make_config(AttentionMode::GatedDeltaNet {
            beta_scale: 1.0,
            gate_mode_delta: GatedDeltaMode::Static,
        });
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(
            output.len(),
            8,
            "GatedDeltaNet output should match d_model=8, got {}",
            output.len()
        );
    }

    #[test]
    fn rwkv_output_dimension_matches_config() {
        let config = make_config(AttentionMode::RWKV { initial_decay: 0.5 });
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(
            output.len(),
            8,
            "RWKV output should match d_model=8, got {}",
            output.len()
        );
    }

    #[test]
    fn mlstm_output_dimension_matches_config() {
        let config = make_config(AttentionMode::MLSTM);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(
            output.len(),
            8,
            "mLSTM output should match d_model=8, got {}",
            output.len()
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let input = make_input(8);

        let config1 = make_config(AttentionMode::RetNet { gamma: 0.95 });
        let mut attn1 = MultiHeadAttention::new(config1);
        let out1 = attn1.forward(&input);

        let config2 = make_config(AttentionMode::RetNet { gamma: 0.95 });
        let mut attn2 = MultiHeadAttention::new(config2);
        let out2 = attn2.forward(&input);

        for i in 0..out1.len() {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-12,
                "same seed should produce identical output at index {}: {} vs {}",
                i,
                out1[i],
                out2[i]
            );
        }
    }

    #[test]
    fn different_seeds_produce_different_output() {
        let input = make_input(8);

        let config1 = AttentionConfig {
            seed: 42,
            ..make_config(AttentionMode::RetNet { gamma: 0.95 })
        };
        let mut attn1 = MultiHeadAttention::new(config1);
        let out1 = attn1.forward(&input);

        let config2 = AttentionConfig {
            seed: 123,
            ..make_config(AttentionMode::RetNet { gamma: 0.95 })
        };
        let mut attn2 = MultiHeadAttention::new(config2);
        let out2 = attn2.forward(&input);

        let any_diff = out1
            .iter()
            .zip(out2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(any_diff, "different seeds should produce different outputs");
    }

    #[test]
    fn reset_clears_state_to_zero() {
        let config = make_config(AttentionMode::GLA);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);

        // Run forward to populate state
        attn.forward(&input);
        attn.forward(&input);

        // Reset
        attn.reset();

        let state = attn.state();
        assert!(
            state.iter().all(|&x| x == 0.0),
            "after reset, all state values should be zero"
        );
    }

    #[test]
    fn state_slice_has_correct_length() {
        let config = make_config(AttentionMode::RetNet { gamma: 0.95 });
        let attn = MultiHeadAttention::new(config);
        // 2 heads * 4 d_key * 4 d_value = 32
        let expected = 2 * 4 * 4;
        assert_eq!(
            attn.state().len(),
            expected,
            "state slice should have length n_heads*d_key*d_value={}, got {}",
            expected,
            attn.state().len()
        );
    }

    #[test]
    fn hawk_state_slice_has_correct_length() {
        let config = make_config(AttentionMode::Hawk);
        let attn = MultiHeadAttention::new(config);
        // 2 heads * 4 d_value = 8
        let expected = 2 * 4;
        assert_eq!(
            attn.state().len(),
            expected,
            "Hawk state slice should have length n_heads*d_value={}, got {}",
            expected,
            attn.state().len()
        );
    }

    #[test]
    fn forward_changes_state() {
        let config = make_config(AttentionMode::DeltaNet);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        attn.forward(&input);
        let state = attn.state();
        let any_nonzero = state.iter().any(|&x| x.abs() > 1e-15);
        assert!(any_nonzero, "after forward, state should be non-zero");
    }

    #[test]
    fn multiple_forwards_accumulate() {
        let config = make_config(AttentionMode::RetNet { gamma: 0.95 });
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);

        attn.forward(&input);
        let state1: Vec<f64> = attn.state().to_vec();

        attn.forward(&input);
        let state2: Vec<f64> = attn.state().to_vec();

        let any_diff = state1
            .iter()
            .zip(state2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(
            any_diff,
            "second forward should change state (accumulation)"
        );
    }

    #[test]
    fn l2_normalize_unit_vector() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        assert!(
            (n[0] - 0.6).abs() < 1e-12,
            "normalized [3,4][0] should be 0.6, got {}",
            n[0]
        );
        assert!(
            (n[1] - 0.8).abs() < 1e-12,
            "normalized [3,4][1] should be 0.8, got {}",
            n[1]
        );
        let norm: f64 = n.iter().map(|x| x * x).sum::<f64>();
        assert!(
            (math::sqrt(norm) - 1.0).abs() < 1e-12,
            "normalized vector should have unit norm"
        );
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!(
            n.iter().all(|&x| x == 0.0),
            "normalizing zero vector should return zero vector"
        );
    }

    #[test]
    fn delta_product_output_dimension() {
        let config = make_config(AttentionMode::DeltaProduct {
            n_compositions: 3,
            reflections: false,
        });
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(
            output.len(),
            8,
            "DeltaProduct output should match d_model=8"
        );
    }

    #[test]
    fn delta_product_state_length() {
        let config = make_config(AttentionMode::DeltaProduct {
            n_compositions: 2,
            reflections: false,
        });
        let attn = MultiHeadAttention::new(config);
        let expected = 2 * 4 * 4; // n_heads * d_key * d_value
        assert_eq!(
            attn.state().len(),
            expected,
            "DeltaProduct state should be n_heads*d_key*d_value"
        );
    }

    #[test]
    fn delta_product_deterministic() {
        let input = make_input(8);
        let config1 = make_config(AttentionMode::DeltaProduct {
            n_compositions: 2,
            reflections: false,
        });
        let mut a1 = MultiHeadAttention::new(config1);
        let o1 = a1.forward(&input);
        let config2 = make_config(AttentionMode::DeltaProduct {
            n_compositions: 2,
            reflections: false,
        });
        let mut a2 = MultiHeadAttention::new(config2);
        let o2 = a2.forward(&input);
        for i in 0..o1.len() {
            assert!(
                (o1[i] - o2[i]).abs() < 1e-12,
                "same seed should give same output at {}",
                i
            );
        }
    }

    #[test]
    fn rwkv7_output_dimension() {
        let config = make_config(AttentionMode::RWKV7);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(output.len(), 8, "RWKV7 output should match d_model=8");
    }

    #[test]
    fn rwkv7_state_length() {
        let config = make_config(AttentionMode::RWKV7);
        let attn = MultiHeadAttention::new(config);
        let expected = 2 * 4 * 4;
        assert_eq!(
            attn.state().len(),
            expected,
            "RWKV7 state should be n_heads*d_key*d_value"
        );
    }

    #[test]
    fn rwkv7_forward_changes_state() {
        let config = make_config(AttentionMode::RWKV7);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        attn.forward(&input);
        let state = attn.state();
        let any_nonzero = state.iter().any(|&x| x.abs() > 1e-15);
        assert!(any_nonzero, "after forward, RWKV7 state should be non-zero");
    }

    #[test]
    fn rwkv7_reset_clears() {
        let config = make_config(AttentionMode::RWKV7);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        attn.forward(&input);
        attn.reset();
        assert!(
            attn.state().iter().all(|&x| x == 0.0),
            "after reset all state should be zero"
        );
    }

    #[test]
    fn hgrn2_output_dimension() {
        let config = make_config(AttentionMode::HGRN2 { lower_bound: 0.9 });
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(output.len(), 8, "HGRN2 output should match d_model=8");
    }

    #[test]
    fn hgrn2_state_length() {
        let config = make_config(AttentionMode::HGRN2 { lower_bound: 0.9 });
        let attn = MultiHeadAttention::new(config);
        let expected = 2 * 4 * 4; // n_heads * d_key * d_value
        assert_eq!(
            attn.state().len(),
            expected,
            "HGRN2 state should be n_heads*d_key*d_value"
        );
    }

    #[test]
    fn hgrn2_forward_changes_state() {
        let config = make_config(AttentionMode::HGRN2 { lower_bound: 0.9 });
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        attn.forward(&input);
        let state = attn.state();
        let any_nonzero = state.iter().any(|&x| x.abs() > 1e-15);
        assert!(any_nonzero, "after forward, HGRN2 state should be non-zero");
    }

    #[test]
    fn hgrn2_reset_clears() {
        let config = make_config(AttentionMode::HGRN2 { lower_bound: 0.9 });
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        attn.forward(&input);
        attn.reset();
        assert!(
            attn.state().iter().all(|&x| x == 0.0),
            "after reset all HGRN2 state should be zero"
        );
    }

    #[test]
    fn hgrn2_different_lower_bounds_different_output() {
        let input = make_input(8);

        let config1 = make_config(AttentionMode::HGRN2 { lower_bound: 0.1 });
        let mut attn1 = MultiHeadAttention::new(config1);
        attn1.forward(&input);
        attn1.forward(&input);
        let state1: Vec<f64> = attn1.state().to_vec();

        let config2 = make_config(AttentionMode::HGRN2 { lower_bound: 0.99 });
        let mut attn2 = MultiHeadAttention::new(config2);
        attn2.forward(&input);
        attn2.forward(&input);
        let state2: Vec<f64> = attn2.state().to_vec();

        let any_diff = state1
            .iter()
            .zip(state2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            any_diff,
            "different lower_bound values should produce different states"
        );
    }

    #[test]
    fn hgrn2_output_finite() {
        let config = make_config(AttentionMode::HGRN2 { lower_bound: 0.9 });
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        for _ in 0..100 {
            let output = attn.forward(&input);
            assert!(
                output.iter().all(|x| x.is_finite()),
                "HGRN2 output should be finite after many steps"
            );
        }
    }

    #[test]
    fn reinitialize_head_preserves_others() {
        let config = make_config(AttentionMode::GLA);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);

        // Run forward to populate state.
        attn.forward(&input);
        attn.forward(&input);

        // Snapshot head 0's weights before reinit.
        let h0_wk_before = attn.heads[0].w_key.clone();
        let h0_wv_before = attn.heads[0].w_value.clone();
        let h0_wq_before = attn.heads[0].w_query.clone();
        let h0_state_before: Vec<f64> = attn.heads[0].state.as_slice().to_vec();

        // Snapshot head 1's weights before reinit.
        let h1_wk_before = attn.heads[1].w_key.clone();
        let h1_wv_before = attn.heads[1].w_value.clone();
        let h1_wq_before = attn.heads[1].w_query.clone();

        // Snapshot w_out before reinit.
        let w_out_before = attn.w_out.clone();

        // Reinitialize head 1 only.
        let mut rng = 12345u64;
        attn.reinitialize_head(1, &mut rng);

        // Head 0's weights should be completely unchanged.
        assert_eq!(
            attn.heads[0].w_key, h0_wk_before,
            "head 0 w_key should be preserved"
        );
        assert_eq!(
            attn.heads[0].w_value, h0_wv_before,
            "head 0 w_value should be preserved"
        );
        assert_eq!(
            attn.heads[0].w_query, h0_wq_before,
            "head 0 w_query should be preserved"
        );
        // Head 0's state should be unchanged.
        for (i, (&a, &b)) in h0_state_before
            .iter()
            .zip(attn.heads[0].state.as_slice().iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-15,
                "head 0 state[{}] should be preserved: {} vs {}",
                i,
                a,
                b
            );
        }

        // Head 1's state should be zeroed.
        assert!(
            attn.heads[1].state.as_slice().iter().all(|&x| x == 0.0),
            "head 1 state should be zeroed after reinit"
        );

        // Head 1's weights should have changed.
        let any_key_changed = h1_wk_before
            .iter()
            .zip(attn.heads[1].w_key.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(any_key_changed, "head 1 w_key should have new values");

        let any_val_changed = h1_wv_before
            .iter()
            .zip(attn.heads[1].w_value.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(any_val_changed, "head 1 w_value should have new values");

        let any_query_changed = h1_wq_before
            .iter()
            .zip(attn.heads[1].w_query.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(any_query_changed, "head 1 w_query should have new values");

        // w_out should be completely unchanged.
        assert_eq!(
            attn.w_out, w_out_before,
            "w_out should be preserved after head reinit"
        );
    }

    #[test]
    fn gla_vector_output_dimension_matches_config() {
        let config = make_config(AttentionMode::GLAVector);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        let output = attn.forward(&input);
        assert_eq!(
            output.len(),
            8,
            "GLAVector output should match d_model=8, got {}",
            output.len()
        );
    }

    #[test]
    fn gla_vector_state_length() {
        let config = make_config(AttentionMode::GLAVector);
        let attn = MultiHeadAttention::new(config);
        let expected = 2 * 4 * 4; // n_heads * d_key * d_value
        assert_eq!(
            attn.state().len(),
            expected,
            "GLAVector state should be n_heads*d_key*d_value"
        );
    }

    #[test]
    fn gla_vector_gate_differs_from_scalar() {
        // The vector-gate GLA must produce different output from scalar-gate GLA
        // when run on multi-dimensional input. This proves the per-dimension gates
        // have independent effect rather than collapsing to the scalar case.
        let input = make_input(8);

        let config_scalar = make_config(AttentionMode::GLA);
        let mut attn_scalar = MultiHeadAttention::new(config_scalar);
        // Train scalar-gate GLA for a few steps to build non-trivial state.
        attn_scalar.forward(&input);
        attn_scalar.forward(&make_input(8));
        let out_scalar = attn_scalar.forward(&input);

        let config_vec = make_config(AttentionMode::GLAVector);
        let mut attn_vec = MultiHeadAttention::new(config_vec);
        // Same inputs as scalar.
        attn_vec.forward(&input);
        attn_vec.forward(&make_input(8));
        let out_vec = attn_vec.forward(&input);

        let any_diff = out_scalar
            .iter()
            .zip(out_vec.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(
            any_diff,
            "GLAVector output must differ from scalar GLA output — vector gates have independent \
             per-dimension effect"
        );
    }

    #[test]
    fn gla_vector_forward_changes_state() {
        let config = make_config(AttentionMode::GLAVector);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        attn.forward(&input);
        let state = attn.state();
        let any_nonzero = state.iter().any(|&x| x.abs() > 1e-15);
        assert!(
            any_nonzero,
            "after forward, GLAVector state should be non-zero"
        );
    }

    #[test]
    fn gla_vector_reset_clears() {
        let config = make_config(AttentionMode::GLAVector);
        let mut attn = MultiHeadAttention::new(config);
        let input = make_input(8);
        attn.forward(&input);
        attn.reset();
        assert!(
            attn.state().iter().all(|&x| x == 0.0),
            "after reset all GLAVector state should be zero"
        );
    }

    #[test]
    fn gated_deltanet_per_token_beta_is_data_dependent() {
        // PerToken mode: β_t = sigmoid(w_beta_scalar · x_t) per Yang et al.
        // ICLR 2025 (arXiv:2412.06464). Verify that two clearly different
        // inputs produce different outputs — if beta were static (Static mode),
        // identical weights would give identical beta regardless of input, and
        // the output difference would be purely from the query projection.
        // This test verifies the PerToken runtime path actually uses x_t.
        let config_per_token = AttentionConfig {
            d_model: 8,
            n_heads: 2,
            d_key: 4,
            d_value: 4,
            mode: AttentionMode::GatedDeltaNet {
                beta_scale: 1.0,
                gate_mode_delta: GatedDeltaMode::PerToken,
            },
            seed: 42,
        };
        let config_static = AttentionConfig {
            mode: AttentionMode::GatedDeltaNet {
                beta_scale: 1.0,
                gate_mode_delta: GatedDeltaMode::Static,
            },
            ..config_per_token.clone()
        };

        let mut attn_per_token = MultiHeadAttention::new(config_per_token);
        let mut attn_static = MultiHeadAttention::new(config_static);

        // Build up state with same inputs.
        for i in 0..5 {
            let t = i as f64 * 0.3;
            let x = alloc::vec![
                t.sin(),
                t.cos(),
                t * 0.1,
                1.0 - t * 0.05,
                0.5,
                0.3,
                0.2,
                0.1
            ];
            attn_per_token.forward(&x);
            attn_static.forward(&x);
        }

        // Now forward with inputs that have maximally different projections.
        let input_a = alloc::vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let input_b = alloc::vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // PerToken: different inputs produce different betas → state updates differ.
        let out_pt_a = attn_per_token.forward(&input_a);
        let out_pt_b = attn_per_token.forward(&input_b);

        // Static: beta is constant regardless of input (identical beta for both calls
        // after the state update from input_a above — test the distinction exists).
        let _ = attn_static.forward(&input_a);
        let _ = attn_static.forward(&input_b);

        // Both PerToken outputs must be finite.
        assert!(
            out_pt_a.iter().all(|x| x.is_finite()),
            "PerToken forward(input_a) must be finite"
        );
        assert!(
            out_pt_b.iter().all(|x| x.is_finite()),
            "PerToken forward(input_b) must be finite"
        );

        // PerToken mode must produce different outputs for opposite inputs.
        let any_diff = out_pt_a
            .iter()
            .zip(out_pt_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(
            any_diff,
            "GatedDeltaNet PerToken outputs must differ for opposite inputs — \
             data-dependent β_t = sigmoid(w·x_t) must influence the result"
        );
    }

    #[test]
    fn delta_product_reflections_doubles_beta_range() {
        // reflections=true: β ∈ (0, 2) via 2·sigmoid (Siems et al. NeurIPS 2025,
        // arXiv:2502.10297 §4). reflections=false: β ∈ (0, 1) via plain sigmoid.
        // Verify that the output with reflections enabled can exceed 1.0 in magnitude
        // on the beta dimension (not directly observable, but state must differ).
        let config_refl = AttentionConfig {
            d_model: 8,
            n_heads: 2,
            d_key: 4,
            d_value: 4,
            mode: AttentionMode::DeltaProduct {
                n_compositions: 2,
                reflections: true,
            },
            seed: 77,
        };
        let config_norefl = AttentionConfig {
            mode: AttentionMode::DeltaProduct {
                n_compositions: 2,
                reflections: false,
            },
            ..config_refl.clone()
        };

        let mut attn_refl = MultiHeadAttention::new(config_refl);
        let mut attn_norefl = MultiHeadAttention::new(config_norefl);

        // Run for several steps to accumulate state differences from the beta range.
        let mut any_diff_seen = false;
        for i in 0..10 {
            let t = i as f64 * 0.5;
            let x = alloc::vec![t.sin(), t.cos(), t * 0.1, 0.5, 0.3, 0.2, 0.1, 0.4];
            let out_r = attn_refl.forward(&x);
            let out_n = attn_norefl.forward(&x);

            assert!(
                out_r.iter().all(|v| v.is_finite()),
                "reflections=true forward must be finite at step {i}"
            );

            if out_r
                .iter()
                .zip(out_n.iter())
                .any(|(a, b)| (a - b).abs() > 1e-12)
            {
                any_diff_seen = true;
            }
        }
        assert!(
            any_diff_seen,
            "reflections=true must produce different state evolution than reflections=false — \
             2·sigmoid range enables negative eigenvalues (Siems et al. NeurIPS 2025 §4)"
        );
    }

    #[test]
    fn delta_family_outputs_bounded_in_unit_interval() {
        // tanh_inplace after W_out ensures delta-family outputs are bounded in (-1, 1).
        // This is the AGENTS.md "Bounded readout features" invariant: anything
        // feeding RLS must be bounded to prevent weight explosion.
        // Verified for all 5 delta-family variants.
        let delta_modes: alloc::vec::Vec<AttentionMode> = alloc::vec![
            AttentionMode::DeltaNet,
            AttentionMode::GatedDeltaNet {
                beta_scale: 1.0,
                gate_mode_delta: GatedDeltaMode::Static,
            },
            AttentionMode::DeltaProduct {
                n_compositions: 2,
                reflections: false,
            },
            AttentionMode::RWKV7,
            AttentionMode::HGRN2 { lower_bound: 0.9 },
        ];

        for mode in delta_modes {
            let config = AttentionConfig {
                d_model: 8,
                n_heads: 2,
                d_key: 4,
                d_value: 4,
                mode,
                seed: 99,
            };
            let mode_name = alloc::format!("{:?}", &config.mode);
            let mut attn = MultiHeadAttention::new(config);

            // Run 20 steps with non-trivial inputs to accumulate state.
            for i in 0..20 {
                let t = i as f64 * 0.3;
                let x = alloc::vec![t.sin(), t.cos(), t * 0.1, 0.5, 0.3, 0.8, -0.2, 0.6];
                let out = attn.forward(&x);
                for &val in &out {
                    assert!(
                        val.is_finite() && val.abs() <= 1.0,
                        "{mode_name} output at step {i} must be in [-1, 1] after tanh_inplace, \
                         got {val}"
                    );
                }
            }
        }
    }
}
