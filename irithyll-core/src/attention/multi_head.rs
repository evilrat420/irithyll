//! Multi-head streaming linear attention.
//!
//! [`MultiHeadAttention`] composes multiple attention head instances with
//! an output projection, implementing the [`AttentionLayer`] trait. Each head
//! independently maintains state and computes its update rule according to
//! the configured [`AttentionMode`].

use alloc::vec;
use alloc::vec::Vec;

use super::config::{AttentionConfig, AttentionMode};
use super::gating::{
    exponential_gate, fixed_decay, init_weights, lstm_gates, mat_vec, sigmoid_gate, Xorshift64,
};
use super::state::AttentionState;
use super::update_rules;
use super::AttentionLayer;
use crate::math;

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

            let w_key = init_weights(&mut rng, d_key * d_model);
            let w_value = init_weights(&mut rng, d_value * d_model);
            let w_query = init_weights(&mut rng, d_key * d_model);

            // Mode-dependent gate weights
            let w_gate = match &config.mode {
                AttentionMode::GLA | AttentionMode::GatedDeltaNet | AttentionMode::MLSTM => {
                    init_weights(&mut rng, d_model)
                }
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
            });
        }

        let concat_dim = n_heads * d_value;
        let w_out = init_weights(&mut rng, d_model * concat_dim);

        // Compute total state size for cache
        let state_size = match &config.mode {
            AttentionMode::Hawk => n_heads * d_value,
            _ => n_heads * d_key * d_value,
        };

        Self {
            config,
            heads,
            w_out,
            state_cache: vec![0.0; state_size],
        }
    }

    /// Update the flat state cache from all head states.
    fn update_state_cache(&mut self) {
        let mut offset = 0;
        for head in &self.heads {
            let slice = head.state.as_slice();
            let len = slice.len();
            self.state_cache[offset..offset + len].copy_from_slice(slice);
            offset += len;
        }
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

        let mut concat_output = vec![0.0; concat_dim];

        for (h, head) in self.heads.iter_mut().enumerate() {
            // Project input to key, value, query
            let mut k = vec![0.0; d_key];
            let mut v = vec![0.0; d_value];
            let mut q = vec![0.0; d_key];
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
                AttentionMode::DeltaNet => {
                    // Normalize key for stable delta rule
                    let k_norm = l2_normalize(&k);
                    update_rules::delta_update(&mut head.state, &k_norm, &v);
                    head.state.query(&q)
                }
                AttentionMode::GatedDeltaNet => {
                    let decay = sigmoid_gate(&head.w_gate, input);
                    let k_norm = l2_normalize(&k);
                    update_rules::gated_delta_update(&mut head.state, &k_norm, &v, decay);
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
            };

            // Copy head output to concatenated buffer
            let offset = h * d_value;
            concat_output[offset..offset + d_value].copy_from_slice(&head_output);
        }

        // Output projection: w_out * concat_output
        let mut output = vec![0.0; d_model];
        mat_vec(
            &self.w_out,
            &concat_output,
            d_model,
            concat_dim,
            &mut output,
        );

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
        let config = make_config(AttentionMode::GatedDeltaNet);
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
}
