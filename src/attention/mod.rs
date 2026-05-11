//! Streaming linear attention models.
//!
//! This module provides [`StreamingAttentionModel`], a streaming machine learning
//! model that uses multi-head linear attention as its temporal feature extractor,
//! feeding into a Recursive Least Squares (RLS) readout layer. It integrates
//! with irithyll's [`StreamingLearner`](crate::learner::StreamingLearner) trait
//! and [`StreamingPreprocessor`](crate::pipeline::StreamingPreprocessor) trait.
//!
//! # Architecture
//!
//! ```text
//! input features ──→ [MultiHeadAttention] ──→ temporal features ──→ [RLS] ──→ prediction
//!   (d_model)            (recurrent state)       (d_model)          (1)
//! ```
//!
//! The attention layer processes each feature vector as a timestep, maintaining
//! per-head recurrent state that captures temporal dependencies via linear
//! attention mechanisms (RetNet, Hawk, GLA, DeltaNet, GatedDeltaNet, RWKV, mLSTM,
//! DeltaProduct, RWKV7, HGRN2).
//! The RLS readout learns a linear mapping from the attention output to the target.
//!
//! # Components
//!
//! - [`StreamingAttentionModel`] -- full model implementing `StreamingLearner`
//! - [`AttentionPreprocessor`] -- attention-only preprocessor implementing `StreamingPreprocessor`
//! - [`StreamingAttentionConfig`] / [`StreamingAttentionConfigBuilder`] -- validated configuration
//!
//! # Example
//!
//! ```ignore
//! use irithyll::attention::{StreamingAttentionModel, StreamingAttentionConfig, AttentionMode};
//! use irithyll::learner::StreamingLearner;
//!
//! let config = StreamingAttentionConfig::builder()
//!     .d_model(8)
//!     .n_heads(2)
//!     .mode(AttentionMode::GLA)
//!     .build()
//!     .unwrap();
//!
//! let mut model = StreamingAttentionModel::new(config);
//! model.train(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 5.0);
//! let pred = model.predict(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//! assert!(pred.is_finite());
//! ```

pub mod attention_config;
pub mod attention_preprocessor;
pub mod streaming_attention;

pub use attention_config::{StreamingAttentionConfig, StreamingAttentionConfigBuilder};
pub use attention_preprocessor::AttentionPreprocessor;
pub use streaming_attention::StreamingAttentionModel;

// Re-export core types
pub use irithyll_core::attention::{
    default_lambda_init, AttentionConfig, AttentionLayer, AttentionMode, GateMode, GatedDeltaMode,
    LogLinearAttention, LogLinearState, MultiHeadAttention, DEFAULT_MAX_LEVELS, DEFAULT_TAU,
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Create a Gated Linear Attention model (SOTA streaming attention).
///
/// GLA uses data-dependent gating for adaptive forgetting, providing strong
/// performance across a wide range of sequence modeling tasks.
///
/// ```ignore
/// use irithyll::attention::gla;
/// use irithyll::learner::StreamingLearner;
///
/// let mut model = gla(8, 2);
/// model.train(&[1.0; 8], 0.5);
/// ```
pub fn gla(d_model: usize, n_heads: usize) -> StreamingAttentionModel {
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .n_heads(n_heads)
            .mode(AttentionMode::GLA)
            .build()
            .expect("gla() factory: invalid parameters"),
    )
}

/// Create a Gated DeltaNet model (NVIDIA, strongest retrieval).
///
/// GatedDeltaNet combines delta rule updates with gating for the strongest
/// associative recall performance among linear attention variants.
///
/// ```ignore
/// use irithyll::attention::delta_net;
/// use irithyll::learner::StreamingLearner;
///
/// let mut model = delta_net(8, 2);
/// model.train(&[1.0; 8], 0.5);
/// ```
pub fn delta_net(d_model: usize, n_heads: usize) -> StreamingAttentionModel {
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .n_heads(n_heads)
            .mode(AttentionMode::GatedDeltaNet {
                beta_scale: 1.0,
                gate_mode_delta: GatedDeltaMode::Static,
            })
            .build()
            .expect("delta_net() factory: invalid parameters"),
    )
}

/// Create a DeltaProduct model (strongest tunable linear attention).
///
/// DeltaProduct applies `n_compositions` sequential delta rule steps per token,
/// with spectrally bounded Householder transitions. Higher `n_compositions`
/// increases expressivity at the cost of per-token compute.
///
/// ```ignore
/// use irithyll::attention::delta_product;
/// use irithyll::learner::StreamingLearner;
///
/// let mut model = delta_product(8, 2, 3);
/// model.train(&[1.0; 8], 0.5);
/// ```
pub fn delta_product(
    d_model: usize,
    n_heads: usize,
    n_compositions: usize,
) -> StreamingAttentionModel {
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .n_heads(n_heads)
            .mode(AttentionMode::DeltaProduct {
                n_compositions,
                reflections: false,
            })
            .build()
            .expect("delta_product() factory: invalid parameters"),
    )
}

/// Create an RWKV-7 model (vector-gated delta rule with DPLR transitions).
///
/// RWKV-7 uses per-dimension vector decay, vector in-context learning rate,
/// and decoupled removal/replacement keys for state-of-the-art in-context
/// learning performance.
///
/// ```ignore
/// use irithyll::attention::rwkv7;
/// use irithyll::learner::StreamingLearner;
///
/// let mut model = rwkv7(8, 2);
/// model.train(&[1.0; 8], 0.5);
/// ```
pub fn rwkv7(d_model: usize, n_heads: usize) -> StreamingAttentionModel {
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .n_heads(n_heads)
            .mode(AttentionMode::RWKV7)
            .build()
            .expect("rwkv7() factory: invalid parameters"),
    )
}

/// Create an HGRN2 model (lower-bounded gated linear RNN with state expansion).
///
/// HGRN2 uses per-dimension forget gates with a lower bound ensuring minimum
/// memory retention. The outer-product state update provides expressivity
/// comparable to GLA, while the lower bound prevents catastrophic forgetting.
///
/// ```ignore
/// use irithyll::attention::hgrn2;
/// use irithyll::learner::StreamingLearner;
///
/// let mut model = hgrn2(8, 2, 0.9);
/// model.train(&[1.0; 8], 0.5);
/// ```
pub fn hgrn2(d_model: usize, n_heads: usize, lower_bound: f64) -> StreamingAttentionModel {
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .n_heads(n_heads)
            .mode(AttentionMode::HGRN2 { lower_bound })
            .build()
            .expect("hgrn2() factory: invalid parameters"),
    )
}

/// Create a Hawk model (lightest, vector state).
///
/// Hawk uses a single-head vector state (no KV matrices), making it the
/// most memory-efficient linear attention variant. Best for resource-constrained
/// environments.
///
/// ```ignore
/// use irithyll::attention::hawk;
/// use irithyll::learner::StreamingLearner;
///
/// let mut model = hawk(8);
/// model.train(&[1.0; 8], 0.5);
/// ```
pub fn hawk(d_model: usize) -> StreamingAttentionModel {
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .n_heads(1)
            .mode(AttentionMode::Hawk)
            .build()
            .expect("hawk() factory: invalid parameters"),
    )
}

/// Create a RetNet model (simplest, fixed decay).
///
/// RetNet uses exponential decay with a fixed gamma parameter. Simplest
/// linear attention variant -- good baseline with predictable behavior.
///
/// ```ignore
/// use irithyll::attention::ret_net;
/// use irithyll::learner::StreamingLearner;
///
/// let mut model = ret_net(8, 0.99);
/// model.train(&[1.0; 8], 0.5);
/// ```
pub fn ret_net(d_model: usize, gamma: f64) -> StreamingAttentionModel {
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .n_heads(1)
            .mode(AttentionMode::RetNet { gamma })
            .build()
            .expect("ret_net() factory: invalid parameters"),
    )
}

/// Create a generic streaming attention model with any mode.
///
/// For full control over all hyperparameters, use
/// [`StreamingAttentionConfig::builder()`] directly.
///
/// ```ignore
/// use irithyll::attention::{streaming_attention, AttentionMode};
/// use irithyll::learner::StreamingLearner;
///
/// let mut model = streaming_attention(8, AttentionMode::MLSTM);
/// model.train(&[1.0; 8], 0.5);
/// ```
pub fn streaming_attention(d_model: usize, mode: AttentionMode) -> StreamingAttentionModel {
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .mode(mode)
            .build()
            .expect("streaming_attention() factory: invalid parameters"),
    )
}

/// Create a Log-Linear Attention model (Han Guo et al., ICLR 2026).
///
/// Wraps any inner linear-attention rule with an O(log T) hierarchical
/// Fenwick state. The headline novelty of irithyll v10: bridges
/// linear-attention efficiency and softmax expressivity. State memory
/// is `max_levels * d_k * d_v * n_heads` per layer, padded to a
/// constant shape regardless of stream length (paper §3.4 stability
/// choice).
///
/// # Arguments
///
/// - `d_model` — model dimension.
/// - `n_heads` — number of attention heads.
/// - `inner` — inner linear-attention mode wrapped per token. Must NOT
///   itself be `AttentionMode::LogLinear`. Recommended: `GatedDeltaNet`
///   for strongest associative recall, `GLA` for stability.
/// - `max_levels` — Fenwick depth cap; 32 covers streams up to ~4 G
///   tokens.
///
/// # Paper reference
///
/// Han Guo, Songlin Yang, Tarushii Goel, Eric P. Xing, Tri Dao, Yoon
/// Kim. *Log-Linear Attention*. ICLR 2026. arXiv:2506.04761.
///
/// ```ignore
/// use irithyll::attention::{log_linear, AttentionMode};
/// use irithyll::learner::StreamingLearner;
///
/// let mut model = log_linear(
///     8,
///     2,
///     AttentionMode::GatedDeltaNet {
///         beta_scale: 1.0,
///         gate_mode_delta: irithyll::attention::GatedDeltaMode::Static,
///     },
///     32,
/// );
/// model.train(&[1.0; 8], 0.5);
/// ```
pub fn log_linear(
    d_model: usize,
    n_heads: usize,
    inner: AttentionMode,
    max_levels: usize,
) -> StreamingAttentionModel {
    let lambda_init = default_lambda_init(max_levels);
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .n_heads(n_heads)
            .mode(AttentionMode::LogLinear {
                inner: Box::new(inner),
                max_levels,
                lambda_init,
            })
            .build()
            .expect("log_linear() factory: invalid parameters"),
    )
}
