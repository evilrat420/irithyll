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
//! attention mechanisms (RetNet, Hawk, GLA, DeltaNet, GatedDeltaNet, RWKV, mLSTM).
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
    AttentionConfig, AttentionLayer, AttentionMode, MultiHeadAttention,
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
            .mode(AttentionMode::GatedDeltaNet)
            .build()
            .expect("delta_net() factory: invalid parameters"),
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
