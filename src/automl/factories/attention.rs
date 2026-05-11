//! Attention factory: Attention, DeltaProduct, RWKV7.

use crate::automl::space::{categorical, int_range, linear_range, ParamMap, SearchSpace};
use irithyll_core::attention::AttentionMode;
use irithyll_core::learner::StreamingLearner;

use super::{Factory, FactoryError};

/// Build the canonical linear-attention search space.
///
/// **Source for ranges:** `n_heads` ∈ {1, 2, 4, 8} matches the typical
/// transformer head sweep (Vaswani et al. 2017 §3.2). Forgetting factor
/// log-half-life range matches Cont (2001). Warmup is the canonical
/// streaming-attention warmup window (Yang et al. 2024 GLA).
fn linear_attention_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("n_heads", categorical(&[1u32, 2, 4, 8]))
        .param("forgetting_factor", linear_range(0.95, 0.9999))
        .param("warmup", int_range(5, 50))
        .build()
        .expect("linear_attention_search_space: builder produces a valid space by construction")
}

fn delta_product_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("n_heads", int_range(1, 8))
        .param("n_compositions", int_range(1, 4))
        .param("forgetting_factor", linear_range(0.95, 0.9999))
        .param("warmup", int_range(5, 50))
        .build()
        .expect("delta_product_search_space: builder produces a valid space by construction")
}

fn rwkv7_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("n_heads", int_range(1, 8))
        .param("forgetting_factor", linear_range(0.95, 0.9999))
        .param("warmup", int_range(5, 50))
        .build()
        .expect("rwkv7_search_space: builder produces a valid space by construction")
}

impl Factory {
    /// Create a factory for streaming linear attention (GLA mode).
    pub fn attention(d_model: usize) -> Self {
        Self {
            algorithm: super::Algorithm::Attention,
            n_features: d_model,
            space: linear_attention_search_space(),
            warmup: 10,
            complexity: 8000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for DeltaProduct attention (product of Householder delta rules).
    pub fn delta_product(d_model: usize) -> Self {
        Self {
            algorithm: super::Algorithm::DeltaProduct,
            n_features: d_model,
            space: delta_product_search_space(),
            warmup: 10,
            complexity: 8000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for RWKV-7 attention (vector-gated delta rule with DPLR).
    pub fn rwkv7(d_model: usize) -> Self {
        Self {
            algorithm: super::Algorithm::Rwkv7,
            n_features: d_model,
            space: rwkv7_search_space(),
            warmup: 10,
            complexity: 5000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    pub(crate) fn create_attention(
        &self,
        params: &ParamMap,
    ) -> Result<Box<dyn StreamingLearner>, FactoryError> {
        use crate::attention::{StreamingAttentionConfig, StreamingAttentionModel};

        match self.algorithm {
            super::Algorithm::Attention => {
                // n_heads is a categorical of {1, 2, 4, 8}.
                let n_heads_str = params.category("n_heads")?;
                let n_heads: usize =
                    n_heads_str
                        .as_str()
                        .parse()
                        .map_err(|_| FactoryError::IncompatibleArm {
                            reason: format!(
                            "attention: n_heads category '{n_heads_str}' is not a valid integer"
                        ),
                        })?;
                let forgetting_factor = params.float("forgetting_factor")?;
                let warmup = params.usize("warmup")?;

                // n_heads must divide d_model — enforce as IncompatibleArm.
                let d_model = self.n_features;
                if d_model > 0 && n_heads > 0 && d_model % n_heads != 0 {
                    return Err(FactoryError::IncompatibleArm {
                        reason: format!("n_heads={} does not divide d_model={}", n_heads, d_model),
                    });
                }

                let attn_config = StreamingAttentionConfig::builder()
                    .d_model(d_model)
                    .n_heads(n_heads)
                    .mode(AttentionMode::GLA)
                    .forgetting_factor(forgetting_factor)
                    .warmup(warmup)
                    .build()?;

                Ok(Box::new(StreamingAttentionModel::new(attn_config)))
            }
            super::Algorithm::DeltaProduct => {
                let n_heads = params.usize("n_heads")?.max(1);
                let n_compositions = params.usize("n_compositions")?.max(1);
                let forgetting_factor = params.float("forgetting_factor")?;
                let warmup = params.usize("warmup")?;

                // n_heads must divide d_model — enforce as IncompatibleArm.
                let d_model = self.n_features;
                if d_model > 0 && d_model % n_heads != 0 {
                    return Err(FactoryError::IncompatibleArm {
                        reason: format!("n_heads={} does not divide d_model={}", n_heads, d_model),
                    });
                }

                let attn_config = StreamingAttentionConfig::builder()
                    .d_model(d_model)
                    .n_heads(n_heads)
                    .mode(AttentionMode::DeltaProduct {
                        n_compositions,
                        reflections: false,
                    })
                    .forgetting_factor(forgetting_factor)
                    .warmup(warmup)
                    .build()?;

                Ok(Box::new(StreamingAttentionModel::new(attn_config)))
            }
            super::Algorithm::Rwkv7 => {
                let n_heads = params.usize("n_heads")?.max(1);
                let forgetting_factor = params.float("forgetting_factor")?;
                let warmup = params.usize("warmup")?;

                // n_heads must divide d_model — enforce as IncompatibleArm.
                let d_model = self.n_features;
                if d_model > 0 && d_model % n_heads != 0 {
                    return Err(FactoryError::IncompatibleArm {
                        reason: format!("n_heads={} does not divide d_model={}", n_heads, d_model),
                    });
                }

                let attn_config = StreamingAttentionConfig::builder()
                    .d_model(d_model)
                    .n_heads(n_heads)
                    .mode(AttentionMode::RWKV7)
                    .forgetting_factor(forgetting_factor)
                    .warmup(warmup)
                    .build()?;

                Ok(Box::new(StreamingAttentionModel::new(attn_config)))
            }
            _ => panic!("create_attention called on non-attention algorithm"),
        }
    }
}
