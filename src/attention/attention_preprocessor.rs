//! Attention-based streaming preprocessor for pipeline composition.
//!
//! [`AttentionPreprocessor`] wraps a multi-head linear attention layer and
//! exposes it as a [`StreamingPreprocessor`], enabling use in irithyll pipelines.
//! This allows the attention layer's temporal feature extraction to be chained
//! with any downstream learner (SGBT, RLS, etc.).
//!
//! # Architecture
//!
//! ```text
//! raw features ──→ [AttentionPreprocessor] ──→ temporal features ──→ [any learner]
//!   (d_model)                                      (output_dim)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use irithyll::attention::{AttentionPreprocessor, AttentionMode};
//! use irithyll::pipeline::StreamingPreprocessor;
//!
//! let mut pre = AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42);
//! let temporal = pre.update_and_transform(&[1.0, 2.0, 3.0, 4.0]);
//! assert_eq!(temporal.len(), 4);
//! ```

use irithyll_core::attention::{
    AttentionConfig, AttentionLayer, AttentionMode, MultiHeadAttention,
};

use crate::pipeline::StreamingPreprocessor;

/// Multi-head linear attention as a streaming preprocessor for pipeline composition.
///
/// During `update_and_transform`, the attention layer processes the input and
/// advances its recurrent state. During `transform`, it returns the cached output
/// from the last `update_and_transform` call without modifying state.
///
/// # Example
///
/// ```ignore
/// use irithyll::{pipe, rls};
/// use irithyll::attention::{AttentionPreprocessor, AttentionMode};
/// use irithyll::learner::StreamingLearner;
///
/// // Attention preprocessor -> RLS learner pipeline
/// let mut pipeline = pipe(
///     AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42)
/// ).learner(rls(0.99));
/// pipeline.train(&[1.0, 2.0, 3.0, 4.0], 5.0);
/// let pred = pipeline.predict(&[1.0, 2.0, 3.0, 4.0]);
/// assert!(pred.is_finite());
/// ```
pub struct AttentionPreprocessor {
    /// Multi-head linear attention for temporal feature extraction.
    attention: MultiHeadAttention,
    /// Cached output from the last update_and_transform call.
    last_output: Vec<f64>,
    /// Number of samples seen.
    samples_seen: u64,
}

impl AttentionPreprocessor {
    /// Create a new AttentionPreprocessor.
    ///
    /// # Arguments
    ///
    /// * `d_model` -- input/output feature dimension
    /// * `n_heads` -- number of attention heads
    /// * `mode` -- attention variant (GLA, DeltaNet, Hawk, etc.)
    /// * `seed` -- random seed for attention weight initialization
    pub fn new(d_model: usize, n_heads: usize, mode: AttentionMode, seed: u64) -> Self {
        let d_head = if d_model >= n_heads {
            d_model / n_heads
        } else {
            1
        };
        let config = AttentionConfig {
            d_model,
            n_heads,
            d_key: d_head,
            d_value: d_head,
            mode,
            seed,
        };
        let attention = MultiHeadAttention::new(config);
        let out_dim = attention.output_dim();
        let last_output = vec![0.0; out_dim];
        Self {
            attention,
            last_output,
            samples_seen: 0,
        }
    }

    /// Get the current attention recurrent state.
    pub fn attention_state(&self) -> &[f64] {
        self.attention.state()
    }

    /// Number of samples processed so far.
    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }
}

impl StreamingPreprocessor for AttentionPreprocessor {
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        let output = self.attention.forward(features);
        self.last_output = output.clone();
        self.samples_seen += 1;
        output
    }

    fn transform(&self, _features: &[f64]) -> Vec<f64> {
        // Return cached output without advancing attention state.
        // This matches the StreamingPreprocessor contract: transform() is read-only.
        self.last_output.clone()
    }

    fn output_dim(&self) -> Option<usize> {
        Some(self.attention.output_dim())
    }

    fn reset(&mut self) {
        self.attention.reset();
        for v in self.last_output.iter_mut() {
            *v = 0.0;
        }
        self.samples_seen = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_correct_dim() {
        let pre = AttentionPreprocessor::new(8, 2, AttentionMode::GLA, 42);
        assert_eq!(pre.output_dim(), Some(8));
    }

    #[test]
    fn update_and_transform_returns_correct_dim() {
        let mut pre = AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42);
        let out = pre.update_and_transform(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.len(), 4, "output dim should match d_model");
    }

    #[test]
    fn update_and_transform_produces_finite_output() {
        let mut pre = AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42);
        let out = pre.update_and_transform(&[1.0, -1.0, 0.5, 0.0]);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] should be finite, got {}", i, v);
        }
    }

    #[test]
    fn transform_returns_cached_output() {
        let mut pre = AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42);
        let updated = pre.update_and_transform(&[1.0, 2.0, 3.0, 4.0]);
        let transformed = pre.transform(&[99.0, 99.0, 99.0, 99.0]);
        assert_eq!(updated.len(), transformed.len(), "dimensions should match");
        for (i, (&u, &t)) in updated.iter().zip(transformed.iter()).enumerate() {
            assert!(
                (u - t).abs() < 1e-15,
                "transform should return cached output: [{}] {} vs {}",
                i,
                u,
                t
            );
        }
    }

    #[test]
    fn transform_before_any_update_returns_zeros() {
        let pre = AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42);
        let out = pre.transform(&[1.0, 2.0, 3.0, 4.0]);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.abs() < 1e-15,
                "transform before update should return zeros: [{}] = {}",
                i,
                v
            );
        }
    }

    #[test]
    fn reset_clears_state_and_cache() {
        let mut pre = AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42);
        let _ = pre.update_and_transform(&[5.0, 10.0, 15.0, 20.0]);
        assert_eq!(pre.samples_seen(), 1);

        pre.reset();
        assert_eq!(pre.samples_seen(), 0);

        // Attention state should be zero
        for &h in pre.attention_state() {
            assert!(
                h.abs() < 1e-15,
                "attention state should be zero after reset"
            );
        }
        // Cached output should be zero
        let out = pre.transform(&[0.0, 0.0, 0.0, 0.0]);
        for &v in &out {
            assert!(v.abs() < 1e-15, "cached output should be zero after reset");
        }
    }

    #[test]
    fn sequential_updates_change_output() {
        let mut pre = AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42);
        let out1 = pre.update_and_transform(&[1.0, 0.0, 0.0, 0.0]);
        let out2 = pre.update_and_transform(&[1.0, 0.0, 0.0, 0.0]);
        // Second call has different state, so output should differ
        let diff: f64 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(
            diff > 1e-20,
            "sequential updates should produce different outputs: {:?} vs {:?}",
            out1,
            out2
        );
    }

    #[test]
    fn pipeline_integration() {
        // Test that AttentionPreprocessor works in a pipeline
        use crate::learner::StreamingLearner;
        use crate::learners::RecursiveLeastSquares;
        use crate::pipeline::Pipeline;

        let pre = AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42);
        let rls = RecursiveLeastSquares::new(0.99);
        let mut pipeline = Pipeline::builder().pipe(pre).learner(rls);

        // Train a few samples
        pipeline.train(&[1.0, 2.0, 3.0, 4.0], 5.0);
        pipeline.train(&[2.0, 3.0, 4.0, 5.0], 7.0);

        let pred = pipeline.predict(&[1.5, 2.5, 3.5, 4.5]);
        assert!(
            pred.is_finite(),
            "pipeline prediction should be finite, got {}",
            pred
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut pre1 = AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42);
        let mut pre2 = AttentionPreprocessor::new(4, 2, AttentionMode::GLA, 42);
        let input = [3.0, -1.0, 0.5, 2.0];
        let out1 = pre1.update_and_transform(&input);
        let out2 = pre2.update_and_transform(&input);
        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-15,
                "same seed should produce identical output: [{}] {} vs {}",
                i,
                a,
                b
            );
        }
    }
}
