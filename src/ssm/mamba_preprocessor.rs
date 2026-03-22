//! SSM-based streaming preprocessor for pipeline composition.
//!
//! [`MambaPreprocessor`] wraps a selective SSM and exposes it as a
//! [`StreamingPreprocessor`], enabling use in irithyll pipelines. This allows
//! the SSM's temporal feature extraction to be chained with any downstream
//! learner (SGBT, RLS, etc.).
//!
//! # Architecture
//!
//! ```text
//! raw features ──→ [MambaPreprocessor (SSM)] ──→ temporal features ──→ [any learner]
//!   (d_in)                                           (d_in)
//! ```
//!
//! # Usage
//!
//! ```
//! use irithyll::ssm::MambaPreprocessor;
//! use irithyll::pipeline::StreamingPreprocessor;
//!
//! let mut pre = MambaPreprocessor::new(4, 16, 42);
//! let temporal = pre.update_and_transform(&[1.0, 2.0, 3.0, 4.0]);
//! assert_eq!(temporal.len(), 4);
//! ```

use irithyll_core::ssm::{SSMLayer, SelectiveSSM};

use crate::pipeline::StreamingPreprocessor;

/// Selective SSM as a streaming preprocessor for pipeline composition.
///
/// During `update_and_transform`, the SSM processes the input and advances its
/// hidden state. During `transform`, it returns the cached output from the last
/// `update_and_transform` call without modifying state.
///
/// # Example
///
/// ```
/// use irithyll::{pipe, rls};
/// use irithyll::ssm::MambaPreprocessor;
/// use irithyll::learner::StreamingLearner;
///
/// // SSM preprocessor → RLS learner pipeline
/// let mut pipeline = pipe(MambaPreprocessor::new(3, 8, 42)).learner(rls(0.99));
/// pipeline.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = pipeline.predict(&[1.0, 2.0, 3.0]);
/// assert!(pred.is_finite());
/// ```
pub struct MambaPreprocessor {
    /// Selective SSM for temporal feature extraction.
    ssm: SelectiveSSM,
    /// Cached output from the last update_and_transform call.
    last_output: Vec<f64>,
    /// Input/output dimension.
    d_in: usize,
}

impl MambaPreprocessor {
    /// Create a new MambaPreprocessor.
    ///
    /// # Arguments
    ///
    /// * `d_in` -- input/output feature dimension
    /// * `n_state` -- hidden state dimension per channel
    /// * `seed` -- random seed for SSM weight initialization
    pub fn new(d_in: usize, n_state: usize, seed: u64) -> Self {
        let ssm = SelectiveSSM::new(d_in, n_state, seed);
        let last_output = vec![0.0; d_in];
        Self {
            ssm,
            last_output,
            d_in,
        }
    }

    /// Get the current SSM hidden state.
    pub fn ssm_state(&self) -> &[f64] {
        self.ssm.state()
    }
}

impl StreamingPreprocessor for MambaPreprocessor {
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        let output = self.ssm.forward(features);
        self.last_output = output.clone();
        output
    }

    fn transform(&self, _features: &[f64]) -> Vec<f64> {
        // Return cached output without advancing SSM state.
        // This matches the StreamingPreprocessor contract: transform() is read-only.
        self.last_output.clone()
    }

    fn output_dim(&self) -> Option<usize> {
        Some(self.d_in)
    }

    fn reset(&mut self) {
        self.ssm.reset();
        for v in self.last_output.iter_mut() {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_correct_dim() {
        let pre = MambaPreprocessor::new(5, 8, 42);
        assert_eq!(pre.output_dim(), Some(5));
        assert_eq!(pre.d_in, 5);
    }

    #[test]
    fn update_and_transform_returns_correct_dim() {
        let mut pre = MambaPreprocessor::new(3, 8, 42);
        let out = pre.update_and_transform(&[1.0, 2.0, 3.0]);
        assert_eq!(out.len(), 3, "output dim should match d_in");
    }

    #[test]
    fn update_and_transform_produces_finite_output() {
        let mut pre = MambaPreprocessor::new(3, 8, 42);
        let out = pre.update_and_transform(&[1.0, -1.0, 0.5]);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] should be finite, got {}", i, v);
        }
    }

    #[test]
    fn transform_returns_cached_output() {
        let mut pre = MambaPreprocessor::new(2, 4, 42);
        let updated = pre.update_and_transform(&[1.0, 2.0]);
        let transformed = pre.transform(&[99.0, 99.0]); // different input, same cached output
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
        let pre = MambaPreprocessor::new(3, 8, 42);
        let out = pre.transform(&[1.0, 2.0, 3.0]);
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
        let mut pre = MambaPreprocessor::new(2, 4, 42);
        let _ = pre.update_and_transform(&[5.0, 10.0]);
        pre.reset();

        // SSM state should be zero
        for &h in pre.ssm_state() {
            assert!(h.abs() < 1e-15, "SSM state should be zero after reset");
        }
        // Cached output should be zero
        let out = pre.transform(&[0.0, 0.0]);
        for &v in &out {
            assert!(v.abs() < 1e-15, "cached output should be zero after reset");
        }
    }

    #[test]
    fn sequential_updates_change_output() {
        let mut pre = MambaPreprocessor::new(2, 4, 42);
        let out1 = pre.update_and_transform(&[1.0, 0.0]);
        let out2 = pre.update_and_transform(&[1.0, 0.0]);
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
        // Test that MambaPreprocessor works in a pipeline
        use crate::learner::StreamingLearner;
        use crate::learners::RecursiveLeastSquares;
        use crate::pipeline::Pipeline;

        let pre = MambaPreprocessor::new(2, 4, 42);
        let rls = RecursiveLeastSquares::new(0.99);
        let mut pipeline = Pipeline::builder().pipe(pre).learner(rls);

        // Train a few samples
        pipeline.train(&[1.0, 2.0], 3.0);
        pipeline.train(&[2.0, 3.0], 5.0);

        let pred = pipeline.predict(&[1.5, 2.5]);
        assert!(
            pred.is_finite(),
            "pipeline prediction should be finite, got {}",
            pred
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut pre1 = MambaPreprocessor::new(2, 4, 42);
        let mut pre2 = MambaPreprocessor::new(2, 4, 42);
        let input = [3.0, -1.0];
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
