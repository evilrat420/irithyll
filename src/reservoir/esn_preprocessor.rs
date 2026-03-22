//! ESN as a streaming preprocessor for pipeline composition.
//!
//! [`ESNPreprocessor`] wraps a [`CycleReservoir`] as a
//! [`StreamingPreprocessor`],
//! allowing it to be used in preprocessing pipelines before a readout learner.
//!
//! During `update_and_transform`, the reservoir is driven forward one step and
//! the resulting state is returned as the transformed feature vector. During
//! `transform`, the current state is returned without advancing the reservoir.
//!
//! This enables composing ESN preprocessing with arbitrary readout models:
//!
//! ```
//! use irithyll::reservoir::{ESNPreprocessor, ESNConfig};
//! use irithyll::pipeline::Pipeline;
//! use irithyll::learners::RecursiveLeastSquares;
//! use irithyll::learner::StreamingLearner;
//!
//! let config = ESNConfig::builder()
//!     .n_reservoir(50)
//!     .warmup(0)
//!     .build()
//!     .unwrap();
//!
//! let mut pipeline = Pipeline::builder()
//!     .pipe(ESNPreprocessor::new(config))
//!     .learner(RecursiveLeastSquares::new(0.999));
//!
//! pipeline.train(&[1.0], 2.0);
//! let pred = pipeline.predict(&[1.5]);
//! assert!(pred.is_finite());
//! ```

use crate::pipeline::StreamingPreprocessor;
use irithyll_core::reservoir::CycleReservoir;

use super::esn_config::ESNConfig;

/// ESN reservoir as a streaming preprocessor.
///
/// Wraps a [`CycleReservoir`] to produce reservoir state vectors as transformed
/// features. The output dimension equals `n_reservoir` (optionally plus the
/// input dimension if `passthrough_input` is enabled).
///
/// # Usage
///
/// Insert into a pipeline before any readout learner. The reservoir transforms
/// raw input features into a rich nonlinear temporal representation.
pub struct ESNPreprocessor {
    /// The underlying cycle reservoir.
    reservoir: CycleReservoir,
    /// Whether to append input features to the reservoir state.
    passthrough_input: bool,
    /// Number of reservoir neurons.
    n_reservoir: usize,
    /// Input dimension (set on first update).
    n_inputs: Option<usize>,
    /// Full configuration for lazy re-initialization.
    config: ESNConfig,
}

impl ESNPreprocessor {
    /// Create a new ESN preprocessor from the given configuration.
    ///
    /// The reservoir is initialized lazily on the first call to
    /// `update_and_transform`, when the input dimension becomes known.
    pub fn new(config: ESNConfig) -> Self {
        let n_reservoir = config.n_reservoir;
        let passthrough_input = config.passthrough_input;

        // Create a placeholder reservoir with 1 input — will be re-created
        // when we see the actual input dimension.
        let reservoir = CycleReservoir::new(
            config.n_reservoir,
            1,
            config.spectral_radius,
            config.input_scaling,
            config.leak_rate,
            config.bias_scaling,
            config.seed,
        );

        Self {
            reservoir,
            passthrough_input,
            n_reservoir,
            n_inputs: None,
            config,
        }
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

    /// Build the output vector from the current reservoir state.
    fn build_output(&self, input: &[f64]) -> Vec<f64> {
        let state = self.reservoir.state();
        if self.passthrough_input {
            let mut out = Vec::with_capacity(state.len() + input.len());
            out.extend_from_slice(state);
            out.extend_from_slice(input);
            out
        } else {
            state.to_vec()
        }
    }
}

impl StreamingPreprocessor for ESNPreprocessor {
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        self.ensure_reservoir(features.len());
        self.reservoir.update(features);
        self.build_output(features)
    }

    fn transform(&self, features: &[f64]) -> Vec<f64> {
        // Return current state without advancing the reservoir.
        self.build_output(features)
    }

    fn output_dim(&self) -> Option<usize> {
        self.n_inputs.map(|d| {
            if self.passthrough_input {
                self.n_reservoir + d
            } else {
                self.n_reservoir
            }
        })
    }

    fn reset(&mut self) {
        self.reservoir.reset();
        // Keep n_inputs and weights — only reset the dynamic state.
    }
}

impl std::fmt::Debug for ESNPreprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ESNPreprocessor")
            .field("n_reservoir", &self.n_reservoir)
            .field("passthrough_input", &self.passthrough_input)
            .field("output_dim", &self.output_dim())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::StreamingLearner;
    use crate::learners::RecursiveLeastSquares;
    use crate::pipeline::Pipeline;

    fn default_config() -> ESNConfig {
        ESNConfig::builder()
            .n_reservoir(20)
            .warmup(0) // No warmup for preprocessor tests.
            .build()
            .unwrap()
    }

    #[test]
    fn update_and_transform_produces_correct_dim() {
        let mut pre = ESNPreprocessor::new(default_config());
        let out = pre.update_and_transform(&[1.0, 2.0]);

        // passthrough_input=true by default: 20 (reservoir) + 2 (input) = 22.
        assert_eq!(out.len(), 22, "expected 22 features, got {}", out.len());
        assert_eq!(pre.output_dim(), Some(22));
    }

    #[test]
    fn transform_does_not_advance_reservoir() {
        let mut pre = ESNPreprocessor::new(default_config());

        // First, advance the reservoir.
        let out1 = pre.update_and_transform(&[1.0]);
        // Now transform without advancing.
        let out2 = pre.transform(&[1.0]);

        // Reservoir state should be the same (transform is read-only).
        assert_eq!(out1.len(), out2.len());
        for (i, (&a, &b)) in out1.iter().zip(out2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-15,
                "feature {} differs: {} vs {}",
                i,
                a,
                b,
            );
        }

        // But after another update, state should change.
        let out3 = pre.update_and_transform(&[100.0]);
        let differs = out1
            .iter()
            .zip(out3.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(
            differs,
            "state should change after another update_and_transform",
        );
    }

    #[test]
    fn no_passthrough_reduces_dim() {
        let config = ESNConfig::builder()
            .n_reservoir(20)
            .warmup(0)
            .passthrough_input(false)
            .build()
            .unwrap();
        let mut pre = ESNPreprocessor::new(config);
        let out = pre.update_and_transform(&[1.0, 2.0, 3.0]);

        assert_eq!(
            out.len(),
            20,
            "without passthrough, output should equal n_reservoir"
        );
        assert_eq!(pre.output_dim(), Some(20));
    }

    #[test]
    fn reset_zeros_state() {
        let mut pre = ESNPreprocessor::new(default_config());
        pre.update_and_transform(&[1.0]);
        pre.update_and_transform(&[2.0]);

        pre.reset();

        // After reset, the reservoir state should be all zeros.
        let out = pre.transform(&[1.0]);
        // The first n_reservoir elements are the reservoir state.
        let state = &out[..20];
        for &s in state {
            assert_eq!(s, 0.0, "reservoir state should be zero after reset");
        }
    }

    #[test]
    fn output_dim_none_before_first_update() {
        let pre = ESNPreprocessor::new(default_config());
        assert_eq!(pre.output_dim(), None);
    }

    #[test]
    fn pipeline_integration() {
        let config = ESNConfig::builder()
            .n_reservoir(30)
            .warmup(0)
            .seed(42)
            .build()
            .unwrap();

        let mut pipeline = Pipeline::builder()
            .pipe(ESNPreprocessor::new(config))
            .learner(RecursiveLeastSquares::new(0.999));

        // Train on a simple linear relationship.
        for i in 0..100 {
            let x = i as f64 * 0.01;
            pipeline.train(&[x], 2.0 * x + 1.0);
        }

        let pred = pipeline.predict(&[0.5]);
        assert!(
            pred.is_finite(),
            "pipeline prediction should be finite, got {}",
            pred,
        );
    }

    #[test]
    fn multiple_inputs_work() {
        let mut pre = ESNPreprocessor::new(default_config());
        let out = pre.update_and_transform(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // 20 reservoir + 5 input = 25
        assert_eq!(out.len(), 25);
    }
}
