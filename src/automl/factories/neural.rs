//! Neural network factory: SpikeNet, KAN, TTT.

use crate::automl::space::{int_range, linear_range, log_range, ParamMap, SearchSpace};
use irithyll_core::learner::StreamingLearner;

use super::{Factory, FactoryError};

/// Build the canonical SpikeNet search space.
///
/// **Source for ranges:** alpha (LIF leak factor) ∈ [0.8, 0.999] is the
/// standard range from Bellec et al. (2020) "A solution to the learning
/// dilemma" (e-prop). v_thr ∈ [0.2, 0.8] keeps the threshold within the
/// stable spiking regime per the same paper. Learning rate log-spans
/// [1e-4, 1e-2] following e-prop's reported sweet spot. Hidden size ∈ [16, 256]
/// covers small-to-medium online networks.
fn spike_net_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("n_hidden", int_range(16, 256))
        .param("alpha", linear_range(0.8, 0.999))
        .param("eta", log_range(0.0001, 0.01))
        .param("v_thr", linear_range(0.2, 0.8))
        .build()
        .expect("spike_net_search_space: builder produces a valid space by construction")
}

/// Build the canonical KAN search space.
///
/// **Source for ranges:** grid_size ∈ [3, 10] and spline_order ∈ [2, 4] are
/// the spans used in Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks"
/// §4 for tabular benchmarks. learning_rate log-spans [1e-3, 1e-1] following
/// the same paper's reported optima.
fn kan_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("hidden_size", int_range(4, 32))
        .param("grid_size", int_range(3, 10))
        .param("learning_rate", log_range(0.001, 0.1))
        .param("spline_order", int_range(2, 4))
        .build()
        .expect("kan_search_space: builder produces a valid space by construction")
}

/// Build the canonical streaming TTT search space.
///
/// **Source for ranges:** d_model ∈ [8, 64] covers compact streaming models,
/// learning_rate log-spans [1e-3, 0.1] and alpha ∈ [1e-4, 1e-2] match the
/// hyperparameter sweep ranges in Sun et al. (2024) "Learning to (Learn at
/// Test Time)" §3.3.
fn ttt_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("d_model", int_range(8, 64))
        .param("learning_rate", log_range(0.001, 0.1))
        .param("alpha", linear_range(0.0001, 0.01))
        .build()
        .expect("ttt_search_space: builder produces a valid space by construction")
}

impl Factory {
    /// Create a factory for spiking neural networks (e-prop learning).
    pub fn spike_net() -> Self {
        Self {
            algorithm: super::Algorithm::SpikeNet,
            n_features: 0,
            space: spike_net_search_space(),
            warmup: 20,
            complexity: 16000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for streaming KAN.
    pub fn kan(n_features: usize) -> Self {
        Self {
            algorithm: super::Algorithm::Kan,
            n_features,
            space: kan_search_space(),
            warmup: 20,
            complexity: 2000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for streaming TTT (test-time training).
    pub fn ttt(n_features: usize) -> Self {
        Self {
            algorithm: super::Algorithm::Ttt,
            n_features,
            space: ttt_search_space(),
            warmup: 10,
            complexity: 3000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    pub(crate) fn create_neural(
        &self,
        params: &ParamMap,
    ) -> Result<Box<dyn StreamingLearner>, FactoryError> {
        use crate::snn::{SpikeNet, SpikeNetConfig};

        match self.algorithm {
            super::Algorithm::SpikeNet => {
                let n_hidden = params.usize("n_hidden")?;
                let alpha = params.float("alpha")?;
                let eta = params.float("eta")?;
                let v_thr = params.float("v_thr")?;

                let spike_config = SpikeNetConfig::builder()
                    .n_hidden(n_hidden)
                    .alpha(alpha)
                    .learning_rate(eta)
                    .v_thr(v_thr)
                    .seed(self.seed)
                    .build()?;

                Ok(Box::new(SpikeNet::new(spike_config)))
            }
            super::Algorithm::Kan => {
                let hidden_size = params.usize("hidden_size")?;
                let grid_size = params.usize("grid_size")?;
                let lr = params.float("learning_rate")?;
                let spline_order = params.usize("spline_order")?;

                let kan_config = crate::kan::KANConfig::builder()
                    .layer_sizes(vec![self.n_features, hidden_size, 1])
                    .grid_size(grid_size)
                    .learning_rate(lr)
                    .spline_order(spline_order)
                    .seed(self.seed)
                    .build()?;

                Ok(Box::new(crate::kan::StreamingKAN::new(kan_config)))
            }
            super::Algorithm::Ttt => {
                let d_model = params.usize("d_model")?;
                let eta = params.float("learning_rate")?;
                let alpha = params.float("alpha")?;

                let ttt_config = crate::ttt::TTTConfig::builder()
                    .d_model(d_model)
                    .learning_rate(eta)
                    .alpha(alpha)
                    .warmup(self.warmup)
                    .seed(self.seed)
                    .build()?;

                Ok(Box::new(crate::ttt::StreamingTTT::new(ttt_config)))
            }
            _ => panic!("create_neural called on non-neural algorithm"),
        }
    }
}
