//! SSM/Recurrent factory: Mamba, Mamba3, MambaBD, sLSTM, mGRADE, ESN.

use crate::automl::space::{int_range, linear_range, log_range, ParamMap, SearchSpace};
use irithyll_core::learner::StreamingLearner;

use super::{Factory, FactoryError};

/// Build the canonical ESN search space.
///
/// **Source for ranges:** spectral radius ∈ [0.5, 0.999] is the classical
/// echo-state-property regime (Jaeger 2001 "The 'echo state' approach"),
/// keeping below 1.0 to preserve the contractive map. Leak rate ∈ [0.05, 1.0]
/// covers low-frequency to fully-instantaneous reservoirs (Lukoševičius 2012).
/// Input scaling log-spans [0.1, 5.0] following standard reservoir-tuning
/// practice. Reservoir size ∈ [20, 500] covers small online to medium-sized
/// streaming problems.
fn esn_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("n_reservoir", int_range(20, 500))
        .param("spectral_radius", linear_range(0.5, 0.999))
        .param("leak_rate", linear_range(0.05, 1.0))
        .param("input_scaling", log_range(0.1, 5.0))
        .build()
        .expect("esn_search_space: builder produces a valid space by construction")
}

/// Common Mamba V1 search space (n_state, forgetting_factor, warmup).
///
/// **Source for forgetting-factor range:** [0.95, 0.9999] corresponds to EWMA
/// half-lives ∈ [13.5, ~6900] samples (per Cont 2001 "Empirical properties"),
/// covering quick-adapt to slow-drift regimes. n_state ∈ [4, 64] is the
/// SSM hidden-rank span used in Mamba (Gu & Dao 2024).
fn mamba_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("n_state", int_range(4, 64))
        .param("forgetting_factor", linear_range(0.95, 0.9999))
        .param("warmup", int_range(5, 50))
        .build()
        .expect("mamba_search_space: builder produces a valid space by construction")
}

/// Build a Mamba-3 search space with `n_groups` capped at `max_groups`.
///
/// `max_groups` is derived from `d_in / 2` (group size at least 2), matching
/// Mamba-3 §3.2 (Mamba-3 Team 2025) "MIMO group decomposition with minimum
/// group dimensionality 2".
fn mamba3_search_space(max_groups: i64) -> SearchSpace {
    SearchSpace::builder()
        .param("n_state", int_range(4, 64))
        .param("n_groups", int_range(1, max_groups.max(1)))
        .param("forgetting_factor", linear_range(0.95, 0.9999))
        .param("warmup", int_range(5, 50))
        .build()
        .expect("mamba3_search_space: builder produces a valid space by construction")
}

fn slstm_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("d_model", int_range(8, 256))
        .param("forgetting_factor", linear_range(0.95, 0.9999))
        .param("warmup", int_range(5, 100))
        .build()
        .expect("slstm_search_space: builder produces a valid space by construction")
}

fn mgrade_search_space() -> SearchSpace {
    SearchSpace::builder()
        .param("d_hidden", int_range(4, 64))
        .param("kernel_size", int_range(2, 8))
        .param("forgetting_factor", linear_range(0.95, 0.9999))
        .param("warmup", int_range(5, 50))
        .build()
        .expect("mgrade_search_space: builder produces a valid space by construction")
}

impl Factory {
    /// Create a factory for echo state networks (reservoir computing).
    pub fn esn() -> Self {
        Self {
            algorithm: super::Algorithm::Esn,
            n_features: 0,
            space: esn_search_space(),
            warmup: 50,
            complexity: 10000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for streaming Mamba (selective state space model).
    pub fn mamba(d_in: usize) -> Self {
        Self {
            algorithm: super::Algorithm::Mamba,
            n_features: d_in,
            space: mamba_search_space(),
            warmup: 10,
            complexity: 4000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for streaming Mamba-3 (MIMO groups, complex states).
    pub fn mamba3(d_in: usize) -> Self {
        let max_groups = (d_in / 2).max(1);
        Self {
            algorithm: super::Algorithm::Mamba3,
            n_features: d_in,
            space: mamba3_search_space(max_groups as i64),
            warmup: 10,
            complexity: 5000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for streaming sLSTM.
    pub fn slstm(n_features: usize) -> Self {
        Self {
            algorithm: super::Algorithm::Slstm,
            n_features,
            space: slstm_search_space(),
            warmup: 10,
            complexity: 2500,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for streaming Mamba with BD-LRU (block-diagonal recurrence).
    pub fn mamba_bd(d_in: usize) -> Self {
        Self {
            algorithm: super::Algorithm::MambaBD,
            n_features: d_in,
            space: mamba_search_space(),
            warmup: 10,
            complexity: 6000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for streaming mGRADE.
    pub fn mgrade(d_in: usize) -> Self {
        Self {
            algorithm: super::Algorithm::Mgrade,
            n_features: d_in,
            space: mgrade_search_space(),
            warmup: 10,
            complexity: 2000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    pub(crate) fn create_ssm(
        &self,
        params: &ParamMap,
    ) -> Result<Box<dyn StreamingLearner>, FactoryError> {
        use crate::reservoir::{ESNConfig, EchoStateNetwork};
        use crate::ssm::{MambaConfig, MambaVersion, StreamingMamba};

        match self.algorithm {
            super::Algorithm::Esn => {
                let n_reservoir = params.usize("n_reservoir")?;
                let spectral_radius = params.float("spectral_radius")?;
                let leak_rate = params.float("leak_rate")?;
                let input_scaling = params.float("input_scaling")?;

                let esn_config = ESNConfig::builder()
                    .n_reservoir(n_reservoir)
                    .spectral_radius(spectral_radius)
                    .leak_rate(leak_rate)
                    .input_scaling(input_scaling)
                    .seed(self.seed)
                    .build()?;

                Ok(Box::new(EchoStateNetwork::new(esn_config)))
            }
            super::Algorithm::Mamba => {
                let n_state = params.usize("n_state")?;
                let forgetting_factor = params.float("forgetting_factor")?;
                let warmup = params.usize("warmup")?;

                let mamba_config = MambaConfig::builder()
                    .d_in(self.n_features)
                    .n_state(n_state)
                    .forgetting_factor(forgetting_factor)
                    .warmup(warmup)
                    .build()?;

                Ok(Box::new(StreamingMamba::new(mamba_config)))
            }
            super::Algorithm::Mamba3 => {
                let n_state = params.usize("n_state")?;
                let n_groups = params.usize("n_groups")?;
                let forgetting_factor = params.float("forgetting_factor")?;
                let warmup = params.usize("warmup")?;

                // n_groups must divide d_in. The search space gates this with a
                // constraint at sample time, but a manually constructed ParamMap
                // could still violate it — fail explicitly.
                let d_in = self.n_features;
                if d_in > 0 && n_groups > 0 && d_in % n_groups != 0 {
                    return Err(FactoryError::IncompatibleArm {
                        reason: format!("n_groups={} does not divide d_in={}", n_groups, d_in),
                    });
                }

                let mamba_config = MambaConfig::builder()
                    .d_in(d_in)
                    .n_state(n_state)
                    .version(MambaVersion::V3)
                    .n_groups(n_groups.max(1))
                    .forgetting_factor(forgetting_factor)
                    .warmup(warmup)
                    .build()?;

                Ok(Box::new(StreamingMamba::new(mamba_config)))
            }
            super::Algorithm::Slstm => {
                let d_model = params.usize("d_model")?;
                let ff = params.float("forgetting_factor")?;
                let warmup = params.usize("warmup")?;

                let slstm_config = crate::lstm::SLSTMConfig::builder()
                    .d_model(d_model)
                    .forgetting_factor(ff)
                    .warmup(warmup)
                    .seed(self.seed)
                    .build()?;

                Ok(Box::new(crate::lstm::StreamingsLSTM::new(slstm_config)))
            }
            super::Algorithm::MambaBD => {
                let n_state = params.usize("n_state")?;
                let ff = params.float("forgetting_factor")?;
                let warmup = params.usize("warmup")?;

                let d_in = self.n_features;
                let block_size = [8, 4, 2]
                    .iter()
                    .copied()
                    .find(|&bs| d_in % bs == 0)
                    .unwrap_or(2);

                let mamba_config = MambaConfig::builder()
                    .d_in(d_in)
                    .n_state(n_state)
                    .version(MambaVersion::BlockDiagonal { block_size })
                    .block_size(block_size)
                    .forgetting_factor(ff)
                    .warmup(warmup)
                    .seed(self.seed)
                    .build()?;

                Ok(Box::new(StreamingMamba::new(mamba_config)))
            }
            super::Algorithm::Mgrade => {
                let d_hidden = params.usize("d_hidden")?;
                let kernel_size = params.usize("kernel_size")?;
                let ff = params.float("forgetting_factor")?;
                let warmup = params.usize("warmup")?;

                let mgrade_config = crate::mgrade::MGradeConfig::builder()
                    .d_in(self.n_features)
                    .d_hidden(d_hidden)
                    .kernel_size(kernel_size)
                    .forgetting_factor(ff)
                    .warmup(warmup)
                    .seed(self.seed)
                    .build()?;

                Ok(Box::new(crate::mgrade::StreamingMGrade::new(mgrade_config)))
            }
            _ => panic!("create_ssm called on non-SSM algorithm"),
        }
    }
}
