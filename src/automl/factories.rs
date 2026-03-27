//! Model factories for AutoML hyperparameter search.
//!
//! Each factory implements [`ModelFactory`] for a specific streaming learner type,
//! defining its hyperparameter search space and how to construct instances from
//! sampled configurations.

use crate::automl::{ConfigSpace, HyperConfig, HyperParam, ModelFactory};
use crate::ensemble::config::SGBTConfig;
use crate::ensemble::distributional::DistributionalSGBT;
use crate::learner::SGBTLearner;
use crate::reservoir::{ESNConfig, EchoStateNetwork};
use crate::snn::{SpikeNet, SpikeNetConfig};
use crate::ssm::{MambaConfig, StreamingMamba};
use irithyll_core::attention::AttentionMode;

use crate::attention::{StreamingAttentionConfig, StreamingAttentionModel};

// ===========================================================================
// SgbtFactory
// ===========================================================================

/// Factory for streaming gradient boosted trees.
///
/// Creates [`SGBTLearner`] instances with squared loss from sampled
/// hyperparameter configurations. The search space covers the six most
/// impactful SGBT hyperparameters.
///
/// # Config Space
///
/// | Index | Name | Type | Range | Scale |
/// |-------|------|------|-------|-------|
/// | 0 | `learning_rate` | Float | [0.001, 0.3] | log |
/// | 1 | `n_steps` | Int | [10, 500] | -- |
/// | 2 | `max_depth` | Int | [3, 10] | -- |
/// | 3 | `n_bins` | Int | [16, 256] | -- |
/// | 4 | `lambda` | Float | [0.01, 10.0] | log |
/// | 5 | `feature_subsample_rate` | Float | [0.3, 1.0] | linear |
#[deprecated(since = "9.7.2", note = "Use Factory::sgbt() instead")]
pub struct SgbtFactory {
    #[allow(dead_code)]
    n_features: usize,
}

#[allow(deprecated)]
impl SgbtFactory {
    /// Create a new SGBT factory.
    ///
    /// `n_features` is stored for documentation and future use (e.g.,
    /// feature-dependent search space adaptation).
    pub fn new(n_features: usize) -> Self {
        Self { n_features }
    }
}

#[allow(deprecated)]
impl ModelFactory for SgbtFactory {
    fn config_space(&self) -> ConfigSpace {
        ConfigSpace::new()
            .push(HyperParam::Float {
                name: "learning_rate",
                low: 0.001,
                high: 0.3,
                log_scale: true,
            })
            .push(HyperParam::Int {
                name: "n_steps",
                low: 10,
                high: 500,
            })
            .push(HyperParam::Int {
                name: "max_depth",
                low: 3,
                high: 10,
            })
            .push(HyperParam::Int {
                name: "n_bins",
                low: 16,
                high: 256,
            })
            .push(HyperParam::Float {
                name: "lambda",
                low: 0.01,
                high: 10.0,
                log_scale: true,
            })
            .push(HyperParam::Float {
                name: "feature_subsample_rate",
                low: 0.3,
                high: 1.0,
                log_scale: false,
            })
    }

    fn create(&self, config: &HyperConfig) -> Box<dyn irithyll_core::learner::StreamingLearner> {
        let learning_rate = config.get(0);
        let n_steps = config.get(1) as usize;
        let max_depth = config.get(2) as usize;
        let n_bins = config.get(3) as usize;
        let lambda = config.get(4);
        let feature_subsample_rate = config.get(5);

        let sgbt_config = SGBTConfig::builder()
            .learning_rate(learning_rate)
            .n_steps(n_steps)
            .max_depth(max_depth)
            .n_bins(n_bins)
            .lambda(lambda)
            .feature_subsample_rate(feature_subsample_rate)
            .build()
            .expect("SgbtFactory::create: invalid config from search space");

        Box::new(SGBTLearner::from_config(sgbt_config))
    }

    fn name(&self) -> &str {
        "SGBT"
    }

    fn complexity_hint(&self) -> usize {
        500
    }
}

// ===========================================================================
// EsnFactory
// ===========================================================================

/// Factory for echo state networks.
///
/// Creates [`EchoStateNetwork`] instances from sampled hyperparameter
/// configurations. The search space covers the four primary ESN
/// hyperparameters.
///
/// # Config Space
///
/// | Index | Name | Type | Range | Scale |
/// |-------|------|------|-------|-------|
/// | 0 | `n_reservoir` | Int | [20, 500] | -- |
/// | 1 | `spectral_radius` | Float | [0.5, 0.999] | linear |
/// | 2 | `leak_rate` | Float | [0.05, 1.0] | linear |
/// | 3 | `input_scaling` | Float | [0.1, 5.0] | log |
#[deprecated(since = "9.7.2", note = "Use Factory::esn() instead")]
pub struct EsnFactory {
    seed: u64,
}

#[allow(deprecated)]
impl EsnFactory {
    /// Create a new ESN factory with default seed (42).
    pub fn new() -> Self {
        Self { seed: 42 }
    }

    /// Create a new ESN factory with a custom seed.
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

#[allow(deprecated)]
impl Default for EsnFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(deprecated)]
impl ModelFactory for EsnFactory {
    fn warmup_hint(&self) -> usize {
        50
    }

    fn config_space(&self) -> ConfigSpace {
        ConfigSpace::new()
            .push(HyperParam::Int {
                name: "n_reservoir",
                low: 20,
                high: 500,
            })
            .push(HyperParam::Float {
                name: "spectral_radius",
                low: 0.5,
                high: 0.999,
                log_scale: false,
            })
            .push(HyperParam::Float {
                name: "leak_rate",
                low: 0.05,
                high: 1.0,
                log_scale: false,
            })
            .push(HyperParam::Float {
                name: "input_scaling",
                low: 0.1,
                high: 5.0,
                log_scale: true,
            })
    }

    fn create(&self, config: &HyperConfig) -> Box<dyn irithyll_core::learner::StreamingLearner> {
        let n_reservoir = config.get(0) as usize;
        let spectral_radius = config.get(1);
        let leak_rate = config.get(2);
        let input_scaling = config.get(3);

        let esn_config = ESNConfig::builder()
            .n_reservoir(n_reservoir)
            .spectral_radius(spectral_radius)
            .leak_rate(leak_rate)
            .input_scaling(input_scaling)
            .seed(self.seed)
            .build()
            .expect("EsnFactory::create: invalid config from search space");

        Box::new(EchoStateNetwork::new(esn_config))
    }

    fn name(&self) -> &str {
        "ESN"
    }

    fn complexity_hint(&self) -> usize {
        10000
    }
}

// ===========================================================================
// MambaFactory
// ===========================================================================

/// Factory for streaming Mamba (selective state space) models.
///
/// Creates [`StreamingMamba`] instances from sampled hyperparameter
/// configurations. The input dimension `d_in` is fixed at factory
/// construction time (not tuned).
///
/// # Config Space
///
/// | Index | Name | Type | Range | Scale |
/// |-------|------|------|-------|-------|
/// | 0 | `n_state` | Int | [4, 64] | -- |
/// | 1 | `forgetting_factor` | Float | [0.95, 0.9999] | linear |
/// | 2 | `warmup` | Int | [5, 50] | -- |
#[deprecated(since = "9.7.2", note = "Use Factory::mamba() instead")]
pub struct MambaFactory {
    d_in: usize,
}

#[allow(deprecated)]
impl MambaFactory {
    /// Create a new Mamba factory with the given input dimension.
    ///
    /// `d_in` is the number of input features, which is fixed and not
    /// part of the hyperparameter search.
    pub fn new(d_in: usize) -> Self {
        Self { d_in }
    }
}

#[allow(deprecated)]
impl ModelFactory for MambaFactory {
    fn warmup_hint(&self) -> usize {
        10
    }

    fn config_space(&self) -> ConfigSpace {
        ConfigSpace::new()
            .push(HyperParam::Int {
                name: "n_state",
                low: 4,
                high: 64,
            })
            .push(HyperParam::Float {
                name: "forgetting_factor",
                low: 0.95,
                high: 0.9999,
                log_scale: false,
            })
            .push(HyperParam::Int {
                name: "warmup",
                low: 5,
                high: 50,
            })
    }

    fn create(&self, config: &HyperConfig) -> Box<dyn irithyll_core::learner::StreamingLearner> {
        let n_state = config.get(0) as usize;
        let forgetting_factor = config.get(1);
        let warmup = config.get(2) as usize;

        let mamba_config = MambaConfig::builder()
            .d_in(self.d_in)
            .n_state(n_state)
            .forgetting_factor(forgetting_factor)
            .warmup(warmup)
            .build()
            .expect("MambaFactory::create: invalid config from search space");

        Box::new(StreamingMamba::new(mamba_config))
    }

    fn name(&self) -> &str {
        "Mamba"
    }

    fn complexity_hint(&self) -> usize {
        4000
    }
}

// ===========================================================================
// AttentionFactory
// ===========================================================================

/// Factory for streaming linear attention models.
///
/// Creates [`StreamingAttentionModel`] instances with GLA mode from sampled
/// hyperparameter configurations. The model dimension `d_model` is fixed at
/// factory construction time (not tuned).
///
/// # Config Space
///
/// | Index | Name | Type | Range | Scale |
/// |-------|------|------|-------|-------|
/// | 0 | `n_heads` | Categorical | {1, 2, 4, 8} | -- |
/// | 1 | `forgetting_factor` | Float | [0.95, 0.9999] | linear |
/// | 2 | `warmup` | Int | [5, 50] | -- |
#[deprecated(since = "9.7.2", note = "Use Factory::attention() instead")]
pub struct AttentionFactory {
    d_model: usize,
}

/// Number of attention heads choices for the categorical parameter.
const ATTENTION_HEAD_CHOICES: [usize; 4] = [1, 2, 4, 8];

#[allow(deprecated)]
impl AttentionFactory {
    /// Create a new attention factory with the given model dimension.
    ///
    /// `d_model` is the input feature dimension, which is fixed and not
    /// part of the hyperparameter search. It must be divisible by all
    /// candidate `n_heads` values (1, 2, 4, 8) for the search to work
    /// correctly.
    pub fn new(d_model: usize) -> Self {
        Self { d_model }
    }
}

#[allow(deprecated)]
impl ModelFactory for AttentionFactory {
    fn warmup_hint(&self) -> usize {
        10
    }

    fn config_space(&self) -> ConfigSpace {
        ConfigSpace::new()
            .push(HyperParam::Categorical {
                name: "n_heads",
                n_choices: ATTENTION_HEAD_CHOICES.len(),
            })
            .push(HyperParam::Float {
                name: "forgetting_factor",
                low: 0.95,
                high: 0.9999,
                log_scale: false,
            })
            .push(HyperParam::Int {
                name: "warmup",
                low: 5,
                high: 50,
            })
    }

    fn create(&self, config: &HyperConfig) -> Box<dyn irithyll_core::learner::StreamingLearner> {
        let head_idx = config.get(0) as usize;
        let n_heads = ATTENTION_HEAD_CHOICES[head_idx.min(ATTENTION_HEAD_CHOICES.len() - 1)];
        let forgetting_factor = config.get(1);
        let warmup = config.get(2) as usize;

        let attn_config = StreamingAttentionConfig::builder()
            .d_model(self.d_model)
            .n_heads(n_heads)
            .mode(AttentionMode::GLA)
            .forgetting_factor(forgetting_factor)
            .warmup(warmup)
            .build()
            .expect("AttentionFactory::create: invalid config from search space");

        Box::new(StreamingAttentionModel::new(attn_config))
    }

    fn name(&self) -> &str {
        "Attention"
    }

    fn complexity_hint(&self) -> usize {
        8000
    }
}

// ===========================================================================
// SpikeNetFactory
// ===========================================================================

/// Factory for spiking neural networks.
///
/// Creates [`SpikeNet`] instances from sampled hyperparameter configurations.
/// The search space covers the four most impactful SNN hyperparameters.
///
/// # Config Space
///
/// | Index | Name | Type | Range | Scale |
/// |-------|------|------|-------|-------|
/// | 0 | `n_hidden` | Int | [16, 256] | -- |
/// | 1 | `alpha` | Float | [0.8, 0.999] | linear |
/// | 2 | `eta` | Float | [0.0001, 0.01] | log |
/// | 3 | `v_thr` | Float | [0.2, 0.8] | linear |
#[deprecated(since = "9.7.2", note = "Use Factory::spike_net() instead")]
pub struct SpikeNetFactory {
    seed: u64,
}

#[allow(deprecated)]
impl SpikeNetFactory {
    /// Create a new SpikeNet factory with default seed (42).
    pub fn new() -> Self {
        Self { seed: 42 }
    }

    /// Create a new SpikeNet factory with a custom seed.
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

#[allow(deprecated)]
impl Default for SpikeNetFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(deprecated)]
impl ModelFactory for SpikeNetFactory {
    fn warmup_hint(&self) -> usize {
        20
    }

    fn config_space(&self) -> ConfigSpace {
        ConfigSpace::new()
            .push(HyperParam::Int {
                name: "n_hidden",
                low: 16,
                high: 256,
            })
            .push(HyperParam::Float {
                name: "alpha",
                low: 0.8,
                high: 0.999,
                log_scale: false,
            })
            .push(HyperParam::Float {
                name: "eta",
                low: 0.0001,
                high: 0.01,
                log_scale: true,
            })
            .push(HyperParam::Float {
                name: "v_thr",
                low: 0.2,
                high: 0.8,
                log_scale: false,
            })
    }

    fn create(&self, config: &HyperConfig) -> Box<dyn irithyll_core::learner::StreamingLearner> {
        let n_hidden = config.get(0) as usize;
        let alpha = config.get(1);
        let eta = config.get(2);
        let v_thr = config.get(3);

        let spike_config = SpikeNetConfig::builder()
            .n_hidden(n_hidden)
            .alpha(alpha)
            .eta(eta)
            .v_thr(v_thr)
            .seed(self.seed)
            .build()
            .expect("SpikeNetFactory::create: invalid config from search space");

        Box::new(SpikeNet::new(spike_config))
    }

    fn name(&self) -> &str {
        "SpikeNet"
    }

    fn complexity_hint(&self) -> usize {
        16000
    }
}

// ===========================================================================
// Algorithm + Unified Factory
// ===========================================================================

/// Algorithm type for the unified factory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Streaming gradient boosted trees.
    Sgbt,
    /// Distributional SGBT (Gaussian output with mu + sigma).
    Distributional,
    /// Echo state network (reservoir computing).
    Esn,
    /// Streaming Mamba (selective state space model).
    Mamba,
    /// Streaming linear attention (GLA mode).
    Attention,
    /// Spiking neural network (e-prop learning).
    SpikeNet,
}

/// Unified model factory for AutoML.
///
/// Replaces the separate per-algorithm factory types with a single type
/// that covers all algorithms via constructor methods.
///
/// # Examples
///
/// ```no_run
/// use irithyll::automl::Factory;
///
/// // Simple: auto-tune SGBT
/// let f = Factory::sgbt(5);
///
/// // Custom search space
/// let f = Factory::esn()
///     .with_warmup(100);
/// ```
pub struct Factory {
    algorithm: Algorithm,
    n_features: usize,
    space: ConfigSpace,
    warmup: usize,
    complexity: usize,
    seed: u64,
}

impl Factory {
    /// Create a factory for streaming gradient boosted trees.
    ///
    /// `n_features` is stored for documentation and future use.
    pub fn sgbt(n_features: usize) -> Self {
        let space = ConfigSpace::new()
            .push(HyperParam::Float {
                name: "learning_rate",
                low: 0.001,
                high: 0.3,
                log_scale: true,
            })
            .push(HyperParam::Int {
                name: "n_steps",
                low: 10,
                high: 500,
            })
            .push(HyperParam::Int {
                name: "max_depth",
                low: 3,
                high: 10,
            })
            .push(HyperParam::Int {
                name: "n_bins",
                low: 16,
                high: 256,
            })
            .push(HyperParam::Float {
                name: "lambda",
                low: 0.01,
                high: 10.0,
                log_scale: true,
            })
            .push(HyperParam::Float {
                name: "feature_subsample_rate",
                low: 0.3,
                high: 1.0,
                log_scale: false,
            });

        Self {
            algorithm: Algorithm::Sgbt,
            n_features,
            space,
            warmup: 0,
            complexity: 500,
            seed: 42,
        }
    }

    /// Create a factory for distributional SGBT (Gaussian output with mu + sigma).
    ///
    /// Uses the same hyperparameter space as SGBT (location chain uses the same
    /// hyperparameters). The scale chain uses default settings.
    pub fn distributional(n_features: usize) -> Self {
        let space = ConfigSpace::new()
            .push(HyperParam::Float {
                name: "learning_rate",
                low: 0.001,
                high: 0.3,
                log_scale: true,
            })
            .push(HyperParam::Int {
                name: "n_steps",
                low: 10,
                high: 500,
            })
            .push(HyperParam::Int {
                name: "max_depth",
                low: 3,
                high: 10,
            })
            .push(HyperParam::Int {
                name: "n_bins",
                low: 16,
                high: 256,
            })
            .push(HyperParam::Float {
                name: "lambda",
                low: 0.01,
                high: 10.0,
                log_scale: true,
            })
            .push(HyperParam::Float {
                name: "feature_subsample_rate",
                low: 0.3,
                high: 1.0,
                log_scale: false,
            });

        Self {
            algorithm: Algorithm::Distributional,
            n_features,
            space,
            warmup: 0,
            complexity: 1000,
            seed: 42,
        }
    }

    /// Create a factory for echo state networks (reservoir computing).
    ///
    /// Input dimension is auto-detected from the first training sample.
    pub fn esn() -> Self {
        let space = ConfigSpace::new()
            .push(HyperParam::Int {
                name: "n_reservoir",
                low: 20,
                high: 500,
            })
            .push(HyperParam::Float {
                name: "spectral_radius",
                low: 0.5,
                high: 0.999,
                log_scale: false,
            })
            .push(HyperParam::Float {
                name: "leak_rate",
                low: 0.05,
                high: 1.0,
                log_scale: false,
            })
            .push(HyperParam::Float {
                name: "input_scaling",
                low: 0.1,
                high: 5.0,
                log_scale: true,
            });

        Self {
            algorithm: Algorithm::Esn,
            n_features: 0,
            space,
            warmup: 50,
            complexity: 10000,
            seed: 42,
        }
    }

    /// Create a factory for streaming Mamba (selective state space model).
    ///
    /// `d_in` is the number of input features, which is fixed and not
    /// part of the hyperparameter search.
    pub fn mamba(d_in: usize) -> Self {
        let space = ConfigSpace::new()
            .push(HyperParam::Int {
                name: "n_state",
                low: 4,
                high: 64,
            })
            .push(HyperParam::Float {
                name: "forgetting_factor",
                low: 0.95,
                high: 0.9999,
                log_scale: false,
            })
            .push(HyperParam::Int {
                name: "warmup",
                low: 5,
                high: 50,
            });

        Self {
            algorithm: Algorithm::Mamba,
            n_features: d_in,
            space,
            warmup: 10,
            complexity: 4000,
            seed: 42,
        }
    }

    /// Create a factory for streaming linear attention (GLA mode).
    ///
    /// `d_model` is the input feature dimension, which must be divisible
    /// by all candidate `n_heads` values (1, 2, 4, 8).
    pub fn attention(d_model: usize) -> Self {
        let space = ConfigSpace::new()
            .push(HyperParam::Categorical {
                name: "n_heads",
                n_choices: ATTENTION_HEAD_CHOICES.len(),
            })
            .push(HyperParam::Float {
                name: "forgetting_factor",
                low: 0.95,
                high: 0.9999,
                log_scale: false,
            })
            .push(HyperParam::Int {
                name: "warmup",
                low: 5,
                high: 50,
            });

        Self {
            algorithm: Algorithm::Attention,
            n_features: d_model,
            space,
            warmup: 10,
            complexity: 8000,
            seed: 42,
        }
    }

    /// Create a factory for spiking neural networks (e-prop learning).
    ///
    /// Input dimension is auto-detected from the first training sample.
    pub fn spike_net() -> Self {
        let space = ConfigSpace::new()
            .push(HyperParam::Int {
                name: "n_hidden",
                low: 16,
                high: 256,
            })
            .push(HyperParam::Float {
                name: "alpha",
                low: 0.8,
                high: 0.999,
                log_scale: false,
            })
            .push(HyperParam::Float {
                name: "eta",
                low: 0.0001,
                high: 0.01,
                log_scale: true,
            })
            .push(HyperParam::Float {
                name: "v_thr",
                low: 0.2,
                high: 0.8,
                log_scale: false,
            });

        Self {
            algorithm: Algorithm::SpikeNet,
            n_features: 0,
            space,
            warmup: 20,
            complexity: 16000,
            seed: 42,
        }
    }

    // -----------------------------------------------------------------------
    // Builder-style overrides
    // -----------------------------------------------------------------------

    /// Override the default search space.
    pub fn with_space(mut self, space: ConfigSpace) -> Self {
        self.space = space;
        self
    }

    /// Override the default warmup hint.
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    /// Override the default complexity hint.
    pub fn with_complexity(mut self, complexity: usize) -> Self {
        self.complexity = complexity;
        self
    }

    /// Override the default seed for algorithms that use one (ESN, SpikeNet).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns the algorithm variant this factory builds.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }
}

impl ModelFactory for Factory {
    fn config_space(&self) -> ConfigSpace {
        self.space.clone()
    }

    fn name(&self) -> &str {
        match self.algorithm {
            Algorithm::Sgbt => "SGBT",
            Algorithm::Distributional => "Distributional",
            Algorithm::Esn => "ESN",
            Algorithm::Mamba => "Mamba",
            Algorithm::Attention => "Attention",
            Algorithm::SpikeNet => "SpikeNet",
        }
    }

    fn warmup_hint(&self) -> usize {
        self.warmup
    }

    fn complexity_hint(&self) -> usize {
        self.complexity
    }

    fn create(&self, config: &HyperConfig) -> Box<dyn irithyll_core::learner::StreamingLearner> {
        match self.algorithm {
            Algorithm::Sgbt => {
                let learning_rate = config.get(0);
                let n_steps = config.get(1) as usize;
                let max_depth = config.get(2) as usize;
                let n_bins = config.get(3) as usize;
                let lambda = config.get(4);
                let feature_subsample_rate = config.get(5);

                let sgbt_config = SGBTConfig::builder()
                    .learning_rate(learning_rate)
                    .n_steps(n_steps)
                    .max_depth(max_depth)
                    .n_bins(n_bins)
                    .lambda(lambda)
                    .feature_subsample_rate(feature_subsample_rate)
                    .build()
                    .expect("Factory::create(Sgbt): invalid config from search space");

                Box::new(SGBTLearner::from_config(sgbt_config))
            }
            Algorithm::Distributional => {
                let learning_rate = config.get(0);
                let n_steps = config.get(1) as usize;
                let max_depth = config.get(2) as usize;
                let n_bins = config.get(3) as usize;
                let lambda = config.get(4);
                let feature_subsample_rate = config.get(5);

                let sgbt_config = SGBTConfig::builder()
                    .learning_rate(learning_rate)
                    .n_steps(n_steps)
                    .max_depth(max_depth)
                    .n_bins(n_bins)
                    .lambda(lambda)
                    .feature_subsample_rate(feature_subsample_rate)
                    .build()
                    .expect("Factory::create(Distributional): invalid config from search space");

                Box::new(DistributionalSGBT::new(sgbt_config))
            }
            Algorithm::Esn => {
                let n_reservoir = config.get(0) as usize;
                let spectral_radius = config.get(1);
                let leak_rate = config.get(2);
                let input_scaling = config.get(3);

                let esn_config = ESNConfig::builder()
                    .n_reservoir(n_reservoir)
                    .spectral_radius(spectral_radius)
                    .leak_rate(leak_rate)
                    .input_scaling(input_scaling)
                    .seed(self.seed)
                    .build()
                    .expect("Factory::create(Esn): invalid config from search space");

                Box::new(EchoStateNetwork::new(esn_config))
            }
            Algorithm::Mamba => {
                let n_state = config.get(0) as usize;
                let forgetting_factor = config.get(1);
                let warmup = config.get(2) as usize;

                let mamba_config = MambaConfig::builder()
                    .d_in(self.n_features)
                    .n_state(n_state)
                    .forgetting_factor(forgetting_factor)
                    .warmup(warmup)
                    .build()
                    .expect("Factory::create(Mamba): invalid config from search space");

                Box::new(StreamingMamba::new(mamba_config))
            }
            Algorithm::Attention => {
                let head_idx = config.get(0) as usize;
                let n_heads =
                    ATTENTION_HEAD_CHOICES[head_idx.min(ATTENTION_HEAD_CHOICES.len() - 1)];
                let forgetting_factor = config.get(1);
                let warmup = config.get(2) as usize;

                let attn_config = StreamingAttentionConfig::builder()
                    .d_model(self.n_features)
                    .n_heads(n_heads)
                    .mode(AttentionMode::GLA)
                    .forgetting_factor(forgetting_factor)
                    .warmup(warmup)
                    .build()
                    .expect("Factory::create(Attention): invalid config from search space");

                Box::new(StreamingAttentionModel::new(attn_config))
            }
            Algorithm::SpikeNet => {
                let n_hidden = config.get(0) as usize;
                let alpha = config.get(1);
                let eta = config.get(2);
                let v_thr = config.get(3);

                let spike_config = SpikeNetConfig::builder()
                    .n_hidden(n_hidden)
                    .alpha(alpha)
                    .eta(eta)
                    .v_thr(v_thr)
                    .seed(self.seed)
                    .build()
                    .expect("Factory::create(SpikeNet): invalid config from search space");

                Box::new(SpikeNet::new(spike_config))
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::automl::ConfigSampler;
    use irithyll_core::learner::StreamingLearner;

    // ===================================================================
    // Legacy factory tests (deprecated types, kept for backwards compat)
    // ===================================================================

    /// Verify SgbtFactory config space has 6 parameters with correct names.
    #[test]
    fn sgbt_factory_config_space() {
        let factory = SgbtFactory::new(10);
        let space = factory.config_space();
        assert_eq!(space.n_params(), 6, "SGBT should have 6 hyperparameters");
        assert_eq!(
            space.params()[0].name(),
            "learning_rate",
            "first param should be learning_rate"
        );
        assert_eq!(
            space.params()[1].name(),
            "n_steps",
            "second param should be n_steps"
        );
        assert_eq!(
            space.params()[2].name(),
            "max_depth",
            "third param should be max_depth"
        );
        assert_eq!(
            space.params()[3].name(),
            "n_bins",
            "fourth param should be n_bins"
        );
        assert_eq!(
            space.params()[4].name(),
            "lambda",
            "fifth param should be lambda"
        );
        assert_eq!(
            space.params()[5].name(),
            "feature_subsample_rate",
            "sixth param should be feature_subsample_rate"
        );
    }

    /// Verify SgbtFactory creates a model that can train and predict.
    #[test]
    fn sgbt_factory_create_and_predict() {
        let factory = SgbtFactory::new(3);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        model.train(&[1.0, 2.0, 3.0], 4.0);
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.is_finite(),
            "SGBT prediction should be finite, got {pred}"
        );
    }

    /// Verify SgbtFactory name returns "SGBT".
    #[test]
    fn sgbt_factory_name() {
        let factory = SgbtFactory::new(5);
        assert_eq!(factory.name(), "SGBT", "factory name should be SGBT");
    }

    /// Verify EsnFactory config space has 4 parameters with correct names.
    #[test]
    fn esn_factory_config_space() {
        let factory = EsnFactory::new();
        let space = factory.config_space();
        assert_eq!(space.n_params(), 4, "ESN should have 4 hyperparameters");
        assert_eq!(
            space.params()[0].name(),
            "n_reservoir",
            "first param should be n_reservoir"
        );
        assert_eq!(
            space.params()[1].name(),
            "spectral_radius",
            "second param should be spectral_radius"
        );
        assert_eq!(
            space.params()[2].name(),
            "leak_rate",
            "third param should be leak_rate"
        );
        assert_eq!(
            space.params()[3].name(),
            "input_scaling",
            "fourth param should be input_scaling"
        );
    }

    /// Verify EsnFactory creates a model that can train and predict.
    #[test]
    fn esn_factory_create_and_predict() {
        let factory = EsnFactory::with_seed(123);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        // ESN needs warmup samples before reliable predictions.
        for i in 0..60 {
            model.train(&[i as f64 * 0.1], 0.0);
        }
        let pred = model.predict(&[1.0]);
        assert!(
            pred.is_finite(),
            "ESN prediction should be finite, got {pred}"
        );
    }

    /// Verify EsnFactory name returns "ESN".
    #[test]
    fn esn_factory_name() {
        let factory = EsnFactory::new();
        assert_eq!(factory.name(), "ESN", "factory name should be ESN");
    }

    /// Verify MambaFactory config space has 3 parameters with correct names.
    #[test]
    fn mamba_factory_config_space() {
        let factory = MambaFactory::new(4);
        let space = factory.config_space();
        assert_eq!(space.n_params(), 3, "Mamba should have 3 hyperparameters");
        assert_eq!(
            space.params()[0].name(),
            "n_state",
            "first param should be n_state"
        );
        assert_eq!(
            space.params()[1].name(),
            "forgetting_factor",
            "second param should be forgetting_factor"
        );
        assert_eq!(
            space.params()[2].name(),
            "warmup",
            "third param should be warmup"
        );
    }

    /// Verify MambaFactory creates a model that can train and predict.
    #[test]
    fn mamba_factory_create_and_predict() {
        let factory = MambaFactory::new(3);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        model.train(&[1.0, 2.0, 3.0], 4.0);
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.is_finite(),
            "Mamba prediction should be finite, got {pred}"
        );
    }

    /// Verify MambaFactory name returns "Mamba".
    #[test]
    fn mamba_factory_name() {
        let factory = MambaFactory::new(4);
        assert_eq!(factory.name(), "Mamba", "factory name should be Mamba");
    }

    /// Verify AttentionFactory config space has 3 parameters with correct names.
    #[test]
    fn attention_factory_config_space() {
        let factory = AttentionFactory::new(8);
        let space = factory.config_space();
        assert_eq!(
            space.n_params(),
            3,
            "Attention should have 3 hyperparameters"
        );
        assert_eq!(
            space.params()[0].name(),
            "n_heads",
            "first param should be n_heads"
        );
        assert_eq!(
            space.params()[1].name(),
            "forgetting_factor",
            "second param should be forgetting_factor"
        );
        assert_eq!(
            space.params()[2].name(),
            "warmup",
            "third param should be warmup"
        );
    }

    /// Verify AttentionFactory creates a model that can train and predict.
    #[test]
    fn attention_factory_create_and_predict() {
        // d_model=8 is divisible by all candidate n_heads (1, 2, 4, 8).
        let factory = AttentionFactory::new(8);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        model.train(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 5.0);
        let pred = model.predict(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(
            pred.is_finite(),
            "Attention prediction should be finite, got {pred}"
        );
    }

    /// Verify AttentionFactory name returns "Attention".
    #[test]
    fn attention_factory_name() {
        let factory = AttentionFactory::new(8);
        assert_eq!(
            factory.name(),
            "Attention",
            "factory name should be Attention"
        );
    }

    /// Verify SpikeNetFactory config space has 4 parameters with correct names.
    #[test]
    fn spikenet_factory_config_space() {
        let factory = SpikeNetFactory::new();
        let space = factory.config_space();
        assert_eq!(
            space.n_params(),
            4,
            "SpikeNet should have 4 hyperparameters"
        );
        assert_eq!(
            space.params()[0].name(),
            "n_hidden",
            "first param should be n_hidden"
        );
        assert_eq!(
            space.params()[1].name(),
            "alpha",
            "second param should be alpha"
        );
        assert_eq!(space.params()[2].name(), "eta", "third param should be eta");
        assert_eq!(
            space.params()[3].name(),
            "v_thr",
            "fourth param should be v_thr"
        );
    }

    /// Verify SpikeNetFactory creates a model that can train and predict.
    #[test]
    fn spikenet_factory_create_and_predict() {
        let factory = SpikeNetFactory::with_seed(99);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        model.train(&[0.5, -0.3], 1.0);
        let pred = model.predict(&[0.5, -0.3]);
        assert!(
            pred.is_finite(),
            "SpikeNet prediction should be finite, got {pred}"
        );
    }

    /// Verify SpikeNetFactory name returns "SpikeNet".
    #[test]
    fn spikenet_factory_name() {
        let factory = SpikeNetFactory::new();
        assert_eq!(
            factory.name(),
            "SpikeNet",
            "factory name should be SpikeNet"
        );
    }

    /// Verify all factories implement Send + Sync (required by ModelFactory).
    #[test]
    fn factories_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SgbtFactory>();
        assert_send_sync::<EsnFactory>();
        assert_send_sync::<MambaFactory>();
        assert_send_sync::<AttentionFactory>();
        assert_send_sync::<SpikeNetFactory>();
        assert_send_sync::<Factory>();
    }

    /// Verify factories can be used as trait objects.
    #[test]
    fn factory_as_trait_object() {
        let factory: Box<dyn ModelFactory> = Box::new(SgbtFactory::new(5));
        let space = factory.config_space();
        assert_eq!(
            space.n_params(),
            6,
            "trait object config_space should return 6 params for SGBT"
        );
        assert_eq!(factory.name(), "SGBT", "trait object name should be SGBT");
    }

    /// Verify EsnFactory default matches new().
    #[test]
    fn esn_factory_default() {
        let a = EsnFactory::new();
        let b = EsnFactory::default();
        assert_eq!(a.seed, b.seed, "default and new should produce same seed");
    }

    /// Verify SpikeNetFactory default matches new().
    #[test]
    fn spikenet_factory_default() {
        let a = SpikeNetFactory::new();
        let b = SpikeNetFactory::default();
        assert_eq!(a.seed, b.seed, "default and new should produce same seed");
    }

    // ===================================================================
    // Unified Factory tests
    // ===================================================================

    /// Factory::sgbt creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_sgbt() {
        let factory = Factory::sgbt(3);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        model.train(&[1.0, 2.0, 3.0], 4.0);
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.is_finite(),
            "unified SGBT prediction should be finite, got {pred}"
        );
    }

    /// Factory::esn creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_esn() {
        let factory = Factory::esn();
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..100 {
            model.train(&[i as f64 * 0.1], 0.0);
        }
        let pred = model.predict(&[1.0]);
        assert!(
            pred.is_finite(),
            "unified ESN prediction should be finite, got {pred}"
        );
    }

    /// Factory::distributional creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_distributional() {
        let factory = Factory::distributional(3);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        model.train(&[1.0, 2.0, 3.0], 4.0);
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.is_finite(),
            "unified Distributional prediction should be finite, got {pred}"
        );
    }

    /// Factory::mamba creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_mamba() {
        let factory = Factory::mamba(3);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        model.train(&[1.0, 2.0, 3.0], 4.0);
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.is_finite(),
            "unified Mamba prediction should be finite, got {pred}"
        );
    }

    /// Factory::spike_net creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_spike_net() {
        let factory = Factory::spike_net();
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        model.train(&[0.5, -0.3], 1.0);
        let pred = model.predict(&[0.5, -0.3]);
        assert!(
            pred.is_finite(),
            "unified SpikeNet prediction should be finite, got {pred}"
        );
    }

    /// Factory::attention creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_attention() {
        let factory = Factory::attention(8);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        model.train(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 5.0);
        let pred = model.predict(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(
            pred.is_finite(),
            "unified Attention prediction should be finite, got {pred}"
        );
    }

    /// Builder-style overrides apply correctly.
    #[test]
    fn unified_factory_with_overrides() {
        let factory = Factory::sgbt(3).with_warmup(50).with_complexity(200);
        assert_eq!(
            factory.warmup_hint(),
            50,
            "with_warmup should override warmup_hint"
        );
        assert_eq!(
            factory.complexity_hint(),
            200,
            "with_complexity should override complexity_hint"
        );
    }

    /// Each algorithm returns the expected complexity_hint.
    #[test]
    fn unified_factory_complexity_hint() {
        assert_eq!(
            Factory::sgbt(3).complexity_hint(),
            500,
            "SGBT complexity should be 500"
        );
        assert_eq!(
            Factory::distributional(3).complexity_hint(),
            1000,
            "Distributional complexity should be 1000"
        );
        assert_eq!(
            Factory::esn().complexity_hint(),
            10000,
            "ESN complexity should be 10000"
        );
        assert_eq!(
            Factory::mamba(3).complexity_hint(),
            4000,
            "Mamba complexity should be 4000"
        );
        assert_eq!(
            Factory::attention(8).complexity_hint(),
            8000,
            "Attention complexity should be 8000"
        );
        assert_eq!(
            Factory::spike_net().complexity_hint(),
            16000,
            "SpikeNet complexity should be 16000"
        );
    }

    /// Each algorithm returns the expected name.
    #[test]
    fn unified_factory_names() {
        assert_eq!(Factory::sgbt(3).name(), "SGBT", "SGBT name mismatch");
        assert_eq!(
            Factory::distributional(3).name(),
            "Distributional",
            "Distributional name mismatch"
        );
        assert_eq!(Factory::esn().name(), "ESN", "ESN name mismatch");
        assert_eq!(Factory::mamba(3).name(), "Mamba", "Mamba name mismatch");
        assert_eq!(
            Factory::attention(8).name(),
            "Attention",
            "Attention name mismatch"
        );
        assert_eq!(
            Factory::spike_net().name(),
            "SpikeNet",
            "SpikeNet name mismatch"
        );
    }

    /// Factory works as a ModelFactory inside auto_tune().
    #[test]
    fn unified_factory_in_auto_tuner() {
        let mut tuner = crate::auto_tune(Factory::sgbt(3));
        tuner.train(&[1.0, 2.0, 3.0], 4.0);
        let pred = tuner.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.is_finite(),
            "auto_tune with unified Factory should produce finite prediction, got {pred}"
        );
    }
}
