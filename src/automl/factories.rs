//! Model factory for AutoML hyperparameter search.
//!
//! The [`Factory`] type implements [`ModelFactory`] for all streaming learner
//! algorithms, defining hyperparameter search spaces and constructing instances
//! from sampled configurations.

use crate::automl::{ConfigSpace, HyperConfig, HyperParam, ModelFactory};
use crate::ensemble::config::SGBTConfig;
use crate::ensemble::distributional::DistributionalSGBT;
use crate::learner::SGBTLearner;
use crate::projection::{ProjectedLearner, ProjectionConfig};
use crate::reservoir::{ESNConfig, EchoStateNetwork};
use crate::snn::{SpikeNet, SpikeNetConfig};
use crate::ssm::{MambaConfig, MambaVersion, StreamingMamba};
use irithyll_core::attention::AttentionMode;

use crate::attention::{StreamingAttentionConfig, StreamingAttentionModel};

/// Number of attention heads choices for the categorical parameter.
const ATTENTION_HEAD_CHOICES: [usize; 4] = [1, 2, 4, 8];

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
    /// Streaming KAN (B-spline edge activations).
    Kan,
    /// Streaming TTT (test-time training with fast weights).
    Ttt,
    /// Streaming Mamba-3 (MIMO groups, complex states, trapezoidal discretization).
    Mamba3,
    /// DeltaProduct attention (product of Householder delta rules).
    DeltaProduct,
    /// RWKV-7 attention (vector-gated delta rule with DPLR transitions).
    Rwkv7,
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
    accuracy_based_pruning: bool,
    proactive_prune_interval: Option<u64>,
    prune_half_life: Option<usize>,
    /// Optional PAST projection wrapping: (d_in, config).
    /// When set, `create()` wraps the inner model in a [`ProjectedLearner`].
    projection: Option<(usize, ProjectionConfig)>,
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
            })
            .push(HyperParam::Int {
                name: "grace_period",
                low: 3,
                high: 200,
            });

        Self {
            algorithm: Algorithm::Sgbt,
            n_features,
            space,
            warmup: 0,
            complexity: 500,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
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
            })
            .push(HyperParam::Int {
                name: "grace_period",
                low: 3,
                high: 200,
            });

        Self {
            algorithm: Algorithm::Distributional,
            n_features,
            space,
            warmup: 0,
            complexity: 1000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
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
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
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
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for streaming Mamba-3 (MIMO groups, complex states).
    ///
    /// `d_in` is the number of input features, which is fixed and not
    /// part of the hyperparameter search.
    ///
    /// # Config Space (4 params)
    ///
    /// | Index | Name | Type | Range | Scale |
    /// |-------|------|------|-------|-------|
    /// | 0 | `n_state` | Int | [4, 64] | -- |
    /// | 1 | `n_groups` | Int | [1, d_in/2] | -- |
    /// | 2 | `forgetting_factor` | Float | [0.95, 0.9999] | linear |
    /// | 3 | `warmup` | Int | [5, 50] | -- |
    pub fn mamba3(d_in: usize) -> Self {
        let max_groups = (d_in / 2).max(1);
        let space = ConfigSpace::new()
            .push(HyperParam::Int {
                name: "n_state",
                low: 4,
                high: 64,
            })
            .push(HyperParam::Int {
                name: "n_groups",
                low: 1,
                high: max_groups as i64,
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
            algorithm: Algorithm::Mamba3,
            n_features: d_in,
            space,
            warmup: 10,
            complexity: 5000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
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
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for DeltaProduct attention (product of Householder delta rules).
    ///
    /// `d_model` is the input feature dimension, which must be divisible
    /// by all candidate `n_heads` values (1, 2, 4, 8).
    ///
    /// # Config Space (4 params)
    ///
    /// | Index | Name | Type | Range | Scale |
    /// |-------|------|------|-------|-------|
    /// | 0 | `n_heads` | Int | [1, 8] | -- |
    /// | 1 | `n_compositions` | Int | [1, 4] | -- |
    /// | 2 | `forgetting_factor` | Float | [0.95, 0.9999] | log |
    /// | 3 | `warmup` | Int | [5, 50] | -- |
    pub fn delta_product(d_model: usize) -> Self {
        let space = ConfigSpace::new()
            .push(HyperParam::Int {
                name: "n_heads",
                low: 1,
                high: 8,
            })
            .push(HyperParam::Int {
                name: "n_compositions",
                low: 1,
                high: 4,
            })
            .push(HyperParam::Float {
                name: "forgetting_factor",
                low: 0.95,
                high: 0.9999,
                log_scale: true,
            })
            .push(HyperParam::Int {
                name: "warmup",
                low: 5,
                high: 50,
            });

        Self {
            algorithm: Algorithm::DeltaProduct,
            n_features: d_model,
            space,
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
    ///
    /// `d_model` is the input feature dimension, which must be divisible
    /// by all candidate `n_heads` values (1, 2, 4, 8).
    ///
    /// # Config Space (3 params)
    ///
    /// | Index | Name | Type | Range | Scale |
    /// |-------|------|------|-------|-------|
    /// | 0 | `n_heads` | Int | [1, 8] | -- |
    /// | 1 | `forgetting_factor` | Float | [0.95, 0.9999] | log |
    /// | 2 | `warmup` | Int | [5, 50] | -- |
    pub fn rwkv7(d_model: usize) -> Self {
        let space = ConfigSpace::new()
            .push(HyperParam::Int {
                name: "n_heads",
                low: 1,
                high: 8,
            })
            .push(HyperParam::Float {
                name: "forgetting_factor",
                low: 0.95,
                high: 0.9999,
                log_scale: true,
            })
            .push(HyperParam::Int {
                name: "warmup",
                low: 5,
                high: 50,
            });

        Self {
            algorithm: Algorithm::Rwkv7,
            n_features: d_model,
            space,
            warmup: 10,
            complexity: 5000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
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
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
        }
    }

    /// Create a factory for streaming KAN.
    ///
    /// # Config Space (4 params)
    /// | Index | Name | Type | Range |
    /// |-------|------|------|-------|
    /// | 0 | `hidden_size` | Int | [4, 32] |
    /// | 1 | `grid_size` | Int | [3, 10] |
    /// | 2 | `lr` | Float | [0.001, 0.1] log |
    /// | 3 | `spline_order` | Int | [2, 4] |
    pub fn kan(n_features: usize) -> Self {
        let space = ConfigSpace::new()
            .push(HyperParam::Int {
                name: "hidden_size",
                low: 4,
                high: 32,
            })
            .push(HyperParam::Int {
                name: "grid_size",
                low: 3,
                high: 10,
            })
            .push(HyperParam::Float {
                name: "lr",
                low: 0.001,
                high: 0.1,
                log_scale: true,
            })
            .push(HyperParam::Int {
                name: "spline_order",
                low: 2,
                high: 4,
            });

        Self {
            algorithm: Algorithm::Kan,
            n_features,
            space,
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
    ///
    /// # Config Space (3 params)
    /// | Index | Name | Type | Range |
    /// |-------|------|------|-------|
    /// | 0 | `d_model` | Int | [8, 64] |
    /// | 1 | `eta` | Float | [0.001, 0.1] log |
    /// | 2 | `alpha` | Float | [0.0, 0.01] linear |
    pub fn ttt(n_features: usize) -> Self {
        let space = ConfigSpace::new()
            .push(HyperParam::Int {
                name: "d_model",
                low: 8,
                high: 64,
            })
            .push(HyperParam::Float {
                name: "eta",
                low: 0.001,
                high: 0.1,
                log_scale: true,
            })
            .push(HyperParam::Float {
                name: "alpha",
                low: 0.0,
                high: 0.01,
                log_scale: false,
            });

        Self {
            algorithm: Algorithm::Ttt,
            n_features,
            space,
            warmup: 10,
            complexity: 3000,
            seed: 42,
            accuracy_based_pruning: false,
            proactive_prune_interval: None,
            prune_half_life: None,
            projection: None,
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

    /// Enable accuracy-based pruning for SGBT/Distributional factories.
    ///
    /// When enabled, proactive pruning replaces the tree with the most negative
    /// contribution alignment instead of the tree with lowest prediction variance.
    /// Has no effect on non-tree algorithms.
    pub fn with_accuracy_based_pruning(mut self, enabled: bool) -> Self {
        self.accuracy_based_pruning = enabled;
        self
    }

    /// Set the proactive prune interval for SGBT/Distributional factories.
    ///
    /// Every `interval` samples, the worst-contributing tree is replaced.
    /// Has no effect on non-tree algorithms. `None` (default) disables proactive pruning.
    pub fn with_proactive_prune_interval(mut self, interval: u64) -> Self {
        self.proactive_prune_interval = Some(interval);
        self
    }

    /// Set the prune half-life for the contribution accuracy EWMA.
    ///
    /// Overrides the automatic derivation used by proactive pruning.
    /// Has no effect on non-tree algorithms.
    pub fn with_prune_half_life(mut self, hl: usize) -> Self {
        self.prune_half_life = Some(hl);
        self
    }

    /// Returns the algorithm variant this factory builds.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Override the bounds of a named hyperparameter in the config space.
    ///
    /// Panics if the parameter name is not found in this factory's space.
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll::automl::Factory;
    ///
    /// let factory = Factory::sgbt(4)
    ///     .with_config_range("learning_rate", 0.01, 0.1)
    ///     .with_config_range("n_steps", 20.0, 100.0);
    /// ```
    pub fn with_config_range(mut self, name: &str, low: f64, high: f64) -> Self {
        self.space.set_range(name, low, high);
        self
    }

    // -----------------------------------------------------------------------
    // Projection wrapping
    // -----------------------------------------------------------------------

    /// Wrap the factory's output model in a PAST-based projection learner.
    ///
    /// The projection reduces the input to `rank` dimensions using online
    /// subspace tracking (PAST algorithm). The wrapped model sees
    /// `rank`-dimensional features instead of the original input.
    ///
    /// For algorithms that require an explicit input dimension (Mamba, Attention,
    /// KAN, TTT), this method also resets the inner model's `n_features` to
    /// `rank` so the inner model is configured for the projected input size.
    ///
    /// # Arguments
    /// * `d_in` -- original input dimension (before projection)
    /// * `rank` -- projection dimension (what the inner model sees)
    /// * `lambda` -- PAST forgetting factor (0.999 typical)
    pub fn with_projection(mut self, d_in: usize, rank: usize, lambda: f64) -> Self {
        let config = ProjectionConfig {
            rank,
            lambda,
            ..ProjectionConfig::default()
        };
        self.projection = Some((d_in, config));
        // Inner model sees rank-dimensional features, not d_in.
        self.n_features = rank;
        self
    }

    /// Wrap with projection, providing a full [`ProjectionConfig`].
    ///
    /// Like [`with_projection`](Self::with_projection) but allows control
    /// over all PAST parameters (delta, warmup, seed).
    pub fn with_projection_config(mut self, d_in: usize, config: ProjectionConfig) -> Self {
        let rank = config.rank;
        self.projection = Some((d_in, config));
        self.n_features = rank;
        self
    }

    // -----------------------------------------------------------------------
    // Projected convenience constructors
    // -----------------------------------------------------------------------

    /// Create a projected Mamba factory.
    ///
    /// Equivalent to `Factory::mamba(rank).with_projection(d_in, rank, 0.999)`.
    /// The inner Mamba sees `rank`-dimensional projected features.
    pub fn projected_mamba(d_in: usize, rank: usize) -> Self {
        Factory::mamba(rank).with_projection(d_in, rank, 0.999)
    }

    /// Create a projected Mamba-3 factory.
    ///
    /// Equivalent to `Factory::mamba3(rank).with_projection(d_in, rank, 0.999)`.
    /// The inner Mamba-3 sees `rank`-dimensional projected features.
    pub fn projected_mamba3(d_in: usize, rank: usize) -> Self {
        Factory::mamba3(rank).with_projection(d_in, rank, 0.999)
    }

    /// Create a projected TTT factory.
    ///
    /// Equivalent to `Factory::ttt(rank).with_projection(d_in, rank, 0.999)`.
    /// The inner TTT sees `rank`-dimensional projected features.
    pub fn projected_ttt(d_in: usize, rank: usize) -> Self {
        Factory::ttt(rank).with_projection(d_in, rank, 0.999)
    }

    /// Create a projected KAN factory.
    ///
    /// Equivalent to `Factory::kan(rank).with_projection(d_in, rank, 0.999)`.
    /// The inner KAN sees `rank`-dimensional projected features.
    pub fn projected_kan(d_in: usize, rank: usize) -> Self {
        Factory::kan(rank).with_projection(d_in, rank, 0.999)
    }

    /// Create a projected Attention factory.
    ///
    /// Equivalent to `Factory::attention(rank).with_projection(d_in, rank, 0.999)`.
    /// `rank` must be divisible by all candidate `n_heads` values (1, 2, 4, 8).
    pub fn projected_attention(d_in: usize, rank: usize) -> Self {
        Factory::attention(rank).with_projection(d_in, rank, 0.999)
    }

    /// Create a projected DeltaProduct factory.
    ///
    /// Equivalent to `Factory::delta_product(rank).with_projection(d_in, rank, 0.999)`.
    /// The inner DeltaProduct sees `rank`-dimensional projected features.
    pub fn projected_delta_product(d_in: usize, rank: usize) -> Self {
        Factory::delta_product(rank).with_projection(d_in, rank, 0.999)
    }

    /// Create a projected RWKV-7 factory.
    ///
    /// Equivalent to `Factory::rwkv7(rank).with_projection(d_in, rank, 0.999)`.
    /// The inner RWKV-7 sees `rank`-dimensional projected features.
    pub fn projected_rwkv7(d_in: usize, rank: usize) -> Self {
        Factory::rwkv7(rank).with_projection(d_in, rank, 0.999)
    }

    /// Create a projected ESN factory.
    ///
    /// Equivalent to `Factory::esn().with_projection(d_in, rank, 0.999)`.
    /// The inner ESN sees `rank`-dimensional projected features.
    pub fn projected_esn(d_in: usize, rank: usize) -> Self {
        Factory::esn().with_projection(d_in, rank, 0.999)
    }

    /// Create a projected SGBT factory.
    ///
    /// Equivalent to `Factory::sgbt(rank).with_projection(d_in, rank, 0.999)`.
    /// The inner SGBT sees `rank`-dimensional projected features.
    pub fn projected_sgbt(d_in: usize, rank: usize) -> Self {
        Factory::sgbt(rank).with_projection(d_in, rank, 0.999)
    }
}

impl ModelFactory for Factory {
    fn config_space(&self) -> ConfigSpace {
        self.space.clone()
    }

    fn name(&self) -> &str {
        if self.projection.is_some() {
            match self.algorithm {
                Algorithm::Sgbt => "Projected<SGBT>",
                Algorithm::Distributional => "Projected<Distributional>",
                Algorithm::Esn => "Projected<ESN>",
                Algorithm::Mamba => "Projected<Mamba>",
                Algorithm::Mamba3 => "Projected<Mamba3>",
                Algorithm::Attention => "Projected<Attention>",
                Algorithm::SpikeNet => "Projected<SpikeNet>",
                Algorithm::Kan => "Projected<KAN>",
                Algorithm::Ttt => "Projected<TTT>",
                Algorithm::DeltaProduct => "Projected<DeltaProduct>",
                Algorithm::Rwkv7 => "Projected<RWKV7>",
            }
        } else {
            match self.algorithm {
                Algorithm::Sgbt => "SGBT",
                Algorithm::Distributional => "Distributional",
                Algorithm::Esn => "ESN",
                Algorithm::Mamba => "Mamba",
                Algorithm::Mamba3 => "Mamba3",
                Algorithm::Attention => "Attention",
                Algorithm::SpikeNet => "SpikeNet",
                Algorithm::Kan => "KAN",
                Algorithm::Ttt => "TTT",
                Algorithm::DeltaProduct => "DeltaProduct",
                Algorithm::Rwkv7 => "RWKV7",
            }
        }
    }

    fn warmup_hint(&self) -> usize {
        self.warmup
    }

    fn complexity_hint(&self) -> usize {
        self.complexity
    }

    fn n_features_hint(&self) -> usize {
        self.n_features
    }

    fn create(&self, config: &HyperConfig) -> Box<dyn irithyll_core::learner::StreamingLearner> {
        let inner: Box<dyn irithyll_core::learner::StreamingLearner> = match self.algorithm {
            Algorithm::Sgbt => {
                let learning_rate = config.get(0);
                let n_steps = config.get(1) as usize;
                let max_depth = config.get(2) as usize;
                let n_bins = config.get(3) as usize;
                let lambda = config.get(4);
                let feature_subsample_rate = config.get(5);
                let grace_period = config.get(6) as usize;

                let mut builder = SGBTConfig::builder()
                    .learning_rate(learning_rate)
                    .n_steps(n_steps)
                    .max_depth(max_depth)
                    .n_bins(n_bins)
                    .lambda(lambda)
                    .feature_subsample_rate(feature_subsample_rate)
                    .grace_period(grace_period)
                    .error_weight_alpha(0.01)
                    .shadow_warmup(100)
                    .accuracy_based_pruning(self.accuracy_based_pruning);
                if let Some(interval) = self.proactive_prune_interval {
                    builder = builder.proactive_prune_interval(interval);
                }
                if let Some(hl) = self.prune_half_life {
                    builder = builder.prune_half_life(hl);
                }
                let sgbt_config = builder
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
                let grace_period = config.get(6) as usize;

                let mut builder = SGBTConfig::builder()
                    .learning_rate(learning_rate)
                    .n_steps(n_steps)
                    .max_depth(max_depth)
                    .n_bins(n_bins)
                    .lambda(lambda)
                    .feature_subsample_rate(feature_subsample_rate)
                    .grace_period(grace_period)
                    .error_weight_alpha(0.01)
                    .shadow_warmup(100)
                    .accuracy_based_pruning(self.accuracy_based_pruning);
                if let Some(interval) = self.proactive_prune_interval {
                    builder = builder.proactive_prune_interval(interval);
                }
                if let Some(hl) = self.prune_half_life {
                    builder = builder.prune_half_life(hl);
                }
                let sgbt_config = builder
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
            Algorithm::Mamba3 => {
                let n_state = config.get(0) as usize;
                let mut n_groups = config.get(1) as usize;
                let forgetting_factor = config.get(2);
                let warmup = config.get(3) as usize;

                // Ensure n_groups divides d_in evenly; snap to nearest valid divisor.
                let d_in = self.n_features;
                if d_in > 0 && n_groups > 0 && d_in % n_groups != 0 {
                    // Find the nearest divisor of d_in that is <= n_groups.
                    n_groups = (1..=n_groups).rev().find(|&g| d_in % g == 0).unwrap_or(1);
                }

                let mamba_config = MambaConfig::builder()
                    .d_in(d_in)
                    .n_state(n_state)
                    .version(MambaVersion::V3)
                    .n_groups(n_groups.max(1))
                    .forgetting_factor(forgetting_factor)
                    .warmup(warmup)
                    .build()
                    .expect("Factory::create(Mamba3): invalid config from search space");

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
            Algorithm::Kan => {
                let hidden_size = config.get(0) as usize;
                let grid_size = config.get(1) as usize;
                let lr = config.get(2);
                let spline_order = config.get(3) as usize;

                let kan_config = crate::kan::KANConfig::builder()
                    .layer_sizes(vec![self.n_features, hidden_size, 1])
                    .grid_size(grid_size)
                    .lr(lr)
                    .spline_order(spline_order)
                    .seed(self.seed)
                    .build()
                    .expect("Factory::create(Kan): invalid config");

                Box::new(crate::kan::StreamingKAN::new(kan_config))
            }
            Algorithm::Ttt => {
                let d_model = config.get(0) as usize;
                let eta = config.get(1);
                let alpha = config.get(2);

                let ttt_config = crate::ttt::TTTConfig::builder()
                    .d_model(d_model)
                    .eta(eta)
                    .alpha(alpha)
                    .warmup(self.warmup)
                    .seed(self.seed)
                    .build()
                    .expect("Factory::create(Ttt): invalid config");

                Box::new(crate::ttt::StreamingTTT::new(ttt_config))
            }
            Algorithm::DeltaProduct => {
                let n_heads = (config.get(0) as usize).max(1);
                let n_compositions = (config.get(1) as usize).max(1);
                let forgetting_factor = config.get(2);
                let warmup = config.get(3) as usize;

                let attn_config = StreamingAttentionConfig::builder()
                    .d_model(self.n_features)
                    .n_heads(n_heads)
                    .mode(AttentionMode::DeltaProduct { n_compositions })
                    .forgetting_factor(forgetting_factor)
                    .warmup(warmup)
                    .build()
                    .expect("Factory::create(DeltaProduct): invalid config from search space");

                Box::new(StreamingAttentionModel::new(attn_config))
            }
            Algorithm::Rwkv7 => {
                let n_heads = (config.get(0) as usize).max(1);
                let forgetting_factor = config.get(1);
                let warmup = config.get(2) as usize;

                let attn_config = StreamingAttentionConfig::builder()
                    .d_model(self.n_features)
                    .n_heads(n_heads)
                    .mode(AttentionMode::RWKV7)
                    .forgetting_factor(forgetting_factor)
                    .warmup(warmup)
                    .build()
                    .expect("Factory::create(Rwkv7): invalid config from search space");

                Box::new(StreamingAttentionModel::new(attn_config))
            }
        };

        // Wrap in ProjectedLearner if projection is configured.
        if let Some((d_in, ref proj_config)) = self.projection {
            Box::new(ProjectedLearner::new(inner, d_in, proj_config.clone()))
        } else {
            inner
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automl::ConfigSampler;
    use irithyll_core::learner::StreamingLearner;

    /// Verify Factory implements Send + Sync (required by ModelFactory).
    #[test]
    fn factory_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Factory>();
    }

    /// Verify Factory can be used as a trait object.
    #[test]
    fn factory_as_trait_object() {
        let factory: Box<dyn ModelFactory> = Box::new(Factory::sgbt(5));
        let space = factory.config_space();
        assert_eq!(
            space.n_params(),
            7,
            "trait object config_space should return 7 params for SGBT"
        );
        assert_eq!(factory.name(), "SGBT", "trait object name should be SGBT");
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
        assert_eq!(
            Factory::kan(3).complexity_hint(),
            2000,
            "KAN complexity should be 2000"
        );
        assert_eq!(
            Factory::ttt(3).complexity_hint(),
            3000,
            "TTT complexity should be 3000"
        );
        assert_eq!(
            Factory::mamba3(8).complexity_hint(),
            5000,
            "Mamba3 complexity should be 5000"
        );
        assert_eq!(
            Factory::delta_product(8).complexity_hint(),
            8000,
            "DeltaProduct complexity should be 8000"
        );
        assert_eq!(
            Factory::rwkv7(8).complexity_hint(),
            5000,
            "RWKV7 complexity should be 5000"
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
        assert_eq!(Factory::mamba3(8).name(), "Mamba3", "Mamba3 name mismatch");
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
        assert_eq!(Factory::kan(3).name(), "KAN", "KAN name mismatch");
        assert_eq!(Factory::ttt(3).name(), "TTT", "TTT name mismatch");
        assert_eq!(
            Factory::delta_product(8).name(),
            "DeltaProduct",
            "DeltaProduct name mismatch"
        );
        assert_eq!(Factory::rwkv7(8).name(), "RWKV7", "RWKV7 name mismatch");
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

    /// Factory::kan creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_kan() {
        let factory = Factory::kan(3);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..50 {
            let x = [i as f64 * 0.1, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1] - x[2];
            model.train(&x, y);
        }
        let pred = model.predict(&[0.5, 0.5_f64.sin(), 0.5_f64.cos()]);
        assert!(
            pred.is_finite(),
            "unified KAN prediction should be finite, got {pred}"
        );
    }

    /// Factory::ttt creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_ttt() {
        let factory = Factory::ttt(3);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..50 {
            let x = [i as f64 * 0.1, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1] - x[2];
            model.train(&x, y);
        }
        let pred = model.predict(&[0.5, 0.5_f64.sin(), 0.5_f64.cos()]);
        assert!(
            pred.is_finite(),
            "unified TTT prediction should be finite, got {pred}"
        );
    }

    /// Factory::kan works inside auto_tune().
    #[test]
    fn kan_in_auto_tuner() {
        let mut tuner = crate::auto_tune(Factory::kan(3));
        for i in 0..200 {
            let x = [i as f64 * 0.01, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 3.0 + x[1];
            tuner.train(&x, y);
        }
        let pred = tuner.predict(&[0.5, 0.5_f64.sin(), 0.5_f64.cos()]);
        assert!(
            pred.is_finite(),
            "auto_tune with KAN should produce finite prediction, got {pred}"
        );
    }

    /// Factory::ttt works inside auto_tune().
    #[test]
    fn ttt_in_auto_tuner() {
        let mut tuner = crate::auto_tune(Factory::ttt(3));
        for i in 0..200 {
            let x = [i as f64 * 0.01, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 3.0 + x[1];
            tuner.train(&x, y);
        }
        let pred = tuner.predict(&[0.5, 0.5_f64.sin(), 0.5_f64.cos()]);
        assert!(
            pred.is_finite(),
            "auto_tune with TTT should produce finite prediction, got {pred}"
        );
    }

    /// KAN expert works inside NeuralMoE.
    #[test]
    fn kan_in_neural_moe() {
        use crate::kan::{KANConfig, StreamingKAN};
        use crate::moe::NeuralMoE;

        let kan = StreamingKAN::new(
            KANConfig::builder()
                .layer_sizes(vec![3, 8, 1])
                .lr(0.01)
                .build()
                .unwrap(),
        );
        let kan2 = StreamingKAN::new(
            KANConfig::builder()
                .layer_sizes(vec![3, 12, 1])
                .lr(0.005)
                .build()
                .unwrap(),
        );

        let mut moe = NeuralMoE::builder()
            .expert_with_warmup(kan, 20)
            .expert_with_warmup(kan2, 20)
            .build();

        for i in 0..100 {
            let x = [i as f64 * 0.1, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 2.0 + x[1];
            moe.train(&x, y);
        }
        let pred = moe.predict(&[0.5, 0.5_f64.sin(), 0.5_f64.cos()]);
        assert!(
            pred.is_finite(),
            "NeuralMoE with KAN experts should produce finite prediction, got {pred}"
        );
    }

    /// Multi-factory racing with SGBT + KAN + TTT.
    #[test]
    fn multi_factory_with_kan_ttt() {
        let mut tuner = crate::automl::AutoTuner::builder()
            .factory(Factory::sgbt(3))
            .add_factory(Factory::kan(3))
            .add_factory(Factory::ttt(3))
            .build();

        for i in 0..200 {
            let x = [i as f64 * 0.01, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 3.0 + x[1];
            tuner.train(&x, y);
        }
        let pred = tuner.predict(&[0.5, 0.5_f64.sin(), 0.5_f64.cos()]);
        assert!(
            pred.is_finite(),
            "multi-factory racing (SGBT+KAN+TTT) should produce finite prediction, got {pred}"
        );
    }

    // ===================================================================
    // Projected factory tests
    // ===================================================================

    /// Factory::projected_mamba creates a projected model that trains and predicts.
    #[test]
    fn projected_mamba_factory_create_and_predict() {
        let factory = Factory::projected_mamba(8, 4);
        assert_eq!(
            factory.name(),
            "Projected<Mamba>",
            "projected mamba factory name should include Projected<>"
        );
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        // Feed 8-dim input; inner Mamba sees 4-dim projected features.
        for i in 0..100 {
            let x: Vec<f64> = (0..8).map(|j| (i * j) as f64 * 0.01).collect();
            model.train(&x, i as f64 * 0.1);
        }
        let x: Vec<f64> = (0..8).map(|j| j as f64 * 0.05).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "projected Mamba prediction should be finite, got {pred}"
        );
    }

    /// Factory::projected_ttt creates a projected model that trains and predicts.
    #[test]
    fn projected_ttt_factory_create_and_predict() {
        let factory = Factory::projected_ttt(8, 4);
        assert_eq!(
            factory.name(),
            "Projected<TTT>",
            "projected TTT factory name should include Projected<>"
        );
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..100 {
            let x: Vec<f64> = (0..8).map(|j| (i * j) as f64 * 0.01).collect();
            model.train(&x, i as f64 * 0.1);
        }
        let x: Vec<f64> = (0..8).map(|j| j as f64 * 0.05).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "projected TTT prediction should be finite, got {pred}"
        );
    }

    /// Factory::projected_kan creates a projected model that trains and predicts.
    #[test]
    fn projected_kan_factory_create_and_predict() {
        let factory = Factory::projected_kan(8, 4);
        assert_eq!(
            factory.name(),
            "Projected<KAN>",
            "projected KAN factory name should include Projected<>"
        );
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..100 {
            let x: Vec<f64> = (0..8).map(|j| (i * j) as f64 * 0.01).collect();
            model.train(&x, i as f64 * 0.1);
        }
        let x: Vec<f64> = (0..8).map(|j| j as f64 * 0.05).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "projected KAN prediction should be finite, got {pred}"
        );
    }

    /// Factory::projected_sgbt creates a projected model that trains and predicts.
    #[test]
    fn projected_sgbt_factory_create_and_predict() {
        let factory = Factory::projected_sgbt(8, 4);
        assert_eq!(
            factory.name(),
            "Projected<SGBT>",
            "projected SGBT factory name should include Projected<>"
        );
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..100 {
            let x: Vec<f64> = (0..8).map(|j| (i * j) as f64 * 0.01).collect();
            model.train(&x, i as f64 * 0.1);
        }
        let x: Vec<f64> = (0..8).map(|j| j as f64 * 0.05).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "projected SGBT prediction should be finite, got {pred}"
        );
    }

    /// with_projection builder applies to any algorithm.
    #[test]
    fn with_projection_builder_on_esn() {
        let factory = Factory::esn().with_projection(10, 4, 0.998);
        assert_eq!(
            factory.name(),
            "Projected<ESN>",
            "with_projection on ESN should give Projected<ESN> name"
        );
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..100 {
            let x: Vec<f64> = (0..10).map(|j| (i * j) as f64 * 0.01).collect();
            model.train(&x, i as f64 * 0.1);
        }
        let x: Vec<f64> = (0..10).map(|j| j as f64 * 0.05).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "projected ESN prediction should be finite, got {pred}"
        );
    }

    /// with_projection_config allows full ProjectionConfig control.
    #[test]
    fn with_projection_config_builder() {
        let proj_cfg = ProjectionConfig {
            rank: 3,
            lambda: 0.995,
            warmup: 20,
            ..ProjectionConfig::default()
        };
        let factory = Factory::mamba(3).with_projection_config(8, proj_cfg);
        assert_eq!(
            factory.name(),
            "Projected<Mamba>",
            "with_projection_config should give Projected<Mamba> name"
        );
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..60 {
            let x: Vec<f64> = (0..8).map(|j| (i * j) as f64 * 0.01).collect();
            model.train(&x, i as f64 * 0.1);
        }
        let x: Vec<f64> = (0..8).map(|j| j as f64 * 0.05).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "projected Mamba (full config) prediction should be finite, got {pred}"
        );
    }

    /// Projected factory works inside AutoTuner.
    #[test]
    fn projected_factory_in_auto_tuner() {
        let mut tuner = crate::auto_tune(Factory::projected_mamba(8, 4));
        for i in 0..200 {
            let x: Vec<f64> = (0..8).map(|j| (i * j) as f64 * 0.01).collect();
            tuner.train(&x, i as f64 * 0.1);
        }
        let x: Vec<f64> = (0..8).map(|j| j as f64 * 0.05).collect();
        let pred = tuner.predict(&x);
        assert!(
            pred.is_finite(),
            "auto_tune with projected Mamba should produce finite prediction, got {pred}"
        );
    }

    /// Multi-factory racing with projected and non-projected factories.
    #[test]
    fn multi_factory_with_projected() {
        let mut tuner = crate::automl::AutoTuner::builder()
            .factory(Factory::sgbt(8))
            .add_factory(Factory::projected_mamba(8, 4))
            .add_factory(Factory::projected_kan(8, 4))
            .build();

        for i in 0..200 {
            let x: Vec<f64> = (0..8).map(|j| (i * j) as f64 * 0.01).collect();
            let y = x[0] * 3.0 + x[1];
            tuner.train(&x, y);
        }
        let x: Vec<f64> = (0..8).map(|j| j as f64 * 0.05).collect();
        let pred = tuner.predict(&x);
        assert!(
            pred.is_finite(),
            "multi-factory with projected should produce finite prediction, got {pred}"
        );
    }

    // ===================================================================
    // Mamba3 factory tests
    // ===================================================================

    /// Factory::mamba3 config space has 4 parameters.
    #[test]
    fn mamba3_factory_config_space() {
        let factory = Factory::mamba3(8);
        let space = factory.config_space();
        assert_eq!(
            space.n_params(),
            4,
            "Mamba3 should have 4 hyperparameters, got {}",
            space.n_params()
        );
        assert_eq!(
            space.params()[0].name(),
            "n_state",
            "first param should be n_state"
        );
        assert_eq!(
            space.params()[1].name(),
            "n_groups",
            "second param should be n_groups"
        );
        assert_eq!(
            space.params()[2].name(),
            "forgetting_factor",
            "third param should be forgetting_factor"
        );
        assert_eq!(
            space.params()[3].name(),
            "warmup",
            "fourth param should be warmup"
        );
    }

    /// Factory::mamba3 has correct complexity and warmup.
    #[test]
    fn mamba3_factory_defaults() {
        let factory = Factory::mamba3(8);
        assert_eq!(
            factory.complexity_hint(),
            5000,
            "Mamba3 complexity should be 5000"
        );
        assert_eq!(factory.warmup_hint(), 10, "Mamba3 warmup should be 10");
        assert_eq!(factory.name(), "Mamba3", "Mamba3 name should be 'Mamba3'");
        assert_eq!(
            factory.algorithm(),
            Algorithm::Mamba3,
            "algorithm should be Mamba3"
        );
    }

    /// Factory::mamba3 creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_mamba3() {
        let factory = Factory::mamba3(8);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..50 {
            let x: Vec<f64> = (0..8)
                .map(|k| (i as f64 * 0.1 + k as f64 * 0.3).sin())
                .collect();
            let y = x[0] * 2.0 + x[1];
            model.train(&x, y);
        }
        let x: Vec<f64> = (0..8).map(|k| (5.0 + k as f64 * 0.3).sin()).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "unified Mamba3 prediction should be finite, got {pred}"
        );
    }

    /// Factory::projected_mamba3 creates a projected model.
    #[test]
    fn projected_mamba3_factory_create_and_predict() {
        let factory = Factory::projected_mamba3(8, 4);
        assert_eq!(
            factory.name(),
            "Projected<Mamba3>",
            "projected mamba3 factory name should be Projected<Mamba3>"
        );
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..100 {
            let x: Vec<f64> = (0..8).map(|j| (i * j) as f64 * 0.01).collect();
            model.train(&x, i as f64 * 0.1);
        }
        let x: Vec<f64> = (0..8).map(|j| j as f64 * 0.05).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "projected Mamba3 prediction should be finite, got {pred}"
        );
    }

    // ===================================================================
    // DeltaProduct factory tests
    // ===================================================================

    /// Factory::delta_product config space has 4 parameters.
    #[test]
    fn delta_product_factory_config_space() {
        let factory = Factory::delta_product(8);
        let space = factory.config_space();
        assert_eq!(
            space.n_params(),
            4,
            "DeltaProduct should have 4 hyperparameters, got {}",
            space.n_params()
        );
        assert_eq!(
            space.params()[0].name(),
            "n_heads",
            "first param should be n_heads"
        );
        assert_eq!(
            space.params()[1].name(),
            "n_compositions",
            "second param should be n_compositions"
        );
        assert_eq!(
            space.params()[2].name(),
            "forgetting_factor",
            "third param should be forgetting_factor"
        );
        assert_eq!(
            space.params()[3].name(),
            "warmup",
            "fourth param should be warmup"
        );
    }

    /// Factory::delta_product has correct complexity, warmup, and name.
    #[test]
    fn delta_product_factory_defaults() {
        let factory = Factory::delta_product(8);
        assert_eq!(
            factory.complexity_hint(),
            8000,
            "DeltaProduct complexity should be 8000"
        );
        assert_eq!(
            factory.warmup_hint(),
            10,
            "DeltaProduct warmup should be 10"
        );
        assert_eq!(
            factory.name(),
            "DeltaProduct",
            "DeltaProduct name should be 'DeltaProduct'"
        );
        assert_eq!(
            factory.algorithm(),
            Algorithm::DeltaProduct,
            "algorithm should be DeltaProduct"
        );
    }

    /// Factory::delta_product creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_delta_product() {
        let factory = Factory::delta_product(8);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..50 {
            let x: Vec<f64> = (0..8)
                .map(|k| (i as f64 * 0.1 + k as f64 * 0.3).sin())
                .collect();
            let y = x[0] * 2.0 + x[1];
            model.train(&x, y);
        }
        let x: Vec<f64> = (0..8).map(|k| (5.0 + k as f64 * 0.3).sin()).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "unified DeltaProduct prediction should be finite, got {pred}"
        );
    }

    /// Factory::projected_delta_product creates a projected model.
    #[test]
    fn projected_delta_product_factory_create_and_predict() {
        let factory = Factory::projected_delta_product(8, 4);
        assert_eq!(
            factory.name(),
            "Projected<DeltaProduct>",
            "projected delta_product factory name should be Projected<DeltaProduct>"
        );
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..100 {
            let x: Vec<f64> = (0..8).map(|j| (i * j) as f64 * 0.01).collect();
            model.train(&x, i as f64 * 0.1);
        }
        let x: Vec<f64> = (0..8).map(|j| j as f64 * 0.05).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "projected DeltaProduct prediction should be finite, got {pred}"
        );
    }

    // ===================================================================
    // RWKV7 factory tests
    // ===================================================================

    /// Factory::rwkv7 config space has 3 parameters.
    #[test]
    fn rwkv7_factory_config_space() {
        let factory = Factory::rwkv7(8);
        let space = factory.config_space();
        assert_eq!(
            space.n_params(),
            3,
            "RWKV7 should have 3 hyperparameters, got {}",
            space.n_params()
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

    /// Factory::rwkv7 has correct complexity, warmup, and name.
    #[test]
    fn rwkv7_factory_defaults() {
        let factory = Factory::rwkv7(8);
        assert_eq!(
            factory.complexity_hint(),
            5000,
            "RWKV7 complexity should be 5000"
        );
        assert_eq!(factory.warmup_hint(), 10, "RWKV7 warmup should be 10");
        assert_eq!(factory.name(), "RWKV7", "RWKV7 name should be 'RWKV7'");
        assert_eq!(
            factory.algorithm(),
            Algorithm::Rwkv7,
            "algorithm should be Rwkv7"
        );
    }

    /// Factory::rwkv7 creates model that trains and predicts finite values.
    #[test]
    fn unified_factory_rwkv7() {
        let factory = Factory::rwkv7(8);
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..50 {
            let x: Vec<f64> = (0..8)
                .map(|k| (i as f64 * 0.1 + k as f64 * 0.3).sin())
                .collect();
            let y = x[0] * 2.0 + x[1];
            model.train(&x, y);
        }
        let x: Vec<f64> = (0..8).map(|k| (5.0 + k as f64 * 0.3).sin()).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "unified RWKV7 prediction should be finite, got {pred}"
        );
    }

    /// Factory::projected_rwkv7 creates a projected model.
    #[test]
    fn projected_rwkv7_factory_create_and_predict() {
        let factory = Factory::projected_rwkv7(8, 4);
        assert_eq!(
            factory.name(),
            "Projected<RWKV7>",
            "projected rwkv7 factory name should be Projected<RWKV7>"
        );
        let space = factory.config_space();
        let mut sampler = ConfigSampler::new(space, 42);
        let config = sampler.random();
        let mut model = factory.create(&config);

        for i in 0..100 {
            let x: Vec<f64> = (0..8).map(|j| (i * j) as f64 * 0.01).collect();
            model.train(&x, i as f64 * 0.1);
        }
        let x: Vec<f64> = (0..8).map(|j| j as f64 * 0.05).collect();
        let pred = model.predict(&x);
        assert!(
            pred.is_finite(),
            "projected RWKV7 prediction should be finite, got {pred}"
        );
    }
}
