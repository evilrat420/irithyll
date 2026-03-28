//! Streaming AutoML: champion-challenger racing with bandit-guided hyperparameter search.
//!
//! Implements online hyperparameter optimization for streaming machine learning models.
//! A champion model always provides predictions while challengers with different
//! hyperparameter configurations are evaluated in parallel. The best challenger
//! is promoted to champion when it consistently outperforms.
//!
//! # Architecture
//!
//! - [`ModelFactory`] -- trait for creating model instances from hyperparameter configs
//! - [`ConfigSpace`] / [`HyperConfig`] -- hyperparameter search space and configurations
//! - [`AutoTuner`] -- top-level orchestrator (implements `StreamingLearner`)
//! - [`RewardNormalizer`] -- maps metric values to \[0,1\] for bandit consumption
//!
//! # References
//!
//! - Wu et al. (2021) "ChaCha for Online AutoML" ICML -- champion-challenger framework
//! - Qi et al. (2023) "Discounted Thompson Sampling" -- non-stationary bandit selection
//! - Wilson et al. (2026) "SUHEN" IEEE TAI -- successive halving for streaming

mod auto_tuner;
mod config_space;
mod factories;
mod reward;

pub use auto_tuner::{
    AutoTuner, AutoTunerBuilder, AutoTunerConfig, AutoTunerSnapshot, CandidateSnapshot,
};
pub use config_space::{ConfigSampler, ConfigSpace, HyperConfig, HyperParam};
#[allow(deprecated)]
pub use factories::{
    Algorithm, AttentionFactory, EsnFactory, Factory, MambaFactory, SgbtFactory, SpikeNetFactory,
};
pub use reward::RewardNormalizer;

/// Metric to optimize during auto-tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoMetric {
    /// Mean Absolute Error (lower is better).
    MAE,
    /// Mean Squared Error (lower is better).
    MSE,
    /// Root Mean Squared Error (lower is better).
    RMSE,
}

/// Factory for creating streaming learner instances from hyperparameter configurations.
///
/// Implementations define the hyperparameter search space and how to construct
/// a model from a given configuration point.
pub trait ModelFactory: Send + Sync {
    /// The hyperparameter search space for this model type.
    fn config_space(&self) -> ConfigSpace;

    /// Create a new model instance from a hyperparameter configuration.
    ///
    /// The `config` values correspond to the parameters in `config_space()`.
    fn create(&self, config: &HyperConfig) -> Box<dyn irithyll_core::learner::StreamingLearner>;

    /// Human-readable name for this model type (e.g., "SGBT", "ESN").
    fn name(&self) -> &str;

    /// Minimum samples a new model needs before its metrics are meaningful.
    ///
    /// Candidates that have seen fewer than `warmup_hint()` samples are
    /// protected from elimination during tournament rounds. This prevents
    /// neural architectures with warmup phases (ESN, Mamba, SpikeNet) from
    /// being prematurely killed by models that start predicting immediately.
    ///
    /// The default is 0 (no warmup protection).
    fn warmup_hint(&self) -> usize {
        0
    }

    /// Approximate model complexity (effective parameter count).
    ///
    /// Used for complexity-adjusted elimination: models with higher complexity
    /// are penalized more when evaluation data is scarce. This naturally
    /// favors simpler models on sparse data and lets complex models prove
    /// themselves when data is abundant.
    ///
    /// The default is 100 (moderate complexity).
    fn complexity_hint(&self) -> usize {
        100
    }
}
