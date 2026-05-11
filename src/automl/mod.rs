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

pub mod adaptation_bus;
pub mod auto_builder;
pub mod auto_tuner;
pub mod budget;
pub mod cohort;
mod config_space;
mod factories;
mod lipschitz_verification;
pub mod meta_learner;
pub mod racing;
mod reward;
pub mod space;

pub use adaptation_bus::{
    AdaptContext, AdaptationBus, BusError, CriticalGuard, DriftRateAdapter, MetaAdapter,
    NoOpAdapter, PlasticityAdapter, ThetaDelta,
};
pub use auto_builder::{
    ConfigBounds, ConfigDiagnostics, DiagnosticAdaptor, DiagnosticLearner, FeasibleRegion,
    MetaObjective, RaceResults, SmoothAdjustments, StructuralChange, TerminateAfter, WelfordRace,
    WelfordStats,
};
#[cfg(feature = "distill")]
#[cfg_attr(docsrs, doc(cfg(feature = "distill")))]
pub use auto_builder::{DistillationConfig, DistillationStats};
pub use auto_tuner::{
    AutoTuner, AutoTunerBuilder, AutoTunerConfig, AutoTunerSnapshot, CandidateSnapshot,
};
pub use budget::{ArmBudget, BudgetLedger, BudgetStatus};
pub use cohort::{ChampionCohort, CohortMember, CohortMemberSnapshot, CohortWeight, COHORT_K};
#[allow(deprecated)]
pub use config_space::{ConfigSampler, ConfigSpace, HyperConfig, HyperParam};
pub use factories::{Algorithm, Factory, FactoryError};
pub use meta_learner::{
    ComplexityClass, FactoryMetaLearner, MetaLearner, MetaScore, MetaSearch, NoOpMetaLearner,
    Objective, SgbtClassificationMetaLearner, SgbtMetaLearner,
};
pub use racing::{
    bernstein_compare, bernstein_halfwidth, bernstein_promotion_test, empirical_bernstein_ci,
    ewma_bernstein_ci, ArmStats, EwmaWelfordTracker, PromotionVerdict, WelfordTracker,
    BERNSTEIN_DELTA, MIN_SAMPLES_FOR_BERNSTEIN,
};
pub use reward::RewardNormalizer;
pub use space::{
    categorical, int_range, linear_range, log_range, when, Category, Condition, ConditionBuilder,
    Constraint, ParamDef, ParamMap, ParamValue, SamplerError, Scale, SearchSpace,
    SearchSpaceBuilder, SpaceError,
};

/// Metric to optimize during auto-tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AutoMetric {
    /// Mean Absolute Error (lower is better).
    MAE,
    /// Mean Squared Error (lower is better).
    MSE,
    /// Root Mean Squared Error (lower is better).
    RMSE,
}

/// Trait for models that can provide diagnostic signals for the auto-builder.
///
/// Models implementing this trait get full auto-builder benefit:
/// streaming config adaptation based on model internals. Models that
/// don't implement it still get FeasibleRegion + WelfordRace.
pub trait DiagnosticSource {
    /// Return config diagnostics, or `None` if not supported.
    fn config_diagnostics(&self) -> Option<auto_builder::ConfigDiagnostics>;
}

/// Factory for creating streaming learner instances from hyperparameter configurations.
///
/// Implementations define the hyperparameter search space (a typed
/// [`SearchSpace`]) and how to construct a model from a sampled
/// [`ParamMap`].
///
/// # Migration from positional `HyperConfig`
///
/// Pre-v10 factories returned a [`ConfigSpace`] of positional `HyperParam`
/// entries and consumed a [`HyperConfig`] (a `Vec<f64>` indexed by position).
/// That API is deprecated in favor of typed, named-access [`SearchSpace`] /
/// [`ParamMap`]. The legacy types remain for one release cycle behind
/// `#[deprecated]` to give downstream crates time to migrate.
pub trait ModelFactory: Send + Sync {
    /// The hyperparameter search space for this model type.
    fn config_space(&self) -> SearchSpace;

    /// Create a new model instance from a sampled parameter map.
    ///
    /// The `params` are values drawn from [`Self::config_space`]. Factories
    /// access them by name via [`ParamMap::float`] / [`ParamMap::int`] /
    /// [`ParamMap::category`]. Conditional parameters whose gate did not fire
    /// are absent from the map; factories must use the `_optional` variants
    /// for those reads.
    ///
    /// Returns `Err(FactoryError)` when the sampled hyperparameter combination
    /// is structurally invalid. The AutoML racing layer catches this error,
    /// logs a warning, and skips the offending arm rather than panicking.
    fn create(
        &self,
        params: &ParamMap,
    ) -> Result<Box<dyn irithyll_core::learner::StreamingLearner>, FactoryError>;

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

    /// Number of input features this factory expects.
    ///
    /// Used by the auto-builder to initialize the [`FeasibleRegion`] with
    /// correct dimensionality, ensuring config bounds (especially grace period
    /// and lambda) are properly calibrated.
    ///
    /// The default is 1 (conservative estimate).
    fn n_features_hint(&self) -> usize {
        1
    }

    /// Return `true` if the SPSA auto-builder (`FeasibleRegion` + `DiagnosticLearner`)
    /// is meaningful for this factory.
    ///
    /// The auto-builder is designed for the SGBT family. Non-SGBT factories
    /// should return `false` (the default) so that the `AutoTuner` can log a
    /// warning and skip activating the adaptor rather than silently no-oping.
    fn supports_auto_builder(&self) -> bool {
        false
    }
}
