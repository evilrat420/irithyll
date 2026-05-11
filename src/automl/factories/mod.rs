//! Model factory for AutoML hyperparameter search.
//!
//! The [`Factory`] type implements [`ModelFactory`] for all streaming learner
//! algorithms, defining hyperparameter search spaces and constructing instances
//! from sampled configurations.

use crate::automl::meta_learner::{
    ComplexityClass, FactoryMetaLearner, MetaLearner, NoOpMetaLearner,
    SgbtClassificationMetaLearner, SgbtMetaLearner,
};
use crate::automl::space::{ParamMap, SearchSpace};
use crate::automl::ModelFactory;
use crate::projection::{ProjectedLearner, ProjectionConfig};
use irithyll_core::learner::StreamingLearner;

mod attention;
mod error;
mod neural;
mod ssm;
mod trees;

pub use error::FactoryError;

// ===========================================================================
// Algorithm + Unified Factory
// ===========================================================================

/// Algorithm type for the unified factory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    /// Streaming sLSTM (exponential gating with log-domain stabilization).
    Slstm,
    /// Streaming Mamba BD-LRU (block-diagonal linear recurrence).
    MambaBD,
    /// Streaming mGRADE (minimal recurrent gating with delay convolutions).
    Mgrade,
    /// Multi-class SGBT (one-vs-rest committee, softmax outputs).
    MulticlassSgbt,
}

impl Algorithm {
    /// Return `true` if this algorithm belongs to the SGBT family.
    ///
    /// The SPSA auto-builder (`FeasibleRegion` / `DiagnosticLearner`) is
    /// designed for the SGBT family only. Activating it for other algorithms
    /// is a silent no-op because their `adjust_config` and `apply_structural_change`
    /// default-impl to no-ops.
    pub fn is_sgbt_family(self) -> bool {
        matches!(
            self,
            Algorithm::Sgbt | Algorithm::Distributional | Algorithm::MulticlassSgbt
        )
    }
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
    pub(crate) algorithm: Algorithm,
    pub(crate) n_features: usize,
    pub(crate) space: SearchSpace,
    pub(crate) warmup: usize,
    pub(crate) complexity: usize,
    pub(crate) seed: u64,
    pub(crate) accuracy_based_pruning: bool,
    pub(crate) proactive_prune_interval: Option<u64>,
    pub(crate) prune_half_life: Option<usize>,
    /// Optional PAST projection wrapping: (d_in, config).
    /// When set, `create()` wraps the inner model in a [`ProjectedLearner`].
    pub(crate) projection: Option<(usize, ProjectionConfig)>,
}

// ===========================================================================
// Trait Implementation + Common Builders
// ===========================================================================

impl ModelFactory for Factory {
    fn config_space(&self) -> SearchSpace {
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
                Algorithm::Slstm => "Projected<sLSTM>",
                Algorithm::MambaBD => "Projected<MambaBD>",
                Algorithm::Mgrade => "Projected<mGRADE>",
                Algorithm::MulticlassSgbt => "Projected<MulticlassSGBT>",
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
                Algorithm::Slstm => "sLSTM",
                Algorithm::MambaBD => "MambaBD",
                Algorithm::Mgrade => "mGRADE",
                Algorithm::MulticlassSgbt => "MulticlassSGBT",
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

    fn supports_auto_builder(&self) -> bool {
        self.algorithm.is_sgbt_family()
    }

    fn create(&self, params: &ParamMap) -> Result<Box<dyn StreamingLearner>, FactoryError> {
        self.create_inner(params)
    }
}

// ===========================================================================
// MetaLearner declarations per algorithm family.
//
// Every Factory carries a per-family MetaLearner declaration. SGBT family
// (regression + classification) declares an SPSA-derived Lipschitz bound and
// the relevant objective surface. Every other family explicitly opts out via
// NoOpMetaLearner with a cited rationale, observable at audit time.
//
// The default `FactoryMetaLearner` blanket impl returns NoOpMetaLearner; this
// override provides SGBT-aware declarations and surfaces a per-family rationale
// for non-SGBT opt-outs (e.g. ESN spectral radius cannot be tuned mid-stream
// without violating the no-mid-stream-cell-reset principle in `AGENTS.md`).
//
// Refs: Banach (1922), Spall (1998), AM-R2 §2-§4.
// ===========================================================================

impl FactoryMetaLearner for Factory {
    fn meta_learner(&self) -> Box<dyn MetaLearner> {
        let complexity = ComplexityClass::from_hint(self.complexity);
        match self.algorithm {
            Algorithm::Sgbt | Algorithm::Distributional => {
                Box::new(SgbtMetaLearner::new(complexity))
            }
            Algorithm::MulticlassSgbt => Box::new(SgbtClassificationMetaLearner::new(complexity)),
            // Reservoir computing: spectral_radius and reservoir_size are
            // structural; mid-stream change requires reservoir reset which
            // violates AGENTS.md "no-mid-stream-cell-reset" principle. Only
            // input_scaling and output_regularization would be warm-tunable;
            // not yet wired through Tunable::adjust_config for this family.
            Algorithm::Esn => Box::new(NoOpMetaLearner::new(
                "ESN: spectral radius is structural; warm-tunable scalings \
                 not yet wired through Tunable::adjust_config",
                complexity,
            )),
            // SSM family: continuous knobs (forgetting_factor, lr) are tunable
            // in principle but Tunable::adjust_config is not yet wired through
            // these models (PR-AM-15 Wave 5c will land them family by family).
            // Until wired, declaring lipschitz<1 would let the bus emit
            // lr_multiplier deltas with no runtime effect — a silent failure
            // mode (R9 P8). Explicit no-op until wiring completes.
            Algorithm::Mamba
            | Algorithm::Mamba3
            | Algorithm::MambaBD
            | Algorithm::Slstm
            | Algorithm::Mgrade => Box::new(NoOpMetaLearner::new(
                "SSM family: continuous knobs not yet wired through \
                 Tunable::adjust_config; awaiting Wave 5c per-family wiring",
                complexity,
            )),
            // Attention family: attention multi-rate adaptation (supervised_lr,
            // gate_lr) is principled per AM-R2 §4 but requires single-knob SPSA
            // wiring per rate. Not yet wired through Tunable::adjust_config.
            Algorithm::Attention | Algorithm::DeltaProduct | Algorithm::Rwkv7 => {
                Box::new(NoOpMetaLearner::new(
                    "Attention family: multi-rate adaptation requires per-rate \
                     SPSA wiring through Tunable::adjust_config; awaiting Wave 5c",
                    complexity,
                ))
            }
            // Neural / spiking: SPSA on lr is tractable (AM-R2 §4 declares
            // L_neural = 0.9) but Tunable::adjust_config is not yet wired
            // through SpikeNet / KAN / TTT. Explicit no-op until wiring lands.
            Algorithm::SpikeNet | Algorithm::Kan | Algorithm::Ttt => {
                Box::new(NoOpMetaLearner::new(
                    "Neural / spiking family: SPSA on lr requires Tunable wiring; \
                     not yet landed for this family",
                    complexity,
                ))
            }
        }
    }
}

impl Factory {
    // -----------------------------------------------------------------------
    // Builder-style overrides
    // -----------------------------------------------------------------------

    /// Override the default search space.
    pub fn with_space(mut self, space: SearchSpace) -> Self {
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

    /// Override the bounds of a named float hyperparameter in the search space.
    ///
    /// Panics if the parameter name is not found, or if the parameter is not
    /// a float (use [`with_config_int_range`](Self::with_config_int_range) for
    /// integer parameters).
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll::automl::Factory;
    ///
    /// let factory = Factory::sgbt(4)
    ///     .with_config_range("learning_rate", 0.01, 0.1);
    /// ```
    pub fn with_config_range(mut self, name: &str, low: f64, high: f64) -> Self {
        self.space
            .set_float_range(name, low, high)
            .unwrap_or_else(|e| panic!("Factory::with_config_range: {e}"));
        self
    }

    /// Override the bounds of a named integer hyperparameter in the search space.
    ///
    /// Panics if the parameter name is not found, or if the parameter is not
    /// an integer.
    pub fn with_config_int_range(mut self, name: &str, low: i64, high: i64) -> Self {
        self.space
            .set_int_range(name, low, high)
            .unwrap_or_else(|e| panic!("Factory::with_config_int_range: {e}"));
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

    /// Internal create method dispatched by ModelFactory::create.
    fn create_inner(&self, params: &ParamMap) -> Result<Box<dyn StreamingLearner>, FactoryError> {
        let inner: Box<dyn StreamingLearner> = match self.algorithm {
            Algorithm::Sgbt | Algorithm::Distributional | Algorithm::MulticlassSgbt => {
                self.create_tree(params)?
            }

            Algorithm::Esn => self.create_ssm(params)?,

            Algorithm::Mamba
            | Algorithm::Mamba3
            | Algorithm::MambaBD
            | Algorithm::Slstm
            | Algorithm::Mgrade => self.create_ssm(params)?,

            Algorithm::Attention | Algorithm::DeltaProduct | Algorithm::Rwkv7 => {
                self.create_attention(params)?
            }

            Algorithm::SpikeNet | Algorithm::Kan | Algorithm::Ttt => self.create_neural(params)?,
        };

        // Wrap in ProjectedLearner if projection is configured.
        if let Some((d_in, ref proj_config)) = self.projection {
            Ok(Box::new(ProjectedLearner::new(
                inner,
                d_in,
                proj_config.clone(),
            )))
        } else {
            Ok(inner)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Factory works with projection via the builder chain.
    #[test]
    fn projection_factory_in_auto_tuner() {
        let mut tuner = crate::auto_tune(Factory::mamba(8).with_projection(8, 4, 0.999));
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
            .add_factory(Factory::mamba(8).with_projection(8, 4, 0.999))
            .add_factory(Factory::kan(8).with_projection(8, 4, 0.999))
            .build()
            .expect("valid config");

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

    // -----------------------------------------------------------------------
    // FactoryError tests (PR-AM-7) — typed-config edition
    // -----------------------------------------------------------------------

    use crate::automl::space::ParamValue;

    /// Build a [`ParamMap`] from `(name, ParamValue)` pairs.
    fn pm(pairs: &[(&str, ParamValue)]) -> ParamMap {
        let mut m = ParamMap::new();
        for (name, v) in pairs {
            m.insert((*name).to_string(), v.clone());
        }
        m
    }

    /// SGBT rejects learning_rate=0 instead of panicking.
    ///
    /// learning_rate=0 violates SGBTConfig's "must be > 0" invariant, so
    /// `SGBTConfig::builder().build()` returns a `ConfigError`. The factory
    /// must propagate it as `FactoryError::InvalidConfig`.
    #[test]
    fn sgbt_arm_config_rejects_out_of_range() {
        let factory = Factory::sgbt(3);
        let bad = pm(&[
            ("learning_rate", ParamValue::Float(0.0)),
            ("n_steps", ParamValue::Int(10)),
            ("max_depth", ParamValue::Int(3)),
            ("n_bins", ParamValue::Int(16)),
            ("lambda", ParamValue::Float(1.0)),
            ("feature_subsample_rate", ParamValue::Float(0.5)),
            ("grace_period", ParamValue::Int(10)),
        ]);
        let result = factory.create(&bad);
        let err = result.err().expect("learning_rate=0 must be rejected");
        assert!(
            matches!(err, FactoryError::InvalidConfig(_)),
            "expected FactoryError::InvalidConfig for learning_rate=0, got: {err}"
        );
    }

    /// Distributional rejects learning_rate=0 (parallel of SGBT path).
    #[test]
    fn distributional_arm_config_rejects_out_of_range() {
        let factory = Factory::distributional(3);
        let bad = pm(&[
            ("learning_rate", ParamValue::Float(0.0)),
            ("n_steps", ParamValue::Int(10)),
            ("max_depth", ParamValue::Int(3)),
            ("n_bins", ParamValue::Int(16)),
            ("lambda", ParamValue::Float(1.0)),
            ("feature_subsample_rate", ParamValue::Float(0.5)),
            ("grace_period", ParamValue::Int(10)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::InvalidConfig(_))),
            "Distributional must reject learning_rate=0"
        );
    }

    /// MulticlassSGBT rejects learning_rate=0.
    #[test]
    fn multiclass_sgbt_arm_config_rejects_out_of_range() {
        let factory = Factory::multiclass_sgbt(3, 5);
        let bad = pm(&[
            ("learning_rate", ParamValue::Float(0.0)),
            ("n_steps", ParamValue::Int(10)),
            ("max_depth", ParamValue::Int(3)),
            ("n_bins", ParamValue::Int(16)),
            ("lambda", ParamValue::Float(1.0)),
            ("n_classes", ParamValue::Int(3)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::InvalidConfig(_))),
            "MulticlassSGBT must reject learning_rate=0"
        );
    }

    /// ESN rejects spectral_radius=0 (must be > 0 per ESNConfig).
    #[test]
    fn esn_arm_config_rejects_out_of_range() {
        let factory = Factory::esn();
        let bad = pm(&[
            ("n_reservoir", ParamValue::Int(50)),
            ("spectral_radius", ParamValue::Float(0.0)),
            ("leak_rate", ParamValue::Float(0.5)),
            ("input_scaling", ParamValue::Float(1.0)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::InvalidConfig(_))),
            "ESN must reject spectral_radius=0"
        );
    }

    /// Mamba rejects forgetting_factor>=1 (must be < 1.0 per MambaConfig).
    #[test]
    fn mamba_arm_config_rejects_out_of_range() {
        let factory = Factory::mamba(4);
        let bad = pm(&[
            ("n_state", ParamValue::Int(8)),
            ("forgetting_factor", ParamValue::Float(1.5)),
            ("warmup", ParamValue::Int(5)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::InvalidConfig(_))),
            "Mamba must reject forgetting_factor=1.5"
        );
    }

    /// Mamba3 rejects n_groups that does not divide d_in.
    #[test]
    fn mamba3_arm_config_rejects_out_of_range() {
        // d_in = 5: n_groups must be 1 or 5. Choose n_groups=2 to violate.
        let factory = Factory::mamba3(5);
        let bad = pm(&[
            ("n_state", ParamValue::Int(8)),
            ("n_groups", ParamValue::Int(2)),
            ("forgetting_factor", ParamValue::Float(0.99)),
            ("warmup", ParamValue::Int(5)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::IncompatibleArm { .. })),
            "Mamba3 must reject n_groups=2 for d_in=5"
        );
    }

    /// MambaBD rejects forgetting_factor=0.0 (must be > 0).
    #[test]
    fn mamba_bd_arm_config_rejects_out_of_range() {
        let factory = Factory::mamba_bd(8);
        let bad = pm(&[
            ("n_state", ParamValue::Int(8)),
            ("forgetting_factor", ParamValue::Float(0.0)),
            ("warmup", ParamValue::Int(5)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::InvalidConfig(_))),
            "MambaBD must reject forgetting_factor=0.0"
        );
    }

    /// sLSTM rejects forgetting_factor=1.5 (must be in (0, 1]).
    #[test]
    fn slstm_arm_config_rejects_out_of_range() {
        let factory = Factory::slstm(4);
        let bad = pm(&[
            ("d_model", ParamValue::Int(16)),
            ("forgetting_factor", ParamValue::Float(1.5)),
            ("warmup", ParamValue::Int(5)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::InvalidConfig(_))),
            "sLSTM must reject forgetting_factor=1.5"
        );
    }

    /// mGRADE rejects forgetting_factor=2.0 (must be ≤ 1).
    #[test]
    fn mgrade_arm_config_rejects_out_of_range() {
        let factory = Factory::mgrade(4);
        let bad = pm(&[
            ("d_hidden", ParamValue::Int(8)),
            ("kernel_size", ParamValue::Int(3)),
            ("forgetting_factor", ParamValue::Float(2.0)),
            ("warmup", ParamValue::Int(5)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::InvalidConfig(_))),
            "mGRADE must reject forgetting_factor=2.0"
        );
    }

    /// Attention rejects n_heads (categorical) when not dividing d_model.
    #[test]
    fn attention_arm_config_rejects_out_of_range() {
        // d_model=5, n_heads=4 → 5 % 4 != 0 → IncompatibleArm.
        let factory = Factory::attention(5);
        let bad = pm(&[
            (
                "n_heads",
                ParamValue::Category(crate::automl::Category::from("4")),
            ),
            ("forgetting_factor", ParamValue::Float(0.99)),
            ("warmup", ParamValue::Int(5)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::IncompatibleArm { .. })),
            "attention must reject n_heads=4 for d_model=5"
        );
    }

    /// DeltaProduct rejects n_heads not dividing d_model.
    #[test]
    fn delta_product_arm_config_rejects_out_of_range() {
        let factory = Factory::delta_product(5);
        let bad = pm(&[
            ("n_heads", ParamValue::Int(3)),
            ("n_compositions", ParamValue::Int(2)),
            ("forgetting_factor", ParamValue::Float(0.99)),
            ("warmup", ParamValue::Int(10)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::IncompatibleArm { .. })),
            "DeltaProduct must reject n_heads=3 for d_model=5"
        );
    }

    /// RWKV-7 rejects n_heads not dividing d_model.
    #[test]
    fn rwkv7_arm_config_rejects_out_of_range() {
        let factory = Factory::rwkv7(5);
        let bad = pm(&[
            ("n_heads", ParamValue::Int(2)),
            ("forgetting_factor", ParamValue::Float(0.99)),
            ("warmup", ParamValue::Int(5)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::IncompatibleArm { .. })),
            "RWKV7 must reject n_heads=2 for d_model=5"
        );
    }

    /// SpikeNet rejects v_thr=0 (LIF must have positive threshold).
    #[test]
    fn spike_net_arm_config_rejects_out_of_range() {
        let factory = Factory::spike_net();
        let bad = pm(&[
            ("n_hidden", ParamValue::Int(32)),
            ("alpha", ParamValue::Float(0.9)),
            ("eta", ParamValue::Float(0.001)),
            ("v_thr", ParamValue::Float(0.0)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::InvalidConfig(_))),
            "SpikeNet must reject v_thr=0.0"
        );
    }

    /// KAN rejects learning_rate=0 (must be > 0).
    #[test]
    fn kan_arm_config_rejects_out_of_range() {
        let factory = Factory::kan(3);
        let bad = pm(&[
            ("hidden_size", ParamValue::Int(8)),
            ("grid_size", ParamValue::Int(5)),
            ("learning_rate", ParamValue::Float(0.0)),
            ("spline_order", ParamValue::Int(3)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::InvalidConfig(_))),
            "KAN must reject learning_rate=0"
        );
    }

    /// TTT rejects learning_rate=0 (must be > 0).
    #[test]
    fn ttt_arm_config_rejects_out_of_range() {
        let factory = Factory::ttt(4);
        let bad = pm(&[
            ("d_model", ParamValue::Int(8)),
            ("learning_rate", ParamValue::Float(0.0)),
            ("alpha", ParamValue::Float(0.001)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::InvalidConfig(_))),
            "TTT must reject learning_rate=0"
        );
    }

    /// ParamMap missing a required param produces IncompatibleArm.
    ///
    /// Replaces the silent `config.get(i)` panic of the positional API with
    /// an explicit, propagated factory error.
    #[test]
    fn factory_rejects_missing_param() {
        let factory = Factory::sgbt(3);
        // Missing learning_rate.
        let bad = pm(&[
            ("n_steps", ParamValue::Int(10)),
            ("max_depth", ParamValue::Int(3)),
            ("n_bins", ParamValue::Int(16)),
            ("lambda", ParamValue::Float(1.0)),
            ("feature_subsample_rate", ParamValue::Float(0.5)),
            ("grace_period", ParamValue::Int(10)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::IncompatibleArm { .. })),
            "missing required param must produce IncompatibleArm"
        );
    }

    /// ParamMap with wrong type for a param produces IncompatibleArm.
    ///
    /// Float requested, Int provided.
    #[test]
    fn factory_rejects_wrong_type_param() {
        let factory = Factory::sgbt(3);
        let bad = pm(&[
            // learning_rate is a Float but we provide an Int.
            ("learning_rate", ParamValue::Int(0)),
            ("n_steps", ParamValue::Int(10)),
            ("max_depth", ParamValue::Int(3)),
            ("n_bins", ParamValue::Int(16)),
            ("lambda", ParamValue::Float(1.0)),
            ("feature_subsample_rate", ParamValue::Float(0.5)),
            ("grace_period", ParamValue::Int(10)),
        ]);
        let result = factory.create(&bad);
        assert!(
            matches!(result, Err(FactoryError::IncompatibleArm { .. })),
            "wrong type for required param must produce IncompatibleArm"
        );
    }

    /// AutoTuner does not panic when factory creates encounter invalid configs.
    ///
    /// Integration smoke test: run AutoTuner with SGBT for 100 samples.
    /// Even if a perturbed config is internally invalid, the tournament must
    /// continue rather than crashing.
    #[test]
    fn auto_tuner_survives_invalid_factory_config() {
        let mut tuner = crate::automl::AutoTuner::builder()
            .factory(Factory::sgbt(3))
            .build()
            .expect("valid config");

        for i in 0..100 {
            let x = [i as f64 * 0.1, 0.5, 1.0];
            let y = x[0] * 2.0 + 1.0;
            tuner.train(&x, y);
        }
        let pred = tuner.predict(&[1.0, 0.5, 1.0]);
        assert!(
            pred.is_finite(),
            "AutoTuner must produce finite prediction after 100 samples, got {pred}"
        );
    }

    /// FactoryError::InvalidConfig wraps ConfigError and displays correctly.
    #[test]
    fn factory_error_display() {
        use crate::automl::FactoryError;
        use irithyll_core::error::ConfigError;

        let ce = ConfigError::out_of_range("learning_rate", "must be > 0", 0.0);
        let fe = FactoryError::InvalidConfig(ce);
        let msg = fe.to_string();
        assert!(
            msg.contains("learning_rate"),
            "FactoryError display must include param name, got: {msg}"
        );

        let fe2 = FactoryError::IncompatibleArm {
            reason: "n_heads=3 does not divide d_model=5".into(),
        };
        let msg2 = fe2.to_string();
        assert!(
            msg2.contains("n_heads=3"),
            "IncompatibleArm display must include reason, got: {msg2}"
        );
    }
}
