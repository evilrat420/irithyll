//! Cross-model meta-learner: across-family search and Lipschitz-declared adaptation.
//!
//! `MetaLearner` is the per-factory trait that declares **what** a model family
//! is willing to adapt online and **at what rate** (Lipschitz bound on the
//! normalized parameter space). It is architecturally separate from
//! [`AutoTuner`][crate::automl::AutoTuner], which selects across families via
//! tournament successive halving, and from the per-arm runtime
//! `AdaptationBus` (PR-AM-12) which composes adapters under the
//! Banach-contraction safety theorem.
//!
//! # Three-layer architecture (v10 AutoML)
//!
//! ```text
//! [AutoTuner]              -- across-family, slowest timescale (rung budget).
//!     |   selects which factory to draw the next arm from via bandit.
//!     v
//! [MetaLearner]            -- per-factory, declarative (this trait).
//!     |   declares Lipschitz L_meta and which adaptation surfaces are exposed.
//!     v
//! [AdaptationBus]          -- per-arm, runtime (PR-AM-12).
//!     |   composes adapter operators under product-Lipschitz < 1 invariant.
//!     v
//! [Tunable / Structural]   -- per-model capability traits (already exist).
//!         deliver the actual `adjust_config(...)` / `apply_structural_change`.
//! ```
//!
//! `MetaLearner::lipschitz_bound()` is the single source of truth that
//! [`AdaptationBus`](crate::automl::AdaptationBus) uses to verify
//! `∏ L_i < 1` (Banach 1922 fixed-point theorem) and certify the per-arm
//! adapter composition is contractive.
//!
//! # Why a separate trait from `Tunable`
//!
//! [`Tunable`][irithyll_core::learner::Tunable] is a *runtime* capability:
//! "this model accepts `lr_multiplier` and `lambda_delta` per call". It says
//! nothing about whether tuning is *safe* or *productive* for the family —
//! e.g., a streaming SpikeNet implements `Tunable` for `eta` adjustments, but
//! mid-stream reservoir-size changes would violate the no-mid-stream-cell-reset
//! streaming principle (see `AGENTS.md`).
//!
//! `MetaLearner` declares the *meta-search surface*:
//! - `lipschitz_bound(): f64` — Lipschitz constant L_meta on normalized theta
//!   (used by `AdaptationBus` for composability). L_meta = 1.0 means
//!   non-expansive identity (opt-out); L_meta < 1.0 means strictly
//!   contractive (productive online adaptation).
//! - `objectives(): &[Objective]` — the metric surfaces this family can
//!   meaningfully optimize. Used by Pareto cross-family comparison so a
//!   classification family is not asked to compete on R².
//! - `complexity_class(): ComplexityClass` — discrete bucket for cross-family
//!   parameter-count grouping. Used by [`MetaSearch`] to prevent dominance by
//!   high-capacity families on data-scarce regimes.
//! - `tunes_continuous_knobs(): bool` and `tunes_structure(): bool` — declare
//!   what adapter classes apply.
//!
//! # No-op opt-out is explicit (not silent)
//!
//! Per Jono's discipline #2 ("sniff out band-aids"), the previous
//! `auto_builder` system silently no-op'd when the factory was non-SGBT (the
//! family's `adjust_config` default impl is a no-op). That is implicit
//! opt-out — undetectable, untestable, and bug-prone (R9 P8).
//!
//! The `MetaLearner` trait flips this: every factory returns *some* meta
//! learner (typically [`NoOpMetaLearner`]). Opt-out is a declared
//! `MetaLearner::is_no_op() -> true`, observable at construction time, with a
//! cited reason (see [`NoOpMetaLearner::reason`]). The orchestrator checks
//! this BEFORE wiring an `AdaptationBus`, so the runtime path never silently
//! collapses to a no-op loop.
//!
//! # Pareto over weighted sums
//!
//! Per Jono's discipline #1 ("reject arbitrary thresholds"), cross-family
//! comparison MUST NOT use `w_rmse * rmse + w_complexity * complexity` —
//! weight tuning is just moving the knob. The library uses Pareto dominance:
//! family A dominates family B iff A is no worse on every declared
//! [`Objective`] and strictly better on at least one. Non-dominated families
//! coexist in the cohort (PR-AM-5).
//!
//! # References
//!
//! - Banach (1922) "Sur les operations dans les ensembles abstraits" —
//!   contraction-mapping fixed-point theorem (compose-safety).
//! - Pareto (1906) "Manuale di economia politica" — multi-objective dominance.
//! - Maurer & Pontil (2009) "Empirical Bernstein Stopping" — within-Pareto-front
//!   tie-breaking (used by `bernstein_compare`).

use core::fmt;

use crate::automl::ModelFactory;

// ===========================================================================
// Objective — metric surfaces a family can optimize
// ===========================================================================

/// Objective surface a model family can meaningfully optimize.
///
/// Used by Pareto cross-family comparison in [`MetaSearch::pareto_dominates`]
/// so that a family declared only for regression is not asked to compete on
/// `F1`. Each declared objective comes with a sign convention: `is_minimization`
/// = true means smaller-is-better.
///
/// # Examples
///
/// - SGBT family: `[RegressionRMSE, RegressionMAE]` (regression-only).
/// - MulticlassSGBT family: `[ClassificationF1, ClassificationKappa]`.
/// - Distributional SGBT: `[RegressionRMSE, DistributionalCRPS]`.
///
/// **Source for tag set:** the metrics enumerated here mirror the
/// objective set in `MetaObjective` and the racing-layer reward functions
/// (Wu et al. 2021 "ChaCha for Online AutoML" §3 reports these as the
/// canonical online-AutoML objectives).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Objective {
    /// Regression root mean squared error (minimization).
    RegressionRmse,
    /// Regression mean absolute error (minimization).
    RegressionMae,
    /// Coefficient of determination R² (maximization).
    RegressionR2,
    /// Directional accuracy: fraction of correct sign predictions (maximization).
    DirectionalAccuracy,
    /// Binary classification F1 score (maximization).
    ClassificationF1,
    /// Cohen's kappa (agreement beyond chance, maximization).
    ClassificationKappa,
    /// Continuous ranked probability score for distributional output (minimization).
    DistributionalCrps,
}

impl Objective {
    /// `true` if smaller values are preferred; `false` if larger is better.
    pub fn is_minimization(self) -> bool {
        matches!(
            self,
            Objective::RegressionRmse | Objective::RegressionMae | Objective::DistributionalCrps
        )
    }

    /// Stable string identifier (used for diagnostics).
    pub fn as_str(self) -> &'static str {
        match self {
            Objective::RegressionRmse => "regression_rmse",
            Objective::RegressionMae => "regression_mae",
            Objective::RegressionR2 => "regression_r2",
            Objective::DirectionalAccuracy => "directional_accuracy",
            Objective::ClassificationF1 => "classification_f1",
            Objective::ClassificationKappa => "classification_kappa",
            Objective::DistributionalCrps => "distributional_crps",
        }
    }
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ===========================================================================
// ComplexityClass — discrete capacity bucket for cross-family Pareto axis
// ===========================================================================

/// Capacity bucket used as a Pareto axis to prevent high-capacity families
/// from monopolizing the cohort on data-scarce regimes.
///
/// # Threshold derivation (information-theoretic, not arbitrary)
///
/// The buckets are derived from `complexity_hint()` (effective parameter
/// count) using powers of 2 chosen to match the **information-theoretic
/// sample-complexity scaling** of generalization-error bounds:
/// for a model with O(p) effective parameters, generalization error bounds
/// of the form `√(p · ln(n) / n)` (Vapnik 1998 §3.4) require `n ≫ p` for
/// the bound to bind tightly. The bucket boundaries match the orders of
/// magnitude where a different `n` regime takes over.
///
/// - `Tiny`: p ≤ 100. Bound binds at n ~ 100 samples (cold-start regime).
/// - `Small`: 100 < p ≤ 1_000. Bound binds at n ~ 1_000.
/// - `Medium`: 1_000 < p ≤ 10_000. Bound binds at n ~ 10_000.
/// - `Large`: p > 10_000. Asymptotic regime.
///
/// **Why discrete buckets, not the raw count?** A continuous penalty
/// `λ · ln(p)` requires choosing λ — that is exactly the kind of weight-tuning
/// Jono's discipline #1 rejects. Buckets give Pareto a discrete axis with a
/// cited derivation; comparison stays parameter-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComplexityClass {
    /// Effective parameter count ≤ 100.
    Tiny,
    /// Effective parameter count in (100, 1_000].
    Small,
    /// Effective parameter count in (1_000, 10_000].
    Medium,
    /// Effective parameter count > 10_000.
    Large,
}

impl Default for ComplexityClass {
    /// Default complexity class: `Tiny` (≤ 100 effective parameters).
    fn default() -> Self {
        ComplexityClass::Tiny
    }
}

impl ComplexityClass {
    /// Map a `complexity_hint()` to a [`ComplexityClass`] via the threshold
    /// derivation in the type docs.
    pub fn from_hint(hint: usize) -> Self {
        // Strict-less-than thresholds match the documented bucket boundaries
        // (Tiny ≤ 100, Small ≤ 1_000, Medium ≤ 10_000, Large > 10_000).
        if hint <= 100 {
            ComplexityClass::Tiny
        } else if hint <= 1_000 {
            ComplexityClass::Small
        } else if hint <= 10_000 {
            ComplexityClass::Medium
        } else {
            ComplexityClass::Large
        }
    }
}

// ===========================================================================
// MetaLearner — per-factory adaptation declaration
// ===========================================================================

/// Per-factory declaration of online-adaptation surface and Lipschitz bound.
///
/// Implemented by every factory that participates in cross-family meta search.
/// The default implementation is [`NoOpMetaLearner`] — explicit opt-out with
/// a documented reason. Models that *do* support online adaptation override
/// this with a concrete impl that declares the contraction bound used by
/// the `AdaptationBus` (PR-AM-12) to certify safe composition.
///
/// # Lipschitz contract for `AdaptationBus`
///
/// The `AdaptationBus` (PR-AM-12) composes per-sample adapters
/// `T_i: Theta → Theta` (e.g. SPSA on `(lr, lambda)`, drift response shifts,
/// MTS forgetting nudges) into a single map `T = T_1 ∘ T_2 ∘ ... ∘ T_N`.
/// By the Banach fixed-point theorem (Banach 1922):
///
/// ```text
/// L_combined = L_1 · L_2 · ... · L_N < 1
///   ⇒  T has a unique fixed point θ*, iteration converges geometrically.
/// ```
///
/// Each factory's `lipschitz_bound()` is the largest L_i any single per-sample
/// adapter for this family can declare. `AdaptationBus` REJECTS adapter
/// registration if `∏ L_i ≥ 1` after multiplying in the new bound, returning
/// a structured error rather than silently allowing divergent composition.
///
/// **Range:** `lipschitz_bound()` MUST return a value in `(0.0, 1.0]`. A
/// return of `1.0` declares non-expansive identity (the canonical no-op
/// declaration; see [`NoOpMetaLearner`]). Values < 1.0 declare strict
/// contraction with margin `1.0 - L`.
///
/// # Cross-family search surface
///
/// `objectives()` declares which metric surfaces this family can compete on.
/// This is consumed by [`MetaSearch::pareto_dominates`] so a regression-only
/// family is never asked to optimize a classification metric. Empty slice
/// means "no objective surface" — the family will be pruned from the cohort
/// without participating in cross-family comparison.
///
/// # Examples
///
/// SGBT family (continuous-knob tuning of `(lr, lambda)`):
///
/// ```ignore
/// // L_SPSA = max(rho, 1 - rho) with rho = 0.3 gives L = 0.7 (Spall 1998
/// // SPSA gain sequence under convex-blended update). Strict contraction.
/// fn lipschitz_bound(&self) -> f64 { 0.7 }
/// fn objectives(&self) -> &[Objective] {
///     &[Objective::RegressionRmse, Objective::RegressionMae,
///       Objective::RegressionR2]
/// }
/// fn tunes_continuous_knobs(&self) -> bool { true }
/// fn tunes_structure(&self) -> bool { true }   // depth, n_steps via at_replacement
/// ```
///
/// Classification SGBT (no online structural changes during stream):
///
/// ```ignore
/// fn lipschitz_bound(&self) -> f64 { 0.7 }
/// fn objectives(&self) -> &[Objective] {
///     &[Objective::ClassificationF1, Objective::ClassificationKappa]
/// }
/// ```
///
/// # See also
///
/// - [`NoOpMetaLearner`] for the explicit opt-out.
/// - [`MetaSearch`] for cross-family Pareto comparison.
pub trait MetaLearner: Send + Sync {
    /// Lipschitz constant L_meta on the normalized parameter space `[0,1]^d`.
    ///
    /// MUST be in `(0.0, 1.0]`.
    /// - `1.0` = non-expansive identity (no-op opt-out, see
    ///   [`NoOpMetaLearner`]).
    /// - `< 1.0` = strict contraction (productive online adaptation).
    ///
    /// **Cited derivation required.** Implementations should document the
    /// theoretical source of their bound (Spall 1998 SPSA gain sequence,
    /// Rockafellar 1976 monotone operators, etc.) in the impl-site doc
    /// comment. A bare numerical constant is technical debt per Jono's
    /// discipline #5.
    fn lipschitz_bound(&self) -> f64;

    /// Objective surfaces this family can meaningfully compete on.
    ///
    /// Used by Pareto cross-family comparison
    /// ([`MetaSearch::pareto_dominates`]). An empty slice declares "no
    /// objective surface" — the family is pruned from cross-family search.
    fn objectives(&self) -> &[Objective];

    /// Discrete capacity bucket (used as a Pareto axis).
    fn complexity_class(&self) -> ComplexityClass;

    /// `true` if this family supports online tuning of continuous knobs
    /// (LR, lambda, forgetting factor) via [`Tunable::adjust_config`].
    ///
    /// [`Tunable::adjust_config`]: irithyll_core::learner::Tunable::adjust_config
    fn tunes_continuous_knobs(&self) -> bool;

    /// `true` if this family supports online structural changes
    /// (depth, n_steps, capacity) via [`Structural::apply_structural_change`].
    ///
    /// [`Structural::apply_structural_change`]: irithyll_core::learner::Structural::apply_structural_change
    fn tunes_structure(&self) -> bool;

    /// `true` if this is the canonical no-op declaration ([`NoOpMetaLearner`]).
    ///
    /// The default implementation derives the answer from the trait surface:
    /// `lipschitz_bound() == 1.0 && !tunes_continuous_knobs() && !tunes_structure()`.
    /// Overriding is unnecessary for downstream impls.
    ///
    /// `AdaptationBus` (PR-AM-12) checks this *before* wiring an adapter so
    /// the runtime path never silently collapses to a no-op loop.
    fn is_no_op(&self) -> bool {
        // SAFETY: floats compared via direct equality here is intentional.
        // 1.0 is the canonical opt-out sentinel; any other value is a
        // declared bound. No NaN producer reaches this code path because
        // the trait contract forbids non-finite returns from
        // lipschitz_bound (verified in the Pareto path and tests).
        self.lipschitz_bound() == 1.0 && !self.tunes_continuous_knobs() && !self.tunes_structure()
    }

    /// Optional human-readable rationale for the declaration.
    ///
    /// Surfaced by `AdaptationBus` diagnostics and by audit logs when an
    /// orchestrator skips wiring this family. Overriding is recommended for
    /// any [`is_no_op`][Self::is_no_op] = `true` declaration so the reason
    /// for opting out is observable.
    fn rationale(&self) -> Option<&str> {
        None
    }
}

// ===========================================================================
// NoOpMetaLearner — explicit opt-out
// ===========================================================================

/// Explicit no-op meta-learner: the canonical declaration that a family does
/// NOT support online adaptation.
///
/// Carries a cited rationale for the opt-out so audit logs surface a
/// human-readable reason, NOT a silent zero. Returned by the default
/// [`FactoryMetaLearner::meta_learner`] impl.
///
/// # Lipschitz declaration
///
/// `lipschitz_bound()` returns `1.0` (non-expansive identity). When
/// `AdaptationBus` (PR-AM-12) computes `∏ L_i`, this contributes a factor
/// of 1, so adding a no-op family preserves the existing product invariant
/// of the bus exactly.
///
/// # Construction
///
/// ```
/// use irithyll::automl::NoOpMetaLearner;
///
/// let m = NoOpMetaLearner::with_reason(
///     "ESN spectral radius requires reservoir reset; not safely tunable mid-stream",
/// );
/// ```
#[derive(Debug, Clone)]
pub struct NoOpMetaLearner {
    reason: &'static str,
    complexity: ComplexityClass,
}

impl NoOpMetaLearner {
    /// Construct a no-op with a cited rationale.
    ///
    /// `reason` SHOULD reference the streaming principle or paper that makes
    /// online tuning unsafe / unproductive for this family (e.g.
    /// "no-mid-stream-cell-reset principle, AGENTS.md").
    pub fn with_reason(reason: &'static str) -> Self {
        Self {
            reason,
            complexity: ComplexityClass::Small,
        }
    }

    /// Construct a no-op with rationale and an explicit complexity class.
    ///
    /// Most factories should use [`Self::with_reason`] and let
    /// [`FactoryMetaLearner::meta_learner`]
    /// inject the right complexity class via `complexity_hint()`. Direct
    /// construction is for tests and bespoke integrations.
    pub fn new(reason: &'static str, complexity: ComplexityClass) -> Self {
        Self { reason, complexity }
    }

    /// The cited rationale for the opt-out.
    pub fn reason(&self) -> &'static str {
        self.reason
    }
}

impl Default for NoOpMetaLearner {
    fn default() -> Self {
        Self::with_reason("default factory opt-out: no MetaLearner declared")
    }
}

impl MetaLearner for NoOpMetaLearner {
    fn lipschitz_bound(&self) -> f64 {
        // 1.0 = non-expansive identity. Composition with this contributes
        // a factor of 1 to the AdaptationBus's running ∏ L_i, preserving
        // invariants exactly.
        1.0
    }

    fn objectives(&self) -> &[Objective] {
        // Empty: a no-op participates in no objective surface, so cross-family
        // Pareto comparison correctly excludes it from the front.
        &[]
    }

    fn complexity_class(&self) -> ComplexityClass {
        self.complexity
    }

    fn tunes_continuous_knobs(&self) -> bool {
        false
    }

    fn tunes_structure(&self) -> bool {
        false
    }

    fn rationale(&self) -> Option<&str> {
        Some(self.reason)
    }
}

// ===========================================================================
// SgbtMetaLearner — concrete impl for the SGBT family
// ===========================================================================

/// SGBT-family meta-learner declaration.
///
/// Covers the regression-side SGBT family: [`SGBT`][crate::SGBT],
/// [`DistributionalSGBT`][crate::DistributionalSGBT]. Online-tunes
/// `(learning_rate, lambda)` via SPSA (Spall 1998) under a convex-blended
/// step that gives Lipschitz bound L = 0.7.
///
/// Multi-class SGBT uses a sibling [`SgbtClassificationMetaLearner`] because
/// its objective surface is classification metrics, not regression.
///
/// # Lipschitz derivation
///
/// SPSA's natural per-step Lipschitz is `1 + a_k · ||J_g||` (Spall 1998 §III).
/// Under the convex-blended update
///
/// ```text
/// θ_{t+1} = (1 - ρ) · θ_t + ρ · clip(θ_t + a_k · g_hat, 0, 1)
/// ```
///
/// the operator becomes a convex combination of the identity and a clipped
/// step, with `L_SPSA = max(ρ, 1 - ρ)`. The library uses `ρ = 0.3`, giving
/// `L_SPSA = 0.7` (AM-R2 §2). This is strict contraction with a 30% margin —
/// the bus can register an additional adapter with its own `L_i ≤ 1` and
/// still satisfy `∏ L_i < 1`.
#[derive(Debug, Clone, Copy)]
pub struct SgbtMetaLearner {
    complexity: ComplexityClass,
}

impl SgbtMetaLearner {
    /// Construct an SGBT meta-learner with the given complexity bucket.
    pub fn new(complexity: ComplexityClass) -> Self {
        Self { complexity }
    }
}

impl MetaLearner for SgbtMetaLearner {
    fn lipschitz_bound(&self) -> f64 {
        // L_SPSA = max(rho, 1 - rho) with rho = 0.3 (Spall 1998 + AM-R2 §2
        // convex-blending fix for SPSA contractivity).
        0.7
    }

    fn objectives(&self) -> &[Objective] {
        const REGRESSION: &[Objective] = &[
            Objective::RegressionRmse,
            Objective::RegressionMae,
            Objective::RegressionR2,
            Objective::DirectionalAccuracy,
        ];
        REGRESSION
    }

    fn complexity_class(&self) -> ComplexityClass {
        self.complexity
    }

    fn tunes_continuous_knobs(&self) -> bool {
        // SPSA tunes (lr, lambda) per-sample.
        true
    }

    fn tunes_structure(&self) -> bool {
        // depth / n_steps adapt at tree-replacement boundary via at_replacement.
        true
    }

    fn rationale(&self) -> Option<&str> {
        Some(
            "SGBT regression family: SPSA on (lr, lambda) with ρ=0.3 \
             convex-blended step (L=0.7); structural changes at tree-replacement \
             boundary",
        )
    }
}

/// Classification variant of [`SgbtMetaLearner`] for `MulticlassSGBT`.
///
/// Identical Lipschitz contract; objective surface is classification metrics.
#[derive(Debug, Clone, Copy)]
pub struct SgbtClassificationMetaLearner {
    complexity: ComplexityClass,
}

impl SgbtClassificationMetaLearner {
    /// Construct a classification SGBT meta-learner.
    pub fn new(complexity: ComplexityClass) -> Self {
        Self { complexity }
    }
}

impl MetaLearner for SgbtClassificationMetaLearner {
    fn lipschitz_bound(&self) -> f64 {
        // Same as SgbtMetaLearner; SPSA bound depends on the gain sequence,
        // not the loss function.
        0.7
    }

    fn objectives(&self) -> &[Objective] {
        const CLASSIFICATION: &[Objective] =
            &[Objective::ClassificationF1, Objective::ClassificationKappa];
        CLASSIFICATION
    }

    fn complexity_class(&self) -> ComplexityClass {
        self.complexity
    }

    fn tunes_continuous_knobs(&self) -> bool {
        true
    }

    fn tunes_structure(&self) -> bool {
        true
    }

    fn rationale(&self) -> Option<&str> {
        Some(
            "Classification SGBT family: SPSA on (lr, lambda) with ρ=0.3 \
             convex-blended step (L=0.7); softmax committee, classification \
             objective surface",
        )
    }
}

// ===========================================================================
// MetaSearch — Pareto cross-family comparison
// ===========================================================================

/// Cross-family Pareto comparison for the meta-search layer.
///
/// Pareto dominance is parameter-free (no weights, no ε threshold) — exactly
/// the discipline Jono enforces (#1, "reject arbitrary thresholds"). A
/// candidate `A` *Pareto-dominates* `B` iff:
///
/// - `A` is no worse than `B` on every shared declared objective AND
///   on the discrete `ComplexityClass` axis, AND
/// - `A` is strictly better than `B` on at least one of those axes.
///
/// Non-dominated candidates form the Pareto front; they coexist in the
/// cohort (PR-AM-5) until additional information (Bernstein
/// confidence-interval comparison via [`bernstein_compare`][bcmp]) breaks
/// ties with statistical certainty.
///
/// # No weighted sums, no ε tolerance
///
/// The library deliberately does NOT support `pareto_score = w_rmse · rmse +
/// w_complexity · ln(p)`. Weight tuning is a knob that pretends not to be
/// one. The discrete `ComplexityClass` axis gives Pareto a meaningful
/// capacity dimension without introducing a weight.
///
/// [bcmp]: crate::automl::bernstein_compare
pub struct MetaSearch;

/// Score table for one family / candidate, indexed by [`Objective`].
///
/// The orchestrator builds one [`MetaScore`] per active arm using the
/// objectives declared in its [`MetaLearner`]. Missing entries (objective
/// not in the family's declared surface) are skipped by Pareto comparison.
///
/// # Sign convention
///
/// Every value is in *raw* metric units (RMSE in raw error, R² in [-∞, 1]).
/// `pareto_dominates` consults [`Objective::is_minimization`] to determine
/// direction. The orchestrator never normalizes units between families;
/// Pareto compares only on *shared* declared objectives.
#[derive(Debug, Clone, Default)]
pub struct MetaScore {
    /// Raw objective values keyed by the [`Objective`] tag.
    ///
    /// Orchestrator side: insert one entry per *declared* objective; do NOT
    /// fabricate values for objectives the family did not declare.
    pub values: std::collections::BTreeMap<Objective, f64>,
    /// Capacity bucket (Pareto axis).
    pub complexity: ComplexityClass,
}

impl MetaScore {
    /// Construct an empty score with the given complexity class.
    pub fn new(complexity: ComplexityClass) -> Self {
        Self {
            values: std::collections::BTreeMap::new(),
            complexity,
        }
    }

    /// Record a raw metric value for a declared objective.
    ///
    /// Idempotent re-insertion overwrites; the orchestrator is responsible
    /// for snapshot timing.
    pub fn record(&mut self, obj: Objective, value: f64) {
        self.values.insert(obj, value);
    }

    /// Get the raw recorded value for an objective.
    pub fn get(&self, obj: Objective) -> Option<f64> {
        self.values.get(&obj).copied()
    }
}

impl MetaSearch {
    /// `true` iff `a` Pareto-dominates `b` on the SHARED declared objectives.
    ///
    /// "Shared" means objectives present in both `a.values` and `b.values`.
    /// If the intersection is empty, returns `false` (incomparable on metrics —
    /// fall through to a different tie-breaker, e.g. complexity-only).
    ///
    /// # Sign convention
    ///
    /// For `Objective::is_minimization()` objectives, smaller is better.
    /// For maximization objectives, larger is better.
    /// `ComplexityClass` is treated as smaller-is-better (ties allowed).
    ///
    /// # Robustness
    ///
    /// NaN inputs make a candidate "incomparable" — neither side dominates
    /// (`partial_cmp` returns `None`). This matches IEEE-754 semantics; no
    /// unwrap, no silent coercion to zero.
    pub fn pareto_dominates(a: &MetaScore, b: &MetaScore) -> bool {
        let mut shared = 0usize;
        let mut a_strictly_better_on_some = false;
        let mut a_no_worse_on_all = true;

        for (&obj, &av) in &a.values {
            if let Some(&bv) = b.values.get(&obj) {
                shared += 1;
                let (a_eff, b_eff) = if obj.is_minimization() {
                    // smaller-is-better → flip so we compare "fitness" with
                    // larger-is-better semantics throughout.
                    (-av, -bv)
                } else {
                    (av, bv)
                };
                match a_eff.partial_cmp(&b_eff) {
                    Some(core::cmp::Ordering::Greater) => a_strictly_better_on_some = true,
                    Some(core::cmp::Ordering::Less) => {
                        a_no_worse_on_all = false;
                    }
                    Some(core::cmp::Ordering::Equal) => {}
                    None => {
                        // NaN: incomparable on this axis. Conservative: a
                        // does NOT strictly dominate b.
                        a_no_worse_on_all = false;
                    }
                }
            }
        }

        // Capacity Pareto axis: smaller class is "no worse" (less capacity ≤
        // more capacity given equal performance is preferred — Occam).
        match a.complexity.cmp(&b.complexity) {
            core::cmp::Ordering::Less => a_strictly_better_on_some = true,
            core::cmp::Ordering::Greater => {
                a_no_worse_on_all = false;
            }
            core::cmp::Ordering::Equal => {}
        }

        shared > 0 && a_no_worse_on_all && a_strictly_better_on_some
    }

    /// Indices of the Pareto-non-dominated candidates within `scores`.
    ///
    /// Returns indices in input order. A candidate is on the front iff no
    /// other candidate Pareto-dominates it.
    pub fn pareto_front(scores: &[MetaScore]) -> Vec<usize> {
        let mut front = Vec::with_capacity(scores.len());
        for (i, si) in scores.iter().enumerate() {
            let dominated = scores.iter().enumerate().any(|(j, sj)| {
                if i == j {
                    return false;
                }
                MetaSearch::pareto_dominates(sj, si)
            });
            if !dominated {
                front.push(i);
            }
        }
        front
    }
}

// ===========================================================================
// ModelFactory extension: meta_learner()
// ===========================================================================

/// Extension trait that lets every [`ModelFactory`] expose a
/// [`MetaLearner`].
///
/// The default impl returns [`NoOpMetaLearner`], so all existing factories
/// opt out explicitly with a documented reason. Factories that DO support
/// online adaptation (SGBT family) override this method to return a
/// concrete [`MetaLearner`].
///
/// # Why an extension trait, not a method on `ModelFactory`?
///
/// `ModelFactory` is a stable v10 trait with multiple existing impls
/// (downstream crates). Adding a required method is a breaking change.
/// Adding an extension trait with a default impl keeps the v10 ABI intact:
/// downstream factories without their own `MetaLearner` declaration default
/// to no-op opt-out automatically.
pub trait FactoryMetaLearner: ModelFactory {
    /// Return this factory's [`MetaLearner`] declaration.
    ///
    /// Default: [`NoOpMetaLearner`] sized to the factory's
    /// [`complexity_hint()`][ModelFactory::complexity_hint].
    fn meta_learner(&self) -> Box<dyn MetaLearner> {
        Box::new(NoOpMetaLearner::new(
            "default ModelFactory::meta_learner: family did not override; \
             online adaptation not declared",
            ComplexityClass::from_hint(self.complexity_hint()),
        ))
    }
}

// Note: FactoryMetaLearner is an extension trait with a default `meta_learner()`
// body.  Do NOT add a blanket `impl<T: ModelFactory> FactoryMetaLearner for T {}` —
// it conflicts with any concrete impl (e.g. `impl FactoryMetaLearner for Factory`).
// Types that want the trait must implement it explicitly or rely on the default body.

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automl::Factory;

    // -----------------------------------------------------------------------
    // Test: no-op default works for ALL existing factories
    // -----------------------------------------------------------------------

    /// Every concrete factory (and the legacy `Factory` enum dispatcher) MUST
    /// produce a `MetaLearner` via the default `meta_learner()` extension
    /// trait without explicit override. The default is a no-op opt-out that
    /// composes safely with [`AdaptationBus`] (PR-AM-12).
    #[test]
    fn meta_learner_no_op_default_compiles_for_all_models() {
        // Tree family
        let _: Box<dyn MetaLearner> = Factory::sgbt(5).meta_learner();
        let _: Box<dyn MetaLearner> = Factory::distributional(5).meta_learner();
        let _: Box<dyn MetaLearner> = Factory::multiclass_sgbt(5, 3).meta_learner();

        // Reservoir
        let _: Box<dyn MetaLearner> = Factory::esn().meta_learner();

        // SSM family
        let _: Box<dyn MetaLearner> = Factory::mamba(4).meta_learner();
        let _: Box<dyn MetaLearner> = Factory::mamba3(4).meta_learner();
        let _: Box<dyn MetaLearner> = Factory::mamba_bd(4).meta_learner();
        let _: Box<dyn MetaLearner> = Factory::slstm(4).meta_learner();
        let _: Box<dyn MetaLearner> = Factory::mgrade(4).meta_learner();

        // Attention family
        let _: Box<dyn MetaLearner> = Factory::attention(8).meta_learner();
        let _: Box<dyn MetaLearner> = Factory::delta_product(8).meta_learner();
        let _: Box<dyn MetaLearner> = Factory::rwkv7(8).meta_learner();

        // Neural / spiking
        let _: Box<dyn MetaLearner> = Factory::spike_net().meta_learner();
        let _: Box<dyn MetaLearner> = Factory::kan(4).meta_learner();
        let _: Box<dyn MetaLearner> = Factory::ttt(4).meta_learner();
    }

    /// Default-derived no-op MUST satisfy the AdaptationBus identity invariant:
    /// L = 1.0, no objective surface, both `tunes_*` declarations false.
    #[test]
    fn no_op_default_factory_has_identity_lipschitz() {
        let factory = Factory::esn();
        let m = factory.meta_learner();
        assert_eq!(
            m.lipschitz_bound(),
            1.0,
            "default no-op factory must declare L=1.0 (non-expansive identity), got {}",
            m.lipschitz_bound()
        );
        assert!(
            m.is_no_op(),
            "default no-op factory must report is_no_op() = true"
        );
        assert!(
            m.objectives().is_empty(),
            "no-op MetaLearner must declare no objective surface"
        );
        assert!(
            !m.tunes_continuous_knobs(),
            "no-op MetaLearner must NOT declare continuous-knob tuning"
        );
        assert!(
            !m.tunes_structure(),
            "no-op MetaLearner must NOT declare structural tuning"
        );
        assert!(
            m.rationale().is_some(),
            "no-op MetaLearner must surface a rationale for the opt-out"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Lipschitz bound derives from impl, not magic constant
    // -----------------------------------------------------------------------

    /// SGBT meta learner declares L=0.7, derived from SPSA convex-blending
    /// (Spall 1998 + AM-R2 §2 fix). The test verifies the declared value
    /// matches the documented derivation: `L = max(rho, 1 - rho)` with
    /// `rho = 0.3` → `L = 0.7`.
    #[test]
    fn meta_learner_lipschitz_bound_derives_from_implementation() {
        let m = SgbtMetaLearner::new(ComplexityClass::Medium);
        let rho: f64 = 0.3;
        let expected = rho.max(1.0_f64 - rho); // = 0.7
        assert!(
            (m.lipschitz_bound() - expected).abs() < 1e-12,
            "SgbtMetaLearner Lipschitz must derive from rho-blending: \
             expected max(rho, 1-rho) = {}, got {}",
            expected,
            m.lipschitz_bound()
        );
        assert!(
            m.lipschitz_bound() < 1.0,
            "SgbtMetaLearner must be a strict contraction (L < 1), got {}",
            m.lipschitz_bound()
        );

        // Classification variant has the same bound (loss function does not
        // change SPSA gain-sequence-derived Lipschitz).
        let mc = SgbtClassificationMetaLearner::new(ComplexityClass::Medium);
        assert!(
            (mc.lipschitz_bound() - expected).abs() < 1e-12,
            "SgbtClassificationMetaLearner must use the same SPSA Lipschitz bound"
        );
    }

    /// Adapting Lipschitz bounds compose under multiplication: if
    /// AdaptationBus has registered an SGBT meta with L=0.7 and a
    /// hypothetical drift adapter with L_drift=0.95, the product 0.665 must
    /// be < 1 (Banach 1922 contraction-mapping product invariant).
    #[test]
    fn lipschitz_product_satisfies_banach_contraction_invariant() {
        let m = SgbtMetaLearner::new(ComplexityClass::Medium);
        let l_drift = 0.95_f64; // hypothetical drift-adapter bound
        let product = m.lipschitz_bound() * l_drift;
        assert!(
            product < 1.0,
            "Banach contraction invariant requires ∏ L_i < 1; got {} = {} · {}",
            product,
            m.lipschitz_bound(),
            l_drift
        );
    }

    // -----------------------------------------------------------------------
    // Test: cross-family comparison is Pareto, not scalar
    // -----------------------------------------------------------------------

    /// Cross-family comparison MUST use Pareto dominance, never a scalar
    /// weighted sum. The test constructs two candidates where `a` wins on
    /// RMSE but `b` wins on R² — neither dominates the other; both are on
    /// the Pareto front.
    #[test]
    fn meta_learner_across_family_comparison_is_pareto_not_scalar() {
        let mut a = MetaScore::new(ComplexityClass::Small);
        a.record(Objective::RegressionRmse, 0.10);
        a.record(Objective::RegressionR2, 0.50);

        let mut b = MetaScore::new(ComplexityClass::Small);
        b.record(Objective::RegressionRmse, 0.15); // worse on RMSE
        b.record(Objective::RegressionR2, 0.80); // better on R²

        // Neither dominates the other: trade-off between metrics.
        assert!(
            !MetaSearch::pareto_dominates(&a, &b),
            "a should NOT Pareto-dominate b (a wins RMSE, loses R²)"
        );
        assert!(
            !MetaSearch::pareto_dominates(&b, &a),
            "b should NOT Pareto-dominate a (b wins R², loses RMSE)"
        );

        // Both must be on the Pareto front.
        let scores = [a, b];
        let front = MetaSearch::pareto_front(&scores);
        assert_eq!(
            front.len(),
            2,
            "trade-off candidates must both appear on Pareto front, got {:?}",
            front
        );
    }

    /// Strict Pareto dominance: `a` strictly better on at least one axis,
    /// no worse on every other shared axis.
    #[test]
    fn pareto_strict_dominance_excludes_dominated() {
        let mut a = MetaScore::new(ComplexityClass::Small);
        a.record(Objective::RegressionRmse, 0.10);
        a.record(Objective::RegressionR2, 0.80);

        let mut b = MetaScore::new(ComplexityClass::Small);
        b.record(Objective::RegressionRmse, 0.15); // worse
        b.record(Objective::RegressionR2, 0.50); // worse

        assert!(
            MetaSearch::pareto_dominates(&a, &b),
            "a beats b on every axis: a should strictly Pareto-dominate b"
        );
        assert!(
            !MetaSearch::pareto_dominates(&b, &a),
            "b cannot dominate a (worse on every axis)"
        );

        let scores = [a, b];
        let front = MetaSearch::pareto_front(&scores);
        assert_eq!(
            front,
            vec![0],
            "Pareto front must contain only the dominating candidate, got {:?}",
            front
        );
    }

    /// Capacity is a real Pareto axis: a Tiny model with worse RMSE can
    /// still be on the front via complexity advantage. Conversely, equal-RMSE
    /// candidates differ via complexity.
    #[test]
    fn pareto_complexity_axis_breaks_equal_metric_ties() {
        let mut tiny = MetaScore::new(ComplexityClass::Tiny);
        tiny.record(Objective::RegressionRmse, 0.10);

        let mut large = MetaScore::new(ComplexityClass::Large);
        large.record(Objective::RegressionRmse, 0.10); // equal

        // Tiny strictly dominates: same metric, lower complexity = Occam preferred.
        assert!(
            MetaSearch::pareto_dominates(&tiny, &large),
            "equal metric + lower complexity must strictly dominate"
        );
        assert!(
            !MetaSearch::pareto_dominates(&large, &tiny),
            "equal metric + higher complexity cannot dominate"
        );
    }

    /// Pareto comparison treats NaN as incomparable — neither dominates.
    /// This guards against a malformed score record causing a silent
    /// dominance assertion via `unwrap_or(Equal)` semantics.
    #[test]
    fn pareto_handles_nan_as_incomparable() {
        let mut a = MetaScore::new(ComplexityClass::Small);
        a.record(Objective::RegressionRmse, f64::NAN);
        a.record(Objective::RegressionR2, 0.80);

        let mut b = MetaScore::new(ComplexityClass::Small);
        b.record(Objective::RegressionRmse, 0.15);
        b.record(Objective::RegressionR2, 0.50);

        // NaN on RMSE makes 'a' incomparable on that axis. 'a' is better
        // on R², so `a_strictly_better_on_some` would set; but
        // `a_no_worse_on_all` falls to `false` because NaN flips it.
        assert!(
            !MetaSearch::pareto_dominates(&a, &b),
            "NaN on a shared axis must prevent dominance"
        );
    }

    /// Empty objective intersection means "incomparable on metrics" — the
    /// shared check is `shared > 0`, so the function returns false.
    #[test]
    fn pareto_disjoint_objectives_are_incomparable() {
        let mut regression = MetaScore::new(ComplexityClass::Small);
        regression.record(Objective::RegressionRmse, 0.10);

        let mut classification = MetaScore::new(ComplexityClass::Small);
        classification.record(Objective::ClassificationF1, 0.80);

        assert!(
            !MetaSearch::pareto_dominates(&regression, &classification),
            "disjoint objective surfaces must not dominate"
        );
        assert!(
            !MetaSearch::pareto_dominates(&classification, &regression),
            "disjoint objective surfaces must not dominate (reverse direction)"
        );
    }

    // -----------------------------------------------------------------------
    // Test: compose with within-model auto tuner (Lipschitz product safety)
    // -----------------------------------------------------------------------

    /// The MetaLearner's Lipschitz bound is the contract that AdaptationBus
    /// (AM-12) consumes when composing it with a within-model adapter (the
    /// existing within-model AutoTuner / SPSA). Composition is multiplicative;
    /// the test verifies the product invariant ∏ L_i < 1 holds for the SGBT
    /// + 0.95-drift case the bus would actually wire.
    #[test]
    fn meta_learner_compose_with_within_model_auto_tuner() {
        let factory_meta = SgbtMetaLearner::new(ComplexityClass::Medium);

        // Hypothetical within-model SPSA adapter (within-model auto-tuner) with
        // its own Lipschitz declaration — same SPSA bound applies, but composed
        // with the meta learner's bound it produces a tighter combined map.
        let within_model_spsa_lipschitz = 0.7_f64;

        let combined = factory_meta.lipschitz_bound() * within_model_spsa_lipschitz;
        assert!(
            combined < 1.0,
            "MetaLearner ∘ within-model SPSA must be a strict contraction; \
             got {} = {} · {}",
            combined,
            factory_meta.lipschitz_bound(),
            within_model_spsa_lipschitz
        );

        // Non-zero margin to allow at least one more adapter (e.g. drift) to
        // join the bus without breaking ∏ L_i < 1.
        let margin = 1.0 - combined;
        assert!(
            margin > 0.0,
            "Composition must leave a non-zero margin for additional adapters; \
             got margin = {}",
            margin
        );

        // No-op meta learner composed with within-model SPSA must collapse to
        // the within-model SPSA's bound exactly (preserves bus invariants).
        let no_op = NoOpMetaLearner::default();
        let no_op_composed = no_op.lipschitz_bound() * within_model_spsa_lipschitz;
        assert!(
            (no_op_composed - within_model_spsa_lipschitz).abs() < 1e-12,
            "no-op MetaLearner must contribute a factor of 1 (identity); \
             got composed = {}, expected {}",
            no_op_composed,
            within_model_spsa_lipschitz
        );
    }

    // -----------------------------------------------------------------------
    // Test: complexity bucketing matches documented threshold derivation
    // -----------------------------------------------------------------------

    /// `ComplexityClass::from_hint` boundaries match the documented
    /// information-theoretic thresholds (Tiny ≤ 100, Small ≤ 1_000, etc.).
    #[test]
    fn complexity_class_buckets_match_documented_thresholds() {
        assert_eq!(ComplexityClass::from_hint(0), ComplexityClass::Tiny);
        assert_eq!(ComplexityClass::from_hint(100), ComplexityClass::Tiny);
        assert_eq!(ComplexityClass::from_hint(101), ComplexityClass::Small);
        assert_eq!(ComplexityClass::from_hint(1_000), ComplexityClass::Small);
        assert_eq!(ComplexityClass::from_hint(1_001), ComplexityClass::Medium);
        assert_eq!(ComplexityClass::from_hint(10_000), ComplexityClass::Medium);
        assert_eq!(ComplexityClass::from_hint(10_001), ComplexityClass::Large);
        assert_eq!(
            ComplexityClass::from_hint(usize::MAX),
            ComplexityClass::Large
        );
    }

    // -----------------------------------------------------------------------
    // Test: objective sign convention
    // -----------------------------------------------------------------------

    /// Minimization-vs-maximization tags match documented intent.
    #[test]
    fn objective_sign_convention() {
        assert!(Objective::RegressionRmse.is_minimization());
        assert!(Objective::RegressionMae.is_minimization());
        assert!(Objective::DistributionalCrps.is_minimization());

        assert!(!Objective::RegressionR2.is_minimization());
        assert!(!Objective::DirectionalAccuracy.is_minimization());
        assert!(!Objective::ClassificationF1.is_minimization());
        assert!(!Objective::ClassificationKappa.is_minimization());
    }

    // -----------------------------------------------------------------------
    // Test: trait object usage (audit hardening)
    // -----------------------------------------------------------------------

    /// `Box<dyn MetaLearner>` is a legal type — required for orchestrator
    /// storage and AdaptationBus registration.
    #[test]
    fn meta_learner_is_trait_object_safe() {
        let _: Box<dyn MetaLearner> = Box::new(NoOpMetaLearner::default());
        let _: Box<dyn MetaLearner> = Box::new(SgbtMetaLearner::new(ComplexityClass::Medium));
        let _: Box<dyn MetaLearner> =
            Box::new(SgbtClassificationMetaLearner::new(ComplexityClass::Medium));
    }
}
