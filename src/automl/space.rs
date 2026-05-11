//! Typed search-space API for AutoML factories.
//!
//! Replaces the legacy positional [`HyperConfig`][crate::automl::HyperConfig]
//! (a [`Vec<f64>`] indexed by position) with a named, typed surface:
//!
//! - [`SearchSpace`] declares which parameters a factory accepts and their
//!   ranges, optional log-scaling, conditional activation, and cross-parameter
//!   constraints.
//! - [`ParamMap`] is the sampled point — a name->value map produced by the
//!   sampler. Factories read values by name (`params.float("learning_rate")?`)
//!   instead of by position (`config.get(0)`).
//! - [`ParamDef`] is the declarative parameter type, constructed via free
//!   functions ([`linear_range`], [`log_range`], [`int_range`],
//!   [`categorical`]).
//!
//! # Why named, typed access
//!
//! Positional access was the silent-bug attractor of the previous API.
//! Inserting a new parameter mid-space silently shifted every downstream
//! `config.get(i)` call. Comments like `// | 0 | learning_rate |` had to be
//! kept in sync by hand. With named access:
//!
//! - The factory states what it expects: `params.float("learning_rate")?`.
//! - Missing-name and wrong-type are explicit, returned errors — never silent
//!   reads of the wrong slot.
//! - The space is inspectable as a data structure (iterating
//!   `space.params()` returns `&ParamDef`), enabling serialization and
//!   meta-learning that the closure-based define-by-run alternative
//!   (Optuna-style) cannot offer.
//!
//! # Conditional parameters
//!
//! Conditional parameters are declared with [`SearchSpaceBuilder::conditional`]
//! and a [`Condition`] built via the [`when`] DSL:
//!
//! ```text
//!   when("use_complex").equals("true")
//!   when("model_kind").in_values(&["svm", "rf"])
//!   when("threshold").greater_than(0.5)
//! ```
//!
//! When a conditional parameter's condition evaluates false, the parameter is
//! **absent** from the [`ParamMap`] (not present with a sentinel default).
//! Factories must use [`ParamMap::float_optional`] (or the int / category
//! optional variants) for conditional reads.
//!
//! # Constraints
//!
//! Cross-parameter invariants like `n_heads divides d_model` are expressed as
//! [`Constraint`] closures that receive the [`ParamMap`] under construction
//! and return `true` for feasible combinations:
//!
//! ```text
//!   .constraint("heads_divide_d_model", |c| {
//!       c.int_unchecked("d_model") % c.int_unchecked("n_heads") == 0
//!   })
//! ```
//!
//! At sample time, the sampler uses **rejection sampling**: draw a candidate,
//! check feasibility, discard and retry on violation. After
//! `MAX_REJECTION_ATTEMPTS = 100` consecutive rejections the sampler returns
//! [`SamplerError::ConstraintUnsatisfiable`] — the racing layer treats this
//! as an evicted-this-round factory, not a panic.
//!
//! # References
//!
//! - Bischl et al. (2023). "Hyperparameter optimization: Foundations,
//!   algorithms, best practices, and open challenges." WIREs DM&KD.
//! - automl/ConfigSpace. Python DSL for configuration spaces.
//!   <https://automl.github.io/ConfigSpace/>
//! - López-Ibáñez et al. (2016). "The irace Package." Operations Research
//!   Perspectives. <https://mlopez-ibanez.github.io/irace/>

use core::fmt;
use std::collections::BTreeMap;
use std::sync::Arc;

use irithyll_core::rng::{standard_normal, xorshift64, xorshift64_f64};

use super::FactoryError;

// ---------------------------------------------------------------------------
// Constants (named, derived where possible)
// ---------------------------------------------------------------------------

/// Maximum rejection-sampling attempts before declaring a constraint
/// unsatisfiable.
///
/// Chosen to bound the worst-case wall-clock cost of a single sample at 100
/// rejections — at typical low-dimensional spaces (≤10 params) and practical
/// constraint geometries (e.g. `d_model % n_heads == 0` rejects roughly half
/// of candidates), 100 attempts gives a probability ≤ `2^-100` of false
/// rejection for any feasible space, which is far below any realistic
/// failure-mode rate. If 100 fails, the constraint geometry is genuinely
/// degenerate — a hard error is the right signal.
pub const MAX_REJECTION_ATTEMPTS: usize = 100;

// ---------------------------------------------------------------------------
// ParamDef: declarative parameter type
// ---------------------------------------------------------------------------

/// Sampling scale for a continuous float parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    /// Linear sampling: uniform in `[low, high]`.
    Linear,
    /// Log-uniform sampling: uniform in `[ln(low), ln(high)]`, then `exp`.
    Log,
}

/// Declarative definition of a single search-space parameter.
///
/// Constructed via [`linear_range`], [`log_range`], [`int_range`], and
/// [`categorical`] free functions rather than struct literal syntax — the
/// free-function form is shorter at every call site and prevents users from
/// constructing inconsistent variants by hand (e.g. log-scaled ints).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ParamDef {
    /// Continuous float in `[low, high]` (linear or log).
    Float {
        /// Lower bound (inclusive).
        low: f64,
        /// Upper bound (inclusive).
        high: f64,
        /// Sampling scale (linear or log).
        scale: Scale,
    },
    /// Integer in `[low, high]` (inclusive).
    Int {
        /// Lower bound (inclusive).
        low: i64,
        /// Upper bound (inclusive).
        high: i64,
    },
    /// Categorical: the parameter value is one of `choices`.
    Categorical {
        /// Possible label values, in declaration order.
        choices: Vec<Category>,
    },
}

impl ParamDef {
    /// Verify the definition is well-formed.
    fn validate(&self, name: &str) -> Result<(), SpaceError> {
        match self {
            ParamDef::Float { low, high, scale } => {
                if !low.is_finite() || !high.is_finite() {
                    return Err(SpaceError::InvalidRange {
                        name: name.to_string(),
                        low: *low,
                        high: *high,
                    });
                }
                if low > high {
                    return Err(SpaceError::InvalidRange {
                        name: name.to_string(),
                        low: *low,
                        high: *high,
                    });
                }
                if matches!(scale, Scale::Log) && (*low <= 0.0 || *high <= 0.0) {
                    return Err(SpaceError::InvalidRange {
                        name: name.to_string(),
                        low: *low,
                        high: *high,
                    });
                }
                Ok(())
            }
            ParamDef::Int { low, high } => {
                if low > high {
                    return Err(SpaceError::InvalidIntRange {
                        name: name.to_string(),
                        low: *low,
                        high: *high,
                    });
                }
                Ok(())
            }
            ParamDef::Categorical { choices } => {
                if choices.is_empty() {
                    return Err(SpaceError::EmptyChoices(name.to_string()));
                }
                Ok(())
            }
        }
    }
}

/// Continuous float parameter on the linear scale, sampled uniformly in
/// `[low, high]`.
#[inline]
pub fn linear_range(low: f64, high: f64) -> ParamDef {
    ParamDef::Float {
        low,
        high,
        scale: Scale::Linear,
    }
}

/// Continuous float parameter on the log scale, sampled uniformly in
/// `[ln(low), ln(high)]` and then exponentiated.
///
/// Both `low` and `high` must be strictly positive; the validator will reject
/// non-positive bounds at [`SearchSpaceBuilder::build`] time.
#[inline]
pub fn log_range(low: f64, high: f64) -> ParamDef {
    ParamDef::Float {
        low,
        high,
        scale: Scale::Log,
    }
}

/// Integer parameter in `[low, high]` (inclusive), sampled uniformly.
#[inline]
pub fn int_range(low: i64, high: i64) -> ParamDef {
    ParamDef::Int { low, high }
}

/// Categorical parameter over the given labels.
///
/// Each label is converted to a [`Category`] via [`From`]. Common inputs:
///
/// - `categorical(&["gated", "ungated"])` for `&str` labels
/// - `categorical(&[1u32, 2, 4, 8])` for integer-valued choices
pub fn categorical<T: Clone + Into<Category>>(choices: &[T]) -> ParamDef {
    ParamDef::Categorical {
        choices: choices.iter().cloned().map(Into::into).collect(),
    }
}

// ---------------------------------------------------------------------------
// Category: opaque label for categorical values
// ---------------------------------------------------------------------------

/// An opaque categorical label.
///
/// Wraps an `Arc<str>` so cloning is cheap and labels can flow through the
/// sampler / [`ParamMap`] / factory pipeline without per-clone allocation.
/// Compared by string equality.
#[derive(Debug, Clone)]
pub struct Category(Arc<str>);

impl Category {
    /// The underlying string label.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl PartialEq for Category {
    fn eq(&self, other: &Self) -> bool {
        *self.0 == *other.0
    }
}

impl Eq for Category {}

impl PartialEq<str> for Category {
    fn eq(&self, other: &str) -> bool {
        &*self.0 == other
    }
}

impl PartialEq<&str> for Category {
    fn eq(&self, other: &&str) -> bool {
        &*self.0 == *other
    }
}

impl From<&str> for Category {
    fn from(s: &str) -> Self {
        Category(Arc::from(s))
    }
}

impl From<String> for Category {
    fn from(s: String) -> Self {
        Category(Arc::from(s.as_str()))
    }
}

// Common convenience: int-valued categoricals (e.g. attention head counts).
impl From<u32> for Category {
    fn from(n: u32) -> Self {
        Category(Arc::from(n.to_string().as_str()))
    }
}

impl From<usize> for Category {
    fn from(n: usize) -> Self {
        Category(Arc::from(n.to_string().as_str()))
    }
}

// ---------------------------------------------------------------------------
// Conditional activation
// ---------------------------------------------------------------------------

/// Predicate over parent-parameter values that gates a conditional parameter.
///
/// Build via the [`when`] DSL rather than constructing variants directly:
///
/// ```text
///   when("model_kind").equals("svm")
///   when("model_kind").in_values(&["svm", "rf"])
///   when("threshold").greater_than(0.5)
///   when("a").equals("x").and(when("b").equals("y"))
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Condition {
    /// `parent == value`.
    Equals {
        /// The parent parameter name.
        parent: String,
        /// The category the parent must equal.
        value: Category,
    },
    /// `parent ∈ values`.
    InValues {
        /// The parent parameter name.
        parent: String,
        /// The categories the parent may match.
        values: Vec<Category>,
    },
    /// `parent > threshold` (for float parents).
    FloatGt {
        /// The parent parameter name.
        parent: String,
        /// The threshold value.
        threshold: f64,
    },
    /// `parent ≤ threshold` (for float parents).
    FloatLte {
        /// The parent parameter name.
        parent: String,
        /// The threshold value.
        threshold: f64,
    },
    /// Boolean conjunction of two conditions (both must hold).
    And(Box<Condition>, Box<Condition>),
    /// Boolean disjunction of two conditions (either may hold).
    Or(Box<Condition>, Box<Condition>),
}

impl Condition {
    /// Compose this condition with another via boolean `AND`.
    pub fn and(self, other: Condition) -> Condition {
        Condition::And(Box::new(self), Box::new(other))
    }

    /// Compose this condition with another via boolean `OR`.
    pub fn or(self, other: Condition) -> Condition {
        Condition::Or(Box::new(self), Box::new(other))
    }

    /// Evaluate this condition against a partially-sampled [`ParamMap`].
    ///
    /// Returns `false` if any referenced parent parameter is absent — the
    /// caller is expected to evaluate conditions in topological (parent-first)
    /// order, so absence here means "parent is itself a conditional whose
    /// gate didn't fire", and the conjunction-style semantic is correct.
    fn evaluate(&self, params: &ParamMap) -> bool {
        match self {
            Condition::Equals { parent, value } => params
                .category_optional(parent)
                .map(|c| c == value)
                .unwrap_or(false),
            Condition::InValues { parent, values } => params
                .category_optional(parent)
                .map(|c| values.iter().any(|v| c == v))
                .unwrap_or(false),
            Condition::FloatGt { parent, threshold } => params
                .float_optional(parent)
                .map(|v| v > *threshold)
                .unwrap_or(false),
            Condition::FloatLte { parent, threshold } => params
                .float_optional(parent)
                .map(|v| v <= *threshold)
                .unwrap_or(false),
            Condition::And(a, b) => a.evaluate(params) && b.evaluate(params),
            Condition::Or(a, b) => a.evaluate(params) || b.evaluate(params),
        }
    }

    /// Names of parent parameters this condition references.
    fn referenced_parents(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_parents(&mut out);
        out
    }

    fn collect_parents<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Condition::Equals { parent, .. }
            | Condition::InValues { parent, .. }
            | Condition::FloatGt { parent, .. }
            | Condition::FloatLte { parent, .. } => out.push(parent),
            Condition::And(a, b) | Condition::Or(a, b) => {
                a.collect_parents(out);
                b.collect_parents(out);
            }
        }
    }
}

/// Builder DSL for [`Condition`]: `when("parent").equals("value")`.
pub struct ConditionBuilder {
    parent: String,
}

impl ConditionBuilder {
    /// `parent == value`.
    pub fn equals(self, value: impl Into<Category>) -> Condition {
        Condition::Equals {
            parent: self.parent,
            value: value.into(),
        }
    }

    /// `parent ∈ values`.
    pub fn in_values<T: Clone + Into<Category>>(self, values: &[T]) -> Condition {
        Condition::InValues {
            parent: self.parent,
            values: values.iter().cloned().map(Into::into).collect(),
        }
    }

    /// `parent > threshold` (for float parents).
    pub fn greater_than(self, threshold: f64) -> Condition {
        Condition::FloatGt {
            parent: self.parent,
            threshold,
        }
    }

    /// `parent ≤ threshold` (for float parents).
    pub fn at_most(self, threshold: f64) -> Condition {
        Condition::FloatLte {
            parent: self.parent,
            threshold,
        }
    }
}

/// Start a conditional clause: `when("parent").equals("value")`.
pub fn when(parent: impl Into<String>) -> ConditionBuilder {
    ConditionBuilder {
        parent: parent.into(),
    }
}

// ---------------------------------------------------------------------------
// Constraint: cross-parameter feasibility predicate
// ---------------------------------------------------------------------------

/// Cross-parameter feasibility predicate evaluated at sample time.
///
/// A [`Constraint`] takes a fully-sampled [`ParamMap`] and returns `true` for
/// feasible combinations. Sampling proceeds by rejection: draw a candidate,
/// evaluate every constraint, discard and resample on violation. After
/// [`MAX_REJECTION_ATTEMPTS`] consecutive rejections the sampler returns
/// [`SamplerError::ConstraintUnsatisfiable`].
///
/// Constraints are stored as boxed closures rather than enum variants so
/// users can express any cross-parameter invariant — divisibility,
/// monotonicity, ordering — without extending the [`SearchSpace`] surface.
pub struct Constraint {
    name: String,
    predicate: Arc<dyn Fn(&ParamMap) -> bool + Send + Sync + 'static>,
}

impl Constraint {
    /// Create a constraint with a human-readable name (used in error messages).
    pub fn new<F>(name: impl Into<String>, predicate: F) -> Self
    where
        F: Fn(&ParamMap) -> bool + Send + Sync + 'static,
    {
        Constraint {
            name: name.into(),
            predicate: Arc::new(predicate),
        }
    }

    /// The constraint's display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Evaluate the constraint against a candidate `ParamMap`.
    pub fn check(&self, params: &ParamMap) -> bool {
        (self.predicate)(params)
    }
}

impl Clone for Constraint {
    fn clone(&self) -> Self {
        Constraint {
            name: self.name.clone(),
            predicate: Arc::clone(&self.predicate),
        }
    }
}

impl fmt::Debug for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Constraint")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// SpaceError: build-time errors
// ---------------------------------------------------------------------------

/// Errors raised at [`SearchSpaceBuilder::build`] time.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SpaceError {
    /// Two parameters share the same name.
    DuplicateName(String),
    /// A categorical was declared with no choices.
    EmptyChoices(String),
    /// A float parameter has `low > high` or non-finite bounds, or a
    /// log-scaled parameter has a non-positive bound.
    InvalidRange {
        /// The offending parameter name.
        name: String,
        /// The supplied lower bound.
        low: f64,
        /// The supplied upper bound.
        high: f64,
    },
    /// An integer parameter has `low > high`.
    InvalidIntRange {
        /// The offending parameter name.
        name: String,
        /// The supplied lower bound.
        low: i64,
        /// The supplied upper bound.
        high: i64,
    },
    /// A conditional parameter references a parent that does not exist.
    ConditionalParentNotFound {
        /// The conditional child parameter.
        child: String,
        /// The missing parent.
        parent: String,
    },
    /// Conditional parameters form a cycle (parent depends on child, child
    /// depends on parent).
    CyclicCondition {
        /// One of the parameter names involved in the cycle.
        name: String,
    },
}

impl fmt::Display for SpaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpaceError::DuplicateName(name) => write!(f, "duplicate parameter name '{name}'"),
            SpaceError::EmptyChoices(name) => {
                write!(f, "categorical parameter '{name}' has no choices")
            }
            SpaceError::InvalidRange { name, low, high } => write!(
                f,
                "parameter '{name}' has invalid float range [{low}, {high}]"
            ),
            SpaceError::InvalidIntRange { name, low, high } => write!(
                f,
                "parameter '{name}' has invalid int range [{low}, {high}]"
            ),
            SpaceError::ConditionalParentNotFound { child, parent } => write!(
                f,
                "conditional parameter '{child}' references unknown parent '{parent}'"
            ),
            SpaceError::CyclicCondition { name } => {
                write!(f, "conditional parameter '{name}' is part of a cycle")
            }
        }
    }
}

impl std::error::Error for SpaceError {}

// ---------------------------------------------------------------------------
// SamplerError: sample-time errors
// ---------------------------------------------------------------------------

/// Errors raised at sample time.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SamplerError {
    /// Rejection sampling exceeded [`MAX_REJECTION_ATTEMPTS`] without finding
    /// a feasible point.
    ConstraintUnsatisfiable {
        /// Last-violated constraint name (best-effort).
        last_violated: Option<String>,
        /// Number of attempts before giving up.
        attempts: usize,
    },
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplerError::ConstraintUnsatisfiable {
                last_violated,
                attempts,
            } => {
                if let Some(name) = last_violated {
                    write!(
                        f,
                        "constraint '{name}' unsatisfiable after {attempts} attempts"
                    )
                } else {
                    write!(f, "constraints unsatisfiable after {attempts} attempts")
                }
            }
        }
    }
}

impl std::error::Error for SamplerError {}

// ---------------------------------------------------------------------------
// Search-space internal entries
// ---------------------------------------------------------------------------

/// One declared parameter, with optional conditional gate.
#[derive(Debug, Clone)]
struct ParamEntry {
    name: String,
    def: ParamDef,
    /// Conditional activation, if any.
    condition: Option<Condition>,
}

// ---------------------------------------------------------------------------
// SearchSpaceBuilder
// ---------------------------------------------------------------------------

/// Builder for [`SearchSpace`].
#[derive(Default)]
pub struct SearchSpaceBuilder {
    entries: Vec<ParamEntry>,
    constraints: Vec<Constraint>,
}

impl SearchSpaceBuilder {
    /// Declare an unconditional parameter.
    pub fn param(mut self, name: impl Into<String>, def: ParamDef) -> Self {
        self.entries.push(ParamEntry {
            name: name.into(),
            def,
            condition: None,
        });
        self
    }

    /// Declare a conditional parameter, active only when `condition` holds.
    pub fn conditional(
        mut self,
        name: impl Into<String>,
        def: ParamDef,
        condition: Condition,
    ) -> Self {
        self.entries.push(ParamEntry {
            name: name.into(),
            def,
            condition: Some(condition),
        });
        self
    }

    /// Add a cross-parameter feasibility predicate.
    pub fn constraint<F>(mut self, name: impl Into<String>, predicate: F) -> Self
    where
        F: Fn(&ParamMap) -> bool + Send + Sync + 'static,
    {
        self.constraints.push(Constraint::new(name, predicate));
        self
    }

    /// Validate and finalize the [`SearchSpace`].
    pub fn build(self) -> Result<SearchSpace, SpaceError> {
        // 1. Per-parameter validity.
        let mut seen: BTreeMap<&str, ()> = BTreeMap::new();
        for entry in &self.entries {
            if seen.insert(entry.name.as_str(), ()).is_some() {
                return Err(SpaceError::DuplicateName(entry.name.clone()));
            }
            entry.def.validate(&entry.name)?;
        }

        // 2. Conditional parents must exist.
        for entry in &self.entries {
            if let Some(cond) = &entry.condition {
                for parent in cond.referenced_parents() {
                    if !seen.contains_key(parent) {
                        return Err(SpaceError::ConditionalParentNotFound {
                            child: entry.name.clone(),
                            parent: parent.to_string(),
                        });
                    }
                }
            }
        }

        // 3. Topological sort (detects cycles). Parent edges first.
        let order = topological_order(&self.entries)?;

        Ok(SearchSpace {
            entries: Arc::new(self.entries),
            order: Arc::new(order),
            constraints: Arc::new(self.constraints),
        })
    }
}

/// Compute a topological ordering: parents before children.
fn topological_order(entries: &[ParamEntry]) -> Result<Vec<usize>, SpaceError> {
    use std::collections::HashMap;

    let name_to_idx: HashMap<&str, usize> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.name.as_str(), i))
        .collect();

    let n = entries.len();
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, entry) in entries.iter().enumerate() {
        if let Some(cond) = &entry.condition {
            for parent in cond.referenced_parents() {
                if let Some(&p_idx) = name_to_idx.get(parent) {
                    adj[p_idx].push(i);
                    in_degree[i] += 1;
                }
            }
        }
    }

    // Kahn's algorithm.
    let mut order = Vec::with_capacity(n);
    let mut frontier: Vec<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, d)| **d == 0)
        .map(|(i, _)| i)
        .collect();

    while let Some(idx) = frontier.pop() {
        order.push(idx);
        for &child in &adj[idx] {
            in_degree[child] -= 1;
            if in_degree[child] == 0 {
                frontier.push(child);
            }
        }
    }

    if order.len() != n {
        // The first vertex with non-zero in-degree is part of a cycle.
        let stuck = in_degree
            .iter()
            .position(|d| *d > 0)
            .map(|i| entries[i].name.clone())
            .unwrap_or_else(|| "<unknown>".to_string());
        return Err(SpaceError::CyclicCondition { name: stuck });
    }
    Ok(order)
}

// ---------------------------------------------------------------------------
// SearchSpace: immutable, clone-cheap, sample-time interface
// ---------------------------------------------------------------------------

/// Immutable, clone-cheap search-space description.
///
/// Cloning is `O(1)` (Arc-shared internals), so factories can return a
/// `SearchSpace` from `config_space()` without per-call allocation.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    entries: Arc<Vec<ParamEntry>>,
    order: Arc<Vec<usize>>,
    constraints: Arc<Vec<Constraint>>,
}

impl SearchSpace {
    /// Start a new builder.
    pub fn builder() -> SearchSpaceBuilder {
        SearchSpaceBuilder::default()
    }

    /// Number of declared parameters (including conditional ones).
    pub fn n_params(&self) -> usize {
        self.entries.len()
    }

    /// Iterator over `(name, &ParamDef, Option<&Condition>)`.
    pub fn params(&self) -> impl Iterator<Item = (&str, &ParamDef, Option<&Condition>)> {
        self.entries
            .iter()
            .map(|e| (e.name.as_str(), &e.def, e.condition.as_ref()))
    }

    /// Look up a parameter definition by name.
    pub fn get(&self, name: &str) -> Option<&ParamDef> {
        self.entries.iter().find(|e| e.name == name).map(|e| &e.def)
    }

    /// All declared constraints (in declaration order).
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Override a float parameter's range in-place.
    ///
    /// Returns an error if `name` does not exist, names a non-float parameter,
    /// or if `low > high`.
    pub fn set_float_range(&mut self, name: &str, low: f64, high: f64) -> Result<(), SpaceError> {
        let entries =
            Arc::get_mut(&mut self.entries).expect("set_float_range requires unique ownership");
        let entry = entries.iter_mut().find(|e| e.name == name).ok_or_else(|| {
            SpaceError::ConditionalParentNotFound {
                child: "<set_float_range>".to_string(),
                parent: name.to_string(),
            }
        })?;
        match &mut entry.def {
            ParamDef::Float {
                low: lo,
                high: hi,
                scale,
            } => {
                let new_def = ParamDef::Float {
                    low,
                    high,
                    scale: *scale,
                };
                new_def.validate(name)?;
                *lo = low;
                *hi = high;
                Ok(())
            }
            _ => Err(SpaceError::InvalidRange {
                name: name.to_string(),
                low,
                high,
            }),
        }
    }

    /// Override an integer parameter's range in-place.
    pub fn set_int_range(&mut self, name: &str, low: i64, high: i64) -> Result<(), SpaceError> {
        let entries =
            Arc::get_mut(&mut self.entries).expect("set_int_range requires unique ownership");
        let entry = entries.iter_mut().find(|e| e.name == name).ok_or_else(|| {
            SpaceError::ConditionalParentNotFound {
                child: "<set_int_range>".to_string(),
                parent: name.to_string(),
            }
        })?;
        match &mut entry.def {
            ParamDef::Int { low: lo, high: hi } => {
                let new_def = ParamDef::Int { low, high };
                new_def.validate(name)?;
                *lo = low;
                *hi = high;
                Ok(())
            }
            _ => Err(SpaceError::InvalidIntRange {
                name: name.to_string(),
                low,
                high,
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Sampling primitives
    // -----------------------------------------------------------------------

    /// Sample one parameter map drawn from the unconditional, unconstrained
    /// distribution, then activate conditionals based on parent values.
    ///
    /// Constraints are NOT enforced here — this is the inner draw used by
    /// [`Self::sample`].
    fn draw_raw(&self, rng: &mut u64) -> ParamMap {
        let mut map = ParamMap::new();

        // Iterate in topological order so parents resolve before children.
        for &idx in self.order.iter() {
            let entry = &self.entries[idx];

            // Conditional gate: if not active, skip (the param is absent).
            if let Some(cond) = &entry.condition {
                if !cond.evaluate(&map) {
                    continue;
                }
            }

            let value = sample_param(&entry.def, rng);
            map.insert(entry.name.clone(), value);
        }

        map
    }

    /// Sample one feasible parameter map (rejection sampling against the
    /// declared constraints).
    pub fn sample(&self, rng: &mut u64) -> Result<ParamMap, SamplerError> {
        let mut last_violated: Option<String> = None;
        for _ in 0..MAX_REJECTION_ATTEMPTS {
            let candidate = self.draw_raw(rng);
            let mut feasible = true;
            for constraint in self.constraints.iter() {
                if !constraint.check(&candidate) {
                    last_violated = Some(constraint.name.clone());
                    feasible = false;
                    break;
                }
            }
            if feasible {
                return Ok(candidate);
            }
        }
        Err(SamplerError::ConstraintUnsatisfiable {
            last_violated,
            attempts: MAX_REJECTION_ATTEMPTS,
        })
    }

    /// Generate `n` configurations using stratified Latin-hypercube draws on
    /// the *unconstrained* distribution, then filter by feasibility. The
    /// returned vector contains only the feasible LHS draws — fewer than `n`
    /// if some were rejected. This trades the exact-coverage property of pure
    /// LHS for compatibility with constraints; for typical low-dimensional
    /// search spaces with mild constraint geometries the loss is negligible.
    pub fn latin_hypercube(&self, n: usize, rng: &mut u64) -> Vec<ParamMap> {
        if n == 0 {
            return Vec::new();
        }

        let total_dims = self.entries.len();
        // Pre-compute strata for each dimension. Each stratum is a uniform draw
        // from `[i/n, (i+1)/n)`, and strata are shuffled per dimension.
        let mut stratified: Vec<Vec<f64>> = Vec::with_capacity(total_dims);
        for _ in 0..total_dims {
            let mut column: Vec<f64> = (0..n)
                .map(|i| {
                    let lo = i as f64 / n as f64;
                    let hi = (i + 1) as f64 / n as f64;
                    let u = xorshift64_f64(rng);
                    lo + u * (hi - lo)
                })
                .collect();
            // Fisher-Yates shuffle (in-place).
            for i in (1..n).rev() {
                let j = (xorshift64(rng) as usize) % (i + 1);
                column.swap(i, j);
            }
            stratified.push(column);
        }

        let mut out: Vec<ParamMap> = Vec::with_capacity(n);
        // `sample_i` indexes multiple parallel columns of `stratified` -- the
        // loop body reads `stratified[idx][sample_i]` for every active param.
        // An iterator over a single column would lose the row-cursor semantics.
        #[allow(clippy::needless_range_loop)]
        for sample_i in 0..n {
            let mut map = ParamMap::new();
            for &idx in self.order.iter() {
                let entry = &self.entries[idx];
                if let Some(cond) = &entry.condition {
                    if !cond.evaluate(&map) {
                        continue;
                    }
                }
                let u = stratified[idx][sample_i];
                let value = map_unit_to_param(u, &entry.def, rng);
                map.insert(entry.name.clone(), value);
            }
            // Filter by feasibility.
            if self.constraints.iter().all(|c| c.check(&map)) {
                out.push(map);
            }
        }
        out
    }

    /// Perturb an existing [`ParamMap`] by adding Gaussian noise scaled by
    /// `sigma` (in the parameter's natural scale). Returns a feasible
    /// neighbour, retrying up to [`MAX_REJECTION_ATTEMPTS`] on infeasibility.
    pub fn perturb(
        &self,
        params: &ParamMap,
        sigma: f64,
        rng: &mut u64,
    ) -> Result<ParamMap, SamplerError> {
        let mut last_violated: Option<String> = None;
        for _ in 0..MAX_REJECTION_ATTEMPTS {
            let candidate = self.perturb_raw(params, sigma, rng);
            let mut feasible = true;
            for constraint in self.constraints.iter() {
                if !constraint.check(&candidate) {
                    last_violated = Some(constraint.name.clone());
                    feasible = false;
                    break;
                }
            }
            if feasible {
                return Ok(candidate);
            }
        }
        Err(SamplerError::ConstraintUnsatisfiable {
            last_violated,
            attempts: MAX_REJECTION_ATTEMPTS,
        })
    }

    /// Per-parameter perturbation without feasibility check.
    fn perturb_raw(&self, params: &ParamMap, sigma: f64, rng: &mut u64) -> ParamMap {
        let mut map = ParamMap::new();
        for &idx in self.order.iter() {
            let entry = &self.entries[idx];
            // Conditionals respect the gate even under perturbation.
            if let Some(cond) = &entry.condition {
                if !cond.evaluate(&map) {
                    continue;
                }
            }
            let value = match params.get(&entry.name) {
                Some(current) => perturb_value(current, &entry.def, sigma, rng),
                None => sample_param(&entry.def, rng),
            };
            map.insert(entry.name.clone(), value);
        }
        map
    }
}

// ---------------------------------------------------------------------------
// ParamValue + ParamMap
// ---------------------------------------------------------------------------

/// A single sampled parameter value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ParamValue {
    /// Continuous float value.
    Float(f64),
    /// Integer value.
    Int(i64),
    /// Categorical label.
    Category(Category),
}

impl ParamValue {
    /// Return the float value or `None` if this is not a float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParamValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Return the int value or `None` if this is not an int.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParamValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Return the categorical value or `None` if this is not a category.
    pub fn as_category(&self) -> Option<&Category> {
        match self {
            ParamValue::Category(v) => Some(v),
            _ => None,
        }
    }
}

/// A sampled point in the search space — name to value map.
///
/// Conditional parameters whose activation gate did not fire are *absent*
/// from the map (use [`Self::float_optional`] / [`Self::int_optional`] /
/// [`Self::category_optional`] to read them safely).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParamMap {
    values: BTreeMap<String, ParamValue>,
}

impl ParamMap {
    /// Construct an empty `ParamMap`.
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    /// Number of stored values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether this map has any stored values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Insert or overwrite a value for `name`.
    pub fn insert(&mut self, name: String, value: ParamValue) {
        self.values.insert(name, value);
    }

    /// Borrow a stored value by name, regardless of variant.
    pub fn get(&self, name: &str) -> Option<&ParamValue> {
        self.values.get(name)
    }

    /// Iterate over `(name, value)` pairs in lexicographic name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ParamValue)> {
        self.values.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Whether the parameter is present in this map.
    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    // -----------------------------------------------------------------------
    // Required accessors (return `FactoryError` on missing / wrong-type)
    // -----------------------------------------------------------------------

    /// Read a required float parameter.
    pub fn float(&self, name: &str) -> Result<f64, FactoryError> {
        match self.values.get(name) {
            Some(ParamValue::Float(v)) => Ok(*v),
            Some(other) => Err(FactoryError::IncompatibleArm {
                reason: format!(
                    "param '{name}' expected Float, found {}",
                    variant_name(other)
                ),
            }),
            None => Err(FactoryError::IncompatibleArm {
                reason: format!("required param '{name}' is missing from ParamMap"),
            }),
        }
    }

    /// Read a required int parameter.
    pub fn int(&self, name: &str) -> Result<i64, FactoryError> {
        match self.values.get(name) {
            Some(ParamValue::Int(v)) => Ok(*v),
            Some(other) => Err(FactoryError::IncompatibleArm {
                reason: format!("param '{name}' expected Int, found {}", variant_name(other)),
            }),
            None => Err(FactoryError::IncompatibleArm {
                reason: format!("required param '{name}' is missing from ParamMap"),
            }),
        }
    }

    /// Read a required usize parameter (validates non-negative int).
    pub fn usize(&self, name: &str) -> Result<usize, FactoryError> {
        let v = self.int(name)?;
        if v < 0 {
            return Err(FactoryError::IncompatibleArm {
                reason: format!("param '{name}' must be non-negative, got {v}"),
            });
        }
        Ok(v as usize)
    }

    /// Read a required categorical parameter.
    pub fn category(&self, name: &str) -> Result<&Category, FactoryError> {
        match self.values.get(name) {
            Some(ParamValue::Category(v)) => Ok(v),
            Some(other) => Err(FactoryError::IncompatibleArm {
                reason: format!(
                    "param '{name}' expected Category, found {}",
                    variant_name(other)
                ),
            }),
            None => Err(FactoryError::IncompatibleArm {
                reason: format!("required param '{name}' is missing from ParamMap"),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Optional accessors (return `None` on missing; type mismatch panics —
    // calls inside constraint closures should use these only for parameters
    // whose presence is conditional, not as a wrong-type guard)
    // -----------------------------------------------------------------------

    /// Read an optional float (returns `None` if the parameter is absent or
    /// of a different type).
    pub fn float_optional(&self, name: &str) -> Option<f64> {
        self.values.get(name).and_then(ParamValue::as_float)
    }

    /// Read an optional int (returns `None` if the parameter is absent or
    /// of a different type).
    pub fn int_optional(&self, name: &str) -> Option<i64> {
        self.values.get(name).and_then(ParamValue::as_int)
    }

    /// Read an optional category (returns `None` if the parameter is absent
    /// or of a different type).
    pub fn category_optional(&self, name: &str) -> Option<&Category> {
        self.values.get(name).and_then(ParamValue::as_category)
    }

    // -----------------------------------------------------------------------
    // Constraint-closure helpers (panic on missing — caller's responsibility
    // inside a closure that runs after sampling, where every referenced
    // unconditional param is guaranteed present)
    // -----------------------------------------------------------------------

    /// Read a float without error wrapping. Panics if the parameter is
    /// missing or of the wrong type. Intended for use inside [`Constraint`]
    /// closures, where the parameter's presence is guaranteed by sampling
    /// order.
    pub fn float_unchecked(&self, name: &str) -> f64 {
        self.float_optional(name)
            .unwrap_or_else(|| panic!("ParamMap::float_unchecked('{name}'): missing or wrong type"))
    }

    /// Read an int without error wrapping. See [`Self::float_unchecked`].
    pub fn int_unchecked(&self, name: &str) -> i64 {
        self.int_optional(name)
            .unwrap_or_else(|| panic!("ParamMap::int_unchecked('{name}'): missing or wrong type"))
    }

    /// Read a category without error wrapping. See [`Self::float_unchecked`].
    pub fn category_unchecked(&self, name: &str) -> &Category {
        self.category_optional(name).unwrap_or_else(|| {
            panic!("ParamMap::category_unchecked('{name}'): missing or wrong type")
        })
    }
}

fn variant_name(v: &ParamValue) -> &'static str {
    match v {
        ParamValue::Float(_) => "Float",
        ParamValue::Int(_) => "Int",
        ParamValue::Category(_) => "Category",
    }
}

// ---------------------------------------------------------------------------
// Sampling helpers (free functions)
// ---------------------------------------------------------------------------

/// Map a uniform `[0, 1)` value to a concrete [`ParamValue`].
fn map_unit_to_param(u: f64, def: &ParamDef, rng: &mut u64) -> ParamValue {
    match def {
        ParamDef::Float { low, high, scale } => {
            let v = match scale {
                Scale::Linear => low + u * (high - low),
                Scale::Log => {
                    let ln_low = low.ln();
                    let ln_high = high.ln();
                    (ln_low + u * (ln_high - ln_low)).exp()
                }
            };
            ParamValue::Float(v)
        }
        ParamDef::Int { low, high } => {
            let range = (*high - *low + 1) as f64;
            let v = (*low as f64 + (u * range).floor()).clamp(*low as f64, *high as f64);
            ParamValue::Int(v as i64)
        }
        ParamDef::Categorical { choices } => {
            let n = choices.len();
            let idx = ((u * n as f64).floor() as usize).min(n - 1);
            // The unused `rng` param keeps signatures aligned with `sample_param`.
            let _ = rng;
            ParamValue::Category(choices[idx].clone())
        }
    }
}

/// Draw a single value for a parameter from `rng`.
fn sample_param(def: &ParamDef, rng: &mut u64) -> ParamValue {
    let u = xorshift64_f64(rng);
    map_unit_to_param(u, def, rng)
}

/// Perturb `current` by Gaussian noise of strength `sigma`. Categoricals
/// are perturbed by Bernoulli-flip with probability `min(sigma, 1.0)`.
fn perturb_value(current: &ParamValue, def: &ParamDef, sigma: f64, rng: &mut u64) -> ParamValue {
    match (def, current) {
        (
            ParamDef::Float {
                low,
                high,
                scale: Scale::Linear,
            },
            ParamValue::Float(v),
        ) => {
            let noise = standard_normal(rng) * sigma * (high - low);
            ParamValue::Float((v + noise).clamp(*low, *high))
        }
        (
            ParamDef::Float {
                low,
                high,
                scale: Scale::Log,
            },
            ParamValue::Float(v),
        ) => {
            let ln_low = low.ln();
            let ln_high = high.ln();
            let ln_v = v.max(*low).ln();
            let noise = standard_normal(rng) * sigma * (ln_high - ln_low);
            let v_new = (ln_v + noise).exp().clamp(*low, *high);
            ParamValue::Float(v_new)
        }
        (ParamDef::Int { low, high }, ParamValue::Int(v)) => {
            let range = (*high - *low) as f64;
            let noise = standard_normal(rng) * sigma * range;
            let perturbed = (*v as f64 + noise).round();
            ParamValue::Int(perturbed.clamp(*low as f64, *high as f64) as i64)
        }
        (ParamDef::Categorical { choices }, ParamValue::Category(current_cat)) => {
            let p = xorshift64_f64(rng);
            if choices.len() > 1 && p < sigma.min(1.0) {
                // Pick a different choice.
                let n = choices.len();
                let current_idx = choices.iter().position(|c| c == current_cat).unwrap_or(0);
                let alt = (xorshift64(rng) as usize) % (n - 1);
                let new_idx = if alt >= current_idx { alt + 1 } else { alt };
                ParamValue::Category(choices[new_idx].clone())
            } else {
                ParamValue::Category(current_cat.clone())
            }
        }
        // Type mismatch (paramdef vs current value): redraw from definition.
        _ => sample_param(def, rng),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_space_builder_basic_roundtrip() {
        let space = SearchSpace::builder()
            .param("learning_rate", log_range(1e-4, 1e-1))
            .param("n_trees", int_range(5, 50))
            .param("activation", categorical(&["relu", "tanh", "gelu"]))
            .build()
            .expect("valid space");

        assert_eq!(space.n_params(), 3, "expected 3 declared params");
        let names: Vec<&str> = space.params().map(|(n, _, _)| n).collect();
        assert!(names.contains(&"learning_rate"));
        assert!(names.contains(&"n_trees"));
        assert!(names.contains(&"activation"));
    }

    #[test]
    fn search_space_rejects_duplicate_name() {
        let result = SearchSpace::builder()
            .param("lr", log_range(1e-4, 1e-1))
            .param("lr", log_range(1e-3, 1e-2))
            .build();
        assert!(
            matches!(result, Err(SpaceError::DuplicateName(_))),
            "expected DuplicateName error, got {:?}",
            result
        );
    }

    #[test]
    fn search_space_rejects_inverted_float_range() {
        let result = SearchSpace::builder()
            .param("lr", linear_range(1.0, 0.5))
            .build();
        assert!(
            matches!(result, Err(SpaceError::InvalidRange { .. })),
            "expected InvalidRange error, got {:?}",
            result
        );
    }

    #[test]
    fn search_space_rejects_log_with_nonpositive_low() {
        let result = SearchSpace::builder()
            .param("lr", log_range(0.0, 1e-1))
            .build();
        assert!(
            matches!(result, Err(SpaceError::InvalidRange { .. })),
            "log scale must reject low <= 0: got {:?}",
            result
        );
    }

    #[test]
    fn search_space_rejects_inverted_int_range() {
        let result = SearchSpace::builder().param("k", int_range(10, 5)).build();
        assert!(
            matches!(result, Err(SpaceError::InvalidIntRange { .. })),
            "expected InvalidIntRange error, got {:?}",
            result
        );
    }

    #[test]
    fn search_space_rejects_empty_categorical() {
        let result = SearchSpace::builder()
            .param("act", categorical::<&str>(&[]))
            .build();
        assert!(
            matches!(result, Err(SpaceError::EmptyChoices(_))),
            "expected EmptyChoices error, got {:?}",
            result
        );
    }

    #[test]
    fn conditional_parent_must_exist() {
        let result = SearchSpace::builder()
            .param("model", categorical(&["a", "b"]))
            .conditional(
                "child",
                log_range(1e-4, 1e-1),
                when("nonexistent").equals("a"),
            )
            .build();
        assert!(
            matches!(result, Err(SpaceError::ConditionalParentNotFound { .. })),
            "expected ConditionalParentNotFound, got {:?}",
            result
        );
    }

    #[test]
    fn cyclic_conditionals_detected() {
        // Build A conditioned on B and B conditioned on A.
        let result = SearchSpace::builder()
            .conditional("a", log_range(1e-4, 1e-1), when("b").equals("y"))
            .conditional("b", categorical(&["x", "y"]), when("a").greater_than(0.5))
            .build();
        assert!(
            matches!(result, Err(SpaceError::CyclicCondition { .. })),
            "expected CyclicCondition, got {:?}",
            result
        );
    }

    #[test]
    fn sampling_produces_in_bounds_values() {
        let space = SearchSpace::builder()
            .param("lr", linear_range(0.0, 1.0))
            .param("k", int_range(2, 10))
            .param("act", categorical(&["relu", "tanh"]))
            .build()
            .unwrap();

        let mut rng = 42u64;
        for _ in 0..200 {
            let m = space.sample(&mut rng).expect("sample succeeds");
            let lr = m.float("lr").unwrap();
            assert!((0.0..=1.0).contains(&lr), "lr={lr} out of range");
            let k = m.int("k").unwrap();
            assert!((2..=10).contains(&k), "k={k} out of range");
            let act = m.category("act").unwrap();
            assert!(act == "relu" || act == "tanh", "unexpected act={act}");
        }
    }

    #[test]
    fn log_scale_sampling_in_bounds() {
        let space = SearchSpace::builder()
            .param("lr", log_range(1e-5, 1.0))
            .build()
            .unwrap();

        let mut rng = 77u64;
        for _ in 0..200 {
            let m = space.sample(&mut rng).unwrap();
            let lr = m.float("lr").unwrap();
            assert!((1e-5..=1.0).contains(&lr), "log-scale lr={lr} out of range");
        }
    }

    #[test]
    fn conditional_param_present_only_when_active() {
        let space = SearchSpace::builder()
            .param("model", categorical(&["svm", "rf"]))
            .conditional("svm_c", log_range(1e-3, 1e3), when("model").equals("svm"))
            .conditional("rf_depth", int_range(2, 32), when("model").equals("rf"))
            .build()
            .unwrap();

        let mut rng = 1234u64;
        let mut saw_svm = false;
        let mut saw_rf = false;
        for _ in 0..200 {
            let m = space.sample(&mut rng).unwrap();
            let model = m.category("model").unwrap();
            if model == "svm" {
                saw_svm = true;
                assert!(m.contains("svm_c"), "svm config must include svm_c");
                assert!(
                    !m.contains("rf_depth"),
                    "svm config must NOT include rf_depth"
                );
            } else if model == "rf" {
                saw_rf = true;
                assert!(m.contains("rf_depth"), "rf config must include rf_depth");
                assert!(!m.contains("svm_c"), "rf config must NOT include svm_c");
            }
        }
        assert!(
            saw_svm && saw_rf,
            "should observe both model types in 200 draws"
        );
    }

    #[test]
    fn constraint_rejection_yields_only_feasible() {
        // d_model in {2,4,6,8}, n_heads in {1,2,4,8}: enforce divisibility.
        let space = SearchSpace::builder()
            .param("d_model", int_range(2, 8))
            .param("n_heads", int_range(1, 4))
            .constraint("heads_divide_d_model", |c| {
                let d = c.int_unchecked("d_model");
                let h = c.int_unchecked("n_heads");
                h > 0 && d % h == 0
            })
            .build()
            .unwrap();

        let mut rng = 5555u64;
        for _ in 0..100 {
            let m = space
                .sample(&mut rng)
                .expect("constraint should be satisfiable");
            let d = m.int("d_model").unwrap();
            let h = m.int("n_heads").unwrap();
            assert!(d % h == 0, "constraint violated: d_model={d}, n_heads={h}");
        }
    }

    #[test]
    fn unsatisfiable_constraint_returns_error() {
        // Constraint: 0 == 1 — never feasible.
        let space = SearchSpace::builder()
            .param("x", int_range(1, 10))
            .constraint("never", |_| false)
            .build()
            .unwrap();

        let mut rng = 777u64;
        let result = space.sample(&mut rng);
        assert!(
            matches!(result, Err(SamplerError::ConstraintUnsatisfiable { .. })),
            "expected ConstraintUnsatisfiable, got {:?}",
            result
        );
    }

    #[test]
    fn perturb_stays_in_bounds_and_feasible() {
        let space = SearchSpace::builder()
            .param("lr", linear_range(0.001, 1.0))
            .param("depth", int_range(1, 20))
            .param("act", categorical(&["a", "b", "c"]))
            .build()
            .unwrap();

        let mut rng = 11u64;
        let base = space.sample(&mut rng).unwrap();
        for _ in 0..100 {
            let p = space.perturb(&base, 0.3, &mut rng).unwrap();
            let lr = p.float("lr").unwrap();
            assert!(
                (0.001..=1.0).contains(&lr),
                "lr={lr} out of range after perturb"
            );
            let depth = p.int("depth").unwrap();
            assert!(
                (1..=20).contains(&depth),
                "depth={depth} out of range after perturb"
            );
        }
    }

    #[test]
    fn param_map_required_accessors_reject_missing() {
        let m = ParamMap::new();
        assert!(matches!(
            m.float("x"),
            Err(FactoryError::IncompatibleArm { .. })
        ));
        assert!(matches!(
            m.int("x"),
            Err(FactoryError::IncompatibleArm { .. })
        ));
        assert!(matches!(
            m.category("x"),
            Err(FactoryError::IncompatibleArm { .. })
        ));
    }

    #[test]
    fn param_map_rejects_wrong_type() {
        let mut m = ParamMap::new();
        m.insert("k".into(), ParamValue::Int(5));
        assert!(matches!(
            m.float("k"),
            Err(FactoryError::IncompatibleArm { .. })
        ));
        assert!(matches!(
            m.category("k"),
            Err(FactoryError::IncompatibleArm { .. })
        ));
        assert_eq!(m.int("k").unwrap(), 5);
    }

    #[test]
    fn search_space_set_float_range() {
        let mut space = SearchSpace::builder()
            .param("lr", log_range(1e-4, 1e-1))
            .build()
            .unwrap();
        space.set_float_range("lr", 1e-3, 1e-2).unwrap();
        // Sample should now lie in the new range.
        let mut rng = 99u64;
        for _ in 0..100 {
            let m = space.sample(&mut rng).unwrap();
            let lr = m.float("lr").unwrap();
            assert!((1e-3..=1e-2).contains(&lr), "lr={lr} out of new range");
        }
    }

    #[test]
    fn search_space_set_float_range_rejects_inverted() {
        let mut space = SearchSpace::builder()
            .param("lr", log_range(1e-4, 1e-1))
            .build()
            .unwrap();
        let result = space.set_float_range("lr", 1.0, 0.5);
        assert!(matches!(result, Err(SpaceError::InvalidRange { .. })));
    }

    #[test]
    fn deterministic_sampling_with_seed() {
        let make = || {
            SearchSpace::builder()
                .param("lr", log_range(1e-4, 1e-1))
                .param("k", int_range(1, 100))
                .build()
                .unwrap()
        };
        let s1 = make();
        let s2 = make();
        let mut rng1 = 31337u64;
        let mut rng2 = 31337u64;
        for _ in 0..50 {
            let m1 = s1.sample(&mut rng1).unwrap();
            let m2 = s2.sample(&mut rng2).unwrap();
            assert_eq!(m1.float("lr").unwrap(), m2.float("lr").unwrap());
            assert_eq!(m1.int("k").unwrap(), m2.int("k").unwrap());
        }
    }

    #[test]
    fn latin_hypercube_returns_feasible_subset() {
        let space = SearchSpace::builder()
            .param("x", linear_range(0.0, 1.0))
            .param("y", linear_range(0.0, 1.0))
            .build()
            .unwrap();
        let mut rng = 12u64;
        let configs = space.latin_hypercube(20, &mut rng);
        assert_eq!(configs.len(), 20, "no constraints — all 20 should pass");
        for cfg in &configs {
            let x = cfg.float("x").unwrap();
            let y = cfg.float("y").unwrap();
            assert!((0.0..=1.0).contains(&x));
            assert!((0.0..=1.0).contains(&y));
        }
    }

    #[test]
    fn category_string_equality() {
        let c: Category = "abc".into();
        assert_eq!(c, "abc");
        let c2 = c.clone();
        assert_eq!(c, c2);
    }
}
