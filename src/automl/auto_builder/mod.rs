//! Auto-builder: diagnostic-driven config adaptation for streaming AutoML.
//!
//! Replaces random config sampling with mathematically principled derivation
//! from data characteristics and model diagnostics. The [`DiagnosticLearner`]
//! uses SPSA (Simultaneous Perturbation Stochastic Approximation) to optimize
//! learning rate and lambda directly from observed performance, replacing
//! the hardcoded signal-to-adjustment rules of the previous `DiagnosticAdaptor`.
//!
//! # Architecture
//!
//! 1. **[`FeasibleRegion`]** -- derives config bounds from (n_samples, n_features, variance)
//! 2. **[`WelfordRace`]** -- batch evaluation with center + directional perturbations
//! 3. **[`DiagnosticLearner`]** -- SPSA optimizer discovers config adjustments from performance

mod feasible;

#[cfg(feature = "distill")]
mod distillation;

#[cfg(feature = "distill")]
#[cfg_attr(docsrs, doc(cfg(feature = "distill")))]
pub use distillation::{DistillationConfig, DistillationStats};

// The CandidateDistillState type is used by WelfordRace internals when the
// distill feature is enabled.  Not re-exported (private to auto_builder).
#[cfg(feature = "distill")]
use distillation::CandidateDistillState;

pub use feasible::FeasibleRegion;

use crate::automl::ModelFactory;
use crate::ensemble::config::SGBTConfig;
use crate::learner::SGBTLearner;
use irithyll_core::learner::StreamingLearner;
use std::collections::VecDeque;
use tracing::warn;

// ---------------------------------------------------------------------------
// AM-16: Race-level drift detection (§4.5 Hetzner handoff 2026-05-07)
// ---------------------------------------------------------------------------
//
// Buffer size 1024 is spec-mandated (V10_LOCAL_CHANGES.md §4.5). It is NOT a
// tuning knob — defer configurable window to v11 per the spec's open question.
// The 1e-12 epsilon in the drift formula prevents division by zero (numerical
// stability), not an empirical threshold.

/// Error-history window for race-level drift detection (AM-16, §4.5).
///
/// Spec-mandated at 1024 samples. Halved at 512 to form the recent/baseline
/// split in [`WelfordRace::race_drift_score`]. Defer configurable window to
/// v11 (see V10_LOCAL_CHANGES.md §4.5 open questions).
const DRIFT_WINDOW: usize = 1024;

// ===========================================================================
// ConfigDiagnostics
// ===========================================================================

/// Model diagnostics consumed by the auto-builder.
///
/// Models that implement diagnostic extraction populate this struct.
/// Fields are optional signals -- return 0.0 if not available.
#[derive(Debug, Clone, Default)]
pub struct ConfigDiagnostics {
    /// Gradient alignment between consecutive steps (-1.0 to 1.0).
    /// Positive = model learning efficiently. Negative = overshooting.
    pub residual_alignment: f64,
    /// How much regularization dominates predictions (0.0 = none, high = over-regularized).
    pub regularization_sensitivity: f64,
    /// Within/between variance ratio. High = need more depth. Low = depth sufficient.
    pub depth_sufficiency: f64,
    /// Effective degrees of freedom (model complexity measure).
    pub effective_dof: f64,
    /// Uncertainty measure (honest_sigma or equivalent).
    pub uncertainty: f64,
}

// ===========================================================================
// ConfigBounds
// ===========================================================================

/// Derived bounds for each config parameter, produced by [`FeasibleRegion::config_bounds`].
///
/// Each field is a `(min, max)` tuple. The center of each range is the
/// baseline config; perturbations probe the extremes.
#[derive(Debug, Clone)]
pub struct ConfigBounds {
    /// Allowed range for `max_depth` (tree depth cap).
    ///
    /// **Range:** derived from `log2(budget)`, clamped to [2, 6].
    /// **Default center:** midpoint of the range.
    pub max_depth: (usize, usize),
    /// Allowed range for `n_steps` (number of boosting trees).
    ///
    /// **Range:** derived from `budget / 4`, clamped to [3, 50].
    /// **Default center:** midpoint of the range.
    pub n_steps: (usize, usize),
    /// Allowed range for `grace_period` (Hoeffding bound samples before split).
    ///
    /// **Range:** derived from Hoeffding bound with `delta=0.05`, clamped to [3, 200].
    /// **Default center:** midpoint of the range.
    pub grace_period: (usize, usize),
    /// Allowed range for the boosting learning rate (shrinkage).
    ///
    /// **Range:** [0.05, 0.3] (fixed — empirically safe for SGBT).
    /// **Default center:** geometric mean of min and max.
    pub learning_rate: (f64, f64),
    /// Allowed range for L2 regularization lambda.
    ///
    /// **Range:** derived from target standard deviation, clamped to [0.1, 5.0].
    /// **Default center:** geometric mean of min and max.
    pub lambda: (f64, f64),
    /// Allowed range for the number of histogram bins per feature.
    ///
    /// **Range:** [8, min(64, n_samples/4)].
    /// **Default center:** midpoint of the range.
    pub n_bins: (usize, usize),
    /// Allowed range for the feature subsample rate (column subsampling).
    ///
    /// **Range:** [0.5, 1.0] (fixed — always use at least half the features).
    /// **Default center:** midpoint of the range.
    pub feature_subsample: (f64, f64),
}

// ===========================================================================
// WelfordStats
// ===========================================================================

/// Running statistics for race evaluation using Welford's online algorithm.
#[derive(Debug, Clone, Default)]
pub struct WelfordStats {
    /// Number of observations.
    pub n: u64,
    /// Running mean of error.
    pub mean_error: f64,
    /// Running M2 for variance computation.
    pub m2: f64,
    /// Count of samples where the prediction sign matched the target sign
    /// (or both were zero). Tracked for AM-14 Pareto directional accuracy signal.
    ///
    /// Directional accuracy = `dir_correct / n` (NaN when `n == 0`).
    pub dir_correct: u64,
}

impl WelfordStats {
    /// Update running statistics with a new error value.
    pub fn update(&mut self, error: f64) {
        self.n += 1;
        let delta = error - self.mean_error;
        self.mean_error += delta / self.n as f64;
        let delta2 = error - self.mean_error;
        self.m2 += delta * delta2;
    }

    /// Record whether the prediction sign matches the target sign.
    ///
    /// Called in [`WelfordRace::feed`] alongside [`update`](Self::update).
    /// Kept separate so error tracking and direction tracking are independent.
    pub fn update_dir(&mut self, prediction: f64, target: f64) {
        // Both non-negative OR both negative → correct direction.
        // Both zero falls under non-negative → also correct.
        if (prediction >= 0.0) == (target >= 0.0) {
            self.dir_correct += 1;
        }
    }

    /// Directional accuracy ∈ [0.0, 1.0]: fraction of samples where prediction
    /// and target had the same sign.
    ///
    /// Returns `f64::NAN` when no samples have been fed (safe: NaN in any
    /// Pareto signal field causes that candidate to be excluded from the front).
    pub fn dir_accuracy(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.dir_correct as f64 / self.n as f64
        }
    }

    /// Sample variance (Bessel-corrected).
    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / (self.n - 1) as f64
        } else {
            0.0
        }
    }

    /// Standard error of the mean.
    pub fn std_error(&self) -> f64 {
        if self.n > 1 {
            (self.variance() / self.n as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }
}

// ===========================================================================
// RaceCandidate (private)
// ===========================================================================

/// A single candidate in a Welford race.
struct RaceCandidate {
    model: Box<dyn StreamingLearner>,
    stats: WelfordStats,
    config_idx: usize,
}

// Manual Debug impl because Box<dyn StreamingLearner> does not impl Debug.
impl core::fmt::Debug for RaceCandidate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RaceCandidate")
            .field("stats", &self.stats)
            .field("config_idx", &self.config_idx)
            .finish_non_exhaustive()
    }
}

// ===========================================================================
// RaceResults
// ===========================================================================

/// Results from a completed Welford race.
#[derive(Debug, Clone)]
pub struct RaceResults {
    /// Index of the winning config.
    pub winner_idx: usize,
    /// Mean error of the winner.
    pub winner_mean_error: f64,
    /// Per-config results: (config_idx, mean_error, std_error, n_samples).
    pub all_results: Vec<(usize, f64, f64, u64)>,
}

// ===========================================================================
// TerminateAfter
// ===========================================================================

/// Termination criterion for a [`WelfordRace`].
///
/// Controls when `feed()` becomes a no-op and `is_terminated()` returns `true`.
/// Once terminated, the race's frozen winner state continues to serve
/// predictions and diagnostics — only learning stops.
///
/// The default is [`TerminateAfter::Never`], which preserves backward-compatible
/// behaviour: the race runs indefinitely and the caller selects the winner
/// explicitly via [`WelfordRace::select_winner`].
///
/// # Monotonicity invariant
///
/// `is_terminated()` is monotonic: once it returns `true`, it never returns
/// `false`. Calls to `feed()` and `signal_correction()` after termination are
/// no-ops; they never panic.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TerminateAfter {
    /// Race terminates after `n` total `feed()` calls.
    ///
    /// Useful when the caller knows the dataset size ahead of time.
    Samples(u64),

    /// Race terminates after `k` explicit `signal_correction()` calls.
    ///
    /// The high-value variant: the caller's "epoch" is a domain concept
    /// (e.g. a correction cycle in a production training loop) rather than
    /// a raw sample count.
    Corrections(usize),

    /// Race terminates after wall-clock duration elapses since the first
    /// `feed()` call.
    ///
    /// The clock starts at the first `feed()`, not at construction, so the
    /// race does not expire before any data arrives.
    ///
    /// # Note on determinism
    ///
    /// `Duration`-based termination depends on real wall-clock time and is
    /// therefore non-deterministic in tests. Tests using this variant should
    /// be marked `#[ignore]` unless a mock-clock abstraction is provided.
    Duration(std::time::Duration),

    /// Race never auto-terminates. Default for backward compatibility.
    ///
    /// The caller must consume the winner via [`WelfordRace::select_winner`].
    Never,
}

// ===========================================================================
// WelfordRace
// ===========================================================================

/// Welford-based batch race: all candidates see all samples.
///
/// Uses center + directional perturbations from the feasible region.
/// No early elimination -- every config is fully evaluated.
pub struct WelfordRace {
    candidates: Vec<RaceCandidate>,
    // --- AM-13 termination state (load-bearing-explicit) ---
    termination: TerminateAfter,
    /// Monotonic termination flag. Set by `recompute_termination()`, never cleared.
    terminated: bool,
    /// Number of explicit `signal_correction()` calls received.
    correction_count: usize,
    /// Wall-clock instant of the first `feed()` call (None until then).
    first_feed_at: Option<std::time::Instant>,

    // --- AM-15 knowledge distillation (`distill` feature, default OFF) ---
    /// Active distillation config, or `None` when the feature has not been
    /// configured via `with_distillation`.  The `distill` feature flag gates
    /// compilation of the distillation module; this field stores runtime state.
    #[cfg(feature = "distill")]
    distill_cfg: Option<DistillationConfig>,
    /// Per-candidate distillation state (replay buffers, domination counters).
    #[cfg(feature = "distill")]
    distill_state: Vec<CandidateDistillState>,
    /// Aggregate distillation telemetry.
    #[cfg(feature = "distill")]
    distill_stats: DistillationStats,

    // --- AM-16: Race-level drift introspection (§4.5, load-bearing-explicit) ---
    //
    // All three are tracked explicitly — NOT derived lazily — to avoid the
    // band-aid class of bugs where state is inferred from timing assumptions.
    /// Config index of the winner at the end of the last `feed()` call.
    /// `None` until the first sample arrives.
    last_winner_idx: Option<usize>,
    /// Total number of times the winner changed. Monotonically non-decreasing.
    winner_change_count: u64,
    /// Value of `n_samples()` when the current winner last took the lead.
    /// `samples_since_last_winner_change()` derives from this on read.
    samples_at_last_winner_change: u64,
    /// Ring buffer of the winner's squared prediction errors, bounded at [`DRIFT_WINDOW`].
    ///
    /// Squared error (not absolute) for sensitivity to large outliers,
    /// which are the dominant regime-change signature (§4.5 recommendation).
    drift_recent_errors: VecDeque<f64>,
}

// Manual Debug impl to match RaceCandidate.
impl core::fmt::Debug for WelfordRace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WelfordRace")
            .field("n_candidates", &self.candidates.len())
            .field(
                "n_samples",
                &self.candidates.first().map(|c| c.stats.n).unwrap_or(0),
            )
            .field("terminated", &self.terminated)
            .field("termination", &self.termination)
            .finish()
    }
}

impl WelfordRace {
    /// Create a race from feasible region perturbation configs (SGBT-specific).
    ///
    /// Each config produces an [`SGBTLearner`] with squared loss.
    pub fn new(configs: Vec<SGBTConfig>) -> Self {
        let candidates = configs
            .into_iter()
            .enumerate()
            .map(|(i, config)| RaceCandidate {
                model: Box::new(SGBTLearner::from_config(config)),
                stats: WelfordStats::default(),
                config_idx: i,
            })
            .collect();
        Self {
            candidates,
            termination: TerminateAfter::Never,
            terminated: false,
            correction_count: 0,
            first_feed_at: None,
            #[cfg(feature = "distill")]
            distill_cfg: None,
            #[cfg(feature = "distill")]
            distill_state: Vec::new(),
            #[cfg(feature = "distill")]
            distill_stats: DistillationStats {
                disabled: true,
                n_distillations_triggered: 0,
                last_distillation_at_samples: None,
                candidates_currently_distilling: Vec::new(),
            },
            // AM-16 drift state — cold start.
            last_winner_idx: None,
            winner_change_count: 0,
            samples_at_last_winner_change: 0,
            drift_recent_errors: VecDeque::with_capacity(DRIFT_WINDOW),
        }
    }

    /// Create a race from a [`ModelFactory`] (for non-SGBT models).
    ///
    /// Uses `k` random configs from the factory's
    /// [`SearchSpace`][crate::automl::SearchSpace]. Configs that fail factory
    /// validation or constraint sampling are skipped with a logged warning;
    /// the race may contain fewer than `k` candidates if many configs are
    /// rejected.
    pub fn from_factory(factory: &dyn ModelFactory, k: usize, seed: u64) -> Self {
        let space = factory.config_space();
        // Seed must be non-zero for xorshift64.
        let mut rng = if seed == 0 { 1 } else { seed };
        let mut candidates = Vec::with_capacity(k);
        for i in 0..k {
            let params = match space.sample(&mut rng) {
                Ok(p) => p,
                Err(e) => {
                    warn!(
                        factory = factory.name(),
                        error = %e,
                        "search-space sampler unsatisfiable in WelfordRace::from_factory; skipping slot"
                    );
                    continue;
                }
            };
            match factory.create(&params) {
                Ok(model) => {
                    candidates.push(RaceCandidate {
                        model,
                        stats: WelfordStats::default(),
                        config_idx: i,
                    });
                }
                Err(e) => {
                    warn!(
                        factory = factory.name(),
                        error = %e,
                        "factory rejected config in WelfordRace::from_factory; skipping slot"
                    );
                }
            }
        }
        Self {
            candidates,
            termination: TerminateAfter::Never,
            terminated: false,
            correction_count: 0,
            first_feed_at: None,
            #[cfg(feature = "distill")]
            distill_cfg: None,
            #[cfg(feature = "distill")]
            distill_state: Vec::new(),
            #[cfg(feature = "distill")]
            distill_stats: DistillationStats {
                disabled: true,
                n_distillations_triggered: 0,
                last_distillation_at_samples: None,
                candidates_currently_distilling: Vec::new(),
            },
            // AM-16 drift state — cold start.
            last_winner_idx: None,
            winner_change_count: 0,
            samples_at_last_winner_change: 0,
            drift_recent_errors: VecDeque::with_capacity(DRIFT_WINDOW),
        }
    }

    /// Feed one sample to ALL candidates (predict-before-train).
    ///
    /// Each candidate predicts, the absolute error is recorded via Welford,
    /// then the candidate trains on the sample.
    ///
    /// If the race is terminated (via a [`TerminateAfter`] criterion set with
    /// [`WelfordRace::with_termination`]), this call is a no-op. Predictions
    /// and diagnostics continue to be served from the frozen winner state.
    pub fn feed(&mut self, features: &[f64], target: f64) {
        if self.terminated {
            return;
        }
        // Record wall-clock start on first feed (for Duration termination).
        if self.first_feed_at.is_none() {
            self.first_feed_at = Some(std::time::Instant::now());
        }
        for c in &mut self.candidates {
            let pred = c.model.predict(features);
            let error = (target - pred).abs();
            c.stats.update(error);
            c.stats.update_dir(pred, target);
            c.model.train_one(features, target, 1.0);
        }

        // AM-15 distillation pass (compiled only with `distill` feature).
        // Uses scalar min-error winner as the distillation target until AM-14
        // Pareto wiring lands.  The Pareto front reduces to {winner} here, which
        // is correct: the Pareto winner is trivially a front of one when
        // scalar ordering is used.
        #[cfg(feature = "distill")]
        if self.distill_cfg.is_some() {
            let winner_idx = self
                .candidates
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.stats
                        .mean_error
                        .partial_cmp(&b.stats.mean_error)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            let pareto_front = vec![winner_idx];
            self.run_distillation_pass(features, target, &pareto_front, winner_idx);
        }

        // AM-16: Update race-level drift introspection state.
        //
        // Uses the scalar min-error winner (current_winner_idx_scalar) so that
        // AM-14's future Pareto changes to winner semantics do not invalidate
        // these counters. The scalar back-compat winner is the stable anchor
        // that both AM-14 and AM-16 coordinate around.
        let current_winner = self.current_winner_idx_scalar();
        if current_winner != self.last_winner_idx {
            self.winner_change_count += 1;
            self.samples_at_last_winner_change = self.n_samples();
            self.last_winner_idx = current_winner;
        }
        // Record the winner's squared prediction error into the drift ring buffer.
        // `current_winner_idx_scalar()` returns config_idx; resolve to a vec
        // position first to split the borrow from the mutating push_back.
        if let Some(winner_cfg_idx) = current_winner {
            let winner_pos = self
                .candidates
                .iter()
                .position(|c| c.config_idx == winner_cfg_idx);
            if let Some(pos) = winner_pos {
                // Post-train re-predict for telemetry — does not affect
                // training state or Welford accumulators.
                let sq_err = {
                    let winner_pred = self.candidates[pos].model.predict(features);
                    (target - winner_pred).powi(2)
                };
                if self.drift_recent_errors.len() >= DRIFT_WINDOW {
                    self.drift_recent_errors.pop_front();
                }
                self.drift_recent_errors.push_back(sq_err);
            }
        }

        self.recompute_termination();
    }

    /// Select winner by lowest Welford mean error.
    ///
    /// Consumes the race and returns the winning model along with full results.
    pub fn select_winner(self) -> (Box<dyn StreamingLearner>, RaceResults) {
        let mut results: Vec<(usize, f64, f64, u64)> = self
            .candidates
            .iter()
            .map(|c| {
                (
                    c.config_idx,
                    c.stats.mean_error,
                    c.stats.std_error(),
                    c.stats.n,
                )
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let winner_idx = results[0].0;
        let winner_mean = results[0].1;

        let winner_model = self
            .candidates
            .into_iter()
            .find(|c| c.config_idx == winner_idx)
            .map(|c| c.model)
            .expect("winner must exist in candidates");

        (
            winner_model,
            RaceResults {
                winner_idx,
                winner_mean_error: winner_mean,
                all_results: results,
            },
        )
    }

    /// Number of candidates in the race.
    pub fn n_candidates(&self) -> usize {
        self.candidates.len()
    }

    /// Number of samples fed so far (from the first candidate).
    pub fn n_samples(&self) -> u64 {
        self.candidates.first().map(|c| c.stats.n).unwrap_or(0)
    }

    /// Private helper: config_idx of the current scalar (min mean-error) winner.
    ///
    /// Used by AM-16 race-level drift introspection as a stable back-compat
    /// anchor. Returns `None` when no samples have been fed yet.
    fn current_winner_idx_scalar(&self) -> Option<usize> {
        self.candidates
            .iter()
            .filter(|c| c.stats.n > 0)
            .min_by(|a, b| {
                a.stats
                    .mean_error
                    .partial_cmp(&b.stats.mean_error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.config_idx)
    }
}

// ===========================================================================
// WelfordRace — AM-13 termination (separate impl block, parallel-agent safe)
// ===========================================================================

impl WelfordRace {
    /// Attach a [`TerminateAfter`] criterion to this race (builder-style).
    ///
    /// Once the criterion is met, [`WelfordRace::feed`] becomes a no-op and
    /// [`WelfordRace::is_terminated`] returns `true`. The frozen winner state
    /// continues to serve predictions and diagnostics.
    ///
    /// The default criterion is [`TerminateAfter::Never`], which preserves
    /// backward-compatible behaviour.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use irithyll::automl::{WelfordRace, TerminateAfter, FeasibleRegion};
    /// let region = FeasibleRegion::from_data(200, 2, 1.0);
    /// let mut race = WelfordRace::new(region.perturbation_configs())
    ///     .with_termination(TerminateAfter::Corrections(10));
    /// assert!(!race.is_terminated());
    /// ```
    pub fn with_termination(mut self, criterion: TerminateAfter) -> Self {
        self.termination = criterion;
        self
    }

    /// Returns `true` if a [`TerminateAfter`] criterion has been met.
    ///
    /// This flag is **monotonic**: once `true`, it is never `false` again.
    /// Calls to [`WelfordRace::feed`] and [`WelfordRace::signal_correction`]
    /// after termination are no-ops.
    #[inline]
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }

    /// Notify the race that one correction cycle has completed.
    ///
    /// Increments the internal correction counter. When a
    /// [`TerminateAfter::Corrections(k)`][TerminateAfter::Corrections]
    /// criterion is active and the counter reaches `k`, the race terminates
    /// on this call.
    ///
    /// No-op if the race is already terminated.
    pub fn signal_correction(&mut self) {
        if self.terminated {
            return;
        }
        self.correction_count += 1;
        self.recompute_termination();
    }

    /// Returns the number of remaining `feed()` calls until termination, or
    /// `None` if the criterion is not sample-based (i.e. [`TerminateAfter::Never`],
    /// [`TerminateAfter::Corrections`], or [`TerminateAfter::Duration`]).
    ///
    /// Returns `Some(0)` when the race is already terminated by a
    /// [`TerminateAfter::Samples`] criterion.
    pub fn samples_until_termination(&self) -> Option<u64> {
        match self.termination {
            TerminateAfter::Samples(n) => {
                let seen = self.n_samples();
                Some(n.saturating_sub(seen))
            }
            _ => None,
        }
    }

    /// Recompute the termination flag from the current criterion and state.
    ///
    /// Called internally after each `feed()` and `signal_correction()`.
    /// The flag is monotonic: it can only transition false → true.
    fn recompute_termination(&mut self) {
        if self.terminated {
            return; // already terminated; never flip back
        }
        self.terminated = match self.termination {
            TerminateAfter::Samples(n) => self.n_samples() >= n,
            TerminateAfter::Corrections(k) => self.correction_count >= k,
            TerminateAfter::Duration(d) => self
                .first_feed_at
                .map(|t| t.elapsed() >= d)
                .unwrap_or(false),
            TerminateAfter::Never => false,
        };
    }
}

// ===========================================================================
// WelfordRace — AM-16 race-level drift detection (separate impl block)
// V10_LOCAL_CHANGES.md §4.5 (Hetzner handoff 2026-05-07)
// ===========================================================================

impl WelfordRace {
    /// Number of `feed()` calls since the current winner last changed.
    ///
    /// Higher = race has converged on a stable winner; callers may safely
    /// avoid a re-race. Resets to zero on every winner change.
    ///
    /// Returns `0` before any samples have been fed (no winner exists yet).
    pub fn samples_since_last_winner_change(&self) -> u64 {
        self.n_samples()
            .saturating_sub(self.samples_at_last_winner_change)
    }

    /// Total number of times the winner has changed since construction.
    ///
    /// Monotonically non-decreasing. High count over a short window indicates
    /// a high-variance regime where the race has not converged on a stable winner.
    pub fn winner_change_count(&self) -> u64 {
        self.winner_change_count
    }

    /// Dimensionless drift score over the last 1024 winner prediction errors.
    ///
    /// # Formula (verbatim per V10_LOCAL_CHANGES.md §4.5)
    ///
    /// ```text
    /// drift_score = (recent_half_mean_error - baseline_half_mean_error)
    ///             / (baseline_half_mean_error.abs() + 1e-12)
    /// ```
    ///
    /// Errors are **squared** (not absolute) for sensitivity to large outliers,
    /// which are the dominant regime-change signature (§4.5 recommendation).
    /// The `1e-12` epsilon prevents division by zero (numerical stability —
    /// not a tuning knob).
    ///
    /// # Interpretation
    ///
    /// - **Positive** (> 0): recent errors larger than baseline — drift is occurring.
    /// - **Zero** (= 0): returned when fewer than 512 errors collected (cold start).
    /// - **Negative** (< 0): recent errors smaller than baseline — winner still improving.
    ///
    /// A score `> 0.5` is a reasonable re-race trigger: schedule a fresh race
    /// with refreshed [`FeasibleRegion`] bounds (V10_LOCAL_CHANGES.md §4.5).
    ///
    /// # Cold-start convention
    ///
    /// Returns `0.0` when fewer than `1024 / 2` (512) errors have been collected.
    /// The buffer must reach half-full before a meaningful baseline/recent split
    /// exists — `0.0` is the correct cold-start value per spec, not a placeholder.
    pub fn race_drift_score(&self) -> f64 {
        let half = DRIFT_WINDOW / 2; // 512 — minimum for a valid baseline/recent split
        if self.drift_recent_errors.len() < half {
            return 0.0;
        }
        let mid = self.drift_recent_errors.len() / 2;
        let baseline: f64 = self.drift_recent_errors.iter().take(mid).sum::<f64>() / mid as f64;
        let recent: f64 = self.drift_recent_errors.iter().skip(mid).sum::<f64>()
            / (self.drift_recent_errors.len() - mid) as f64;
        (recent - baseline) / (baseline.abs() + 1e-12)
    }
}

// ===========================================================================
// AM-14: Pareto winner selection
// ===========================================================================

/// Five-dimensional signal vector for one race candidate.
///
/// Signals: `mean_error` down, `se_error` down, `empirical_sigma` down,
/// `n_steps` up, `dir_accuracy` up. NaN excludes candidate from Pareto front.
#[derive(Debug, Clone, Copy)]
struct CandidateSignals {
    mean_error: f64,
    se_error: f64,
    empirical_sigma: f64,
    n_steps: f64,
    dir_accuracy: f64,
}

impl CandidateSignals {
    /// Extract the five Pareto signals from a [`RaceCandidate`].
    /// `empirical_sigma` = `diagnostics_array()[4]`; 0.0 for non-distributional models.
    #[allow(deprecated)]
    fn from_candidate(c: &RaceCandidate) -> Self {
        let diag = c.model.diagnostics_array();
        CandidateSignals {
            mean_error: c.stats.mean_error,
            se_error: c.stats.std_error(),
            empirical_sigma: diag[4],
            n_steps: c.stats.n as f64,
            dir_accuracy: c.stats.dir_accuracy(),
        }
    }

    fn has_nan(&self) -> bool {
        self.mean_error.is_nan()
            || self.se_error.is_nan()
            || self.empirical_sigma.is_nan()
            || self.n_steps.is_nan()
            || self.dir_accuracy.is_nan()
    }
}

/// `a` strictly Pareto-dominates `b`: no-worse on all, strictly better on one.
/// epsilon-Pareto rejected per no-arbitrary-threshold discipline (2026-05-06).
fn pareto_dominates(a: &CandidateSignals, b: &CandidateSignals) -> bool {
    let no_worse_lower = a.mean_error <= b.mean_error
        && a.se_error <= b.se_error
        && a.empirical_sigma <= b.empirical_sigma;
    let no_worse_higher = a.n_steps >= b.n_steps && a.dir_accuracy >= b.dir_accuracy;
    let strictly_better = a.mean_error < b.mean_error
        || a.se_error < b.se_error
        || a.empirical_sigma < b.empirical_sigma
        || a.n_steps > b.n_steps
        || a.dir_accuracy > b.dir_accuracy;
    no_worse_lower && no_worse_higher && strictly_better
}

/// AM-14 Pareto winner selection -- separate impl block (parallel-agent safe).
///
/// Fixed signal vector: `(mean_error down, se_error down, empirical_sigma down,
/// n_steps up, dir_accuracy up)`. Layers on top of AM-2 Bernstein racing.
impl WelfordRace {
    /// Indices of all Pareto-nondominated candidates. Empty only when no candidates exist.
    pub fn pareto_front(&self) -> Vec<usize> {
        let signals: Vec<Option<CandidateSignals>> = self
            .candidates
            .iter()
            .map(|c| {
                let sig = CandidateSignals::from_candidate(c);
                if sig.has_nan() {
                    None
                } else {
                    Some(sig)
                }
            })
            .collect();

        (0..self.candidates.len())
            .filter(|&i| {
                let Some(sig_i) = signals[i] else {
                    return false;
                };
                !signals.iter().enumerate().any(|(j, sig_j_opt)| {
                    if j == i {
                        return false;
                    }
                    match sig_j_opt {
                        Some(sig_j) => pareto_dominates(sig_j, &sig_i),
                        None => false,
                    }
                })
            })
            .collect()
    }

    /// Winning candidate index: Pareto front then Bernstein tiebreak (Maurer & Pontil 2009).
    /// Falls back to min(mean_error) if Bernstein is inconclusive.
    /// Returns `None` when no samples have been fed.
    pub fn pareto_winner_idx(&self) -> Option<usize> {
        let front = self.pareto_front();
        if front.is_empty() {
            return None;
        }
        if front.len() == 1 {
            return Some(front[0]);
        }

        use crate::automl::racing::{bernstein_compare, BERNSTEIN_DELTA};
        let arm_stats: Vec<crate::automl::racing::ArmStats> = front
            .iter()
            .map(|&idx| {
                let c = &self.candidates[idx];
                let n = c.stats.n;
                let range = if n > 1 {
                    4.0 * (c.stats.m2 / (n - 1) as f64).sqrt()
                } else {
                    0.0
                };
                (c.stats.mean_error, c.stats.m2, n, range)
            })
            .collect();

        if let Some(front_slot) = bernstein_compare(&arm_stats, BERNSTEIN_DELTA) {
            return Some(front[front_slot]);
        }

        front.into_iter().min_by(|&a, &b| {
            self.candidates[a]
                .stats
                .mean_error
                .partial_cmp(&self.candidates[b].stats.mean_error)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Inject raw Welford stats directly into a candidate (test-only).
    ///
    /// Bypasses `feed()` to set precise signal values for Pareto tests.
    /// Uses `idx` as the position in `self.candidates` (NOT `config_idx`).
    #[cfg(test)]
    pub(super) fn inject_stats_for_test(
        &mut self,
        pos: usize,
        n: u64,
        mean_error: f64,
        m2: f64,
        dir_correct: u64,
    ) {
        self.candidates[pos].stats.n = n;
        self.candidates[pos].stats.mean_error = mean_error;
        self.candidates[pos].stats.m2 = m2;
        self.candidates[pos].stats.dir_correct = dir_correct;
    }
}

// ===========================================================================
// SmoothAdjustments + StructuralChange
// ===========================================================================

/// Smooth parameter adjustments (applied every sample).
#[derive(Debug, Clone, Default)]
pub struct SmoothAdjustments {
    /// Learning rate multiplier (0.5 = halve, 2.0 = double, 1.0 = no change).
    pub lr_multiplier: f64,
    /// Lambda direction: positive = increase, negative = decrease, 0.0 = no change.
    pub lambda_direction: f64,
}

/// Structural changes (queued for next tree replacement).
#[derive(Debug, Clone, Default)]
pub struct StructuralChange {
    /// Suggested depth change (+1, -1, or 0).
    pub depth_delta: i32,
    /// Suggested n_steps change (+2, -2, or 0).
    pub steps_delta: i32,
}

// ===========================================================================
// MetaObjective
// ===========================================================================

/// Configurable optimization objective for the meta-learner.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub enum MetaObjective {
    /// Minimize root mean squared error (default for regression).
    #[default]
    MinimizeRMSE,
    /// Maximize R-squared (coefficient of determination).
    MaximizeR2,
    /// Maximize directional accuracy (correct sign prediction).
    MaximizeDirection,
    /// Maximize F1 score (harmonic mean of precision and recall).
    MaximizeF1,
    /// Maximize Cohen's kappa (agreement beyond chance).
    MaximizeKappa,
    /// Weighted combination of multiple objectives.
    Composite {
        /// Weight for RMSE component (lower is better, so this is negated internally).
        rmse_weight: f64,
        /// Weight for R-squared component.
        r2_weight: f64,
        /// Weight for directional accuracy component.
        dir_weight: f64,
    },
}

// ===========================================================================
// SPSAPhase
// ===========================================================================

/// Phase of the SPSA optimization cycle.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SPSAPhase {
    /// Collecting initial performance variance (first 50 samples).
    Init,
    /// Evaluating performance under theta + c*delta perturbation.
    PerturbPlus,
    /// Evaluating performance under theta - c*delta perturbation.
    PerturbMinus,
}

// ===========================================================================
// DiagnosticLearner
// ===========================================================================

/// SPSA optimizer that discovers config adjustments from performance signals.
///
/// Replaces the hardcoded signal-to-adjustment rules of the previous
/// `DiagnosticAdaptor` with SPSA (Simultaneous Perturbation Stochastic
/// Approximation) that optimizes learning rate and lambda directly from
/// observed performance, using only 2 function evaluations per iteration
/// regardless of parameter dimensionality.
///
/// # SPSA optimization cycle
///
/// 1. Baselines always update (every sample)
/// 2. Performance trackers always update (RMSE, R², direction, F1, kappa)
/// 3. Init phase (first 50 samples): calibrate perturbation from noise variance
/// 4. PerturbPlus: evaluate performance under theta + c*delta
/// 5. PerturbMinus: evaluate performance under theta - c*delta
/// 6. Gradient estimate + theta update with divergence guard and CUSUM regime detection
#[derive(Debug)]
pub struct DiagnosticLearner {
    /// Running EWMA of uncertainty for baseline comparison.
    uncertainty_ewma: f64,
    /// Running EWMA of residual alignment.
    alignment_ewma: f64,
    /// Running EWMA of regularization sensitivity.
    reg_sensitivity_ewma: f64,
    /// Running EWMA of depth sufficiency signal.
    depth_signal_ewma: f64,
    /// Running EWMA of effective DOF.
    dof_ewma: f64,
    /// EWMA decay factor.
    alpha: f64,
    /// Current feasible region.
    region: FeasibleRegion,
    /// Samples seen.
    n_samples: u64,
    /// Whether initialization phase (first 50 samples = pure observation).
    initialized: bool,

    // --- Observation interval ---
    /// Minimum samples between phase transitions (from grace_period center).
    observation_interval: u64,

    // --- SPSA config optimization ---
    /// \[lr_log_normalized, lambda_normalized\] in \[0, 1\].
    theta: [f64; 2],
    /// Best theta found so far.
    theta_best: [f64; 2],
    /// Best performance observed.
    best_performance: f64,
    /// Initial step size.
    a_init: f64,
    /// Current step size (may be halved by divergence guard).
    a: f64,
    /// Initial perturbation magnitude (calibrated from noise).
    c_init: f64,
    /// Per-dimension perturbation floor.
    c_floor: [f64; 2],
    /// Local iteration counter (reset on regime change).
    k_local: u64,
    /// Stability constant for gain sequence.
    big_a: f64,
    /// Current SPSA phase.
    phase: SPSAPhase,
    /// Bernoulli +/-1 perturbation vector.
    current_delta: [f64; 2],
    /// Performance recorded during PerturbPlus phase.
    perf_plus: f64,
    /// Performance recorded during PerturbMinus phase.
    perf_minus: f64,
    /// Samples accumulated in the current phase.
    samples_in_phase: u64,
    // --- Regime detection (CUSUM) ---
    /// CUSUM statistic for regime change detection.
    cusum_s: f64,
    /// Baseline performance for CUSUM.
    perf_ewma_baseline: f64,
    /// Running variance of performance.
    perf_variance: f64,
    // --- Config tracking for delta emission ---
    /// Last theta used for adjustment emission.
    last_emitted_theta: [f64; 2],
    /// Total SPSA steps completed.
    total_steps: u64,
    /// xorshift64 RNG state.
    rng_state: u64,

    /// Optimization objective.
    objective: MetaObjective,

    // --- Performance trackers ---
    /// EWMA of squared errors (for RMSE).
    squared_error_ewma: f64,
    /// EWMA of target values (for R²).
    target_ewma: f64,
    /// EWMA of squared target deviation from mean (for R²).
    target_var_ewma: f64,
    /// EWMA of correct direction predictions (for directional accuracy).
    direction_ewma: f64,
    /// EWMA of true positives (for F1).
    tp_ewma: f64,
    /// EWMA of false positives (for F1).
    fp_ewma: f64,
    /// EWMA of false negatives (for F1).
    fn_ewma: f64,
    /// EWMA of observed accuracy (for kappa).
    accuracy_ewma: f64,
    /// EWMA of positive rate in targets (for kappa).
    pos_rate_ewma: f64,
    /// EWMA of positive rate in predictions (for kappa).
    pred_pos_rate_ewma: f64,
}

impl DiagnosticLearner {
    /// Create a new learner backed by a feasible region with default objective.
    pub fn new(region: FeasibleRegion) -> Self {
        Self::with_objective(region, MetaObjective::default())
    }

    /// Create a new learner backed by a feasible region with a specific objective.
    pub fn with_objective(region: FeasibleRegion, objective: MetaObjective) -> Self {
        let bounds = region.config_bounds();

        // Observation interval: center of grace_period range, minimum 1.
        // Capped at 50 to ensure the SPSA optimizer gets enough gradient
        // updates within typical stream lengths (1000-5000 samples).
        let observation_interval =
            ((bounds.grace_period.0 + bounds.grace_period.1) / 2).clamp(1, 50) as u64;

        // SPSA gain sequence parameters.
        let big_a = 10.0;
        let a_init = 0.05 * (big_a + 1.0_f64).powf(0.602);
        let c_floor = [0.001; 2];

        Self {
            uncertainty_ewma: 0.0,
            alignment_ewma: 0.0,
            reg_sensitivity_ewma: 0.0,
            depth_signal_ewma: 0.0,
            dof_ewma: 0.0,
            alpha: 1.0 - (-2.0 / observation_interval as f64).exp(),
            region,
            n_samples: 0,
            initialized: false,
            observation_interval,
            theta: [0.5, 0.5],
            theta_best: [0.5, 0.5],
            best_performance: f64::NEG_INFINITY,
            a_init,
            a: a_init,
            c_init: 0.1,
            c_floor,
            k_local: 0,
            big_a,
            phase: SPSAPhase::Init,
            current_delta: [0.0; 2],
            perf_plus: 0.0,
            perf_minus: 0.0,
            samples_in_phase: 0,
            cusum_s: 0.0,
            perf_ewma_baseline: 0.0,
            perf_variance: 0.0,
            last_emitted_theta: [0.5, 0.5],
            total_steps: 0,
            rng_state: 0xDEAD_BEEF_CAFE_1234,
            objective,
            squared_error_ewma: 0.0,
            target_ewma: 0.0,
            target_var_ewma: 0.0,
            direction_ewma: 0.5,
            tp_ewma: 0.0,
            fp_ewma: 0.0,
            fn_ewma: 0.0,
            accuracy_ewma: 0.5,
            pos_rate_ewma: 0.5,
            pred_pos_rate_ewma: 0.5,
        }
    }

    /// Process diagnostics after each `train_one()`. Returns smooth adjustments.
    ///
    /// The SPSA optimizer cycles through Init -> PerturbPlus -> PerturbMinus
    /// phases, estimating gradients from paired performance evaluations and
    /// updating theta (normalized config parameters) accordingly.
    /// During the first 50 samples, only baselines are updated (no adjustments).
    pub fn after_train(
        &mut self,
        diagnostics: &ConfigDiagnostics,
        prediction: f64,
        target: f64,
    ) -> SmoothAdjustments {
        self.n_samples += 1;
        let a = self.alpha;

        // 1. Always update diagnostic baselines (EWMAs).
        self.uncertainty_ewma = a * diagnostics.uncertainty + (1.0 - a) * self.uncertainty_ewma;
        self.alignment_ewma = a * diagnostics.residual_alignment + (1.0 - a) * self.alignment_ewma;
        self.reg_sensitivity_ewma =
            a * diagnostics.regularization_sensitivity + (1.0 - a) * self.reg_sensitivity_ewma;
        self.depth_signal_ewma =
            a * diagnostics.depth_sufficiency + (1.0 - a) * self.depth_signal_ewma;
        self.dof_ewma = a * diagnostics.effective_dof + (1.0 - a) * self.dof_ewma;

        // 2. Always update performance trackers.
        let error = target - prediction;
        self.squared_error_ewma = a * (error * error) + (1.0 - a) * self.squared_error_ewma;

        let old_target_ewma = self.target_ewma;
        self.target_ewma = a * target + (1.0 - a) * self.target_ewma;
        let dev = target - old_target_ewma;
        self.target_var_ewma = a * (dev * dev) + (1.0 - a) * self.target_var_ewma;

        // Directional accuracy: both have the same sign (or both zero).
        let correct_dir = if (prediction * target) >= 0.0 {
            1.0
        } else {
            0.0
        };
        self.direction_ewma = a * correct_dir + (1.0 - a) * self.direction_ewma;

        // F1 components.
        let predicted_positive = prediction > 0.5;
        let actual_positive = target > 0.5;
        let tp = if predicted_positive && actual_positive {
            1.0
        } else {
            0.0
        };
        let fp = if predicted_positive && !actual_positive {
            1.0
        } else {
            0.0
        };
        let fn_ = if !predicted_positive && actual_positive {
            1.0
        } else {
            0.0
        };
        self.tp_ewma = a * tp + (1.0 - a) * self.tp_ewma;
        self.fp_ewma = a * fp + (1.0 - a) * self.fp_ewma;
        self.fn_ewma = a * fn_ + (1.0 - a) * self.fn_ewma;

        // Kappa components.
        let correct = if (predicted_positive && actual_positive)
            || (!predicted_positive && !actual_positive)
        {
            1.0
        } else {
            0.0
        };
        self.accuracy_ewma = a * correct + (1.0 - a) * self.accuracy_ewma;
        self.pos_rate_ewma =
            a * (if actual_positive { 1.0 } else { 0.0 }) + (1.0 - a) * self.pos_rate_ewma;
        self.pred_pos_rate_ewma =
            a * (if predicted_positive { 1.0 } else { 0.0 }) + (1.0 - a) * self.pred_pos_rate_ewma;

        // 3. Increment phase sample counter.
        self.samples_in_phase += 1;

        // 4. SPSA phase state machine.
        let no_op = SmoothAdjustments {
            lr_multiplier: 1.0,
            lambda_direction: 0.0,
        };

        match self.phase {
            SPSAPhase::Init => {
                // Accumulate performance variance for c_init calibration.
                let perf = self.current_performance();
                self.perf_variance =
                    a * (perf - self.perf_ewma_baseline).powi(2) + (1.0 - a) * self.perf_variance;
                self.perf_ewma_baseline = a * perf + (1.0 - a) * self.perf_ewma_baseline;

                if self.samples_in_phase >= 50 {
                    // Calibrate c_init from observed noise.
                    // Clamped conservatively to avoid large config swings that
                    // destabilize the champion during exploration phases.
                    let noise_std = self.perf_variance.sqrt();
                    self.c_init = (2.0 * noise_std).clamp(0.005, 0.08);
                    self.initialized = true;

                    // Generate first perturbation and transition to PerturbPlus.
                    self.generate_delta();
                    self.phase = SPSAPhase::PerturbPlus;
                    self.samples_in_phase = 0;

                    // Apply theta + c*delta config.
                    let target_theta = self.perturbed_theta(1.0);
                    return self.adjustment_for_theta(&target_theta);
                }
                no_op
            }

            SPSAPhase::PerturbPlus => {
                if self.samples_in_phase >= self.observation_interval {
                    // Record performance under theta + c*delta.
                    self.perf_plus = self.current_performance();

                    // Transition to PerturbMinus: apply theta - c*delta.
                    self.phase = SPSAPhase::PerturbMinus;
                    self.samples_in_phase = 0;

                    let target_theta = self.perturbed_theta(-1.0);
                    return self.adjustment_for_theta(&target_theta);
                }
                no_op
            }

            SPSAPhase::PerturbMinus => {
                if self.samples_in_phase >= self.observation_interval {
                    // Record performance under theta - c*delta.
                    self.perf_minus = self.current_performance();

                    // Do SPSA gradient update.
                    self.do_spsa_update();

                    // Transition back to PerturbPlus with new delta.
                    self.generate_delta();
                    self.phase = SPSAPhase::PerturbPlus;
                    self.samples_in_phase = 0;

                    // Apply new theta + c*delta config.
                    let target_theta = self.perturbed_theta(1.0);
                    return self.adjustment_for_theta(&target_theta);
                }
                no_op
            }
        }
    }

    /// Fallback for callers that cannot provide prediction/target.
    ///
    /// Passes zero prediction and zero target, which still updates diagnostic
    /// baselines and interval gating but provides no useful performance signal
    /// to the meta-learner.
    pub fn after_train_diagnostics_only(
        &mut self,
        diagnostics: &ConfigDiagnostics,
    ) -> SmoothAdjustments {
        self.after_train(diagnostics, 0.0, 0.0)
    }

    /// Evaluate structural changes at tree replacement boundary.
    ///
    /// Returns `Some` if the diagnostics suggest depth or step count changes,
    /// `None` if the current structure is adequate.
    ///
    /// Structural changes remain rule-based (not learned) because they are
    /// infrequent, discrete events that the meta-learner cannot observe
    /// often enough to learn from.
    pub fn at_replacement(&mut self, diagnostics: &ConfigDiagnostics) -> Option<StructuralChange> {
        if !self.initialized {
            return None;
        }

        // Update feasible region with current sample count.
        self.region.update(self.n_samples as usize);
        let bounds = self.region.config_bounds();

        // Depth sufficiency: compare current signal to baseline.
        let needs_more_depth = diagnostics.depth_sufficiency > self.depth_signal_ewma * 1.5
            && bounds.max_depth.1 > bounds.max_depth.0; // room to grow

        // DOF ratio: effective DOF relative to data.
        let dof_ratio = if self.n_samples > 0 {
            diagnostics.effective_dof / self.n_samples as f64
        } else {
            0.0
        };

        // Target DOF ratio derived from feasible region budget.
        let target_dof_ratio = (self.region.budget() / self.n_samples as f64).clamp(0.01, 0.5);
        let needs_more_steps = dof_ratio < target_dof_ratio * 0.5;
        let needs_fewer_steps = dof_ratio > target_dof_ratio * 2.0;

        if needs_more_depth || needs_more_steps || needs_fewer_steps {
            Some(StructuralChange {
                depth_delta: if needs_more_depth { 1 } else { 0 },
                steps_delta: if needs_more_steps {
                    2
                } else if needs_fewer_steps {
                    -2
                } else {
                    0
                },
            })
        } else {
            None
        }
    }

    /// Reset the learner state (all EWMA baselines, SPSA state, and performance trackers).
    pub fn reset(&mut self) {
        self.uncertainty_ewma = 0.0;
        self.alignment_ewma = 0.0;
        self.reg_sensitivity_ewma = 0.0;
        self.depth_signal_ewma = 0.0;
        self.dof_ewma = 0.0;
        self.n_samples = 0;
        self.initialized = false;
        // SPSA state
        self.theta = [0.5, 0.5];
        self.theta_best = [0.5, 0.5];
        self.best_performance = f64::NEG_INFINITY;
        self.a = self.a_init;
        self.c_init = 0.1;
        self.k_local = 0;
        self.phase = SPSAPhase::Init;
        self.current_delta = [0.0; 2];
        self.perf_plus = 0.0;
        self.perf_minus = 0.0;
        self.samples_in_phase = 0;
        self.cusum_s = 0.0;
        self.perf_ewma_baseline = 0.0;
        self.perf_variance = 0.0;
        self.last_emitted_theta = [0.5, 0.5];
        self.total_steps = 0;
        self.rng_state = 0xDEAD_BEEF_CAFE_1234;
        // Performance trackers
        self.squared_error_ewma = 0.0;
        self.target_ewma = 0.0;
        self.target_var_ewma = 0.0;
        self.direction_ewma = 0.5;
        self.tp_ewma = 0.0;
        self.fp_ewma = 0.0;
        self.fn_ewma = 0.0;
        self.accuracy_ewma = 0.5;
        self.pos_rate_ewma = 0.5;
        self.pred_pos_rate_ewma = 0.5;
    }

    /// Current feasible region.
    pub fn region(&self) -> &FeasibleRegion {
        &self.region
    }

    /// Update the feasible region with current data characteristics.
    ///
    /// Called by the AutoTuner as it accumulates target variance estimates.
    /// This recalibrates config bounds (lambda, grace period) to match the
    /// actual data distribution rather than the conservative initial guess.
    pub fn update_region(&mut self, n_samples: usize, target_variance: f64) {
        self.region.update(n_samples);
        self.region.update_variance(target_variance);
    }

    /// Total SPSA optimization steps completed.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Current optimization objective.
    pub fn objective(&self) -> MetaObjective {
        self.objective
    }

    /// Current SPSA phase (for testing).
    #[cfg(test)]
    fn phase(&self) -> SPSAPhase {
        self.phase
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Compute current performance value based on the configured objective.
    ///
    /// Higher is always better (for "minimize" objectives we negate).
    fn current_performance(&self) -> f64 {
        match self.objective {
            MetaObjective::MinimizeRMSE => -self.squared_error_ewma.sqrt(),
            MetaObjective::MaximizeR2 => {
                if self.target_var_ewma > 1e-15 {
                    1.0 - self.squared_error_ewma / self.target_var_ewma
                } else {
                    0.0
                }
            }
            MetaObjective::MaximizeDirection => self.direction_ewma,
            MetaObjective::MaximizeF1 => {
                let denom = 2.0 * self.tp_ewma + self.fp_ewma + self.fn_ewma;
                if denom > 1e-15 {
                    2.0 * self.tp_ewma / denom
                } else {
                    0.0
                }
            }
            MetaObjective::MaximizeKappa => {
                let expected = self.pos_rate_ewma * self.pred_pos_rate_ewma
                    + (1.0 - self.pos_rate_ewma) * (1.0 - self.pred_pos_rate_ewma);
                if (1.0 - expected).abs() > 1e-15 {
                    (self.accuracy_ewma - expected) / (1.0 - expected)
                } else {
                    0.0
                }
            }
            MetaObjective::Composite {
                rmse_weight,
                r2_weight,
                dir_weight,
            } => {
                let rmse_score = -self.squared_error_ewma.sqrt();
                let r2_score = if self.target_var_ewma > 1e-15 {
                    1.0 - self.squared_error_ewma / self.target_var_ewma
                } else {
                    0.0
                };
                let dir_score = self.direction_ewma;
                rmse_weight * rmse_score + r2_weight * r2_score + dir_weight * dir_score
            }
        }
    }

    /// Perform the SPSA gradient update after both perturbation evaluations.
    fn do_spsa_update(&mut self) {
        let a_k = self.a / (self.big_a + self.k_local as f64 + 1.0).powf(0.602);
        let c_k_base = self.c_init / (self.k_local as f64 + 1.0).powf(0.101);

        for i in 0..2 {
            let c_k = c_k_base.max(self.c_floor[i]);
            if self.current_delta[i].abs() > 0.5 {
                let g_hat =
                    (self.perf_plus - self.perf_minus) / (2.0 * c_k * self.current_delta[i]);
                self.theta[i] += a_k * g_hat; // MAXIMIZE
                self.theta[i] = self.theta[i].clamp(0.0, 1.0);
            }
        }

        // Ito-Dhaene divergence guard.
        if self.perf_plus < self.best_performance && self.perf_minus < self.best_performance {
            self.a *= 0.5;
            self.theta = self.theta_best;
        } else {
            let best = self.perf_plus.max(self.perf_minus);
            if best > self.best_performance {
                self.best_performance = best;
                self.theta_best = self.theta;
            }
        }

        // CUSUM regime detection.
        let drift_margin = 0.5 * self.perf_variance.sqrt();
        let drift_threshold = 5.0 * self.perf_variance.sqrt();
        let current = self.current_performance();
        self.cusum_s = (self.cusum_s + (self.perf_ewma_baseline - current) - drift_margin).max(0.0);
        if self.cusum_s > drift_threshold && drift_threshold > 1e-15 {
            self.k_local = 0;
            self.a = self.a_init;
            self.cusum_s = 0.0;
            self.perf_ewma_baseline = current;
        }

        self.k_local += 1;
        self.total_steps += 1;

        // Update variance tracking.
        let perf = self.current_performance();
        let a = self.alpha;
        self.perf_variance =
            a * (perf - self.perf_ewma_baseline).powi(2) + (1.0 - a) * self.perf_variance;
        self.perf_ewma_baseline = a * perf + (1.0 - a) * self.perf_ewma_baseline;
    }

    /// Convert normalized theta to actual (lr, lambda) config values.
    fn theta_to_config(&self, theta: &[f64; 2]) -> (f64, f64) {
        let bounds = self.region.config_bounds();
        let lr = bounds.learning_rate.0
            * (bounds.learning_rate.1 / bounds.learning_rate.0.max(1e-15)).powf(theta[0]);
        let lambda = bounds.lambda.0 + theta[1] * (bounds.lambda.1 - bounds.lambda.0);
        (lr.max(1e-10), lambda.max(0.0))
    }

    /// Compute the adjustment to move from last_emitted_theta to the target theta.
    ///
    /// Adjustments are dampened to prevent large config swings from
    /// destabilizing the champion during SPSA exploration phases.
    fn adjustment_for_theta(&mut self, target: &[f64; 2]) -> SmoothAdjustments {
        let (target_lr, target_lambda) = self.theta_to_config(target);
        let (last_lr, last_lambda) = self.theta_to_config(&self.last_emitted_theta);
        self.last_emitted_theta = *target;

        // Dampen: blend the raw multiplier toward 1.0 (no change).
        // Factor of 0.3 means we apply only 30% of the suggested change,
        // preventing large LR swings while still allowing gradient-guided drift.
        let raw_mult = target_lr / last_lr.max(1e-15);
        let dampened_mult = 1.0 + 0.3 * (raw_mult - 1.0);

        let raw_dir = target_lambda - last_lambda;
        let dampened_dir = 0.3 * raw_dir;

        SmoothAdjustments {
            lr_multiplier: dampened_mult,
            lambda_direction: dampened_dir,
        }
    }

    /// Compute theta perturbed by `sign * c_k * delta`, clamped to [0, 1].
    fn perturbed_theta(&self, sign: f64) -> [f64; 2] {
        let c_k_base = self.c_init / (self.k_local as f64 + 1.0).powf(0.101);
        let mut result = [0.0; 2];
        for (i, val) in result.iter_mut().enumerate() {
            let c_k = c_k_base.max(self.c_floor[i]);
            *val = (self.theta[i] + sign * c_k * self.current_delta[i]).clamp(0.0, 1.0);
        }
        result
    }

    /// Generate Bernoulli +/-1 perturbation using xorshift64.
    fn generate_delta(&mut self) {
        for d in &mut self.current_delta {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            *d = if self.rng_state % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
}

/// Backward compatibility alias.
pub type DiagnosticAdaptor = DiagnosticLearner;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feasible_region_sparse_data() {
        let region = FeasibleRegion::from_data(100, 3, 1.0);
        let bounds = region.config_bounds();
        assert!(
            bounds.max_depth.1 <= 4,
            "sparse data (n=100) should have tight depth: got max {}",
            bounds.max_depth.1
        );
        assert!(
            bounds.n_steps.1 <= 15,
            "sparse data (n=100) should have tight n_steps: got max {}",
            bounds.n_steps.1
        );
    }

    #[test]
    fn feasible_region_abundant_data() {
        let region = FeasibleRegion::from_data(10_000, 3, 1.0);
        let bounds = region.config_bounds();
        assert!(
            bounds.max_depth.1 >= 4,
            "abundant data (n=10000) should allow deeper trees: got max {}",
            bounds.max_depth.1
        );
        assert!(
            bounds.n_steps.1 >= 20,
            "abundant data (n=10000) should allow more steps: got max {}",
            bounds.n_steps.1
        );
    }

    #[test]
    fn feasible_region_center_config_valid() {
        let region = FeasibleRegion::from_data(500, 5, 2.0);
        let config = region.center_config();
        assert!(config.n_steps > 0, "center n_steps must be > 0");
        assert!(config.max_depth > 0, "center max_depth must be > 0");
        assert!(
            config.learning_rate > 0.0 && config.learning_rate <= 1.0,
            "center learning_rate must be in (0, 1]"
        );
    }

    #[test]
    fn feasible_region_perturbations() {
        let region = FeasibleRegion::from_data(500, 5, 2.0);
        let configs = region.perturbation_configs();
        assert!(
            configs.len() > 1,
            "perturbation_configs should produce > 1 configs, got {}",
            configs.len()
        );
        for (i, cfg) in configs.iter().enumerate() {
            assert!(cfg.n_steps > 0, "config[{i}] n_steps must be > 0");
            assert!(cfg.max_depth > 0, "config[{i}] max_depth must be > 0");
            assert!(
                cfg.learning_rate > 0.0,
                "config[{i}] learning_rate must be > 0"
            );
        }
    }

    #[test]
    fn feasible_region_update_expands() {
        let mut region = FeasibleRegion::from_data(100, 3, 1.0);
        let budget_before = region.budget();
        region.update(10_000);
        assert!(
            region.budget() > budget_before,
            "budget should increase with more data: before={budget_before}, after={}",
            region.budget()
        );
    }

    #[test]
    fn welford_race_all_see_all() {
        let region = FeasibleRegion::from_data(200, 2, 1.0);
        let configs = region.perturbation_configs();
        let n_configs = configs.len();
        let mut race = WelfordRace::new(configs);

        for i in 0..100 {
            let x = i as f64 * 0.1;
            race.feed(&[x, x * 0.5], x * 2.0 + 1.0);
        }

        assert_eq!(race.n_candidates(), n_configs);
        assert_eq!(race.n_samples(), 100);

        let (_winner, results) = race.select_winner();
        for (idx, _mean, _se, n) in &results.all_results {
            assert_eq!(
                *n, 100,
                "config {idx} should have seen 100 samples, got {n}"
            );
        }
    }

    #[test]
    fn welford_race_selects_best() {
        let region = FeasibleRegion::from_data(500, 1, 1.0);
        let configs = region.perturbation_configs();
        let mut race = WelfordRace::new(configs);

        for i in 0..200 {
            let x = i as f64 * 0.01;
            let noise = ((i * 7 + 3) % 11) as f64 * 0.001 - 0.005;
            race.feed(&[x], 2.0 * x + noise);
        }

        let (_winner, results) = race.select_winner();
        let winner_mean = results.winner_mean_error;
        for (_, mean, _, _) in &results.all_results {
            assert!(
                winner_mean <= *mean + 1e-12,
                "winner mean {winner_mean} should be <= all others, found {mean}"
            );
        }
    }

    #[test]
    fn welford_stats_accuracy() {
        let mut stats = WelfordStats::default();
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        for v in &values {
            stats.update(*v);
        }

        let expected_mean = 5.0;
        let expected_variance = 4.571428571428571;
        assert!(
            (stats.mean_error - expected_mean).abs() < 1e-10,
            "mean should be {expected_mean}, got {}",
            stats.mean_error
        );
        assert!(
            (stats.variance() - expected_variance).abs() < 1e-10,
            "variance should be {expected_variance}, got {}",
            stats.variance()
        );
    }

    // =======================================================================
    // TerminateAfter tests (AM-13)
    // =======================================================================

    /// Helper: build a minimal race (2+ configs) for termination tests.
    fn make_race() -> WelfordRace {
        let region = FeasibleRegion::from_data(200, 2, 1.0);
        WelfordRace::new(region.perturbation_configs())
    }

    #[test]
    fn terminate_after_samples_freezes_state() {
        let mut race = make_race().with_termination(TerminateAfter::Samples(500));
        assert!(
            !race.is_terminated(),
            "should not be terminated before any feed"
        );
        assert_eq!(
            race.samples_until_termination(),
            Some(500),
            "should report 500 remaining at start"
        );

        for i in 0..499 {
            let x = i as f64 * 0.001;
            race.feed(&[x, x], x * 2.0);
        }
        assert!(
            !race.is_terminated(),
            "should not terminate after 499 samples (threshold is 500)"
        );
        assert_eq!(
            race.samples_until_termination(),
            Some(1),
            "should report 1 remaining after 499 feeds"
        );

        race.feed(&[1.0, 1.0], 2.0);
        assert!(
            race.is_terminated(),
            "should terminate after exactly 500 samples"
        );
        assert_eq!(
            race.samples_until_termination(),
            Some(0),
            "samples_until_termination should be 0 after termination"
        );

        let n_before = race.n_samples();
        race.feed(&[2.0, 2.0], 4.0);
        race.feed(&[3.0, 3.0], 6.0);
        assert_eq!(
            race.n_samples(),
            n_before,
            "feed after termination must be a no-op: samples should not increase"
        );
        assert!(
            race.is_terminated(),
            "is_terminated must remain true after extra feeds (monotonic)"
        );
    }

    #[test]
    fn terminate_after_corrections_freezes_state() {
        const K: usize = 5;
        let mut race = make_race().with_termination(TerminateAfter::Corrections(K));

        for i in 0..50 {
            let x = i as f64 * 0.01;
            race.feed(&[x, x], x * 2.0);
        }
        assert!(
            !race.is_terminated(),
            "should not terminate before K corrections"
        );

        for _ in 0..(K - 1) {
            race.signal_correction();
        }
        assert!(
            !race.is_terminated(),
            "should not terminate after K-1 corrections (threshold is K)"
        );

        race.signal_correction();
        assert!(
            race.is_terminated(),
            "should terminate on the K-th signal_correction call"
        );

        race.signal_correction();
        assert!(
            race.is_terminated(),
            "is_terminated must remain true after extra signal_correction (monotonic)"
        );
    }

    #[test]
    #[ignore = "Duration-based termination is non-deterministic on real wall-clock time"]
    fn terminate_after_duration_freezes_state() {
        use std::time::Duration;
        let mut race =
            make_race().with_termination(TerminateAfter::Duration(Duration::from_millis(50)));

        race.feed(&[1.0, 1.0], 2.0);
        std::thread::sleep(Duration::from_millis(100));
        race.feed(&[2.0, 2.0], 4.0);
        assert!(
            race.is_terminated(),
            "should terminate after wall-clock duration elapses"
        );

        let n_before = race.n_samples();
        race.feed(&[3.0, 3.0], 6.0);
        assert_eq!(
            race.n_samples(),
            n_before,
            "feed after duration-termination must be a no-op"
        );
    }

    #[test]
    fn terminate_after_never_default_back_compat() {
        let mut race = make_race(); // default: TerminateAfter::Never

        for i in 0..1_000 {
            let x = i as f64 * 0.001;
            race.feed(&[x, x], x * 2.0);
        }
        assert!(
            !race.is_terminated(),
            "TerminateAfter::Never must never terminate, even after 1000 feeds"
        );
        assert_eq!(
            race.samples_until_termination(),
            None,
            "samples_until_termination must be None for TerminateAfter::Never"
        );
    }

    #[test]
    fn is_terminated_is_monotonic() {
        let mut race = make_race().with_termination(TerminateAfter::Samples(10));

        for i in 0..10 {
            race.feed(&[i as f64, i as f64], i as f64 * 2.0);
        }
        assert!(race.is_terminated(), "should be terminated after 10 feeds");

        race.feed(&[99.0, 99.0], 198.0);
        race.signal_correction();
        race.feed(&[100.0, 100.0], 200.0);

        assert!(
            race.is_terminated(),
            "is_terminated must remain true (monotonic invariant violated)"
        );
    }

    #[test]
    fn samples_until_termination_decrements_correctly() {
        let mut race = make_race().with_termination(TerminateAfter::Samples(5));

        assert_eq!(
            race.samples_until_termination(),
            Some(5),
            "remaining should be 5 at start"
        );

        for step in 0..5usize {
            let x = step as f64;
            race.feed(&[x, x], x * 2.0);
            let remaining = race.samples_until_termination();
            let expected = Some((4 - step) as u64);
            assert_eq!(
                remaining,
                expected,
                "after {} feeds, remaining should be {:?}, got {:?}",
                step + 1,
                expected,
                remaining
            );
        }
        assert!(
            race.is_terminated(),
            "race must be terminated after all 5 feeds"
        );
    }

    // =======================================================================
    // DiagnosticLearner tests (SPSA)
    // =======================================================================

    #[test]
    fn diagnostic_learner_init_phase_no_adjustments() {
        // No adjustments during Init phase (first 50 samples).
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let mut learner = DiagnosticLearner::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            regularization_sensitivity: 1.0,
            depth_sufficiency: 0.5,
            effective_dof: 10.0,
            uncertainty: 0.1,
        };

        // First 49 samples: all should be no-op.
        for _ in 0..49 {
            let adj = learner.after_train(&diag, 0.5, 1.0);
            assert_eq!(
                adj.lr_multiplier, 1.0,
                "during init phase, lr_multiplier should be 1.0"
            );
            assert_eq!(
                adj.lambda_direction, 0.0,
                "during init phase, lambda_direction should be 0.0"
            );
        }

        assert_eq!(
            learner.phase(),
            SPSAPhase::Init,
            "should still be in Init phase after 49 samples"
        );
    }

    #[test]
    fn diagnostic_learner_phase_cycling() {
        // Verify Init -> PerturbPlus -> PerturbMinus -> PerturbPlus...
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let bounds = region.config_bounds();
        // Match the cap applied in DiagnosticLearner::with_objective.
        let interval = ((bounds.grace_period.0 + bounds.grace_period.1) / 2).clamp(1, 50) as u64;
        let mut learner = DiagnosticLearner::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            regularization_sensitivity: 1.0,
            depth_sufficiency: 0.5,
            effective_dof: 10.0,
            uncertainty: 0.1,
        };

        // Init phase: 50 samples.
        for i in 0..50 {
            learner.after_train(&diag, i as f64 * 0.01, i as f64 * 0.01 + 0.1);
        }
        // After 50 samples, should transition to PerturbPlus.
        assert_eq!(
            learner.phase(),
            SPSAPhase::PerturbPlus,
            "should be PerturbPlus after init phase"
        );

        // Feed observation_interval samples to complete PerturbPlus.
        for i in 0..interval {
            let idx = 50 + i;
            learner.after_train(&diag, idx as f64 * 0.01, idx as f64 * 0.01 + 0.1);
        }
        assert_eq!(
            learner.phase(),
            SPSAPhase::PerturbMinus,
            "should be PerturbMinus after completing PerturbPlus"
        );

        // Feed observation_interval samples to complete PerturbMinus.
        for i in 0..interval {
            let idx = 50 + interval + i;
            learner.after_train(&diag, idx as f64 * 0.01, idx as f64 * 0.01 + 0.1);
        }
        assert_eq!(
            learner.phase(),
            SPSAPhase::PerturbPlus,
            "should cycle back to PerturbPlus after PerturbMinus"
        );
    }

    #[test]
    fn diagnostic_learner_theta_bounds_clamping() {
        // Theta must stay in [0, 1] even with many SPSA iterations.
        let region = FeasibleRegion::from_data(10_000, 5, 1.0);
        let mut learner = DiagnosticLearner::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.9,
            regularization_sensitivity: 0.1,
            depth_sufficiency: 0.5,
            effective_dof: 10.0,
            uncertainty: 0.1,
        };

        for i in 0..5_000 {
            let pred = i as f64 * 0.01;
            let target = pred + 0.01;
            learner.after_train(&diag, pred, target);
        }

        // Theta must be within [0, 1].
        assert!(
            learner.theta[0] >= 0.0 && learner.theta[0] <= 1.0,
            "theta[0] must be in [0, 1], got {}",
            learner.theta[0]
        );
        assert!(
            learner.theta[1] >= 0.0 && learner.theta[1] <= 1.0,
            "theta[1] must be in [0, 1], got {}",
            learner.theta[1]
        );
    }

    #[test]
    fn diagnostic_learner_backward_compat_alias() {
        // DiagnosticAdaptor alias should work identically to DiagnosticLearner.
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let mut adaptor: DiagnosticAdaptor = DiagnosticAdaptor::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            ..Default::default()
        };

        // Should compile and behave identically.
        let adj = adaptor.after_train(&diag, 0.0, 0.0);
        assert_eq!(
            adj.lr_multiplier, 1.0,
            "backward compat alias: init phase should return no-op"
        );

        // after_train_diagnostics_only should also work via alias.
        let adj2 = adaptor.after_train_diagnostics_only(&diag);
        assert_eq!(
            adj2.lr_multiplier, 1.0,
            "backward compat alias: diagnostics_only should return no-op"
        );
    }

    #[test]
    fn diagnostic_learner_structural_change() {
        let region = FeasibleRegion::from_data(10_000, 3, 10.0);
        let mut learner = DiagnosticLearner::new(region);

        let init_diag = ConfigDiagnostics {
            depth_sufficiency: 0.1,
            effective_dof: 5.0,
            ..Default::default()
        };
        for _ in 0..500 {
            learner.after_train(&init_diag, 0.5, 1.0);
        }

        let mut check_region = learner.region().clone();
        check_region.update(500);
        let bounds = check_region.config_bounds();
        assert!(
            bounds.max_depth.1 > bounds.max_depth.0,
            "region must have depth headroom for this test: bounds={:?}",
            bounds.max_depth
        );

        let high_depth_diag = ConfigDiagnostics {
            depth_sufficiency: 1.0,
            effective_dof: 5.0,
            ..Default::default()
        };
        let change = learner.at_replacement(&high_depth_diag);
        assert!(
            change.is_some(),
            "high depth_sufficiency should trigger structural change"
        );
        let change = change.unwrap();
        assert!(
            change.depth_delta > 0,
            "should suggest increasing depth, got delta={}",
            change.depth_delta
        );
    }

    #[test]
    fn diagnostic_learner_reset_clears_state() {
        let region = FeasibleRegion::from_data(1_000, 3, 1.0);
        let mut learner = DiagnosticLearner::new(region);

        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            ..Default::default()
        };

        for i in 0..200 {
            learner.after_train(&diag, i as f64 * 0.01, i as f64 * 0.01 + 0.1);
        }

        learner.reset();

        // After reset, total_steps should be 0.
        assert_eq!(
            learner.total_steps(),
            0,
            "total_steps should be 0 after reset"
        );

        // After reset, phase should be Init.
        assert_eq!(
            learner.phase(),
            SPSAPhase::Init,
            "phase should be Init after reset"
        );

        // After reset, theta should be [0.5, 0.5].
        assert_eq!(
            learner.theta,
            [0.5, 0.5],
            "theta should be [0.5, 0.5] after reset"
        );

        // After reset, init phase should be active again (no-op adjustments).
        let adj = learner.after_train(&diag, 0.0, 0.0);
        assert_eq!(
            adj.lr_multiplier, 1.0,
            "after reset, first sample should be init-phase no-op, got lr={}",
            adj.lr_multiplier
        );
        assert_eq!(
            adj.lambda_direction, 0.0,
            "after reset, first sample should be init-phase no-op, got lambda={}",
            adj.lambda_direction
        );
    }

    #[test]
    fn diagnostic_learner_meta_objective_default() {
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let learner = DiagnosticLearner::new(region);
        assert!(
            matches!(learner.objective(), MetaObjective::MinimizeRMSE),
            "default objective should be MinimizeRMSE"
        );
    }

    #[test]
    fn diagnostic_learner_with_custom_objective() {
        let region = FeasibleRegion::from_data(200, 3, 1.0);
        let learner = DiagnosticLearner::with_objective(region, MetaObjective::MaximizeF1);
        assert!(
            matches!(learner.objective(), MetaObjective::MaximizeF1),
            "objective should be MaximizeF1"
        );
    }

    // =======================================================================
    // SPSA convergence and bounds tests
    // =======================================================================

    /// Helper: run a learner for `n_calls` post-init samples with constant
    /// diagnostics. Returns the product of all emitted lr_multipliers.
    fn run_learner_total_lr(n_calls: u64, diag: &ConfigDiagnostics) -> f64 {
        let region = FeasibleRegion::from_data(50_000, 5, 1.0);
        let mut learner = DiagnosticLearner::new(region);

        // Burn through init phase.
        for i in 0..50 {
            learner.after_train(diag, i as f64 * 0.01, i as f64 * 0.01 + 0.1);
        }

        let mut total_lr_log = 0.0_f64;
        for i in 0..n_calls {
            let pred = (50 + i) as f64 * 0.01;
            let target = pred + 0.1;
            let adj = learner.after_train(diag, pred, target);
            total_lr_log += adj.lr_multiplier.ln();
        }
        total_lr_log.exp()
    }

    #[test]
    fn spsa_bounded_total_adjustment() {
        let diag = ConfigDiagnostics {
            residual_alignment: 0.5,
            ..Default::default()
        };

        let total_100 = run_learner_total_lr(100, &diag);
        let total_10000 = run_learner_total_lr(10_000, &diag);
        let total_40000 = run_learner_total_lr(40_000, &diag);

        assert!(
            total_100.is_finite(),
            "total LR after 100 calls must be finite, got {total_100}"
        );
        assert!(
            total_10000.is_finite(),
            "total LR after 10000 calls must be finite, got {total_10000}"
        );
        assert!(
            total_40000.is_finite(),
            "total LR after 40000 calls must be finite, got {total_40000}"
        );
    }

    // =======================================================================
    // AM-15 distillation tests
    // =======================================================================

    /// `train_one_weighted` default delegates to `train_one`.  Not feature-gated:
    /// the trait method must compile regardless of the `distill` flag.
    #[test]
    fn train_one_weighted_default_ignores_weight() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let mut race = WelfordRace::new(region.perturbation_configs());
        race.feed(&[1.0], 2.0);
        let n_before = race.n_samples();
        race.feed(&[1.0], 2.0);
        let n_after = race.n_samples();
        assert_eq!(
            n_after,
            n_before + 1,
            "n_samples must increment by 1 per feed regardless of distillation state"
        );
    }

    /// Without `with_distillation`, feeding samples must not alter behavior.
    /// When `distill` feature is ON, `distillation_stats().disabled` must be `true`.
    #[test]
    fn distillation_disabled_by_default() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let mut race = WelfordRace::new(region.perturbation_configs());
        for i in 0..10 {
            race.feed(&[i as f64 * 0.1], i as f64 * 0.2);
        }
        #[cfg(feature = "distill")]
        {
            let stats = race.distillation_stats();
            assert!(
                stats.disabled,
                "distillation_stats().disabled must be true when with_distillation not called"
            );
            assert_eq!(
                stats.n_distillations_triggered, 0,
                "no distillations should have triggered without configuration"
            );
        }
        // Flag OFF: the feed loop + n_samples check IS the test; the absence
        // of a distillation_stats() call is a compile-time guarantee.
        assert_eq!(
            race.n_samples(),
            10,
            "n_samples must equal 10 after 10 feeds"
        );
    }

    #[cfg(feature = "distill")]
    #[test]
    fn distillation_replay_buffer_respects_size_limit() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let cfg = DistillationConfig {
            trigger_after_dominated_samples: 500,
            replay_buffer_size: 5,
            distill_weight: 0.3,
        };
        let mut race = WelfordRace::new(region.perturbation_configs()).with_distillation(cfg);
        for i in 0..20 {
            race.feed(&[i as f64 * 0.1], i as f64 * 0.2);
        }
        let stats = race.distillation_stats();
        assert!(
            !stats.disabled,
            "distillation_stats().disabled must be false after with_distillation"
        );
        assert!(
            stats.candidates_currently_distilling.is_empty(),
            "no candidates should be distilling after 20 samples with trigger=500"
        );
        // Buffer cap: if VecDeque grew unbounded we would OOM; reaching here with
        // 20 feeds and cap=5 is the structural correctness signal.
    }

    #[cfg(feature = "distill")]
    #[test]
    fn distillation_triggers_after_dominated_samples() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let trigger = 3u64;
        let cfg = DistillationConfig {
            trigger_after_dominated_samples: trigger,
            replay_buffer_size: 100,
            distill_weight: 0.3,
        };
        let mut race = WelfordRace::new(region.perturbation_configs()).with_distillation(cfg);
        for i in 0..(trigger + 5) {
            let x = i as f64 * 0.1;
            race.feed(&[x], x * 2.0 + 1.0);
        }
        let stats = race.distillation_stats();
        assert!(
            stats.n_distillations_triggered > 0,
            "distillation must trigger after {} dominated samples; got {} triggers",
            trigger,
            stats.n_distillations_triggered
        );
    }

    #[cfg(feature = "distill")]
    #[test]
    fn distillation_distill_weight_clamped_to_unit_interval() {
        // Weight > 1.0 must be clamped to 1.0.
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let cfg = DistillationConfig {
            trigger_after_dominated_samples: 500,
            replay_buffer_size: 100,
            distill_weight: 5.0,
        };
        let race = WelfordRace::new(region.perturbation_configs()).with_distillation(cfg);
        assert!(
            !race.distillation_stats().disabled,
            "race with clamped weight must be active (disabled=false)"
        );
        // Negative weight must also be clamped to f64::MIN_POSITIVE.
        let region2 = FeasibleRegion::from_data(200, 1, 1.0);
        let cfg2 = DistillationConfig {
            trigger_after_dominated_samples: 500,
            replay_buffer_size: 100,
            distill_weight: -1.0,
        };
        let race2 = WelfordRace::new(region2.perturbation_configs()).with_distillation(cfg2);
        assert!(
            !race2.distillation_stats().disabled,
            "race with negative weight (clamped) must be active"
        );
    }

    // =======================================================================
    // AM-14 Pareto winner selection tests
    // =======================================================================

    /// Single candidate race: always wins the Pareto front and pareto_winner_idx.
    #[test]
    fn pareto_single_candidate_wins() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let configs = vec![region.center_config()];
        let mut race = WelfordRace::new(configs);
        for i in 0..50 {
            let x = i as f64 * 0.01;
            race.feed(&[x], 2.0 * x + 0.1);
        }
        let front = race.pareto_front();
        assert_eq!(
            front.len(),
            1,
            "single-candidate race must have front of size 1, got {}",
            front.len()
        );
        assert_eq!(front[0], 0, "single candidate must be at position 0");
        let winner = race.pareto_winner_idx();
        assert_eq!(
            winner,
            Some(0),
            "single-candidate pareto_winner_idx must be Some(0), got {winner:?}"
        );
    }

    /// No samples fed: pareto_winner_idx returns None (NaN dir_accuracy excludes all).
    #[test]
    fn pareto_no_samples_returns_none() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let race = WelfordRace::new(region.perturbation_configs());
        // No feeds -> all candidates have n=0 -> dir_accuracy() returns NaN -> excluded.
        let winner = race.pareto_winner_idx();
        assert_eq!(
            winner, None,
            "pareto_winner_idx before any feed must be None, got {winner:?}"
        );
    }

    /// Dominated candidate excluded: A wins on all five signals, B is excluded from front.
    #[test]
    fn pareto_dominated_candidate_excluded() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let configs = vec![region.center_config(), region.center_config()];
        let mut race = WelfordRace::new(configs);

        // Candidate 0 (A): low error, tight CI, many samples, high dir_accuracy.
        // Candidate 1 (B): high error, same n, low dir_accuracy.
        // Both get same n to avoid n_steps tie being load-bearing.
        let n = 200u64;
        let m2_a = 0.01 * (n - 1) as f64; // low variance
        let m2_b = 0.50 * (n - 1) as f64; // high variance
        race.inject_stats_for_test(0, n, 0.10, m2_a, 180); // 90% dir correct
        race.inject_stats_for_test(1, n, 0.80, m2_b, 100); // 50% dir correct

        // A dominates B: lower mean_error (0.10<0.80), lower se_error, same empirical_sigma
        // (both 0.0 from non-distributional model), same n_steps, higher dir_accuracy (0.9>0.5).
        let front = race.pareto_front();
        assert!(
            front.contains(&0),
            "candidate 0 (lower error + higher dir_accuracy) must be on front: front={front:?}"
        );
        assert!(
            !front.contains(&1),
            "candidate 1 (dominated) must NOT be on front: front={front:?}"
        );
        let winner = race.pareto_winner_idx();
        assert_eq!(
            winner,
            Some(0),
            "pareto_winner_idx must pick the dominating candidate (0): got {winner:?}"
        );
    }

    /// Non-dominated pair: A better on mean_error, B better on dir_accuracy.
    /// Both survive to front; tiebreak selects A (lower mean_error).
    #[test]
    fn pareto_non_dominated_pair_returns_both_in_front() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let configs = vec![region.center_config(), region.center_config()];
        let mut race = WelfordRace::new(configs);

        let n = 200u64;
        let m2 = 0.10 * (n - 1) as f64;
        // A: lower mean_error but worse dir_accuracy.
        race.inject_stats_for_test(0, n, 0.20, m2, 100); // 50% dir correct
                                                         // B: higher mean_error but better dir_accuracy.
        race.inject_stats_for_test(1, n, 0.50, m2, 180); // 90% dir correct

        // A does not dominate B (B has better dir_accuracy).
        // B does not dominate A (A has lower mean_error and se_error).
        // Both are on the front.
        let front = race.pareto_front();
        assert!(
            front.contains(&0),
            "candidate 0 (lower mean_error) must be on front: front={front:?}"
        );
        assert!(
            front.contains(&1),
            "candidate 1 (better dir_accuracy) must be on front: front={front:?}"
        );
        assert_eq!(
            front.len(),
            2,
            "non-dominated pair must produce front of size 2, got {}",
            front.len()
        );
        // Tiebreak: min(mean_error) picks candidate 0.
        let winner = race.pareto_winner_idx();
        assert_eq!(
            winner,
            Some(0),
            "min(mean_error) tiebreak must select candidate 0 (0.20 < 0.50): got {winner:?}"
        );
    }

    /// When only mean_error varies (all other signals equal), Pareto winner == scalar winner.
    #[test]
    fn pareto_matches_scalar_when_single_metric_varies() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let configs = vec![region.center_config(), region.center_config()];
        let mut race = WelfordRace::new(configs);

        let n = 200u64;
        let m2 = 0.10 * (n - 1) as f64;
        let dir_correct = 150u64; // same for both
                                  // Candidate 0 wins on mean_error; all others equal.
        race.inject_stats_for_test(0, n, 0.15, m2, dir_correct);
        race.inject_stats_for_test(1, n, 0.45, m2, dir_correct);

        let pareto_winner = race.pareto_winner_idx();
        let scalar_winner = race.current_winner_idx_scalar();
        assert_eq!(
            pareto_winner, scalar_winner,
            "when only mean_error differs, Pareto winner must match scalar winner: \
             pareto={pareto_winner:?}, scalar={scalar_winner:?}"
        );
        assert_eq!(
            pareto_winner,
            Some(0),
            "lower mean_error (0.15) must win: got {pareto_winner:?}"
        );
    }

    /// NaN in any signal field excludes that candidate from the front.
    #[test]
    fn pareto_handles_nan_signals() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let configs = vec![region.center_config(), region.center_config()];
        let mut race = WelfordRace::new(configs);

        // Candidate 0: valid stats.
        let n = 200u64;
        let m2 = 0.10 * (n - 1) as f64;
        race.inject_stats_for_test(0, n, 0.20, m2, 150);
        // Candidate 1: n=0 -> dir_accuracy() returns NaN -> excluded from front.
        // (We don't inject; it stays at default n=0.)

        let front = race.pareto_front();
        assert!(
            front.contains(&0),
            "valid candidate must be on front: front={front:?}"
        );
        assert!(
            !front.contains(&1),
            "NaN-signal candidate (n=0 -> NaN dir_accuracy) must be excluded: front={front:?}"
        );

        let winner = race.pareto_winner_idx();
        assert_eq!(
            winner,
            Some(0),
            "valid candidate must win when other has NaN signal: got {winner:?}"
        );
    }

    /// Multi-front with clear Bernstein winner: the statistically certain arm wins.
    #[test]
    fn pareto_invokes_bernstein_tiebreak_for_multi_front() {
        // Construct a non-dominated pair where both are on the Pareto front,
        // then give arm 0 a much lower mean_error with tight CI so Bernstein
        // can declare it the statistical winner.
        //
        // The arm_stats passed to bernstein_compare use range = 4*sigma estimate.
        // For arm 0: mean=0.10, var=0.0001, n=2000 -> range~= 4*0.01 = 0.04.
        //   halfwidth = sqrt(2*0.0001*3.69/2000) + 7*0.04*3.69/(3*1999) ~ 0.00061+0.000172 ~ 0.00078
        //   CI ~ (0.0992, 0.1008)
        // For arm 1: mean=0.50, var=0.0001, n=2000 -> same halfwidth.
        //   CI ~ (0.4992, 0.5008)
        // hi_0 (0.1008) << lo_1 (0.4992) -> Bernstein declares arm 0 the winner.
        //
        // Make arm 1 have better dir_accuracy so both are non-dominated.
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let configs = vec![region.center_config(), region.center_config()];
        let mut race = WelfordRace::new(configs);

        let n = 2000u64;
        let var = 0.0001f64;
        let m2 = var * (n - 1) as f64;
        // Arm 0: lower mean_error, worse dir_accuracy -> non-dominated with arm 1.
        race.inject_stats_for_test(0, n, 0.10, m2, 1000); // 50% dir correct
                                                          // Arm 1: higher mean_error, better dir_accuracy -> non-dominated with arm 0.
        race.inject_stats_for_test(1, n, 0.50, m2, 1800); // 90% dir correct

        let front = race.pareto_front();
        assert_eq!(
            front.len(),
            2,
            "both candidates must be on the front for Bernstein tiebreak to trigger: front={front:?}"
        );

        // With the above stats, Bernstein should select arm 0 (hi_0 < lo_1).
        let winner = race.pareto_winner_idx();
        assert_eq!(
            winner,
            Some(0),
            "Bernstein tiebreak must select arm 0 (hi_ci_0 < lo_ci_1): got {winner:?}"
        );
    }

    // =======================================================================
    // AM-16 race-level drift detection tests (§4.5)
    // =======================================================================

    #[test]
    fn samples_since_change_increments_per_feed() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let mut race = WelfordRace::new(region.perturbation_configs());

        // Before any feeds, counter is 0.
        assert_eq!(
            race.samples_since_last_winner_change(),
            0,
            "before feeds, samples_since_last_winner_change should be 0"
        );

        let mut prev = race.samples_since_last_winner_change();
        let mut any_increment = false;

        for i in 0..50 {
            let x = i as f64 * 0.1;
            race.feed(&[x], x * 2.0 + 1.0);
            let now = race.samples_since_last_winner_change();
            if now > prev {
                any_increment = true;
            }
            prev = now;
        }

        // The counter must have incremented at least once during 50 feeds.
        assert!(
            any_increment,
            "samples_since_last_winner_change must increment per feed during stable regime"
        );
    }

    #[test]
    fn samples_since_change_resets_on_winner_flip() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let mut race = WelfordRace::new(region.perturbation_configs());

        for i in 0..20 {
            let x = i as f64 * 0.1;
            race.feed(&[x], x * 2.0 + 1.0);
        }

        // samples_since_last_winner_change must always be <= n_samples total.
        // A reset (flip) makes it strictly less; stable makes it equal.
        let counter = race.samples_since_last_winner_change();
        let total = race.n_samples();
        assert!(
            counter <= total,
            "samples_since_last_winner_change ({counter}) must be <= n_samples ({total})"
        );
    }

    #[test]
    fn winner_change_count_is_monotonically_non_decreasing() {
        let region = FeasibleRegion::from_data(200, 1, 1.0);
        let mut race = WelfordRace::new(region.perturbation_configs());

        let mut prev_count = race.winner_change_count();
        for i in 0..100 {
            let x = i as f64 * 0.1;
            // Alternate between two target patterns to encourage winner churn.
            let target = if i % 20 < 10 { x * 2.0 + 1.0 } else { -x * 0.5 };
            race.feed(&[x], target);
            let now = race.winner_change_count();
            assert!(
                now >= prev_count,
                "winner_change_count must be monotonically non-decreasing: was {prev_count}, now {now} at step {i}"
            );
            prev_count = now;
        }
    }

    #[test]
    fn race_drift_score_returns_zero_until_buffer_half_full() {
        let region = FeasibleRegion::from_data(5000, 1, 1.0);
        let mut race = WelfordRace::new(region.perturbation_configs());

        // Feed 511 samples — one below DRIFT_WINDOW / 2 = 512.
        for i in 0..511 {
            let x = i as f64 * 0.01;
            race.feed(&[x], x * 2.0 + 1.0);
        }
        // Buffer has < 512 entries — score must be 0.0 per cold-start convention.
        assert_eq!(
            race.race_drift_score(),
            0.0,
            "drift_score must be 0.0 when fewer than 512 errors collected (cold-start convention)"
        );
    }

    #[test]
    fn race_drift_score_positive_when_error_growing() {
        // Strategy: fill the 1024-entry buffer's baseline half with low-error
        // samples, then inject exactly 512 samples where any smooth model fails
        // (rapid sign oscillation at ±1000 magnitude). The recent half of the
        // split contains these enormous errors; the baseline half retains the
        // low-error phase. Score must be > 0.
        let region = FeasibleRegion::from_data(5000, 1, 1.0);
        let mut race = WelfordRace::new(region.perturbation_configs());

        // Phase 1: fill buffer with 1200 low-error linear samples.
        for i in 0..1200 {
            let x = i as f64 * 0.001;
            race.feed(&[x], x * 2.0 + 0.5);
        }

        // Phase 2: alternating ±1000 target at period 2 — no smooth model can
        // fit this, so squared errors stay enormous for all 512 samples.
        for i in 0..512 {
            let sign = if i % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            race.feed(&[i as f64 * 0.001], sign * 1000.0);
        }

        let score = race.race_drift_score();
        assert!(
            score > 0.0,
            "drift_score should be positive when errors are growing (got {score})"
        );
    }

    #[test]
    fn race_drift_score_negative_when_winner_improving() {
        let region = FeasibleRegion::from_data(5000, 1, 1.0);
        let mut race = WelfordRace::new(region.perturbation_configs());

        // Phase 1: noisy signal — large errors fill baseline half.
        for i in 0..700 {
            let x = i as f64 * 0.01;
            let noise = ((i * 31 + 7) % 100) as f64 * 10.0;
            race.feed(&[x], x * 2.0 + noise);
        }
        // Phase 2: clean linear signal — small errors fill recent half.
        for i in 700..1400 {
            let x = i as f64 * 0.01;
            race.feed(&[x], x * 2.0 + 1.0);
        }

        let score = race.race_drift_score();
        assert!(
            score < 0.0,
            "drift_score should be negative when winner is improving (got {score})"
        );
    }
}
