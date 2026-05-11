/// Knowledge distillation for [`WelfordRace`] (AM-15, `distill` feature).
///
/// When a race candidate has been continuously Pareto-dominated for
/// `trigger_after_dominated_samples` samples, it transitions into a
/// *teaching* role: its recent predictions are replayed as pseudo-targets
/// into the current Pareto winner's model using down-weighted training calls.
/// The dominated candidate's own parameters freeze (no direct training); its
/// *learned function* transfers.
///
/// This implements Jono's "lego" framing: pieces of the loser's knowledge
/// compose into the new champion rather than being silently discarded.
///
/// # Feature gate
///
/// This entire module and the `WelfordRace` methods it provides are compiled
/// **only** when the `distill` feature is enabled.  Default: OFF.  Enable only
/// after post-restart telemetry confirms winner flipping between two or more
/// configs that capture genuinely different market regimes.
///
/// # Default values and their justification
///
/// | Field | Default | Justification |
/// |-------|---------|---------------|
/// | `trigger_after_dominated_samples` | 500 | Hetzner §4.4 spec mandate. Chosen as ~10% of a 5000-sample race window — enough sustained dominance to be statistically meaningful, short enough to transfer knowledge before the regime ends. No theoretical derivation exists in the spec; treat as a conservative starting point and adjust via `with_distillation`. |
/// | `replay_buffer_size` | 1000 | Hetzner §4.4 spec mandate. Two seconds of 500 Hz streaming data; enough to capture a full micro-regime. Bounded to prevent unbounded memory growth. Adjust proportionally to your correction cadence. |
/// | `distill_weight` | 0.3 | Hetzner §4.4 spec mandate. Keeps pseudo-target contribution below direct-target contribution (weight < 1.0), preventing the winner from overfitting to stale regimes. No stronger theoretical derivation in the spec. Flag for future calibration via empirical pinball comparison. |
///
/// **Discipline note:** all three defaults are spec-mandated starting points,
/// not auto-derived. They should be validated empirically before permanent
/// deployment.  See `DistillationStats::disabled` for the explicit no-op
/// signal when the flag is off.
use std::collections::VecDeque;

use super::WelfordRace;

// ===========================================================================
// DistillationConfig
// ===========================================================================

/// Configuration for WelfordRace knowledge distillation (AM-15).
///
/// All fields are spec-mandated defaults from Hetzner §4.4.  See module-level
/// doc for per-field justification.
#[derive(Clone, Debug)]
pub struct DistillationConfig {
    /// Samples of sustained Pareto domination before distillation triggers.
    ///
    /// A candidate that re-enters the Pareto front resets this counter.
    /// Default: 500 (Hetzner §4.4).
    pub trigger_after_dominated_samples: u64,

    /// Maximum number of `(features, target)` pairs retained per candidate.
    ///
    /// Oldest entries are evicted FIFO when the buffer reaches capacity.
    /// Default: 1000 (Hetzner §4.4).
    pub replay_buffer_size: usize,

    /// Loss weight applied to distilled (pseudo-target) training calls.
    ///
    /// Must be in (0.0, 1.0]. Values >= 1.0 are clamped to 1.0.
    /// Default: 0.3 (Hetzner §4.4).
    pub distill_weight: f64,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            trigger_after_dominated_samples: 500,
            replay_buffer_size: 1000,
            distill_weight: 0.3,
        }
    }
}

// ===========================================================================
// DistillationStats
// ===========================================================================

/// Observable state of the distillation subsystem.
///
/// When the `distill` feature is OFF this struct is not produced.  When the
/// feature is ON but `with_distillation` has not been called, `disabled` is
/// `true` and all counters are zero — making the off-vs-idle distinction
/// explicit at the API boundary rather than via ambiguous zeros.
#[derive(Debug, Clone)]
pub struct DistillationStats {
    /// `true` when distillation has not been configured (no call to
    /// `with_distillation`).  Distinguishes "disabled" from "idle"
    /// (configured but no distillations triggered yet).
    pub disabled: bool,

    /// Total number of distillation passes that have fired since the race
    /// started.  A single pass replays all buffered samples from one dominated
    /// candidate into the winner.
    pub n_distillations_triggered: u64,

    /// Sample index (from `WelfordRace::n_samples`) at which the most recent
    /// distillation pass fired.  `None` if no distillation has fired yet.
    pub last_distillation_at_samples: Option<u64>,

    /// Indices of candidates currently in the distilling state (i.e.
    /// dominated for >= `trigger_after_dominated_samples` samples).
    pub candidates_currently_distilling: Vec<usize>,
}

// ===========================================================================
// Per-candidate distillation state (private)
// ===========================================================================

/// Distillation state attached to each [`RaceCandidate`] when the feature is on.
pub(super) struct CandidateDistillState {
    /// Bounded ring buffer of recent (features, target) pairs.
    pub replay: VecDeque<(Vec<f64>, f64)>,
    /// Consecutive samples this candidate has been outside the Pareto front.
    pub samples_dominated: u64,
    /// Whether distillation has triggered for this candidate.
    pub is_distilling: bool,
}

impl CandidateDistillState {
    pub fn new() -> Self {
        Self {
            replay: VecDeque::new(),
            samples_dominated: 0,
            is_distilling: false,
        }
    }

    /// Push a sample into the replay buffer, evicting oldest if at capacity.
    pub fn push(&mut self, features: &[f64], target: f64, capacity: usize) {
        if self.replay.len() >= capacity {
            self.replay.pop_front();
        }
        self.replay.push_back((features.to_vec(), target));
    }
}

// ===========================================================================
// WelfordRace distillation API  (AM-15)
// ===========================================================================

impl WelfordRace {
    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// Attach a [`DistillationConfig`] to this race.
    ///
    /// Calling this method allocates per-candidate distillation state and
    /// activates the distillation pass in [`feed`](Self::feed).
    ///
    /// Must be called before feeding any samples. Calling after samples have
    /// been fed is safe but the replay buffers will be empty at that point.
    pub fn with_distillation(mut self, cfg: DistillationConfig) -> Self {
        // Clamp distill_weight to (0, 1] — values outside this range are
        // mathematically unsound (negative: inverts gradient; > 1: amplifies
        // pseudo-targets above real targets).
        let weight = cfg.distill_weight.clamp(f64::MIN_POSITIVE, 1.0);
        self.distill_cfg = Some(DistillationConfig {
            distill_weight: weight,
            ..cfg
        });
        // Allocate per-candidate state for every candidate already present.
        self.distill_state = self
            .candidates
            .iter()
            .map(|_| CandidateDistillState::new())
            .collect();
        // Reset stats to idle (not disabled).
        self.distill_stats = DistillationStats {
            disabled: false,
            n_distillations_triggered: 0,
            last_distillation_at_samples: None,
            candidates_currently_distilling: Vec::new(),
        };
        self
    }

    // -----------------------------------------------------------------------
    // Telemetry
    // -----------------------------------------------------------------------

    /// Return the current distillation statistics.
    ///
    /// When distillation has not been configured (`with_distillation` not
    /// called), `disabled: true` is set explicitly so callers can distinguish
    /// "disabled" from "configured but no distillations yet".
    pub fn distillation_stats(&self) -> DistillationStats {
        self.distill_stats.clone()
    }

    // -----------------------------------------------------------------------
    // Internal: called inside WelfordRace::feed (guarded at call site)
    // -----------------------------------------------------------------------

    /// Run the distillation pass for the current feed step.
    ///
    /// Called from `WelfordRace::feed` **after** normal predict-error-train for
    /// all candidates. The `pareto_front` argument is the set of indices that
    /// are currently non-dominated (computed by the caller so the front is
    /// consistent with this feed step).
    ///
    /// Steps:
    /// 1. Update `samples_dominated` / `is_distilling` for every candidate.
    /// 2. Replay dominated-and-distilling candidates' buffers into the winner.
    /// 3. Push the current (features, target) into every candidate's buffer.
    pub(super) fn run_distillation_pass(
        &mut self,
        features: &[f64],
        target: f64,
        pareto_front: &[usize],
        winner_idx: usize,
    ) {
        let cfg = match &self.distill_cfg {
            Some(c) => c.clone(),
            None => return,
        };

        let n = self.candidates.len();

        // Ensure state vec is in sync (defensive: handles candidates added
        // after `with_distillation` via `from_factory` edge cases).
        while self.distill_state.len() < n {
            self.distill_state.push(CandidateDistillState::new());
        }

        // Step 1: Update domination counters.
        for i in 0..n {
            if pareto_front.contains(&i) {
                self.distill_state[i].samples_dominated = 0;
                self.distill_state[i].is_distilling = false;
            } else {
                self.distill_state[i].samples_dominated += 1;
                if self.distill_state[i].samples_dominated >= cfg.trigger_after_dominated_samples {
                    self.distill_state[i].is_distilling = true;
                }
            }
        }

        // Step 2: Replay distilling candidates' predictions into the winner.
        for i in 0..n {
            if i == winner_idx || !self.distill_state[i].is_distilling {
                continue;
            }
            // Collect replay pairs so we can borrow candidates mutably below.
            let replay_pairs: Vec<(Vec<f64>, f64)> =
                self.distill_state[i].replay.iter().cloned().collect();

            if replay_pairs.is_empty() {
                continue;
            }

            for (feat, _orig_target) in &replay_pairs {
                // Pseudo-target = dominated candidate's current prediction.
                // This transfers the candidate's *learned function*, not just
                // the raw targets it trained on (the "lego" framing).
                let pseudo_target = self.candidates[i].model.predict(feat);
                self.candidates[winner_idx].model.train_one_weighted(
                    feat,
                    pseudo_target,
                    cfg.distill_weight,
                );
            }

            // Update global stats.
            let n_samples_now = self.candidates.first().map(|c| c.stats.n).unwrap_or(0);
            self.distill_stats.n_distillations_triggered += 1;
            self.distill_stats.last_distillation_at_samples = Some(n_samples_now);
        }

        // Step 3: Push current sample into all replay buffers.
        for i in 0..n {
            self.distill_state[i].push(features, target, cfg.replay_buffer_size);
        }

        // Step 4: Refresh the "currently distilling" index list.
        self.distill_stats.candidates_currently_distilling = (0..n)
            .filter(|&i| self.distill_state[i].is_distilling)
            .collect();
    }
}
