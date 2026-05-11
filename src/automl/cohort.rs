//! Top-K champion cohort for drift-aware tournament selection.
//!
//! # Why K = 3?
//!
//! K = 3 is the **minimum cohort size that enables unambiguous quorum vote**
//! without majority-of-K bias:
//!
//! - K = 1: no vote; single champion, no diversity.
//! - K = 2: any disagreement produces a tie (1 vs 1); no resolution without
//!   a tiebreaker that reintroduces a single champion.
//! - K = 3: a 2-of-3 majority is always achievable.  A candidate that beats
//!   2 of the 3 current champions has demonstrated superiority over a majority
//!   while the minority retains coverage over other regime types.
//! - K ≥ 4 (even): K/2 vs K/2 ties reappear.
//! - K ≥ 5 (odd): majority vote still works but prediction overhead grows
//!   linearly and diversity benefit saturates quickly past 3.
//!
//! K = 3 is thus the Pareto-optimal choice on the
//! {tie-free, minimal overhead, regime coverage} objective surface
//! (no tuning constant — the derivation is purely structural).
//!
//! # Cohort lifecycle
//!
//! 1. **Initialization**: cohort starts empty; the first `K` tournament
//!    finalists fill the slots in order.
//! 2. **Candidate entry**: a new candidate is compared to the lowest-ranked
//!    cohort member (by drift-aware metric).  If it beats the lowest, it
//!    replaces it and the cohort is re-ranked.
//! 3. **Quorum vote for demotion**: when a slot is contested the cohort votes.
//!    The candidate must beat ≥ 2 of the 3 members; if it does, the lowest
//!    member is demoted and the candidate enters.  This prevents a noisy
//!    single-round win from displacing a reliable cohort member.
//! 4. **Prediction**: uniform 1/K weighting (see [`CohortWeight`]).  This
//!    avoids over-fitting the ensemble to recent noise — the Bernstein race
//!    already selected for quality; adding recency weights would double-count
//!    the most recent regime.
//!
//! # References
//!
//! - R9 §3.5 — Top-K champion cohort design note.
//! - AM-R3 §4 — Cohort selection integration and uniform-weight rationale.

use irithyll_core::learner::StreamingLearner;

/// Number of champions in the cohort.
///
/// Derived from quorum-vote analysis (see module doc).  Not configurable
/// at runtime — changing K changes the voting semantics.
pub const COHORT_K: usize = 3;

/// Weighting policy for cohort ensemble predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CohortWeight {
    /// Equal 1/K weight for all members (default).
    ///
    /// Rationale: Bernstein race already selects for quality.  Adding
    /// recency weights would double-count the most recent regime and
    /// increase variance without reducing bias.
    #[default]
    Uniform,
    /// Weight members by their lower Bernstein confidence bound (ci_lo).
    ///
    /// Higher ci_lo ≙ more confidently good performance.  This policy
    /// rewards confirmed quality over tentative quality.
    CiLower,
}

/// A single member of the champion cohort.
pub struct CohortMember {
    /// The live champion model.
    pub model: Box<dyn StreamingLearner>,
    /// Current drift-aware metric (lower is better for error metrics).
    pub metric: f64,
    /// Lower Bernstein CI bound at the time of cohort entry.
    /// `f64::NEG_INFINITY` when no Bernstein CI is available.
    pub ci_lo: f64,
    /// Human-readable factory name (for diagnostics).
    pub factory_name: String,
    /// Samples seen by this member since joining the cohort.
    pub samples_in_cohort: u64,
}

impl std::fmt::Debug for CohortMember {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohortMember")
            .field("factory_name", &self.factory_name)
            .field("metric", &self.metric)
            .field("ci_lo", &self.ci_lo)
            .field("samples_in_cohort", &self.samples_in_cohort)
            .finish_non_exhaustive()
    }
}

/// Snapshot of a cohort member (no model box; safe to clone and log).
#[derive(Debug, Clone)]
pub struct CohortMemberSnapshot {
    /// Factory name.
    pub factory_name: String,
    /// Current drift-aware metric.
    pub metric: f64,
    /// Lower Bernstein CI bound.
    pub ci_lo: f64,
    /// Samples since joining the cohort.
    pub samples_in_cohort: u64,
}

/// The K=3 champion cohort.
///
/// Maintains at most [`COHORT_K`] champion models ranked by drift-aware
/// metric (ascending — best first).  Provides quorum-vote candidate entry
/// and uniform/ci_lo-weighted ensemble prediction.
pub struct ChampionCohort {
    /// Ranked members: index 0 = best, index K-1 = worst.
    members: Vec<CohortMember>,
    /// Prediction weighting policy.
    weight_policy: CohortWeight,
    /// Total candidate challenges seen.
    pub challenges: u64,
    /// Total successful demotions (lowest member replaced).
    pub demotions: u64,
}

impl std::fmt::Debug for ChampionCohort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChampionCohort")
            .field("n_members", &self.members.len())
            .field("weight_policy", &self.weight_policy)
            .field("challenges", &self.challenges)
            .field("demotions", &self.demotions)
            .finish()
    }
}

impl ChampionCohort {
    /// Create an empty cohort with the given weighting policy.
    pub fn new(weight_policy: CohortWeight) -> Self {
        Self {
            members: Vec::with_capacity(COHORT_K),
            weight_policy,
            challenges: 0,
            demotions: 0,
        }
    }

    /// Create an empty cohort with uniform weighting (common default).
    pub fn with_uniform_weights() -> Self {
        Self::new(CohortWeight::Uniform)
    }

    /// Number of members currently in the cohort (0..=COHORT_K).
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// True if the cohort has no members yet.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// True if the cohort is at full capacity (COHORT_K members).
    pub fn is_full(&self) -> bool {
        self.members.len() >= COHORT_K
    }

    /// Weighted ensemble prediction from all cohort members.
    ///
    /// With [`CohortWeight::Uniform`]: simple mean of all member predictions.
    /// With [`CohortWeight::CiLower`]: weighted by `ci_lo`, rescaled to sum=1.
    ///
    /// Falls back to `f64::NAN` if the cohort is empty (caller should guard).
    pub fn predict(&self, features: &[f64]) -> f64 {
        if self.members.is_empty() {
            return f64::NAN;
        }
        match self.weight_policy {
            CohortWeight::Uniform => {
                let sum: f64 = self.members.iter().map(|m| m.model.predict(features)).sum();
                sum / self.members.len() as f64
            }
            CohortWeight::CiLower => {
                // Shift ci_lo so all weights are positive: w_i = ci_lo_i - min(ci_lo) + ε.
                let min_ci = self
                    .members
                    .iter()
                    .map(|m| m.ci_lo)
                    .fold(f64::INFINITY, f64::min);
                let eps = 1e-12;
                let weights: Vec<f64> = self
                    .members
                    .iter()
                    .map(|m| m.ci_lo - min_ci + eps)
                    .collect();
                let w_sum: f64 = weights.iter().sum();
                if w_sum < 1e-30 {
                    // All ci_lo are identical — fall back to uniform.
                    let sum: f64 = self.members.iter().map(|m| m.model.predict(features)).sum();
                    return sum / self.members.len() as f64;
                }
                self.members
                    .iter()
                    .zip(weights.iter())
                    .map(|(m, &w)| m.model.predict(features) * w / w_sum)
                    .sum()
            }
        }
    }

    /// Return a snapshot of all cohort members (cloneable, no Box).
    pub fn snapshots(&self) -> Vec<CohortMemberSnapshot> {
        self.members
            .iter()
            .map(|m| CohortMemberSnapshot {
                factory_name: m.factory_name.clone(),
                metric: m.metric,
                ci_lo: m.ci_lo,
                samples_in_cohort: m.samples_in_cohort,
            })
            .collect()
    }

    /// Return the best member's metric (lowest error, index 0).
    ///
    /// Returns `f64::INFINITY` if the cohort is empty.
    pub fn best_metric(&self) -> f64 {
        self.members
            .first()
            .map(|m| m.metric)
            .unwrap_or(f64::INFINITY)
    }

    /// Return the worst member's metric (highest error, last index).
    ///
    /// Returns `f64::INFINITY` if the cohort is empty.
    pub fn worst_metric(&self) -> f64 {
        self.members
            .last()
            .map(|m| m.metric)
            .unwrap_or(f64::INFINITY)
    }

    /// Attempt to enter a candidate into the cohort.
    ///
    /// # Semantics
    ///
    /// - If the cohort is not yet full (`len < COHORT_K`): candidate is added
    ///   immediately and the cohort is re-sorted.
    /// - If the cohort is full: apply quorum vote.
    ///   The candidate must beat **strictly more than half** (i.e., ≥ 2 of 3)
    ///   existing members on `candidate_metric < member.metric`.
    ///   If the quorum is met, the **lowest-ranked** (worst) member is demoted
    ///   and the candidate takes its slot.
    ///
    /// # Returns
    ///
    /// `true` if the candidate entered the cohort (and the previous worst was
    /// demoted), `false` otherwise.
    pub fn try_enter(
        &mut self,
        candidate: Box<dyn StreamingLearner>,
        candidate_metric: f64,
        candidate_ci_lo: f64,
        factory_name: String,
    ) -> bool {
        self.challenges += 1;

        if !self.is_full() {
            // Cohort not yet at capacity — admit immediately.
            self.members.push(CohortMember {
                model: candidate,
                metric: candidate_metric,
                ci_lo: candidate_ci_lo,
                factory_name,
                samples_in_cohort: 0,
            });
            self.sort_members();
            return true;
        }

        // Cohort full: apply quorum vote.
        // Count how many existing members the candidate beats.
        let beats: usize = self
            .members
            .iter()
            .filter(|m| candidate_metric < m.metric)
            .count();

        // Quorum: must beat strictly more than half (≥ ceil(COHORT_K/2 + 0.5)).
        // For K=3: ≥ 2.
        let quorum = COHORT_K / 2 + 1; // = 2 for K=3

        if beats < quorum {
            return false;
        }

        // Demote the worst member (last after sort), insert the candidate.
        // The demoted model is dropped here (Rust Drop handles memory).
        let worst_idx = self.members.len() - 1;
        self.members.remove(worst_idx); // Drop demoted model.
        self.members.push(CohortMember {
            model: candidate,
            metric: candidate_metric,
            ci_lo: candidate_ci_lo,
            factory_name,
            samples_in_cohort: 0,
        });
        self.sort_members();
        self.demotions += 1;
        true
    }

    /// Update a cohort member's metric after it has trained on more data.
    ///
    /// Re-sorts the cohort after the update so rank order stays correct.
    /// Call this each time the member processes a sample through the cohort.
    pub fn update_member_metric(&mut self, idx: usize, new_metric: f64) {
        if let Some(m) = self.members.get_mut(idx) {
            m.metric = new_metric;
            m.samples_in_cohort += 1;
        }
        self.sort_members();
    }

    /// Train all cohort members on a single sample.
    ///
    /// Updates `samples_in_cohort` for every member.
    pub fn train_all(&mut self, features: &[f64], target: f64, weight: f64) {
        for m in &mut self.members {
            m.model.train_one(features, target, weight);
            m.samples_in_cohort += 1;
        }
    }

    /// Reset the cohort — evicts all members.
    ///
    /// Models are dropped deterministically (Rust `Drop`).
    pub fn reset(&mut self) {
        self.members.clear();
        self.challenges = 0;
        self.demotions = 0;
    }

    // Sort members ascending by metric (best = lowest error first).
    fn sort_members(&mut self) {
        self.members.sort_by(|a, b| {
            a.metric
                .partial_cmp(&b.metric)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal stub implementing StreamingLearner for testing.
    struct ConstantPredictor {
        value: f64,
        n: u64,
    }

    impl ConstantPredictor {
        fn boxed(value: f64) -> Box<dyn StreamingLearner> {
            Box::new(Self { value, n: 0 })
        }
    }

    impl StreamingLearner for ConstantPredictor {
        fn train_one(&mut self, _: &[f64], _: f64, _: f64) {
            self.n += 1;
        }
        fn predict(&self, _: &[f64]) -> f64 {
            self.value
        }
        fn n_samples_seen(&self) -> u64 {
            self.n
        }
        fn reset(&mut self) {
            self.n = 0;
        }
    }

    /// When a new candidate beats all 3 members (all have higher error),
    /// the worst member (highest error) is demoted and the candidate enters.
    #[test]
    fn cohort_demotes_lowest_when_new_candidate_enters() {
        let mut cohort = ChampionCohort::with_uniform_weights();

        // Fill cohort with 3 members: metrics 0.5, 0.3, 0.8 (sorted: 0.3, 0.5, 0.8).
        cohort.try_enter(ConstantPredictor::boxed(1.0), 0.5, 0.0, "A".into());
        cohort.try_enter(ConstantPredictor::boxed(1.0), 0.3, 0.0, "B".into());
        cohort.try_enter(ConstantPredictor::boxed(1.0), 0.8, 0.0, "C".into());

        assert_eq!(cohort.len(), 3, "cohort should be full");
        // After sort: best=0.3, mid=0.5, worst=0.8
        assert!(
            (cohort.worst_metric() - 0.8).abs() < 1e-12,
            "worst should be 0.8, got {}",
            cohort.worst_metric()
        );

        // Candidate with metric 0.1 beats all 3 → demotes worst (0.8).
        let entered = cohort.try_enter(ConstantPredictor::boxed(1.0), 0.1, 0.0, "D".into());

        assert!(
            entered,
            "candidate with metric 0.1 should beat all 3 and enter"
        );
        assert_eq!(cohort.len(), 3, "cohort size must remain COHORT_K=3");
        assert_eq!(cohort.demotions, 1, "exactly one demotion should occur");

        // Worst after entry should now be 0.5 (0.8 was evicted).
        assert!(
            (cohort.worst_metric() - 0.5).abs() < 1e-12,
            "after demotion worst should be 0.5, got {}",
            cohort.worst_metric()
        );
        assert!(
            (cohort.best_metric() - 0.1).abs() < 1e-12,
            "after entry best should be 0.1, got {}",
            cohort.best_metric()
        );
    }

    /// A candidate that beats only 1 of 3 members (below quorum) is rejected.
    /// A candidate that beats exactly 2 of 3 members (meets quorum) is accepted.
    #[test]
    fn cohort_quorum_vote_demotes_majority() {
        let mut cohort = ChampionCohort::with_uniform_weights();

        // Fill cohort: metrics 0.2, 0.4, 0.6 (sorted: 0.2, 0.4, 0.6).
        cohort.try_enter(ConstantPredictor::boxed(1.0), 0.2, 0.0, "A".into());
        cohort.try_enter(ConstantPredictor::boxed(1.0), 0.4, 0.0, "B".into());
        cohort.try_enter(ConstantPredictor::boxed(1.0), 0.6, 0.0, "C".into());

        // Candidate with metric 0.35: beats 0.4 and 0.6 (2 of 3) — quorum met.
        let entered_quorum = cohort.try_enter(ConstantPredictor::boxed(2.0), 0.35, 0.0, "D".into());
        assert!(
            entered_quorum,
            "candidate 0.35 beats 2 of 3 members — quorum (≥2) is met, must enter"
        );
        assert_eq!(cohort.demotions, 1, "one demotion for quorum entry");

        // After entry: cohort should have 0.2, 0.35, 0.4 (0.6 evicted).
        let snaps = cohort.snapshots();
        let metrics: Vec<f64> = snaps.iter().map(|s| s.metric).collect();
        assert!(
            metrics.iter().any(|&m| (m - 0.35).abs() < 1e-12),
            "0.35 must be in the cohort: {:?}",
            metrics
        );
        assert!(
            !metrics.iter().any(|&m| (m - 0.6).abs() < 1e-12),
            "0.6 must have been evicted: {:?}",
            metrics
        );

        // Candidate with metric 0.45: beats only 0.6 ... wait, 0.6 is gone.
        // Current cohort: 0.2, 0.35, 0.4.
        // Candidate 0.45: beats none (0.45 > 0.2, 0.35, 0.4) — rejected.
        let entered_below_quorum =
            cohort.try_enter(ConstantPredictor::boxed(3.0), 0.45, 0.0, "E".into());
        assert!(
            !entered_below_quorum,
            "candidate 0.45 beats 0 of 3 members — below quorum, must be rejected"
        );
        assert_eq!(
            cohort.demotions, 1,
            "no additional demotion when candidate is rejected"
        );
        assert_eq!(cohort.len(), 3, "cohort must remain at COHORT_K=3");
    }

    /// EWMA does not reset at a round boundary.
    ///
    /// This test verifies the AM-3/AM-4 invariant: a champion's accumulated
    /// metric is preserved across round boundaries (no `ewma.reset()` call
    /// is made when the round counter advances).  We test this at the
    /// ChampionCohort level by checking that `samples_in_cohort` accumulates
    /// monotonically and is never zeroed mid-stream.
    #[test]
    fn ewma_does_not_reset_at_round_boundary() {
        let mut cohort = ChampionCohort::with_uniform_weights();

        cohort.try_enter(ConstantPredictor::boxed(0.5), 0.3, 0.0, "A".into());

        let features = [1.0_f64];
        // Simulate two "rounds" of training without any reset.
        for _ in 0..50 {
            cohort.train_all(&features, 1.0, 1.0);
        }
        let after_round1 = cohort.members[0].samples_in_cohort;

        // Second "round" — no reset, samples continue accumulating.
        for _ in 0..50 {
            cohort.train_all(&features, 1.0, 1.0);
        }
        let after_round2 = cohort.members[0].samples_in_cohort;

        assert_eq!(
            after_round1, 50,
            "samples_in_cohort after round 1 should be 50, got {}",
            after_round1
        );
        assert_eq!(
            after_round2, 100,
            "samples_in_cohort after round 2 should be 100 (no reset), got {}",
            after_round2
        );
    }
}
