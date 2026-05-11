//! Tournament scheduling: elimination rounds, promotion, bracket management.

use super::AutoTuner;
use crate::automl::budget::BudgetStatus;
use crate::metrics::ewma::EwmaRegressionMetrics;
use tracing::warn;

/// Snapshot of the AutoTuner's current state for diagnostics.
#[derive(Debug, Clone)]
pub struct AutoTunerSnapshot {
    /// Name of the champion's model factory.
    pub champion_factory: String,
    /// Champion's current EWMA metric value.
    pub champion_metric: f64,
    /// Samples the champion has trained on.
    pub champion_samples: u64,

    /// Current tournament candidates.
    pub candidates: Vec<super::racing::CandidateSnapshot>,

    /// Current tournament round (0-indexed).
    pub current_round: usize,
    /// Samples processed in current round.
    pub samples_in_round: u64,
    /// Current adaptive bracket size.
    pub effective_n_initial: usize,

    /// Total tournaments completed.
    pub tournaments_completed: u64,
    /// Total champion promotions.
    pub promotions: u64,
    /// Total samples processed.
    pub total_samples: u64,

    /// Registered factory names.
    pub factory_names: Vec<String>,
}

impl std::fmt::Display for AutoTunerSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== AutoTuner Snapshot ===")?;
        writeln!(
            f,
            "Champion: {} (metric: {:.6}, samples: {})",
            self.champion_factory, self.champion_metric, self.champion_samples
        )?;
        writeln!(
            f,
            "Tournament: round {}, {}/{} candidates, {} samples in round",
            self.current_round,
            self.candidates.len(),
            self.effective_n_initial,
            self.samples_in_round
        )?;
        writeln!(
            f,
            "History: {} tournaments, {} promotions, {} total samples",
            self.tournaments_completed, self.promotions, self.total_samples
        )?;
        if !self.candidates.is_empty() {
            writeln!(f, "Candidates:")?;
            for (i, c) in self.candidates.iter().enumerate() {
                let warmup_tag = if c.in_warmup { " [warmup]" } else { "" };
                writeln!(
                    f,
                    "  [{i}] {} metric={:.6} samples={}{warmup_tag}",
                    c.factory_name, c.metric, c.samples_trained
                )?;
            }
        }
        Ok(())
    }
}

/// Generate a snapshot of the current AutoTuner state.
pub(crate) fn snapshot_impl(tuner: &AutoTuner) -> AutoTunerSnapshot {
    let metric_type = tuner.config.metric;

    let champion_metric = super::racing::get_metric(&tuner.champion_ewma, metric_type);
    let champion_factory = tuner
        .factories
        .get(tuner.champion_factory_idx)
        .map(|f| f.name().to_string())
        .unwrap_or_default();

    let candidates: Vec<super::racing::CandidateSnapshot> = tuner
        .candidates
        .iter()
        .map(|c| {
            let factory_name = tuner
                .factories
                .get(c.factory_idx)
                .map(|f| f.name().to_string())
                .unwrap_or_default();
            let warmup_hint = tuner
                .factories
                .get(c.factory_idx)
                .map(|f| f.warmup_hint())
                .unwrap_or(0);
            super::racing::CandidateSnapshot {
                factory_name,
                metric: super::racing::get_metric(&c.ewma, metric_type),
                samples_trained: c.ewma.n_samples(),
                in_warmup: (c.ewma.n_samples() as usize) < warmup_hint,
            }
        })
        .collect();

    AutoTunerSnapshot {
        champion_factory,
        champion_metric,
        champion_samples: tuner.champion.n_samples_seen(),
        candidates,
        current_round: tuner.current_round,
        samples_in_round: tuner.samples_in_round,
        effective_n_initial: tuner.effective_n_initial,
        tournaments_completed: tuner.tournaments_completed,
        promotions: tuner.promotions,
        total_samples: tuner.total_samples,
        factory_names: tuner
            .factories
            .iter()
            .map(|f| f.name().to_string())
            .collect(),
    }
}

/// Eliminate the bottom half of candidates after a round completes.
///
/// Candidates still in warmup (fewer samples than their factory's
/// `warmup_hint()`) are protected and always survive elimination.
///
/// # AM-4: Budget-gated elimination
///
/// Before the metric-based sort, the budget ledger is recomputed.  Candidates
/// with [`BudgetStatus::Exhausted`] are immediately marked as ineligible and
/// cannot survive unless no non-exhausted candidates exist.  This makes the
/// budget constraint load-bearing-explicit: the tuner KNOWS which arms are
/// over their fair share and cannot silently de-prioritize them.
///
/// # AM-3/AM-4: EWMA continuity invariant
///
/// Candidate EWMAs are NOT reset at round boundaries.  The finalist's
/// accumulated metric must be comparable to the champion's long-run metric;
/// resetting destroys cross-round comparability.
pub(crate) fn eliminate_round(tuner: &mut AutoTuner) {
    if tuner.candidates.is_empty() {
        start_tournament(tuner);
        return;
    }

    // AM-4: Recompute budget status before elimination.
    // This updates BudgetStatus::Exhausted for all over-budget arms.
    tuner.budget_ledger.recompute();

    // Pre-compute warmup hints to avoid borrow issues.
    let warmup_hints: Vec<usize> = tuner
        .candidates
        .iter()
        .map(|c| {
            tuner
                .factories
                .get(c.factory_idx)
                .map(|f| f.warmup_hint())
                .unwrap_or(0)
        })
        .collect();

    // Partition into:
    //   protected (warmup): always survive
    //   exhausted (over budget): immediate elimination unless no alternatives
    //   eligible: ranked by budget-adjusted metric
    let metric_type = tuner.config.metric;
    let mut protected_indices: Vec<usize> = Vec::new();
    let mut eligible: Vec<(usize, f64)> = Vec::new();
    let mut exhausted_indices: Vec<usize> = Vec::new();

    for (i, c) in tuner.candidates.iter().enumerate() {
        if (c.err_count as usize) < warmup_hints[i] {
            // Still in warmup — protected regardless of budget.
            protected_indices.push(i);
        } else if tuner.budget_ledger.status(c.budget_idx) == BudgetStatus::Exhausted {
            // AM-4: Explicitly exhausted — cannot compete for survivor slots.
            exhausted_indices.push(i);
        } else {
            let raw = super::racing::get_metric(&c.ewma, metric_type);
            // AM-4: Use budget-adjusted metric (derived from sample fractions,
            // not an arbitrary constant).  Scale = raw metric so the budget
            // penalty is in the same units.
            let adj = tuner
                .budget_ledger
                .adjusted_metric(c.budget_idx, raw, raw.abs().max(1e-12));
            eligible.push((i, adj));
        }
    }

    // Sort eligible best first (lowest adjusted metric).
    eligible.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // From the eligible + protected pool, keep top half (ceiling division).
    let total_to_keep = tuner.candidates.len().div_ceil(2);
    let eligible_to_keep = total_to_keep.saturating_sub(protected_indices.len());

    // If exhaustion left no eligible candidates, fall back to keeping at least 1
    // from the exhausted pool (we cannot leave the tournament with zero candidates).
    let have_eligible = !eligible.is_empty();
    let eligible_to_keep = eligible_to_keep.max(if have_eligible { 1 } else { 0 });

    let mut keep_set: Vec<usize> = protected_indices;
    keep_set.extend(eligible.iter().take(eligible_to_keep).map(|(i, _)| *i));

    // If still empty (all arms were exhausted and warmup-unprotected), keep the
    // single least-exhausted arm so the tournament can finalize.
    if keep_set.is_empty() && !exhausted_indices.is_empty() {
        keep_set.push(exhausted_indices[0]);
    }

    // Rebuild candidates with only survivors.
    let old = std::mem::take(&mut tuner.candidates);
    tuner.candidates = old
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep_set.contains(i))
        .map(|(_, c)| c)
        .collect();

    tuner.current_round += 1;
    tuner.samples_in_round = 0;

    // If 1 candidate left -> finals.
    if tuner.candidates.len() <= 1 {
        finalize_tournament(tuner);
    }
    // AM-3/AM-4: Do NOT reset EWMAs at round boundaries. Cross-round EWMA
    // continuity is required for stable champion promotion: the finalist's
    // accumulated metric must be comparable to the champion's long-run metric.
    // Resetting here destroys that comparability and destabilizes promotion.
}

/// Compare the lone finalist to the champion and possibly promote.
pub(crate) fn finalize_tournament(tuner: &mut AutoTuner) {
    use crate::bandits::Bandit;

    let mut promoted = false;

    if let Some(finalist) = tuner.candidates.pop() {
        let metric_type = tuner.config.metric;
        let finalist_metric = super::racing::get_metric(&finalist.ewma, metric_type);
        let champion_metric = super::racing::get_metric(&tuner.champion_ewma, metric_type);

        if finalist_metric < champion_metric && finalist.ewma.n_samples() >= 10 {
            // Promote.
            tuner.champion = finalist.model;
            tuner.champion_ewma = finalist.ewma;
            tuner.champion_params = finalist.params;
            tuner.champion_factory_idx = finalist.factory_idx;
            tuner.promotions += 1;
            promoted = true;
        }

        // Update bandit: reward the finalist's factory.
        let reward = tuner
            .normalizer
            .normalize(finalist_metric.min(champion_metric));
        if finalist.factory_idx < tuner.bandit.n_arms() {
            tuner.bandit.update(finalist.factory_idx, reward);
        }
    }

    // Adaptive bracket sizing.
    if promoted {
        // Champion was beaten -- landscape is shifting, explore more.
        tuner.effective_n_initial = (tuner.effective_n_initial * 2).min(tuner.config.max_n_initial);
    } else {
        // Champion held -- it's strong, reduce exploration.
        tuner.effective_n_initial = (tuner.effective_n_initial / 2).max(tuner.config.min_n_initial);
    }

    tuner.candidates.clear();
    tuner.tournaments_completed += 1;
    start_tournament(tuner);
}

/// Spawn a new tournament with `effective_n_initial` candidates.
///
/// AM-4: Resets the budget ledger and assigns each new challenger a fresh
/// budget index so per-arm sample × complexity accounting starts clean.
pub(crate) fn start_tournament(tuner: &mut AutoTuner) {
    use crate::bandits::Bandit;

    let n = tuner.effective_n_initial;
    let n_perturb = n / 2;
    let n_factories = tuner.factories.len();
    let ewma_span = tuner.config.ewma_span;
    let sigma = tuner.config.perturb_sigma;

    // AM-4: Reset ledger at tournament start so per-arm accounting is fresh.
    tuner.budget_ledger.reset();

    let mut candidates = Vec::with_capacity(n);

    // 50%: perturbations of champion params (from champion's factory).
    let champ_idx = tuner.champion_factory_idx;
    for _ in 0..n_perturb {
        let space = tuner.factories[champ_idx].config_space();
        let params = match space.perturb(
            &tuner.champion_params,
            sigma,
            &mut tuner.sampler_rngs[champ_idx],
        ) {
            Ok(p) => p,
            Err(e) => {
                warn!(
                    factory = tuner.factories[champ_idx].name(),
                    error = %e,
                    "search-space perturbation unsatisfiable; skipping this slot"
                );
                continue;
            }
        };
        let complexity = tuner.factories[champ_idx].complexity_hint();
        match tuner.factories[champ_idx].create(&params) {
            Ok(model) => {
                let budget_idx = tuner.budget_ledger.add_arm(complexity);
                candidates.push(super::Challenger {
                    model,
                    ewma: EwmaRegressionMetrics::new(ewma_span),
                    params,
                    factory_idx: champ_idx,
                    err_mean: 0.0,
                    err_m2: 0.0,
                    err_count: 0,
                    budget_idx,
                });
            }
            Err(e) => {
                warn!(
                    factory = tuner.factories[champ_idx].name(),
                    error = %e,
                    "factory rejected perturbed arm config; skipping this slot"
                );
            }
        }
    }

    // 50%: random from factories (bandit-guided selection if multi-factory).
    for _ in n_perturb..n {
        let factory_idx = if n_factories > 1 {
            tuner.bandit.select_arm() % n_factories
        } else {
            0
        };
        let space = tuner.factories[factory_idx].config_space();
        let params = match space.sample(&mut tuner.sampler_rngs[factory_idx]) {
            Ok(p) => p,
            Err(e) => {
                warn!(
                    factory = tuner.factories[factory_idx].name(),
                    error = %e,
                    "search-space sampler unsatisfiable; skipping this slot"
                );
                continue;
            }
        };
        let complexity = tuner.factories[factory_idx].complexity_hint();
        match tuner.factories[factory_idx].create(&params) {
            Ok(model) => {
                let budget_idx = tuner.budget_ledger.add_arm(complexity);
                candidates.push(super::Challenger {
                    model,
                    ewma: EwmaRegressionMetrics::new(ewma_span),
                    params,
                    factory_idx,
                    err_mean: 0.0,
                    err_m2: 0.0,
                    err_count: 0,
                    budget_idx,
                });
            }
            Err(e) => {
                warn!(
                    factory = tuner.factories[factory_idx].name(),
                    error = %e,
                    "factory rejected random arm config; skipping this slot"
                );
            }
        }
    }

    tuner.candidates = candidates;
    tuner.current_round = 0;
    tuner.samples_in_round = 0;
    // AM-3/AM-4: Do NOT reset champion_ewma here. The champion EWMA must
    // remain continuous across tournaments so that promotion comparisons
    // at finalize_tournament() use the champion's true long-run metric.
}
