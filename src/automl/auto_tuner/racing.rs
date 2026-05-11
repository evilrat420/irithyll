//! Statistical racing: early elimination using Welford's online statistics.

use super::AutoTuner;
use crate::automl::AutoMetric;
use crate::metrics::ewma::EwmaRegressionMetrics;

/// Snapshot of a single tournament candidate.
#[derive(Debug, Clone)]
pub struct CandidateSnapshot {
    /// Factory name that produced this candidate.
    pub factory_name: String,
    /// Current EWMA metric value.
    pub metric: f64,
    /// Samples this candidate has trained on.
    pub samples_trained: u64,
    /// Whether this candidate is still in warmup.
    pub in_warmup: bool,
}

/// Extract the relevant metric from an EWMA tracker.
///
/// Free function to avoid borrow checker issues with `&self` + `&self.candidates[i].ewma`.
pub fn get_metric(ewma: &EwmaRegressionMetrics, metric: AutoMetric) -> f64 {
    match metric {
        AutoMetric::MAE => ewma.mae(),
        AutoMetric::MSE => ewma.mse(),
        AutoMetric::RMSE => ewma.rmse(),
    }
}

/// Attempt statistical early elimination of clearly outclassed candidates.
///
/// Uses Welford's online statistics to compute a z-test against the median
/// error.  Candidates significantly worse than the median (z > 2.0) are
/// removed early, saving compute.  Warmup-protected candidates are exempt.
///
/// AM-4: Uses budget-adjusted metric from the tournament ledger instead of
/// the legacy `metric + complexity/n_seen` heuristic.
pub(crate) fn try_early_elimination(tuner: &mut AutoTuner) {
    if tuner.candidates.len() <= 1 {
        return;
    }

    let min_samples_for_test: u64 = 30;

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

    // Compute median of budget-adjusted error across candidates that are
    // PAST warmup and have enough samples.  Candidates still in warmup are
    // excluded from the median to prevent their noisy cold-start metrics
    // from skewing the baseline.
    let mut adjusted_errors: Vec<f64> = tuner
        .candidates
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            c.err_count >= min_samples_for_test && (c.err_count as usize) >= warmup_hints[*i]
        })
        .map(|(_, c)| {
            // AM-4: derive penalty from fair-share accounting, not 1/n heuristic.
            tuner.budget_ledger.adjusted_metric(
                c.budget_idx,
                c.err_mean,
                c.err_mean.abs().max(1e-12),
            )
        })
        .collect();
    if adjusted_errors.len() < 2 {
        return;
    }
    adjusted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_adj_err = adjusted_errors[adjusted_errors.len() / 2];

    let z_threshold = 2.0;

    // Pre-snapshot budget indices to avoid borrow conflict inside filter_map.
    let budget_indices: Vec<usize> = tuner.candidates.iter().map(|c| c.budget_idx).collect();
    let err_means: Vec<f64> = tuner.candidates.iter().map(|c| c.err_mean).collect();
    let err_counts: Vec<u64> = tuner.candidates.iter().map(|c| c.err_count).collect();
    let err_m2s: Vec<f64> = tuner.candidates.iter().map(|c| c.err_m2).collect();

    // Collect indices to remove (avoids borrow issues).
    let remove_indices: Vec<usize> = (0..tuner.candidates.len())
        .filter(|&i| {
            // Protected: not enough data.
            if err_counts[i] < min_samples_for_test {
                return false;
            }
            // Protected: still warming up.
            if (err_counts[i] as usize) < warmup_hints[i] {
                return false;
            }
            // AM-4: budget-adjusted error (fair-share accounting, not 1/n heuristic).
            let adj_err = tuner.budget_ledger.adjusted_metric(
                budget_indices[i],
                err_means[i],
                err_means[i].abs().max(1e-12),
            );
            let variance = if err_counts[i] > 1 {
                err_m2s[i] / (err_counts[i] - 1) as f64
            } else {
                0.0
            };
            let std_err = (variance / err_counts[i] as f64).sqrt();
            if std_err < 1e-12 {
                return false; // avoid division by zero
            }
            let z = (adj_err - median_adj_err) / std_err;
            z > z_threshold
        })
        .collect();

    if !remove_indices.is_empty() {
        let old = std::mem::take(&mut tuner.candidates);
        tuner.candidates = old
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !remove_indices.contains(i))
            .map(|(_, c)| c)
            .collect();
    }

    // If early elimination reduced to 1 or 0, finalize.
    if tuner.candidates.len() <= 1 {
        super::scheduler::finalize_tournament(tuner);
    }
}
