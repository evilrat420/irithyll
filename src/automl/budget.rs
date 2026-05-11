//! Samples × complexity budget accounting for AutoML arm pruning.
//!
//! # Design rationale
//!
//! The legacy `adjusted_metric + complexity/n_seen` heuristic implicitly
//! penalizes complex arms but has no load-bearing budget concept: it uses
//! a continuous additive penalty that never actually terminates an arm.
//!
//! This module tracks the *samples invested per arm weighted by complexity*
//! and provides an **explicit** [`BudgetStatus`] enum.  An arm whose
//! weighted-sample share exceeds its fair portion of the total budget is
//! `BudgetStatus::Exhausted`; the scheduler must KNOW that explicitly and
//! cannot silently de-prioritize it (Jono discipline 2: no band-aids).
//!
//! # Fair-share formula
//!
//! For an arm with complexity `c_i` (param count proxy) that has seen
//! `n_i` samples, define its *cost* as `cost_i = n_i × c_i`.
//!
//! The *total cost* is `T = Σ cost_i`.
//!
//! The *fair-share cost* for arm i is `T × (c_i / Σ c_j)`.
//!
//! An arm is over-budget iff `cost_i > T × (c_i / Σ c_j)` which
//! simplifies to `n_i / (Σ n_j) > 1 / k` for k equal-complexity arms,
//! and proportionally adjusts for heterogeneous complexity.
//!
//! In other words: arm i is exhausted when its *sample fraction* exceeds
//! its *complexity fraction*.  Complex arms get fewer samples before
//! exhaustion; simple arms get more.  This is the natural information-
//! theoretic tradeoff: a complex arm costs more per sample to compare
//! fairly, so fewer samples suffice for a confident comparison.
//!
//! The formula requires no tuning constants; it derives entirely from
//! observed complexity hints and sample counts.

/// Whether an arm has remaining budget or has exhausted its fair share.
///
/// This is load-bearing-explicit state (not an implicit signal derived
/// from a soft penalty).  The scheduler must check this before evaluating
/// an arm and treat `Exhausted` as a hard gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStatus {
    /// The arm has remaining budget relative to its complexity share.
    Active,
    /// The arm has consumed more than its fair share of the total
    /// samples × complexity budget.  It must not receive further
    /// evaluation samples until the budget is rebalanced (new tournament).
    Exhausted,
}

/// Per-arm budget accounting entry.
///
/// Stored in the [`BudgetLedger`]; one entry per active arm.
#[derive(Debug, Clone)]
pub struct ArmBudget {
    /// Approximate parameter count (complexity proxy) from the factory.
    pub complexity: usize,
    /// Samples this arm has received.
    pub samples: u64,
    /// Explicit budget status — the load-bearing field.
    pub status: BudgetStatus,
}

impl ArmBudget {
    /// Create a new arm budget entry.
    pub fn new(complexity: usize) -> Self {
        Self {
            complexity,
            samples: 0,
            status: BudgetStatus::Active,
        }
    }
}

/// Ledger tracking the samples × complexity budget across all active arms.
///
/// Call [`BudgetLedger::record_sample`] each time an arm processes a sample,
/// then call [`BudgetLedger::recompute`] after each round boundary to
/// update `BudgetStatus` for all arms.
#[derive(Debug, Default, Clone)]
pub struct BudgetLedger {
    entries: Vec<ArmBudget>,
}

impl BudgetLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new arm with the given complexity.
    ///
    /// Returns the index assigned to this arm (used for all subsequent calls).
    pub fn add_arm(&mut self, complexity: usize) -> usize {
        let idx = self.entries.len();
        self.entries.push(ArmBudget::new(complexity));
        idx
    }

    /// Record that arm `idx` processed one sample.
    pub fn record_sample(&mut self, idx: usize) {
        if let Some(entry) = self.entries.get_mut(idx) {
            entry.samples += 1;
        }
    }

    /// Recompute [`BudgetStatus`] for all arms based on current sample counts.
    ///
    /// An arm is [`BudgetStatus::Exhausted`] when its weighted sample fraction
    /// exceeds its complexity fraction:
    ///
    /// ```text
    /// exhausted_i  ⟺  n_i / Σ n_j  >  c_i / Σ c_j
    /// ```
    ///
    /// Equivalently: `n_i × Σ c_j > c_i × Σ n_j`.
    ///
    /// This form uses only integer arithmetic (no division), avoiding
    /// floating-point precision issues when sample counts are small.
    ///
    /// Arms with zero total samples remain `Active` (no data yet to judge).
    pub fn recompute(&mut self) {
        let total_samples: u64 = self.entries.iter().map(|e| e.samples).sum();
        let total_complexity: u64 = self.entries.iter().map(|e| e.complexity as u64).sum();

        if total_samples == 0 || total_complexity == 0 {
            // No data yet — everyone is active.
            for e in &mut self.entries {
                e.status = BudgetStatus::Active;
            }
            return;
        }

        for entry in &mut self.entries {
            // exhausted iff n_i × Σ c_j > c_i × Σ n_j
            let lhs = entry.samples * total_complexity;
            let rhs = (entry.complexity as u64) * total_samples;
            entry.status = if lhs > rhs {
                BudgetStatus::Exhausted
            } else {
                BudgetStatus::Active
            };
        }
    }

    /// Return the budget status for arm `idx`.
    pub fn status(&self, idx: usize) -> BudgetStatus {
        self.entries
            .get(idx)
            .map(|e| e.status)
            .unwrap_or(BudgetStatus::Active)
    }

    /// Return an immutable reference to the arm's budget entry.
    pub fn arm(&self, idx: usize) -> Option<&ArmBudget> {
        self.entries.get(idx)
    }

    /// Number of arms currently tracked.
    pub fn n_arms(&self) -> usize {
        self.entries.len()
    }

    /// Reset the ledger (called at tournament start).
    pub fn reset(&mut self) {
        self.entries.clear();
    }

    /// Return the total samples × complexity cost in the ledger.
    ///
    /// Useful for diagnostics; reflects actual usage, not predicted.
    pub fn total_cost(&self) -> u64 {
        self.entries
            .iter()
            .map(|e| e.samples * e.complexity as u64)
            .sum()
    }

    /// Adjusted metric for arm `idx` using the budget-normalized penalty.
    ///
    /// This replaces the legacy `metric + complexity/n_seen` heuristic with a
    /// term grounded in the fair-share accounting.  The penalty is proportional
    /// to how far the arm is *over budget* (positive) or *under budget*
    /// (negative, i.e., a bonus for under-evaluated arms).
    ///
    /// ```text
    /// penalty_i = (n_i / Σ n_j) - (c_i / Σ c_j)
    /// ```
    ///
    /// When the arm is exactly at its fair share the penalty is zero.
    /// Over-budget arms get a positive penalty (pushed up in error ranking).
    ///
    /// The penalty is scaled by `scale` to keep it in the same units as the
    /// metric.  The natural scale is the metric itself: `scale = base_metric`
    /// so that a fully-exhausted arm (penalty = 1.0 in normalized form) would
    /// have its metric doubled.  Callers may pass any scale.
    ///
    /// Returns `base_metric` unchanged if there is no data yet.
    pub fn adjusted_metric(&self, idx: usize, base_metric: f64, scale: f64) -> f64 {
        let total_samples: u64 = self.entries.iter().map(|e| e.samples).sum();
        let total_complexity: u64 = self.entries.iter().map(|e| e.complexity as u64).sum();

        if total_samples == 0 || total_complexity == 0 {
            return base_metric;
        }

        let Some(entry) = self.entries.get(idx) else {
            return base_metric;
        };

        let sample_frac = entry.samples as f64 / total_samples as f64;
        let complexity_frac = entry.complexity as f64 / total_complexity as f64;
        let penalty = sample_frac - complexity_frac;

        base_metric + scale * penalty
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// An arm that has received proportionally more samples than its complexity
    /// share deserves is marked Exhausted after recompute().
    #[test]
    fn budget_exhausts_arm_when_overrun() {
        let mut ledger = BudgetLedger::new();
        let simple = ledger.add_arm(10); // low complexity
        let complex = ledger.add_arm(90); // high complexity

        // Give the simple arm many samples but the complex arm very few.
        // Simple arm sample fraction = 90/100 = 0.90
        // Simple arm complexity fraction = 10/100 = 0.10
        // 0.90 > 0.10 → Exhausted
        for _ in 0..90 {
            ledger.record_sample(simple);
        }
        for _ in 0..10 {
            ledger.record_sample(complex);
        }

        ledger.recompute();

        assert_eq!(
            ledger.status(simple),
            BudgetStatus::Exhausted,
            "simple arm got 90% of samples but only 10% of complexity share — must be Exhausted"
        );
        assert_eq!(
            ledger.status(complex),
            BudgetStatus::Active,
            "complex arm got 10% of samples and 90% of complexity share — must remain Active"
        );
    }

    /// Equal-complexity arms are both Active when samples are evenly split.
    #[test]
    fn budget_normalizes_across_arms_with_different_complexity() {
        let mut ledger = BudgetLedger::new();
        let a = ledger.add_arm(100);
        let b = ledger.add_arm(200);
        let c = ledger.add_arm(300);

        // Distribute samples proportional to complexity: 100, 200, 300 samples.
        // Each arm's sample fraction == complexity fraction → all Active.
        for _ in 0..100 {
            ledger.record_sample(a);
        }
        for _ in 0..200 {
            ledger.record_sample(b);
        }
        for _ in 0..300 {
            ledger.record_sample(c);
        }

        ledger.recompute();

        assert_eq!(
            ledger.status(a),
            BudgetStatus::Active,
            "arm a: 100 samples, complexity 100 — at fair share, must be Active"
        );
        assert_eq!(
            ledger.status(b),
            BudgetStatus::Active,
            "arm b: 200 samples, complexity 200 — at fair share, must be Active"
        );
        assert_eq!(
            ledger.status(c),
            BudgetStatus::Active,
            "arm c: 300 samples, complexity 300 — at fair share, must be Active"
        );

        // Now overrun arm a significantly.
        for _ in 0..500 {
            ledger.record_sample(a);
        }
        ledger.recompute();

        assert_eq!(
            ledger.status(a),
            BudgetStatus::Exhausted,
            "arm a overrun: now 600 samples vs 100 complexity — must be Exhausted"
        );
    }

    /// The total_cost() reports actual usage (samples × complexity), not predicted.
    #[test]
    fn budget_total_cost_reflects_actual_usage() {
        let mut ledger = BudgetLedger::new();
        let idx = ledger.add_arm(50);
        for _ in 0..20 {
            ledger.record_sample(idx);
        }
        // 20 samples × complexity 50 = 1000
        assert_eq!(
            ledger.total_cost(),
            1000,
            "total cost should be 20 × 50 = 1000"
        );
    }
}
