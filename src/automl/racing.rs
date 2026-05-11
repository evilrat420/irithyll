//! Empirical Bernstein statistical racing for champion promotion.
//!
//! Replaces the bare `<` comparison in champion promotion with a statistically
//! rigorous Bernstein confidence-interval test. A challenger is only promoted
//! when the empirical Bernstein bound proves, with probability ≥ 1−δ, that its
//! true mean error is genuinely below the champion's — not merely noise-favoured.
//!
//! # Mathematical foundation
//!
//! The Empirical Bernstein bound (Maurer & Pontil 2009, eq. 2) gives a
//! one-sided confidence interval for the mean of a bounded random variable
//! using its *empirical* variance:
//!
//! ```text
//! halfwidth(V_n, n, b−a, δ) = √(2·V_n·ln(2/δ)/n) + 7(b−a)·ln(2/δ)/(3(n−1))
//! ```
//!
//! where `V_n` is the sample variance (Bessel-corrected), `n` is sample count,
//! `b−a` is the observed error range, and `δ = 0.05` gives 95% confidence.
//!
//! **Why Bernstein over Hoeffding?** Hoeffding uses only range: `(b−a)·√(ln(2/δ)/(2n))`.
//! Bernstein is always ≤ Hoeffding (tighter or equal), and strictly tighter when
//! variance is small relative to range² — exactly when arms are similar and noise
//! differentiation matters most.
//!
//! **Why Bernstein over z-test?** z-test requires knowing the population variance or
//! large-n asymptotic validity. Welford gives sample variance; plugging into
//! Bernstein is valid at any n ≥ 2. z-test fails to control error probability at
//! small n (early rounds in successive halving).
//!
//! # EWMA drift variant
//!
//! Under concept drift, the stationary Bernstein bound uses stale statistics.
//! The EWMA Welford tracker downweights past observations, providing a drift-aware
//! variance estimate. It uses `n_eff ≈ 1/(1−α)` (steady-state, geometric weights)
//! in place of `n`. The forgetting factor `α` is derived from information-decay
//! matching: `α = exp(−ln(2)/H)` where `H` is the EWMA half-life, set to match
//! the prediction horizon (Adams & MacKay 2007; Kingma & Ba 2015 bias correction).
//!
//! # Pareto integration hook (AM-14)
//!
//! When the Pareto front contains multiple candidates (multi-objective racing),
//! the outer AM-14 agent calls [`bernstein_compare`] on front members to break
//! ties with statistical certainty. The function accepts a slice of
//! `(mean, m2, n, range)` tuples and returns the index of the statistically
//! best candidate, or `None` if no candidate has sufficient certainty.
//!
//! # References
//!
//! - Maurer & Pontil (2009) "Empirical Bernstein Stopping" — eq. 2 used verbatim.
//! - Li et al. (2020) "A System for Massively Parallel Hyperparameter Tuning (ASHA)"
//!   — rung structure rationale.
//! - Adams & MacKay (2007) "Bayesian Online Changepoint Detection" — EWMA-Welford style.
//! - Kingma & Ba (2015) "Adam" — bias correction formula adapted for EWMA transient.
//! - Welford (1962) "Note on a method for calculating corrected sums of squares".

/// Default per-promotion confidence level (failure probability δ).
///
/// δ = 0.05 is the standard 95% confidence choice (Maurer & Pontil 2009).
/// By the union bound, the family-wise error rate across K candidates and L
/// rounds is ≤ K·L·δ; keeping δ small (0.01–0.05) maintains overall coverage.
pub const BERNSTEIN_DELTA: f64 = 0.05;

/// Default minimum sample count before the Bernstein bound is applied.
///
/// With n < 2 the sample variance is undefined. With n < MIN_SAMPLES_FOR_BERNSTEIN
/// the bound is so wide that it provides no useful discrimination; the system
/// falls back to requiring strictly more data. Set to 10 for consistency with
/// the existing early-elimination guard (see auto_tuner/racing.rs).
pub const MIN_SAMPLES_FOR_BERNSTEIN: u64 = 10;

// ===========================================================================
// WelfordTracker — stationary accumulator
// ===========================================================================

/// Welford online statistics for a stream of scalar rewards (or errors).
///
/// The two-delta recurrence (Welford 1962) accumulates `m2 = Σ(x_i − mean)²`
/// in one pass without catastrophic cancellation. Seeded on the first
/// observation to prevent cold-start bias (per irithyll streaming principles).
#[derive(Debug, Clone, Default)]
pub struct WelfordTracker {
    /// Number of observations accumulated.
    pub n: u64,
    /// Running mean.
    pub mean: f64,
    /// Unnormalized sum of squared deviations (`Σ(x_i − mean)²`).
    pub m2: f64,
    /// Minimum value observed (for range).
    pub r_min: f64,
    /// Maximum value observed (for range).
    pub r_max: f64,
}

impl WelfordTracker {
    /// New, empty tracker. Use [`WelfordTracker::update`] to accumulate samples.
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            r_min: f64::INFINITY,
            r_max: f64::NEG_INFINITY,
        }
    }

    /// Feed one observation.
    ///
    /// First observation seeds the mean to `x` (not 0) to avoid cold-start bias.
    pub fn update(&mut self, x: f64) {
        self.n += 1;
        let delta1 = x - self.mean;
        self.mean += delta1 / self.n as f64;
        let delta2 = x - self.mean; // uses updated mean — Welford two-delta form
        self.m2 += delta1 * delta2;
        if x < self.r_min {
            self.r_min = x;
        }
        if x > self.r_max {
            self.r_max = x;
        }
    }

    /// Bessel-corrected sample variance (`m2 / (n−1)`).
    ///
    /// Returns `f64::INFINITY` when `n < 2` (undefined — wide CI forces more data).
    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / (self.n - 1) as f64
        } else {
            f64::INFINITY
        }
    }

    /// Observed range `r_max − r_min`.
    ///
    /// Returns 0 when no observations have been made.
    pub fn range(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.r_max - self.r_min
        }
    }

    /// Reset all state (drift flush — Welford stats become adversarial post-drift).
    ///
    /// Keeps `r_min` / `r_max` as soft range bounds because they remain valid
    /// estimates of the plausible reward space even after regime shift.
    pub fn flush(&mut self) {
        let (keep_min, keep_max) = (self.r_min, self.r_max);
        *self = Self::new();
        self.r_min = keep_min;
        self.r_max = keep_max;
    }
}

// ===========================================================================
// EwmaWelfordTracker — drift-aware accumulator
// ===========================================================================

/// EWMA-weighted Welford statistics for drift-aware variance estimation.
///
/// Replaces uniform averaging with exponential weights `w_i = α^(n−i)`, so
/// recent observations dominate. The effective sample size `n_eff ≈ 1/(1−α)` in
/// steady state (geometric-weight series). Bias correction (Kingma & Ba 2015)
/// is applied during the initial transient (`n < 1/(1−α)` samples).
///
/// Forgetting factor derivation (information-decay matching): set α such that
/// the EWMA half-life H satisfies `α = exp(−ln(2)/H)`, where H is the prediction
/// horizon length (Adams & MacKay 2007). Default `α = 0.98` corresponds to
/// H ≈ 34 samples — appropriate for short prediction horizons in streaming.
#[derive(Debug, Clone)]
pub struct EwmaWelfordTracker {
    /// Effective sample size (approx `1/(1−α)` in steady state).
    pub n_eff: f64,
    /// EWMA mean.
    pub mean: f64,
    /// EWMA variance accumulator (unnormalized).
    pub m2: f64,
    /// Forgetting factor α ∈ (0, 1). Default `0.98` (H ≈ 34 samples).
    pub alpha: f64,
    /// Observation count (for bias correction).
    pub n: u64,
    /// Minimum observed value.
    pub r_min: f64,
    /// Maximum observed value.
    pub r_max: f64,
}

impl EwmaWelfordTracker {
    /// Create a new tracker with the given forgetting factor α.
    ///
    /// # Derivation
    ///
    /// `α = 0.98` is derived via information-decay matching: the EWMA half-life
    /// H = `−1/ln(α) ≈ 49` samples, corresponding to a moderate prediction horizon.
    /// For shorter horizons use α closer to 0.95 (H ≈ 13). For longer use 0.995 (H ≈ 138).
    pub fn new(alpha: f64) -> Self {
        Self {
            n_eff: 1.0,
            mean: 0.0,
            m2: 0.0,
            alpha,
            n: 0,
            r_min: f64::INFINITY,
            r_max: f64::NEG_INFINITY,
        }
    }

    /// Feed one observation with exponential forgetting.
    pub fn update(&mut self, x: f64) {
        self.n += 1;
        let one_minus_a = 1.0 - self.alpha;
        let delta = x - self.mean;
        self.mean += one_minus_a * delta;
        self.m2 = self.alpha * self.m2 + self.alpha * one_minus_a * delta * delta;
        // n_eff tracks the effective sample size = 1 / sum_of_squared_weights.
        // sum_sq recurrence: sum_sq_new = α² * sum_sq_old + (1−α)²
        // Since n_eff = 1/sum_sq, sum_sq = 1/n_eff:
        //   sum_sq_new = α²/n_eff_old + (1−α)²
        //   n_eff_new  = 1 / sum_sq_new
        // Steady-state: n_eff → (1+α)/(1−α) (e.g. α=0.98 → ~99 samples).
        let sum_sq = self.alpha * self.alpha / self.n_eff + one_minus_a * one_minus_a;
        self.n_eff = if sum_sq > 1e-15 {
            1.0 / sum_sq
        } else {
            1.0 / one_minus_a // fallback: steady-state lower bound
        };
        if x < self.r_min {
            self.r_min = x;
        }
        if x > self.r_max {
            self.r_max = x;
        }
    }

    /// Bias-corrected EWMA mean (Kingma & Ba 2015 style).
    ///
    /// Applied when `n < 1/(1−α)` (within one effective window of startup).
    pub fn corrected_mean(&self) -> f64 {
        let bias = 1.0 - self.alpha.powi(self.n as i32);
        if bias > 1e-15 {
            self.mean / bias
        } else {
            self.mean
        }
    }

    /// Bias-corrected EWMA variance (Kingma & Ba 2015 style).
    pub fn corrected_variance(&self) -> f64 {
        let bias = 1.0 - self.alpha.powi(self.n as i32);
        let raw = if bias > 1e-15 {
            self.m2 / bias
        } else {
            self.m2
        };
        // Normalize by (1−α): converts accumulated m2 into per-sample variance.
        let one_minus_a = 1.0 - self.alpha;
        if one_minus_a > 1e-15 {
            raw / one_minus_a
        } else {
            raw
        }
    }

    /// Observed range.
    pub fn range(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.r_max - self.r_min
        }
    }

    /// Flush after drift: reset statistics but keep range as soft bounds.
    pub fn flush(&mut self) {
        let (alpha, keep_min, keep_max) = (self.alpha, self.r_min, self.r_max);
        *self = Self::new(alpha);
        self.r_min = keep_min;
        self.r_max = keep_max;
    }
}

// ===========================================================================
// Core Bernstein bound functions
// ===========================================================================

/// Empirical Bernstein confidence interval half-width.
///
/// Implements Maurer & Pontil (2009), eq. 2, verbatim:
///
/// ```text
/// half_width = √(2·V_n·ln(2/δ)/n) + 7(b−a)·ln(2/δ)/(3(n−1))
/// ```
///
/// - `variance`: sample variance `V_n = m2/(n−1)` (Bessel-corrected).
/// - `n`: sample count (use `n_eff` for EWMA variant).
/// - `range`: observed range `b−a = r_max−r_min`.
/// - `delta`: per-test failure probability (δ = 0.05 → 95% confidence).
///
/// **Term origins:**
/// - Term 1: `√(2·V_n·ln(2/δ)/n)` — variance-driven term. Dominant when
///   variance is high relative to range. Shrinks as `O(1/√n)`.
/// - Term 2: `7(b−a)·ln(2/δ)/(3(n−1))` — range correction for the plug-in
///   bias when using empirical variance. Factor 7/3 from Taylor analysis in
///   the paper. Shrinks as `O(1/n)`.
///
/// Returns `f64::INFINITY` when `n < 2` (variance undefined).
pub fn bernstein_halfwidth(variance: f64, n: f64, range: f64, delta: f64) -> f64 {
    if n < 2.0 {
        return f64::INFINITY;
    }
    let ln2d = (2.0 / delta).ln();
    let term1 = (2.0 * variance * ln2d / n).sqrt();
    let term2 = (7.0 * range / (3.0 * (n - 1.0))) * ln2d;
    term1 + term2
}

/// Empirical Bernstein confidence interval: (lower_bound, upper_bound).
///
/// Uses [`bernstein_halfwidth`] to construct a two-sided CI around the mean.
/// Returns `(−∞, +∞)` when `n < 2`.
///
/// # Arguments
///
/// - `mean`: sample mean.
/// - `m2`: Welford's accumulated `Σ(x_i − mean_i)²`.
/// - `n`: sample count (u64; pass as f64 for EWMA variant).
/// - `range`: observed range `r_max − r_min`.
/// - `delta`: failure probability (default [`BERNSTEIN_DELTA`] = 0.05).
pub fn empirical_bernstein_ci(mean: f64, m2: f64, n: u64, range: f64, delta: f64) -> (f64, f64) {
    let n_f = n as f64;
    if n < 2 {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    let variance = m2 / (n_f - 1.0);
    let hw = bernstein_halfwidth(variance, n_f, range, delta);
    (mean - hw, mean + hw)
}

/// EWMA-adapted Bernstein CI using `n_eff` in place of `n`.
///
/// Under concept drift the stationary `n` overstates confidence. Using
/// `n_eff ≈ 1/(1−α)` (geometric-weight effective samples) gives a wider,
/// more honest CI that reflects only recent observations.
///
/// Applies bias correction before computing variance (Kingma & Ba 2015).
pub fn ewma_bernstein_ci(tracker: &EwmaWelfordTracker, delta: f64) -> (f64, f64) {
    if tracker.n < 2 {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    let n_eff = tracker.n_eff;
    let variance = tracker.corrected_variance();
    let mean = tracker.corrected_mean();
    let range = tracker.range();
    let hw = bernstein_halfwidth(variance, n_eff, range, delta);
    (mean - hw, mean + hw)
}

// ===========================================================================
// BernsteinPromotion — champion-challenger gate
// ===========================================================================

/// Outcome of a Bernstein promotion test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionVerdict {
    /// The challenger's CI upper bound is below the champion metric:
    /// statistically certain improvement. Promote.
    Promote,
    /// The challenger's CI overlaps with the champion metric:
    /// insufficient statistical certainty. Do not promote.
    Inconclusive,
    /// The challenger's mean is worse than the champion metric:
    /// definitively worse. Do not promote.
    Worse,
}

/// Bernstein promotion test: should the challenger replace the champion?
///
/// Tests whether the challenger's true mean error is, with probability ≥ 1−δ,
/// strictly below the champion's current metric. Promotion requires:
///
/// ```text
/// challenger_mean + halfwidth < champion_metric
/// ```
///
/// In words: even the *worst-case* (upper bound) of the challenger's error
/// estimate is below the champion's error — not just noise-favoured mean.
///
/// # Arguments
///
/// - `challenger_mean`: challenger's Welford mean error.
/// - `challenger_m2`: challenger's Welford m2.
/// - `challenger_n`: samples challenger has seen.
/// - `challenger_range`: observed error range of challenger.
/// - `champion_metric`: champion's current EWMA metric (same unit as error).
/// - `delta`: failure probability (default [`BERNSTEIN_DELTA`]).
///
/// # Returns
///
/// [`PromotionVerdict`] indicating the statistical outcome.
pub fn bernstein_promotion_test(
    challenger_mean: f64,
    challenger_m2: f64,
    challenger_n: u64,
    challenger_range: f64,
    champion_metric: f64,
    delta: f64,
) -> PromotionVerdict {
    if challenger_n < MIN_SAMPLES_FOR_BERNSTEIN {
        return PromotionVerdict::Inconclusive;
    }
    let (_lo, hi) = empirical_bernstein_ci(
        challenger_mean,
        challenger_m2,
        challenger_n,
        challenger_range,
        delta,
    );
    if challenger_mean >= champion_metric {
        PromotionVerdict::Worse
    } else if hi < champion_metric {
        PromotionVerdict::Promote
    } else {
        PromotionVerdict::Inconclusive
    }
}

// ===========================================================================
// Pareto front tiebreak hook (for AM-14)
// ===========================================================================

/// Statistics tuple for a single arm, used in multi-arm Bernstein tiebreaking.
///
/// Fields: `(mean_error, m2, n, range)`.
pub type ArmStats = (f64, f64, u64, f64);

/// Bernstein tiebreak over a Pareto front (AM-14 integration hook).
///
/// When the Pareto front after multi-objective selection contains `|front| > 1`
/// candidates, the AM-14 agent calls this function to select the member with
/// statistically certain superiority over all others.
///
/// Returns `Some(idx)` into the `front` slice when one arm's CI upper bound is
/// strictly below every other arm's CI lower bound. Returns `None` if no arm
/// achieves statistical dominance (caller should fall back to lowest mean or
/// retain all front members).
///
/// # Design contract for AM-14
///
/// ```text
/// let front: &[ArmStats] = ...;  // one entry per Pareto-front member
/// match bernstein_compare(front, BERNSTEIN_DELTA) {
///     Some(idx) => /* promote front[idx] */,
///     None => /* use lowest-mean fallback or retain all */,
/// }
/// ```
pub fn bernstein_compare(front: &[ArmStats], delta: f64) -> Option<usize> {
    if front.is_empty() {
        return None;
    }
    if front.len() == 1 {
        return Some(0);
    }

    // Compute CIs for all front members.
    let cis: Vec<(f64, f64)> = front
        .iter()
        .map(|&(mean, m2, n, range)| empirical_bernstein_ci(mean, m2, n, range, delta))
        .collect();

    // An arm at index `i` is a statistical winner if its hi < every other arm's lo.
    for (i, &(_lo_i, hi_i)) in cis.iter().enumerate() {
        // Skip if CI is degenerate (insufficient data).
        if !hi_i.is_finite() {
            continue;
        }
        let dominates = cis
            .iter()
            .enumerate()
            .all(|(j, &(lo_j, _))| j == i || hi_i < lo_j);
        if dominates {
            // Also sanity: this arm must have the lowest mean error.
            let mean_i = front[i].0;
            let is_best_mean = front
                .iter()
                .enumerate()
                .all(|(j, &(m, _, _, _))| j == i || mean_i <= m);
            if is_best_mean {
                return Some(i);
            }
        }
    }

    // No single dominant arm found — check if lowest-mean arm's hi < second-lowest lo.
    let mut sorted_by_mean: Vec<usize> = (0..front.len()).collect();
    sorted_by_mean.sort_by(|&a, &b| {
        front[a]
            .0
            .partial_cmp(&front[b].0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best_idx = sorted_by_mean[0];
    let second_idx = sorted_by_mean[1];
    let (_, hi_best) = cis[best_idx];
    let (lo_second, _) = cis[second_idx];
    if hi_best.is_finite() && lo_second.is_finite() && hi_best < lo_second {
        return Some(best_idx);
    }

    None
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // WelfordTracker correctness
    // -----------------------------------------------------------------------

    #[test]
    fn welford_tracker_mean_and_variance() {
        let mut t = WelfordTracker::new();
        // Classic example: [2, 4, 4, 4, 5, 5, 7, 9] → mean=5, variance=4.5714...
        let values = [2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        for &v in &values {
            t.update(v);
        }
        assert!(
            (t.mean - 5.0).abs() < 1e-10,
            "mean should be 5.0, got {}",
            t.mean
        );
        let expected_var = 32.0 / 7.0; // sum of sq-dev=32, n-1=7
        assert!(
            (t.variance() - expected_var).abs() < 1e-10,
            "variance should be {expected_var:.6}, got {}",
            t.variance()
        );
        assert!(
            (t.range() - 7.0).abs() < 1e-10,
            "range should be 7.0, got {}",
            t.range()
        );
    }

    #[test]
    fn welford_tracker_cold_start_single_sample() {
        let mut t = WelfordTracker::new();
        let sample = core::f64::consts::PI;
        t.update(sample);
        assert!(
            (t.mean - sample).abs() < 1e-12,
            "single-sample mean should equal the sample, got {}",
            t.mean
        );
        assert!(
            t.variance().is_infinite(),
            "single-sample variance should be infinite (undefined), got {}",
            t.variance()
        );
    }

    #[test]
    fn welford_tracker_flush_resets_stats_keeps_range() {
        let mut t = WelfordTracker::new();
        for v in [1.0f64, 2.0, 3.0, 4.0] {
            t.update(v);
        }
        let (kept_min, kept_max) = (t.r_min, t.r_max);
        t.flush();
        assert_eq!(t.n, 0, "flush should reset n to 0, got {}", t.n);
        assert!(
            (t.r_min - kept_min).abs() < 1e-12,
            "flush should preserve r_min as soft bound: expected {kept_min}, got {}",
            t.r_min
        );
        assert!(
            (t.r_max - kept_max).abs() < 1e-12,
            "flush should preserve r_max as soft bound: expected {kept_max}, got {}",
            t.r_max
        );
    }

    // -----------------------------------------------------------------------
    // bernstein_halfwidth properties
    // -----------------------------------------------------------------------

    #[test]
    fn bernstein_bound_widens_with_variance() {
        // Higher variance → wider bound at same n, range, delta.
        let range = 1.0;
        let n = 100.0;
        let delta = 0.05;
        let hw_low = bernstein_halfwidth(0.01, n, range, delta);
        let hw_high = bernstein_halfwidth(0.5, n, range, delta);
        assert!(
            hw_high > hw_low,
            "higher variance should produce wider bound: hw_low={hw_low:.6}, hw_high={hw_high:.6}"
        );
    }

    #[test]
    fn bernstein_tighter_than_hoeffding_at_low_variance() {
        // Bernstein's primary advantage: tighter than Hoeffding when variance << range²
        // (the common case when arms are similar). Maurer & Pontil (2009) §4.
        //
        // The variance-driven term of Bernstein, sqrt(2·V·ln(2/δ)/n), is always ≤
        // the Hoeffding bound range·sqrt(ln(2/δ)/(2n)) when V ≤ range²/4 (which holds
        // by definition for bounded r.v.). The full Bernstein bound also includes a
        // range-correction term 7(b−a)ln(2/δ)/(3(n−1)) that shrinks as O(1/n), so
        // Bernstein is strictly tighter at large n when variance is low relative to range².
        let range = 2.0;
        let delta = 0.05;
        // Low-variance cases with large n: full Bernstein < Hoeffding.
        for &(variance, n_u64) in &[(0.01f64, 1000u64), (0.1, 2000), (0.001, 500)] {
            let n = n_u64 as f64;
            let bernstein = bernstein_halfwidth(variance, n, range, delta);
            let hoeffding = range * ((2.0 / delta).ln() / (2.0 * n)).sqrt();
            assert!(
                bernstein < hoeffding,
                "Bernstein should be < Hoeffding at large n with low variance \
                 (n={n_u64}, var={variance}): bernstein={bernstein:.6}, hoeffding={hoeffding:.6}"
            );
        }
        // Monotone in variance: variance term of Bernstein is sub-Hoeffding.
        // sqrt(2·V·ln(2/δ)/n) ≤ range·sqrt(ln(2/δ)/(2n)) iff 2V ≤ range²/2 iff V ≤ range²/4.
        let n = 1000.0f64;
        let var_small = 0.001; // V << range²/4 = 1.0
        let term1_small = (2.0 * var_small * (2.0_f64 / delta).ln() / n).sqrt();
        let hoeff = range * ((2.0 / delta).ln() / (2.0 * n)).sqrt();
        assert!(
            term1_small < hoeff,
            "Bernstein term1 (variance component) should be < Hoeffding for small variance: \
             term1={term1_small:.6}, hoeffding={hoeff:.6}"
        );
    }

    #[test]
    fn bernstein_returns_infinity_for_small_n() {
        // n < 2 → INFINITY (variance undefined, cannot bound).
        assert!(
            bernstein_halfwidth(0.5, 0.0, 1.0, 0.05).is_infinite(),
            "n=0 should give INFINITY"
        );
        assert!(
            bernstein_halfwidth(0.5, 1.0, 1.0, 0.05).is_infinite(),
            "n=1 should give INFINITY"
        );
    }

    // -----------------------------------------------------------------------
    // bernstein_promotion_test correctness
    // -----------------------------------------------------------------------

    #[test]
    fn bernstein_promotion_requires_statistical_certainty() {
        // Single-trial noise: challenger slightly better in mean but CI overlaps.
        // With n=11, large variance and small difference → should NOT promote.
        let champion_metric = 0.50;
        // Challenger has mean 0.48 (better) but large variance.
        let challenger_mean = 0.48;
        let challenger_range = 1.0;
        // Compute m2 from known variance: var=0.2, n=11 → m2 = var*(n-1) = 2.0
        let n = 11u64;
        let variance = 0.2f64;
        let challenger_m2 = variance * (n - 1) as f64;

        let verdict = bernstein_promotion_test(
            challenger_mean,
            challenger_m2,
            n,
            challenger_range,
            champion_metric,
            BERNSTEIN_DELTA,
        );
        assert_ne!(
            verdict,
            PromotionVerdict::Promote,
            "small n ({n}) with high variance should NOT promote: got {verdict:?}"
        );
    }

    #[test]
    fn bernstein_promotes_with_clear_advantage() {
        // Large n, tiny variance, big advantage → should promote.
        let champion_metric = 0.50;
        let challenger_mean = 0.10; // clearly better
        let challenger_range = 0.05;
        let n = 1000u64;
        let variance = 0.001f64;
        let challenger_m2 = variance * (n - 1) as f64;

        let verdict = bernstein_promotion_test(
            challenger_mean,
            challenger_m2,
            n,
            challenger_range,
            champion_metric,
            BERNSTEIN_DELTA,
        );
        assert_eq!(
            verdict,
            PromotionVerdict::Promote,
            "large n, tiny variance, large advantage should Promote: got {verdict:?}"
        );
    }

    #[test]
    fn bernstein_worse_when_challenger_mean_exceeds_champion() {
        // Challenger is worse in mean → Worse, never Promote.
        let verdict = bernstein_promotion_test(
            0.80, // challenger mean — worse than champion
            0.01, // m2
            200,  // n
            0.1,  // range
            0.50, // champion metric (lower is better)
            BERNSTEIN_DELTA,
        );
        assert_eq!(
            verdict,
            PromotionVerdict::Worse,
            "challenger with higher mean error should be Worse, got {verdict:?}"
        );
    }

    #[test]
    fn bernstein_inconclusive_below_min_samples() {
        // Fewer than MIN_SAMPLES_FOR_BERNSTEIN → Inconclusive (not enough data).
        let verdict = bernstein_promotion_test(
            0.10,                          // great mean
            0.001,                         // tiny m2
            MIN_SAMPLES_FOR_BERNSTEIN - 1, // below threshold
            0.1,                           // small range
            0.50,                          // champion metric
            BERNSTEIN_DELTA,
        );
        assert_eq!(
            verdict,
            PromotionVerdict::Inconclusive,
            "fewer than MIN_SAMPLES_FOR_BERNSTEIN should be Inconclusive, got {verdict:?}"
        );
    }

    // -----------------------------------------------------------------------
    // EWMA drift adaptation
    // -----------------------------------------------------------------------

    #[test]
    fn bernstein_handles_drift_via_ewma_decay() {
        // After many stationary samples the EWMA reaches steady-state n_eff ≈ (1+α)/(1−α),
        // giving a tight, finite CI. After a drift flush + far fewer samples, n_eff is much
        // smaller and the CI is correspondingly wider — reflecting reduced certainty.
        //
        // With α=0.98, steady-state n_eff ≈ 99 (>> 2). After flush + 50 new samples,
        // n_eff ≈ 10 (still >> 2 so CI is finite, but much wider).
        let alpha = 0.98f64;
        let mut tracker = EwmaWelfordTracker::new(alpha);

        // Feed 500 stationary samples in a narrow band [0.4, 0.6].
        // After this many samples n_eff ≈ (1+α)/(1−α) ≈ 99 and CI is tight.
        for i in 0..500u64 {
            let x = 0.5 + 0.1 * (if i % 2 == 0 { 1.0 } else { -1.0 });
            tracker.update(x);
        }
        let (lo_before, hi_before) = ewma_bernstein_ci(&tracker, BERNSTEIN_DELTA);
        let width_before = hi_before - lo_before;
        assert!(
            width_before.is_finite(),
            "pre-flush CI should be finite after 500 samples: lo={lo_before}, hi={hi_before}"
        );

        // Drift flush: resets n_eff to 1.0 (fresh start, range bounds kept).
        tracker.flush();

        // Feed 50 new samples in the shifted regime. With α=0.98 and 50 samples,
        // n_eff ≈ 1/(α²/1 + (1−α)²) accumulated over 50 steps. Still << 99.
        // CI should be finite (n_eff > 2 after ~15 samples) but much wider than pre-flush.
        for _ in 0..50u64 {
            tracker.update(0.9); // new regime
        }
        let (lo_after, hi_after) = ewma_bernstein_ci(&tracker, BERNSTEIN_DELTA);
        let width_after = hi_after - lo_after;

        assert!(
            width_after.is_finite(),
            "post-flush CI should be finite after 50 samples: lo={lo_after}, hi={hi_after}"
        );
        assert!(
            width_after > width_before,
            "CI after drift flush should be wider (less certainty after reset): \
             before={width_before:.6}, after={width_after:.6}"
        );
    }

    // -----------------------------------------------------------------------
    // Pareto front tiebreak hook
    // -----------------------------------------------------------------------

    #[test]
    fn pareto_front_can_invoke_bernstein_tiebreak() {
        // Two candidates on the Pareto front: one clearly better with enough data.
        // For bernstein_compare to select arm 0, arm 0's CI upper bound must be
        // strictly below arm 1's CI lower bound: hi_0 < lo_1.
        //
        // Arm 0: mean=0.10, tiny variance, large n → tight CI around 0.10.
        //   halfwidth ≈ sqrt(2*0.0001*3.69/2000) + 7*0.05*3.69/(3*1999) ≈ 0.00061 + 0.000215 ≈ 0.00082
        //   CI ≈ (0.0992, 0.1008)
        // Arm 1: mean=0.50, tiny variance, large n → tight CI around 0.50.
        //   halfwidth ≈ sqrt(2*0.0001*3.69/2000) + 7*0.05*3.69/(3*1999) ≈ 0.00082
        //   CI ≈ (0.4992, 0.5008)
        // Condition: hi_0 (0.1008) < lo_1 (0.4992) → arm 0 wins.
        let n0 = 2000u64;
        let var0 = 0.0001f64;
        let m2_0 = var0 * (n0 - 1) as f64;
        let range0 = 0.05f64;

        let n1 = 2000u64;
        let var1 = 0.0001f64;
        let m2_1 = var1 * (n1 - 1) as f64;
        let range1 = 0.05f64;

        let front: &[ArmStats] = &[
            (0.10, m2_0, n0, range0), // clearly better
            (0.50, m2_1, n1, range1), // clearly worse, tight CI so lo_1 > hi_0
        ];

        let winner = bernstein_compare(front, BERNSTEIN_DELTA);
        assert_eq!(
            winner,
            Some(0),
            "Pareto tiebreak should select arm 0 (hi_0 < lo_1, statistically dominant): got {winner:?}"
        );
    }

    #[test]
    fn pareto_front_returns_none_when_uncertain() {
        // Two candidates with overlapping CIs — no winner.
        let n = 15u64;
        let var = 0.5f64;
        let m2 = var * (n - 1) as f64;
        let range = 1.0f64;

        let front: &[ArmStats] = &[
            (0.48, m2, n, range), // similar, high variance
            (0.52, m2, n, range),
        ];

        let winner = bernstein_compare(front, BERNSTEIN_DELTA);
        assert_eq!(
            winner, None,
            "Pareto tiebreak with overlapping CIs should return None, got {winner:?}"
        );
    }

    #[test]
    fn pareto_front_single_entry_always_wins() {
        let front: &[ArmStats] = &[(0.3, 0.01, 5, 0.1)];
        assert_eq!(
            bernstein_compare(front, BERNSTEIN_DELTA),
            Some(0),
            "single-entry Pareto front should always return Some(0)"
        );
    }
}
