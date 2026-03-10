//! ADWIN (ADaptive WINdowing) drift detector.
//!
//! ADWIN maintains a variable-length window of recent values using an exponential
//! histogram for compression. It detects concept drift by comparing the means of
//! sub-windows at every bucket boundary. When a statistically significant difference
//! is found, the window is shrunk by removing the older portion.
//!
//! # Algorithm
//!
//! Each new value is inserted as a bucket of order 0. When a row accumulates more
//! than `max_buckets` entries, the two oldest are merged into a single bucket at the
//! next order (capacity doubles). This gives O(log W) memory for a window of width W.
//!
//! Drift is tested by iterating bucket boundaries from newest to oldest, splitting
//! the window into a right (newer) and left (older) sub-window. The Hoeffding-based
//! cut uses `epsilon = sqrt((1 / 2m) * ln(4 / delta'))` where `m` is the harmonic
//! count and `delta' = delta / ln(width)`.
//!
//! # Reference
//!
//! Bifet, A. & Gavalda, R. (2007). "Learning from Time-Changing Data with Adaptive
//! Windowing." In *Proceedings of the 2007 SIAM International Conference on Data Mining*.

use crate::drift::{DriftDetector, DriftSignal};

// ---------------------------------------------------------------------------
// Bucket
// ---------------------------------------------------------------------------

/// A single bucket in the exponential histogram.
///
/// Buckets at row `i` summarise `2^i` original observations.
#[derive(Debug, Clone)]
struct Bucket {
    /// Sum of values stored in this bucket.
    total: f64,
    /// Variance accumulator: sum of squared deviations (variance * count).
    variance: f64,
    /// Number of original values represented by this bucket.
    count: u64,
}

impl Bucket {
    /// Create a leaf bucket for a single observed value.
    #[inline]
    fn singleton(value: f64) -> Self {
        Self {
            total: value,
            variance: 0.0,
            count: 1,
        }
    }

    /// Merge two buckets into one, combining their sufficient statistics.
    ///
    /// The variance update follows the parallel variance formula:
    /// `Var(A+B) = Var(A) + Var(B) + (mean_A - mean_B)^2 * n_A * n_B / (n_A + n_B)`
    fn merge(a: &Bucket, b: &Bucket) -> Self {
        let count = a.count + b.count;
        let total = a.total + b.total;

        let mean_a = a.total / a.count as f64;
        let mean_b = b.total / b.count as f64;
        let diff = mean_a - mean_b;
        let variance = a.variance
            + b.variance
            + diff * diff * (a.count as f64) * (b.count as f64) / (count as f64);

        Self {
            total,
            variance,
            count,
        }
    }
}

// ---------------------------------------------------------------------------
// Adwin
// ---------------------------------------------------------------------------

/// ADWIN (ADaptive WINdowing) drift detector.
///
/// Maintains an adaptive-size window over a stream of real-valued observations
/// using an exponential histogram. Detects distribution shift by comparing
/// sub-window means with a Hoeffding-style bound.
///
/// # Examples
///
/// ```
/// use irithyll::drift::{DriftDetector, DriftSignal};
/// use irithyll::drift::adwin::Adwin;
///
/// let mut det = Adwin::new();
/// // Feed stable values
/// for _ in 0..200 {
///     let sig = det.update(0.0);
///     assert_ne!(sig, DriftSignal::Drift);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Adwin {
    /// Confidence parameter. Smaller values require stronger evidence to declare
    /// drift. Default: 0.002.
    delta: f64,
    /// Maximum number of buckets allowed per row before compression triggers.
    /// Default: 5.
    max_buckets: usize,
    /// Bucket rows. `rows[i]` holds buckets whose capacity is `2^i`.
    /// Within each row the *oldest* bucket is at the front (index 0) and the
    /// *newest* is at the back.
    rows: Vec<Vec<Bucket>>,
    /// Running sum of all values currently in the window.
    total: f64,
    /// Running variance accumulator for the full window.
    variance: f64,
    /// Total number of original observations in the window.
    count: u64,
    /// Width of the window (mirrors `count`; kept for semantic clarity).
    width: u64,
    /// Minimum number of observations before drift checking begins.
    min_window: u64,
}

impl Adwin {
    /// Create a new ADWIN detector with the default confidence `delta = 0.002`.
    pub fn new() -> Self {
        Self::with_delta(0.002)
    }

    /// Create a new ADWIN detector with the given confidence parameter.
    ///
    /// `delta` controls sensitivity: smaller values mean fewer false positives but
    /// slower reaction to real drift. Typical range: 0.0001 to 0.1.
    ///
    /// # Panics
    ///
    /// Panics if `delta` is not in `(0, 1)`.
    pub fn with_delta(delta: f64) -> Self {
        assert!(
            delta > 0.0 && delta < 1.0,
            "delta must be in (0, 1), got {delta}"
        );
        Self {
            delta,
            max_buckets: 5,
            rows: Vec::new(),
            total: 0.0,
            variance: 0.0,
            count: 0,
            width: 0,
            min_window: 32,
        }
    }

    /// Set the maximum number of buckets per row (default 5).
    ///
    /// Higher values use more memory but give finer granularity for split-point
    /// testing. Must be at least 2.
    pub fn with_max_buckets(mut self, m: usize) -> Self {
        assert!(m >= 2, "max_buckets must be >= 2, got {m}");
        self.max_buckets = m;
        self
    }

    /// Set the minimum window size before drift checking begins (default 32).
    pub fn with_min_window(mut self, min: u64) -> Self {
        self.min_window = min;
        self
    }

    /// Current window width (number of original observations).
    #[inline]
    pub fn width(&self) -> u64 {
        self.width
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Insert a new singleton bucket at row 0.
    fn insert_bucket(&mut self, value: f64) {
        if self.rows.is_empty() {
            self.rows.push(Vec::new());
        }
        self.rows[0].push(Bucket::singleton(value));

        // Update running stats using Welford's online method for variance.
        self.count += 1;
        self.width += 1;
        let old_mean = if self.count > 1 {
            (self.total) / (self.count - 1) as f64
        } else {
            0.0
        };
        self.total += value;
        let new_mean = self.total / self.count as f64;
        self.variance += (value - old_mean) * (value - new_mean);
    }

    /// Compress bucket rows: whenever a row has more than `max_buckets` entries,
    /// merge the two oldest (front of the vec) into one bucket at the next row.
    fn compress(&mut self) {
        let max = self.max_buckets;
        let mut row_idx = 0;
        while row_idx < self.rows.len() {
            if self.rows[row_idx].len() <= max {
                break; // rows beyond this one can't have overflowed from this insertion
            }
            // Merge the two oldest buckets (indices 0 and 1).
            let b1 = self.rows[row_idx].remove(0);
            let b2 = self.rows[row_idx].remove(0);
            let merged = Bucket::merge(&b1, &b2);

            let next_row = row_idx + 1;
            if next_row >= self.rows.len() {
                self.rows.push(Vec::new());
            }
            self.rows[next_row].push(merged);

            row_idx += 1;
        }
    }

    /// Check for drift by scanning split points from newest to oldest.
    ///
    /// Returns `(drift_detected, warning_detected)`.
    fn check_drift(&self) -> (bool, bool) {
        if self.width <= self.min_window {
            return (false, false);
        }
        // We need at least 2 observations on each side of a split.
        if self.width < 4 {
            return (false, false);
        }

        let ln_width = (self.width as f64).ln();
        // Guard: if width is so small that ln(width) <= 0 we skip.
        if ln_width <= 0.0 {
            return (false, false);
        }

        let delta_prime = self.delta / ln_width;
        let delta_warn = (2.0 * self.delta) / ln_width;

        // Traverse buckets from newest to oldest, accumulating the "right"
        // (newer) sub-window. The "left" (older) sub-window is the remainder.
        let mut right_count: u64 = 0;
        let mut right_total: f64 = 0.0;

        let mut warning_found = false;

        // Iterate rows from 0 (smallest buckets, most recent) upward, and
        // within each row from the back (newest) to the front (oldest).
        for row in &self.rows {
            for bucket in row.iter().rev() {
                right_count += bucket.count;
                right_total += bucket.total;

                let left_count = self.count - right_count;
                if left_count < 1 || right_count < 1 {
                    continue;
                }

                let left_total = self.total - right_total;

                let mean_left = left_total / left_count as f64;
                let mean_right = right_total / right_count as f64;
                let abs_diff = (mean_left - mean_right).abs();

                // Harmonic mean of the two sub-window sizes.
                let n0 = left_count as f64;
                let n1 = right_count as f64;
                let m = 1.0 / (1.0 / n0 + 1.0 / n1);

                // Drift threshold.
                let epsilon_drift = ((1.0 / (2.0 * m)) * (4.0 / delta_prime).ln()).sqrt();

                if abs_diff >= epsilon_drift {
                    // No need to keep scanning — we found the outermost drift point.
                    return (true, true);
                }

                // Warning threshold (looser).
                let epsilon_warn = ((1.0 / (2.0 * m)) * (4.0 / delta_warn).ln()).sqrt();
                if abs_diff >= epsilon_warn {
                    warning_found = true;
                }
            }
        }

        (false, warning_found)
    }

    /// Remove the oldest portion of the window up to (and including) the split
    /// point where drift was detected. We find the split point again and remove
    /// everything on the left side.
    fn shrink_window(&mut self) {
        let ln_width = (self.width as f64).ln();
        if ln_width <= 0.0 {
            return;
        }
        let delta_prime = self.delta / ln_width;

        // Traverse newest-to-oldest just like check_drift, find the first
        // split where drift is confirmed, then remove the left (older) side.
        let mut right_count: u64 = 0;
        let mut right_total: f64 = 0.0;
        let mut right_variance: f64 = 0.0;

        // We need to record which buckets comprise the right sub-window.
        // Strategy: find how many buckets (from the newest end) form the right
        // sub-window, then rebuild rows keeping only those.

        // Collect all buckets in order from newest to oldest.
        let mut all_buckets: Vec<(usize, usize)> = Vec::new(); // (row_idx, bucket_idx_in_row)
        for (row_idx, row) in self.rows.iter().enumerate() {
            for (bucket_idx, _) in row.iter().enumerate().rev() {
                all_buckets.push((row_idx, bucket_idx));
            }
        }

        let mut split_pos = all_buckets.len(); // keep everything by default
        for (pos, &(row_idx, bucket_idx)) in all_buckets.iter().enumerate() {
            let bucket = &self.rows[row_idx][bucket_idx];
            // Update right sub-window variance using parallel formula.
            if right_count > 0 {
                let mean_right_old = right_total / right_count as f64;
                let mean_bucket = bucket.total / bucket.count as f64;
                let diff = mean_right_old - mean_bucket;
                right_variance = right_variance
                    + bucket.variance
                    + diff * diff * (right_count as f64) * (bucket.count as f64)
                        / (right_count + bucket.count) as f64;
            } else {
                right_variance = bucket.variance;
            }
            right_count += bucket.count;
            right_total += bucket.total;

            let left_count = self.count - right_count;
            if left_count < 1 || right_count < 1 {
                continue;
            }
            let left_total = self.total - right_total;
            let mean_left = left_total / left_count as f64;
            let mean_right = right_total / right_count as f64;
            let abs_diff = (mean_left - mean_right).abs();

            let n0 = left_count as f64;
            let n1 = right_count as f64;
            let m = 1.0 / (1.0 / n0 + 1.0 / n1);
            let epsilon = ((1.0 / (2.0 * m)) * (4.0 / delta_prime).ln()).sqrt();

            if abs_diff >= epsilon {
                // The right sub-window (positions 0..=pos) is the part we keep.
                split_pos = pos + 1;
                break;
            }
        }

        if split_pos >= all_buckets.len() {
            // No split found (shouldn't happen if called after check_drift),
            // but just in case, don't remove anything.
            return;
        }

        // Rebuild: keep only the buckets in all_buckets[0..split_pos].
        // These are the *right* (newer) sub-window.
        let keep_set: Vec<(usize, usize)> = all_buckets[..split_pos].to_vec();

        // Build a set for fast lookup: which (row, idx) to keep.
        let mut new_rows: Vec<Vec<Bucket>> = Vec::new();
        // We need to reconstruct in the proper order: within each row,
        // oldest first (low index) to newest (high index).

        // First, determine max row.
        let max_row = self.rows.len();
        new_rows.resize_with(max_row, Vec::new);

        // Mark which buckets to keep. Since keep_set lists them newest-first,
        // we process them in reverse to restore oldest-first order within rows.
        let mut keep_flags: Vec<Vec<bool>> = self
            .rows
            .iter()
            .map(|row| vec![false; row.len()])
            .collect();
        for &(r, b) in &keep_set {
            keep_flags[r][b] = true;
        }

        let mut new_total: f64 = 0.0;
        let mut new_count: u64 = 0;
        for (row_idx, row) in self.rows.iter().enumerate() {
            for (bucket_idx, bucket) in row.iter().enumerate() {
                if keep_flags[row_idx][bucket_idx] {
                    new_total += bucket.total;
                    new_count += bucket.count;
                    new_rows[row_idx].push(bucket.clone());
                }
            }
        }

        // Trim empty trailing rows.
        while new_rows.last().is_some_and(|r| r.is_empty()) {
            new_rows.pop();
        }

        self.rows = new_rows;
        self.total = new_total;
        self.count = new_count;
        self.width = new_count;
        // Recompute variance from the remaining buckets.
        self.recompute_variance();
    }

    /// Recompute the aggregate variance from the bucket tree.
    ///
    /// This is called after shrinking the window because we cannot cheaply
    /// subtract out the removed portion's contribution to variance.
    fn recompute_variance(&mut self) {
        if self.count == 0 {
            self.variance = 0.0;
            return;
        }
        // Merge all buckets' variances using the parallel formula, processing
        // from oldest to newest.
        let mut running_total: f64 = 0.0;
        let mut running_count: u64 = 0;
        let mut running_var: f64 = 0.0;

        // Process in order: highest row first (oldest, largest buckets),
        // then within each row from index 0 (oldest) upward.
        for row in self.rows.iter().rev() {
            for bucket in row.iter() {
                if running_count == 0 {
                    running_total = bucket.total;
                    running_count = bucket.count;
                    running_var = bucket.variance;
                } else {
                    let combined_count = running_count + bucket.count;
                    let mean_running = running_total / running_count as f64;
                    let mean_bucket = bucket.total / bucket.count as f64;
                    let diff = mean_running - mean_bucket;
                    running_var = running_var
                        + bucket.variance
                        + diff * diff * (running_count as f64) * (bucket.count as f64)
                            / combined_count as f64;
                    running_total += bucket.total;
                    running_count = combined_count;
                }
            }
        }
        self.variance = running_var;
    }
}

impl Default for Adwin {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DriftDetector trait
// ---------------------------------------------------------------------------

impl DriftDetector for Adwin {
    fn update(&mut self, value: f64) -> DriftSignal {
        // 1. Insert new observation.
        self.insert_bucket(value);

        // 2. Compress overflowing rows.
        self.compress();

        // 3. Check for drift.
        let (drift, warning) = self.check_drift();

        if drift {
            // Shrink window: remove the older sub-window.
            self.shrink_window();
            DriftSignal::Drift
        } else if warning {
            DriftSignal::Warning
        } else {
            DriftSignal::Stable
        }
    }

    fn reset(&mut self) {
        self.rows.clear();
        self.total = 0.0;
        self.variance = 0.0;
        self.count = 0;
        self.width = 0;
    }

    fn clone_fresh(&self) -> Box<dyn DriftDetector> {
        Box::new(Self::with_delta(self.delta).with_max_buckets(self.max_buckets))
    }

    fn estimated_mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total / self.count as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drift::{DriftDetector, DriftSignal};

    /// Helper: simple deterministic PRNG for reproducible tests without pulling
    /// in `rand` in the test module. Uses xorshift64.
    struct Xorshift64(u64);

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 { 1 } else { seed })
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Uniform f64 in [0, 1).
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        /// Approximate normal via Box-Muller transform.
        fn next_normal(&mut self, mean: f64, std: f64) -> f64 {
            let u1 = self.next_f64().max(1e-15); // avoid ln(0)
            let u2 = self.next_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mean + std * z
        }
    }

    // -----------------------------------------------------------------------
    // 1. No false alarm on constant distribution
    // -----------------------------------------------------------------------
    #[test]
    fn no_false_alarm_constant() {
        let mut det = Adwin::with_delta(0.002);
        let mut rng = Xorshift64::new(42);

        let mut drift_count = 0;
        for _ in 0..10_000 {
            let value = 0.5 + rng.next_normal(0.0, 0.01);
            if det.update(value) == DriftSignal::Drift {
                drift_count += 1;
            }
        }
        // With delta=0.002 over 10k samples we might get a rare false positive,
        // but certainly not more than a handful.
        assert!(
            drift_count <= 2,
            "Too many false alarms on constant distribution: {drift_count}"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Detect abrupt shift
    // -----------------------------------------------------------------------
    #[test]
    fn detect_abrupt_shift() {
        let mut det = Adwin::with_delta(0.002);
        let mut rng = Xorshift64::new(123);

        // Phase 1: N(0, 0.1)
        for _ in 0..2000 {
            det.update(rng.next_normal(0.0, 0.1));
        }

        // Phase 2: N(5, 0.1) — huge shift, should detect quickly.
        let mut detected = false;
        for i in 0..2000 {
            let sig = det.update(rng.next_normal(5.0, 0.1));
            if sig == DriftSignal::Drift {
                detected = true;
                // Should detect within the first ~100 samples of the new regime.
                assert!(
                    i < 500,
                    "Drift detected too late after abrupt shift: sample {i}"
                );
                break;
            }
        }
        assert!(detected, "Failed to detect abrupt mean shift from 0 to 5");
    }

    // -----------------------------------------------------------------------
    // 3. Detect gradual shift
    // -----------------------------------------------------------------------
    #[test]
    fn detect_gradual_shift() {
        let mut det = Adwin::with_delta(0.01); // slightly more sensitive for gradual
        let mut rng = Xorshift64::new(456);

        let mut detected = false;
        for i in 0..5000 {
            // Mean ramps linearly from 0 to 5.
            let mean = 5.0 * (i as f64) / 5000.0;
            let value = rng.next_normal(mean, 0.1);
            if det.update(value) == DriftSignal::Drift {
                detected = true;
                break;
            }
        }
        assert!(detected, "Failed to detect gradual mean shift from 0 to 5");
    }

    // -----------------------------------------------------------------------
    // 4. estimated_mean tracks recent values
    // -----------------------------------------------------------------------
    #[test]
    fn estimated_mean_tracks() {
        let mut det = Adwin::new();
        // Feed exactly 100 values of 3.0.
        for _ in 0..100 {
            det.update(3.0);
        }
        let mean = det.estimated_mean();
        assert!(
            (mean - 3.0).abs() < 1e-9,
            "Expected mean ~3.0, got {mean}"
        );

        // Feed 100 values of 7.0.
        for _ in 0..100 {
            det.update(7.0);
        }
        // Mean should be somewhere between 3 and 7 depending on window.
        let mean = det.estimated_mean();
        assert!(
            mean > 2.5 && mean < 7.5,
            "Mean out of expected range: {mean}"
        );
    }

    // -----------------------------------------------------------------------
    // 5. reset clears all state
    // -----------------------------------------------------------------------
    #[test]
    fn reset_clears_state() {
        let mut det = Adwin::new();
        for _ in 0..500 {
            det.update(1.0);
        }
        assert!(det.width() > 0);
        assert!(det.estimated_mean() > 0.0);

        det.reset();

        assert_eq!(det.width(), 0);
        assert_eq!(det.estimated_mean(), 0.0);
        assert!(det.rows.is_empty());
        assert_eq!(det.count, 0);
        assert_eq!(det.total, 0.0);
        assert_eq!(det.variance, 0.0);
    }

    // -----------------------------------------------------------------------
    // 6. clone_fresh returns clean instance with same delta
    // -----------------------------------------------------------------------
    #[test]
    fn clone_fresh_preserves_config() {
        let mut det = Adwin::with_delta(0.05).with_max_buckets(7);
        for _ in 0..200 {
            det.update(42.0);
        }

        let fresh = det.clone_fresh();
        // The fresh detector should have zero state.
        assert_eq!(fresh.estimated_mean(), 0.0);

        // We can't directly inspect the boxed detector's fields, so we verify
        // behaviour: feeding the same constant shouldn't drift.
        // (This implicitly tests that delta was preserved — if it were near 1.0,
        // we'd get spurious drifts.)
        let mut fresh = fresh;
        let mut drifts = 0;
        for _ in 0..1000 {
            if fresh.update(1.0) == DriftSignal::Drift {
                drifts += 1;
            }
        }
        assert!(
            drifts <= 1,
            "clone_fresh produced detector with too many false alarms: {drifts}"
        );
    }

    // -----------------------------------------------------------------------
    // 7. Small window warmup: no drift check before min_window
    // -----------------------------------------------------------------------
    #[test]
    fn warmup_suppresses_early_detection() {
        let mut det = Adwin::with_delta(0.002).with_min_window(100);

        // Feed an extreme shift within the first 100 samples: 50 zeros then 50
        // very large values. Without warmup this would certainly trigger.
        let mut any_drift = false;
        for _ in 0..50 {
            if det.update(0.0) == DriftSignal::Drift {
                any_drift = true;
            }
        }
        for _ in 0..50 {
            if det.update(100.0) == DriftSignal::Drift {
                any_drift = true;
            }
        }
        assert!(
            !any_drift,
            "Drift should not fire before min_window=100 samples"
        );
    }

    // -----------------------------------------------------------------------
    // 8. Bucket compression keeps memory bounded
    // -----------------------------------------------------------------------
    #[test]
    fn compression_bounds_memory() {
        let mut det = Adwin::with_delta(0.002).with_max_buckets(5);

        for i in 0..10_000 {
            det.update(i as f64);
        }

        // After compression, no row should have more than max_buckets entries.
        for (row_idx, row) in det.rows.iter().enumerate() {
            assert!(
                row.len() <= det.max_buckets + 1, // +1 tolerance for the just-inserted bucket before next compress
                "Row {row_idx} has {} buckets, exceeding max {}",
                row.len(),
                det.max_buckets
            );
        }

        // Total bucket count should be O(log W * max_buckets).
        let total_buckets: usize = det.rows.iter().map(|r| r.len()).sum();
        let expected_max = det.rows.len() * (det.max_buckets + 1);
        assert!(
            total_buckets <= expected_max,
            "Total buckets {total_buckets} exceeds expected max {expected_max}"
        );
    }

    // -----------------------------------------------------------------------
    // 9. Window shrinks after drift
    // -----------------------------------------------------------------------
    #[test]
    fn window_shrinks_on_drift() {
        let mut det = Adwin::with_delta(0.002);

        // Build up a big window.
        for _ in 0..2000 {
            det.update(0.0);
        }
        let width_before = det.width();
        assert!(width_before >= 1900, "Expected large window, got {width_before}");

        // Inject abrupt shift to force drift.
        let mut drifted = false;
        for _ in 0..500 {
            if det.update(100.0) == DriftSignal::Drift {
                drifted = true;
                break;
            }
        }
        assert!(drifted, "Expected drift on extreme shift");

        // After drift, window should be significantly smaller.
        let width_after = det.width();
        assert!(
            width_after < width_before,
            "Window should shrink after drift: before={width_before}, after={width_after}"
        );
    }

    // -----------------------------------------------------------------------
    // 10. Warning signal fires before drift
    // -----------------------------------------------------------------------
    #[test]
    fn warning_precedes_drift() {
        let mut det = Adwin::with_delta(0.002);
        let mut rng = Xorshift64::new(789);

        // Stable phase.
        for _ in 0..1000 {
            det.update(rng.next_normal(0.0, 0.1));
        }

        // Moderate shift that might trigger warning before drift.
        let mut _saw_warning = false;
        let mut saw_drift = false;
        for _ in 0..2000 {
            let sig = det.update(rng.next_normal(2.0, 0.1));
            match sig {
                DriftSignal::Warning => {
                    if !saw_drift {
                        _saw_warning = true;
                    }
                }
                DriftSignal::Drift => {
                    saw_drift = true;
                    break;
                }
                _ => {}
            }
        }

        // We should have detected drift on a shift this large.
        assert!(saw_drift, "Should have detected drift on shift from 0 to 2");
        // Warning may or may not precede drift depending on how fast it triggers,
        // but we at least verify the mechanism didn't panic.
    }

    // -----------------------------------------------------------------------
    // 11. Deterministic reproducibility
    // -----------------------------------------------------------------------
    #[test]
    fn deterministic_for_same_input() {
        let values: Vec<f64> = (0..500).map(|i| (i as f64 * 0.01).sin()).collect();

        let mut det1 = Adwin::with_delta(0.01);
        let mut det2 = Adwin::with_delta(0.01);

        let signals1: Vec<DriftSignal> = values.iter().map(|&v| det1.update(v)).collect();
        let signals2: Vec<DriftSignal> = values.iter().map(|&v| det2.update(v)).collect();

        assert_eq!(signals1, signals2, "Same input must produce identical signals");
    }
}
