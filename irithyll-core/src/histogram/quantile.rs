//! Greenwald-Khanna streaming quantile sketch for bin edge computation.
//!
//! The GK sketch maintains a compact summary of a data stream that answers
//! quantile queries with bounded error epsilon. It stores tuples (value, gap, delta)
//! where `gap` is the minimum rank difference from the predecessor and `delta` is
//! the maximum rank uncertainty.
//!
//! **Space complexity:** O((1/epsilon) * log(epsilon * N))
//! **Per-insert time:** O(log(1/epsilon) + (1/epsilon)) amortized (binary search + compress)
//! **Query time:** O(1/epsilon) (linear scan of summary)

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::histogram::{BinEdges, BinningStrategy};

/// A single entry in the GK summary.
///
/// Invariant: for any entry `t_i` (not the first), the true rank of `t_i.value`
/// lies in `[rank_min, rank_min + t_i.delta]` where `rank_min` is the sum of
/// all `gap` values from `t_0` through `t_i`.
#[derive(Debug, Clone)]
struct GKTuple {
    value: f64,
    /// g_i: minimum possible rank difference from the predecessor entry.
    gap: u64,
    /// delta_i: maximum uncertainty in the rank of this value.
    delta: u64,
}

/// Greenwald-Khanna streaming quantile sketch.
///
/// Provides epsilon-approximate quantile queries on a data stream using
/// O((1/epsilon) * log(epsilon * N)) space, where N is the number of observed
/// values and epsilon is the error tolerance.
///
/// A query for quantile phi returns a value whose true rank r satisfies
/// `|r - phi*N| <= epsilon * N`.
///
/// # Example
///
/// ```ignore
/// let mut sketch = QuantileBinning::new();
/// for v in 0..1000 {
///     sketch.observe(v as f64);
/// }
/// let edges = sketch.compute_edges(4);
/// // edges are approximately at the 25th, 50th, 75th percentiles
/// ```
#[derive(Debug, Clone)]
pub struct QuantileBinning {
    /// The GK summary: sorted tuples summarizing the rank distribution.
    summary: Vec<GKTuple>,
    /// Total number of values observed.
    count: u64,
    /// Error tolerance. A query for quantile phi returns a value whose
    /// true rank r satisfies |r - phi*N| <= epsilon * N.
    epsilon: f64,
    /// Counter for triggering periodic compression.
    inserts_since_compress: u64,
}

impl QuantileBinning {
    /// Create a new `QuantileBinning` with the default epsilon of 0.01 (1% error).
    pub fn new() -> Self {
        Self::with_epsilon(0.01)
    }

    /// Create a new `QuantileBinning` with a custom error tolerance.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` is not in the range `(0.0, 1.0)`.
    pub fn with_epsilon(epsilon: f64) -> Self {
        assert!(
            epsilon > 0.0 && epsilon < 1.0,
            "epsilon must be in (0.0, 1.0), got {epsilon}"
        );
        Self {
            summary: Vec::new(),
            count: 0,
            epsilon,
            inserts_since_compress: 0,
        }
    }

    /// The capacity threshold: floor(2 * epsilon * count).
    /// This is the maximum allowed value for `gap + delta` in any summary entry
    /// (except the first and last).
    #[inline]
    fn band_width(&self) -> u64 {
        libm::floor(2.0 * self.epsilon * self.count as f64) as u64
    }

    /// How often to trigger compression: every floor(1 / (2 * epsilon)) inserts.
    #[inline]
    fn compress_interval(&self) -> u64 {
        libm::floor(1.0 / (2.0 * self.epsilon)) as u64
    }

    /// Insert a value into the summary.
    fn insert(&mut self, value: f64) {
        self.count += 1;

        if self.summary.is_empty() {
            self.summary.push(GKTuple {
                value,
                gap: 1,
                delta: 0,
            });
            self.inserts_since_compress += 1;
            return;
        }

        // Binary search: find the first entry with value >= new value.
        let pos = self.summary.partition_point(|t| t.value < value);

        // Compute delta for the new tuple.
        let delta = if pos == 0 || pos == self.summary.len() {
            // Inserting at the very front or very back: no uncertainty.
            0
        } else {
            // Interior insertion: delta = floor(2 * epsilon * count) - 1.
            // We subtract 1 because count has already been incremented and the
            // new tuple's gap is 1, so delta = band_width - 1 keeps the invariant.
            self.band_width().saturating_sub(1)
        };

        self.summary.insert(
            pos,
            GKTuple {
                value,
                gap: 1,
                delta,
            },
        );

        self.inserts_since_compress += 1;

        // Periodic compression to keep summary compact.
        let interval = self.compress_interval().max(1);
        if self.inserts_since_compress >= interval {
            self.compress();
            self.inserts_since_compress = 0;
        }
    }

    /// Compress the summary by merging adjacent entries where the merge
    /// would not violate the GK invariant.
    ///
    /// Scans backwards (highest to lowest index) merging entry `i` into
    /// entry `i+1` when `summary[i].gap + summary[i+1].gap + summary[i+1].delta`
    /// does not exceed `floor(2 * epsilon * count)`.
    ///
    /// The first and last entries are never removed.
    fn compress(&mut self) {
        if self.summary.len() < 3 {
            return;
        }

        let threshold = self.band_width();

        // Walk backwards from second-to-last to second entry.
        // We check whether entry i can be merged into entry i+1.
        let mut i = self.summary.len() - 2;
        while i >= 1 {
            let merge_cost =
                self.summary[i].gap + self.summary[i + 1].gap + self.summary[i + 1].delta;

            if merge_cost <= threshold {
                // Absorb entry i into entry i+1: transfer its gap contribution.
                let absorbed_gap = self.summary[i].gap;
                self.summary[i + 1].gap += absorbed_gap;
                self.summary.remove(i);
            }

            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    /// Query the approximate value at quantile `phi` in `[0.0, 1.0]`.
    ///
    /// Returns `None` if the summary is empty.
    ///
    /// The returned value has true rank r such that `|r - phi*N| <= epsilon * N`.
    pub fn quantile(&self, phi: f64) -> Option<f64> {
        if self.summary.is_empty() {
            return None;
        }

        // Clamp phi to [0, 1].
        let phi = phi.clamp(0.0, 1.0);

        let target_rank = libm::ceil(phi * self.count as f64) as u64;
        let tolerance = libm::floor(self.epsilon * self.count as f64) as u64;

        // Walk through summary, tracking cumulative rank (sum of gaps).
        let mut rank: u64 = 0;
        let mut best_idx = 0;

        for (i, tuple) in self.summary.iter().enumerate() {
            rank += tuple.gap;

            // Check whether this entry could satisfy the quantile query.
            // We want the largest i where rank_i + delta_i < target_rank + tolerance,
            // or if we've passed the target, return the best candidate seen so far.
            if rank + tuple.delta > target_rank + tolerance {
                // We've overshot: the previous valid entry is the answer.
                return Some(self.summary[best_idx].value);
            }

            best_idx = i;
        }

        // If we walked all the way through, return the last entry.
        Some(self.summary[best_idx].value)
    }

    /// Return the number of values observed so far.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Return the current number of tuples in the summary.
    #[inline]
    pub fn summary_len(&self) -> usize {
        self.summary.len()
    }
}

impl Default for QuantileBinning {
    fn default() -> Self {
        Self::new()
    }
}

impl BinningStrategy for QuantileBinning {
    /// Observe a single value from the stream.
    fn observe(&mut self, value: f64) {
        self.insert(value);
    }

    /// Compute bin edges by querying quantiles at evenly spaced positions.
    ///
    /// For `n_bins` bins, queries quantiles at `1/n_bins, 2/n_bins, ..., (n_bins-1)/n_bins`
    /// and deduplicates adjacent equal edges. Returns at least one edge when data
    /// has been observed.
    fn compute_edges(&self, n_bins: usize) -> BinEdges {
        if self.summary.is_empty() || n_bins <= 1 {
            return BinEdges { edges: vec![] };
        }

        let mut edges = Vec::with_capacity(n_bins - 1);

        for i in 1..n_bins {
            let phi = i as f64 / n_bins as f64;
            if let Some(val) = self.quantile(phi) {
                // Deduplicate: skip if this edge equals the previous one.
                if edges
                    .last()
                    .map_or(true, |&last: &f64| (val - last).abs() > f64::EPSILON)
                {
                    edges.push(val);
                }
            }
        }

        BinEdges { edges }
    }

    /// Reset the sketch to its initial empty state (preserving epsilon).
    fn reset(&mut self) {
        self.summary.clear();
        self.count = 0;
        self.inserts_since_compress = 0;
    }

    /// Create a fresh empty instance with the same epsilon.
    fn clone_fresh(&self) -> Box<dyn BinningStrategy> {
        Box::new(QuantileBinning::with_epsilon(self.epsilon))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Construction & defaults
    // ---------------------------------------------------------------

    #[test]
    fn new_starts_empty() {
        let q = QuantileBinning::new();
        assert_eq!(q.count(), 0);
        assert_eq!(q.summary_len(), 0);
        assert!((q.epsilon - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn with_epsilon_stores_value() {
        let q = QuantileBinning::with_epsilon(0.05);
        assert!((q.epsilon - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "epsilon must be in (0.0, 1.0)")]
    fn with_epsilon_zero_panics() {
        QuantileBinning::with_epsilon(0.0);
    }

    #[test]
    #[should_panic(expected = "epsilon must be in (0.0, 1.0)")]
    fn with_epsilon_one_panics() {
        QuantileBinning::with_epsilon(1.0);
    }

    #[test]
    #[should_panic(expected = "epsilon must be in (0.0, 1.0)")]
    fn with_epsilon_negative_panics() {
        QuantileBinning::with_epsilon(-0.1);
    }

    // ---------------------------------------------------------------
    // Single value / edge cases
    // ---------------------------------------------------------------

    #[test]
    fn single_value_quantile() {
        let mut q = QuantileBinning::new();
        q.observe(42.0);
        assert_eq!(q.quantile(0.0), Some(42.0));
        assert_eq!(q.quantile(0.5), Some(42.0));
        assert_eq!(q.quantile(1.0), Some(42.0));
    }

    #[test]
    fn empty_state_quantile_returns_none() {
        let q = QuantileBinning::new();
        assert_eq!(q.quantile(0.5), None);
    }

    #[test]
    fn two_values() {
        let mut q = QuantileBinning::new();
        q.observe(10.0);
        q.observe(20.0);
        assert_eq!(q.count(), 2);

        // Median should be one of the two values.
        let med = q.quantile(0.5).unwrap();
        assert!(med == 10.0 || med == 20.0);
    }

    // ---------------------------------------------------------------
    // Sequential insertion: median accuracy
    // ---------------------------------------------------------------

    #[test]
    fn sequential_1000_median_close_to_500() {
        let mut q = QuantileBinning::new(); // epsilon = 0.01
        for v in 0..1000 {
            q.observe(v as f64);
        }

        let median = q.quantile(0.5).unwrap();
        // True median is 500. With epsilon=0.01 and N=1000, error bound is 10.
        assert!(
            (median - 500.0).abs() <= 10.0,
            "median = {median}, expected ~500.0 (within +/-10)"
        );
    }

    #[test]
    fn sequential_1000_quartiles() {
        let mut q = QuantileBinning::new();
        for v in 0..1000 {
            q.observe(v as f64);
        }

        let tolerance = 0.01 * 1000.0; // = 10

        let q25 = q.quantile(0.25).unwrap();
        assert!(
            (q25 - 250.0).abs() <= tolerance,
            "q25 = {q25}, expected ~250 (tol {tolerance})"
        );

        let q75 = q.quantile(0.75).unwrap();
        assert!(
            (q75 - 750.0).abs() <= tolerance,
            "q75 = {q75}, expected ~750 (tol {tolerance})"
        );
    }

    // ---------------------------------------------------------------
    // Reverse insertion order (worst case for some sketches)
    // ---------------------------------------------------------------

    #[test]
    fn reverse_insertion_median() {
        let mut q = QuantileBinning::new();
        for v in (0..1000).rev() {
            q.observe(v as f64);
        }

        let median = q.quantile(0.5).unwrap();
        assert!(
            (median - 500.0).abs() <= 10.0,
            "median = {median}, expected ~500.0 (within +/-10)"
        );
    }

    // ---------------------------------------------------------------
    // Compression effectiveness
    // ---------------------------------------------------------------

    #[test]
    fn compression_keeps_summary_bounded() {
        let mut q = QuantileBinning::new(); // epsilon = 0.01
        for v in 0..10_000 {
            q.observe(v as f64);
        }

        // Theoretical bound: O((1/epsilon) * log(epsilon * N))
        // = O(100 * log(100)) ~ O(100 * 4.6) ~ 460.
        // In practice it should be well under 10000.
        let len = q.summary_len();
        assert!(
            len < 2000,
            "summary length {len} should be much less than 10000"
        );
        // Also verify it's significantly compressed.
        assert!(
            len < 10_000 / 3,
            "summary length {len} should be at most ~1/3 of N"
        );
    }

    #[test]
    fn compression_large_epsilon() {
        let mut q = QuantileBinning::with_epsilon(0.1);
        for v in 0..10_000 {
            q.observe(v as f64);
        }

        // Larger epsilon -> more aggressive compression -> smaller summary.
        let len = q.summary_len();
        assert!(
            len < 500,
            "with epsilon=0.1, summary length {len} should be < 500"
        );
    }

    // ---------------------------------------------------------------
    // compute_edges (BinningStrategy trait)
    // ---------------------------------------------------------------

    #[test]
    fn compute_edges_4_bins_roughly_quartiles() {
        let mut q = QuantileBinning::new();
        for v in 0..1000 {
            q.observe(v as f64);
        }

        let edges = q.compute_edges(4);
        // Should have ~3 edges (quartile boundaries) after dedup.
        assert!(!edges.edges.is_empty(), "should produce at least one edge");
        assert!(
            edges.edges.len() <= 3,
            "4 bins -> at most 3 edges, got {}",
            edges.edges.len()
        );

        let tolerance = 0.01 * 1000.0;

        if edges.edges.len() == 3 {
            assert!(
                (edges.edges[0] - 250.0).abs() <= tolerance,
                "first edge {} should be ~250",
                edges.edges[0]
            );
            assert!(
                (edges.edges[1] - 500.0).abs() <= tolerance,
                "second edge {} should be ~500",
                edges.edges[1]
            );
            assert!(
                (edges.edges[2] - 750.0).abs() <= tolerance,
                "third edge {} should be ~750",
                edges.edges[2]
            );
        }
    }

    #[test]
    fn compute_edges_sorted() {
        let mut q = QuantileBinning::new();
        for v in 0..500 {
            q.observe(v as f64);
        }

        let edges = q.compute_edges(10);
        for i in 1..edges.edges.len() {
            assert!(
                edges.edges[i] > edges.edges[i - 1],
                "edges must be strictly increasing: edges[{}]={} <= edges[{}]={}",
                i,
                edges.edges[i],
                i - 1,
                edges.edges[i - 1]
            );
        }
    }

    #[test]
    fn compute_edges_empty_sketch() {
        let q = QuantileBinning::new();
        let edges = q.compute_edges(4);
        assert!(edges.edges.is_empty());
    }

    #[test]
    fn compute_edges_single_bin() {
        let mut q = QuantileBinning::new();
        q.observe(1.0);
        q.observe(2.0);
        let edges = q.compute_edges(1);
        assert!(edges.edges.is_empty(), "1 bin means 0 edges");
    }

    // ---------------------------------------------------------------
    // Deduplication in compute_edges
    // ---------------------------------------------------------------

    #[test]
    fn compute_edges_deduplicates_identical_values() {
        let mut q = QuantileBinning::new();
        // All identical values: every quantile query returns the same value.
        for _ in 0..1000 {
            q.observe(7.0);
        }

        let edges = q.compute_edges(4);
        // Should be deduplicated to at most 1 unique edge.
        assert!(
            edges.edges.len() <= 1,
            "all-same input should dedup to <=1 edge, got {}",
            edges.edges.len()
        );
    }

    #[test]
    fn compute_edges_deduplicates_two_clusters() {
        let mut q = QuantileBinning::new();
        // Two clusters: 500 values of 1.0, then 500 values of 100.0.
        for _ in 0..500 {
            q.observe(1.0);
        }
        for _ in 0..500 {
            q.observe(100.0);
        }

        let edges = q.compute_edges(10);
        // With heavy dedup, we might get very few unique edges.
        // The key test: no adjacent duplicates.
        for i in 1..edges.edges.len() {
            assert!(
                (edges.edges[i] - edges.edges[i - 1]).abs() > f64::EPSILON,
                "adjacent edges should not be equal"
            );
        }
    }

    // ---------------------------------------------------------------
    // Error bound with epsilon = 0.05
    // ---------------------------------------------------------------

    #[test]
    fn error_within_epsilon_005() {
        let mut q = QuantileBinning::with_epsilon(0.05);
        let n = 2000u64;
        for v in 0..n {
            q.observe(v as f64);
        }

        let tolerance = 0.05 * n as f64; // = 100

        // Check several quantiles.
        for &phi in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let result = q.quantile(phi).unwrap();
            let expected = phi * n as f64;
            assert!(
                (result - expected).abs() <= tolerance,
                "phi={phi}: result={result}, expected~{expected}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn error_within_epsilon_001_large_n() {
        let mut q = QuantileBinning::with_epsilon(0.01);
        let n = 5000u64;
        for v in 0..n {
            q.observe(v as f64);
        }

        let tolerance = 0.01 * n as f64; // = 50

        for &phi in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let result = q.quantile(phi).unwrap();
            let expected = phi * n as f64;
            assert!(
                (result - expected).abs() <= tolerance,
                "phi={phi}: result={result}, expected~{expected}, tolerance={tolerance}"
            );
        }
    }

    // ---------------------------------------------------------------
    // Extremes: quantile(0) and quantile(1)
    // ---------------------------------------------------------------

    #[test]
    fn quantile_zero_returns_minimum() {
        let mut q = QuantileBinning::new();
        for v in 10..110 {
            q.observe(v as f64);
        }
        let min_approx = q.quantile(0.0).unwrap();
        // Should be the actual minimum or very close.
        assert!(
            (min_approx - 10.0).abs() <= 1.0,
            "quantile(0.0) = {min_approx}, expected ~10.0"
        );
    }

    #[test]
    fn quantile_one_returns_maximum() {
        let mut q = QuantileBinning::new();
        for v in 10..110 {
            q.observe(v as f64);
        }
        let max_approx = q.quantile(1.0).unwrap();
        // Should be the actual maximum or very close.
        assert!(
            (max_approx - 109.0).abs() <= 1.0,
            "quantile(1.0) = {max_approx}, expected ~109.0"
        );
    }

    // ---------------------------------------------------------------
    // reset / clone_fresh
    // ---------------------------------------------------------------

    #[test]
    fn reset_clears_state() {
        let mut q = QuantileBinning::new();
        for v in 0..100 {
            q.observe(v as f64);
        }
        assert_eq!(q.count(), 100);
        assert!(q.summary_len() > 0);

        q.reset();
        assert_eq!(q.count(), 0);
        assert_eq!(q.summary_len(), 0);
        assert_eq!(q.quantile(0.5), None);
    }

    #[test]
    fn clone_fresh_preserves_epsilon() {
        let mut q = QuantileBinning::with_epsilon(0.05);
        for v in 0..100 {
            q.observe(v as f64);
        }

        let fresh = q.clone_fresh();
        let edges = fresh.compute_edges(4);
        assert!(edges.edges.is_empty(), "fresh instance should be empty");
    }

    // ---------------------------------------------------------------
    // find_bin integration
    // ---------------------------------------------------------------

    #[test]
    fn find_bin_with_quantile_edges() {
        let mut q = QuantileBinning::new();
        for v in 0..1000 {
            q.observe(v as f64);
        }
        let edges = q.compute_edges(4);

        // Values below the first edge go into bin 0.
        assert_eq!(edges.find_bin(0.0), 0);

        // Values above the last edge go into the last bin.
        assert_eq!(edges.find_bin(999.0), edges.n_bins() - 1);
    }

    // ---------------------------------------------------------------
    // Shuffled / random-ish input
    // ---------------------------------------------------------------

    #[test]
    fn shuffled_input_median() {
        let mut q = QuantileBinning::new();

        // Simple deterministic pseudo-shuffle using a stride coprime to N.
        let n = 1000u64;
        let stride = 397; // coprime to 1000
        let mut val = 0u64;
        for _ in 0..n {
            q.observe(val as f64);
            val = (val + stride) % n;
        }

        let median = q.quantile(0.5).unwrap();
        let tolerance = 0.01 * n as f64;
        assert!(
            (median - 500.0).abs() <= tolerance,
            "shuffled median = {median}, expected ~500 (tol {tolerance})"
        );
    }

    // ---------------------------------------------------------------
    // Negative values
    // ---------------------------------------------------------------

    #[test]
    fn negative_range_quantiles() {
        let mut q = QuantileBinning::new();
        for v in -500..500 {
            q.observe(v as f64);
        }

        let median = q.quantile(0.5).unwrap();
        let tolerance = 0.01 * 1000.0;
        assert!(
            median.abs() <= tolerance,
            "median = {median}, expected ~0.0 (tol {tolerance})"
        );
    }

    // ---------------------------------------------------------------
    // Floating point values
    // ---------------------------------------------------------------

    #[test]
    fn floating_point_values() {
        let mut q = QuantileBinning::new();
        for i in 0..1000 {
            q.observe(i as f64 * 0.001); // 0.000, 0.001, ..., 0.999
        }

        let median = q.quantile(0.5).unwrap();
        let tolerance = 0.01 * 1.0; // epsilon * range
        assert!(
            (median - 0.5).abs() <= tolerance,
            "median = {median}, expected ~0.5 (tol {tolerance})"
        );
    }

    // ---------------------------------------------------------------
    // Many bins
    // ---------------------------------------------------------------

    #[test]
    fn many_bins_fine_grained() {
        // Use a small epsilon so the sketch has enough resolution for 100 bins.
        // With epsilon=0.001 and N=10000, error bound is 10, so quantile queries
        // spaced 100 apart (for 100 bins over range 10000) are well-separated.
        let mut q = QuantileBinning::with_epsilon(0.001);
        for v in 0..10_000 {
            q.observe(v as f64);
        }

        let edges = q.compute_edges(100);
        // Should produce close to 99 edges. With epsilon=0.001 and 10000 distinct
        // values spaced 100 apart per bin, dedup should be minimal.
        assert!(
            edges.edges.len() >= 90,
            "100 bins with 10000 distinct values (eps=0.001) should give ~99 edges, got {}",
            edges.edges.len()
        );

        // All edges should be strictly increasing.
        for i in 1..edges.edges.len() {
            assert!(
                edges.edges[i] > edges.edges[i - 1],
                "edges must be strictly increasing"
            );
        }
    }
}
