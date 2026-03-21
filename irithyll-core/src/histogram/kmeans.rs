//! K-means binning strategy (Labovich 2025).
//!
//! Maximizes worst-case explained variance under Lipschitz assumptions,
//! yielding up to 55% MSE reduction on skewed data compared to quantile
//! binning, and is never statistically worse. Uses online k-means clustering
//! over a reservoir sample to find bin centers that minimize within-bin
//! variance; bin edges are midpoints between adjacent sorted cluster centers.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::histogram::{BinEdges, BinningStrategy};

/// Default maximum number of values retained in the reservoir.
const DEFAULT_MAX_RESERVOIR: usize = 10_000;

/// Default number of Lloyd's iterations for k-means refinement.
const DEFAULT_MAX_ITERS: usize = 50;

/// Convergence tolerance: if no center moves more than this, stop early.
const EPSILON: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Xorshift64 PRNG -- lightweight, deterministic, no allocation.
// ---------------------------------------------------------------------------

/// Advance an xorshift64 state and return the new value.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

// ---------------------------------------------------------------------------
// KMeansBinning
// ---------------------------------------------------------------------------

/// K-means binning strategy: uses online k-means clustering to find
/// bin centers that minimize within-bin variance. Bin edges are midpoints
/// between adjacent sorted cluster centers.
///
/// Particularly effective for skewed distributions where equal-width
/// and even quantile binning waste bins on sparse regions.
///
/// # Algorithm
///
/// 1. **Observe:** Values are collected into a fixed-capacity reservoir via
///    Vitter's Algorithm R (reservoir sampling), so memory is bounded
///    regardless of stream length.
/// 2. **Compute edges:** Cluster centers are initialized at quantile
///    positions, then refined with Lloyd's algorithm. Bin edges are the
///    midpoints between adjacent sorted centers.
///
/// # Example
///
/// ```ignore
/// use irithyll::histogram::kmeans::KMeansBinning;
/// use irithyll::histogram::BinningStrategy;
///
/// let mut binner = KMeansBinning::new();
/// for i in 0..5000 {
///     binner.observe((i as f64).sin() * 10.0);
/// }
/// let edges = binner.compute_edges(8);
/// assert!(edges.edges.len() <= 7);
/// ```
#[derive(Debug, Clone)]
pub struct KMeansBinning {
    /// Reservoir of observed values (capped for memory).
    reservoir: Vec<f64>,
    /// Maximum reservoir size.
    max_reservoir: usize,
    /// Number of Lloyd's iterations for refinement.
    max_iters: usize,
    /// Total values observed (including those discarded by sampling).
    count: u64,
    /// Xorshift64 PRNG state for reservoir sampling.
    rng_state: u64,
}

impl KMeansBinning {
    /// Create a new `KMeansBinning` with default parameters
    /// (reservoir = 10 000, max iterations = 50).
    pub fn new() -> Self {
        Self::with_params(DEFAULT_MAX_RESERVOIR, DEFAULT_MAX_ITERS)
    }

    /// Create a `KMeansBinning` with explicit reservoir capacity and iteration
    /// limit.
    ///
    /// # Panics
    ///
    /// Panics if `max_reservoir` is 0 or `max_iters` is 0.
    pub fn with_params(max_reservoir: usize, max_iters: usize) -> Self {
        assert!(max_reservoir > 0, "max_reservoir must be > 0");
        assert!(max_iters > 0, "max_iters must be > 0");
        Self {
            reservoir: Vec::with_capacity(max_reservoir.min(1024)),
            max_reservoir,
            max_iters,
            count: 0,
            // Seed with a non-zero constant; any nonzero value works for xorshift64.
            rng_state: 0xDEAD_BEEF_CAFE_BABE,
        }
    }

    /// Return the current reservoir size (for testing / diagnostics).
    #[inline]
    pub fn reservoir_len(&self) -> usize {
        self.reservoir.len()
    }

    /// Return the total number of values observed.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for KMeansBinning {
    fn default() -> Self {
        Self::new()
    }
}

impl BinningStrategy for KMeansBinning {
    /// Observe a single value from the stream.
    ///
    /// If the reservoir is not yet full, the value is appended directly.
    /// Once full, Vitter's Algorithm R is used: with probability
    /// `max_reservoir / count`, a uniformly random element is replaced.
    fn observe(&mut self, value: f64) {
        self.count += 1;

        if self.reservoir.len() < self.max_reservoir {
            self.reservoir.push(value);
        } else {
            // Reservoir sampling: replace element at random index with
            // probability max_reservoir / count.
            let r = xorshift64(&mut self.rng_state) % self.count;
            if (r as usize) < self.max_reservoir {
                let idx = r as usize;
                self.reservoir[idx] = value;
            }
        }
    }

    /// Compute bin edges from the reservoir using k-means (Lloyd's algorithm).
    ///
    /// Returns `BinEdges` with at most `n_bins - 1` edges. When the reservoir
    /// contains fewer distinct values than requested bins, a smaller number of
    /// edges is returned gracefully.
    fn compute_edges(&self, n_bins: usize) -> BinEdges {
        if n_bins == 0 {
            return BinEdges { edges: Vec::new() };
        }

        // Degenerate: nothing observed yet.
        if self.reservoir.is_empty() {
            return BinEdges { edges: Vec::new() };
        }

        // Sort a working copy of the reservoir.
        let mut sorted: Vec<f64> = self.reservoir.clone();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        // Deduplicate to unique values.
        sorted.dedup();

        let n_unique = sorted.len();

        // If fewer unique values than bins, use unique values as edges directly
        // (the midpoints between consecutive unique values).
        if n_unique <= n_bins {
            if n_unique <= 1 {
                return BinEdges { edges: Vec::new() };
            }
            let edges: Vec<f64> = sorted.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
            return BinEdges { edges };
        }

        // --- Quantile initialization for k = n_bins centers ---
        let k = n_bins;
        let data = &self.reservoir; // use full (unsorted-deduped) reservoir for clustering
        let n = data.len();

        // Sorted copy for quantile picking (with duplicates preserved for
        // correct quantile positions).
        let mut sorted_full: Vec<f64> = data.to_vec();
        sorted_full.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let mut centers: Vec<f64> = (0..k)
            .map(|i| {
                let q = (i as f64 + 0.5) / k as f64;
                let idx_f = q * (n - 1) as f64;
                let lo = libm::floor(idx_f) as usize;
                let hi = (lo + 1).min(n - 1);
                let frac = idx_f - lo as f64;
                sorted_full[lo] * (1.0 - frac) + sorted_full[hi] * frac
            })
            .collect();

        // Pre-allocate assignment buffers (reused across iterations).
        let mut sums = vec![0.0_f64; k];
        let mut counts = vec![0_usize; k];

        // --- Lloyd's iterations ---
        for _ in 0..self.max_iters {
            // Reset accumulators.
            for j in 0..k {
                sums[j] = 0.0;
                counts[j] = 0;
            }

            // Assign each point to nearest center, accumulate sum and count.
            for &x in data.iter() {
                let mut best_j = 0;
                let mut best_dist = (x - centers[0]).abs();
                for (j, &center) in centers.iter().enumerate().skip(1) {
                    let d = (x - center).abs();
                    if d < best_dist {
                        best_dist = d;
                        best_j = j;
                    }
                }
                sums[best_j] += x;
                counts[best_j] += 1;
            }

            // Recompute centers and track maximum movement.
            let mut max_move: f64 = 0.0;
            for j in 0..k {
                if counts[j] > 0 {
                    let new_center = sums[j] / counts[j] as f64;
                    let move_dist = (new_center - centers[j]).abs();
                    if move_dist > max_move {
                        max_move = move_dist;
                    }
                    centers[j] = new_center;
                }
                // If a center has no assigned points, leave it unchanged.
            }

            // Early convergence check.
            if max_move <= EPSILON {
                break;
            }
        }

        // Sort centers and compute midpoint edges.
        centers.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        // Deduplicate centers (may happen if initial quantiles coincide).
        centers.dedup_by(|a, b| (*a - *b).abs() <= EPSILON);

        if centers.len() <= 1 {
            return BinEdges { edges: Vec::new() };
        }

        let mut edges: Vec<f64> = centers.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

        // Deduplicate edges (possible with very tight center clusters).
        edges.dedup_by(|a, b| (*a - *b).abs() <= EPSILON);

        BinEdges { edges }
    }

    /// Reset observed state, clearing reservoir and count.
    fn reset(&mut self) {
        self.reservoir.clear();
        self.count = 0;
        self.rng_state = 0xDEAD_BEEF_CAFE_BABE;
    }

    /// Create a fresh instance with the same configuration but no data.
    fn clone_fresh(&self) -> Box<dyn BinningStrategy> {
        Box::new(KMeansBinning::with_params(
            self.max_reservoir,
            self.max_iters,
        ))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: check that edges are strictly sorted.
    fn assert_sorted(edges: &[f64]) {
        for i in 1..edges.len() {
            assert!(
                edges[i] > edges[i - 1],
                "edges not strictly sorted: edges[{}]={} <= edges[{}]={}",
                i,
                edges[i],
                i - 1,
                edges[i - 1],
            );
        }
    }

    // ----- 1. Standard normal-ish data, 8 bins, edges are sorted ----------

    #[test]
    fn normal_ish_data_edges_sorted() {
        let mut binner = KMeansBinning::new();

        // Deterministic pseudo-normal: sum of 4 uniform-ish values via simple LCG.
        let mut lcg: u64 = 12345;
        for _ in 0..5000 {
            let mut sum = 0.0_f64;
            for _ in 0..4 {
                lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (lcg >> 33) as f64 / (1u64 << 31) as f64; // [0, 1)
                sum += u;
            }
            // sum is in [0, 4), shift to [-2, 2) -- roughly bell-shaped.
            binner.observe(sum - 2.0);
        }

        let edges = binner.compute_edges(8);
        assert!(!edges.edges.is_empty());
        assert!(edges.edges.len() <= 7, "at most n_bins-1 edges");
        assert_sorted(&edges.edges);
    }

    // ----- 2. Skewed data: k-means concentrates where data is dense -------

    #[test]
    fn skewed_data_denser_bins_at_low_end() {
        let mut binner = KMeansBinning::new();

        // Exponential-like: i*i for i in 0..1000 gives values 0..999_001
        // Most data is clustered near 0.
        for i in 0..1000u64 {
            binner.observe((i * i) as f64);
        }

        let edges = binner.compute_edges(8);
        assert_sorted(&edges.edges);

        // The first bin edge should be much closer to 0 than to the max,
        // because k-means follows data density. With equal-width bins the
        // first edge would be at ~124875. K-means should place it well below
        // that -- typically under 60% of the equal-width edge.
        let max_val = 999.0 * 999.0;
        let equal_width_first = max_val / 8.0;

        assert!(
            edges.edges[0] < equal_width_first * 0.6,
            "first k-means edge {} should be well below equal-width edge {} (60% threshold = {})",
            edges.edges[0],
            equal_width_first,
            equal_width_first * 0.6,
        );
    }

    // ----- 3. Uniform data: edges roughly equal-width ----------------------

    #[test]
    fn uniform_data_roughly_equal_width() {
        let mut binner = KMeansBinning::new();

        for i in 0..2000 {
            binner.observe(i as f64);
        }

        let edges = binner.compute_edges(5);
        assert_sorted(&edges.edges);
        assert_eq!(edges.edges.len(), 4, "5 bins -> 4 edges");

        // With 5 bins over [0, 1999], ideal edges near 400, 800, 1200, 1600.
        // Allow generous tolerance (10% of range = 200).
        let expected = [400.0, 800.0, 1200.0, 1600.0];
        for (got, want) in edges.edges.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 200.0,
                "edge {} too far from expected {} for uniform data",
                got,
                want,
            );
        }

        // Also check that spacing is roughly even: max gap / min gap < 2.0.
        let gaps: Vec<f64> = core::iter::once(edges.edges[0])
            .chain(edges.edges.windows(2).map(|w| w[1] - w[0]))
            .collect();
        let min_gap = gaps.iter().cloned().fold(f64::MAX, f64::min);
        let max_gap = gaps.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            max_gap / min_gap < 2.0,
            "gaps should be roughly even for uniform data, got min={} max={}",
            min_gap,
            max_gap,
        );
    }

    // ----- 4. Reservoir capping --------------------------------------------

    #[test]
    fn reservoir_capped_at_max() {
        let cap = 500;
        let mut binner = KMeansBinning::with_params(cap, 20);

        for i in 0..100_000u64 {
            binner.observe(i as f64);
        }

        assert_eq!(binner.count(), 100_000);
        assert!(
            binner.reservoir_len() <= cap,
            "reservoir {} exceeds cap {}",
            binner.reservoir_len(),
            cap,
        );
        assert_eq!(binner.reservoir_len(), cap);
    }

    // ----- 5. Small data: fewer samples than bins --------------------------

    #[test]
    fn fewer_samples_than_bins() {
        let mut binner = KMeansBinning::new();
        binner.observe(1.0);
        binner.observe(3.0);
        binner.observe(5.0);

        // Request 10 bins but only 3 unique values.
        let edges = binner.compute_edges(10);

        // Should get at most 2 edges (midpoints between 3 unique values).
        assert!(
            edges.edges.len() <= 2,
            "expected at most 2 edges, got {}",
            edges.edges.len(),
        );
        assert_sorted(&edges.edges);

        // Edges should be midpoints: 2.0, 4.0.
        if edges.edges.len() == 2 {
            assert!((edges.edges[0] - 2.0).abs() < 1e-10);
            assert!((edges.edges[1] - 4.0).abs() < 1e-10);
        }
    }

    // ----- 6. Single value repeated ----------------------------------------

    #[test]
    fn single_value_repeated() {
        let mut binner = KMeansBinning::new();
        for _ in 0..1000 {
            binner.observe(42.0);
        }

        let edges = binner.compute_edges(8);
        // All values identical -> 1 unique -> no meaningful edges.
        assert!(
            edges.edges.is_empty(),
            "expected no edges for constant data, got {:?}",
            edges.edges,
        );
    }

    // ----- 7. Convergence: centers stabilize within max_iters --------------

    #[test]
    fn convergence_within_iters() {
        // Use a very small max_iters and verify results are still reasonable.
        let mut binner_fast = KMeansBinning::with_params(5000, 3);
        let mut binner_full = KMeansBinning::with_params(5000, 200);

        // Same deterministic data for both.
        let mut lcg: u64 = 99999;
        let data: Vec<f64> = (0..3000)
            .map(|_| {
                lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
                (lcg >> 33) as f64 / (1u64 << 31) as f64
            })
            .collect();

        for &x in &data {
            binner_fast.observe(x);
            binner_full.observe(x);
        }

        let edges_fast = binner_fast.compute_edges(6);
        let edges_full = binner_full.compute_edges(6);

        // Both should produce valid sorted edges.
        assert_sorted(&edges_fast.edges);
        assert_sorted(&edges_full.edges);

        // Same number of edges.
        assert_eq!(edges_fast.edges.len(), edges_full.edges.len());

        // With uniform-ish data and quantile init, even 3 iterations should
        // give edges reasonably close to the fully-converged solution.
        for (a, b) in edges_fast.edges.iter().zip(edges_full.edges.iter()) {
            let range = 1.0; // data in [0, 1)
            let tolerance = range * 0.1; // 10% of range
            assert!(
                (a - b).abs() < tolerance,
                "3-iter edge {} too far from converged edge {} (tol {})",
                a,
                b,
                tolerance,
            );
        }
    }

    // ----- Additional edge-case tests --------------------------------------

    #[test]
    fn empty_binner_returns_no_edges() {
        let binner = KMeansBinning::new();
        let edges = binner.compute_edges(8);
        assert!(edges.edges.is_empty());
    }

    #[test]
    fn zero_bins_returns_empty() {
        let mut binner = KMeansBinning::new();
        binner.observe(1.0);
        binner.observe(2.0);
        let edges = binner.compute_edges(0);
        assert!(edges.edges.is_empty());
    }

    #[test]
    fn one_bin_returns_no_edges() {
        let mut binner = KMeansBinning::new();
        for i in 0..100 {
            binner.observe(i as f64);
        }
        let edges = binner.compute_edges(1);
        // 1 bin means 0 internal edges (but k-means with k=1 gives a single
        // center, so no midpoints).
        assert!(edges.edges.is_empty());
    }

    #[test]
    fn two_values_two_bins() {
        let mut binner = KMeansBinning::new();
        binner.observe(0.0);
        binner.observe(10.0);

        let edges = binner.compute_edges(2);
        assert_eq!(edges.edges.len(), 1);
        assert!((edges.edges[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn reset_clears_all_state() {
        let mut binner = KMeansBinning::with_params(100, 10);
        for i in 0..200 {
            binner.observe(i as f64);
        }
        assert_eq!(binner.count(), 200);
        assert_eq!(binner.reservoir_len(), 100);

        binner.reset();
        assert_eq!(binner.count(), 0);
        assert_eq!(binner.reservoir_len(), 0);

        // After reset, should behave like fresh.
        let edges = binner.compute_edges(4);
        assert!(edges.edges.is_empty());
    }

    #[test]
    fn clone_fresh_preserves_params() {
        let mut binner = KMeansBinning::with_params(256, 25);
        for i in 0..500 {
            binner.observe(i as f64);
        }

        let fresh = binner.clone_fresh();
        let edges = fresh.compute_edges(4);
        // Fresh has no data, so no edges.
        assert!(edges.edges.is_empty());
    }

    #[test]
    fn bimodal_data_finds_two_clusters() {
        let mut binner = KMeansBinning::new();

        // Two tight clusters around 10.0 and 90.0.
        let mut lcg: u64 = 77777;
        for _ in 0..2500 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((lcg >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
            binner.observe(10.0 + noise);
        }
        for _ in 0..2500 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((lcg >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
            binner.observe(90.0 + noise);
        }

        let edges = binner.compute_edges(4);
        assert_sorted(&edges.edges);

        // With bimodal data at ~10 and ~90, there should be an edge somewhere
        // in the middle gap (between ~11 and ~89). At least one edge should
        // fall in [20, 80].
        let has_gap_edge = edges.edges.iter().any(|&e| e > 20.0 && e < 80.0);
        assert!(
            has_gap_edge,
            "expected at least one edge in the gap between clusters, edges={:?}",
            edges.edges,
        );
    }

    #[test]
    fn find_bin_integration() {
        let mut binner = KMeansBinning::new();
        for i in 0..1000 {
            binner.observe(i as f64);
        }

        let edges = binner.compute_edges(5);
        assert_sorted(&edges.edges);

        // Every bin index must be in [0, n_bins).
        let n_bins = edges.n_bins();
        for i in 0..1000 {
            let bin = edges.find_bin(i as f64);
            assert!(
                bin < n_bins,
                "find_bin({}) = {} but n_bins = {}",
                i,
                bin,
                n_bins,
            );
        }
    }
}
