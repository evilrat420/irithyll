//! Equal-width (uniform) binning strategy.
//!
//! The simplest binning approach: observe the data range (min, max), then
//! divide it into `n_bins` equal-width intervals. Fast and deterministic,
//! but can produce poor splits when data is heavily skewed. Serves as the
//! baseline strategy and fallback when quantile/k-means binning is unnecessary.

use crate::histogram::{BinEdges, BinningStrategy};

/// Equal-width binning: observes min/max, divides range into equal bins.
///
/// # Example
///
/// ```ignore
/// let mut binner = UniformBinning::new();
/// for &v in &[1.0, 3.0, 5.0, 7.0, 9.0] {
///     binner.observe(v);
/// }
/// let edges = binner.compute_edges(4);
/// // edges.edges == [3.0, 5.0, 7.0]  (dividing [1.0, 9.0] into 4 bins)
/// ```
#[derive(Debug, Clone)]
pub struct UniformBinning {
    min: f64,
    max: f64,
    count: u64,
}

impl UniformBinning {
    /// Create a new `UniformBinning` in its initial (empty) state.
    pub fn new() -> Self {
        Self {
            min: f64::MAX,
            max: f64::MIN,
            count: 0,
        }
    }
}

impl Default for UniformBinning {
    fn default() -> Self {
        Self::new()
    }
}

impl BinningStrategy for UniformBinning {
    /// Observe a single value, updating the running min/max.
    fn observe(&mut self, value: f64) {
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
        self.count += 1;
    }

    /// Compute bin edges by dividing `[min, max]` into `n_bins` equal-width intervals.
    ///
    /// Returns `n_bins - 1` internal boundary values. If fewer than 2 values have
    /// been observed or all values are identical, returns a single edge at min
    /// (producing 2 bins with all data in one of them), which is the degenerate
    /// but safe fallback.
    fn compute_edges(&self, n_bins: usize) -> BinEdges {
        // Degenerate cases: not enough data to define a range
        if self.count < 2 || (self.max - self.min).abs() < f64::EPSILON {
            return BinEdges {
                edges: vec![self.min],
            };
        }

        let range = self.max - self.min;
        let bin_width = range / n_bins as f64;
        let n_edges = n_bins.saturating_sub(1);

        let edges: Vec<f64> = (1..=n_edges)
            .map(|i| self.min + bin_width * i as f64)
            .collect();

        BinEdges { edges }
    }

    /// Reset to the initial empty state.
    fn reset(&mut self) {
        self.min = f64::MAX;
        self.max = f64::MIN;
        self.count = 0;
    }

    /// Create a fresh (empty) instance.
    fn clone_fresh(&self) -> Box<dyn BinningStrategy> {
        Box::new(UniformBinning::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_empty() {
        let b = UniformBinning::new();
        assert_eq!(b.count, 0);
        assert_eq!(b.min, f64::MAX);
        assert_eq!(b.max, f64::MIN);
    }

    #[test]
    fn observe_updates_min_max() {
        let mut b = UniformBinning::new();
        b.observe(5.0);
        assert_eq!(b.min, 5.0);
        assert_eq!(b.max, 5.0);
        assert_eq!(b.count, 1);

        b.observe(2.0);
        assert_eq!(b.min, 2.0);
        assert_eq!(b.max, 5.0);

        b.observe(8.0);
        assert_eq!(b.min, 2.0);
        assert_eq!(b.max, 8.0);
        assert_eq!(b.count, 3);
    }

    #[test]
    fn edges_evenly_spaced() {
        let mut b = UniformBinning::new();
        for &v in &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0] {
            b.observe(v);
        }

        // 5 bins over [0.0, 10.0] -> width = 2.0, edges at 2.0, 4.0, 6.0, 8.0
        let edges = b.compute_edges(5);
        assert_eq!(edges.edges.len(), 4);
        assert_eq!(edges.n_bins(), 5);

        let expected = [2.0, 4.0, 6.0, 8.0];
        for (got, want) in edges.edges.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "edge {got} != expected {want}"
            );
        }

        // Verify equal spacing
        let width = edges.edges[1] - edges.edges[0];
        for i in 1..edges.edges.len() {
            let gap = edges.edges[i] - edges.edges[i - 1];
            assert!(
                (gap - width).abs() < 1e-12,
                "uneven spacing: gap {gap} != width {width}"
            );
        }
    }

    #[test]
    fn edges_two_bins() {
        let mut b = UniformBinning::new();
        b.observe(0.0);
        b.observe(10.0);

        // 2 bins -> 1 edge at midpoint
        let edges = b.compute_edges(2);
        assert_eq!(edges.edges.len(), 1);
        assert!((edges.edges[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn edges_single_value_degenerate() {
        let mut b = UniformBinning::new();
        b.observe(42.0);
        b.observe(42.0);
        b.observe(42.0);

        let edges = b.compute_edges(4);
        // All values identical -> degenerate fallback: single edge at min
        assert_eq!(edges.edges.len(), 1);
        assert!((edges.edges[0] - 42.0).abs() < 1e-12);
        assert_eq!(edges.n_bins(), 2);
    }

    #[test]
    fn edges_too_few_observations() {
        let mut b = UniformBinning::new();
        b.observe(5.0);

        // Only 1 observation -> degenerate
        let edges = b.compute_edges(4);
        assert_eq!(edges.edges.len(), 1);
        assert!((edges.edges[0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn edges_empty_state() {
        let b = UniformBinning::new();
        let edges = b.compute_edges(4);
        // count == 0 -> degenerate, single edge at f64::MAX
        assert_eq!(edges.edges.len(), 1);
        assert_eq!(edges.n_bins(), 2);
    }

    #[test]
    fn reset_clears_state() {
        let mut b = UniformBinning::new();
        b.observe(1.0);
        b.observe(10.0);
        assert_eq!(b.count, 2);

        b.reset();
        assert_eq!(b.count, 0);
        assert_eq!(b.min, f64::MAX);
        assert_eq!(b.max, f64::MIN);
    }

    #[test]
    fn clone_fresh_returns_empty() {
        let mut b = UniformBinning::new();
        b.observe(1.0);
        b.observe(10.0);

        let fresh = b.clone_fresh();
        // Fresh should produce degenerate edges (empty state)
        let edges = fresh.compute_edges(4);
        assert_eq!(edges.edges.len(), 1);
    }

    #[test]
    fn many_bins_fine_grained() {
        let mut b = UniformBinning::new();
        b.observe(0.0);
        b.observe(100.0);

        let edges = b.compute_edges(100);
        assert_eq!(edges.edges.len(), 99);
        assert_eq!(edges.n_bins(), 100);

        // Each edge should be at 1.0, 2.0, ..., 99.0
        for (i, &e) in edges.edges.iter().enumerate() {
            let expected = (i + 1) as f64;
            assert!(
                (e - expected).abs() < 1e-10,
                "edge[{i}] = {e}, expected {expected}"
            );
        }
    }

    #[test]
    fn negative_range() {
        let mut b = UniformBinning::new();
        b.observe(-10.0);
        b.observe(-2.0);

        // 4 bins over [-10.0, -2.0], width = 2.0, edges at -8.0, -6.0, -4.0
        let edges = b.compute_edges(4);
        assert_eq!(edges.edges.len(), 3);

        let expected = [-8.0, -6.0, -4.0];
        for (got, want) in edges.edges.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-12,
                "edge {got} != expected {want}"
            );
        }
    }

    #[test]
    fn single_bin_no_edges() {
        let mut b = UniformBinning::new();
        b.observe(0.0);
        b.observe(10.0);

        // 1 bin -> 0 internal edges
        let edges = b.compute_edges(1);
        assert_eq!(edges.edges.len(), 0);
        assert_eq!(edges.n_bins(), 1);
    }

    #[test]
    fn find_bin_with_uniform_edges() {
        let mut b = UniformBinning::new();
        for &v in &[0.0, 10.0] {
            b.observe(v);
        }
        let edges = b.compute_edges(5);
        // edges at 2.0, 4.0, 6.0, 8.0

        assert_eq!(edges.find_bin(0.0), 0); // below first edge
        assert_eq!(edges.find_bin(1.9), 0); // still in first bin
        assert_eq!(edges.find_bin(2.0), 1); // exact edge -> next bin
        assert_eq!(edges.find_bin(3.0), 1); // between 2.0 and 4.0
        assert_eq!(edges.find_bin(6.0), 3); // exact edge at 6.0 -> bin 3
        assert_eq!(edges.find_bin(9.0), 4); // above last edge
        assert_eq!(edges.find_bin(10.0), 4); // at max
    }
}
