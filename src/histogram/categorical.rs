//! Categorical feature binning -- one bin per observed category value.
//!
//! Unlike continuous binning strategies that discretize a range, categorical
//! binning assigns each distinct category value its own bin. Category values
//! are encoded as integers (cast from `f64`) and bins are ordered by value.
//!
//! The bin edges are placed at midpoints between consecutive category values,
//! which allows the existing `BinEdges::find_bin()` to route category values
//! to the correct bin via binary search.
//!
//! # Limits
//!
//! Supports up to 64 distinct category values per feature (matching the `u64`
//! bitmask used for categorical split routing in the tree arena).

use super::{BinEdges, BinningStrategy};

/// Maximum number of distinct categories supported per feature.
///
/// Limited by the `u64` bitmask used for split routing.
pub const MAX_CATEGORIES: usize = 64;

/// Categorical binning: one bin per observed distinct value.
///
/// Tracks observed category values (as sorted integers) and creates bin
/// edges at midpoints between consecutive values.
#[derive(Debug, Clone)]
pub struct CategoricalBinning {
    /// Sorted list of distinct observed category values.
    categories: Vec<i64>,
}

impl CategoricalBinning {
    /// Create a new empty categorical binner.
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
        }
    }

    /// Number of distinct categories observed so far.
    #[inline]
    pub fn n_categories(&self) -> usize {
        self.categories.len()
    }

    /// The sorted list of observed category values.
    pub fn categories(&self) -> &[i64] {
        &self.categories
    }

    /// Find the bin index for a category value.
    ///
    /// Returns the index of `value` in the sorted categories list, or `None`
    /// if the value hasn't been observed.
    pub fn category_index(&self, value: f64) -> Option<usize> {
        let v = value as i64;
        self.categories.binary_search(&v).ok()
    }
}

impl Default for CategoricalBinning {
    fn default() -> Self {
        Self::new()
    }
}

impl BinningStrategy for CategoricalBinning {
    fn observe(&mut self, value: f64) {
        let v = value as i64;
        // Insert in sorted order if not already present, up to MAX_CATEGORIES
        match self.categories.binary_search(&v) {
            Ok(_) => {} // already present
            Err(pos) => {
                if self.categories.len() < MAX_CATEGORIES {
                    self.categories.insert(pos, v);
                }
                // Silently ignore categories beyond MAX_CATEGORIES
            }
        }
    }

    fn compute_edges(&self, _n_bins: usize) -> BinEdges {
        // For categorical features, n_bins is ignored -- we use one bin per category.
        // Bin edges are placed at midpoints between consecutive category values.
        if self.categories.len() <= 1 {
            return BinEdges { edges: Vec::new() };
        }

        let edges: Vec<f64> = self
            .categories
            .windows(2)
            .map(|w| (w[0] as f64 + w[1] as f64) / 2.0)
            .collect();

        BinEdges { edges }
    }

    fn reset(&mut self) {
        self.categories.clear();
    }

    fn clone_fresh(&self) -> Box<dyn BinningStrategy> {
        Box::new(CategoricalBinning::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_binner() {
        let b = CategoricalBinning::new();
        assert_eq!(b.n_categories(), 0);
        let edges = b.compute_edges(10);
        assert!(edges.edges.is_empty());
        assert_eq!(edges.n_bins(), 1);
    }

    #[test]
    fn single_category() {
        let mut b = CategoricalBinning::new();
        b.observe(3.0);
        b.observe(3.0); // duplicate
        assert_eq!(b.n_categories(), 1);
        let edges = b.compute_edges(10);
        assert!(edges.edges.is_empty());
        assert_eq!(edges.n_bins(), 1);
    }

    #[test]
    fn two_categories() {
        let mut b = CategoricalBinning::new();
        b.observe(1.0);
        b.observe(5.0);
        assert_eq!(b.n_categories(), 2);

        let edges = b.compute_edges(10);
        assert_eq!(edges.edges.len(), 1);
        // Midpoint between 1 and 5 = 3.0
        assert!((edges.edges[0] - 3.0).abs() < 1e-10);
        assert_eq!(edges.n_bins(), 2);
    }

    #[test]
    fn multiple_categories_sorted() {
        let mut b = CategoricalBinning::new();
        // Insert out of order
        b.observe(5.0);
        b.observe(1.0);
        b.observe(3.0);
        b.observe(7.0);

        assert_eq!(b.categories(), &[1, 3, 5, 7]);

        let edges = b.compute_edges(10);
        assert_eq!(edges.edges.len(), 3);
        assert!((edges.edges[0] - 2.0).abs() < 1e-10); // midpoint(1,3)
        assert!((edges.edges[1] - 4.0).abs() < 1e-10); // midpoint(3,5)
        assert!((edges.edges[2] - 6.0).abs() < 1e-10); // midpoint(5,7)
    }

    #[test]
    fn find_bin_routes_correctly() {
        let mut b = CategoricalBinning::new();
        for i in 0..5 {
            b.observe(i as f64 * 2.0); // 0, 2, 4, 6, 8
        }

        let edges = b.compute_edges(10);
        // Edges: 1.0, 3.0, 5.0, 7.0 → bins: [<=1], (1,3], (3,5], (5,7], (>7)
        assert_eq!(edges.find_bin(0.0), 0); // category 0
        assert_eq!(edges.find_bin(2.0), 1); // category 2
        assert_eq!(edges.find_bin(4.0), 2); // category 4
        assert_eq!(edges.find_bin(6.0), 3); // category 6
        assert_eq!(edges.find_bin(8.0), 4); // category 8
    }

    #[test]
    fn category_index_lookup() {
        let mut b = CategoricalBinning::new();
        b.observe(10.0);
        b.observe(20.0);
        b.observe(30.0);

        assert_eq!(b.category_index(10.0), Some(0));
        assert_eq!(b.category_index(20.0), Some(1));
        assert_eq!(b.category_index(30.0), Some(2));
        assert_eq!(b.category_index(15.0), None);
    }

    #[test]
    fn max_categories_enforced() {
        let mut b = CategoricalBinning::new();
        for i in 0..100 {
            b.observe(i as f64);
        }
        assert_eq!(b.n_categories(), MAX_CATEGORIES);
    }

    #[test]
    fn reset_clears() {
        let mut b = CategoricalBinning::new();
        b.observe(1.0);
        b.observe(2.0);
        b.observe(3.0);
        b.reset();
        assert_eq!(b.n_categories(), 0);
    }

    #[test]
    fn n_bins_ignored() {
        let mut b = CategoricalBinning::new();
        b.observe(1.0);
        b.observe(2.0);
        b.observe(3.0);

        // n_bins param is ignored for categorical
        let edges_2 = b.compute_edges(2);
        let edges_100 = b.compute_edges(100);
        assert_eq!(edges_2.edges, edges_100.edges);
    }
}
