//! Feature histogram storage with gradient/hessian sums per bin.
//!
//! Each `FeatureHistogram` accumulates gradient, hessian, and count statistics
//! into discrete bins defined by `BinEdges`. The SoA (Structure of Arrays) layout
//! keeps each statistic contiguous in memory for cache-friendly sequential scans
//! during split evaluation.

use crate::histogram::BinEdges;

/// Histogram for a single feature: accumulates grad/hess/count per bin.
/// SoA layout for cache efficiency during split-point evaluation.
#[derive(Debug, Clone)]
pub struct FeatureHistogram {
    /// Gradient sums per bin.
    pub grad_sums: Vec<f64>,
    /// Hessian sums per bin.
    pub hess_sums: Vec<f64>,
    /// Sample counts per bin.
    pub counts: Vec<u64>,
    /// Bin edges for this feature.
    pub edges: BinEdges,
}

impl FeatureHistogram {
    /// Create a new zeroed histogram with the given bin edges.
    pub fn new(edges: BinEdges) -> Self {
        let n = edges.n_bins();
        Self {
            grad_sums: vec![0.0; n],
            hess_sums: vec![0.0; n],
            counts: vec![0; n],
            edges,
        }
    }

    /// Accumulate a single sample into the appropriate bin.
    ///
    /// Finds the bin for `value` using the stored edges, then adds the
    /// gradient, hessian, and increments the count for that bin.
    #[inline]
    pub fn accumulate(&mut self, value: f64, gradient: f64, hessian: f64) {
        let bin = self.edges.find_bin(value);
        self.grad_sums[bin] += gradient;
        self.hess_sums[bin] += hessian;
        self.counts[bin] += 1;
    }

    /// Sum of all gradient accumulators across bins.
    pub fn total_gradient(&self) -> f64 {
        self.grad_sums.iter().sum()
    }

    /// Sum of all hessian accumulators across bins.
    pub fn total_hessian(&self) -> f64 {
        self.hess_sums.iter().sum()
    }

    /// Total sample count across all bins.
    pub fn total_count(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Number of bins in this histogram.
    #[inline]
    pub fn n_bins(&self) -> usize {
        self.edges.n_bins()
    }

    /// Reset all accumulators to zero, preserving the bin edges.
    pub fn reset(&mut self) {
        self.grad_sums.fill(0.0);
        self.hess_sums.fill(0.0);
        self.counts.fill(0);
    }

    /// Histogram subtraction trick: `self - child = sibling`.
    ///
    /// When building a tree, if you know the parent histogram and one child's
    /// histogram, you can derive the other child by subtraction rather than
    /// re-scanning the data. This halves the histogram-building cost at each
    /// split.
    ///
    /// The returned histogram uses edges cloned from `self`.
    pub fn subtract(&self, child: &FeatureHistogram) -> FeatureHistogram {
        debug_assert_eq!(
            self.n_bins(),
            child.n_bins(),
            "cannot subtract histograms with different bin counts"
        );
        let n = self.n_bins();
        let mut grad_sums = Vec::with_capacity(n);
        let mut hess_sums = Vec::with_capacity(n);
        let mut counts = Vec::with_capacity(n);

        for i in 0..n {
            grad_sums.push(self.grad_sums[i] - child.grad_sums[i]);
            hess_sums.push(self.hess_sums[i] - child.hess_sums[i]);
            // Counts are unsigned; parent should always >= child per bin.
            counts.push(self.counts[i].saturating_sub(child.counts[i]));
        }

        FeatureHistogram {
            grad_sums,
            hess_sums,
            counts,
            edges: self.edges.clone(),
        }
    }
}

/// Collection of histograms for all features at a single leaf node.
///
/// During tree construction, each active leaf maintains one `LeafHistograms`
/// that accumulates statistics from all samples routed to that leaf.
#[derive(Debug, Clone)]
pub struct LeafHistograms {
    pub histograms: Vec<FeatureHistogram>,
}

impl LeafHistograms {
    /// Create histograms for all features, one per entry in `edges_per_feature`.
    pub fn new(edges_per_feature: &[BinEdges]) -> Self {
        let histograms = edges_per_feature
            .iter()
            .map(|edges| FeatureHistogram::new(edges.clone()))
            .collect();
        Self { histograms }
    }

    /// Accumulate a sample across all feature histograms.
    ///
    /// `features` must have the same length as the number of histograms.
    /// Each feature value is accumulated into its corresponding histogram
    /// with the shared gradient and hessian.
    pub fn accumulate(&mut self, features: &[f64], gradient: f64, hessian: f64) {
        debug_assert_eq!(
            features.len(),
            self.histograms.len(),
            "feature count mismatch: got {} features but have {} histograms",
            features.len(),
            self.histograms.len(),
        );
        for (hist, &value) in self.histograms.iter_mut().zip(features.iter()) {
            hist.accumulate(value, gradient, hessian);
        }
    }

    /// Total sample count, taken from the first histogram.
    ///
    /// All histograms receive the same samples, so their total counts must agree.
    /// Returns 0 if there are no features.
    pub fn total_count(&self) -> u64 {
        self.histograms
            .first()
            .map_or(0, |h| h.total_count())
    }

    /// Number of features (histograms).
    #[inline]
    pub fn n_features(&self) -> usize {
        self.histograms.len()
    }

    /// Reset all histograms to zero, preserving bin edges.
    pub fn reset(&mut self) {
        for hist in &mut self.histograms {
            hist.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::BinEdges;

    /// Helper: edges at [2.0, 4.0, 6.0] giving 4 bins:
    /// (-inf, 2.0], (2.0, 4.0], (4.0, 6.0], (6.0, +inf)
    fn four_bin_edges() -> BinEdges {
        BinEdges {
            edges: vec![2.0, 4.0, 6.0],
        }
    }

    #[test]
    fn feature_histogram_new_is_zeroed() {
        let h = FeatureHistogram::new(four_bin_edges());
        assert_eq!(h.n_bins(), 4);
        assert_eq!(h.total_gradient(), 0.0);
        assert_eq!(h.total_hessian(), 0.0);
        assert_eq!(h.total_count(), 0);
    }

    #[test]
    fn feature_histogram_accumulate_single() {
        let mut h = FeatureHistogram::new(four_bin_edges());
        // value 3.0 -> bin 1 (between edges 2.0 and 4.0)
        h.accumulate(3.0, 1.5, 0.5);
        assert_eq!(h.counts[1], 1);
        assert_eq!(h.grad_sums[1], 1.5);
        assert_eq!(h.hess_sums[1], 0.5);
        assert_eq!(h.total_count(), 1);
    }

    #[test]
    fn feature_histogram_accumulate_multiple() {
        let mut h = FeatureHistogram::new(four_bin_edges());
        // bin 0: values < 2.0
        h.accumulate(1.0, 0.1, 0.01);
        h.accumulate(0.5, 0.2, 0.02);
        // bin 1: 2.0 < values <= 4.0 (value == 2.0 lands in bin 1 because Ok(0) -> 0+1=1)
        h.accumulate(2.0, 0.3, 0.03);
        // bin 2: 4.0 < values <= 6.0
        h.accumulate(5.0, 0.4, 0.04);
        // bin 3: values > 6.0
        h.accumulate(7.0, 0.5, 0.05);
        h.accumulate(8.0, 0.6, 0.06);

        assert_eq!(h.counts, vec![2, 1, 1, 2]);
        assert_eq!(h.total_count(), 6);

        let total_grad = 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6;
        assert!((h.total_gradient() - total_grad).abs() < 1e-12);

        let total_hess = 0.01 + 0.02 + 0.03 + 0.04 + 0.05 + 0.06;
        assert!((h.total_hessian() - total_hess).abs() < 1e-12);
    }

    #[test]
    fn feature_histogram_subtraction_trick() {
        let edges = four_bin_edges();
        let mut parent = FeatureHistogram::new(edges.clone());
        let mut child = FeatureHistogram::new(edges);

        // Populate parent with 6 samples
        parent.accumulate(1.0, 0.1, 0.01);
        parent.accumulate(3.0, 0.2, 0.02);
        parent.accumulate(5.0, 0.3, 0.03);
        parent.accumulate(7.0, 0.4, 0.04);
        parent.accumulate(1.5, 0.5, 0.05);
        parent.accumulate(5.5, 0.6, 0.06);

        // Child gets 3 of those samples
        child.accumulate(1.0, 0.1, 0.01);
        child.accumulate(5.0, 0.3, 0.03);
        child.accumulate(7.0, 0.4, 0.04);

        let sibling = parent.subtract(&child);

        // Sibling should have the remaining 3 samples
        assert_eq!(sibling.total_count(), 3);
        // bin 0: parent had 2 (1.0, 1.5), child had 1 (1.0) -> sibling 1
        assert_eq!(sibling.counts[0], 1);
        assert!((sibling.grad_sums[0] - 0.5).abs() < 1e-12);
        assert!((sibling.hess_sums[0] - 0.05).abs() < 1e-12);
        // bin 1: parent had 1 (3.0), child had 0 -> sibling 1
        assert_eq!(sibling.counts[1], 1);
        assert!((sibling.grad_sums[1] - 0.2).abs() < 1e-12);
        // bin 2: parent had 2 (5.0, 5.5), child had 1 (5.0) -> sibling 1
        assert_eq!(sibling.counts[2], 1);
        assert!((sibling.grad_sums[2] - 0.6).abs() < 1e-12);
        // bin 3: parent had 1 (7.0), child had 1 (7.0) -> sibling 0
        assert_eq!(sibling.counts[3], 0);
        assert!((sibling.grad_sums[3]).abs() < 1e-12);
    }

    #[test]
    fn feature_histogram_reset() {
        let mut h = FeatureHistogram::new(four_bin_edges());
        h.accumulate(1.0, 1.0, 1.0);
        h.accumulate(3.0, 2.0, 2.0);
        assert_eq!(h.total_count(), 2);

        h.reset();
        assert_eq!(h.total_count(), 0);
        assert_eq!(h.total_gradient(), 0.0);
        assert_eq!(h.total_hessian(), 0.0);
        assert_eq!(h.n_bins(), 4); // edges preserved
    }

    #[test]
    fn leaf_histograms_multi_feature() {
        let edges_f0 = BinEdges {
            edges: vec![5.0],
        }; // 2 bins
        let edges_f1 = BinEdges {
            edges: vec![2.0, 4.0, 6.0],
        }; // 4 bins

        let mut leaf = LeafHistograms::new(&[edges_f0, edges_f1]);
        assert_eq!(leaf.n_features(), 2);

        // Sample 1: f0=3.0 (bin 0), f1=5.0 (bin 2)
        leaf.accumulate(&[3.0, 5.0], 1.0, 0.1);
        // Sample 2: f0=7.0 (bin 1), f1=1.0 (bin 0)
        leaf.accumulate(&[7.0, 1.0], 2.0, 0.2);
        // Sample 3: f0=3.0 (bin 0), f1=3.0 (bin 1)
        leaf.accumulate(&[3.0, 3.0], 3.0, 0.3);

        assert_eq!(leaf.total_count(), 3);

        // Feature 0 checks
        let h0 = &leaf.histograms[0];
        assert_eq!(h0.counts, vec![2, 1]); // two in bin 0, one in bin 1
        assert!((h0.total_gradient() - 6.0).abs() < 1e-12);
        assert!((h0.total_hessian() - 0.6).abs() < 1e-12);

        // Feature 1 checks
        let h1 = &leaf.histograms[1];
        assert_eq!(h1.counts, vec![1, 1, 1, 0]);
        assert!((h1.total_gradient() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn leaf_histograms_reset() {
        let edges = BinEdges {
            edges: vec![3.0],
        };
        let mut leaf = LeafHistograms::new(&[edges.clone(), edges]);
        leaf.accumulate(&[1.0, 5.0], 1.0, 1.0);
        assert_eq!(leaf.total_count(), 1);

        leaf.reset();
        assert_eq!(leaf.total_count(), 0);
        assert_eq!(leaf.n_features(), 2);
    }

    #[test]
    fn leaf_histograms_empty() {
        let leaf = LeafHistograms::new(&[]);
        assert_eq!(leaf.n_features(), 0);
        assert_eq!(leaf.total_count(), 0);
    }

    #[test]
    fn single_edge_histogram() {
        // Single edge -> 2 bins: (-inf, 5.0] and (5.0, +inf)
        let edges = BinEdges {
            edges: vec![5.0],
        };
        let mut h = FeatureHistogram::new(edges);
        assert_eq!(h.n_bins(), 2);

        h.accumulate(3.0, 1.0, 0.1);
        h.accumulate(5.0, 2.0, 0.2); // exact edge -> bin 1
        h.accumulate(7.0, 3.0, 0.3);

        assert_eq!(h.counts, vec![1, 2]);
        assert!((h.grad_sums[0] - 1.0).abs() < 1e-12);
        assert!((h.grad_sums[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn no_edges_single_bin() {
        // No edges -> 1 bin, everything goes to bin 0
        let edges = BinEdges { edges: vec![] };
        let mut h = FeatureHistogram::new(edges);
        assert_eq!(h.n_bins(), 1);

        h.accumulate(42.0, 1.0, 0.5);
        h.accumulate(-100.0, 2.0, 0.3);

        assert_eq!(h.counts, vec![2]);
        assert!((h.total_gradient() - 3.0).abs() < 1e-12);
        assert!((h.total_hessian() - 0.8).abs() < 1e-12);
    }
}
