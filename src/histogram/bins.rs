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
    ///
    /// When `decay_scale != 1.0`, these are stored in un-decayed coordinates.
    /// Multiply by `decay_scale` to recover effective (decayed) values.
    pub grad_sums: Vec<f64>,
    /// Hessian sums per bin (same scaling convention as `grad_sums`).
    pub hess_sums: Vec<f64>,
    /// Sample counts per bin (never decayed).
    pub counts: Vec<u64>,
    /// Bin edges for this feature.
    pub edges: BinEdges,
    /// Accumulated decay factor for lazy forward decay.
    ///
    /// Instead of decaying all bins on every sample (O(n_bins)), we track
    /// a running scale factor and store new samples in un-decayed coordinates.
    /// The decay is materialized (applied to all bins) only when the histogram
    /// is read — typically at split evaluation time, every `grace_period` samples.
    ///
    /// Invariant: `effective_value[i] = grad_sums[i] * decay_scale`.
    /// Value is 1.0 when no decay is pending (default, or after [`materialize_decay`]).
    decay_scale: f64,
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
            decay_scale: 1.0,
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
    ///
    /// Accounts for any pending lazy decay by multiplying by `decay_scale`.
    pub fn total_gradient(&self) -> f64 {
        let raw = {
            #[cfg(feature = "simd")]
            {
                crate::histogram::simd::sum_f64(&self.grad_sums)
            }
            #[cfg(not(feature = "simd"))]
            {
                self.grad_sums.iter().sum::<f64>()
            }
        };
        raw * self.decay_scale
    }

    /// Sum of all hessian accumulators across bins.
    ///
    /// Accounts for any pending lazy decay by multiplying by `decay_scale`.
    pub fn total_hessian(&self) -> f64 {
        let raw = {
            #[cfg(feature = "simd")]
            {
                crate::histogram::simd::sum_f64(&self.hess_sums)
            }
            #[cfg(not(feature = "simd"))]
            {
                self.hess_sums.iter().sum::<f64>()
            }
        };
        raw * self.decay_scale
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

    /// Accumulate with lazy forward decay: O(1) per sample.
    ///
    /// Implements the forward decay scheme (Cormode et al. 2009) using a
    /// deferred scaling technique. Instead of decaying all bins on every
    /// sample, we track a running `decay_scale` and store new samples in
    /// un-decayed coordinates. The actual bin values are materialized only
    /// when read (see `materialize_decay()`).
    ///
    /// Mathematically equivalent to eager decay: a gradient `g` added at
    /// epoch `t` contributes `g * alpha^(T - t)` at read time `T`.
    ///
    /// The `counts` array is NOT decayed — it tracks the raw sample count
    /// for the Hoeffding bound computation.
    #[inline]
    pub fn accumulate_with_decay(&mut self, value: f64, gradient: f64, hessian: f64, alpha: f64) {
        self.decay_scale *= alpha;
        let inv_scale = 1.0 / self.decay_scale;
        let bin = self.edges.find_bin(value);
        self.grad_sums[bin] += gradient * inv_scale;
        self.hess_sums[bin] += hessian * inv_scale;
        self.counts[bin] += 1;

        // Renormalize when scale underflows to prevent precision loss.
        // With alpha ≈ 0.986 (half_life = 50), this fires roughly every
        // 16K samples per leaf — negligible amortized cost.
        if self.decay_scale < 1e-100 {
            self.materialize_decay();
        }
    }

    /// Apply the pending decay factor to all bins and reset the scale.
    ///
    /// After calling this, `grad_sums` and `hess_sums` contain the true
    /// decayed values and can be passed directly to split evaluation.
    /// O(n_bins) — called at split evaluation time, not per sample.
    #[inline]
    pub fn materialize_decay(&mut self) {
        if (self.decay_scale - 1.0).abs() > f64::EPSILON {
            for i in 0..self.grad_sums.len() {
                self.grad_sums[i] *= self.decay_scale;
                self.hess_sums[i] *= self.decay_scale;
            }
            self.decay_scale = 1.0;
        }
    }

    /// Reset all accumulators to zero, preserving the bin edges.
    pub fn reset(&mut self) {
        self.grad_sums.fill(0.0);
        self.hess_sums.fill(0.0);
        self.counts.fill(0);
        self.decay_scale = 1.0;
    }

    /// Histogram subtraction trick: `self - child = sibling`.
    ///
    /// When building a tree, if you know the parent histogram and one child's
    /// histogram, you can derive the other child by subtraction rather than
    /// re-scanning the data. This halves the histogram-building cost at each
    /// split.
    ///
    /// Scale-aware: if either operand has pending lazy decay, the effective
    /// (decayed) values are used. The result has `decay_scale = 1.0`.
    ///
    /// The returned histogram uses edges cloned from `self`.
    pub fn subtract(&self, child: &FeatureHistogram) -> FeatureHistogram {
        debug_assert_eq!(
            self.n_bins(),
            child.n_bins(),
            "cannot subtract histograms with different bin counts"
        );
        let n = self.n_bins();
        let s_self = self.decay_scale;
        let s_child = child.decay_scale;

        // Fast path: both scales are 1.0, use existing SIMD logic directly.
        let scales_trivial = (s_self - 1.0).abs() <= f64::EPSILON
            && (s_child - 1.0).abs() <= f64::EPSILON;

        #[cfg(feature = "simd")]
        {
            if scales_trivial {
                let mut grad_sums = vec![0.0; n];
                let mut hess_sums = vec![0.0; n];
                let mut counts = vec![0u64; n];
                crate::histogram::simd::subtract_f64(&self.grad_sums, &child.grad_sums, &mut grad_sums);
                crate::histogram::simd::subtract_f64(&self.hess_sums, &child.hess_sums, &mut hess_sums);
                crate::histogram::simd::subtract_u64(&self.counts, &child.counts, &mut counts);
                return FeatureHistogram {
                    grad_sums,
                    hess_sums,
                    counts,
                    edges: self.edges.clone(),
                    decay_scale: 1.0,
                };
            }
        }

        // Scale-aware scalar path (also used as non-SIMD fallback).
        let _ = scales_trivial; // suppress unused warning in non-SIMD builds
        let mut grad_sums = Vec::with_capacity(n);
        let mut hess_sums = Vec::with_capacity(n);
        let mut counts = Vec::with_capacity(n);

        for i in 0..n {
            grad_sums.push(self.grad_sums[i] * s_self - child.grad_sums[i] * s_child);
            hess_sums.push(self.hess_sums[i] * s_self - child.hess_sums[i] * s_child);
            counts.push(self.counts[i].saturating_sub(child.counts[i]));
        }

        FeatureHistogram {
            grad_sums,
            hess_sums,
            counts,
            edges: self.edges.clone(),
            decay_scale: 1.0,
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

    /// Accumulate a sample with forward decay across all feature histograms.
    ///
    /// Each histogram is decayed by `alpha` before accumulating the new sample.
    pub fn accumulate_with_decay(&mut self, features: &[f64], gradient: f64, hessian: f64, alpha: f64) {
        debug_assert_eq!(
            features.len(),
            self.histograms.len(),
            "feature count mismatch: got {} features but have {} histograms",
            features.len(),
            self.histograms.len(),
        );
        for (hist, &value) in self.histograms.iter_mut().zip(features.iter()) {
            hist.accumulate_with_decay(value, gradient, hessian, alpha);
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

    /// Materialize pending lazy decay across all feature histograms.
    ///
    /// Call before reading raw bin values (e.g., split evaluation).
    /// O(n_features * n_bins) — called at split evaluation time, not per sample.
    pub fn materialize_decay(&mut self) {
        for hist in &mut self.histograms {
            hist.materialize_decay();
        }
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

    #[test]
    fn accumulate_with_decay_recent_dominates() {
        // 3 bins: edges at [3.0, 7.0] => bins [<3, 3-7, >7]
        let edges = BinEdges { edges: vec![3.0, 7.0] };
        let mut h = FeatureHistogram::new(edges);
        let alpha = 0.9;

        // 100 samples in bin 0 (value=1.0)
        for _ in 0..100 {
            h.accumulate_with_decay(1.0, 1.0, 1.0, alpha);
        }

        // 50 samples in bin 2 (value=8.0) — enough for recent to dominate
        for _ in 0..50 {
            h.accumulate_with_decay(8.0, 1.0, 1.0, alpha);
        }

        // Materialize to get true decayed values.
        h.materialize_decay();

        // With alpha=0.9, old bin 0 decays rapidly while bin 2 accumulates.
        assert!(
            h.grad_sums[2] > h.grad_sums[0],
            "recent bin should dominate: bin2={} > bin0={}",
            h.grad_sums[2], h.grad_sums[0],
        );
    }

    #[test]
    fn lazy_decay_matches_eager() {
        // Compare lazy decay (new) against a manual eager implementation.
        // They must produce identical results (within f64 precision).
        let edges = BinEdges { edges: vec![3.0, 6.0] }; // 3 bins
        let alpha = 0.95;

        // Lazy histogram (our implementation).
        let mut lazy = FeatureHistogram::new(edges.clone());

        // Manual eager: track expected values by applying alpha to all bins
        // each step, then accumulating.
        let n = 3;
        let mut eager_grad = vec![0.0; n];
        let mut eager_hess = vec![0.0; n];

        let samples: Vec<(f64, f64, f64)> = vec![
            (1.0, 0.5, 1.0),   // bin 0
            (4.0, -0.3, 0.8),  // bin 1
            (1.0, 0.7, 1.2),   // bin 0
            (8.0, -1.0, 0.5),  // bin 2
            (5.0, 0.2, 0.9),   // bin 1
            (1.0, 0.1, 1.1),   // bin 0
            (8.0, 0.4, 0.6),   // bin 2
            (4.0, -0.5, 1.0),  // bin 1
        ];

        let edge_vals = [3.0, 6.0];
        for &(value, gradient, hessian) in &samples {
            // Eager: decay all bins, then accumulate.
            for i in 0..n {
                eager_grad[i] *= alpha;
                eager_hess[i] *= alpha;
            }
            let bin = if value <= edge_vals[0] { 0 }
                      else if value <= edge_vals[1] { 1 }
                      else { 2 };
            eager_grad[bin] += gradient;
            eager_hess[bin] += hessian;

            // Lazy: our O(1) implementation.
            lazy.accumulate_with_decay(value, gradient, hessian, alpha);
        }

        // Materialize lazy to get comparable values.
        lazy.materialize_decay();

        for i in 0..n {
            assert!(
                (lazy.grad_sums[i] - eager_grad[i]).abs() < 1e-10,
                "grad_sums[{}]: lazy={}, eager={}", i, lazy.grad_sums[i], eager_grad[i],
            );
            assert!(
                (lazy.hess_sums[i] - eager_hess[i]).abs() < 1e-10,
                "hess_sums[{}]: lazy={}, eager={}", i, lazy.hess_sums[i], eager_hess[i],
            );
        }
    }

    #[test]
    fn lazy_decay_total_gradient_without_materialize() {
        // total_gradient() should return correct decayed value even
        // WITHOUT explicit materialize_decay() call.
        let edges = BinEdges { edges: vec![5.0] }; // 2 bins
        let alpha = 0.9;
        let mut h = FeatureHistogram::new(edges);

        h.accumulate_with_decay(1.0, 1.0, 1.0, alpha); // bin 0
        h.accumulate_with_decay(7.0, 1.0, 1.0, alpha); // bin 1

        // Expected: first sample decayed once more = 1.0 * 0.9 = 0.9
        // Second sample = 1.0, total = 1.9
        let total = h.total_gradient();
        assert!(
            (total - 1.9).abs() < 1e-10,
            "total_gradient should account for decay_scale: got {}", total,
        );
    }

    #[test]
    fn materialize_is_idempotent() {
        let edges = BinEdges { edges: vec![5.0] };
        let mut h = FeatureHistogram::new(edges);
        let alpha = 0.95;

        for _ in 0..50 {
            h.accumulate_with_decay(1.0, 1.0, 1.0, alpha);
        }

        h.materialize_decay();
        let grad_after_first = h.grad_sums.clone();

        h.materialize_decay(); // second call should be no-op
        assert_eq!(h.grad_sums, grad_after_first, "second materialize should be a no-op");
    }

    #[test]
    fn lazy_decay_renormalization() {
        // Use a very aggressive alpha to trigger renormalization quickly.
        let edges = BinEdges { edges: vec![5.0] };
        let mut h = FeatureHistogram::new(edges);
        let alpha = 0.5; // decay_scale halves each step, hits 1e-100 at ~332 steps

        for _ in 0..500 {
            h.accumulate_with_decay(1.0, 1.0, 1.0, alpha);
        }

        // Should not have NaN or Inf despite extreme decay.
        let total = h.total_gradient();
        assert!(total.is_finite(), "gradient should be finite after renormalization, got {}", total);
        assert!(total > 0.0, "gradient should be positive, got {}", total);

        // With alpha=0.5, geometric sum converges to 1/(1-0.5) = 2.0
        assert!(
            (total - 2.0).abs() < 0.1,
            "total gradient should converge to ~2.0, got {}", total,
        );
    }
}
