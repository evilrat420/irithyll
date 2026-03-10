//! Histogram-based feature binning for streaming tree construction.
//!
//! Streaming trees need to evaluate potential splits without storing raw data.
//! Histograms discretize continuous features into bins and accumulate gradient/hessian
//! statistics per bin, enabling efficient split evaluation.

pub mod quantile;
pub mod kmeans;
pub mod uniform;
pub mod bins;

/// Bin edge boundaries computed by a binning strategy.
#[derive(Debug, Clone)]
pub struct BinEdges {
    /// Sorted thresholds defining bin boundaries. `n_bins - 1` values.
    pub edges: Vec<f64>,
}

impl BinEdges {
    /// Find which bin a value falls into. Returns bin index in `[0, n_bins)`.
    #[inline]
    pub fn find_bin(&self, value: f64) -> usize {
        match self.edges.binary_search_by(|e| e.partial_cmp(&value).unwrap()) {
            Ok(i) => i + 1,
            Err(i) => i,
        }
    }

    /// Number of bins (edges.len() + 1).
    #[inline]
    pub fn n_bins(&self) -> usize {
        self.edges.len() + 1
    }
}

/// A strategy for computing histogram bin edges from a stream of values.
pub trait BinningStrategy: Send + Sync + 'static {
    /// Observe a single value from the stream.
    fn observe(&mut self, value: f64);

    /// Compute bin edges from observed values.
    fn compute_edges(&self, n_bins: usize) -> BinEdges;

    /// Reset observed state.
    fn reset(&mut self);

    /// Create a fresh instance with the same configuration.
    fn clone_fresh(&self) -> Box<dyn BinningStrategy>;
}
