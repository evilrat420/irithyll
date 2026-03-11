//! Histogram-based feature binning for streaming tree construction.
//!
//! Streaming trees need to evaluate potential splits without storing raw data.
//! Histograms discretize continuous features into bins and accumulate gradient/hessian
//! statistics per bin, enabling efficient split evaluation.

pub mod bins;
pub mod categorical;
pub mod kmeans;
pub mod quantile;
#[cfg(feature = "simd")]
pub mod simd;
pub mod uniform;

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
        match self
            .edges
            .binary_search_by(|e| e.partial_cmp(&value).unwrap())
        {
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

// ---------------------------------------------------------------------------
// BinnerKind — enum dispatch replacing Box<dyn BinningStrategy>
// ---------------------------------------------------------------------------

/// Concrete binning strategy enum, eliminating `Box<dyn BinningStrategy>`
/// heap allocations per feature per leaf.
///
/// This is used internally by [`HoeffdingTree`](crate::tree::hoeffding::HoeffdingTree)
/// leaf states. Each variant delegates to the corresponding strategy impl.
#[derive(Debug, Clone)]
pub enum BinnerKind {
    /// Equal-width binning (default).
    Uniform(uniform::UniformBinning),

    /// Categorical binning — one bin per observed distinct value.
    Categorical(categorical::CategoricalBinning),

    /// K-means binning (feature-gated).
    #[cfg(feature = "kmeans-binning")]
    KMeans(kmeans::KMeansBinning),
}

impl BinnerKind {
    /// Create a new uniform binner (the default strategy).
    #[inline]
    pub fn uniform() -> Self {
        BinnerKind::Uniform(uniform::UniformBinning::new())
    }

    /// Create a new categorical binner.
    #[inline]
    pub fn categorical() -> Self {
        BinnerKind::Categorical(categorical::CategoricalBinning::new())
    }

    /// Create a new k-means binner.
    #[cfg(feature = "kmeans-binning")]
    #[inline]
    pub fn kmeans() -> Self {
        BinnerKind::KMeans(kmeans::KMeansBinning::new())
    }

    /// Observe a single value.
    #[inline]
    pub fn observe(&mut self, value: f64) {
        match self {
            BinnerKind::Uniform(b) => b.observe(value),
            BinnerKind::Categorical(b) => b.observe(value),
            #[cfg(feature = "kmeans-binning")]
            BinnerKind::KMeans(b) => b.observe(value),
        }
    }

    /// Compute bin edges from observed values.
    #[inline]
    pub fn compute_edges(&self, n_bins: usize) -> BinEdges {
        match self {
            BinnerKind::Uniform(b) => b.compute_edges(n_bins),
            BinnerKind::Categorical(b) => b.compute_edges(n_bins),
            #[cfg(feature = "kmeans-binning")]
            BinnerKind::KMeans(b) => b.compute_edges(n_bins),
        }
    }

    /// Reset observed state.
    #[inline]
    pub fn reset(&mut self) {
        match self {
            BinnerKind::Uniform(b) => b.reset(),
            BinnerKind::Categorical(b) => b.reset(),
            #[cfg(feature = "kmeans-binning")]
            BinnerKind::KMeans(b) => b.reset(),
        }
    }

    /// Whether this binner is for a categorical feature.
    #[inline]
    pub fn is_categorical(&self) -> bool {
        matches!(self, BinnerKind::Categorical(_))
    }

    /// Create a fresh instance with the same variant but no data.
    pub fn clone_fresh(&self) -> Self {
        match self {
            BinnerKind::Uniform(_) => BinnerKind::Uniform(uniform::UniformBinning::new()),
            BinnerKind::Categorical(_) => {
                BinnerKind::Categorical(categorical::CategoricalBinning::new())
            }
            #[cfg(feature = "kmeans-binning")]
            BinnerKind::KMeans(_) => BinnerKind::KMeans(kmeans::KMeansBinning::new()),
        }
    }
}
