//! Streaming clustering algorithms.
//!
//! Three complementary approaches to online clustering:
//!
//! - [`StreamingKMeans`] -- mini-batch K-Means with optional forgetting factor
//!   for non-stationary streams (Sculley, 2010).
//! - [`DBStream`] -- density-based streaming clustering with micro-clusters
//!   and shared-density macro-cluster merging (Hahsler & Bolanos, 2016).
//! - [`CluStream`] -- micro/macro clustering via Cluster Feature vectors
//!   with on-demand weighted K-Means macro-clustering (Aggarwal et al., 2003).

pub mod clustream;
pub mod dbstream;
pub mod kmeans;

pub use clustream::{CluStream, CluStreamConfig, ClusterFeature};
pub use dbstream::{DBStream, DBStreamConfig, MicroCluster};
pub use kmeans::{StreamingKMeans, StreamingKMeansConfig};
