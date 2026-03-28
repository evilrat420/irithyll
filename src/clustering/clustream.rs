//! CluStream: streaming clustering via micro-cluster maintenance.
//!
//! Implements the CluStream algorithm from Aggarwal et al. (2003):
//! "A Framework for Clustering Evolving Data Streams."
//!
//! CluStream operates in two phases:
//!
//! - **Online phase:** Maintain a bounded set of micro-clusters, each
//!   represented by a Cluster Feature (CF) vector of sufficient statistics
//!   `(n, LS, SS)`. Incoming points are either absorbed into the nearest
//!   micro-cluster (if within a radius threshold) or trigger a merge of
//!   the two closest micro-clusters to make room for a new singleton.
//!
//! - **Offline phase:** On demand, run weighted K-Means over the micro-cluster
//!   centers to produce `k` macro-clusters. This is a snapshot operation that
//!   does not modify the online state.
//!
//! # Algorithm
//!
//! For each incoming point *x*:
//!
//! 1. Find the nearest micro-cluster by Euclidean distance to its center.
//! 2. If the distance is within `max_radius_factor * MC.radius()`, absorb
//!    *x* into that micro-cluster (additively update n, LS, SS).
//! 3. Otherwise, if the micro-cluster buffer is not full, create a new
//!    singleton micro-cluster from *x*.
//! 4. Otherwise, merge the two closest micro-clusters and create a new
//!    singleton from *x*, keeping the total count at `max_micro_clusters`.
//!
//! Means and variances are recoverable from the CF triple without storing
//! individual points, giving O(q * d) memory where q = max micro-clusters
//! and d = dimensionality.
//!
//! # References
//!
//! Aggarwal, C. C., Han, J., Wang, J., & Yu, P. S. (2003). A framework for
//! clustering evolving data streams. In *Proceedings of the 29th International
//! Conference on Very Large Data Bases (VLDB)* (pp. 81-92).

// ─── Cluster Feature ────────────────────────────────────────────────────

/// Sufficient statistics for a micro-cluster.
///
/// A Cluster Feature (CF) compactly represents a set of absorbed points via
/// three components:
///
/// - `n` -- number of points absorbed
/// - `linear_sum` -- component-wise sum of all points
/// - `squared_sum` -- component-wise sum of squared components
///
/// From these, the centroid and radius can be recovered in O(d) time without
/// storing individual points (Aggarwal et al., 2003).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-json", derive(serde::Serialize, serde::Deserialize))]
pub struct ClusterFeature {
    /// Number of points absorbed into this micro-cluster.
    pub n: u64,
    /// Linear sum of absorbed points (component-wise).
    pub linear_sum: Vec<f64>,
    /// Sum of squares of absorbed points (component-wise).
    pub squared_sum: Vec<f64>,
}

impl ClusterFeature {
    /// Create an empty cluster feature for `n_features`-dimensional data.
    pub fn new(n_features: usize) -> Self {
        Self {
            n: 0,
            linear_sum: vec![0.0; n_features],
            squared_sum: vec![0.0; n_features],
        }
    }

    /// Absorb a single point into this micro-cluster.
    ///
    /// Updates n, LS, and SS additively. Panics if `point.len()` does not
    /// match the dimensionality.
    pub fn absorb(&mut self, point: &[f64]) {
        debug_assert_eq!(
            point.len(),
            self.linear_sum.len(),
            "point dimensionality mismatch: expected {}, got {}",
            self.linear_sum.len(),
            point.len(),
        );
        self.n += 1;
        for (i, &v) in point.iter().enumerate() {
            self.linear_sum[i] += v;
            self.squared_sum[i] += v * v;
        }
    }

    /// Compute the centroid of this micro-cluster: `LS / n`.
    ///
    /// Returns a zero vector if no points have been absorbed.
    pub fn center(&self) -> Vec<f64> {
        if self.n == 0 {
            return vec![0.0; self.linear_sum.len()];
        }
        let n = self.n as f64;
        self.linear_sum.iter().map(|&ls| ls / n).collect()
    }

    /// Compute the radius of this micro-cluster.
    ///
    /// Defined as `sqrt(mean(SS/n - (LS/n)^2))`, which is the RMS standard
    /// deviation across all dimensions. Returns 0.0 if fewer than 2 points
    /// have been absorbed (variance is undefined for a single point).
    pub fn radius(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let n = self.n as f64;
        let d = self.linear_sum.len() as f64;
        let sum_var: f64 = self
            .linear_sum
            .iter()
            .zip(self.squared_sum.iter())
            .map(|(&ls, &ss)| {
                let mean = ls / n;
                ss / n - mean * mean
            })
            .sum();
        let avg_var = sum_var / d;
        // Guard against small negative values from floating-point arithmetic.
        if avg_var <= 0.0 {
            return 0.0;
        }
        avg_var.sqrt()
    }

    /// Merge another cluster feature into this one additively.
    ///
    /// After merging, this CF represents the union of both point sets.
    pub fn merge(&mut self, other: &ClusterFeature) {
        debug_assert_eq!(
            self.linear_sum.len(),
            other.linear_sum.len(),
            "cannot merge CFs with different dimensionality",
        );
        self.n += other.n;
        for (i, &v) in other.linear_sum.iter().enumerate() {
            self.linear_sum[i] += v;
        }
        for (i, &v) in other.squared_sum.iter().enumerate() {
            self.squared_sum[i] += v;
        }
    }
}

// ─── Configuration ──────────────────────────────────────────────────────

/// Configuration for the CluStream algorithm.
///
/// Use [`CluStreamConfig::builder`] for ergonomic construction with
/// validation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-json", derive(serde::Serialize, serde::Deserialize))]
pub struct CluStreamConfig {
    /// Maximum number of micro-clusters to maintain in the online phase.
    pub max_micro_clusters: usize,
    /// Maximum radius factor: a point is absorbed into a micro-cluster if
    /// its distance to the MC center is less than
    /// `max_radius_factor * MC.radius()`. Default: 2.0.
    pub max_radius_factor: f64,
    /// Number of features (dimensionality). Set to 0 for lazy initialization
    /// from the first sample.
    pub n_features: usize,
}

/// Builder for [`CluStreamConfig`] with validation.
#[derive(Debug, Clone)]
pub struct CluStreamConfigBuilder {
    max_micro_clusters: usize,
    max_radius_factor: f64,
    n_features: usize,
}

impl CluStreamConfig {
    /// Create a builder with the required `max_micro_clusters` parameter.
    ///
    /// # Example
    ///
    /// ```
    /// use irithyll::clustering::clustream::CluStreamConfig;
    ///
    /// let config = CluStreamConfig::builder(10)
    ///     .max_radius_factor(2.5)
    ///     .build()
    ///     .unwrap();
    /// assert_eq!(config.max_micro_clusters, 10);
    /// ```
    pub fn builder(max_micro_clusters: usize) -> CluStreamConfigBuilder {
        CluStreamConfigBuilder {
            max_micro_clusters,
            max_radius_factor: 2.0,
            n_features: 0,
        }
    }
}

impl CluStreamConfigBuilder {
    /// Set the maximum radius factor (default: 2.0).
    pub fn max_radius_factor(mut self, f: f64) -> Self {
        self.max_radius_factor = f;
        self
    }

    /// Set the number of features (default: 0, lazy from first sample).
    pub fn n_features(mut self, d: usize) -> Self {
        self.n_features = d;
        self
    }

    /// Build the configuration, validating all parameters.
    ///
    /// Returns an error if `max_micro_clusters < 2`.
    pub fn build(self) -> Result<CluStreamConfig, String> {
        if self.max_micro_clusters < 2 {
            return Err(format!(
                "max_micro_clusters must be >= 2, got {}",
                self.max_micro_clusters
            ));
        }
        Ok(CluStreamConfig {
            max_micro_clusters: self.max_micro_clusters,
            max_radius_factor: self.max_radius_factor,
            n_features: self.n_features,
        })
    }
}

// ─── CluStream ──────────────────────────────────────────────────────────

/// Streaming clustering via micro-cluster maintenance (Aggarwal et al., 2003).
///
/// Maintains a bounded set of micro-clusters in the online phase, then
/// produces macro-clusters on demand via weighted K-Means over the
/// micro-cluster summaries.
///
/// # Example
///
/// ```
/// use irithyll::clustering::clustream::{CluStreamConfig, CluStream};
///
/// let config = CluStreamConfig::builder(10)
///     .max_radius_factor(2.0)
///     .build()
///     .unwrap();
/// let mut cs = CluStream::new(config);
///
/// // Online phase: stream two well-separated clusters
/// for i in 0..10 {
///     cs.train_one(&[0.0 + i as f64 * 0.01, 0.0]);
///     cs.train_one(&[10.0 + i as f64 * 0.01, 10.0]);
/// }
///
/// // Predict nearest micro-cluster
/// let cluster = cs.predict(&[0.05, 0.0]);
/// assert!(cluster < cs.n_micro_clusters());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-json", derive(serde::Serialize, serde::Deserialize))]
pub struct CluStream {
    config: CluStreamConfig,
    micro_clusters: Vec<ClusterFeature>,
    n_features: usize,
    n_samples: u64,
}

impl CluStream {
    /// Create a new CluStream instance from the given configuration.
    pub fn new(config: CluStreamConfig) -> Self {
        let n_features = config.n_features;
        Self {
            config,
            micro_clusters: Vec::new(),
            n_features,
            n_samples: 0,
        }
    }

    /// Process a single data point in the online phase.
    ///
    /// The point is either absorbed into the nearest micro-cluster (if
    /// within the radius threshold), used to create a new micro-cluster
    /// (if capacity remains), or triggers a merge of the two closest
    /// micro-clusters before being inserted as a new singleton.
    pub fn train_one(&mut self, features: &[f64]) {
        // Lazy dimensionality initialization.
        if self.n_features == 0 {
            self.n_features = features.len();
        }
        self.n_samples += 1;

        // First point ever: create the initial micro-cluster.
        if self.micro_clusters.is_empty() {
            let mut cf = ClusterFeature::new(self.n_features);
            cf.absorb(features);
            self.micro_clusters.push(cf);
            return;
        }

        // Find the nearest micro-cluster.
        let (nearest_idx, nearest_dist) = self.nearest_mc(features);

        // Decide: absorb, create, or merge-then-create.
        let mc = &self.micro_clusters[nearest_idx];
        let r = mc.radius();
        let threshold = self.config.max_radius_factor * r;

        // Absorb if within radius threshold.
        // Special case: for micro-clusters with very small radius (n < 3),
        // absorb unconditionally into the nearest MC since the radius is
        // not yet statistically meaningful.
        if nearest_dist < threshold || mc.n < 3 {
            self.micro_clusters[nearest_idx].absorb(features);
            return;
        }

        // Room for a new micro-cluster.
        if self.micro_clusters.len() < self.config.max_micro_clusters {
            let mut cf = ClusterFeature::new(self.n_features);
            cf.absorb(features);
            self.micro_clusters.push(cf);
            return;
        }

        // At capacity: merge the two closest MCs, then create a new one.
        let (i, j) = self.closest_mc_pair();
        // Merge j into i (keep the lower index, remove the higher).
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        let removed = self.micro_clusters.remove(hi);
        self.micro_clusters[lo].merge(&removed);

        let mut cf = ClusterFeature::new(self.n_features);
        cf.absorb(features);
        self.micro_clusters.push(cf);
    }

    /// Predict the nearest micro-cluster index for a given point.
    ///
    /// Returns the index into the internal micro-cluster vector.
    /// Panics if no micro-clusters exist (i.e., no training data seen).
    pub fn predict(&self, features: &[f64]) -> usize {
        assert!(
            !self.micro_clusters.is_empty(),
            "cannot predict with no micro-clusters -- call train_one first"
        );
        let (idx, _) = self.nearest_mc(features);
        idx
    }

    /// Return a reference to the current micro-clusters.
    pub fn micro_clusters(&self) -> &[ClusterFeature] {
        &self.micro_clusters
    }

    /// Return the current number of micro-clusters.
    pub fn n_micro_clusters(&self) -> usize {
        self.micro_clusters.len()
    }

    /// Total number of samples processed so far.
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Reset all state, clearing micro-clusters and sample count.
    pub fn reset(&mut self) {
        self.micro_clusters.clear();
        self.n_samples = 0;
        // Reset lazy dimensionality if it was set from data.
        if self.config.n_features == 0 {
            self.n_features = 0;
        }
    }

    /// Produce macro-clusters by running weighted K-Means on micro-cluster
    /// centers (offline phase).
    ///
    /// Returns `k` groups, where each group is a `Vec<usize>` of micro-cluster
    /// indices belonging to that macro-cluster. If there are fewer micro-clusters
    /// than `k`, each micro-cluster is placed in its own group.
    pub fn macro_clusters(&self, k: usize) -> Vec<Vec<usize>> {
        let n = self.micro_clusters.len();
        if n == 0 {
            return Vec::new();
        }

        let effective_k = k.min(n);

        let centers: Vec<Vec<f64>> = self.micro_clusters.iter().map(|mc| mc.center()).collect();
        let weights: Vec<u64> = self.micro_clusters.iter().map(|mc| mc.n).collect();

        let assignments = weighted_kmeans(&centers, &weights, effective_k, 100);

        // Group MC indices by their macro-cluster assignment.
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); effective_k];
        for (mc_idx, &cluster_id) in assignments.iter().enumerate() {
            groups[cluster_id].push(mc_idx);
        }

        // Remove any empty groups (can happen if K-Means produces degenerate
        // clusters with no assignments).
        groups.retain(|g| !g.is_empty());
        groups
    }

    // ─── Private helpers ────────────────────────────────────────────────

    /// Find the nearest micro-cluster to a point, returning (index, distance).
    fn nearest_mc(&self, point: &[f64]) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (i, mc) in self.micro_clusters.iter().enumerate() {
            let d = euclidean_distance(point, &mc.center());
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        (best_idx, best_dist)
    }

    /// Find the two closest micro-clusters, returning their indices.
    fn closest_mc_pair(&self) -> (usize, usize) {
        let n = self.micro_clusters.len();
        debug_assert!(n >= 2, "need at least 2 micro-clusters to find a pair");
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_dist = f64::MAX;

        let centers: Vec<Vec<f64>> = self.micro_clusters.iter().map(|mc| mc.center()).collect();

        for i in 0..n {
            for j in (i + 1)..n {
                let d = euclidean_distance(&centers[i], &centers[j]);
                if d < best_dist {
                    best_dist = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        (best_i, best_j)
    }
}

// ─── Weighted K-Means (private) ─────────────────────────────────────────

/// Run weighted K-Means clustering on a set of points.
///
/// Returns an assignment vector mapping each point index to a cluster index
/// in `0..k`. Centroids are initialized from the `k` points with the
/// largest weights, and the algorithm runs for at most `max_iter` iterations
/// or until convergence (no assignment changes).
fn weighted_kmeans(centers: &[Vec<f64>], weights: &[u64], k: usize, max_iter: usize) -> Vec<usize> {
    let n = centers.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let effective_k = k.min(n);
    let d = centers[0].len();

    // Initialize centroids from the k most populated points.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| weights[b].cmp(&weights[a]));
    let mut centroids: Vec<Vec<f64>> = order[..effective_k]
        .iter()
        .map(|&i| centers[i].clone())
        .collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        // Assignment step: each point goes to the nearest centroid.
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0;
            let mut best_dist = f64::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = euclidean_distance(&centers[i], centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step: recompute centroids as weighted means.
        let mut new_centroids = vec![vec![0.0; d]; effective_k];
        let mut total_weight = vec![0.0_f64; effective_k];

        for i in 0..n {
            let c = assignments[i];
            let w = weights[i] as f64;
            total_weight[c] += w;
            for (j, &v) in centers[i].iter().enumerate() {
                new_centroids[c][j] += w * v;
            }
        }

        for c in 0..effective_k {
            if total_weight[c] > 0.0 {
                for val in new_centroids[c].iter_mut().take(d) {
                    *val /= total_weight[c];
                }
            }
            // If a centroid has no assigned points, keep the old one.
            if total_weight[c] == 0.0 {
                new_centroids[c] = centroids[c].clone();
            }
        }

        centroids = new_centroids;
    }

    assignments
}

// ─── Euclidean distance helper ──────────────────────────────────────────

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch in euclidean_distance");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    /// Deterministic PRNG for test data generation (no external deps).
    fn xorshift64(state: &mut u64) -> f64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        (x as f64) / (u64::MAX as f64)
    }

    #[test]
    fn single_point_creates_micro_cluster() {
        let config = CluStreamConfig::builder(5).build().unwrap();
        let mut cs = CluStream::new(config);
        cs.train_one(&[1.0, 2.0, 3.0]);
        assert_eq!(cs.n_micro_clusters(), 1);
        assert_eq!(cs.micro_clusters()[0].n, 1);
        assert_eq!(cs.n_samples_seen(), 1);
    }

    #[test]
    fn cluster_feature_absorb() {
        let mut cf = ClusterFeature::new(2);
        cf.absorb(&[1.0, 4.0]);
        cf.absorb(&[3.0, 6.0]);
        cf.absorb(&[2.0, 5.0]);
        assert_eq!(cf.n, 3);
        // Center should be mean: (1+3+2)/3 = 2.0, (4+6+5)/3 = 5.0
        let center = cf.center();
        assert!(approx_eq(center[0], 2.0));
        assert!(approx_eq(center[1], 5.0));
    }

    #[test]
    fn cluster_feature_merge() {
        let mut cf1 = ClusterFeature::new(2);
        cf1.absorb(&[1.0, 2.0]);
        cf1.absorb(&[3.0, 4.0]);

        let mut cf2 = ClusterFeature::new(2);
        cf2.absorb(&[5.0, 6.0]);

        cf1.merge(&cf2);

        assert_eq!(cf1.n, 3);
        // LS = (1+3+5, 2+4+6) = (9, 12)
        assert!(approx_eq(cf1.linear_sum[0], 9.0));
        assert!(approx_eq(cf1.linear_sum[1], 12.0));
        // SS = (1+9+25, 4+16+36) = (35, 56)
        assert!(approx_eq(cf1.squared_sum[0], 35.0));
        assert!(approx_eq(cf1.squared_sum[1], 56.0));
        // Center = (3, 4)
        let center = cf1.center();
        assert!(approx_eq(center[0], 3.0));
        assert!(approx_eq(center[1], 4.0));
    }

    #[test]
    fn nearby_points_absorbed() {
        let config = CluStreamConfig::builder(5)
            .max_radius_factor(4.0)
            .build()
            .unwrap();
        let mut cs = CluStream::new(config);

        // First 3 points establish a micro-cluster with nonzero radius (~0.008).
        cs.train_one(&[1.0, 1.0]);
        cs.train_one(&[1.01, 1.01]);
        cs.train_one(&[0.99, 0.99]);
        // A nearby fourth point (distance ~0.028, threshold = 4.0 * 0.008 = 0.033)
        // should be absorbed rather than creating a new MC.
        cs.train_one(&[1.02, 1.02]);

        assert_eq!(cs.n_micro_clusters(), 1);
        assert_eq!(cs.micro_clusters()[0].n, 4);
    }

    #[test]
    fn distant_point_creates_new_mc() {
        let config = CluStreamConfig::builder(5)
            .max_radius_factor(2.0)
            .build()
            .unwrap();
        let mut cs = CluStream::new(config);

        // Build a tight cluster first (need n >= 3 for meaningful radius).
        cs.train_one(&[0.0, 0.0]);
        cs.train_one(&[0.01, 0.01]);
        cs.train_one(&[0.02, 0.02]);
        // This point is very far away -- should create a new MC.
        cs.train_one(&[100.0, 100.0]);

        assert_eq!(cs.n_micro_clusters(), 2);
    }

    #[test]
    fn max_micro_clusters_triggers_merge() {
        let config = CluStreamConfig::builder(3)
            .max_radius_factor(2.0)
            .build()
            .unwrap();
        let mut cs = CluStream::new(config);

        // Create 3 well-separated micro-clusters.
        // Each needs >= 3 points so the radius becomes meaningful and
        // subsequent distant points are not absorbed.
        for _ in 0..3 {
            cs.train_one(&[0.0, 0.0]);
        }
        for _ in 0..3 {
            cs.train_one(&[50.0, 50.0]);
        }
        for _ in 0..3 {
            cs.train_one(&[100.0, 100.0]);
        }
        assert_eq!(cs.n_micro_clusters(), 3);

        // Adding a distant 4th cluster-seed should trigger a merge,
        // keeping us at max_micro_clusters = 3.
        cs.train_one(&[200.0, 200.0]);
        assert_eq!(cs.n_micro_clusters(), 3);
    }

    #[test]
    fn macro_clusters_separates_groups() {
        let config = CluStreamConfig::builder(20)
            .max_radius_factor(2.0)
            .build()
            .unwrap();
        let mut cs = CluStream::new(config);

        // Generate two well-separated clusters with deterministic noise.
        let mut rng_state: u64 = 12345;

        // Cluster A around (0, 0)
        for _ in 0..50 {
            let x = xorshift64(&mut rng_state) * 2.0 - 1.0; // [-1, 1]
            let y = xorshift64(&mut rng_state) * 2.0 - 1.0;
            cs.train_one(&[x, y]);
        }

        // Cluster B around (100, 100)
        for _ in 0..50 {
            let x = 100.0 + xorshift64(&mut rng_state) * 2.0 - 1.0;
            let y = 100.0 + xorshift64(&mut rng_state) * 2.0 - 1.0;
            cs.train_one(&[x, y]);
        }

        let groups = cs.macro_clusters(2);
        assert_eq!(groups.len(), 2);

        // Verify separation: each group's MC centers should be in one region.
        for group in &groups {
            let centers: Vec<Vec<f64>> = group
                .iter()
                .map(|&idx| cs.micro_clusters()[idx].center())
                .collect();
            // All centers in a group should be on the same side of x=50.
            let all_low = centers.iter().all(|c| c[0] < 50.0);
            let all_high = centers.iter().all(|c| c[0] >= 50.0);
            assert!(
                all_low || all_high,
                "macro-cluster group mixes centers from both regions"
            );
        }
    }

    #[test]
    fn predict_nearest_micro_cluster() {
        let config = CluStreamConfig::builder(10)
            .max_radius_factor(2.0)
            .build()
            .unwrap();
        let mut cs = CluStream::new(config);

        // Create two well-separated MCs.
        for _ in 0..5 {
            cs.train_one(&[0.0, 0.0]);
        }
        for _ in 0..5 {
            cs.train_one(&[100.0, 100.0]);
        }

        // Points near (0,0) should predict the first MC.
        let idx_near_origin = cs.predict(&[0.1, 0.1]);
        let center_origin = cs.micro_clusters()[idx_near_origin].center();
        assert!(
            center_origin[0] < 50.0,
            "expected prediction near origin, got center {:?}",
            center_origin
        );

        // Points near (100,100) should predict the second MC.
        let idx_near_far = cs.predict(&[99.9, 99.9]);
        let center_far = cs.micro_clusters()[idx_near_far].center();
        assert!(
            center_far[0] >= 50.0,
            "expected prediction near (100,100), got center {:?}",
            center_far
        );
    }

    #[test]
    fn reset_clears_state() {
        let config = CluStreamConfig::builder(5).build().unwrap();
        let mut cs = CluStream::new(config);

        cs.train_one(&[1.0, 2.0]);
        cs.train_one(&[3.0, 4.0]);
        assert_eq!(cs.n_micro_clusters(), 1);
        assert_eq!(cs.n_samples_seen(), 2);

        cs.reset();
        assert_eq!(cs.n_micro_clusters(), 0);
        assert_eq!(cs.n_samples_seen(), 0);
    }

    #[test]
    fn config_builder_validates() {
        // max_micro_clusters < 2 should fail.
        let result = CluStreamConfig::builder(1).build();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be >= 2"));

        let result = CluStreamConfig::builder(0).build();
        assert!(result.is_err());

        // max_micro_clusters = 2 should succeed.
        let result = CluStreamConfig::builder(2).build();
        assert!(result.is_ok());
    }
}
