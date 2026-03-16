//! DBSTREAM density-based streaming clustering (Hahsler & Bolanos, 2016).
//!
//! Maintains weighted micro-clusters and a shared-density graph that enables
//! macro-cluster formation via connected-component analysis. Micro-cluster
//! weights and shared densities decay exponentially over time, and periodic
//! cleanup removes micro-clusters whose weight drops below a configurable
//! threshold.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Builder for [`DBStreamConfig`].
///
/// Created via [`DBStreamConfig::builder`].
#[derive(Debug, Clone)]
pub struct DBStreamConfigBuilder {
    radius: f64,
    decay_rate: f64,
    min_weight: f64,
    cleanup_interval: usize,
    shared_density_threshold: f64,
}

impl DBStreamConfigBuilder {
    /// Set the decay rate (lambda). Higher values forget faster.
    pub fn decay_rate(mut self, d: f64) -> Self {
        self.decay_rate = d;
        self
    }

    /// Set the minimum weight for micro-cluster survival.
    pub fn min_weight(mut self, w: f64) -> Self {
        self.min_weight = w;
        self
    }

    /// Set the cleanup interval in number of samples.
    pub fn cleanup_interval(mut self, n: usize) -> Self {
        self.cleanup_interval = n;
        self
    }

    /// Set the minimum shared density threshold for macro-cluster merging
    /// (expressed as a fraction of combined micro-cluster weight).
    pub fn shared_density_threshold(mut self, t: f64) -> Self {
        self.shared_density_threshold = t;
        self
    }

    /// Validate and build the configuration.
    ///
    /// Returns `Err` if any parameter is out of range.
    pub fn build(self) -> Result<DBStreamConfig, String> {
        if self.radius <= 0.0 {
            return Err("radius must be > 0".to_string());
        }
        if self.decay_rate <= 0.0 {
            return Err("decay_rate must be > 0".to_string());
        }
        if self.min_weight < 0.0 {
            return Err("min_weight must be >= 0".to_string());
        }
        if self.cleanup_interval == 0 {
            return Err("cleanup_interval must be > 0".to_string());
        }
        if self.shared_density_threshold < 0.0 || self.shared_density_threshold > 1.0 {
            return Err("shared_density_threshold must be in [0, 1]".to_string());
        }
        Ok(DBStreamConfig {
            radius: self.radius,
            decay_rate: self.decay_rate,
            min_weight: self.min_weight,
            cleanup_interval: self.cleanup_interval,
            shared_density_threshold: self.shared_density_threshold,
        })
    }
}

/// Configuration for [`DBStream`].
///
/// Use [`DBStreamConfig::builder`] to construct with sensible defaults and
/// validation.
///
/// # Example
///
/// ```
/// use irithyll::clustering::dbstream::DBStreamConfig;
///
/// let config = DBStreamConfig::builder(0.5)
///     .decay_rate(0.01)
///     .min_weight(0.5)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DBStreamConfig {
    /// Radius for micro-cluster neighborhood.
    pub radius: f64,
    /// Decay rate (lambda). Higher values forget faster.
    pub decay_rate: f64,
    /// Minimum weight threshold for micro-cluster survival.
    pub min_weight: f64,
    /// Cleanup interval (in samples). Remove dead MCs every N samples.
    pub cleanup_interval: usize,
    /// Minimum shared density for macro-cluster merging (as fraction of
    /// combined weight).
    pub shared_density_threshold: f64,
}

impl DBStreamConfig {
    /// Create a builder with the given micro-cluster radius.
    pub fn builder(radius: f64) -> DBStreamConfigBuilder {
        DBStreamConfigBuilder {
            radius,
            decay_rate: 0.001,
            min_weight: 1.0,
            cleanup_interval: 100,
            shared_density_threshold: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Micro-cluster
// ---------------------------------------------------------------------------

/// A single micro-cluster maintained by [`DBStream`].
///
/// Represents a weighted centroid in feature space. The weight decays over
/// time and micro-clusters whose weight falls below the configured threshold
/// are removed during cleanup.
#[derive(Debug, Clone)]
pub struct MicroCluster {
    /// Centroid coordinates.
    pub center: Vec<f64>,
    /// Current (decayed) weight.
    pub weight: f64,
    /// Sample index at which this micro-cluster was created.
    pub creation_time: u64,
}

// ---------------------------------------------------------------------------
// DBStream
// ---------------------------------------------------------------------------

/// Density-based streaming clustering.
///
/// Maintains a set of weighted micro-clusters and a shared-density graph.
/// Points that fall within `radius` of an existing micro-cluster are merged
/// into it; otherwise a new micro-cluster is created. Shared density between
/// pairs of micro-clusters accumulates when a single point is within range
/// of both. Macro-clusters are formed by finding connected components in the
/// shared-density graph.
///
/// # Example
///
/// ```
/// use irithyll::clustering::dbstream::{DBStreamConfig, DBStream};
///
/// let config = DBStreamConfig::builder(1.0).build().unwrap();
/// let mut db = DBStream::new(config);
///
/// db.train_one(&[0.0, 0.0]);
/// db.train_one(&[0.1, 0.1]);
/// db.train_one(&[10.0, 10.0]);
///
/// assert_eq!(db.n_micro_clusters(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct DBStream {
    config: DBStreamConfig,
    micro_clusters: Vec<MicroCluster>,
    /// Shared density between micro-cluster pairs. Keys are `(min_id, max_id)`
    /// referencing positions in `micro_clusters`.
    shared_density: HashMap<(usize, usize), f64>,
    n_samples: u64,
}

impl DBStream {
    /// Create a new `DBStream` instance from the given configuration.
    pub fn new(config: DBStreamConfig) -> Self {
        Self {
            config,
            micro_clusters: Vec::new(),
            shared_density: HashMap::new(),
            n_samples: 0,
        }
    }

    /// Process a single sample, updating micro-clusters and shared density.
    ///
    /// This is the core DBSTREAM algorithm:
    /// 1. Find all micro-clusters within `radius` of the point.
    /// 2. If at least one is in range, merge into the nearest and update
    ///    shared density for all in-range pairs.
    /// 3. Otherwise, create a new micro-cluster at the point's location.
    /// 4. Apply exponential decay to all weights and shared densities.
    /// 5. Periodically clean up micro-clusters below `min_weight`.
    pub fn train_one(&mut self, features: &[f64]) {
        self.n_samples += 1;

        // Step 1: find all MCs within radius
        let mut in_range: Vec<(usize, f64)> = Vec::new();
        for (i, mc) in self.micro_clusters.iter().enumerate() {
            let d = euclidean_distance(&mc.center, features);
            if d <= self.config.radius {
                in_range.push((i, d));
            }
        }

        if !in_range.is_empty() {
            // Step 2a: merge into nearest MC
            let nearest_idx = in_range
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;

            let mc = &mut self.micro_clusters[nearest_idx];
            let new_weight = mc.weight + 1.0;
            for (c, f) in mc.center.iter_mut().zip(features.iter()) {
                *c = (*c * mc.weight + f) / new_weight;
            }
            mc.weight = new_weight;

            // Step 2b: update shared density for all pairs of in-range MCs
            for i in 0..in_range.len() {
                for j in (i + 1)..in_range.len() {
                    let a = in_range[i].0;
                    let b = in_range[j].0;
                    let key = make_pair_key(a, b);
                    *self.shared_density.entry(key).or_insert(0.0) += 1.0;
                }
            }
        } else {
            // Step 3: create new micro-cluster
            self.micro_clusters.push(MicroCluster {
                center: features.to_vec(),
                weight: 1.0,
                creation_time: self.n_samples,
            });
        }

        // Step 4: apply decay
        let decay_factor = 2.0_f64.powf(-self.config.decay_rate);
        for mc in &mut self.micro_clusters {
            mc.weight *= decay_factor;
        }
        for sd in self.shared_density.values_mut() {
            *sd *= decay_factor;
        }

        // Step 5: periodic cleanup
        if self.n_samples % self.config.cleanup_interval as u64 == 0 {
            self.cleanup();
        }
    }

    /// Assign the given point to the nearest micro-cluster.
    ///
    /// Returns the index into `micro_clusters()`.
    ///
    /// # Panics
    ///
    /// Panics if there are no micro-clusters.
    pub fn predict(&self, features: &[f64]) -> usize {
        assert!(
            !self.micro_clusters.is_empty(),
            "cannot predict with no micro-clusters"
        );
        self.micro_clusters
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = euclidean_distance(&a.center, features);
                let db = euclidean_distance(&b.center, features);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
            .0
    }

    /// Assign the given point to the nearest micro-cluster, or return `None`
    /// if the point is further than `noise_radius` from all micro-clusters.
    pub fn predict_or_noise(&self, features: &[f64], noise_radius: f64) -> Option<usize> {
        let mut best_idx = None;
        let mut best_dist = f64::INFINITY;

        for (i, mc) in self.micro_clusters.iter().enumerate() {
            let d = euclidean_distance(&mc.center, features);
            if d < best_dist {
                best_dist = d;
                best_idx = Some(i);
            }
        }

        if best_dist <= noise_radius {
            best_idx
        } else {
            None
        }
    }

    /// Return a reference to the current micro-clusters.
    pub fn micro_clusters(&self) -> &[MicroCluster] {
        &self.micro_clusters
    }

    /// Number of active micro-clusters.
    pub fn n_micro_clusters(&self) -> usize {
        self.micro_clusters.len()
    }

    /// Group micro-clusters into macro-clusters via connected components in
    /// the shared-density graph.
    ///
    /// Two micro-clusters `i` and `j` are connected if their shared density
    /// exceeds `shared_density_threshold * (weight_i + weight_j)`. Returns
    /// a `Vec` of groups, where each group is a `Vec<usize>` of micro-cluster
    /// indices.
    pub fn macro_clusters(&self) -> Vec<Vec<usize>> {
        let n = self.micro_clusters.len();
        if n == 0 {
            return Vec::new();
        }

        // Build adjacency list
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (&(i, j), &sd) in &self.shared_density {
            // Both indices must be in range (they should be after cleanup)
            if i >= n || j >= n {
                continue;
            }
            let combined_weight = self.micro_clusters[i].weight + self.micro_clusters[j].weight;
            if sd > self.config.shared_density_threshold * combined_weight {
                adj[i].push(j);
                adj[j].push(i);
            }
        }

        // DFS for connected components
        let mut visited = vec![false; n];
        let mut components: Vec<Vec<usize>> = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut component = Vec::new();
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                if visited[node] {
                    continue;
                }
                visited[node] = true;
                component.push(node);
                for &neighbor in &adj[node] {
                    if !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }

        components
    }

    /// Number of macro-clusters (connected components in the shared-density
    /// graph).
    pub fn n_clusters(&self) -> usize {
        self.macro_clusters().len()
    }

    /// Total number of samples processed so far.
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Reset all state, removing all micro-clusters and shared density.
    pub fn reset(&mut self) {
        self.micro_clusters.clear();
        self.shared_density.clear();
        self.n_samples = 0;
    }

    /// Remove micro-clusters below `min_weight` and rebuild the shared-density
    /// map with remapped indices.
    fn cleanup(&mut self) {
        // Identify which indices survive
        let mut keep_indices: Vec<usize> = Vec::new();
        for (i, mc) in self.micro_clusters.iter().enumerate() {
            if mc.weight >= self.config.min_weight {
                keep_indices.push(i);
            }
        }

        // If nothing was removed, we're done
        if keep_indices.len() == self.micro_clusters.len() {
            return;
        }

        // Build old-index -> new-index mapping
        let mut index_map: HashMap<usize, usize> = HashMap::new();
        for (new_idx, &old_idx) in keep_indices.iter().enumerate() {
            index_map.insert(old_idx, new_idx);
        }

        // Rebuild micro-clusters in-place
        let new_mcs: Vec<MicroCluster> = keep_indices
            .iter()
            .map(|&i| self.micro_clusters[i].clone())
            .collect();
        self.micro_clusters = new_mcs;

        // Rebuild shared density with remapped indices, dropping entries
        // involving removed MCs
        let mut new_sd: HashMap<(usize, usize), f64> = HashMap::new();
        for (&(old_a, old_b), &val) in &self.shared_density {
            if let (Some(&new_a), Some(&new_b)) = (index_map.get(&old_a), index_map.get(&old_b)) {
                let key = make_pair_key(new_a, new_b);
                new_sd.insert(key, val);
            }
        }
        self.shared_density = new_sd;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Euclidean distance between two feature vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Create a canonical pair key where the smaller index comes first.
fn make_pair_key(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn default_config(radius: f64) -> DBStreamConfig {
        DBStreamConfig::builder(radius)
            .decay_rate(0.001)
            .min_weight(0.0) // low threshold so cleanup doesn't interfere
            .cleanup_interval(1000)
            .build()
            .unwrap()
    }

    #[test]
    fn single_point_creates_micro_cluster() {
        let config = default_config(1.0);
        let mut db = DBStream::new(config);
        db.train_one(&[5.0, 5.0]);

        assert_eq!(db.n_micro_clusters(), 1);
        let mc = &db.micro_clusters()[0];
        assert!(approx_eq(mc.center[0], 5.0));
        assert!(approx_eq(mc.center[1], 5.0));
        assert_eq!(db.n_samples_seen(), 1);
    }

    #[test]
    fn nearby_points_merge() {
        let config = default_config(1.0);
        let mut db = DBStream::new(config);

        db.train_one(&[0.0, 0.0]);
        db.train_one(&[0.1, 0.1]);
        db.train_one(&[0.2, 0.2]);

        // All within radius 1.0, should merge into a single MC
        assert_eq!(db.n_micro_clusters(), 1);
        assert_eq!(db.n_samples_seen(), 3);
    }

    #[test]
    fn distant_points_separate() {
        let config = default_config(1.0);
        let mut db = DBStream::new(config);

        db.train_one(&[0.0, 0.0]);
        db.train_one(&[10.0, 10.0]);

        assert_eq!(db.n_micro_clusters(), 2);
    }

    #[test]
    fn decay_reduces_weights() {
        let config = DBStreamConfig::builder(1.0)
            .decay_rate(0.1) // aggressive decay for testing
            .min_weight(0.0)
            .cleanup_interval(10_000)
            .build()
            .unwrap();
        let mut db = DBStream::new(config);

        db.train_one(&[0.0, 0.0]);
        let initial_weight = db.micro_clusters()[0].weight;

        // Feed distant points to avoid merging, which applies additional decay
        for i in 1..20 {
            db.train_one(&[100.0 * i as f64, 100.0 * i as f64]);
        }

        // The first MC should have decayed significantly
        let final_weight = db.micro_clusters()[0].weight;
        assert!(
            final_weight < initial_weight,
            "expected weight to decay: initial={}, final={}",
            initial_weight,
            final_weight
        );
    }

    #[test]
    fn cleanup_removes_light_clusters() {
        let config = DBStreamConfig::builder(1.0)
            .decay_rate(0.5) // very aggressive decay
            .min_weight(0.1)
            .cleanup_interval(5)
            .build()
            .unwrap();
        let mut db = DBStream::new(config);

        // Create an isolated MC
        db.train_one(&[0.0, 0.0]);
        let initial_count = db.n_micro_clusters();
        assert_eq!(initial_count, 1);

        // Feed distant points so the original MC only decays, never gets reinforced.
        // After enough samples + cleanup, it should be removed.
        for i in 1..=20 {
            db.train_one(&[1000.0 * i as f64, 1000.0 * i as f64]);
        }

        // The original MC at (0,0) should have been cleaned up.
        // Check that its center no longer appears.
        let has_origin = db
            .micro_clusters()
            .iter()
            .any(|mc| approx_eq(mc.center[0], 0.0) && approx_eq(mc.center[1], 0.0));
        assert!(
            !has_origin,
            "expected the origin MC to be removed after decay and cleanup"
        );
    }

    #[test]
    fn macro_clusters_merge_shared_density() {
        // Two tight groups far apart. Points within each group overlap in
        // radius, building shared density. The two groups should form
        // two separate macro-clusters.
        let config = DBStreamConfig::builder(1.0)
            .decay_rate(0.0001) // very slow decay
            .min_weight(0.0)
            .cleanup_interval(10_000)
            .shared_density_threshold(0.1)
            .build()
            .unwrap();
        let mut db = DBStream::new(config);

        // Group A: points near the origin
        for _ in 0..10 {
            db.train_one(&[0.0, 0.0]);
            db.train_one(&[0.5, 0.5]);
        }

        // Group B: points far away
        for _ in 0..10 {
            db.train_one(&[10.0, 10.0]);
            db.train_one(&[10.5, 10.5]);
        }

        let macros = db.macro_clusters();
        // Should have at least 2 macro-clusters (groups A and B are not connected)
        assert!(
            macros.len() >= 2,
            "expected at least 2 macro-clusters, got {}",
            macros.len()
        );
    }

    #[test]
    fn predict_returns_nearest() {
        let config = default_config(1.0);
        let mut db = DBStream::new(config);

        db.train_one(&[0.0, 0.0]);
        db.train_one(&[10.0, 10.0]);

        // Point near origin should predict MC 0
        let idx = db.predict(&[0.1, 0.1]);
        let nearest_center = &db.micro_clusters()[idx].center;
        let d_origin = euclidean_distance(nearest_center, &[0.0, 0.0]);
        let d_far = euclidean_distance(nearest_center, &[10.0, 10.0]);
        assert!(
            d_origin < d_far,
            "predicted MC should be closer to origin than to (10,10)"
        );

        // Point near (10,10) should predict the other MC
        let idx2 = db.predict(&[9.9, 9.9]);
        let nearest_center2 = &db.micro_clusters()[idx2].center;
        let d_origin2 = euclidean_distance(nearest_center2, &[0.0, 0.0]);
        let d_far2 = euclidean_distance(nearest_center2, &[10.0, 10.0]);
        assert!(
            d_far2 < d_origin2,
            "predicted MC should be closer to (10,10) than to origin"
        );
    }

    #[test]
    fn predict_or_noise_returns_none() {
        let config = default_config(1.0);
        let mut db = DBStream::new(config);

        db.train_one(&[0.0, 0.0]);

        // Point within noise_radius
        assert!(db.predict_or_noise(&[0.5, 0.5], 2.0).is_some());

        // Point far outside noise_radius
        assert!(db.predict_or_noise(&[100.0, 100.0], 1.0).is_none());
    }

    #[test]
    fn reset_clears_state() {
        let config = default_config(1.0);
        let mut db = DBStream::new(config);

        db.train_one(&[1.0, 2.0]);
        db.train_one(&[3.0, 4.0]);
        assert!(db.n_micro_clusters() > 0);
        assert!(db.n_samples_seen() > 0);

        db.reset();

        assert_eq!(db.n_micro_clusters(), 0);
        assert_eq!(db.n_samples_seen(), 0);
        assert!(db.macro_clusters().is_empty());
    }

    #[test]
    fn config_builder_validates() {
        // radius must be > 0
        assert!(DBStreamConfig::builder(0.0).build().is_err());
        assert!(DBStreamConfig::builder(-1.0).build().is_err());

        // decay_rate must be > 0
        assert!(DBStreamConfig::builder(1.0)
            .decay_rate(0.0)
            .build()
            .is_err());
        assert!(DBStreamConfig::builder(1.0)
            .decay_rate(-1.0)
            .build()
            .is_err());

        // min_weight must be >= 0
        assert!(DBStreamConfig::builder(1.0)
            .min_weight(-1.0)
            .build()
            .is_err());

        // cleanup_interval must be > 0
        assert!(DBStreamConfig::builder(1.0)
            .cleanup_interval(0)
            .build()
            .is_err());

        // shared_density_threshold must be in [0, 1]
        assert!(DBStreamConfig::builder(1.0)
            .shared_density_threshold(-0.1)
            .build()
            .is_err());
        assert!(DBStreamConfig::builder(1.0)
            .shared_density_threshold(1.1)
            .build()
            .is_err());

        // Valid config succeeds
        assert!(DBStreamConfig::builder(1.0).build().is_ok());
    }
}
