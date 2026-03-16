//! Streaming Mini-batch K-Means clustering.
//!
//! An online K-Means implementation based on the mini-batch approach
//! (Sculley, 2010) adapted for single-sample streaming. Each incoming
//! sample is assigned to the nearest centroid, which is then updated
//! with a decaying learning rate proportional to 1/count.
//!
//! Initialization is lazy: the first `k` distinct samples become the
//! initial centroids. An optional forgetting factor enables adaptation
//! to non-stationary data streams by exponentially decaying centroid
//! counts after each update.

/// Squared Euclidean distance between two slices of equal length.
fn euclidean_distance_squared(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(ai, bi)| (ai - bi).powi(2)).sum()
}

/// Builder for [`StreamingKMeansConfig`].
///
/// Constructed via [`StreamingKMeansConfig::builder`]. Call `.build()` to
/// validate and produce the final configuration.
///
/// # Example
///
/// ```
/// use irithyll::clustering::kmeans::StreamingKMeansConfig;
///
/// let config = StreamingKMeansConfig::builder(3)
///     .forgetting_factor(0.95)
///     .build()
///     .unwrap();
/// assert_eq!(config.k, 3);
/// ```
#[derive(Debug, Clone)]
pub struct StreamingKMeansConfigBuilder {
    k: usize,
    forgetting_factor: f64,
    seed: Option<u64>,
}

impl StreamingKMeansConfigBuilder {
    /// Set the forgetting factor in (0, 1].
    ///
    /// Values less than 1.0 cause exponential decay of centroid counts,
    /// allowing centroids to adapt faster to recent data. A value of 1.0
    /// (the default) disables forgetting entirely.
    pub fn forgetting_factor(mut self, f: f64) -> Self {
        self.forgetting_factor = f;
        self
    }

    /// Set an optional seed for deterministic behavior.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate and build the configuration.
    ///
    /// Returns an error if `k < 1` or `forgetting_factor` is outside (0, 1].
    pub fn build(self) -> Result<StreamingKMeansConfig, String> {
        if self.k < 1 {
            return Err("k must be >= 1".to_string());
        }
        if self.forgetting_factor <= 0.0 || self.forgetting_factor > 1.0 {
            return Err(format!(
                "forgetting_factor must be in (0, 1], got {}",
                self.forgetting_factor
            ));
        }
        Ok(StreamingKMeansConfig {
            k: self.k,
            forgetting_factor: self.forgetting_factor,
            seed: self.seed,
        })
    }
}

/// Configuration for [`StreamingKMeans`].
///
/// Use [`StreamingKMeansConfig::builder`] to construct with validation.
#[derive(Debug, Clone)]
pub struct StreamingKMeansConfig {
    /// Number of clusters.
    pub k: usize,
    /// Optional forgetting factor in (0, 1]. 1.0 = no forgetting (default).
    /// Lower values cause centroids to adapt faster to recent data.
    pub forgetting_factor: f64,
    /// Optional seed for deterministic behavior (used if random init is added later).
    pub seed: Option<u64>,
}

impl StreamingKMeansConfig {
    /// Create a new builder with the given number of clusters.
    pub fn builder(k: usize) -> StreamingKMeansConfigBuilder {
        StreamingKMeansConfigBuilder {
            k,
            forgetting_factor: 1.0,
            seed: None,
        }
    }
}

/// Online streaming K-Means clustering.
///
/// Implements the mini-batch K-Means algorithm (Sculley, 2010) adapted for
/// single-sample streaming updates. Centroids are lazily initialized from
/// the first `k` distinct samples observed.
///
/// # Algorithm
///
/// For each incoming sample:
/// 1. If fewer than `k` centroids exist and the sample is distinct from
///    all current centroids, add it as a new centroid.
/// 2. Otherwise, find the nearest centroid by squared Euclidean distance.
/// 3. Increment the centroid's count.
/// 4. Update the centroid: `c[j] += (1 / count[j]) * (x - c[j])`.
/// 5. If a forgetting factor `f < 1.0` is set, multiply all counts by `f`.
///
/// # Example
///
/// ```
/// use irithyll::clustering::kmeans::{StreamingKMeans, StreamingKMeansConfig};
///
/// let config = StreamingKMeansConfig::builder(2).build().unwrap();
/// let mut km = StreamingKMeans::new(config);
///
/// // Initialize centroids with two distinct points
/// km.train_one(&[0.0, 0.0]);
/// km.train_one(&[10.0, 10.0]);
/// assert!(km.is_initialized());
///
/// // Predict cluster assignment
/// let cluster = km.predict(&[1.0, 1.0]);
/// assert_eq!(cluster, 0);
/// ```
#[derive(Debug, Clone)]
pub struct StreamingKMeans {
    config: StreamingKMeansConfig,
    centroids: Vec<Vec<f64>>,
    counts: Vec<f64>, // f64 for forgetting factor decay
    n_samples: u64,
    initialized: bool, // true once k centroids are set
}

impl StreamingKMeans {
    /// Create a new streaming K-Means instance with the given configuration.
    pub fn new(config: StreamingKMeansConfig) -> Self {
        Self {
            centroids: Vec::with_capacity(config.k),
            counts: Vec::with_capacity(config.k),
            n_samples: 0,
            initialized: false,
            config,
        }
    }

    /// Process a single sample: assign to the nearest centroid and update it.
    ///
    /// During initialization (fewer than `k` centroids), distinct samples
    /// are added as new centroids. Once all `k` centroids are set, each
    /// sample updates the nearest centroid with a decaying learning rate.
    pub fn train_one(&mut self, features: &[f64]) {
        self.n_samples += 1;

        // Initialization phase: collect first k distinct samples as centroids.
        if !self.initialized {
            let is_duplicate = self.centroids.iter().any(|c| {
                c.len() == features.len()
                    && c.iter()
                        .zip(features)
                        .all(|(ci, fi)| (ci - fi).abs() < f64::EPSILON)
            });

            if !is_duplicate {
                self.centroids.push(features.to_vec());
                self.counts.push(1.0);
                if self.centroids.len() == self.config.k {
                    self.initialized = true;
                }
                return;
            }

            // Duplicate sample during init: still assign to nearest if we
            // have at least one centroid.
            if self.centroids.is_empty() {
                return;
            }
        }

        // Find nearest centroid.
        let nearest = self.nearest_centroid(features);

        // Update centroid with learning rate eta = 1 / count.
        self.counts[nearest] += 1.0;
        let eta = 1.0 / self.counts[nearest];
        let centroid = &mut self.centroids[nearest];
        for (ci, fi) in centroid.iter_mut().zip(features) {
            *ci += eta * (fi - *ci);
        }

        // Apply forgetting factor to all counts.
        if self.config.forgetting_factor < 1.0 {
            for count in &mut self.counts {
                *count *= self.config.forgetting_factor;
            }
        }
    }

    /// Return the index of the nearest centroid to the given features.
    ///
    /// # Panics
    ///
    /// Panics if no centroids have been initialized (0 samples seen).
    pub fn predict(&self, features: &[f64]) -> usize {
        assert!(
            !self.centroids.is_empty(),
            "cannot predict: no centroids initialized"
        );
        self.nearest_centroid(features)
    }

    /// Return the nearest centroid index and the squared Euclidean distance.
    ///
    /// # Panics
    ///
    /// Panics if no centroids have been initialized.
    pub fn predict_distance(&self, features: &[f64]) -> (usize, f64) {
        assert!(
            !self.centroids.is_empty(),
            "cannot predict: no centroids initialized"
        );
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (i, c) in self.centroids.iter().enumerate() {
            let d = euclidean_distance_squared(features, c);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        (best_idx, best_dist)
    }

    /// Return a reference to the current centroid positions.
    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }

    /// Return the assignment counts for each cluster, rounded to `u64`.
    ///
    /// Counts may be fractional internally when a forgetting factor is
    /// applied; this method rounds each to the nearest integer.
    pub fn cluster_counts(&self) -> Vec<u64> {
        self.counts.iter().map(|c| c.round() as u64).collect()
    }

    /// Return the current number of initialized centroids.
    ///
    /// During initialization this may be less than `k`.
    pub fn n_clusters(&self) -> usize {
        self.centroids.len()
    }

    /// Total number of samples observed via [`train_one`](Self::train_one).
    pub fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Whether all `k` centroids have been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Compute inertia (sum of squared distances) over the given data.
    ///
    /// Each sample is assigned to its nearest centroid and the squared
    /// Euclidean distance is summed. This is a common measure of cluster
    /// quality — lower is better.
    ///
    /// # Panics
    ///
    /// Panics if no centroids have been initialized.
    pub fn inertia(&self, data: &[&[f64]]) -> f64 {
        assert!(
            !self.centroids.is_empty(),
            "cannot compute inertia: no centroids initialized"
        );
        data.iter()
            .map(|sample| {
                self.centroids
                    .iter()
                    .map(|c| euclidean_distance_squared(sample, c))
                    .fold(f64::MAX, f64::min)
            })
            .sum()
    }

    /// Reset all state to the initial empty configuration.
    pub fn reset(&mut self) {
        self.centroids.clear();
        self.counts.clear();
        self.n_samples = 0;
        self.initialized = false;
    }

    /// Find the index of the nearest centroid to `features`.
    fn nearest_centroid(&self, features: &[f64]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (i, c) in self.centroids.iter().enumerate() {
            let d = euclidean_distance_squared(features, c);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    /// Simple deterministic pseudo-random number generator using xorshift64.
    /// Returns values in [0, 1).
    fn xorshift64(state: &mut u64) -> f64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        (x as f64) / (u64::MAX as f64)
    }

    #[test]
    fn initialization_from_first_k_samples() {
        let config = StreamingKMeansConfig::builder(3).build().unwrap();
        let mut km = StreamingKMeans::new(config);

        let points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        for p in &points {
            km.train_one(p);
        }

        assert!(km.is_initialized());
        assert_eq!(km.n_clusters(), 3);

        // Centroids should match the input points exactly (first k distinct samples).
        for (i, p) in points.iter().enumerate() {
            assert!(
                (km.centroids()[i][0] - p[0]).abs() < EPS,
                "centroid {} x mismatch: expected {}, got {}",
                i,
                p[0],
                km.centroids()[i][0]
            );
            assert!(
                (km.centroids()[i][1] - p[1]).abs() < EPS,
                "centroid {} y mismatch: expected {}, got {}",
                i,
                p[1],
                km.centroids()[i][1]
            );
        }
    }

    #[test]
    fn predict_nearest_centroid() {
        let config = StreamingKMeansConfig::builder(3).build().unwrap();
        let mut km = StreamingKMeans::new(config);

        // Initialize centroids at known positions.
        km.train_one(&[0.0, 0.0]);
        km.train_one(&[10.0, 0.0]);
        km.train_one(&[0.0, 10.0]);
        assert!(km.is_initialized());

        // Point near [0, 0] => cluster 0.
        assert_eq!(km.predict(&[0.1, 0.1]), 0);
        // Point near [10, 0] => cluster 1.
        assert_eq!(km.predict(&[9.9, 0.1]), 1);
        // Point near [0, 10] => cluster 2.
        assert_eq!(km.predict(&[0.1, 9.9]), 2);
    }

    #[test]
    fn centroids_converge_on_clusters() {
        let config = StreamingKMeansConfig::builder(2).build().unwrap();
        let mut km = StreamingKMeans::new(config);

        let mut state = 42u64;

        // Generate 100 points around [0, 0] and 100 points around [10, 10]
        // with small noise in [-0.5, 0.5].
        let mut samples = Vec::with_capacity(200);
        for _ in 0..100 {
            let x = (xorshift64(&mut state) - 0.5) * 1.0;
            let y = (xorshift64(&mut state) - 0.5) * 1.0;
            samples.push([x, y]);
        }
        for _ in 0..100 {
            let x = 10.0 + (xorshift64(&mut state) - 0.5) * 1.0;
            let y = 10.0 + (xorshift64(&mut state) - 0.5) * 1.0;
            samples.push([x, y]);
        }

        // Interleave the two clusters to simulate a mixed stream.
        for i in 0..100 {
            km.train_one(&samples[i]);
            km.train_one(&samples[100 + i]);
        }

        assert!(km.is_initialized());

        // Each centroid should be close to one of the two cluster centers.
        let c0 = &km.centroids()[0];
        let c1 = &km.centroids()[1];

        // Determine which centroid is near [0,0] and which is near [10,10].
        let d0_origin = euclidean_distance_squared(c0, &[0.0, 0.0]);
        let d0_far = euclidean_distance_squared(c0, &[10.0, 10.0]);

        let tolerance = 4.0; // squared distance tolerance

        if d0_origin < d0_far {
            // c0 is near origin, c1 should be near [10, 10]
            assert!(
                d0_origin < tolerance,
                "centroid 0 too far from [0,0]: squared distance = {}",
                d0_origin
            );
            let d1_far = euclidean_distance_squared(c1, &[10.0, 10.0]);
            assert!(
                d1_far < tolerance,
                "centroid 1 too far from [10,10]: squared distance = {}",
                d1_far
            );
        } else {
            // c0 is near [10, 10], c1 should be near origin
            assert!(
                d0_far < tolerance,
                "centroid 0 too far from [10,10]: squared distance = {}",
                d0_far
            );
            let d1_origin = euclidean_distance_squared(c1, &[0.0, 0.0]);
            assert!(
                d1_origin < tolerance,
                "centroid 1 too far from [0,0]: squared distance = {}",
                d1_origin
            );
        }
    }

    #[test]
    fn forgetting_factor_adapts_to_drift() {
        // Train with forgetting on cluster at [0, 0], then shift to [10, 10].
        // Centroid should move toward new data faster with low forgetting factor.
        let config_forget = StreamingKMeansConfig::builder(1)
            .forgetting_factor(0.9)
            .build()
            .unwrap();
        let config_no_forget = StreamingKMeansConfig::builder(1).build().unwrap();

        let mut km_forget = StreamingKMeans::new(config_forget);
        let mut km_no_forget = StreamingKMeans::new(config_no_forget);

        // Phase 1: train on [0, 0] for 50 samples.
        for _ in 0..50 {
            km_forget.train_one(&[0.0, 0.0]);
            km_no_forget.train_one(&[0.0, 0.0]);
        }

        // Phase 2: shift to [10, 10] for 50 samples.
        for _ in 0..50 {
            km_forget.train_one(&[10.0, 10.0]);
            km_no_forget.train_one(&[10.0, 10.0]);
        }

        // The forgetting model should have its centroid closer to [10, 10].
        let dist_forget = euclidean_distance_squared(&km_forget.centroids()[0], &[10.0, 10.0]);
        let dist_no_forget =
            euclidean_distance_squared(&km_no_forget.centroids()[0], &[10.0, 10.0]);

        assert!(
            dist_forget < dist_no_forget,
            "forgetting model should be closer to [10,10]: forget dist^2 = {}, no-forget dist^2 = {}",
            dist_forget,
            dist_no_forget
        );
    }

    #[test]
    fn predict_distance_returns_correct_distance() {
        let config = StreamingKMeansConfig::builder(2).build().unwrap();
        let mut km = StreamingKMeans::new(config);

        km.train_one(&[0.0, 0.0]);
        km.train_one(&[10.0, 0.0]);

        let (idx, dist) = km.predict_distance(&[3.0, 4.0]);
        // Distance to [0,0]: 9 + 16 = 25
        // Distance to [10,0]: 49 + 16 = 65
        assert_eq!(idx, 0);
        assert!((dist - 25.0).abs() < EPS, "expected 25.0, got {}", dist);
    }

    #[test]
    fn cluster_counts_track_assignments() {
        let config = StreamingKMeansConfig::builder(2).build().unwrap();
        let mut km = StreamingKMeans::new(config);

        // Initialize with two centroids.
        km.train_one(&[0.0, 0.0]);
        km.train_one(&[10.0, 10.0]);

        // Send 5 points near cluster 0 and 3 near cluster 1.
        for _ in 0..5 {
            km.train_one(&[0.1, 0.1]);
        }
        for _ in 0..3 {
            km.train_one(&[9.9, 9.9]);
        }

        let counts = km.cluster_counts();
        // Cluster 0: 1 (init) + 5 = 6, Cluster 1: 1 (init) + 3 = 4
        assert_eq!(counts[0], 6, "cluster 0 count mismatch: {:?}", counts);
        assert_eq!(counts[1], 4, "cluster 1 count mismatch: {:?}", counts);
        assert_eq!(km.n_samples_seen(), 10);
    }

    #[test]
    fn inertia_decreases_with_training() {
        let config = StreamingKMeansConfig::builder(2).build().unwrap();
        let mut km = StreamingKMeans::new(config);

        let mut state = 123u64;

        // Generate test data: two clusters.
        let mut test_data_vecs = Vec::new();
        for _ in 0..50 {
            let x = (xorshift64(&mut state) - 0.5) * 1.0;
            let y = (xorshift64(&mut state) - 0.5) * 1.0;
            test_data_vecs.push(vec![x, y]);
        }
        for _ in 0..50 {
            let x = 10.0 + (xorshift64(&mut state) - 0.5) * 1.0;
            let y = 10.0 + (xorshift64(&mut state) - 0.5) * 1.0;
            test_data_vecs.push(vec![x, y]);
        }
        let test_data: Vec<&[f64]> = test_data_vecs.iter().map(|v| v.as_slice()).collect();

        // Initialize with first two distinct points from test data.
        km.train_one(&test_data[0]);
        km.train_one(&test_data[50]);
        let inertia_before = km.inertia(&test_data);

        // Train on all data.
        for sample in &test_data {
            km.train_one(sample);
        }
        let inertia_after = km.inertia(&test_data);

        assert!(
            inertia_after <= inertia_before,
            "inertia should decrease with training: before = {}, after = {}",
            inertia_before,
            inertia_after
        );
    }

    #[test]
    fn reset_clears_all_state() {
        let config = StreamingKMeansConfig::builder(2).build().unwrap();
        let mut km = StreamingKMeans::new(config);

        km.train_one(&[1.0, 2.0]);
        km.train_one(&[3.0, 4.0]);
        km.train_one(&[5.0, 6.0]);

        km.reset();

        assert_eq!(km.n_samples_seen(), 0);
        assert_eq!(km.n_clusters(), 0);
        assert!(!km.is_initialized());
        assert!(km.centroids().is_empty());
        assert!(km.cluster_counts().is_empty());
    }

    #[test]
    fn config_builder_validates() {
        // k = 0 should fail.
        let result = StreamingKMeansConfig::builder(0).build();
        assert!(result.is_err(), "k=0 should fail validation");
        assert!(result.unwrap_err().contains("k must be >= 1"));

        // forgetting_factor = 0.0 should fail.
        let result = StreamingKMeansConfig::builder(3)
            .forgetting_factor(0.0)
            .build();
        assert!(result.is_err(), "forgetting_factor=0.0 should fail");

        // forgetting_factor > 1.0 should fail.
        let result = StreamingKMeansConfig::builder(3)
            .forgetting_factor(1.5)
            .build();
        assert!(result.is_err(), "forgetting_factor=1.5 should fail");

        // forgetting_factor < 0 should fail.
        let result = StreamingKMeansConfig::builder(3)
            .forgetting_factor(-0.1)
            .build();
        assert!(result.is_err(), "forgetting_factor=-0.1 should fail");

        // Valid config should succeed.
        let result = StreamingKMeansConfig::builder(5)
            .forgetting_factor(0.95)
            .seed(42)
            .build();
        assert!(result.is_ok(), "valid config should build successfully");
        let config = result.unwrap();
        assert_eq!(config.k, 5);
        assert!((config.forgetting_factor - 0.95).abs() < EPS);
        assert_eq!(config.seed, Some(42));
    }
}
