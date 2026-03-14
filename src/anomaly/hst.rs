//! Half-Space Trees for streaming anomaly detection.
//!
//! Implements the HS-Tree algorithm from Tan, Ting & Liu (2011):
//! "Fast Anomaly Detection for Streaming Data."
//!
//! Half-Space Trees partition the feature space with random axis-aligned cuts,
//! then score anomalies by comparing mass profiles between a reference window
//! and the latest window. Points landing in low-mass regions of the reference
//! model are scored as anomalous.
//!
//! # Algorithm
//!
//! Each tree is built by:
//! 1. Randomly selecting a feature dimension
//! 2. Randomly selecting a split point within the feature's working range
//! 3. Recursively partitioning left/right until max depth
//!
//! Two mass profiles are maintained:
//! - **reference (`r`)**: mass counts from the previous window
//! - **latest (`l`)**: mass counts being accumulated in the current window
//!
//! After `window_size` samples, `l` is copied to `r` and `l` is reset.
//!
//! The anomaly score for a point is: `Σ_nodes(mass_r / 2^depth)` -- nodes
//! with low reference mass at shallow depth contribute most to anomaly score.
//!
//! # References
//!
//! Tan, S. C., Ting, K. M., & Liu, T. F. (2011). Fast anomaly detection for
//! streaming data. In *Proceedings of the Twenty-Second International Joint
//! Conference on Artificial Intelligence* (pp. 1511–1516).

/// Configuration for a Half-Space Trees ensemble.
#[derive(Debug, Clone)]
pub struct HSTConfig {
    /// Number of trees in the ensemble. More trees reduce variance.
    /// Default: 25.
    pub n_trees: usize,

    /// Maximum depth of each tree. Deeper trees give finer partitions.
    /// Default: 15.
    pub max_depth: usize,

    /// Window size for mass profile rotation. After this many samples,
    /// the latest profile becomes the reference. Default: 250.
    pub window_size: usize,

    /// Number of feature dimensions. Must be set before first use.
    pub n_features: usize,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// Anomaly score threshold -- scores above this are flagged.
    /// Default: 0.5 (after normalization to [0, 1]).
    pub threshold: f64,
}

impl HSTConfig {
    /// Create a new config with the given number of features.
    pub fn new(n_features: usize) -> Self {
        Self {
            n_trees: 25,
            max_depth: 15,
            window_size: 250,
            n_features,
            seed: 42,
            threshold: 0.5,
        }
    }

    /// Set the number of trees.
    pub fn n_trees(mut self, n: usize) -> Self {
        self.n_trees = n;
        self
    }

    /// Set the maximum depth.
    pub fn max_depth(mut self, d: usize) -> Self {
        self.max_depth = d;
        self
    }

    /// Set the window size for mass profile rotation.
    pub fn window_size(mut self, w: usize) -> Self {
        self.window_size = w;
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Set the anomaly threshold (0.0 to 1.0).
    pub fn threshold(mut self, t: f64) -> Self {
        self.threshold = t;
        self
    }
}

// ─── Internal tree structures ───────────────────────────────────────────

/// A single node in a half-space tree.
#[derive(Debug, Clone)]
struct HSTNode {
    /// Feature dimension used for the split (leaf nodes store 0 but don't use it).
    feature: usize,
    /// Split value.
    split_value: f64,
    /// Left child index (0 = no child / leaf).
    left: usize,
    /// Right child index (0 = no child / leaf).
    right: usize,
    /// Reference mass count.
    r_mass: u64,
    /// Latest mass count.
    l_mass: u64,
    /// Depth of this node.
    depth: usize,
}

/// A single half-space tree.
#[derive(Debug, Clone)]
struct HSTree {
    nodes: Vec<HSTNode>,
}

impl HSTree {
    /// Build a random tree structure using the given RNG state.
    fn build(
        n_features: usize,
        max_depth: usize,
        rng: &mut SimpleRng,
        work_ranges: &[(f64, f64)],
    ) -> Self {
        let capacity = (1 << (max_depth + 1)) - 1; // max possible nodes
        let mut nodes = Vec::with_capacity(capacity.min(4096));

        // Push root placeholder
        nodes.push(HSTNode {
            feature: 0,
            split_value: 0.0,
            left: 0,
            right: 0,
            r_mass: 0,
            l_mass: 0,
            depth: 0,
        });

        Self::build_recursive(&mut nodes, 0, 0, max_depth, n_features, rng, work_ranges);
        HSTree { nodes }
    }

    fn build_recursive(
        nodes: &mut Vec<HSTNode>,
        node_idx: usize,
        depth: usize,
        max_depth: usize,
        n_features: usize,
        rng: &mut SimpleRng,
        work_ranges: &[(f64, f64)],
    ) {
        if depth >= max_depth {
            return; // leaf
        }

        // Pick random feature
        let feature = rng.next_usize() % n_features;

        // Pick random split within the working range
        let (lo, hi) = work_ranges[feature];
        let split_value = if (hi - lo).abs() < 1e-15 {
            lo
        } else {
            lo + rng.next_f64() * (hi - lo)
        };

        // Create left and right children
        let left_idx = nodes.len();
        nodes.push(HSTNode {
            feature: 0,
            split_value: 0.0,
            left: 0,
            right: 0,
            r_mass: 0,
            l_mass: 0,
            depth: depth + 1,
        });

        let right_idx = nodes.len();
        nodes.push(HSTNode {
            feature: 0,
            split_value: 0.0,
            left: 0,
            right: 0,
            r_mass: 0,
            l_mass: 0,
            depth: depth + 1,
        });

        nodes[node_idx].feature = feature;
        nodes[node_idx].split_value = split_value;
        nodes[node_idx].left = left_idx;
        nodes[node_idx].right = right_idx;

        // Narrow working ranges for children
        let mut left_ranges: Vec<(f64, f64)> = work_ranges.to_vec();
        left_ranges[feature].1 = split_value; // left gets [lo, split)

        let mut right_ranges: Vec<(f64, f64)> = work_ranges.to_vec();
        right_ranges[feature].0 = split_value; // right gets [split, hi)

        Self::build_recursive(
            nodes,
            left_idx,
            depth + 1,
            max_depth,
            n_features,
            rng,
            &left_ranges,
        );
        Self::build_recursive(
            nodes,
            right_idx,
            depth + 1,
            max_depth,
            n_features,
            rng,
            &right_ranges,
        );
    }

    /// Insert a point, incrementing `l_mass` along the traversal path.
    fn update(&mut self, features: &[f64]) {
        let mut idx = 0;
        loop {
            let node = &mut self.nodes[idx];
            node.l_mass += 1;

            if node.left == 0 && node.right == 0 {
                break; // leaf
            }

            if features[node.feature] < node.split_value {
                idx = node.left;
            } else {
                idx = node.right;
            }
        }
    }

    /// Score a point -- lower reference mass at shallow depth = more anomalous.
    /// Returns a raw score (higher = more anomalous).
    fn score(&self, features: &[f64], max_depth: usize) -> f64 {
        let mut idx = 0;
        let mut score = 0.0;

        loop {
            let node = &self.nodes[idx];

            // Mass-based scoring: nodes with low reference mass contribute more
            // Weight by 2^(max_depth - depth) so shallower nodes count more
            let depth_weight = (1u64 << (max_depth - node.depth)) as f64;
            score += node.r_mass as f64 * depth_weight;

            if node.left == 0 && node.right == 0 {
                break;
            }

            if features[node.feature] < node.split_value {
                idx = node.left;
            } else {
                idx = node.right;
            }
        }

        score
    }

    /// Rotate: copy l_mass -> r_mass, zero l_mass.
    fn rotate(&mut self) {
        for node in &mut self.nodes {
            node.r_mass = node.l_mass;
            node.l_mass = 0;
        }
    }
}

// ─── Simple RNG (xorshift64) ───────────────────────────────────────────

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }
}

// ─── Half-Space Tree Ensemble ───────────────────────────────────────────

/// Result of scoring a single sample.
#[derive(Debug, Clone, Copy)]
pub struct AnomalyScore {
    /// Normalized anomaly score in [0, 1]. Higher = more anomalous.
    pub score: f64,
    /// Whether this score exceeds the configured threshold.
    pub is_anomaly: bool,
}

/// A Half-Space Trees ensemble for streaming anomaly detection.
///
/// # Example
///
/// ```
/// use irithyll::anomaly::hst::{HalfSpaceTree, HSTConfig};
///
/// let config = HSTConfig::new(3).n_trees(25).window_size(100);
/// let mut hst = HalfSpaceTree::new(config);
///
/// // Feed normal data to build a reference profile
/// for i in 0..200 {
///     let features = vec![i as f64 * 0.01, 0.5, 0.3];
///     hst.update(&features);
/// }
///
/// // Score a point
/// let result = hst.score(&[0.5, 0.5, 0.3]);
/// println!("anomaly score: {:.3}, is_anomaly: {}", result.score, result.is_anomaly);
/// ```
#[derive(Debug, Clone)]
pub struct HalfSpaceTree {
    config: HSTConfig,
    trees: Vec<HSTree>,
    samples_seen: u64,
    /// Maximum possible score (for normalization).
    max_score: f64,
}

impl HalfSpaceTree {
    /// Create a new Half-Space Tree ensemble.
    pub fn new(config: HSTConfig) -> Self {
        let mut rng = SimpleRng::new(config.seed);
        let n_features = config.n_features;
        let max_depth = config.max_depth;

        // Default working range [0, 1] for all features.
        // Points outside this range still work but get less granular partitioning.
        let work_ranges: Vec<(f64, f64)> = vec![(0.0, 1.0); n_features];

        let trees: Vec<HSTree> = (0..config.n_trees)
            .map(|_| HSTree::build(n_features, max_depth, &mut rng, &work_ranges))
            .collect();

        // Max score: if every node on the path has max mass (window_size),
        // weighted by depth. This is Σ_{d=0}^{D} window_size * 2^(D-d)
        let max_score = {
            let ws = config.window_size as f64;
            let mut s = 0.0;
            for d in 0..=max_depth {
                s += ws * (1u64 << (max_depth - d)) as f64;
            }
            s * config.n_trees as f64
        };

        Self {
            config,
            trees,
            samples_seen: 0,
            max_score,
        }
    }

    /// Create a new ensemble with explicit working ranges per feature.
    ///
    /// `ranges` should contain `(min, max)` for each feature dimension.
    /// This improves partitioning quality when feature ranges are known.
    pub fn with_ranges(config: HSTConfig, ranges: &[(f64, f64)]) -> Self {
        assert_eq!(
            ranges.len(),
            config.n_features,
            "ranges length must match n_features"
        );

        let mut rng = SimpleRng::new(config.seed);
        let max_depth = config.max_depth;

        let trees: Vec<HSTree> = (0..config.n_trees)
            .map(|_| HSTree::build(config.n_features, max_depth, &mut rng, ranges))
            .collect();

        let max_score = {
            let ws = config.window_size as f64;
            let mut s = 0.0;
            for d in 0..=max_depth {
                s += ws * (1u64 << (max_depth - d)) as f64;
            }
            s * config.n_trees as f64
        };

        Self {
            config,
            trees,
            samples_seen: 0,
            max_score,
        }
    }

    /// Feed a sample to the ensemble, updating mass profiles.
    ///
    /// When `window_size` samples have been seen, the latest profile
    /// rotates to become the reference and counting restarts.
    pub fn update(&mut self, features: &[f64]) {
        assert!(
            features.len() >= self.config.n_features,
            "expected at least {} features, got {}",
            self.config.n_features,
            features.len()
        );

        for tree in &mut self.trees {
            tree.update(features);
        }

        self.samples_seen += 1;

        if self.samples_seen % self.config.window_size as u64 == 0 {
            for tree in &mut self.trees {
                tree.rotate();
            }
        }
    }

    /// Score a sample for anomalousness.
    ///
    /// Returns an [`AnomalyScore`] with a normalized score in [0, 1]
    /// (inverted so higher = more anomalous) and an `is_anomaly` flag.
    pub fn score(&self, features: &[f64]) -> AnomalyScore {
        let raw: f64 = self
            .trees
            .iter()
            .map(|t| t.score(features, self.config.max_depth))
            .sum();

        // Invert: high reference mass = normal, low = anomalous
        let normalized = if self.max_score > 0.0 {
            1.0 - (raw / self.max_score)
        } else {
            0.0
        };

        // Clamp to [0, 1]
        let score = normalized.clamp(0.0, 1.0);

        AnomalyScore {
            score,
            is_anomaly: score > self.config.threshold,
        }
    }

    /// Combined update + score in one pass.
    ///
    /// The sample is scored BEFORE being incorporated into the mass profile,
    /// so the score reflects how anomalous it is relative to past data.
    pub fn score_and_update(&mut self, features: &[f64]) -> AnomalyScore {
        let result = self.score(features);
        self.update(features);
        result
    }

    /// Reset all mass profiles to zero.
    pub fn reset(&mut self) {
        for tree in &mut self.trees {
            for node in &mut tree.nodes {
                node.r_mass = 0;
                node.l_mass = 0;
            }
        }
        self.samples_seen = 0;
    }

    /// Total samples processed.
    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Number of complete window rotations performed.
    pub fn windows_completed(&self) -> u64 {
        self.samples_seen / self.config.window_size as u64
    }

    /// The configured threshold.
    pub fn threshold(&self) -> f64 {
        self.config.threshold
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_construction() {
        let config = HSTConfig::new(3).n_trees(10).max_depth(8).seed(123);
        let hst = HalfSpaceTree::new(config);
        assert_eq!(hst.samples_seen(), 0);
        assert_eq!(hst.windows_completed(), 0);
    }

    #[test]
    fn test_update_increments_count() {
        let config = HSTConfig::new(2).n_trees(5).window_size(50);
        let mut hst = HalfSpaceTree::new(config);

        for _ in 0..100 {
            hst.update(&[0.5, 0.5]);
        }

        assert_eq!(hst.samples_seen(), 100);
        assert_eq!(hst.windows_completed(), 2);
    }

    #[test]
    fn test_score_range() {
        let config = HSTConfig::new(3).n_trees(10).window_size(50).seed(42);
        let mut hst = HalfSpaceTree::new(config);

        // Build reference profile with normal data
        for i in 0..100 {
            hst.update(&[0.5 + (i as f64) * 0.001, 0.5, 0.5]);
        }

        let result = hst.score(&[0.5, 0.5, 0.5]);
        assert!(
            (0.0..=1.0).contains(&result.score),
            "score {} not in [0,1]",
            result.score
        );
    }

    #[test]
    fn test_anomaly_detection_basic() {
        let config = HSTConfig::new(2)
            .n_trees(50)
            .max_depth(10)
            .window_size(200)
            .seed(7)
            .threshold(0.5);
        let mut hst = HalfSpaceTree::new(config);

        // Train on normal data clustered around (0.5, 0.5)
        let mut rng = SimpleRng::new(99);
        for _ in 0..400 {
            let x = 0.4 + rng.next_f64() * 0.2; // [0.4, 0.6]
            let y = 0.4 + rng.next_f64() * 0.2;
            hst.update(&[x, y]);
        }

        // Normal point should have lower anomaly score
        let normal = hst.score(&[0.5, 0.5]);
        // Far-out point should have higher anomaly score
        let anomaly = hst.score(&[0.01, 0.99]);

        assert!(
            anomaly.score > normal.score,
            "anomaly score ({:.4}) should exceed normal ({:.4})",
            anomaly.score,
            normal.score
        );
    }

    #[test]
    fn test_score_and_update() {
        let config = HSTConfig::new(2).n_trees(5).window_size(50);
        let mut hst = HalfSpaceTree::new(config);

        for _ in 0..100 {
            let result = hst.score_and_update(&[0.5, 0.5]);
            assert!((0.0..=1.0).contains(&result.score));
        }

        assert_eq!(hst.samples_seen(), 100);
    }

    #[test]
    fn test_reset() {
        let config = HSTConfig::new(2).n_trees(5).window_size(50);
        let mut hst = HalfSpaceTree::new(config);

        for _ in 0..100 {
            hst.update(&[0.5, 0.5]);
        }

        hst.reset();
        assert_eq!(hst.samples_seen(), 0);
        assert_eq!(hst.windows_completed(), 0);
    }

    #[test]
    fn test_with_ranges() {
        let config = HSTConfig::new(2).n_trees(10).window_size(50).seed(42);
        let ranges = vec![(-10.0, 10.0), (0.0, 100.0)];
        let mut hst = HalfSpaceTree::with_ranges(config, &ranges);

        // Data within the specified ranges
        for i in 0..100 {
            hst.update(&[i as f64 * 0.1 - 5.0, i as f64]);
        }

        let result = hst.score(&[0.0, 50.0]);
        assert!((0.0..=1.0).contains(&result.score));
    }

    #[test]
    fn test_deterministic_with_seed() {
        let make = || {
            let config = HSTConfig::new(3).n_trees(10).seed(999).window_size(50);
            let mut hst = HalfSpaceTree::new(config);
            for i in 0..100 {
                hst.update(&[i as f64 * 0.01, 0.5, 0.3]);
            }
            hst.score(&[0.5, 0.5, 0.3]).score
        };

        let s1 = make();
        let s2 = make();
        assert!(
            (s1 - s2).abs() < 1e-12,
            "same seed should produce identical scores"
        );
    }

    #[test]
    fn test_window_rotation() {
        let config = HSTConfig::new(2).n_trees(3).window_size(10).max_depth(4);
        let mut hst = HalfSpaceTree::new(config);

        // First window
        for _ in 0..10 {
            hst.update(&[0.5, 0.5]);
        }
        assert_eq!(hst.windows_completed(), 1);

        // After rotation, reference should have mass
        // Score a normal point -- it should land where mass accumulated
        let normal = hst.score(&[0.5, 0.5]);

        // Score a very different point
        let outlier = hst.score(&[0.01, 0.01]);

        // With mass built up around [0.5, 0.5], the outlier should be more anomalous
        assert!(
            outlier.score >= normal.score,
            "outlier ({:.4}) should score >= normal ({:.4})",
            outlier.score,
            normal.score
        );
    }
}
