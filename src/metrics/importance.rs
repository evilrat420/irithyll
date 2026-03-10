//! Feature importance tracking from accumulated split gains.
//!
//! Maintains a running tally of gain per feature as trees are built,
//! enabling normalized importance scores and top-k queries without
//! storing individual tree structures.

/// Tracks per-feature importance via accumulated split gains.
///
/// Each time a tree splits on a feature, the gain from that split is
/// added to the feature's running total. This produces a simple but
/// effective measure of feature importance for gradient boosted trees.
///
/// # Example
///
/// ```
/// use irithyll::metrics::FeatureImportance;
///
/// let mut fi = FeatureImportance::new(3);
/// fi.update(0, 1.5);
/// fi.update(2, 0.8);
/// fi.update(0, 0.3);
///
/// let top = fi.top_k(2);
/// assert_eq!(top[0].0, 0); // feature 0 has highest gain
/// ```
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Accumulated gain per feature.
    gains: Vec<f64>,
}

impl FeatureImportance {
    /// Create a new tracker for `n_features` features, all initialized to zero.
    pub fn new(n_features: usize) -> Self {
        Self {
            gains: vec![0.0; n_features],
        }
    }

    /// Accumulate `gain` for the feature at `feature_idx`.
    ///
    /// # Panics
    ///
    /// Panics if `feature_idx >= n_features`.
    pub fn update(&mut self, feature_idx: usize, gain: f64) {
        self.gains[feature_idx] += gain;
    }

    /// Raw accumulated gains for all features.
    pub fn importances(&self) -> &[f64] {
        &self.gains
    }

    /// Importances normalized to sum to 1.0.
    ///
    /// Returns a vector of zeros if total accumulated gain is zero.
    pub fn normalized(&self) -> Vec<f64> {
        let total = self.total_gain();
        if total == 0.0 {
            return vec![0.0; self.gains.len()];
        }
        self.gains.iter().map(|&g| g / total).collect()
    }

    /// Top-k features by accumulated gain, sorted descending.
    ///
    /// Returns `(feature_index, gain)` pairs. If `k > n_features`,
    /// all features are returned.
    pub fn top_k(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .gains
            .iter()
            .enumerate()
            .map(|(i, &g)| (i, g))
            .collect();

        // Sort descending by gain, with stable index ordering for ties
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        indexed.truncate(k);
        indexed
    }

    /// Number of features being tracked.
    pub fn n_features(&self) -> usize {
        self.gains.len()
    }

    /// Total accumulated gain across all features.
    pub fn total_gain(&self) -> f64 {
        self.gains.iter().sum()
    }

    /// Reset all feature importances to zero.
    pub fn reset(&mut self) {
        self.gains.iter_mut().for_each(|g| *g = 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn empty_state() {
        let fi = FeatureImportance::new(5);
        assert_eq!(fi.n_features(), 5);
        assert!(approx_eq(fi.total_gain(), 0.0));
        assert!(fi.importances().iter().all(|&g| g == 0.0));
    }

    #[test]
    fn single_update() {
        let mut fi = FeatureImportance::new(3);
        fi.update(1, 2.5);
        assert!(approx_eq(fi.importances()[0], 0.0));
        assert!(approx_eq(fi.importances()[1], 2.5));
        assert!(approx_eq(fi.importances()[2], 0.0));
        assert!(approx_eq(fi.total_gain(), 2.5));
    }

    #[test]
    fn multiple_updates_accumulate() {
        let mut fi = FeatureImportance::new(3);
        fi.update(0, 1.0);
        fi.update(0, 0.5);
        fi.update(2, 3.0);
        assert!(approx_eq(fi.importances()[0], 1.5));
        assert!(approx_eq(fi.importances()[1], 0.0));
        assert!(approx_eq(fi.importances()[2], 3.0));
        assert!(approx_eq(fi.total_gain(), 4.5));
    }

    #[test]
    fn normalized_sums_to_one() {
        let mut fi = FeatureImportance::new(4);
        fi.update(0, 2.0);
        fi.update(1, 3.0);
        fi.update(2, 1.0);
        fi.update(3, 4.0);
        let norm = fi.normalized();
        let sum: f64 = norm.iter().sum();
        assert!(approx_eq(sum, 1.0));
        assert!(approx_eq(norm[0], 0.2));
        assert!(approx_eq(norm[1], 0.3));
        assert!(approx_eq(norm[2], 0.1));
        assert!(approx_eq(norm[3], 0.4));
    }

    #[test]
    fn normalized_zero_total_returns_zeros() {
        let fi = FeatureImportance::new(3);
        let norm = fi.normalized();
        assert_eq!(norm.len(), 3);
        assert!(norm.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn top_k_ordering() {
        let mut fi = FeatureImportance::new(5);
        fi.update(0, 1.0);
        fi.update(1, 5.0);
        fi.update(2, 3.0);
        fi.update(3, 0.5);
        fi.update(4, 4.0);

        let top3 = fi.top_k(3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].0, 1); // gain=5.0
        assert_eq!(top3[1].0, 4); // gain=4.0
        assert_eq!(top3[2].0, 2); // gain=3.0
        assert!(approx_eq(top3[0].1, 5.0));
        assert!(approx_eq(top3[1].1, 4.0));
        assert!(approx_eq(top3[2].1, 3.0));
    }

    #[test]
    fn top_k_exceeds_n_features() {
        let mut fi = FeatureImportance::new(2);
        fi.update(0, 1.0);
        fi.update(1, 2.0);

        let top = fi.top_k(10);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 0);
    }

    #[test]
    fn reset_clears_all_gains() {
        let mut fi = FeatureImportance::new(3);
        fi.update(0, 5.0);
        fi.update(1, 3.0);
        fi.update(2, 1.0);
        fi.reset();
        assert!(approx_eq(fi.total_gain(), 0.0));
        assert!(fi.importances().iter().all(|&g| g == 0.0));
        assert_eq!(fi.n_features(), 3); // size preserved
    }

    #[test]
    fn top_k_empty_gains() {
        let fi = FeatureImportance::new(4);
        let top = fi.top_k(2);
        assert_eq!(top.len(), 2);
        // All gains are 0, so order is by index (stable sort)
        assert!(top.iter().all(|(_, g)| *g == 0.0));
    }

    #[test]
    fn top_k_zero_k() {
        let mut fi = FeatureImportance::new(3);
        fi.update(0, 1.0);
        let top = fi.top_k(0);
        assert!(top.is_empty());
    }
}
