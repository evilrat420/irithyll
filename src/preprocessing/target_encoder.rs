//! Streaming target encoder for categorical features with Bayesian smoothing.
//!
//! [`TargetEncoder`] replaces categorical features with the smoothed mean of
//! the target variable for each observed category. A Bayesian smoothing factor
//! shrinks rare-category estimates toward the global mean, preventing
//! overfitting on categories with few observations.
//!
//! This type does **not** implement [`StreamingPreprocessor`](crate::pipeline::StreamingPreprocessor)
//! because it requires the target value during updates.
//!
//! # Encoding formula
//!
//! ```text
//! global_mean   = global_sum / global_count
//! cat_mean      = cat_sum / cat_count
//! encoded       = (cat_count * cat_mean + smoothing * global_mean) / (cat_count + smoothing)
//! ```
//!
//! Unknown categories and categories with zero observations receive the
//! global mean. If no samples have been seen at all, `0.0` is used.
//!
//! # Example
//!
//! ```
//! use irithyll::preprocessing::TargetEncoder;
//!
//! // Feature 0 is categorical, feature 1 is numeric.
//! let mut enc = TargetEncoder::new(vec![0]);
//!
//! // Train: category 1.0 → target 10, category 2.0 → target 20.
//! for _ in 0..50 {
//!     enc.update(&[1.0, 5.0], 10.0);
//!     enc.update(&[2.0, 5.0], 20.0);
//! }
//!
//! let out = enc.transform(&[1.0, 5.0]);
//! // Feature 0 is replaced with the smoothed target mean for category 1.
//! assert!((out[0] - 10.0).abs() < 1.0);
//! // Feature 1 passes through unchanged.
//! assert!((out[1] - 5.0).abs() < f64::EPSILON);
//! ```

use std::collections::HashMap;

/// Streaming target encoder with Bayesian smoothing.
///
/// Categorical features (identified by index) are replaced with the smoothed
/// running mean of the target for the corresponding category. Numeric features
/// pass through unchanged.
///
/// Category values are discretized via [`f64::to_bits`] before lookup, so each
/// distinct bit pattern maps to a separate category bucket.
#[derive(Clone, Debug)]
pub struct TargetEncoder {
    /// Which feature positions are categorical.
    categorical_indices: Vec<usize>,
    /// Per-categorical-feature map of category → (sum_target, count).
    ///
    /// `category_stats[i]` corresponds to `categorical_indices[i]`.
    /// Keys are the `u64` bit representation of the f64 feature value.
    category_stats: Vec<HashMap<u64, (f64, u64)>>,
    /// Sum of all target values seen.
    global_sum: f64,
    /// Total number of samples incorporated.
    global_count: u64,
    /// Bayesian smoothing factor. Higher values shrink category estimates
    /// more aggressively toward the global mean.
    smoothing: f64,
}

impl TargetEncoder {
    /// Create a target encoder with the given categorical feature indices
    /// and a default smoothing factor of `10.0`.
    ///
    /// ```
    /// use irithyll::preprocessing::TargetEncoder;
    /// let enc = TargetEncoder::new(vec![0, 2]);
    /// assert_eq!(enc.global_mean(), 0.0);
    /// assert!((enc.smoothing() - 10.0).abs() < f64::EPSILON);
    /// ```
    pub fn new(categorical_indices: Vec<usize>) -> Self {
        let n_cats = categorical_indices.len();
        Self {
            categorical_indices,
            category_stats: vec![HashMap::new(); n_cats],
            global_sum: 0.0,
            global_count: 0,
            smoothing: 10.0,
        }
    }

    /// Create a target encoder with custom smoothing.
    ///
    /// Higher `smoothing` values pull rare-category estimates closer to the
    /// global mean. A smoothing of `0.0` disables shrinkage entirely.
    ///
    /// # Panics
    ///
    /// Panics if `smoothing` is negative.
    ///
    /// ```
    /// use irithyll::preprocessing::TargetEncoder;
    /// let enc = TargetEncoder::with_smoothing(vec![0], 5.0);
    /// assert!((enc.smoothing() - 5.0).abs() < f64::EPSILON);
    /// ```
    pub fn with_smoothing(categorical_indices: Vec<usize>, smoothing: f64) -> Self {
        assert!(
            smoothing >= 0.0,
            "TargetEncoder: smoothing must be >= 0.0, got {}",
            smoothing
        );
        let n_cats = categorical_indices.len();
        Self {
            categorical_indices,
            category_stats: vec![HashMap::new(); n_cats],
            global_sum: 0.0,
            global_count: 0,
            smoothing,
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// The input feature indices that are treated as categorical.
    pub fn categorical_indices(&self) -> &[usize] {
        &self.categorical_indices
    }

    /// The Bayesian smoothing parameter.
    pub fn smoothing(&self) -> f64 {
        self.smoothing
    }

    /// Global mean of all targets seen so far.
    ///
    /// Returns `0.0` if no samples have been incorporated.
    pub fn global_mean(&self) -> f64 {
        if self.global_count == 0 {
            0.0
        } else {
            self.global_sum / self.global_count as f64
        }
    }

    /// Number of discovered categories for the given categorical slot.
    ///
    /// `cat_idx` is a positional index into [`categorical_indices`](Self::categorical_indices),
    /// **not** the original feature index.
    ///
    /// # Panics
    ///
    /// Panics if `cat_idx >= categorical_indices.len()`.
    pub fn n_categories(&self, cat_idx: usize) -> usize {
        self.category_stats[cat_idx].len()
    }

    // -----------------------------------------------------------------------
    // Core methods
    // -----------------------------------------------------------------------

    /// Incorporate a single sample into running statistics.
    ///
    /// Updates the global target sum/count and, for each categorical feature,
    /// the per-category sum/count. Category keys use [`f64::to_bits`].
    pub fn update(&mut self, features: &[f64], target: f64) {
        self.global_sum += target;
        self.global_count += 1;

        for (i, &feat_idx) in self.categorical_indices.iter().enumerate() {
            let key = features[feat_idx].to_bits();
            let entry = self.category_stats[i].entry(key).or_insert((0.0, 0));
            entry.0 += target;
            entry.1 += 1;
        }
    }

    /// Replace categorical features with their smoothed target-mean encoding.
    ///
    /// Non-categorical features pass through unchanged. Unknown categories
    /// receive the global mean. If no samples have been seen, `0.0` is used
    /// for all categorical positions.
    pub fn transform(&self, features: &[f64]) -> Vec<f64> {
        let global_mean = self.global_mean();
        let mut out = features.to_vec();

        for (i, &feat_idx) in self.categorical_indices.iter().enumerate() {
            let key = features[feat_idx].to_bits();

            out[feat_idx] = match self.category_stats[i].get(&key) {
                Some(&(cat_sum, cat_count)) if cat_count > 0 => {
                    let cat_mean = cat_sum / cat_count as f64;
                    (cat_count as f64 * cat_mean + self.smoothing * global_mean)
                        / (cat_count as f64 + self.smoothing)
                }
                _ => global_mean,
            };
        }

        out
    }

    /// Update statistics then transform in one step.
    ///
    /// The current sample influences its own encoding (the category count is
    /// already incremented before computing the smoothed mean).
    pub fn update_and_transform(&mut self, features: &[f64], target: f64) -> Vec<f64> {
        self.update(features, target);
        self.transform(features)
    }

    /// Clear all accumulated statistics, returning the encoder to its
    /// initial state.
    ///
    /// Preserves `categorical_indices` and `smoothing` but clears all
    /// category stats and global accumulators.
    pub fn reset(&mut self) {
        for map in &mut self.category_stats {
            map.clear();
        }
        self.global_sum = 0.0;
        self.global_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn passthrough_non_categorical() {
        // No categorical indices: everything passes through unchanged.
        let mut enc = TargetEncoder::new(vec![]);

        enc.update(&[1.0, 2.5, 3.0], 10.0);
        enc.update(&[4.0, 5.5, 6.0], 20.0);

        let out = enc.transform(&[7.0, 8.5, 9.0]);
        assert!((out[0] - 7.0).abs() < EPS);
        assert!((out[1] - 8.5).abs() < EPS);
        assert!((out[2] - 9.0).abs() < EPS);
    }

    #[test]
    fn encodes_with_target_mean() {
        // After many samples the smoothed encoding should approach the
        // category's raw target mean.
        let mut enc = TargetEncoder::new(vec![0]);

        // Category 1.0 always has target 10.0
        for _ in 0..1000 {
            enc.update(&[1.0, 5.0], 10.0);
        }
        // Category 2.0 always has target 50.0
        for _ in 0..1000 {
            enc.update(&[2.0, 5.0], 50.0);
        }

        let out1 = enc.transform(&[1.0, 5.0]);
        let out2 = enc.transform(&[2.0, 5.0]);

        // With 1000 observations per category and smoothing=10, the weight
        // toward the category mean is 1000/(1000+10) ≈ 0.99. The encoded
        // value should be very close to the raw category mean.
        assert!(
            (out1[0] - 10.0).abs() < 0.5,
            "expected ~10.0 for cat 1, got {}",
            out1[0]
        );
        assert!(
            (out2[0] - 50.0).abs() < 0.5,
            "expected ~50.0 for cat 2, got {}",
            out2[0]
        );
        // Numeric feature passes through.
        assert!((out1[1] - 5.0).abs() < EPS);
    }

    #[test]
    fn smoothing_pulls_toward_global_mean() {
        // With few samples the encoded value should sit between the category
        // mean and the global mean.
        let mut enc = TargetEncoder::with_smoothing(vec![0], 10.0);

        // Bulk category: target = 0.0 (100 observations).
        for _ in 0..100 {
            enc.update(&[0.0], 0.0);
        }
        // Rare category: target = 100.0 (2 observations).
        enc.update(&[1.0], 100.0);
        enc.update(&[1.0], 100.0);

        let global_mean = enc.global_mean();
        let cat_mean = 100.0;

        let out = enc.transform(&[1.0]);

        // encoded = (2 * 100 + 10 * global_mean) / (2 + 10)
        // It must be strictly between cat_mean and global_mean.
        assert!(
            out[0] > global_mean && out[0] < cat_mean,
            "encoded {} should be between global_mean {} and cat_mean {}",
            out[0],
            global_mean,
            cat_mean
        );
    }

    #[test]
    fn unknown_category_uses_global_mean() {
        let mut enc = TargetEncoder::new(vec![0]);

        enc.update(&[0.0, 1.0], 10.0);
        enc.update(&[1.0, 2.0], 20.0);

        let global_mean = enc.global_mean();

        // Category 99.0 was never seen.
        let out = enc.transform(&[99.0, 3.0]);
        assert!(
            (out[0] - global_mean).abs() < EPS,
            "unknown category should get global mean {}, got {}",
            global_mean,
            out[0]
        );
        // Numeric feature passes through.
        assert!((out[1] - 3.0).abs() < EPS);
    }

    #[test]
    fn update_and_transform_consistency() {
        // Calling update then transform must yield the same result as
        // update_and_transform.
        let features = [2.0, 7.5];
        let target = 42.0;

        // Path A: update_and_transform in one call.
        let mut enc_a = TargetEncoder::with_smoothing(vec![0], 5.0);
        enc_a.update(&[0.0, 1.0], 10.0);
        enc_a.update(&[1.0, 2.0], 20.0);
        let out_a = enc_a.update_and_transform(&features, target);

        // Path B: separate update + transform.
        let mut enc_b = TargetEncoder::with_smoothing(vec![0], 5.0);
        enc_b.update(&[0.0, 1.0], 10.0);
        enc_b.update(&[1.0, 2.0], 20.0);
        enc_b.update(&features, target);
        let out_b = enc_b.transform(&features);

        assert_eq!(out_a.len(), out_b.len());
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!(
                (a - b).abs() < EPS,
                "mismatch: update_and_transform={}, update+transform={}",
                a,
                b
            );
        }
    }

    #[test]
    fn reset_clears_all_stats() {
        let mut enc = TargetEncoder::new(vec![0]);

        enc.update(&[1.0, 2.0], 10.0);
        enc.update(&[2.0, 3.0], 20.0);

        assert!(enc.global_mean() != 0.0);
        assert!(enc.n_categories(0) > 0);

        enc.reset();

        assert_eq!(enc.global_mean(), 0.0);
        assert_eq!(enc.n_categories(0), 0);
        assert!(enc.category_stats[0].is_empty());
    }

    #[test]
    fn multiple_categorical_features() {
        // Features: [cat_a, numeric, cat_b]
        let mut enc = TargetEncoder::with_smoothing(vec![0, 2], 1.0);

        // cat_a=0, cat_b=10 → target 100
        for _ in 0..50 {
            enc.update(&[0.0, 5.0, 10.0], 100.0);
        }
        // cat_a=1, cat_b=20 → target 200
        for _ in 0..50 {
            enc.update(&[1.0, 5.0, 20.0], 200.0);
        }

        let out = enc.transform(&[0.0, 99.0, 20.0]);

        // Feature 1 (numeric) passes through unchanged.
        assert!(
            (out[1] - 99.0).abs() < EPS,
            "numeric feature should be 99.0, got {}",
            out[1]
        );

        // cat_a=0 should be close to 100 (its category mean).
        assert!(
            (out[0] - 100.0).abs() < 5.0,
            "cat_a=0 expected ~100.0, got {}",
            out[0]
        );

        // cat_b=20 should be close to 200 (its category mean).
        assert!(
            (out[2] - 200.0).abs() < 5.0,
            "cat_b=20 expected ~200.0, got {}",
            out[2]
        );

        // Verify we discovered categories in both slots.
        assert_eq!(enc.n_categories(0), 2); // cat_a: {0, 1}
        assert_eq!(enc.n_categories(1), 2); // cat_b: {10, 20}
    }

    #[test]
    fn zero_smoothing_uses_pure_category_mean() {
        let mut enc = TargetEncoder::with_smoothing(vec![0], 0.0);

        enc.update(&[1.0], 10.0);
        enc.update(&[1.0], 30.0);
        enc.update(&[2.0], 100.0);

        // Category 1 mean = (10 + 30) / 2 = 20.0
        // With smoothing=0: encoded = (2*20 + 0*global) / (2+0) = 20.0
        let out = enc.transform(&[1.0]);
        assert!(
            (out[0] - 20.0).abs() < EPS,
            "with zero smoothing expected exact category mean 20.0, got {}",
            out[0]
        );

        // Category 2 mean = 100.0
        let out2 = enc.transform(&[2.0]);
        assert!(
            (out2[0] - 100.0).abs() < EPS,
            "with zero smoothing expected exact category mean 100.0, got {}",
            out2[0]
        );
    }

    #[test]
    #[should_panic(expected = "smoothing must be >= 0.0")]
    fn negative_smoothing_panics() {
        TargetEncoder::with_smoothing(vec![0], -1.0);
    }
}
