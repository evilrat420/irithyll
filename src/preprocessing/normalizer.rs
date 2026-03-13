//! Welford online mean/variance normalizer for incremental standardization.
//!
//! [`IncrementalNormalizer`] maintains per-feature running statistics using
//! Welford's numerically stable algorithm. Each call to [`update`](IncrementalNormalizer::update)
//! incorporates a new sample in O(n_features) time with zero allocations
//! (after the first sample initialises internal buffers).
//!
//! Transformed outputs are zero-mean, unit-variance:
//!
//! ```text
//! z_i = (x_i - mean_i) / sqrt(variance_i + floor)
//! ```
//!
//! The variance floor (default `1e-12`) prevents division by zero on
//! constant features.
//!
//! # Example
//!
//! ```
//! use irithyll::preprocessing::IncrementalNormalizer;
//!
//! let mut norm = IncrementalNormalizer::new();
//!
//! // Feed samples one at a time.
//! norm.update(&[10.0, 200.0]);
//! norm.update(&[12.0, 180.0]);
//! norm.update(&[11.0, 190.0]);
//!
//! // Transform a new observation using running statistics.
//! let z = norm.transform(&[11.0, 190.0]);
//! assert_eq!(z.len(), 2);
//! ```

/// Welford online normalizer — incremental zero-mean, unit-variance standardization.
///
/// Internally tracks per-feature `mean` and `M2` (sum of squared deviations
/// from the current mean) so that variance is always available as `M2 / count`.
///
/// The struct supports two initialisation modes:
///
/// 1. **Lazy** — call [`new`](Self::new) and let the first [`update`](Self::update)
///    determine `n_features` from the slice length.
/// 2. **Explicit** — call [`with_n_features`](Self::with_n_features) to
///    pre-allocate buffers and enforce a fixed dimensionality.
#[derive(Clone, Debug)]
pub struct IncrementalNormalizer {
    /// Running per-feature means.
    means: Vec<f64>,
    /// Running per-feature M2 (sum of squared deviations from current mean).
    m2s: Vec<f64>,
    /// Total number of samples incorporated.
    count: u64,
    /// Expected feature dimensionality (`None` until first sample in lazy mode).
    n_features: Option<usize>,
    /// Minimum variance denominator to avoid division by zero. Default `1e-12`.
    variance_floor: f64,
}

impl IncrementalNormalizer {
    /// Create a normalizer with lazy dimensionality detection.
    ///
    /// The number of features is inferred from the first call to
    /// [`update`](Self::update).
    ///
    /// ```
    /// use irithyll::preprocessing::IncrementalNormalizer;
    /// let norm = IncrementalNormalizer::new();
    /// assert_eq!(norm.count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            m2s: Vec::new(),
            count: 0,
            n_features: None,
            variance_floor: 1e-12,
        }
    }

    /// Create a normalizer with a known feature count, pre-allocating buffers.
    ///
    /// Subsequent calls to [`update`](Self::update) will panic if the slice
    /// length does not match `n`.
    ///
    /// ```
    /// use irithyll::preprocessing::IncrementalNormalizer;
    /// let norm = IncrementalNormalizer::with_n_features(5);
    /// assert_eq!(norm.n_features(), Some(5));
    /// ```
    pub fn with_n_features(n: usize) -> Self {
        Self {
            means: vec![0.0; n],
            m2s: vec![0.0; n],
            count: 0,
            n_features: Some(n),
            variance_floor: 1e-12,
        }
    }

    /// Create a normalizer with a known feature count and custom variance floor.
    ///
    /// The `floor` is added to variance before taking the square root during
    /// transformation, preventing NaN on constant features.
    ///
    /// ```
    /// use irithyll::preprocessing::IncrementalNormalizer;
    /// let norm = IncrementalNormalizer::with_variance_floor(3, 1e-8);
    /// assert_eq!(norm.n_features(), Some(3));
    /// ```
    pub fn with_variance_floor(n: usize, floor: f64) -> Self {
        Self {
            means: vec![0.0; n],
            m2s: vec![0.0; n],
            count: 0,
            n_features: Some(n),
            variance_floor: floor,
        }
    }

    /// Incorporate a single sample into running statistics.
    ///
    /// Uses Welford's algorithm for numerically stable incremental updates:
    ///
    /// ```text
    /// count  += 1
    /// delta   = x - mean
    /// mean   += delta / count
    /// delta2  = x - mean   (post-update)
    /// M2     += delta * delta2
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `features.len()` does not match the previously established
    /// dimensionality (either from [`with_n_features`](Self::with_n_features)
    /// or from the first [`update`](Self::update) call).
    pub fn update(&mut self, features: &[f64]) {
        match self.n_features {
            None => {
                // Lazy initialisation on first sample.
                let n = features.len();
                self.means = vec![0.0; n];
                self.m2s = vec![0.0; n];
                self.n_features = Some(n);
            }
            Some(n) => {
                assert_eq!(
                    features.len(),
                    n,
                    "IncrementalNormalizer: expected {} features, got {}",
                    n,
                    features.len()
                );
            }
        }

        self.count += 1;
        let count_f = self.count as f64;

        for (j, &fj) in features.iter().enumerate() {
            let delta = fj - self.means[j];
            self.means[j] += delta / count_f;
            let delta2 = fj - self.means[j];
            self.m2s[j] += delta * delta2;
        }
    }

    /// Standardize `features` using current running statistics.
    ///
    /// Returns a new `Vec<f64>` where each element is
    /// `(x_i - mean_i) / sqrt(var_i + floor)`.
    ///
    /// # Panics
    ///
    /// Panics if no samples have been incorporated yet, or if the slice
    /// length does not match the established dimensionality.
    pub fn transform(&self, features: &[f64]) -> Vec<f64> {
        let n = self.check_features_len(features.len());
        assert!(
            self.count > 0,
            "IncrementalNormalizer: cannot transform before any updates"
        );

        let count_f = self.count as f64;
        let mut out = Vec::with_capacity(n);
        for ((&fj, &mean), &m2) in features.iter().zip(self.means.iter()).zip(self.m2s.iter()) {
            let var = m2 / count_f;
            out.push((fj - mean) / (var + self.variance_floor).sqrt());
        }
        out
    }

    /// Standardize `features` in place using current running statistics.
    ///
    /// Equivalent to [`transform`](Self::transform) but writes results back
    /// into the provided mutable slice, avoiding allocation.
    ///
    /// # Panics
    ///
    /// Same conditions as [`transform`](Self::transform).
    pub fn transform_in_place(&self, features: &mut [f64]) {
        let _n = self.check_features_len(features.len());
        assert!(
            self.count > 0,
            "IncrementalNormalizer: cannot transform before any updates"
        );

        let count_f = self.count as f64;
        for ((fj, &mean), &m2) in features
            .iter_mut()
            .zip(self.means.iter())
            .zip(self.m2s.iter())
        {
            let var = m2 / count_f;
            *fj = (*fj - mean) / (var + self.variance_floor).sqrt();
        }
    }

    /// Update statistics with `features` then return the standardized result.
    ///
    /// This is equivalent to calling [`update`](Self::update) followed by
    /// [`transform`](Self::transform), but slightly more efficient since the
    /// sample count is guaranteed to be at least 1 after the update.
    ///
    /// ```
    /// use irithyll::preprocessing::IncrementalNormalizer;
    /// let mut norm = IncrementalNormalizer::new();
    /// let z = norm.update_and_transform(&[5.0, 10.0]);
    /// // After a single sample, variance is 0 so output is ~0 (floor-limited).
    /// assert!(z[0].abs() < 1e6);
    /// ```
    pub fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        self.update(features);
        self.transform(features)
    }

    /// Running mean for feature at `idx`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_features` or if no features have been set.
    pub fn mean(&self, idx: usize) -> f64 {
        self.means[idx]
    }

    /// Running population variance for feature at `idx`.
    ///
    /// Returns `M2[idx] / count`. Returns `0.0` if `count == 0`.
    pub fn variance(&self, idx: usize) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.m2s[idx] / self.count as f64
    }

    /// Running population standard deviation for feature at `idx`.
    ///
    /// Equal to `sqrt(variance(idx))`.
    pub fn std_dev(&self, idx: usize) -> f64 {
        self.variance(idx).sqrt()
    }

    /// Number of features, or `None` if lazy mode has not yet seen a sample.
    pub fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    /// Number of samples incorporated so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset all statistics, returning the normalizer to its initial state.
    ///
    /// Preserves `n_features` and `variance_floor` settings but clears all
    /// running accumulators.
    pub fn reset(&mut self) {
        self.count = 0;
        self.means.fill(0.0);
        self.m2s.fill(0.0);
    }

    /// Validate and return the expected feature count.
    fn check_features_len(&self, len: usize) -> usize {
        let n = self
            .n_features
            .expect("IncrementalNormalizer: n_features not set (call update first)");
        assert_eq!(
            len, n,
            "IncrementalNormalizer: expected {} features, got {}",
            n, len
        );
        n
    }
}

impl Default for IncrementalNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamingPreprocessor impl
// ---------------------------------------------------------------------------

impl crate::pipeline::StreamingPreprocessor for IncrementalNormalizer {
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        IncrementalNormalizer::update_and_transform(self, features)
    }

    fn transform(&self, features: &[f64]) -> Vec<f64> {
        IncrementalNormalizer::transform(self, features)
    }

    fn output_dim(&self) -> Option<usize> {
        self.n_features()
    }

    fn reset(&mut self) {
        IncrementalNormalizer::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn lazy_init_sets_n_features_from_first_update() {
        let mut norm = IncrementalNormalizer::new();
        assert_eq!(norm.n_features(), None);
        norm.update(&[1.0, 2.0, 3.0]);
        assert_eq!(norm.n_features(), Some(3));
        assert_eq!(norm.count(), 1);
    }

    #[test]
    fn explicit_init_pre_allocates() {
        let norm = IncrementalNormalizer::with_n_features(4);
        assert_eq!(norm.n_features(), Some(4));
        assert_eq!(norm.count(), 0);
    }

    #[test]
    fn single_feature_mean_is_correct() {
        let mut norm = IncrementalNormalizer::new();
        let samples = [2.0, 4.0, 6.0, 8.0, 10.0];
        for &s in &samples {
            norm.update(&[s]);
        }
        let expected_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((norm.mean(0) - expected_mean).abs() < EPS);
    }

    #[test]
    fn single_feature_variance_is_correct() {
        let mut norm = IncrementalNormalizer::new();
        let samples = [2.0, 4.0, 6.0, 8.0, 10.0];
        for &s in &samples {
            norm.update(&[s]);
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let expected_var =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert!((norm.variance(0) - expected_var).abs() < EPS);
    }

    #[test]
    fn transform_produces_standardized_output() {
        let mut norm = IncrementalNormalizer::new();
        // Feed enough data for meaningful statistics.
        let data: Vec<[f64; 2]> = (0..100)
            .map(|i| [i as f64, (i as f64) * 2.0 + 50.0])
            .collect();
        for row in &data {
            norm.update(row);
        }

        // Transform the mean itself — should produce values near zero.
        let mean_point = [norm.mean(0), norm.mean(1)];
        let z = norm.transform(&mean_point);
        assert!(z[0].abs() < EPS, "z[0] = {}", z[0]);
        assert!(z[1].abs() < EPS, "z[1] = {}", z[1]);

        // Transform a point one std-dev above the mean — should be ~1.0.
        let one_sd = [
            norm.mean(0) + norm.std_dev(0),
            norm.mean(1) + norm.std_dev(1),
        ];
        let z1 = norm.transform(&one_sd);
        assert!((z1[0] - 1.0).abs() < 0.01, "z1[0] = {}", z1[0]);
        assert!((z1[1] - 1.0).abs() < 0.01, "z1[1] = {}", z1[1]);
    }

    #[test]
    fn variance_floor_prevents_nan() {
        let mut norm = IncrementalNormalizer::new();
        // Single sample: variance = 0, floor prevents NaN.
        norm.update(&[42.0]);
        let z = norm.transform(&[42.0]);
        assert!(z[0].is_finite(), "z[0] should be finite, got {}", z[0]);
    }

    #[test]
    fn update_and_transform_matches_separate_calls() {
        let mut norm_a = IncrementalNormalizer::new();
        let mut norm_b = IncrementalNormalizer::new();

        let samples = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        for row in &samples {
            let z_combined = norm_a.update_and_transform(row);

            norm_b.update(row);
            let z_separate = norm_b.transform(row);

            for (a, b) in z_combined.iter().zip(z_separate.iter()) {
                assert!((a - b).abs() < EPS, "mismatch: {} vs {}", a, b);
            }
        }
    }

    #[test]
    fn reset_clears_statistics() {
        let mut norm = IncrementalNormalizer::with_n_features(2);
        norm.update(&[10.0, 20.0]);
        norm.update(&[30.0, 40.0]);
        assert_eq!(norm.count(), 2);

        norm.reset();
        assert_eq!(norm.count(), 0);
        assert!((norm.mean(0)).abs() < EPS);
        assert!((norm.variance(0)).abs() < EPS);
        // n_features is preserved after reset.
        assert_eq!(norm.n_features(), Some(2));
    }

    #[test]
    fn transform_in_place_matches_transform() {
        let mut norm = IncrementalNormalizer::new();
        for row in &[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]] {
            norm.update(row);
        }

        let input = [2.5, 6.5];
        let z_alloc = norm.transform(&input);

        let mut z_inplace = input;
        norm.transform_in_place(&mut z_inplace);

        for (a, b) in z_alloc.iter().zip(z_inplace.iter()) {
            assert!((a - b).abs() < EPS, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    #[should_panic(expected = "expected 2 features, got 3")]
    fn panics_on_dimension_mismatch() {
        let mut norm = IncrementalNormalizer::with_n_features(2);
        norm.update(&[1.0, 2.0, 3.0]);
    }
}
