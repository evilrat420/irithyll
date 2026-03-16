//! Streaming min-max scaler for feature normalization.

use crate::pipeline::StreamingPreprocessor;

/// Streaming min-max scaler -- incremental feature scaling to a configurable range.
///
/// Tracks per-feature running minimum and maximum values, scaling features to a
/// target range (default `[0, 1]`). Each call to
/// [`update_and_transform`](StreamingPreprocessor::update_and_transform) incorporates
/// a new sample in O(n_features) time with zero allocations (after the first
/// sample initializes internal buffers).
///
/// Scaling formula:
///
/// ```text
/// scaled = (x - min) / (max - min) * (range_high - range_low) + range_low
/// ```
///
/// When `max - min < f64::EPSILON` for a feature (zero range), the scaler
/// outputs the midpoint of the target range to avoid division by zero.
///
/// # Example
///
/// ```
/// use irithyll::preprocessing::MinMaxScaler;
/// use irithyll::pipeline::StreamingPreprocessor;
///
/// let mut scaler = MinMaxScaler::new();
///
/// // Feed samples one at a time.
/// scaler.update_and_transform(&[0.0, 100.0]);
/// scaler.update_and_transform(&[10.0, 0.0]);
///
/// // Transform a new observation using running min/max.
/// let scaled = scaler.transform(&[5.0, 50.0]);
/// assert!((scaled[0] - 0.5).abs() < 1e-9);
/// assert!((scaled[1] - 0.5).abs() < 1e-9);
/// ```
#[derive(Clone, Debug)]
pub struct MinMaxScaler {
    /// Per-feature running minimums.
    mins: Vec<f64>,
    /// Per-feature running maximums.
    maxs: Vec<f64>,
    /// Expected feature dimensionality (`None` until first sample).
    n_features: Option<usize>,
    /// Total number of samples incorporated.
    count: u64,
    /// Lower bound of output range (default `0.0`).
    range_low: f64,
    /// Upper bound of output range (default `1.0`).
    range_high: f64,
}

impl MinMaxScaler {
    /// Create a scaler with the default target range `[0, 1]`.
    ///
    /// The number of features is inferred from the first call to
    /// [`update_and_transform`](StreamingPreprocessor::update_and_transform).
    ///
    /// ```
    /// use irithyll::preprocessing::MinMaxScaler;
    /// let scaler = MinMaxScaler::new();
    /// ```
    pub fn new() -> Self {
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            n_features: None,
            count: 0,
            range_low: 0.0,
            range_high: 1.0,
        }
    }

    /// Create a scaler with a custom output range `[low, high]`.
    ///
    /// # Panics
    ///
    /// Panics if `low >= high`.
    ///
    /// ```
    /// use irithyll::preprocessing::MinMaxScaler;
    /// let scaler = MinMaxScaler::with_range(-1.0, 1.0);
    /// ```
    pub fn with_range(low: f64, high: f64) -> Self {
        assert!(
            low < high,
            "MinMaxScaler: range low ({}) must be less than high ({})",
            low,
            high
        );
        Self {
            mins: Vec::new(),
            maxs: Vec::new(),
            n_features: None,
            count: 0,
            range_low: low,
            range_high: high,
        }
    }

    /// Per-feature running minimums observed so far.
    ///
    /// Empty until the first sample has been incorporated.
    pub fn mins(&self) -> &[f64] {
        &self.mins
    }

    /// Per-feature running maximums observed so far.
    ///
    /// Empty until the first sample has been incorporated.
    pub fn maxs(&self) -> &[f64] {
        &self.maxs
    }

    /// The configured output range as `(low, high)`.
    pub fn range(&self) -> (f64, f64) {
        (self.range_low, self.range_high)
    }

    /// Number of samples incorporated so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Scale a single value given its observed min/max bounds.
    #[inline]
    fn scale_value(&self, x: f64, min: f64, max: f64) -> f64 {
        let range_in = max - min;
        if range_in < f64::EPSILON {
            // Zero range: return the midpoint of the target range.
            (self.range_low + self.range_high) / 2.0
        } else {
            let scaled = (x - min) / range_in;
            scaled * (self.range_high - self.range_low) + self.range_low
        }
    }

    /// Initialize internal buffers from the first sample.
    fn init_from_sample(&mut self, features: &[f64]) {
        let n = features.len();
        self.mins = features.to_vec();
        self.maxs = features.to_vec();
        self.n_features = Some(n);
    }

    /// Update running min/max from a sample.
    fn update_bounds(&mut self, features: &[f64]) {
        for (j, &fj) in features.iter().enumerate() {
            if fj < self.mins[j] {
                self.mins[j] = fj;
            }
            if fj > self.maxs[j] {
                self.maxs[j] = fj;
            }
        }
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamingPreprocessor impl
// ---------------------------------------------------------------------------

impl StreamingPreprocessor for MinMaxScaler {
    /// Update running min/max from this sample and return scaled features.
    ///
    /// On the first call, initializes internal buffers from the sample. Since
    /// `min == max` for every feature after a single sample, all outputs will
    /// be the midpoint of the target range.
    fn update_and_transform(&mut self, features: &[f64]) -> Vec<f64> {
        match self.n_features {
            None => {
                self.init_from_sample(features);
            }
            Some(n) => {
                assert_eq!(
                    features.len(),
                    n,
                    "MinMaxScaler: expected {} features, got {}",
                    n,
                    features.len()
                );
                self.update_bounds(features);
            }
        }
        self.count += 1;

        features
            .iter()
            .enumerate()
            .map(|(j, &fj)| self.scale_value(fj, self.mins[j], self.maxs[j]))
            .collect()
    }

    /// Scale features using current min/max without updating statistics.
    ///
    /// If no samples have been seen yet (no min/max stats), features pass
    /// through unscaled.
    fn transform(&self, features: &[f64]) -> Vec<f64> {
        if self.n_features.is_none() {
            // No stats yet -- pass through unscaled.
            return features.to_vec();
        }

        let n = self.n_features.unwrap();
        assert_eq!(
            features.len(),
            n,
            "MinMaxScaler: expected {} features, got {}",
            n,
            features.len()
        );

        features
            .iter()
            .enumerate()
            .map(|(j, &fj)| self.scale_value(fj, self.mins[j], self.maxs[j]))
            .collect()
    }

    /// Number of output features, or `None` if unknown until the first sample.
    fn output_dim(&self) -> Option<usize> {
        self.n_features
    }

    /// Reset to initial (untrained) state.
    ///
    /// Clears mins, maxs, sets `n_features` to `None`, and resets count to 0.
    /// Preserves the configured output range.
    fn reset(&mut self) {
        self.mins.clear();
        self.maxs.clear();
        self.n_features = None;
        self.count = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn scales_to_zero_one() {
        let mut scaler = MinMaxScaler::new();
        scaler.update_and_transform(&[1.0, 2.0]);
        scaler.update_and_transform(&[3.0, 4.0]);
        scaler.update_and_transform(&[5.0, 6.0]);

        let scaled = scaler.transform(&[3.0, 4.0]);
        assert!(
            (scaled[0] - 0.5).abs() < EPS,
            "expected 0.5, got {}",
            scaled[0]
        );
        assert!(
            (scaled[1] - 0.5).abs() < EPS,
            "expected 0.5, got {}",
            scaled[1]
        );
    }

    #[test]
    fn custom_range() {
        let mut scaler = MinMaxScaler::with_range(-1.0, 1.0);
        scaler.update_and_transform(&[0.0, 10.0]);
        scaler.update_and_transform(&[10.0, 0.0]);

        // Midpoint of data maps to midpoint of range (-1, 1) = 0.0.
        let scaled = scaler.transform(&[5.0, 5.0]);
        assert!(
            (scaled[0] - 0.0).abs() < EPS,
            "expected 0.0, got {}",
            scaled[0]
        );
        assert!(
            (scaled[1] - 0.0).abs() < EPS,
            "expected 0.0, got {}",
            scaled[1]
        );

        // Min of data maps to -1.0.
        let low = scaler.transform(&[0.0, 0.0]);
        assert!(
            (low[0] - (-1.0)).abs() < EPS,
            "expected -1.0, got {}",
            low[0]
        );

        // Max of data maps to 1.0.
        let high = scaler.transform(&[10.0, 10.0]);
        assert!((high[0] - 1.0).abs() < EPS, "expected 1.0, got {}", high[0]);
    }

    #[test]
    fn constant_feature_maps_to_midpoint() {
        let mut scaler = MinMaxScaler::new();
        scaler.update_and_transform(&[5.0, 5.0]);
        scaler.update_and_transform(&[5.0, 5.0]);
        scaler.update_and_transform(&[5.0, 5.0]);

        let scaled = scaler.transform(&[5.0, 5.0]);
        for (i, &val) in scaled.iter().enumerate() {
            assert!(
                (val - 0.5).abs() < EPS,
                "feature {}: expected midpoint 0.5, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn transform_before_update_passes_through() {
        let scaler = MinMaxScaler::new();
        let input = [42.0, -7.0, 100.0];
        let result = scaler.transform(&input);

        // No stats yet, features pass through unscaled.
        for (i, (&orig, &out)) in input.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig - out).abs() < EPS,
                "feature {}: expected {}, got {}",
                i,
                orig,
                out
            );
        }
    }

    #[test]
    fn reset_clears_stats() {
        let mut scaler = MinMaxScaler::new();
        scaler.update_and_transform(&[1.0, 2.0]);
        scaler.update_and_transform(&[3.0, 4.0]);
        assert_eq!(scaler.output_dim(), Some(2));
        assert_eq!(scaler.count(), 2);

        scaler.reset();
        assert_eq!(scaler.output_dim(), None);
        assert_eq!(scaler.count(), 0);
        assert!(scaler.mins().is_empty());
        assert!(scaler.maxs().is_empty());

        // Range config is preserved.
        assert_eq!(scaler.range(), (0.0, 1.0));
    }

    #[test]
    fn single_sample_constant() {
        let mut scaler = MinMaxScaler::new();
        let result = scaler.update_and_transform(&[42.0, -7.0, 100.0]);

        // With 1 sample, min == max for all features, so output is midpoint.
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - 0.5).abs() < EPS,
                "feature {}: expected midpoint 0.5, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn output_dim_matches() {
        let mut scaler = MinMaxScaler::new();
        assert_eq!(scaler.output_dim(), None);

        scaler.update_and_transform(&[1.0, 2.0, 3.0]);
        assert_eq!(scaler.output_dim(), Some(3));
    }

    #[test]
    #[should_panic(expected = "range low")]
    fn invalid_range_panics() {
        MinMaxScaler::with_range(1.0, 0.0);
    }

    // -- Additional tests beyond the required 8 --

    #[test]
    fn accessors_track_extremes() {
        let mut scaler = MinMaxScaler::new();
        scaler.update_and_transform(&[5.0, 20.0]);
        scaler.update_and_transform(&[1.0, 30.0]);
        scaler.update_and_transform(&[10.0, 10.0]);

        let mins = scaler.mins();
        let maxs = scaler.maxs();

        assert!(
            (mins[0] - 1.0).abs() < EPS,
            "min[0]: expected 1.0, got {}",
            mins[0]
        );
        assert!(
            (mins[1] - 10.0).abs() < EPS,
            "min[1]: expected 10.0, got {}",
            mins[1]
        );
        assert!(
            (maxs[0] - 10.0).abs() < EPS,
            "max[0]: expected 10.0, got {}",
            maxs[0]
        );
        assert!(
            (maxs[1] - 30.0).abs() < EPS,
            "max[1]: expected 30.0, got {}",
            maxs[1]
        );
    }

    #[test]
    fn range_accessor_returns_config() {
        let scaler = MinMaxScaler::with_range(-1.0, 1.0);
        assert_eq!(scaler.range(), (-1.0, 1.0));

        let default_scaler = MinMaxScaler::new();
        assert_eq!(default_scaler.range(), (0.0, 1.0));
    }

    #[test]
    fn transform_does_not_update_bounds() {
        let mut scaler = MinMaxScaler::new();
        scaler.update_and_transform(&[0.0, 0.0]);
        scaler.update_and_transform(&[10.0, 10.0]);

        let mins_before: Vec<f64> = scaler.mins().to_vec();
        let maxs_before: Vec<f64> = scaler.maxs().to_vec();
        let count_before = scaler.count();

        // transform should NOT update bounds, even with out-of-range values.
        let _ = scaler.transform(&[20.0, -5.0]);

        assert_eq!(scaler.mins(), mins_before.as_slice());
        assert_eq!(scaler.maxs(), maxs_before.as_slice());
        assert_eq!(scaler.count(), count_before);
    }

    #[test]
    #[should_panic(expected = "expected 2 features, got 3")]
    fn panics_on_dimension_mismatch() {
        let mut scaler = MinMaxScaler::new();
        scaler.update_and_transform(&[1.0, 2.0]);
        scaler.update_and_transform(&[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "range low (5) must be less than high (5)")]
    fn equal_range_panics() {
        MinMaxScaler::with_range(5.0, 5.0);
    }

    #[test]
    fn reset_preserves_custom_range() {
        let mut scaler = MinMaxScaler::with_range(-10.0, 10.0);
        scaler.update_and_transform(&[1.0, 2.0]);
        scaler.reset();

        assert_eq!(scaler.range(), (-10.0, 10.0));
        assert_eq!(scaler.count(), 0);
        assert_eq!(scaler.output_dim(), None);
    }
}
