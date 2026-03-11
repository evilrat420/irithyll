//! Non-crossing multi-quantile regression via parallel SGBT ensembles.
//!
//! Wraps K independent [`SGBT<QuantileLoss>`] instances, one per quantile
//! level. Predictions are post-processed with the Pool Adjacent Violators
//! Algorithm (PAVA) to enforce monotonicity — guaranteeing that the predicted
//! quantile at tau_i <= tau_j implies q_hat(tau_i) <= q_hat(tau_j).
//!
//! Without PAVA enforcement, independently trained quantile models can
//! produce *crossing* predictions (e.g., the 90th percentile prediction
//! falls below the 10th percentile), which is incoherent. PAVA resolves
//! crossings in O(K) time via isotonic regression.
//!
//! # Example
//!
//! ```
//! use irithyll::ensemble::quantile_regressor::QuantileRegressorSGBT;
//! use irithyll::SGBTConfig;
//!
//! let config = SGBTConfig::builder()
//!     .n_steps(10)
//!     .learning_rate(0.1)
//!     .grace_period(10)
//!     .build()
//!     .unwrap();
//!
//! // 90% prediction interval: [5th, 50th, 95th] percentiles
//! let quantiles = vec![0.05, 0.5, 0.95];
//! let mut model = QuantileRegressorSGBT::new(config, &quantiles).unwrap();
//!
//! model.train_one(&irithyll::Sample::new(vec![1.0, 2.0], 3.0));
//! let preds = model.predict(&[1.0, 2.0]);
//! assert_eq!(preds.len(), 3);
//! // Guaranteed: preds[0] <= preds[1] <= preds[2]
//! ```

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::SGBT;
use crate::error::{ConfigError, IrithyllError};
use crate::loss::quantile::QuantileLoss;
use crate::sample::Observation;

// ---------------------------------------------------------------------------
// PAVA — Pool Adjacent Violators Algorithm
// ---------------------------------------------------------------------------

/// Enforce monotonicity on a slice of values via isotonic regression.
///
/// The Pool Adjacent Violators Algorithm (PAVA) scans left-to-right,
/// merging adjacent blocks whose values violate the non-decreasing
/// constraint. When a violation is found, the two blocks are pooled
/// and replaced by their weighted average. This continues until the
/// entire sequence is non-decreasing.
///
/// Runs in O(K) time and O(K) space where K is the number of values.
///
/// # Arguments
///
/// * `values` — mutable slice modified in-place to be non-decreasing
fn enforce_monotonicity(values: &mut [f64]) {
    let n = values.len();
    if n <= 1 {
        return;
    }

    // Each block is (sum, count, start_idx)
    // We use a stack-based approach for efficiency
    let mut block_sums: Vec<f64> = Vec::with_capacity(n);
    let mut block_counts: Vec<usize> = Vec::with_capacity(n);
    let mut block_starts: Vec<usize> = Vec::with_capacity(n);

    for i in 0..n {
        // Push new singleton block
        block_sums.push(values[i]);
        block_counts.push(1);
        block_starts.push(i);

        // Merge backward while violation exists
        while block_sums.len() >= 2 {
            let len = block_sums.len();
            let mean_last = block_sums[len - 1] / block_counts[len - 1] as f64;
            let mean_prev = block_sums[len - 2] / block_counts[len - 2] as f64;

            if mean_prev <= mean_last {
                break; // No violation
            }

            // Pool: merge last block into previous
            block_sums[len - 2] += block_sums[len - 1];
            block_counts[len - 2] += block_counts[len - 1];
            block_sums.pop();
            block_counts.pop();
            block_starts.pop();
        }
    }

    // Expand blocks back into values
    for b in 0..block_sums.len() {
        let mean = block_sums[b] / block_counts[b] as f64;
        let start = block_starts[b];
        let end = if b + 1 < block_starts.len() {
            block_starts[b + 1]
        } else {
            n
        };
        for v in values[start..end].iter_mut() {
            *v = mean;
        }
    }
}

// ---------------------------------------------------------------------------
// QuantileRegressorSGBT
// ---------------------------------------------------------------------------

/// Non-crossing multi-quantile SGBT regressor.
///
/// Maintains K independent `SGBT<QuantileLoss>` models — one per quantile
/// level — and applies the Pool Adjacent Violators Algorithm (PAVA) to
/// predictions to guarantee non-crossing quantile estimates.
///
/// # Non-crossing guarantee
///
/// For quantile levels tau_1 < tau_2 < ... < tau_K, the predicted values
/// q_hat(tau_1) <= q_hat(tau_2) <= ... <= q_hat(tau_K) after PAVA
/// enforcement. This makes the output suitable for prediction intervals,
/// conditional density estimation, and risk quantification.
pub struct QuantileRegressorSGBT {
    /// One SGBT per quantile level, each with its own QuantileLoss.
    models: Vec<SGBT<QuantileLoss>>,
    /// Sorted quantile levels in (0, 1).
    quantiles: Vec<f64>,
    /// Number of quantile levels.
    n_quantiles: usize,
    /// Total samples seen.
    samples_seen: u64,
}

impl Clone for QuantileRegressorSGBT {
    fn clone(&self) -> Self {
        Self {
            models: self.models.clone(),
            quantiles: self.quantiles.clone(),
            n_quantiles: self.n_quantiles,
            samples_seen: self.samples_seen,
        }
    }
}

impl std::fmt::Debug for QuantileRegressorSGBT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantileRegressorSGBT")
            .field("quantiles", &self.quantiles)
            .field("n_quantiles", &self.n_quantiles)
            .field("samples_seen", &self.samples_seen)
            .finish()
    }
}

impl QuantileRegressorSGBT {
    /// Create a new multi-quantile regressor.
    ///
    /// Quantile levels are automatically sorted. Each level must be in (0, 1)
    /// and there must be at least one level.
    ///
    /// # Errors
    ///
    /// Returns [`IrithyllError::InvalidConfig`] if:
    /// - `quantiles` is empty
    /// - any quantile is not in (0, 1)
    /// - duplicate quantile levels exist
    pub fn new(config: SGBTConfig, quantiles: &[f64]) -> crate::error::Result<Self> {
        if quantiles.is_empty() {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "quantiles",
                "must have at least one quantile level",
                0usize,
            )));
        }

        // Validate and sort
        let mut sorted: Vec<f64> = quantiles.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (i, &tau) in sorted.iter().enumerate() {
            if tau <= 0.0 || tau >= 1.0 {
                return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                    "quantiles",
                    "each quantile must be in (0, 1)",
                    tau,
                )));
            }
            // Check for duplicates
            if i > 0 && (sorted[i] - sorted[i - 1]).abs() < 1e-15 {
                return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                    "quantiles",
                    "duplicate quantile levels are not allowed",
                    tau,
                )));
            }
        }

        let n_quantiles = sorted.len();
        let models = sorted
            .iter()
            .map(|&tau| SGBT::with_loss(config.clone(), QuantileLoss::new(tau)))
            .collect();

        Ok(Self {
            models,
            quantiles: sorted,
            n_quantiles,
            samples_seen: 0,
        })
    }

    /// Train all quantile models on a single observation.
    ///
    /// Each model independently learns from the same sample, targeting
    /// its respective quantile level.
    pub fn train_one(&mut self, sample: &impl Observation) {
        self.samples_seen += 1;
        for model in &mut self.models {
            model.train_one(sample);
        }
    }

    /// Train on a batch of observations.
    pub fn train_batch<O: Observation>(&mut self, samples: &[O]) {
        for sample in samples {
            self.train_one(sample);
        }
    }

    /// Predict quantile values with PAVA non-crossing enforcement.
    ///
    /// Returns a `Vec<f64>` of length `n_quantiles` where:
    /// - `result[i]` is the predicted quantile at level `quantiles[i]`
    /// - `result[0] <= result[1] <= ... <= result[K-1]` (guaranteed)
    pub fn predict(&self, features: &[f64]) -> Vec<f64> {
        let mut preds: Vec<f64> = self.models.iter().map(|m| m.predict(features)).collect();
        enforce_monotonicity(&mut preds);
        preds
    }

    /// Predict raw quantile values WITHOUT non-crossing enforcement.
    ///
    /// This may produce crossing predictions. Use [`predict`](Self::predict)
    /// for the PAVA-enforced version.
    pub fn predict_raw(&self, features: &[f64]) -> Vec<f64> {
        self.models.iter().map(|m| m.predict(features)).collect()
    }

    /// Predict a symmetric prediction interval at coverage level `1 - alpha`.
    ///
    /// Returns `(lower, median, upper)` if the model was constructed with
    /// quantile levels that include `alpha/2`, `0.5`, and `1 - alpha/2`.
    ///
    /// If the exact levels are not present, returns the closest available
    /// quantile predictions with PAVA enforcement applied.
    pub fn predict_interval(&self, features: &[f64]) -> (f64, f64, f64) {
        let preds = self.predict(features);
        let lower = preds[0];
        let upper = preds[preds.len() - 1];
        // Use the middle quantile as the point estimate
        let mid_idx = preds.len() / 2;
        let median = preds[mid_idx];
        (lower, median, upper)
    }

    /// Batch prediction with PAVA enforcement.
    pub fn predict_batch(&self, feature_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
        feature_matrix.iter().map(|f| self.predict(f)).collect()
    }

    /// Number of quantile levels.
    #[inline]
    pub fn n_quantiles(&self) -> usize {
        self.n_quantiles
    }

    /// The sorted quantile levels.
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    /// Total samples seen.
    #[inline]
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Access the model for a specific quantile index.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_quantiles`.
    pub fn model(&self, idx: usize) -> &SGBT<QuantileLoss> {
        &self.models[idx]
    }

    /// Access all quantile models.
    pub fn models(&self) -> &[SGBT<QuantileLoss>] {
        &self.models
    }

    /// Reset all quantile models.
    pub fn reset(&mut self) {
        for model in &mut self.models {
            model.reset();
        }
        self.samples_seen = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::Sample;

    fn test_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(10)
            .initial_target_count(5)
            .build()
            .unwrap()
    }

    // -----------------------------------------------------------------------
    // PAVA unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn pava_already_sorted() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        enforce_monotonicity(&mut values);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn pava_single_element() {
        let mut values = vec![42.0];
        enforce_monotonicity(&mut values);
        assert_eq!(values, vec![42.0]);
    }

    #[test]
    fn pava_empty() {
        let mut values: Vec<f64> = vec![];
        enforce_monotonicity(&mut values);
        assert!(values.is_empty());
    }

    #[test]
    fn pava_simple_violation() {
        // [3, 1, 2] -> 3 > 1 violation
        // Pool first two: mean(3,1) = 2.0 -> [2, 2, 2]
        // Then 2 <= 2 OK
        let mut values = vec![3.0, 1.0, 2.0];
        enforce_monotonicity(&mut values);
        // After pooling 3,1 -> 2.0, then 2.0 <= 2.0, so all become 2.0
        assert!((values[0] - 2.0).abs() < 1e-10);
        assert!((values[1] - 2.0).abs() < 1e-10);
        assert!((values[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn pava_reversed() {
        // Fully reversed: [5, 4, 3, 2, 1] -> all become mean = 3.0
        let mut values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        enforce_monotonicity(&mut values);
        let mean = 3.0;
        for v in &values {
            assert!((v - mean).abs() < 1e-10, "expected {mean}, got {v}");
        }
    }

    #[test]
    fn pava_partial_violation() {
        // [1, 5, 3, 4, 6] — violation at 5 > 3
        // Pool 5,3 -> 4.0: [1, 4, 4, 4, 6]
        let mut values = vec![1.0, 5.0, 3.0, 4.0, 6.0];
        enforce_monotonicity(&mut values);
        // Result should be non-decreasing
        for i in 1..values.len() {
            assert!(
                values[i] >= values[i - 1] - 1e-10,
                "violation at index {i}: {} < {}",
                values[i],
                values[i - 1]
            );
        }
        // First should still be 1.0
        assert!((values[0] - 1.0).abs() < 1e-10);
        // Last should still be 6.0
        assert!((values[4] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn pava_equal_values() {
        let mut values = vec![3.0, 3.0, 3.0];
        enforce_monotonicity(&mut values);
        assert_eq!(values, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn pava_two_elements_violation() {
        let mut values = vec![5.0, 1.0];
        enforce_monotonicity(&mut values);
        assert!((values[0] - 3.0).abs() < 1e-10);
        assert!((values[1] - 3.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // QuantileRegressorSGBT tests
    // -----------------------------------------------------------------------

    #[test]
    fn creates_correct_number_of_models() {
        let model = QuantileRegressorSGBT::new(test_config(), &[0.1, 0.5, 0.9]).unwrap();
        assert_eq!(model.n_quantiles(), 3);
        assert_eq!(model.models().len(), 3);
        assert_eq!(model.n_samples_seen(), 0);
    }

    #[test]
    fn quantiles_are_sorted() {
        let model = QuantileRegressorSGBT::new(test_config(), &[0.9, 0.1, 0.5]).unwrap();
        assert_eq!(model.quantiles(), &[0.1, 0.5, 0.9]);
    }

    #[test]
    fn rejects_empty_quantiles() {
        let result = QuantileRegressorSGBT::new(test_config(), &[]);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_invalid_quantile_zero() {
        let result = QuantileRegressorSGBT::new(test_config(), &[0.0, 0.5]);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_invalid_quantile_one() {
        let result = QuantileRegressorSGBT::new(test_config(), &[0.5, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_duplicate_quantiles() {
        let result = QuantileRegressorSGBT::new(test_config(), &[0.5, 0.5, 0.9]);
        assert!(result.is_err());
    }

    #[test]
    fn single_quantile_works() {
        let mut model = QuantileRegressorSGBT::new(test_config(), &[0.5]).unwrap();
        for i in 0..50 {
            let x = i as f64 * 0.1;
            model.train_one(&Sample::new(vec![x], x * 2.0));
        }
        let preds = model.predict(&[0.5]);
        assert_eq!(preds.len(), 1);
        assert!(preds[0].is_finite());
    }

    #[test]
    fn predictions_are_non_crossing() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(10)
            .initial_target_count(5)
            .build()
            .unwrap();

        let mut model = QuantileRegressorSGBT::new(config, &[0.05, 0.25, 0.5, 0.75, 0.95]).unwrap();

        // Train on noisy linear data
        let mut rng: u64 = 42;
        for _ in 0..200 {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let x = (rng >> 33) as f64 / (u32::MAX as f64) * 10.0;
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let noise = ((rng >> 33) as f64 / (u32::MAX as f64) - 0.5) * 2.0;
            let y = 3.0 * x + noise;
            model.train_one(&Sample::new(vec![x], y));
        }

        // Test at multiple points — predictions must be non-decreasing
        let test_points = [0.0, 1.0, 3.0, 5.0, 8.0, 10.0];
        for &x in &test_points {
            let preds = model.predict(&[x]);
            for i in 1..preds.len() {
                assert!(
                    preds[i] >= preds[i - 1] - 1e-10,
                    "crossing at x={x}: q[{i}]={} < q[{}]={}",
                    preds[i],
                    i - 1,
                    preds[i - 1]
                );
            }
        }
    }

    #[test]
    fn raw_predict_may_cross() {
        // Raw predictions don't have PAVA — they may cross
        // (Just verify the method works, crossings aren't guaranteed)
        let mut model = QuantileRegressorSGBT::new(test_config(), &[0.1, 0.5, 0.9]).unwrap();

        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&Sample::new(vec![x], x));
        }

        let raw = model.predict_raw(&[0.5]);
        assert_eq!(raw.len(), 3);
        for v in &raw {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn predict_interval_returns_triple() {
        let mut model = QuantileRegressorSGBT::new(test_config(), &[0.05, 0.5, 0.95]).unwrap();

        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&Sample::new(vec![x], x * 2.0 + 1.0));
        }

        let (lower, median, upper) = model.predict_interval(&[0.5]);
        assert!(lower <= median, "lower={lower} > median={median}");
        assert!(median <= upper, "median={median} > upper={upper}");
    }

    #[test]
    fn batch_prediction() {
        let mut model = QuantileRegressorSGBT::new(test_config(), &[0.1, 0.5, 0.9]).unwrap();

        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&Sample::new(vec![x], x));
        }

        let features = vec![vec![0.5], vec![1.0], vec![2.0]];
        let batch = model.predict_batch(&features);
        assert_eq!(batch.len(), 3);
        for preds in &batch {
            assert_eq!(preds.len(), 3);
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut model = QuantileRegressorSGBT::new(test_config(), &[0.1, 0.5, 0.9]).unwrap();

        for i in 0..100 {
            let x = i as f64;
            model.train_one(&Sample::new(vec![x], x));
        }
        assert!(model.n_samples_seen() > 0);

        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
    }

    #[test]
    fn deterministic_with_same_config() {
        let config = test_config();
        let quantiles = [0.1, 0.5, 0.9];
        let mut model1 = QuantileRegressorSGBT::new(config.clone(), &quantiles).unwrap();
        let mut model2 = QuantileRegressorSGBT::new(config, &quantiles).unwrap();

        let samples: Vec<Sample> = (0..50)
            .map(|i| {
                let x = i as f64 * 0.1;
                Sample::new(vec![x], x * 3.0)
            })
            .collect();

        for s in &samples {
            model1.train_one(s);
            model2.train_one(s);
        }

        let pred1 = model1.predict(&[0.5]);
        let pred2 = model2.predict(&[0.5]);
        for (a, b) in pred1.iter().zip(pred2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "same config should give identical predictions: {a} vs {b}"
            );
        }
    }

    #[test]
    fn higher_quantile_predicts_higher_after_training() {
        let config = SGBTConfig::builder()
            .n_steps(20)
            .learning_rate(0.1)
            .grace_period(10)
            .initial_target_count(5)
            .build()
            .unwrap();

        let mut model = QuantileRegressorSGBT::new(config, &[0.1, 0.5, 0.9]).unwrap();

        // Train on data with spread
        let mut rng: u64 = 99;
        for _ in 0..500 {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let x = (rng >> 33) as f64 / (u32::MAX as f64) * 10.0;
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let noise = ((rng >> 33) as f64 / (u32::MAX as f64) - 0.5) * 4.0;
            model.train_one(&Sample::new(vec![x], x + noise));
        }

        let preds = model.predict(&[5.0]);
        // With enough training, higher quantiles should predict higher
        assert!(
            preds[2] > preds[0],
            "90th percentile ({}) should be > 10th percentile ({})",
            preds[2],
            preds[0]
        );
    }
}
