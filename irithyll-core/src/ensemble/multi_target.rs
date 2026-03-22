//! Multi-target regression via parallel SGBT ensembles.
//!
//! For T targets, maintains T independent `SGBT<L>` models, each trained on
//! a single target dimension. Predictions return a `Vec<f64>` of length T.

use alloc::vec::Vec;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::SGBT;
use crate::error::{ConfigError, IrithyllError};
use crate::loss::squared::SquaredLoss;
use crate::loss::Loss;
use crate::sample::SampleRef;

/// Multi-target regression SGBT.
///
/// Wraps `T` independent `SGBT<L>` models, one per target dimension.
/// Each model is trained and predicts independently, sharing the same
/// configuration and loss function.
///
/// # Examples
///
/// ```text
/// use irithyll::ensemble::multi_target::MultiTargetSGBT;
/// use irithyll::SGBTConfig;
///
/// let config = SGBTConfig::builder()
///     .n_steps(10)
///     .learning_rate(0.1)
///     .grace_period(10)
///     .build()
///     .unwrap();
///
/// let mut model = MultiTargetSGBT::new(config, 3).unwrap();
/// model.train_one(&[1.0, 2.0], &[0.5, 1.0, 1.5]);
/// let preds = model.predict(&[1.0, 2.0]);
/// assert_eq!(preds.len(), 3);
/// ```
#[derive(Debug)]
pub struct MultiTargetSGBT<L: Loss = SquaredLoss> {
    /// One SGBT per target dimension.
    models: Vec<SGBT<L>>,
    /// Number of target dimensions.
    n_targets: usize,
    /// Total samples seen.
    samples_seen: u64,
}

impl<L: Loss + Clone> Clone for MultiTargetSGBT<L> {
    fn clone(&self) -> Self {
        Self {
            models: self.models.clone(),
            n_targets: self.n_targets,
            samples_seen: self.samples_seen,
        }
    }
}

impl MultiTargetSGBT<SquaredLoss> {
    /// Create a new multi-target SGBT with squared loss (default).
    ///
    /// # Errors
    ///
    /// Returns [`IrithyllError::InvalidConfig`] if `n_targets < 1`.
    pub fn new(config: SGBTConfig, n_targets: usize) -> crate::error::Result<Self> {
        Self::with_loss(config, SquaredLoss, n_targets)
    }
}

impl<L: Loss + Clone> MultiTargetSGBT<L> {
    /// Create a new multi-target SGBT with a custom loss function.
    ///
    /// The loss is cloned for each target model.
    ///
    /// # Errors
    ///
    /// Returns [`IrithyllError::InvalidConfig`] if `n_targets < 1`.
    pub fn with_loss(config: SGBTConfig, loss: L, n_targets: usize) -> crate::error::Result<Self> {
        if n_targets < 1 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "n_targets",
                "must be >= 1",
                n_targets,
            )));
        }

        let models = (0..n_targets)
            .map(|_| SGBT::with_loss(config.clone(), loss.clone()))
            .collect();

        Ok(Self {
            models,
            n_targets,
            samples_seen: 0,
        })
    }

    /// Train on a single sample with multiple target values.
    ///
    /// # Panics
    ///
    /// Panics if `targets.len() != n_targets`.
    pub fn train_one(&mut self, features: &[f64], targets: &[f64]) {
        assert_eq!(
            targets.len(),
            self.n_targets,
            "expected {} targets, got {}",
            self.n_targets,
            targets.len()
        );
        self.samples_seen += 1;
        for (model, &target) in self.models.iter_mut().zip(targets.iter()) {
            let sample = SampleRef::new(features, target);
            model.train_one(&sample);
        }
    }

    /// Train on a batch of multi-target samples.
    pub fn train_batch(&mut self, feature_matrix: &[Vec<f64>], target_matrix: &[Vec<f64>]) {
        for (features, targets) in feature_matrix.iter().zip(target_matrix.iter()) {
            self.train_one(features, targets);
        }
    }

    /// Predict all target values for a feature vector.
    pub fn predict(&self, features: &[f64]) -> Vec<f64> {
        self.models.iter().map(|m| m.predict(features)).collect()
    }

    /// Number of target dimensions.
    pub fn n_targets(&self) -> usize {
        self.n_targets
    }

    /// Total samples trained.
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Access the model for a specific target dimension.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_targets`.
    pub fn model(&self, idx: usize) -> &SGBT<L> {
        &self.models[idx]
    }

    /// Access all target models.
    pub fn models(&self) -> &[SGBT<L>] {
        &self.models
    }

    /// Reset all target models.
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
    use alloc::string::ToString;
    use alloc::vec;

    fn test_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(3)
            .n_bins(8)
            .build()
            .unwrap()
    }

    #[test]
    fn new_multi_target_creates_models() {
        let model = MultiTargetSGBT::new(test_config(), 3).unwrap();
        assert_eq!(model.n_targets(), 3);
        assert_eq!(model.n_samples_seen(), 0);
    }

    #[test]
    fn rejects_zero_targets() {
        let err = MultiTargetSGBT::new(test_config(), 0).unwrap_err();
        assert!(
            err.to_string().contains("n_targets"),
            "error should mention n_targets: {}",
            err
        );
    }

    #[test]
    fn single_target_works() {
        let mut model = MultiTargetSGBT::new(test_config(), 1).unwrap();
        model.train_one(&[1.0, 2.0], &[5.0]);
        let preds = model.predict(&[1.0, 2.0]);
        assert_eq!(preds.len(), 1);
    }

    #[test]
    fn train_and_predict() {
        let mut model = MultiTargetSGBT::new(test_config(), 2).unwrap();

        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train_one(&[x, x * 2.0], &[x * 3.0, -x]);
        }

        assert_eq!(model.n_samples_seen(), 100);
        let preds = model.predict(&[1.0, 2.0]);
        assert_eq!(preds.len(), 2);
        assert!(preds[0].is_finite());
        assert!(preds[1].is_finite());
    }

    #[test]
    fn targets_are_independent() {
        let config = test_config();
        let mut multi = MultiTargetSGBT::new(config.clone(), 2).unwrap();
        let mut single = SGBT::new(config);

        let mut rng: u64 = 42;
        for _ in 0..200 {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let x = (rng >> 33) as f64 / (u32::MAX as f64);
            let t0 = 2.0 * x;
            let t1 = -3.0 * x;

            multi.train_one(&[x], &[t0, t1]);
            single.train_one(&Sample::new(vec![x], t0));
        }

        // The first target model should match the single model.
        let pred_multi = multi.predict(&[0.5]);
        let pred_single = single.predict(&[0.5]);
        assert!(
            (pred_multi[0] - pred_single).abs() < 1e-10,
            "target 0 should match independent model: multi={}, single={}",
            pred_multi[0],
            pred_single
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut model = MultiTargetSGBT::new(test_config(), 2).unwrap();
        for i in 0..50 {
            let x = i as f64 * 0.1;
            model.train_one(&[x], &[x, x * 2.0]);
        }
        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        let preds = model.predict(&[1.0]);
        for &p in &preds {
            assert!(p.abs() < 1e-12, "after reset, prediction should be ~0.0");
        }
    }

    #[test]
    fn model_accessor_works() {
        let model = MultiTargetSGBT::new(test_config(), 3).unwrap();
        assert_eq!(model.model(0).n_steps(), 5);
        assert_eq!(model.model(2).n_steps(), 5);
        assert_eq!(model.models().len(), 3);
    }

    #[test]
    #[should_panic(expected = "expected 2 targets")]
    fn wrong_target_count_panics() {
        let mut model = MultiTargetSGBT::new(test_config(), 2).unwrap();
        model.train_one(&[1.0], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn convergence_on_linear_signal() {
        let config = SGBTConfig::builder()
            .n_steps(10)
            .learning_rate(0.1)
            .grace_period(10)
            .max_depth(3)
            .n_bins(16)
            .build()
            .unwrap();
        let mut model = MultiTargetSGBT::new(config, 2).unwrap();

        let mut rng: u64 = 99;
        let mut early_mse = [0.0f64; 2];
        let mut late_mse = [0.0f64; 2];
        let mut early_n = 0;
        let mut late_n = 0;

        for i in 0..500 {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let x0 = (rng >> 33) as f64 / (u32::MAX as f64) * 10.0 - 5.0;
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let x1 = (rng >> 33) as f64 / (u32::MAX as f64) * 10.0 - 5.0;

            let t0 = 2.0 * x0 + 3.0 * x1;
            let t1 = -x0 + 0.5 * x1;

            let preds = model.predict(&[x0, x1]);

            if (50..150).contains(&i) {
                early_mse[0] += (preds[0] - t0).powi(2);
                early_mse[1] += (preds[1] - t1).powi(2);
                early_n += 1;
            }
            if i >= 400 {
                late_mse[0] += (preds[0] - t0).powi(2);
                late_mse[1] += (preds[1] - t1).powi(2);
                late_n += 1;
            }

            model.train_one(&[x0, x1], &[t0, t1]);
        }

        let early_rmse_0 = (early_mse[0] / early_n as f64).sqrt();
        let late_rmse_0 = (late_mse[0] / late_n as f64).sqrt();
        assert!(
            late_rmse_0 < early_rmse_0,
            "target 0 RMSE should improve: early={:.4}, late={:.4}",
            early_rmse_0,
            late_rmse_0
        );

        let early_rmse_1 = (early_mse[1] / early_n as f64).sqrt();
        let late_rmse_1 = (late_mse[1] / late_n as f64).sqrt();
        assert!(
            late_rmse_1 < early_rmse_1,
            "target 1 RMSE should improve: early={:.4}, late={:.4}",
            early_rmse_1,
            late_rmse_1
        );
    }
}
