//! Multi-class classification via one-vs-rest SGBT committees.
//!
//! For C classes, maintains C independent SGBT ensembles, each trained with
//! logistic loss on a binarized target (1.0 for the class, 0.0 otherwise).
//! Final predictions are normalized via softmax across committee outputs.

use alloc::vec::Vec;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::SGBT;
use crate::error::{ConfigError, IrithyllError};
use crate::loss::softmax::SoftmaxLoss;
use crate::sample::{Observation, SampleRef};

/// Multi-class SGBT using one-vs-rest committee of ensembles.
///
/// Each class gets its own `SGBT<SoftmaxLoss>` trained with softmax
/// (logistic per-class) loss. The concrete loss type is monomorphized
/// for each committee -- no `Box<dyn Loss>` overhead.
///
/// Predictions are softmax-normalized across all class committees.
#[derive(Debug)]
pub struct MulticlassSGBT {
    /// One SGBT per class, each with monomorphized SoftmaxLoss.
    committees: Vec<SGBT<SoftmaxLoss>>,
    /// Number of classes.
    n_classes: usize,
    /// Total samples seen.
    samples_seen: u64,
}

impl MulticlassSGBT {
    /// Create a new multi-class SGBT.
    ///
    /// # Errors
    ///
    /// Returns [`IrithyllError::InvalidConfig`] if `n_classes < 2`.
    pub fn new(config: SGBTConfig, n_classes: usize) -> crate::error::Result<Self> {
        if n_classes < 2 {
            return Err(IrithyllError::InvalidConfig(ConfigError::out_of_range(
                "n_classes",
                "must be >= 2",
                n_classes,
            )));
        }

        let committees = (0..n_classes)
            .map(|_| SGBT::with_loss(config.clone(), SoftmaxLoss { n_classes }))
            .collect();

        Ok(Self {
            committees,
            n_classes,
            samples_seen: 0,
        })
    }

    /// Train on a single observation.
    ///
    /// The observation's target should be the class index as f64 (0.0, 1.0, 2.0, ...).
    ///
    /// Uses [`SampleRef`] internally to avoid cloning feature vectors for each
    /// committee (N classes = 0 clones instead of N clones).
    pub fn train_one(&mut self, sample: &impl Observation) {
        self.samples_seen += 1;
        let class_idx = sample.target() as usize;
        let features = sample.features();

        for (c, committee) in self.committees.iter_mut().enumerate() {
            // Binary target: 1.0 for the correct class, 0.0 otherwise
            let binary_target = if c == class_idx { 1.0 } else { 0.0 };
            let binary_ref = SampleRef::new(features, binary_target);
            committee.train_one(&binary_ref);
        }
    }

    /// Train on a batch of observations.
    pub fn train_batch<O: Observation>(&mut self, samples: &[O]) {
        for sample in samples {
            self.train_one(sample);
        }
    }

    /// Predict class probabilities via softmax normalization.
    ///
    /// Returns a vector of length `n_classes` summing to ~1.0.
    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        // Get raw predictions from each committee
        let raw: Vec<f64> = self
            .committees
            .iter()
            .map(|c| c.predict(features))
            .collect();

        // Softmax normalization (numerically stable)
        let max_raw = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = raw.iter().map(|&r| crate::math::exp(r - max_raw)).sum();
        raw.iter()
            .map(|&r| crate::math::exp(r - max_raw) / exp_sum)
            .collect()
    }

    /// Predict the most likely class.
    pub fn predict(&self, features: &[f64]) -> usize {
        let proba = self.predict_proba(features);
        proba
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Total samples trained.
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Reset all committees.
    pub fn reset(&mut self) {
        for committee in &mut self.committees {
            committee.reset();
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
    fn new_multiclass_creates_committees() {
        let model = MulticlassSGBT::new(test_config(), 3).unwrap();
        assert_eq!(model.n_classes(), 3);
    }

    #[test]
    fn new_multiclass_rejects_less_than_two_classes() {
        let err = MulticlassSGBT::new(test_config(), 1).unwrap_err();
        assert!(
            err.to_string().contains("n_classes"),
            "error should mention n_classes: {}",
            err
        );
    }

    #[test]
    fn predict_proba_sums_to_one() {
        let model = MulticlassSGBT::new(test_config(), 3).unwrap();
        let proba = model.predict_proba(&[1.0, 2.0]);
        let sum: f64 = proba.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "probabilities should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn predict_proba_uniform_before_training() {
        let model = MulticlassSGBT::new(test_config(), 3).unwrap();
        let proba = model.predict_proba(&[1.0, 2.0]);
        // All committees predict 0.0, softmax of equal values = uniform
        for &p in &proba {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn train_one_does_not_panic() {
        let mut model = MulticlassSGBT::new(test_config(), 3).unwrap();
        model.train_one(&Sample::new(vec![1.0, 2.0], 0.0));
        model.train_one(&Sample::new(vec![3.0, 4.0], 1.0));
        model.train_one(&Sample::new(vec![5.0, 6.0], 2.0));
        assert_eq!(model.n_samples_seen(), 3);
    }

    #[test]
    fn reset_clears_state() {
        let mut model = MulticlassSGBT::new(test_config(), 3).unwrap();
        for i in 0..20 {
            model.train_one(&Sample::new(vec![i as f64], (i % 3) as f64));
        }
        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        let proba = model.predict_proba(&[1.0]);
        for &p in &proba {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }
}
