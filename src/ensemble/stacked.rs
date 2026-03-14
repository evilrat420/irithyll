//! Polymorphic model stacking meta-learner for streaming ensembles.
//!
//! [`StackedEnsemble`] implements *stacked generalization* (Wolpert, 1992) in a
//! streaming context. Multiple heterogeneous base learners -- any type implementing
//! [`StreamingLearner`] -- produce predictions that are fed as features to a
//! meta-learner which learns to optimally combine them.
//!
//! # Temporal Holdout Stacking
//!
//! In batch stacking, cross-validation prevents the meta-learner from seeing
//! memorized training predictions. In a streaming setting we use **temporal
//! holdout**: for each incoming sample `(x, y, w)`, base predictions are
//! collected *before* the base learners are trained on that sample. This ensures
//! the meta-learner always sees honest, out-of-sample-like predictions rather
//! than memorized values -- the streaming analogue of leave-one-out stacking.
//!
//! # Recursive Stacking
//!
//! Because `StackedEnsemble` itself implements [`StreamingLearner`], it can be
//! used as a base learner inside another `StackedEnsemble`, enabling arbitrarily
//! deep stacking hierarchies.
//!
//! # Example
//!
//! ```
//! use irithyll::learner::{StreamingLearner, SGBTLearner};
//! use irithyll::learners::linear::StreamingLinearModel;
//! use irithyll::ensemble::stacked::StackedEnsemble;
//! use irithyll::SGBTConfig;
//!
//! let config = SGBTConfig::builder()
//!     .n_steps(5)
//!     .learning_rate(0.1)
//!     .grace_period(10)
//!     .max_depth(3)
//!     .n_bins(8)
//!     .build()
//!     .unwrap();
//!
//! let bases: Vec<Box<dyn StreamingLearner>> = vec![
//!     Box::new(SGBTLearner::from_config(config)),
//!     Box::new(StreamingLinearModel::new(0.01)),
//! ];
//! let meta: Box<dyn StreamingLearner> = Box::new(StreamingLinearModel::new(0.01));
//!
//! let mut stack = StackedEnsemble::new(bases, meta);
//! stack.train(&[1.0, 2.0], 3.0);
//! let pred = stack.predict(&[1.0, 2.0]);
//! assert!(pred.is_finite());
//! ```

use std::fmt;

use crate::learner::StreamingLearner;

// ---------------------------------------------------------------------------
// StackedEnsemble
// ---------------------------------------------------------------------------

/// Polymorphic model stacking meta-learner using `Box<dyn StreamingLearner>`.
///
/// Combines predictions from heterogeneous base learners through a trainable
/// meta-learner. Uses temporal holdout to prevent information leakage: base
/// predictions are collected *before* training the bases on each sample.
///
/// # Note on `Clone`
///
/// `StackedEnsemble` cannot implement `Clone` because `Box<dyn StreamingLearner>`
/// is not `Clone`. If you need to snapshot the ensemble, serialize it instead.
pub struct StackedEnsemble {
    /// Base learners -- heterogeneous models wrapped as trait objects.
    base_learners: Vec<Box<dyn StreamingLearner>>,
    /// Meta-learner that combines base predictions.
    meta_learner: Box<dyn StreamingLearner>,
    /// Whether to pass original features alongside base predictions to the meta-learner.
    passthrough: bool,
    /// Total samples trained on.
    samples_seen: u64,
}

// ---------------------------------------------------------------------------
// Constructors and accessors
// ---------------------------------------------------------------------------

impl StackedEnsemble {
    /// Create a new stacked ensemble with passthrough disabled.
    ///
    /// The meta-learner receives only base learner predictions as features.
    ///
    /// # Arguments
    ///
    /// * `base_learners` -- heterogeneous base models (at least one recommended)
    /// * `meta_learner` -- combiner model trained on base predictions
    #[inline]
    pub fn new(
        base_learners: Vec<Box<dyn StreamingLearner>>,
        meta_learner: Box<dyn StreamingLearner>,
    ) -> Self {
        Self {
            base_learners,
            meta_learner,
            passthrough: false,
            samples_seen: 0,
        }
    }

    /// Create a new stacked ensemble with configurable feature passthrough.
    ///
    /// When `passthrough` is `true`, the meta-learner receives both base
    /// predictions *and* the original feature vector, enabling it to learn
    /// corrections that depend on raw inputs.
    ///
    /// # Arguments
    ///
    /// * `base_learners` -- heterogeneous base models
    /// * `meta_learner` -- combiner model
    /// * `passthrough` -- if `true`, original features are appended to meta-features
    #[inline]
    pub fn with_passthrough(
        base_learners: Vec<Box<dyn StreamingLearner>>,
        meta_learner: Box<dyn StreamingLearner>,
        passthrough: bool,
    ) -> Self {
        Self {
            base_learners,
            meta_learner,
            passthrough,
            samples_seen: 0,
        }
    }

    /// Number of base learners in the ensemble.
    #[inline]
    pub fn n_base_learners(&self) -> usize {
        self.base_learners.len()
    }

    /// Whether original features are passed through to the meta-learner.
    #[inline]
    pub fn passthrough(&self) -> bool {
        self.passthrough
    }

    /// Get predictions from each base learner for inspection.
    ///
    /// Returns a vector with one prediction per base learner, in the same
    /// order they were provided at construction time.
    #[inline]
    pub fn base_predictions(&self, features: &[f64]) -> Vec<f64> {
        self.base_learners
            .iter()
            .map(|learner| learner.predict(features))
            .collect()
    }

    /// Build the meta-feature vector from base predictions and optional original features.
    ///
    /// Layout: `[base1_pred, base2_pred, ..., baseK_pred, (original features if passthrough)]`
    fn build_meta_features(&self, features: &[f64], base_preds: &[f64]) -> Vec<f64> {
        if self.passthrough {
            let mut meta_features = Vec::with_capacity(base_preds.len() + features.len());
            meta_features.extend_from_slice(base_preds);
            meta_features.extend_from_slice(features);
            meta_features
        } else {
            base_preds.to_vec()
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl -- enables recursive stacking
// ---------------------------------------------------------------------------

impl StreamingLearner for StackedEnsemble {
    /// Train on a single weighted observation using temporal holdout.
    ///
    /// 1. Collect base predictions **before** training (temporal holdout).
    /// 2. Build meta-features and train the meta-learner on `(meta_features, target, weight)`.
    /// 3. Train each base learner on `(features, target, weight)`.
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // Step 1: Collect pre-training predictions from base learners.
        let base_preds: Vec<f64> = self
            .base_learners
            .iter()
            .map(|learner| learner.predict(features))
            .collect();

        // Step 2: Build meta-features and train the meta-learner.
        let meta_features = self.build_meta_features(features, &base_preds);
        self.meta_learner.train_one(&meta_features, target, weight);

        // Step 3: Train base learners AFTER meta-learner has used their predictions.
        for learner in &mut self.base_learners {
            learner.train_one(features, target, weight);
        }

        self.samples_seen += 1;
    }

    /// Predict by collecting base predictions and passing them through the meta-learner.
    #[inline]
    fn predict(&self, features: &[f64]) -> f64 {
        let base_preds = self.base_predictions(features);
        let meta_features = self.build_meta_features(features, &base_preds);
        self.meta_learner.predict(&meta_features)
    }

    /// Total number of samples trained on since creation or last reset.
    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Reset all base learners, the meta-learner, and the sample counter.
    fn reset(&mut self) {
        for learner in &mut self.base_learners {
            learner.reset();
        }
        self.meta_learner.reset();
        self.samples_seen = 0;
    }
}

// ---------------------------------------------------------------------------
// Debug impl -- manual since Box<dyn StreamingLearner> does not impl Debug
// ---------------------------------------------------------------------------

impl fmt::Debug for StackedEnsemble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StackedEnsemble")
            .field("n_base_learners", &self.base_learners.len())
            .field("passthrough", &self.passthrough)
            .field("samples_seen", &self.samples_seen)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::SGBTLearner;
    use crate::learners::linear::StreamingLinearModel;
    use crate::SGBTConfig;

    /// Shared minimal SGBT config for tests.
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

    /// Create a pair of SGBT base learners as trait objects.
    fn sgbt_bases() -> Vec<Box<dyn StreamingLearner>> {
        vec![
            Box::new(SGBTLearner::from_config(test_config())),
            Box::new(SGBTLearner::from_config(test_config())),
        ]
    }

    /// Create a linear meta-learner as a trait object.
    fn linear_meta() -> Box<dyn StreamingLearner> {
        Box::new(StreamingLinearModel::new(0.01))
    }

    #[test]
    fn test_creation() {
        let stack = StackedEnsemble::new(sgbt_bases(), linear_meta());
        assert_eq!(stack.n_base_learners(), 2);
        assert!(!stack.passthrough());
        assert_eq!(stack.n_samples_seen(), 0);
    }

    #[test]
    fn test_train_and_predict() {
        let mut stack = StackedEnsemble::new(sgbt_bases(), linear_meta());

        // Train on a simple pattern.
        for i in 0..50 {
            let x = i as f64 * 0.1;
            stack.train(&[x, x * 2.0], x * 3.0);
        }

        assert_eq!(stack.n_samples_seen(), 50);

        // Prediction should be finite and non-trivial after training.
        let pred = stack.predict(&[1.0, 2.0]);
        assert!(
            pred.is_finite(),
            "prediction should be finite, got {}",
            pred
        );
    }

    #[test]
    fn test_temporal_holdout() {
        // Verify that the meta-learner sees pre-training predictions by
        // checking that base learner sample counts advance correctly:
        // after training the stack once, each base should have seen 1 sample.
        let mut stack = StackedEnsemble::new(sgbt_bases(), linear_meta());

        // Before training, base learners have seen 0 samples.
        for bp in &stack.base_learners {
            assert_eq!(bp.n_samples_seen(), 0);
        }

        // Train one sample through the stack.
        stack.train(&[1.0, 2.0], 3.0);

        // After training, each base learner has seen exactly 1 sample.
        // The temporal holdout guarantee is that the meta-learner was trained
        // on predictions made *before* this sample was ingested by the bases.
        for bp in &stack.base_learners {
            assert_eq!(bp.n_samples_seen(), 1);
        }
        assert_eq!(stack.meta_learner.n_samples_seen(), 1);
        assert_eq!(stack.n_samples_seen(), 1);

        // Train a second sample and verify counts advance together.
        stack.train(&[3.0, 4.0], 5.0);
        for bp in &stack.base_learners {
            assert_eq!(bp.n_samples_seen(), 2);
        }
        assert_eq!(stack.meta_learner.n_samples_seen(), 2);
        assert_eq!(stack.n_samples_seen(), 2);
    }

    #[test]
    fn test_passthrough() {
        // With passthrough=true, meta-features should include original features.
        // We can verify this indirectly: a passthrough stack with a linear meta
        // should produce a different prediction than a non-passthrough stack,
        // because the meta-learner sees a wider feature vector.
        let bases_a = sgbt_bases();
        let bases_b = sgbt_bases();

        let mut no_pass = StackedEnsemble::new(bases_a, linear_meta());
        let mut with_pass = StackedEnsemble::with_passthrough(bases_b, linear_meta(), true);

        assert!(!no_pass.passthrough());
        assert!(with_pass.passthrough());

        // Train both on the same data.
        for i in 0..30 {
            let x = i as f64 * 0.1;
            let features = [x, x * 2.0];
            let target = x * 3.0 + 1.0;
            no_pass.train(&features, target);
            with_pass.train(&features, target);
        }

        // Verify meta-feature dimensions differ by checking that build_meta_features
        // produces different-length vectors.
        let features = [1.0, 2.0];
        let base_preds = [0.5, 0.7]; // mock base predictions
        let meta_no = no_pass.build_meta_features(&features, &base_preds);
        let meta_yes = with_pass.build_meta_features(&features, &base_preds);

        assert_eq!(meta_no.len(), 2, "no passthrough: only base predictions");
        assert_eq!(
            meta_yes.len(),
            4,
            "passthrough: base predictions + original features"
        );
        assert!(
            (meta_yes[2] - 1.0).abs() < 1e-12,
            "original features appended"
        );
        assert!(
            (meta_yes[3] - 2.0).abs() < 1e-12,
            "original features appended"
        );
    }

    #[test]
    fn test_base_predictions() {
        let mut stack = StackedEnsemble::new(sgbt_bases(), linear_meta());

        // Before training, base predictions should all be zero (untrained models).
        let preds = stack.base_predictions(&[1.0, 2.0]);
        assert_eq!(preds.len(), 2);
        for p in &preds {
            assert!(
                p.abs() < 1e-12,
                "untrained base should predict ~0, got {}",
                p
            );
        }

        // Train a few samples.
        for i in 0..20 {
            let x = i as f64;
            stack.train(&[x, x * 0.5], x * 2.0);
        }

        // Base predictions should still return the correct count.
        let preds_after = stack.base_predictions(&[5.0, 2.5]);
        assert_eq!(preds_after.len(), 2);
        for p in &preds_after {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_reset() {
        let mut stack = StackedEnsemble::new(sgbt_bases(), linear_meta());

        // Train some data.
        for i in 0..30 {
            let x = i as f64 * 0.1;
            stack.train(&[x, x * 2.0], x * 3.0);
        }
        assert_eq!(stack.n_samples_seen(), 30);

        // Reset everything.
        stack.reset();
        assert_eq!(stack.n_samples_seen(), 0);

        // All base learners should be reset.
        for bp in &stack.base_learners {
            assert_eq!(bp.n_samples_seen(), 0);
        }

        // Meta-learner should be reset.
        assert_eq!(stack.meta_learner.n_samples_seen(), 0);

        // Predictions after reset should be near zero (untrained state).
        let pred = stack.predict(&[1.0, 2.0]);
        assert!(
            pred.abs() < 1e-12,
            "prediction after reset should be ~0, got {}",
            pred,
        );
    }

    #[test]
    fn test_n_samples_seen() {
        let mut stack = StackedEnsemble::new(sgbt_bases(), linear_meta());

        assert_eq!(stack.n_samples_seen(), 0);

        for i in 1..=10 {
            stack.train(&[i as f64], i as f64);
            assert_eq!(stack.n_samples_seen(), i);
        }

        // Weighted training also increments by 1 (sample count, not weight sum).
        stack.train_one(&[11.0], 11.0, 5.0);
        assert_eq!(stack.n_samples_seen(), 11);
    }

    #[test]
    fn test_trait_object() {
        // StackedEnsemble itself should work as Box<dyn StreamingLearner>,
        // enabling recursive stacking.
        let stack = StackedEnsemble::new(sgbt_bases(), linear_meta());
        let mut boxed: Box<dyn StreamingLearner> = Box::new(stack);

        boxed.train(&[1.0, 2.0], 3.0);
        assert_eq!(boxed.n_samples_seen(), 1);

        let pred = boxed.predict(&[1.0, 2.0]);
        assert!(pred.is_finite());

        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }

    #[test]
    fn test_heterogeneous_bases() {
        // Mix SGBT and linear base learners -- the core polymorphism use case.
        let bases: Vec<Box<dyn StreamingLearner>> = vec![
            Box::new(SGBTLearner::from_config(test_config())),
            Box::new(StreamingLinearModel::new(0.01)),
            Box::new(StreamingLinearModel::ridge(0.01, 0.001)),
        ];
        let meta = linear_meta();

        let mut stack = StackedEnsemble::new(bases, meta);
        assert_eq!(stack.n_base_learners(), 3);

        // Train on a linear-ish pattern. Both SGBT and linear models should
        // contribute meaningful predictions.
        for i in 0..40 {
            let x = i as f64 * 0.1;
            stack.train(&[x, x * 0.5], 2.0 * x + 1.0);
        }

        assert_eq!(stack.n_samples_seen(), 40);

        let preds = stack.base_predictions(&[2.0, 1.0]);
        assert_eq!(preds.len(), 3);
        for p in &preds {
            assert!(p.is_finite(), "base prediction should be finite, got {}", p);
        }

        let final_pred = stack.predict(&[2.0, 1.0]);
        assert!(final_pred.is_finite());
    }

    #[test]
    fn test_predict_batch() {
        let mut stack = StackedEnsemble::new(sgbt_bases(), linear_meta());

        // Train enough samples for non-trivial predictions.
        for i in 0..30 {
            let x = i as f64 * 0.1;
            stack.train(&[x, x * 2.0], x * 3.0);
        }

        let rows: Vec<&[f64]> = vec![&[0.5, 1.0], &[1.5, 3.0], &[2.5, 5.0]];
        let batch = stack.predict_batch(&rows);

        // Batch results should exactly match individual predictions.
        assert_eq!(batch.len(), rows.len());
        for (i, row) in rows.iter().enumerate() {
            let individual = stack.predict(row);
            assert!(
                (batch[i] - individual).abs() < 1e-12,
                "batch[{}]={} != individual={}",
                i,
                batch[i],
                individual,
            );
        }
    }

    #[test]
    fn test_debug_impl() {
        let stack = StackedEnsemble::new(sgbt_bases(), linear_meta());
        let debug_str = format!("{:?}", stack);
        assert!(debug_str.contains("StackedEnsemble"));
        assert!(debug_str.contains("n_base_learners: 2"));
        assert!(debug_str.contains("passthrough: false"));
        assert!(debug_str.contains("samples_seen: 0"));
    }
}
