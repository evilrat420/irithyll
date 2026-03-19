//! Softmax cross-entropy loss for multi-class classification.
//!
//! In gradient boosting, multi-class classification is typically handled by
//! training a separate committee of trees for each class. Each committee
//! member sees binary targets (1.0 if the sample belongs to that class,
//! 0.0 otherwise) and uses logistic-style gradients.
//!
//! The full softmax normalization across classes happens at the ensemble
//! level after all committees produce their raw outputs. Within each
//! committee, the per-class loss reduces to binary logistic form.

use super::Loss;
use crate::math;

/// Softmax (multi-class cross-entropy) loss.
///
/// `n_classes` controls the number of tree committees. Each committee uses
/// logistic-style gradients on one-hot encoded targets.
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxLoss {
    /// Number of classes in the classification problem.
    pub n_classes: usize,
}

/// Numerically stable sigmoid: 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = math::exp(-x);
        1.0 / (1.0 + z)
    } else {
        let z = math::exp(x);
        z / (1.0 + z)
    }
}

impl Loss for SoftmaxLoss {
    #[inline]
    fn n_outputs(&self) -> usize {
        self.n_classes
    }

    #[inline]
    fn gradient(&self, target: f64, prediction: f64) -> f64 {
        // Per-committee binary logistic gradient.
        // target is 1.0 if this sample belongs to this committee's class, else 0.0.
        let indicator = if target == 1.0 { 1.0 } else { 0.0 };
        sigmoid(prediction) - indicator
    }

    #[inline]
    fn hessian(&self, _target: f64, prediction: f64) -> f64 {
        let p = sigmoid(prediction);
        (p * (1.0 - p)).max(1e-16)
    }

    fn loss(&self, target: f64, prediction: f64) -> f64 {
        // Binary cross-entropy for this committee.
        let indicator = if target == 1.0 { 1.0 } else { 0.0 };
        let p = sigmoid(prediction).clamp(1e-15, 1.0 - 1e-15);
        -indicator * math::ln(p) - (1.0 - indicator) * math::ln(1.0 - p)
    }

    #[inline]
    fn predict_transform(&self, raw: f64) -> f64 {
        // Per-committee sigmoid. Full softmax normalization across classes
        // is handled by the ensemble.
        sigmoid(raw)
    }

    fn initial_prediction(&self, _targets: &[f64]) -> f64 {
        // Start each class committee from zero (no prior bias).
        // The boosting loop will learn class priors through early trees.
        0.0
    }

    fn loss_type(&self) -> Option<super::LossType> {
        Some(super::LossType::Softmax {
            n_classes: self.n_classes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_n_outputs() {
        let loss = SoftmaxLoss { n_classes: 5 };
        assert_eq!(loss.n_outputs(), 5);
    }

    #[test]
    fn test_n_outputs_binary() {
        let loss = SoftmaxLoss { n_classes: 2 };
        assert_eq!(loss.n_outputs(), 2);
    }

    #[test]
    fn test_gradient_correct_class() {
        let loss = SoftmaxLoss { n_classes: 3 };
        let g = loss.gradient(1.0, 0.0);
        assert!((g - (-0.5)).abs() < EPS);
    }

    #[test]
    fn test_gradient_wrong_class() {
        let loss = SoftmaxLoss { n_classes: 3 };
        let g = loss.gradient(0.0, 0.0);
        assert!((g - 0.5).abs() < EPS);
    }

    #[test]
    fn test_gradient_confident_correct() {
        let loss = SoftmaxLoss { n_classes: 3 };
        let g = loss.gradient(1.0, 5.0);
        assert!(g < 0.0);
        assert!(g > -0.01);
    }

    #[test]
    fn test_hessian_positive() {
        let loss = SoftmaxLoss { n_classes: 3 };
        assert!(loss.hessian(0.0, 0.0) > 0.0);
        assert!(loss.hessian(1.0, 5.0) > 0.0);
        assert!(loss.hessian(0.0, -5.0) > 0.0);
        assert!(loss.hessian(1.0, 100.0) > 0.0);
    }

    #[test]
    fn test_hessian_max_at_zero() {
        let loss = SoftmaxLoss { n_classes: 3 };
        let h_zero = loss.hessian(0.0, 0.0);
        let h_large = loss.hessian(0.0, 5.0);
        assert!((h_zero - 0.25).abs() < EPS);
        assert!(h_large < h_zero);
    }

    #[test]
    fn test_loss_value_at_zero() {
        let loss = SoftmaxLoss { n_classes: 3 };
        let l1 = loss.loss(1.0, 0.0);
        let l0 = loss.loss(0.0, 0.0);
        let ln2 = math::ln(2.0);
        assert!((l1 - ln2).abs() < 1e-8);
        assert!((l0 - ln2).abs() < 1e-8);
    }

    #[test]
    fn test_loss_decreases_with_correct_prediction() {
        let loss = SoftmaxLoss { n_classes: 3 };
        let l_zero = loss.loss(1.0, 0.0);
        let l_positive = loss.loss(1.0, 3.0);
        assert!(l_positive < l_zero);
    }

    #[test]
    fn test_predict_transform_is_sigmoid() {
        let loss = SoftmaxLoss { n_classes: 3 };
        assert!((loss.predict_transform(0.0) - 0.5).abs() < EPS);
        assert!(loss.predict_transform(10.0) > 0.99);
        assert!(loss.predict_transform(-10.0) < 0.01);
    }

    #[test]
    fn test_initial_prediction_is_zero() {
        let loss = SoftmaxLoss { n_classes: 3 };
        let targets = [0.0, 1.0, 2.0, 1.0, 0.0];
        assert!((loss.initial_prediction(&targets)).abs() < EPS);
        assert!((loss.initial_prediction(&[])).abs() < EPS);
    }

    #[test]
    fn test_gradient_is_derivative_of_loss() {
        let loss = SoftmaxLoss { n_classes: 3 };
        let target = 1.0;
        let pred = 1.5;
        let h = 1e-7;
        let numerical = (loss.loss(target, pred + h) - loss.loss(target, pred - h)) / (2.0 * h);
        let analytical = loss.gradient(target, pred);
        assert!(
            (numerical - analytical).abs() < 1e-5,
            "numerical={numerical}, analytical={analytical}"
        );

        let target = 0.0;
        let pred = -0.5;
        let numerical = (loss.loss(target, pred + h) - loss.loss(target, pred - h)) / (2.0 * h);
        let analytical = loss.gradient(target, pred);
        assert!(
            (numerical - analytical).abs() < 1e-5,
            "numerical={numerical}, analytical={analytical}"
        );
    }
}
