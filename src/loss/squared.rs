//! Squared error loss for regression.
//!
//! L(y, f) = 0.5 * (f - y)^2
//!
//! The default choice for regression tasks. Gradient is simply the residual,
//! and the hessian is a constant 1.0, making Newton steps equivalent to
//! gradient steps.

use super::Loss;

/// Mean squared error loss (L2 loss, scaled by 0.5).
///
/// Optimal for regression when the noise is Gaussian.
/// Sensitive to outliers — consider [`HuberLoss`](super::huber::HuberLoss)
/// for heavy-tailed distributions.
pub struct SquaredLoss;

impl Loss for SquaredLoss {
    #[inline]
    fn n_outputs(&self) -> usize {
        1
    }

    #[inline]
    fn gradient(&self, target: f64, prediction: f64) -> f64 {
        prediction - target
    }

    #[inline]
    fn hessian(&self, _target: f64, _prediction: f64) -> f64 {
        1.0
    }

    #[inline]
    fn loss(&self, target: f64, prediction: f64) -> f64 {
        let r = prediction - target;
        0.5 * r * r
    }

    #[inline]
    fn predict_transform(&self, raw: f64) -> f64 {
        raw
    }

    fn initial_prediction(&self, targets: &[f64]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        let sum: f64 = targets.iter().sum();
        sum / targets.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_n_outputs() {
        assert_eq!(SquaredLoss.n_outputs(), 1);
    }

    #[test]
    fn test_gradient_at_known_points() {
        let loss = SquaredLoss;
        // prediction == target => gradient is 0
        assert!((loss.gradient(3.0, 3.0)).abs() < EPS);
        // prediction > target => positive gradient
        assert!((loss.gradient(1.0, 4.0) - 3.0).abs() < EPS);
        // prediction < target => negative gradient
        assert!((loss.gradient(5.0, 2.0) - (-3.0)).abs() < EPS);
    }

    #[test]
    fn test_hessian_is_constant() {
        let loss = SquaredLoss;
        assert!((loss.hessian(0.0, 0.0) - 1.0).abs() < EPS);
        assert!((loss.hessian(100.0, -50.0) - 1.0).abs() < EPS);
        assert!((loss.hessian(-7.0, 42.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_loss_value() {
        let loss = SquaredLoss;
        assert!((loss.loss(1.0, 3.0) - 2.0).abs() < EPS); // 0.5 * 4 = 2
        assert!((loss.loss(5.0, 5.0)).abs() < EPS);
        assert!((loss.loss(0.0, 1.0) - 0.5).abs() < EPS);
    }

    #[test]
    fn test_predict_transform_is_identity() {
        let loss = SquaredLoss;
        assert!((loss.predict_transform(42.0) - 42.0).abs() < EPS);
        assert!((loss.predict_transform(-3.25) - (-3.25)).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_is_mean() {
        let loss = SquaredLoss;
        let targets = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((loss.initial_prediction(&targets) - 3.0).abs() < EPS);

        let single = [7.0];
        assert!((loss.initial_prediction(&single) - 7.0).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_empty() {
        let loss = SquaredLoss;
        assert!((loss.initial_prediction(&[])).abs() < EPS);
    }

    #[test]
    fn test_gradient_is_derivative_of_loss() {
        // Numerical gradient check: dL/df ~= (L(f+h) - L(f-h)) / (2h)
        let loss = SquaredLoss;
        let target = 2.5;
        let pred = 4.0;
        let h = 1e-6;
        let numerical = (loss.loss(target, pred + h) - loss.loss(target, pred - h)) / (2.0 * h);
        let analytical = loss.gradient(target, pred);
        assert!((numerical - analytical).abs() < 1e-5);
    }
}
