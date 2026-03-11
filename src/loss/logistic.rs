//! Logistic loss for binary classification.
//!
//! L(y, f) = -y * ln(sigma(f)) - (1-y) * ln(1 - sigma(f))
//!
//! where sigma(f) = 1 / (1 + exp(-f)) is the sigmoid function,
//! y in {0, 1} is the binary target, and f is the raw model output (logit).
//!
//! Also known as binary cross-entropy or log loss.

use super::Loss;

/// Logistic (binary cross-entropy) loss.
///
/// Targets must be `0.0` or `1.0`. Predictions are raw logits (unbounded).
/// The sigmoid transform maps logits to probabilities in `[0, 1]`.
#[derive(Debug, Clone, Copy)]
pub struct LogisticLoss;

/// Numerically stable sigmoid: 1 / (1 + exp(-x)).
///
/// Handles large positive and negative inputs without overflow by branching
/// on the sign of x.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

impl Loss for LogisticLoss {
    #[inline]
    fn n_outputs(&self) -> usize {
        1
    }

    #[inline]
    fn gradient(&self, target: f64, prediction: f64) -> f64 {
        sigmoid(prediction) - target
    }

    #[inline]
    fn hessian(&self, _target: f64, prediction: f64) -> f64 {
        let p = sigmoid(prediction);
        (p * (1.0 - p)).max(1e-16)
    }

    fn loss(&self, target: f64, prediction: f64) -> f64 {
        let p = sigmoid(prediction).clamp(1e-15, 1.0 - 1e-15);
        -target * p.ln() - (1.0 - target) * (1.0 - p).ln()
    }

    #[inline]
    fn predict_transform(&self, raw: f64) -> f64 {
        sigmoid(raw)
    }

    fn initial_prediction(&self, targets: &[f64]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        let sum: f64 = targets.iter().sum();
        let mean = (sum / targets.len() as f64).clamp(1e-7, 1.0 - 1e-7);
        // log-odds: ln(p / (1 - p))
        (mean / (1.0 - mean)).ln()
    }

    fn loss_type(&self) -> Option<super::LossType> {
        Some(super::LossType::Logistic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_n_outputs() {
        assert_eq!(LogisticLoss.n_outputs(), 1);
    }

    #[test]
    fn test_sigmoid_basic() {
        assert!((sigmoid(0.0) - 0.5).abs() < EPS);
        // sigmoid(large) ~ 1
        assert!((sigmoid(100.0) - 1.0).abs() < EPS);
        // sigmoid(-large) ~ 0
        assert!(sigmoid(-100.0).abs() < EPS);
        // sigmoid is symmetric: sigmoid(x) + sigmoid(-x) = 1
        let x = 2.5;
        assert!((sigmoid(x) + sigmoid(-x) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_gradient_target_1_pred_0() {
        let loss = LogisticLoss;
        // sigmoid(0) = 0.5, so gradient = 0.5 - 1.0 = -0.5
        let g = loss.gradient(1.0, 0.0);
        assert!((g - (-0.5)).abs() < EPS);
    }

    #[test]
    fn test_gradient_target_0_pred_0() {
        let loss = LogisticLoss;
        // sigmoid(0) = 0.5, gradient = 0.5 - 0.0 = 0.5
        let g = loss.gradient(0.0, 0.0);
        assert!((g - 0.5).abs() < EPS);
    }

    #[test]
    fn test_gradient_perfect_prediction() {
        let loss = LogisticLoss;
        // target=1, prediction very large => sigmoid ~ 1.0, gradient ~ 0
        let g = loss.gradient(1.0, 20.0);
        assert!(g.abs() < 1e-6);
    }

    #[test]
    fn test_hessian_positive() {
        let loss = LogisticLoss;
        // Hessian should always be positive for logistic loss
        assert!(loss.hessian(0.0, 0.0) > 0.0);
        assert!(loss.hessian(1.0, 5.0) > 0.0);
        assert!(loss.hessian(0.0, -5.0) > 0.0);
        assert!(loss.hessian(1.0, 100.0) > 0.0); // extreme, but still > 0 (clamped)
    }

    #[test]
    fn test_hessian_max_at_pred_zero() {
        let loss = LogisticLoss;
        // p*(1-p) is maximized at p=0.5, i.e. prediction=0
        let h_zero = loss.hessian(0.0, 0.0);
        let h_five = loss.hessian(0.0, 5.0);
        assert!((h_zero - 0.25).abs() < EPS);
        assert!(h_five < h_zero);
    }

    #[test]
    fn test_loss_value() {
        let loss = LogisticLoss;
        // At prediction=0, sigmoid=0.5, loss = -ln(0.5) = ln(2) for both classes
        let l1 = loss.loss(1.0, 0.0);
        let l0 = loss.loss(0.0, 0.0);
        let ln2 = 2.0_f64.ln();
        assert!((l1 - ln2).abs() < 1e-8);
        assert!((l0 - ln2).abs() < 1e-8);
    }

    #[test]
    fn test_predict_transform_is_sigmoid() {
        let loss = LogisticLoss;
        assert!((loss.predict_transform(0.0) - 0.5).abs() < EPS);
        assert!(loss.predict_transform(10.0) > 0.99);
        assert!(loss.predict_transform(-10.0) < 0.01);
    }

    #[test]
    fn test_initial_prediction_balanced() {
        let loss = LogisticLoss;
        // Balanced dataset: mean=0.5, log-odds = ln(1) = 0
        let targets = [0.0, 1.0, 0.0, 1.0];
        assert!(loss.initial_prediction(&targets).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_skewed() {
        let loss = LogisticLoss;
        // 75% positive: mean=0.75, log-odds = ln(3) ~ 1.0986
        let targets = [1.0, 1.0, 1.0, 0.0];
        let init = loss.initial_prediction(&targets);
        let expected = (0.75_f64 / 0.25).ln();
        assert!((init - expected).abs() < 1e-8);
    }

    #[test]
    fn test_initial_prediction_empty() {
        let loss = LogisticLoss;
        assert!((loss.initial_prediction(&[])).abs() < EPS);
    }

    #[test]
    fn test_gradient_is_derivative_of_loss() {
        let loss = LogisticLoss;
        let target = 1.0;
        let pred = 1.5;
        let h = 1e-7;
        let numerical = (loss.loss(target, pred + h) - loss.loss(target, pred - h)) / (2.0 * h);
        let analytical = loss.gradient(target, pred);
        assert!(
            (numerical - analytical).abs() < 1e-5,
            "numerical={numerical}, analytical={analytical}"
        );
    }
}
