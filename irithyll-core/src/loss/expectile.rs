//! Expectile loss for asymmetric regression.
//!
//! L(y, f) = w * (f - y)^2,  where w = tau if f >= y, else w = 1 - tau.
//!
//! Generalizes squared error loss: tau = 0.5 recovers standard MSE (up to a
//! factor of 2). Values of tau > 0.5 penalize under-prediction more heavily,
//! producing models that target conditional expectiles above the mean.
//!
//! Unlike quantile (pinball) loss, the expectile loss has a well-defined
//! positive Hessian everywhere, making it natively compatible with second-order
//! gradient boosting (Newton leaf weights work directly).

use super::{Loss, LossType};

/// Expectile loss with asymmetry parameter `tau`.
///
/// # Parameters
///
/// - `tau` in (0, 1): asymmetry parameter. `tau = 0.5` is equivalent to
///   squared loss (up to scaling). `tau > 0.5` penalizes under-prediction
///   more heavily; `tau < 0.5` penalizes over-prediction more heavily.
#[derive(Debug, Clone, Copy)]
pub struct ExpectileLoss {
    /// Asymmetry parameter in (0, 1).
    pub tau: f64,
}

impl ExpectileLoss {
    /// Create a new expectile loss with the given asymmetry parameter.
    ///
    /// # Panics
    ///
    /// Panics if `tau` is not in (0, 1).
    pub fn new(tau: f64) -> Self {
        assert!(tau > 0.0 && tau < 1.0, "tau must be in (0, 1), got {tau}");
        Self { tau }
    }
}

impl Loss for ExpectileLoss {
    #[inline]
    fn n_outputs(&self) -> usize {
        1
    }

    #[inline]
    fn gradient(&self, target: f64, prediction: f64) -> f64 {
        let r = prediction - target;
        let w = if r >= 0.0 { self.tau } else { 1.0 - self.tau };
        2.0 * w * r
    }

    #[inline]
    fn hessian(&self, target: f64, prediction: f64) -> f64 {
        let w = if prediction >= target {
            self.tau
        } else {
            1.0 - self.tau
        };
        2.0 * w
    }

    #[inline]
    fn loss(&self, target: f64, prediction: f64) -> f64 {
        let r = prediction - target;
        let w = if r >= 0.0 { self.tau } else { 1.0 - self.tau };
        w * r * r
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

    fn loss_type(&self) -> Option<LossType> {
        Some(LossType::Expectile { tau: self.tau })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::squared::SquaredLoss;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_n_outputs() {
        assert_eq!(ExpectileLoss::new(0.5).n_outputs(), 1);
    }

    #[test]
    fn test_gradient_symmetric_at_half() {
        let exp = ExpectileLoss::new(0.5);
        let sq = SquaredLoss;
        assert!((exp.gradient(3.0, 5.0) - sq.gradient(3.0, 5.0)).abs() < EPS);
        assert!((exp.gradient(5.0, 3.0) - sq.gradient(5.0, 3.0)).abs() < EPS);
        assert!((exp.gradient(4.0, 4.0) - sq.gradient(4.0, 4.0)).abs() < EPS);
    }

    #[test]
    fn test_gradient_asymmetric() {
        let loss = ExpectileLoss::new(0.9);
        let g_over = loss.gradient(1.0, 3.0);
        assert!((g_over - 3.6).abs() < EPS);
        let g_under = loss.gradient(3.0, 1.0);
        assert!((g_under - (-0.4)).abs() < EPS);
    }

    #[test]
    fn test_hessian_positive_definite() {
        let loss = ExpectileLoss::new(0.9);
        assert!((loss.hessian(1.0, 3.0) - 1.8).abs() < EPS);
        assert!((loss.hessian(3.0, 1.0) - 0.2).abs() < EPS);
        assert!((loss.hessian(2.0, 2.0) - 1.8).abs() < EPS);

        for &tau in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let l = ExpectileLoss::new(tau);
            assert!(l.hessian(0.0, 1.0) > 0.0);
            assert!(l.hessian(1.0, 0.0) > 0.0);
            assert!(l.hessian(5.0, 5.0) > 0.0);
        }
    }

    #[test]
    fn test_loss_value() {
        let loss = ExpectileLoss::new(0.9);
        assert!((loss.loss(1.0, 3.0) - 3.6).abs() < EPS);
        assert!((loss.loss(3.0, 1.0) - 0.4).abs() < EPS);
        assert!((loss.loss(5.0, 5.0)).abs() < EPS);
    }

    #[test]
    fn test_predict_transform_is_identity() {
        let loss = ExpectileLoss::new(0.5);
        assert!((loss.predict_transform(42.0) - 42.0).abs() < EPS);
        assert!((loss.predict_transform(-3.25) - (-3.25)).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_is_mean() {
        let loss = ExpectileLoss::new(0.9);
        let targets = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((loss.initial_prediction(&targets) - 3.0).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_empty() {
        let loss = ExpectileLoss::new(0.5);
        assert!((loss.initial_prediction(&[])).abs() < EPS);
    }

    #[test]
    fn test_gradient_is_derivative_of_loss() {
        let loss = ExpectileLoss::new(0.75);
        let target = 2.5;
        let pred = 4.0;
        let h = 1e-6;
        let numerical = (loss.loss(target, pred + h) - loss.loss(target, pred - h)) / (2.0 * h);
        let analytical = loss.gradient(target, pred);
        assert!(
            (numerical - analytical).abs() < 1e-5,
            "numerical={numerical}, analytical={analytical}"
        );

        let pred2 = 1.0;
        let numerical2 = (loss.loss(target, pred2 + h) - loss.loss(target, pred2 - h)) / (2.0 * h);
        let analytical2 = loss.gradient(target, pred2);
        assert!(
            (numerical2 - analytical2).abs() < 1e-5,
            "numerical={numerical2}, analytical={analytical2}"
        );
    }

    #[test]
    fn test_loss_type_returns_some() {
        let loss = ExpectileLoss::new(0.75);
        match loss.loss_type() {
            Some(LossType::Expectile { tau }) => assert!((tau - 0.75).abs() < EPS),
            other => panic!("expected Expectile, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "tau must be in (0, 1)")]
    fn test_invalid_tau_zero() {
        ExpectileLoss::new(0.0);
    }

    #[test]
    #[should_panic(expected = "tau must be in (0, 1)")]
    fn test_invalid_tau_one() {
        ExpectileLoss::new(1.0);
    }
}
