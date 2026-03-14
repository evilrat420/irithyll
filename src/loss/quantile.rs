//! Quantile (pinball) loss for conditional quantile regression.
//!
//! L(y, f) = tau * (y - f)     if y >= f   (under-prediction)
//!         = (1-tau) * (f - y)  if y < f    (over-prediction)
//!
//! The pinball loss has zero Hessian everywhere (it is piecewise linear),
//! which breaks the standard Newton leaf formula w = -G/(H+lambda).
//! We use the pseudo-Huber trick: set hessian = 1.0, so leaf weights
//! become w = -G_sum / (count + lambda), effectively gradient descent.
//! This converges to the empirical conditional quantile as more samples
//! arrive at each leaf -- the same approach used by LightGBM and XGBoost.

use super::{Loss, LossType};

/// Quantile (pinball) loss with target quantile `tau`.
///
/// # Parameters
///
/// - `tau` in (0, 1): target quantile. `tau = 0.5` targets the median,
///   `tau = 0.9` targets the 90th percentile, etc.
///
/// # Example
///
/// ```
/// use irithyll::{SGBTConfig, SGBT};
/// use irithyll::loss::quantile::QuantileLoss;
///
/// // Median regression
/// let config = SGBTConfig::builder().n_steps(50).build().unwrap();
/// let model = SGBT::with_loss(config, QuantileLoss::new(0.5));
/// ```
///
/// # Prediction Intervals
///
/// Train two models -- one for the low quantile, one for the high quantile --
/// to produce prediction intervals:
///
/// ```no_run
/// use irithyll::{SGBTConfig, SGBT};
/// use irithyll::loss::quantile::QuantileLoss;
///
/// let config = SGBTConfig::builder().n_steps(50).build().unwrap();
/// let lower = SGBT::with_loss(config.clone(), QuantileLoss::new(0.05));
/// let upper = SGBT::with_loss(config, QuantileLoss::new(0.95));
/// // 90% prediction interval: [lower.predict(x), upper.predict(x)]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QuantileLoss {
    /// Target quantile in (0, 1).
    pub tau: f64,
}

impl QuantileLoss {
    /// Create a new quantile loss with the given target quantile.
    ///
    /// # Panics
    ///
    /// Panics if `tau` is not in (0, 1).
    pub fn new(tau: f64) -> Self {
        assert!(tau > 0.0 && tau < 1.0, "tau must be in (0, 1), got {tau}");
        Self { tau }
    }
}

impl Loss for QuantileLoss {
    #[inline]
    fn n_outputs(&self) -> usize {
        1
    }

    #[inline]
    fn gradient(&self, target: f64, prediction: f64) -> f64 {
        // Subgradient of the pinball loss w.r.t. prediction (dL/df):
        //   L = tau*(y-f) when y >= f  => dL/df = -tau
        //   L = (1-tau)*(f-y) when y < f => dL/df = 1 - tau
        if prediction >= target {
            1.0 - self.tau
        } else {
            -self.tau
        }
    }

    #[inline]
    fn hessian(&self, _target: f64, _prediction: f64) -> f64 {
        // Pseudo-Huber trick: constant hessian makes leaf weights
        // behave as gradient descent, converging to the empirical quantile.
        1.0
    }

    #[inline]
    fn loss(&self, target: f64, prediction: f64) -> f64 {
        let r = target - prediction;
        if r >= 0.0 {
            self.tau * r
        } else {
            (self.tau - 1.0) * r
        }
    }

    #[inline]
    fn predict_transform(&self, raw: f64) -> f64 {
        raw
    }

    fn initial_prediction(&self, targets: &[f64]) -> f64 {
        if targets.is_empty() {
            return 0.0;
        }
        // Compute the empirical tau-quantile
        let mut sorted: Vec<f64> = targets.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((self.tau * sorted.len() as f64) as usize).min(sorted.len() - 1);
        sorted[idx]
    }

    fn loss_type(&self) -> Option<LossType> {
        Some(LossType::Quantile { tau: self.tau })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_n_outputs() {
        assert_eq!(QuantileLoss::new(0.5).n_outputs(), 1);
    }

    #[test]
    fn test_gradient_over_predict() {
        let loss = QuantileLoss::new(0.9);
        // pred > target: gradient = 1 - tau = 0.1
        assert!((loss.gradient(1.0, 3.0) - 0.1).abs() < EPS);
        assert!((loss.gradient(0.0, 100.0) - 0.1).abs() < EPS);
    }

    #[test]
    fn test_gradient_under_predict() {
        let loss = QuantileLoss::new(0.9);
        // pred < target: gradient = -tau = -0.9
        assert!((loss.gradient(3.0, 1.0) - (-0.9)).abs() < EPS);
        assert!((loss.gradient(100.0, 0.0) - (-0.9)).abs() < EPS);
    }

    #[test]
    fn test_gradient_at_exact() {
        let loss = QuantileLoss::new(0.5);
        // pred == target: gradient = 1 - tau = 0.5 (>= branch)
        assert!((loss.gradient(5.0, 5.0) - 0.5).abs() < EPS);
    }

    #[test]
    fn test_hessian_is_one() {
        let loss = QuantileLoss::new(0.9);
        assert!((loss.hessian(0.0, 0.0) - 1.0).abs() < EPS);
        assert!((loss.hessian(100.0, -50.0) - 1.0).abs() < EPS);
        assert!((loss.hessian(-7.0, 42.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_loss_pinball() {
        let loss = QuantileLoss::new(0.9);
        // Under-prediction (target > pred): loss = tau * (target - pred)
        assert!((loss.loss(5.0, 3.0) - 0.9 * 2.0).abs() < EPS);
        // Over-prediction (target < pred): loss = (1-tau) * (pred - target)
        assert!((loss.loss(3.0, 5.0) - 0.1 * 2.0).abs() < EPS);
        // Exact: loss = 0
        assert!((loss.loss(4.0, 4.0)).abs() < EPS);
    }

    #[test]
    fn test_median_loss_is_half_mae() {
        // At tau=0.5, pinball loss = 0.5 * |target - prediction| = MAE/2
        let loss = QuantileLoss::new(0.5);
        assert!((loss.loss(5.0, 3.0) - 1.0).abs() < EPS); // 0.5 * 2
        assert!((loss.loss(3.0, 5.0) - 1.0).abs() < EPS); // 0.5 * 2
    }

    #[test]
    fn test_predict_transform_is_identity() {
        let loss = QuantileLoss::new(0.5);
        assert!((loss.predict_transform(42.0) - 42.0).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_is_quantile() {
        let loss = QuantileLoss::new(0.5);
        let targets = [1.0, 2.0, 3.0, 4.0, 5.0];
        // Median of [1,2,3,4,5] at index floor(0.5*5)=2 => 3.0
        assert!((loss.initial_prediction(&targets) - 3.0).abs() < EPS);

        let loss90 = QuantileLoss::new(0.9);
        // 90th percentile: index floor(0.9*5)=4 => 5.0
        assert!((loss90.initial_prediction(&targets) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_empty() {
        let loss = QuantileLoss::new(0.5);
        assert!((loss.initial_prediction(&[])).abs() < EPS);
    }

    #[test]
    fn test_loss_type_returns_some() {
        let loss = QuantileLoss::new(0.75);
        match loss.loss_type() {
            Some(LossType::Quantile { tau }) => assert!((tau - 0.75).abs() < EPS),
            other => panic!("expected Quantile, got {other:?}"),
        }
    }

    #[test]
    fn test_gradient_is_subderivative_of_loss() {
        // Numerical check away from the kink point (pred != target)
        let loss = QuantileLoss::new(0.75);
        let target = 2.5;

        // Over-prediction side
        let pred = 4.0;
        let h = 1e-6;
        let numerical = (loss.loss(target, pred + h) - loss.loss(target, pred - h)) / (2.0 * h);
        let analytical = loss.gradient(target, pred);
        assert!(
            (numerical - analytical).abs() < 1e-4,
            "over: numerical={numerical}, analytical={analytical}"
        );

        // Under-prediction side
        let pred2 = 1.0;
        let numerical2 = (loss.loss(target, pred2 + h) - loss.loss(target, pred2 - h)) / (2.0 * h);
        let analytical2 = loss.gradient(target, pred2);
        assert!(
            (numerical2 - analytical2).abs() < 1e-4,
            "under: numerical={numerical2}, analytical={analytical2}"
        );
    }

    #[test]
    #[should_panic(expected = "tau must be in (0, 1)")]
    fn test_invalid_tau_zero() {
        QuantileLoss::new(0.0);
    }

    #[test]
    #[should_panic(expected = "tau must be in (0, 1)")]
    fn test_invalid_tau_one() {
        QuantileLoss::new(1.0);
    }
}
