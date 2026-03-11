//! Huber loss for robust regression.
//!
//! A hybrid loss that behaves as squared error for small residuals and
//! as absolute error for large residuals, controlled by the delta parameter:
//!
//! ```text
//! L(y, f) = 0.5 * (f - y)^2           if |f - y| <= delta
//!         = delta * (|f - y| - 0.5 * delta)  otherwise
//! ```
//!
//! This makes gradient boosting more robust to outliers compared to
//! pure squared loss, while retaining the efficiency of squared loss
//! near the optimum.

use super::Loss;

/// Huber loss with configurable transition threshold `delta`.
///
/// At `|residual| <= delta` the loss is quadratic (like [`SquaredLoss`](super::squared::SquaredLoss)).
/// Beyond that threshold the loss grows linearly, bounding the influence of outliers.
#[derive(Debug, Clone, Copy)]
pub struct HuberLoss {
    /// Threshold at which the loss transitions from quadratic to linear.
    /// Must be positive. Default is `1.0`.
    pub delta: f64,
}

impl HuberLoss {
    /// Create a new Huber loss with the given delta threshold.
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl Loss for HuberLoss {
    #[inline]
    fn n_outputs(&self) -> usize {
        1
    }

    #[inline]
    fn gradient(&self, target: f64, prediction: f64) -> f64 {
        let residual = prediction - target;
        if residual.abs() <= self.delta {
            residual
        } else {
            self.delta * residual.signum()
        }
    }

    #[inline]
    fn hessian(&self, target: f64, prediction: f64) -> f64 {
        let residual = prediction - target;
        if residual.abs() <= self.delta {
            1.0
        } else {
            // Use small epsilon instead of exact 0 to avoid division-by-zero
            // in Newton step computations (leaf weight = -G/H).
            1e-6
        }
    }

    #[inline]
    fn loss(&self, target: f64, prediction: f64) -> f64 {
        let r = (prediction - target).abs();
        if r <= self.delta {
            0.5 * r * r
        } else {
            self.delta * (r - 0.5 * self.delta)
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
        let mut sorted = targets.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
    }

    fn loss_type(&self) -> Option<super::LossType> {
        Some(super::LossType::Huber { delta: self.delta })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_n_outputs() {
        assert_eq!(HuberLoss::default().n_outputs(), 1);
    }

    #[test]
    fn test_default_delta() {
        assert!((HuberLoss::default().delta - 1.0).abs() < EPS);
    }

    #[test]
    fn test_custom_delta() {
        let loss = HuberLoss::new(2.5);
        assert!((loss.delta - 2.5).abs() < EPS);
    }

    #[test]
    fn test_gradient_quadratic_region() {
        let loss = HuberLoss::new(1.0);
        // |prediction - target| = 0.5 <= 1.0, so gradient = prediction - target
        assert!((loss.gradient(3.0, 3.5) - 0.5).abs() < EPS);
        assert!((loss.gradient(3.0, 2.5) - (-0.5)).abs() < EPS);
        // Exactly at zero
        assert!((loss.gradient(2.0, 2.0)).abs() < EPS);
    }

    #[test]
    fn test_gradient_linear_region() {
        let loss = HuberLoss::new(1.0);
        // |prediction - target| = 5.0 > 1.0, so gradient = delta * signum(residual)
        assert!((loss.gradient(0.0, 5.0) - 1.0).abs() < EPS);
        assert!((loss.gradient(5.0, 0.0) - (-1.0)).abs() < EPS);
    }

    #[test]
    fn test_gradient_at_boundary() {
        let loss = HuberLoss::new(1.0);
        // |residual| = delta exactly: both branches should agree
        let g_quad = loss.gradient(0.0, 1.0); // residual = 1.0 = delta
        assert!((g_quad - 1.0).abs() < EPS);
    }

    #[test]
    fn test_hessian_quadratic_region() {
        let loss = HuberLoss::new(1.0);
        assert!((loss.hessian(3.0, 3.5) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_hessian_linear_region() {
        let loss = HuberLoss::new(1.0);
        let h = loss.hessian(0.0, 5.0);
        assert!((h - 1e-6).abs() < EPS);
        assert!(h > 0.0); // Must be positive (epsilon, not zero)
    }

    #[test]
    fn test_loss_quadratic_region() {
        let loss = HuberLoss::new(1.0);
        // |r| = 0.5, loss = 0.5 * 0.25 = 0.125
        assert!((loss.loss(3.0, 3.5) - 0.125).abs() < EPS);
    }

    #[test]
    fn test_loss_linear_region() {
        let loss = HuberLoss::new(1.0);
        // |r| = 3.0, loss = 1.0 * (3.0 - 0.5) = 2.5
        assert!((loss.loss(0.0, 3.0) - 2.5).abs() < EPS);
    }

    #[test]
    fn test_loss_continuity_at_boundary() {
        let loss = HuberLoss::new(1.0);
        // At |r| = delta = 1.0:
        // Quadratic: 0.5 * 1.0 = 0.5
        // Linear: 1.0 * (1.0 - 0.5) = 0.5
        assert!((loss.loss(0.0, 1.0) - 0.5).abs() < EPS);
        // Just inside and outside should be close
        let inside = loss.loss(0.0, 0.999);
        let outside = loss.loss(0.0, 1.001);
        assert!((inside - outside).abs() < 0.01);
    }

    #[test]
    fn test_predict_transform_is_identity() {
        let loss = HuberLoss::default();
        assert!((loss.predict_transform(42.0) - 42.0).abs() < EPS);
        assert!((loss.predict_transform(-3.25) - (-3.25)).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_median_odd() {
        let loss = HuberLoss::default();
        let targets = [5.0, 1.0, 3.0, 4.0, 2.0];
        // Sorted: [1, 2, 3, 4, 5], median = 3
        assert!((loss.initial_prediction(&targets) - 3.0).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_median_even() {
        let loss = HuberLoss::default();
        let targets = [1.0, 4.0, 2.0, 3.0];
        // Sorted: [1, 2, 3, 4], median = (2+3)/2 = 2.5
        assert!((loss.initial_prediction(&targets) - 2.5).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_single() {
        let loss = HuberLoss::default();
        assert!((loss.initial_prediction(&[7.0]) - 7.0).abs() < EPS);
    }

    #[test]
    fn test_initial_prediction_empty() {
        let loss = HuberLoss::default();
        assert!((loss.initial_prediction(&[])).abs() < EPS);
    }

    #[test]
    fn test_gradient_is_derivative_of_loss_quadratic() {
        let loss = HuberLoss::new(2.0);
        let target = 3.0;
        let pred = 3.5; // |r| = 0.5 < 2.0, quadratic region
        let h = 1e-7;
        let numerical = (loss.loss(target, pred + h) - loss.loss(target, pred - h)) / (2.0 * h);
        let analytical = loss.gradient(target, pred);
        assert!(
            (numerical - analytical).abs() < 1e-5,
            "numerical={numerical}, analytical={analytical}"
        );
    }

    #[test]
    fn test_gradient_is_derivative_of_loss_linear() {
        let loss = HuberLoss::new(1.0);
        let target = 0.0;
        let pred = 5.0; // |r| = 5.0 > 1.0, linear region
        let h = 1e-7;
        let numerical = (loss.loss(target, pred + h) - loss.loss(target, pred - h)) / (2.0 * h);
        let analytical = loss.gradient(target, pred);
        assert!(
            (numerical - analytical).abs() < 1e-5,
            "numerical={numerical}, analytical={analytical}"
        );
    }

    #[test]
    fn test_huber_with_large_delta_matches_squared() {
        // With a very large delta, Huber should behave like squared loss
        let huber = HuberLoss::new(1000.0);
        let target = 2.0;
        let pred = 5.0;
        let r = pred - target;
        assert!((huber.loss(target, pred) - 0.5 * r * r).abs() < EPS);
        assert!((huber.gradient(target, pred) - r).abs() < EPS);
        assert!((huber.hessian(target, pred) - 1.0).abs() < EPS);
    }
}
