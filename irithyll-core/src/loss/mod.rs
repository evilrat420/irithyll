//! Loss functions for gradient boosting.
//!
//! Each loss provides gradient and hessian computations used by the boosting
//! loop to compute pseudo-residuals for tree fitting.

pub mod expectile;
pub mod huber;
pub mod logistic;
pub mod quantile;
pub mod softmax;
pub mod squared;

// ---------------------------------------------------------------------------
// LossType -- serialization tag for built-in loss functions
// ---------------------------------------------------------------------------

/// Tag identifying a loss function for serialization and reconstruction.
///
/// When saving a model, the loss type is captured so the correct loss function
/// can be reconstructed on load. Built-in losses implement [`Loss::loss_type`]
/// to return `Some(LossType::...)` automatically.
///
/// Custom losses return `None` from `loss_type()` and must be handled manually
/// during serialization.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LossType {
    /// Squared error loss (regression).
    Squared,
    /// Logistic (binary cross-entropy) loss.
    Logistic,
    /// Huber loss with the given delta threshold.
    Huber {
        /// Threshold at which loss transitions from quadratic to linear.
        delta: f64,
    },
    /// Softmax (multi-class cross-entropy) loss.
    Softmax {
        /// Number of classes.
        n_classes: usize,
    },
    /// Expectile loss with asymmetry parameter.
    Expectile {
        /// Asymmetry parameter tau in (0, 1).
        tau: f64,
    },
    /// Quantile (pinball) loss with target quantile.
    Quantile {
        /// Target quantile tau in (0, 1).
        tau: f64,
    },
}

#[cfg(feature = "alloc")]
impl LossType {
    /// Reconstruct the loss function from its serialized tag.
    pub fn into_loss(self) -> alloc::boxed::Box<dyn Loss> {
        match self {
            LossType::Squared => alloc::boxed::Box::new(squared::SquaredLoss),
            LossType::Logistic => alloc::boxed::Box::new(logistic::LogisticLoss),
            LossType::Huber { delta } => alloc::boxed::Box::new(huber::HuberLoss { delta }),
            LossType::Softmax { n_classes } => {
                alloc::boxed::Box::new(softmax::SoftmaxLoss { n_classes })
            }
            LossType::Expectile { tau } => {
                alloc::boxed::Box::new(expectile::ExpectileLoss::new(tau))
            }
            LossType::Quantile { tau } => alloc::boxed::Box::new(quantile::QuantileLoss::new(tau)),
        }
    }
}

// ---------------------------------------------------------------------------
// Loss trait
// ---------------------------------------------------------------------------

/// A differentiable loss function for gradient boosting.
///
/// Implementations must provide first and second derivatives (gradient/hessian)
/// with respect to the prediction, which drive the boosting updates.
pub trait Loss: Send + Sync + 'static {
    /// Number of output dimensions (1 for regression/binary, C for multiclass).
    fn n_outputs(&self) -> usize;

    /// First derivative of the loss with respect to prediction: dL/df.
    fn gradient(&self, target: f64, prediction: f64) -> f64;

    /// Second derivative of the loss with respect to prediction: d^2L/df^2.
    fn hessian(&self, target: f64, prediction: f64) -> f64;

    /// Raw loss value L(target, prediction).
    fn loss(&self, target: f64, prediction: f64) -> f64;

    /// Transform raw model output to final prediction (identity for regression, sigmoid for binary).
    fn predict_transform(&self, raw: f64) -> f64;

    /// Initial constant prediction (before any trees). Typically the optimal constant
    /// that minimizes sum of losses over the given targets.
    fn initial_prediction(&self, _targets: &[f64]) -> f64 {
        0.0
    }

    /// Return the serialization tag for this loss function.
    ///
    /// Built-in losses return `Some(LossType::...)`. Custom losses default to
    /// `None`, which means the model cannot be auto-serialized.
    fn loss_type(&self) -> Option<LossType> {
        None
    }
}

// ---------------------------------------------------------------------------
// Box<dyn Loss> blanket impl -- enables DynSGBT = SGBT<Box<dyn Loss>>
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl Loss for alloc::boxed::Box<dyn Loss> {
    #[inline]
    fn n_outputs(&self) -> usize {
        (**self).n_outputs()
    }

    #[inline]
    fn gradient(&self, target: f64, prediction: f64) -> f64 {
        (**self).gradient(target, prediction)
    }

    #[inline]
    fn hessian(&self, target: f64, prediction: f64) -> f64 {
        (**self).hessian(target, prediction)
    }

    #[inline]
    fn loss(&self, target: f64, prediction: f64) -> f64 {
        (**self).loss(target, prediction)
    }

    #[inline]
    fn predict_transform(&self, raw: f64) -> f64 {
        (**self).predict_transform(raw)
    }

    fn initial_prediction(&self, targets: &[f64]) -> f64 {
        (**self).initial_prediction(targets)
    }

    fn loss_type(&self) -> Option<LossType> {
        (**self).loss_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_type_variants() {
        let sq = LossType::Squared;
        assert_eq!(sq, LossType::Squared);

        let hub = LossType::Huber { delta: 1.5 };
        assert_eq!(hub, LossType::Huber { delta: 1.5 });

        let soft = LossType::Softmax { n_classes: 3 };
        assert_eq!(soft, LossType::Softmax { n_classes: 3 });

        let exp = LossType::Expectile { tau: 0.9 };
        assert_eq!(exp, LossType::Expectile { tau: 0.9 });

        let quant = LossType::Quantile { tau: 0.5 };
        assert_eq!(quant, LossType::Quantile { tau: 0.5 });
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_into_loss_squared() {
        let loss = LossType::Squared.into_loss();
        assert_eq!(loss.n_outputs(), 1);
        assert!((loss.gradient(3.0, 5.0) - 2.0).abs() < 1e-12);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_into_loss_logistic() {
        let loss = LossType::Logistic.into_loss();
        assert_eq!(loss.n_outputs(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_into_loss_huber() {
        let loss = LossType::Huber { delta: 1.0 }.into_loss();
        assert_eq!(loss.n_outputs(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_into_loss_softmax() {
        let loss = LossType::Softmax { n_classes: 5 }.into_loss();
        assert_eq!(loss.n_outputs(), 5);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_into_loss_expectile() {
        let loss = LossType::Expectile { tau: 0.9 }.into_loss();
        assert_eq!(loss.n_outputs(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_into_loss_quantile() {
        let loss = LossType::Quantile { tau: 0.5 }.into_loss();
        assert_eq!(loss.n_outputs(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_box_dyn_loss_blanket() {
        let loss: alloc::boxed::Box<dyn Loss> = alloc::boxed::Box::new(squared::SquaredLoss);
        assert_eq!(loss.n_outputs(), 1);
        assert!((loss.gradient(3.0, 5.0) - 2.0).abs() < 1e-12);
        assert!((loss.hessian(3.0, 5.0) - 1.0).abs() < 1e-12);
        assert!((loss.loss(1.0, 3.0) - 2.0).abs() < 1e-12);
        assert!((loss.predict_transform(42.0) - 42.0).abs() < 1e-12);
        assert!(loss.loss_type().is_some());
    }
}
