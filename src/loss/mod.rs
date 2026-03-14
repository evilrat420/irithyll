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
/// during serialization (see [`SGBT::to_model_state_with`](crate::SGBT::to_model_state_with)).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    any(feature = "serde-json", feature = "serde-bincode"),
    derive(serde::Serialize, serde::Deserialize)
)]
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

impl LossType {
    /// Reconstruct the loss function from its serialized tag.
    pub fn into_loss(self) -> Box<dyn Loss> {
        match self {
            LossType::Squared => Box::new(squared::SquaredLoss),
            LossType::Logistic => Box::new(logistic::LogisticLoss),
            LossType::Huber { delta } => Box::new(huber::HuberLoss { delta }),
            LossType::Softmax { n_classes } => Box::new(softmax::SoftmaxLoss { n_classes }),
            LossType::Expectile { tau } => Box::new(expectile::ExpectileLoss::new(tau)),
            LossType::Quantile { tau } => Box::new(quantile::QuantileLoss::new(tau)),
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
///
/// # Generic SGBT
///
/// [`SGBT`](crate::SGBT) is generic over `L: Loss`, so the loss function is
/// monomorphized into the boosting loop -- no virtual dispatch overhead on the
/// gradient/hessian hot path.
///
/// ```
/// use irithyll::{SGBTConfig, SGBT};
/// use irithyll::loss::logistic::LogisticLoss;
///
/// let config = SGBTConfig::builder().n_steps(10).build().unwrap();
/// let model = SGBT::with_loss(config, LogisticLoss); // no Box::new()!
/// ```
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
    fn initial_prediction(&self, targets: &[f64]) -> f64;

    /// Return the serialization tag for this loss function.
    ///
    /// Built-in losses return `Some(LossType::...)`. Custom losses default to
    /// `None`, which means the model cannot be auto-serialized -- use
    /// [`SGBT::to_model_state_with`](crate::SGBT::to_model_state_with) to
    /// supply the tag manually.
    fn loss_type(&self) -> Option<LossType> {
        None
    }
}

// ---------------------------------------------------------------------------
// Box<dyn Loss> blanket impl -- enables DynSGBT = SGBT<Box<dyn Loss>>
// ---------------------------------------------------------------------------

impl Loss for Box<dyn Loss> {
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
