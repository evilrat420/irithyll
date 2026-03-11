//! Loss functions for gradient boosting.
//!
//! Each loss provides gradient and hessian computations used by the boosting
//! loop to compute pseudo-residuals for tree fitting.

pub mod huber;
pub mod logistic;
pub mod softmax;
pub mod squared;

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
    fn initial_prediction(&self, targets: &[f64]) -> f64;
}
