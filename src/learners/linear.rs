//! Online linear regression via stochastic gradient descent (SGD).
//!
//! [`StreamingLinearModel`] fits a linear model incrementally, one sample at a
//! time, using the classic SGD weight update rule with optional L1, L2, or
//! ElasticNet regularization. Weights are lazily initialized on the first
//! training call, so the model adapts to any dimensionality without an upfront
//! feature count.
//!
//! # Regularization
//!
//! | Variant | Mechanism |
//! |---------|-----------|
//! | [`None`](Regularization::None) | Vanilla SGD -- no penalty |
//! | [`Ridge`](Regularization::Ridge) | L2 weight decay applied before the gradient step |
//! | [`Lasso`](Regularization::Lasso) | L1 proximal (soft-thresholding) after the gradient step |
//! | [`ElasticNet`](Regularization::ElasticNet) | L2 decay + L1 proximal combined |
//!
//! # Usage
//!
//! ```
//! use irithyll::learners::linear::StreamingLinearModel;
//! use irithyll::learner::StreamingLearner;
//!
//! let mut model = StreamingLinearModel::new(0.01);
//! model.train(&[1.0, 2.0], 5.0);
//! model.train(&[3.0, 4.0], 11.0);
//!
//! let pred = model.predict(&[1.0, 2.0]);
//! assert!(pred.is_finite());
//! ```

use std::fmt;

use crate::learner::StreamingLearner;

// ---------------------------------------------------------------------------
// Regularization
// ---------------------------------------------------------------------------

/// Regularization strategy for online linear regression.
///
/// Controls how model weights are penalized during SGD updates. The bias
/// term is **never** regularized regardless of the chosen strategy.
///
/// # Examples
///
/// ```
/// use irithyll::learners::linear::Regularization;
///
/// let none = Regularization::None;
/// let ridge = Regularization::Ridge(0.001);
/// let lasso = Regularization::Lasso(0.001);
/// let elastic = Regularization::ElasticNet { l1: 0.0005, l2: 0.0005 };
/// ```
pub enum Regularization {
    /// No regularization -- vanilla SGD.
    None,
    /// L2 penalty (Ridge). The parameter is the regularization strength lambda.
    ///
    /// Applied as multiplicative weight decay *before* the gradient step:
    /// `w_j *= (1 - lr * lambda)`.
    Ridge(f64),
    /// L1 penalty (Lasso). The parameter is the regularization strength lambda.
    ///
    /// Applied as a proximal (soft-thresholding) operator *after* the gradient
    /// step: `w_j = sign(w_j) * max(0, |w_j| - lr * lambda)`.
    ///
    /// This promotes sparsity by driving small weights to exactly zero.
    Lasso(f64),
    /// Combined L1 + L2 penalty (ElasticNet).
    ///
    /// Applies L2 weight decay before the gradient step and L1 proximal
    /// after it, combining the shrinkage of Ridge with the sparsity of Lasso.
    ElasticNet {
        /// L1 (Lasso) penalty strength.
        l1: f64,
        /// L2 (Ridge) penalty strength.
        l2: f64,
    },
}

// --- Manual Clone impl ---

impl Clone for Regularization {
    fn clone(&self) -> Self {
        match self {
            Self::None => Self::None,
            Self::Ridge(lambda) => Self::Ridge(*lambda),
            Self::Lasso(lambda) => Self::Lasso(*lambda),
            Self::ElasticNet { l1, l2 } => Self::ElasticNet { l1: *l1, l2: *l2 },
        }
    }
}

// --- Manual Debug impl ---

impl fmt::Debug for Regularization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "Regularization::None"),
            Self::Ridge(lambda) => write!(f, "Regularization::Ridge({})", lambda),
            Self::Lasso(lambda) => write!(f, "Regularization::Lasso({})", lambda),
            Self::ElasticNet { l1, l2 } => {
                write!(f, "Regularization::ElasticNet {{ l1: {}, l2: {} }}", l1, l2)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLinearModel
// ---------------------------------------------------------------------------

/// Online linear regression model trained via stochastic gradient descent.
///
/// The model maintains a weight vector and bias term, updated incrementally
/// on each call to [`train_one`](StreamingLearner::train_one). Feature
/// dimensionality is determined lazily on the first training sample -- weights
/// are initialized to zero.
///
/// # Gradient Update
///
/// For each sample *(x, y)* with sample weight *w*:
///
/// 1. Compute prediction: `pred = dot(weights, x) + bias`
/// 2. Compute error: `error = pred - target`
/// 3. Apply L2 weight decay (if applicable): `w_j *= (1 - lr * lambda)`
/// 4. Update weights: `w_j -= lr * w * 2 * error * x_j`
/// 5. Update bias: `bias -= lr * w * 2 * error`
/// 6. Apply L1 proximal operator (if applicable)
///
/// # Examples
///
/// ```
/// use irithyll::learners::linear::StreamingLinearModel;
/// use irithyll::learner::StreamingLearner;
///
/// // Ridge regression with learning rate 0.01 and lambda 0.001
/// let mut model = StreamingLinearModel::ridge(0.01, 0.001);
///
/// for i in 0..100 {
///     let x = i as f64;
///     model.train(&[x], 2.0 * x + 1.0);
/// }
///
/// let pred = model.predict(&[50.0]);
/// assert!(pred.is_finite());
/// ```
pub struct StreamingLinearModel {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    regularization: Regularization,
    n_features: Option<usize>,
    samples_seen: u64,
}

// ---------------------------------------------------------------------------
// Constructors and accessors
// ---------------------------------------------------------------------------

impl StreamingLinearModel {
    /// Create a new linear model with no regularization.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` -- step size for SGD updates (e.g., 0.01)
    ///
    /// # Examples
    ///
    /// ```
    /// use irithyll::learners::linear::StreamingLinearModel;
    ///
    /// let model = StreamingLinearModel::new(0.01);
    /// assert_eq!(model.bias(), 0.0);
    /// ```
    #[inline]
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            regularization: Regularization::None,
            n_features: None,
            samples_seen: 0,
        }
    }

    /// Create a new linear model with L2 (Ridge) regularization.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` -- step size for SGD updates
    /// * `lambda` -- L2 penalty strength
    #[inline]
    pub fn ridge(learning_rate: f64, lambda: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            regularization: Regularization::Ridge(lambda),
            n_features: None,
            samples_seen: 0,
        }
    }

    /// Create a new linear model with L1 (Lasso) regularization.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` -- step size for SGD updates
    /// * `lambda` -- L1 penalty strength
    #[inline]
    pub fn lasso(learning_rate: f64, lambda: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            regularization: Regularization::Lasso(lambda),
            n_features: None,
            samples_seen: 0,
        }
    }

    /// Create a new linear model with ElasticNet (L1 + L2) regularization.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` -- step size for SGD updates
    /// * `l1` -- L1 (Lasso) penalty strength
    /// * `l2` -- L2 (Ridge) penalty strength
    #[inline]
    pub fn elastic_net(learning_rate: f64, l1: f64, l2: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            regularization: Regularization::ElasticNet { l1, l2 },
            n_features: None,
            samples_seen: 0,
        }
    }

    /// Current weight vector.
    ///
    /// Returns an empty slice if no training samples have been seen yet
    /// (weights are lazily initialized).
    #[inline]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Current bias (intercept) term.
    #[inline]
    pub fn bias(&self) -> f64 {
        self.bias
    }

    // --- Private helpers ---

    /// Initialize weights to zero on first training call.
    #[inline]
    fn lazy_init(&mut self, n_features: usize) {
        if self.n_features.is_none() {
            self.n_features = Some(n_features);
            self.weights = vec![0.0; n_features];
        }
    }

    /// Apply L1 soft-thresholding proximal operator to all weights.
    ///
    /// `w_j = sign(w_j) * max(0, |w_j| - threshold)`
    #[inline]
    fn apply_l1_proximal(&mut self, threshold: f64) {
        for w in &mut self.weights {
            let abs_w = w.abs();
            if abs_w <= threshold {
                *w = 0.0;
            } else {
                *w = w.signum() * (abs_w - threshold);
            }
        }
    }

    /// Apply L2 weight decay: `w_j *= (1 - factor)`.
    #[inline]
    fn apply_l2_decay(&mut self, factor: f64) {
        let scale = 1.0 - factor;
        for w in &mut self.weights {
            *w *= scale;
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

impl StreamingLearner for StreamingLinearModel {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        self.lazy_init(features.len());

        // Compute prediction: dot(weights, features) + bias
        let pred: f64 = self
            .weights
            .iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias;

        let error = pred - target;
        let lr = self.learning_rate;

        // Step 1: Apply L2 weight decay (before gradient step)
        match &self.regularization {
            Regularization::Ridge(lambda) => self.apply_l2_decay(lr * lambda),
            Regularization::ElasticNet { l2, .. } => self.apply_l2_decay(lr * l2),
            _ => {}
        }

        // Step 2: SGD gradient update (gradient = 2 * error * x_j, scaled by weight)
        let grad_scale = 2.0 * error * weight * lr;
        for (w, x) in self.weights.iter_mut().zip(features.iter()) {
            *w -= grad_scale * x;
        }

        // Bias update (never regularized)
        self.bias -= grad_scale;

        // Step 3: Apply L1 proximal operator (after gradient step)
        match &self.regularization {
            Regularization::Lasso(lambda) => self.apply_l1_proximal(lr * lambda),
            Regularization::ElasticNet { l1, .. } => self.apply_l1_proximal(lr * l1),
            _ => {}
        }

        self.samples_seen += 1;
    }

    #[inline]
    fn predict(&self, features: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        self.weights.clear();
        self.bias = 0.0;
        self.n_features = None;
        self.samples_seen = 0;
    }
}

// ---------------------------------------------------------------------------
// Clone impl -- manual to match irithyll patterns
// ---------------------------------------------------------------------------

impl Clone for StreamingLinearModel {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            bias: self.bias,
            learning_rate: self.learning_rate,
            regularization: self.regularization.clone(),
            n_features: self.n_features,
            samples_seen: self.samples_seen,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug impl -- manual for consistent formatting
// ---------------------------------------------------------------------------

impl fmt::Debug for StreamingLinearModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamingLinearModel")
            .field("n_features", &self.n_features)
            .field("bias", &self.bias)
            .field("learning_rate", &self.learning_rate)
            .field("regularization", &self.regularization)
            .field("samples_seen", &self.samples_seen)
            .field("n_weights", &self.weights.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let model = StreamingLinearModel::new(0.01);
        assert_eq!(model.n_samples_seen(), 0);
        assert_eq!(model.bias(), 0.0);
        assert!(model.weights().is_empty());
    }

    #[test]
    fn test_lazy_init() {
        let mut model = StreamingLinearModel::new(0.01);
        assert!(model.n_features.is_none());
        assert!(model.weights().is_empty());

        // First training call should initialize weights.
        model.train(&[1.0, 2.0, 3.0], 5.0);
        assert_eq!(model.n_features, Some(3));
        assert_eq!(model.weights().len(), 3);
        assert_eq!(model.n_samples_seen(), 1);
    }

    #[test]
    fn test_no_regularization() {
        // Train on y = 2*x1 + 3*x2 + 1 (true weights: [2, 3], bias: 1)
        let mut model = StreamingLinearModel::new(0.001);

        for i in 0..5000 {
            let x1 = (i as f64 * 0.37).sin();
            let x2 = (i as f64 * 0.73).cos();
            let y = 2.0 * x1 + 3.0 * x2 + 1.0;
            model.train(&[x1, x2], y);
        }

        // Weights should converge near [2, 3] and bias near 1.
        let w = model.weights();
        assert!(
            (w[0] - 2.0).abs() < 0.3,
            "w[0] should be near 2.0, got {}",
            w[0],
        );
        assert!(
            (w[1] - 3.0).abs() < 0.3,
            "w[1] should be near 3.0, got {}",
            w[1],
        );
        assert!(
            (model.bias() - 1.0).abs() < 0.3,
            "bias should be near 1.0, got {}",
            model.bias(),
        );
    }

    #[test]
    fn test_ridge() {
        // Ridge regularization should shrink weights toward zero compared to
        // an unregularized model trained on the same data.
        let mut plain = StreamingLinearModel::new(0.001);
        let mut ridge = StreamingLinearModel::ridge(0.001, 0.1);

        for i in 0..2000 {
            let x1 = (i as f64 * 0.37).sin();
            let x2 = (i as f64 * 0.73).cos();
            let y = 2.0 * x1 + 3.0 * x2 + 1.0;
            plain.train(&[x1, x2], y);
            ridge.train(&[x1, x2], y);
        }

        // Ridge weights should have smaller L2 norm than plain SGD weights.
        let plain_l2: f64 = plain.weights().iter().map(|w| w * w).sum();
        let ridge_l2: f64 = ridge.weights().iter().map(|w| w * w).sum();
        assert!(
            ridge_l2 < plain_l2,
            "Ridge L2 norm ({}) should be less than plain ({})",
            ridge_l2,
            plain_l2,
        );
    }

    #[test]
    fn test_lasso() {
        // Lasso should drive irrelevant feature weights to exactly zero.
        // y = 3*x1 + 0*x2 + 0*x3 (features 2 and 3 are irrelevant noise)
        let mut model = StreamingLinearModel::lasso(0.001, 0.01);

        for i in 0..5000 {
            let x1 = (i as f64 * 0.31).sin();
            let x2 = (i as f64 * 0.97).cos() * 0.01; // tiny irrelevant signal
            let x3 = (i as f64 * 0.53).sin() * 0.01; // tiny irrelevant signal
            let y = 3.0 * x1;
            model.train(&[x1, x2, x3], y);
        }

        let w = model.weights();
        // x2 and x3 have near-zero true coefficients -- Lasso should zero them out.
        assert!(
            w[1].abs() < 0.05,
            "w[1] should be near zero (sparse), got {}",
            w[1],
        );
        assert!(
            w[2].abs() < 0.05,
            "w[2] should be near zero (sparse), got {}",
            w[2],
        );
        // x1 should retain a non-trivial coefficient.
        assert!(w[0].abs() > 0.5, "w[0] should be non-trivial, got {}", w[0],);
    }

    #[test]
    fn test_elastic_net() {
        // ElasticNet combines both L1 and L2 -- weights should be smaller than
        // Ridge alone and Lasso alone (for the same total penalty budget).
        let mut ridge_only = StreamingLinearModel::ridge(0.001, 0.05);
        let mut elastic = StreamingLinearModel::elastic_net(0.001, 0.025, 0.025);

        for i in 0..3000 {
            let x1 = (i as f64 * 0.37).sin();
            let x2 = (i as f64 * 0.73).cos();
            let y = 2.0 * x1 + 3.0 * x2;
            ridge_only.train(&[x1, x2], y);
            elastic.train(&[x1, x2], y);
        }

        // ElasticNet should produce finite, non-zero weights.
        let ew = elastic.weights();
        assert!(ew[0].is_finite() && ew[0].abs() > 0.01);
        assert!(ew[1].is_finite() && ew[1].abs() > 0.01);

        // ElasticNet L2 norm should differ from pure Ridge
        // (confirming the L1 component has an effect).
        let ridge_l2: f64 = ridge_only.weights().iter().map(|w| w * w).sum();
        let elastic_l2: f64 = ew.iter().map(|w| w * w).sum();
        assert!(
            (ridge_l2 - elastic_l2).abs() > 1e-6,
            "ElasticNet and Ridge should produce different weight norms",
        );
    }

    #[test]
    fn test_predict_batch() {
        let mut model = StreamingLinearModel::new(0.01);

        for i in 0..100 {
            let x = i as f64 * 0.1;
            model.train(&[x, x * 2.0], x * 3.0);
        }

        let rows: Vec<&[f64]> = vec![&[1.0, 2.0], &[3.0, 6.0], &[0.5, 1.0]];
        let batch = model.predict_batch(&rows);

        assert_eq!(batch.len(), rows.len());
        for (i, row) in rows.iter().enumerate() {
            let individual = model.predict(row);
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
    fn test_reset() {
        let mut model = StreamingLinearModel::new(0.01);

        // Train some samples.
        for i in 0..50 {
            model.train(&[i as f64, (i as f64) * 2.0], i as f64);
        }
        assert_eq!(model.n_samples_seen(), 50);
        assert!(!model.weights().is_empty());

        // Reset should clear everything.
        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        assert!(model.weights().is_empty());
        assert_eq!(model.bias(), 0.0);

        // After reset, predict should return 0 (no weights).
        let pred = model.predict(&[1.0, 2.0]);
        assert!(
            pred.abs() < 1e-12,
            "prediction after reset should be zero, got {}",
            pred,
        );
    }

    #[test]
    fn test_weighted_samples() {
        // Training with weight=2 should have a stronger effect than weight=1.
        // Both start from the same initial state; we train a single sample
        // and compare the resulting bias shift.
        let mut model_w1 = StreamingLinearModel::new(0.1);
        let mut model_w2 = StreamingLinearModel::new(0.1);

        let features = [1.0, 0.0];
        let target = 10.0;

        // One sample with weight=1
        model_w1.train_one(&features, target, 1.0);

        // One sample with weight=2
        model_w2.train_one(&features, target, 2.0);

        // The weight=2 model should have moved further from zero.
        let shift_w1 = model_w1.bias().abs();
        let shift_w2 = model_w2.bias().abs();
        assert!(
            shift_w2 > shift_w1,
            "weight=2 shift ({}) should exceed weight=1 shift ({})",
            shift_w2,
            shift_w1,
        );

        // Weight=2 gradient is exactly 2x the weight=1 gradient, so the
        // bias change should be exactly double.
        assert!(
            (shift_w2 - 2.0 * shift_w1).abs() < 1e-12,
            "weight=2 shift should be exactly 2x weight=1: {} vs {}",
            shift_w2,
            shift_w1,
        );
    }

    #[test]
    fn test_trait_object() {
        // Verify StreamingLinearModel works behind Box<dyn StreamingLearner>.
        let model = StreamingLinearModel::ridge(0.01, 0.001);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(model);

        boxed.train(&[1.0, 2.0], 5.0);
        assert_eq!(boxed.n_samples_seen(), 1);

        let pred = boxed.predict(&[1.0, 2.0]);
        assert!(pred.is_finite());

        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }
}
