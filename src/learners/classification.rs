//! Classification output layer for streaming learners.
//!
//! Wraps any [`StreamingLearner`] to add binary or multiclass classification
//! without modifying the underlying model. Uses bipolar targets {-1, +1}
//! internally for better MSE-based discrimination (standard in ESN/reservoir
//! computing literature). Binary mode thresholds at 0.0; multiclass uses K
//! one-vs-rest heads with stable softmax at inference.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────┐
//! │  ClassificationWrapper                              │
//! │  ┌─────────────┐  ┌────────────────────────────┐   │
//! │  │ inner model  │  │ extra_heads (K-1 for multi)│   │
//! │  │  (head 0)    │  │  [RLS_1, RLS_2, ..., K-1]  │   │
//! │  └─────────────┘  └────────────────────────────┘   │
//! │       │                     │                       │
//! │       └─────────┬───────────┘                       │
//! │           softmax / sigmoid                         │
//! │              → class prediction                     │
//! └────────────────────────────────────────────────────┘
//! ```
//!
//! # Examples
//!
//! Binary classification with any streaming model:
//!
//! ```
//! use irithyll::{binary_classifier, rls, StreamingLearner};
//!
//! let mut clf = binary_classifier(rls(0.99));
//! // Train with {0.0, 1.0} targets
//! clf.train(&[1.0, 2.0], 1.0);
//! clf.train(&[-1.0, -2.0], 0.0);
//! let pred = clf.predict(&[1.0, 2.0]);
//! // Returns sigmoid(raw_output) thresholded: 0.0 or 1.0
//! assert!(pred == 0.0 || pred == 1.0);
//! ```
//!
//! Multiclass classification:
//!
//! ```
//! use irithyll::{multiclass_classifier, rls, StreamingLearner};
//!
//! let mut clf = multiclass_classifier(rls(0.99), 3);
//! for i in 0..30 {
//!     clf.train(&[i as f64, (i % 3) as f64], (i % 3) as f64);
//! }
//! let pred = clf.predict(&[1.0, 1.0]);
//! assert!(pred >= 0.0 && pred < 3.0);
//! ```

use crate::learner::StreamingLearner;
use crate::learners::rls::RecursiveLeastSquares;

// ---------------------------------------------------------------------------
// ClassificationMode
// ---------------------------------------------------------------------------

/// Classification mode for a wrapped streaming learner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClassificationMode {
    /// Raw regression output (default, no transformation).
    Regression,
    /// Binary classification: bipolar {-1, +1} targets, threshold at 0.0.
    Binary,
    /// Multiclass: K RLS heads, softmax over outputs, return argmax class.
    Multiclass {
        /// Number of classes (K >= 2).
        n_classes: usize,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sigmoid function: 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable softmax: subtract max before exp to prevent overflow.
fn stable_softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&z| (z - max_logit).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// ClassificationWrapper
// ---------------------------------------------------------------------------

/// Generic wrapper that adds classification to any [`StreamingLearner`].
///
/// For binary mode, the inner model is trained with bipolar {-1, +1}
/// targets (mapped from user-supplied {0, 1}) for better MSE-based
/// discrimination. The predicted class is 1 if raw output >= 0.0, else 0.
/// Probabilities are obtained via sigmoid(raw_output).
///
/// For multiclass mode with K classes, the wrapper maintains K-1 additional
/// [`RecursiveLeastSquares`] heads. The inner model serves as head 0. On
/// each `train(features, class_label)` call, every head k is trained with
/// bipolar target `+1.0` if `class_label == k`, else `-1.0`. On `predict`,
/// the K raw outputs are passed through stable softmax and the argmax class
/// index is returned as `f64`.
pub struct ClassificationWrapper {
    /// The wrapped streaming learner (serves as head 0 for multiclass).
    inner: Box<dyn StreamingLearner>,
    /// Classification mode.
    mode: ClassificationMode,
    /// Additional RLS heads for multiclass (heads 1..K-1). Empty for binary.
    extra_heads: Vec<RecursiveLeastSquares>,
    /// Total samples seen (tracked independently for correct reset).
    samples_seen: u64,
}

impl ClassificationWrapper {
    /// Create a binary classification wrapper around any streaming learner.
    ///
    /// User-supplied {0, 1} targets are mapped to bipolar {-1, +1} internally
    /// for better MSE-based discrimination. At prediction time, the raw output
    /// is thresholded at 0.0.
    pub fn binary(model: Box<dyn StreamingLearner>) -> Self {
        Self {
            inner: model,
            mode: ClassificationMode::Binary,
            extra_heads: Vec::new(),
            samples_seen: 0,
        }
    }

    /// Create a multiclass classification wrapper.
    ///
    /// Maintains K independent scalar heads (the inner model as head 0, plus
    /// K-1 additional RLS heads). Each head is trained with bipolar {-1, +1}
    /// targets via one-vs-rest encoding.
    ///
    /// # Panics
    ///
    /// Panics if `n_classes < 2`.
    pub fn multiclass(model: Box<dyn StreamingLearner>, n_classes: usize) -> Self {
        assert!(
            n_classes >= 2,
            "multiclass requires n_classes >= 2, got {n_classes}"
        );
        let extra_heads = (0..n_classes - 1)
            .map(|_| RecursiveLeastSquares::new(0.99))
            .collect();
        Self {
            inner: model,
            mode: ClassificationMode::Multiclass { n_classes },
            extra_heads,
            samples_seen: 0,
        }
    }

    /// The current classification mode.
    pub fn mode(&self) -> ClassificationMode {
        self.mode
    }

    /// Get class probabilities for the given features.
    ///
    /// - **Binary**: returns `[P(class=0), P(class=1)]` via sigmoid on bipolar output.
    /// - **Multiclass**: returns K probabilities via stable softmax on bipolar outputs.
    /// - **Regression**: returns `[raw_prediction]` (no transformation).
    pub fn predict_proba(&self, features: &[f64]) -> Vec<f64> {
        match self.mode {
            ClassificationMode::Regression => {
                vec![self.inner.predict(features)]
            }
            ClassificationMode::Binary => {
                let raw = self.inner.predict(features);
                let p1 = sigmoid(raw);
                vec![1.0 - p1, p1]
            }
            ClassificationMode::Multiclass { n_classes } => {
                let mut logits = Vec::with_capacity(n_classes);
                logits.push(self.inner.predict(features));
                for head in &self.extra_heads {
                    logits.push(head.predict(features));
                }
                stable_softmax(&logits)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

impl StreamingLearner for ClassificationWrapper {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        self.samples_seen += 1;

        match self.mode {
            ClassificationMode::Regression => {
                self.inner.train_one(features, target, weight);
            }
            ClassificationMode::Binary => {
                // Map {0, 1} → {-1, +1} bipolar targets for better MSE
                // discrimination (standard in ESN/reservoir literature).
                let bipolar = if target > 0.5 { 1.0 } else { -1.0 };
                self.inner.train_one(features, bipolar, weight);
            }
            ClassificationMode::Multiclass { n_classes } => {
                let class_idx = target as usize;
                // Head 0 (inner model): bipolar +1 if class == 0, else -1
                let target_0 = if class_idx == 0 { 1.0 } else { -1.0 };
                self.inner.train_one(features, target_0, weight);
                // Heads 1..K-1 (extra_heads[k-1] corresponds to class k)
                for (k_minus_1, head) in self.extra_heads.iter_mut().enumerate() {
                    let class_k = k_minus_1 + 1;
                    let target_k = if class_idx == class_k { 1.0 } else { -1.0 };
                    head.train_one(features, target_k, weight);
                }
                // Validate class index range (debug only)
                debug_assert!(
                    class_idx < n_classes,
                    "class index {} out of range for {} classes",
                    class_idx,
                    n_classes,
                );
            }
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        match self.mode {
            ClassificationMode::Regression => self.inner.predict(features),
            ClassificationMode::Binary => {
                let raw = self.inner.predict(features);
                // Threshold at 0.0 (bipolar decision boundary)
                if raw >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ClassificationMode::Multiclass { n_classes } => {
                let mut logits = Vec::with_capacity(n_classes);
                logits.push(self.inner.predict(features));
                for head in &self.extra_heads {
                    logits.push(head.predict(features));
                }
                let proba = stable_softmax(&logits);
                // Return argmax class index as f64
                proba
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as f64)
                    .unwrap_or(0.0)
            }
        }
    }

    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        self.inner.reset();
        for head in &mut self.extra_heads {
            head.reset();
        }
        self.samples_seen = 0;
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

impl core::fmt::Debug for ClassificationWrapper {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClassificationWrapper")
            .field("mode", &self.mode)
            .field("samples_seen", &self.samples_seen)
            .field("n_extra_heads", &self.extra_heads.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learners::rls::RecursiveLeastSquares;

    // -----------------------------------------------------------------------
    // Unit tests: sigmoid and softmax helpers
    // -----------------------------------------------------------------------

    #[test]
    fn sigmoid_at_zero_is_half() {
        let result = sigmoid(0.0);
        assert!(
            (result - 0.5).abs() < 1e-12,
            "sigmoid(0) should be 0.5, got {result}"
        );
    }

    #[test]
    fn sigmoid_extreme_values_are_finite() {
        let p_high = sigmoid(1000.0);
        let p_low = sigmoid(-1000.0);
        assert!(p_high.is_finite(), "sigmoid(1000) should be finite");
        assert!(p_low.is_finite(), "sigmoid(-1000) should be finite");
        assert!(
            (p_high - 1.0).abs() < 1e-10,
            "sigmoid(1000) should be ~1.0, got {p_high}"
        );
        assert!(
            p_low.abs() < 1e-10,
            "sigmoid(-1000) should be ~0.0, got {p_low}"
        );
    }

    #[test]
    fn softmax_uniform_logits_are_equal() {
        let logits = vec![1.0, 1.0, 1.0];
        let proba = stable_softmax(&logits);
        assert_eq!(proba.len(), 3, "softmax output should have 3 elements");
        for p in &proba {
            assert!(
                (p - 1.0 / 3.0).abs() < 1e-10,
                "uniform logits should give equal probabilities, got {p}"
            );
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![2.0, 1.0, 0.1, -1.0];
        let proba = stable_softmax(&logits);
        let sum: f64 = proba.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax probabilities should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn softmax_extreme_logits_are_stable() {
        let logits = vec![1000.0, 0.0, -1000.0];
        let proba = stable_softmax(&logits);
        assert!(
            proba.iter().all(|p| p.is_finite()),
            "softmax should be finite for extreme logits"
        );
        assert!(
            (proba[0] - 1.0).abs() < 1e-10,
            "dominant logit should have probability ~1.0, got {}",
            proba[0]
        );
    }

    // -----------------------------------------------------------------------
    // Classification mode enum
    // -----------------------------------------------------------------------

    #[test]
    fn classification_mode_equality() {
        assert_eq!(ClassificationMode::Binary, ClassificationMode::Binary);
        assert_eq!(
            ClassificationMode::Multiclass { n_classes: 3 },
            ClassificationMode::Multiclass { n_classes: 3 }
        );
        assert_ne!(ClassificationMode::Binary, ClassificationMode::Regression);
    }

    // -----------------------------------------------------------------------
    // Binary classification
    // -----------------------------------------------------------------------

    #[test]
    fn binary_wrapper_returns_zero_or_one() {
        let model = RecursiveLeastSquares::new(0.99);
        let mut clf = ClassificationWrapper::binary(Box::new(model));

        // Train on simple linearly separable data
        for i in 0..100 {
            let x = i as f64 * 0.1;
            let label = if x > 5.0 { 1.0 } else { 0.0 };
            clf.train(&[x], label);
        }

        let pred = clf.predict(&[8.0]);
        assert!(
            pred == 0.0 || pred == 1.0,
            "binary predict should return 0.0 or 1.0, got {pred}"
        );
    }

    #[test]
    fn binary_wrapper_predict_proba_returns_two_classes() {
        let model = RecursiveLeastSquares::new(0.99);
        let clf = ClassificationWrapper::binary(Box::new(model));
        let proba = clf.predict_proba(&[1.0, 2.0]);
        assert_eq!(
            proba.len(),
            2,
            "binary predict_proba should return 2 probabilities"
        );
        let sum: f64 = proba.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "binary probabilities should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn binary_wrapper_learns_sine_classification() {
        // Classify sin(x) > 0 as class 1, else class 0
        let model = RecursiveLeastSquares::new(0.998);
        let mut clf = ClassificationWrapper::binary(Box::new(model));

        // Training phase
        for i in 0..500 {
            let x = (i as f64) * 0.05;
            let label = if x.sin() > 0.0 { 1.0 } else { 0.0 };
            clf.train(&[x.sin(), x.cos()], label);
        }

        // Test phase: check accuracy on known points
        let mut correct = 0;
        let test_points = 50;
        for i in 0..test_points {
            let x = (i as f64) * 0.1 + 0.05; // offset to avoid boundaries
            let expected = if x.sin() > 0.0 { 1.0 } else { 0.0 };
            let pred = clf.predict(&[x.sin(), x.cos()]);
            if (pred - expected).abs() < 1e-10 {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / test_points as f64;
        assert!(
            accuracy > 0.7,
            "binary sine classification accuracy should be > 70%, got {:.1}%",
            accuracy * 100.0
        );
    }

    // -----------------------------------------------------------------------
    // Multiclass classification
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "n_classes >= 2")]
    fn multiclass_panics_on_fewer_than_two_classes() {
        let model = RecursiveLeastSquares::new(0.99);
        let _ = ClassificationWrapper::multiclass(Box::new(model), 1);
    }

    #[test]
    fn multiclass_wrapper_returns_valid_class_index() {
        let model = RecursiveLeastSquares::new(0.99);
        let mut clf = ClassificationWrapper::multiclass(Box::new(model), 3);

        for i in 0..60 {
            let class = (i % 3) as f64;
            let x0 = if i % 3 == 0 { 1.0 } else { 0.0 };
            let x1 = if i % 3 == 1 { 1.0 } else { 0.0 };
            let x2 = if i % 3 == 2 { 1.0 } else { 0.0 };
            clf.train(&[x0, x1, x2], class);
        }

        let pred = clf.predict(&[1.0, 0.0, 0.0]);
        assert!(
            (0.0..3.0).contains(&pred),
            "multiclass predict should return class index in [0, 3), got {pred}"
        );
        assert!(
            (pred - pred.round()).abs() < 1e-10,
            "multiclass predict should return an integer class index, got {pred}"
        );
    }

    #[test]
    fn multiclass_predict_proba_returns_k_probabilities() {
        let model = RecursiveLeastSquares::new(0.99);
        let clf = ClassificationWrapper::multiclass(Box::new(model), 4);
        let proba = clf.predict_proba(&[1.0, 2.0, 3.0]);
        assert_eq!(
            proba.len(),
            4,
            "multiclass(4) predict_proba should return 4 probabilities"
        );
        let sum: f64 = proba.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "multiclass probabilities should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn multiclass_learns_three_class_data() {
        // 3 classes, each class has a distinct feature pattern
        let model = RecursiveLeastSquares::new(0.998);
        let mut clf = ClassificationWrapper::multiclass(Box::new(model), 3);

        // Training phase: class 0 = high x0, class 1 = high x1, class 2 = high x2
        for _ in 0..200 {
            clf.train(&[1.0, 0.0, 0.0], 0.0);
            clf.train(&[0.0, 1.0, 0.0], 1.0);
            clf.train(&[0.0, 0.0, 1.0], 2.0);
        }

        // Test: each pure feature vector should predict its class
        let pred_0 = clf.predict(&[1.0, 0.0, 0.0]);
        let pred_1 = clf.predict(&[0.0, 1.0, 0.0]);
        let pred_2 = clf.predict(&[0.0, 0.0, 1.0]);

        assert!(
            (pred_0 - 0.0).abs() < 1e-10,
            "pure class 0 features should predict class 0, got {pred_0}"
        );
        assert!(
            (pred_1 - 1.0).abs() < 1e-10,
            "pure class 1 features should predict class 1, got {pred_1}"
        );
        assert!(
            (pred_2 - 2.0).abs() < 1e-10,
            "pure class 2 features should predict class 2, got {pred_2}"
        );
    }

    // -----------------------------------------------------------------------
    // Wrapper lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn wrapper_tracks_samples_seen() {
        let model = RecursiveLeastSquares::new(0.99);
        let mut clf = ClassificationWrapper::binary(Box::new(model));
        assert_eq!(
            clf.n_samples_seen(),
            0,
            "fresh wrapper should have 0 samples"
        );
        clf.train(&[1.0], 1.0);
        clf.train(&[2.0], 0.0);
        assert_eq!(clf.n_samples_seen(), 2, "wrapper should track samples seen");
    }

    #[test]
    fn wrapper_reset_clears_state() {
        let model = RecursiveLeastSquares::new(0.99);
        let mut clf = ClassificationWrapper::binary(Box::new(model));
        clf.train(&[1.0], 1.0);
        clf.train(&[2.0], 0.0);
        clf.reset();
        assert_eq!(
            clf.n_samples_seen(),
            0,
            "samples_seen should be 0 after reset"
        );
    }

    #[test]
    fn multiclass_reset_clears_all_heads() {
        let model = RecursiveLeastSquares::new(0.99);
        let mut clf = ClassificationWrapper::multiclass(Box::new(model), 3);
        for i in 0..30 {
            clf.train(&[1.0, 0.0], (i % 3) as f64);
        }
        assert_eq!(clf.n_samples_seen(), 30);
        clf.reset();
        assert_eq!(clf.n_samples_seen(), 0, "reset should clear all state");
        // After reset, all heads should predict 0 (uninitialized)
        let proba = clf.predict_proba(&[1.0, 0.0]);
        assert_eq!(
            proba.len(),
            3,
            "predict_proba should still return 3 classes after reset"
        );
    }

    #[test]
    fn wrapper_mode_accessor() {
        let model = RecursiveLeastSquares::new(0.99);
        let clf = ClassificationWrapper::binary(Box::new(model));
        assert_eq!(clf.mode(), ClassificationMode::Binary);

        let model2 = RecursiveLeastSquares::new(0.99);
        let clf2 = ClassificationWrapper::multiclass(Box::new(model2), 5);
        assert_eq!(clf2.mode(), ClassificationMode::Multiclass { n_classes: 5 });
    }

    #[test]
    fn wrapper_debug_format() {
        let model = RecursiveLeastSquares::new(0.99);
        let clf = ClassificationWrapper::binary(Box::new(model));
        let debug = format!("{:?}", clf);
        assert!(
            debug.contains("ClassificationWrapper"),
            "debug output should contain struct name, got: {debug}"
        );
        assert!(
            debug.contains("Binary"),
            "debug output should contain mode, got: {debug}"
        );
    }

    // -----------------------------------------------------------------------
    // Regression passthrough
    // -----------------------------------------------------------------------

    #[test]
    fn regression_mode_is_passthrough() {
        // Construct a regression-mode wrapper manually to verify passthrough
        let model = RecursiveLeastSquares::new(0.99);
        let mut clf = ClassificationWrapper {
            inner: Box::new(model),
            mode: ClassificationMode::Regression,
            extra_heads: Vec::new(),
            samples_seen: 0,
        };
        // Train on y = 2*x
        for i in 0..100 {
            let x = i as f64 * 0.1;
            clf.train(&[x], 2.0 * x);
        }
        let pred = clf.predict(&[5.0]);
        assert!(
            (pred - 10.0).abs() < 0.5,
            "regression passthrough should approximate y=2x, got {pred}"
        );
    }
}
