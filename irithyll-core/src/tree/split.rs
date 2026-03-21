//! Split criterion traits and XGBoost gain implementation.
//!
//! This module defines how candidate splits are evaluated from histogram
//! gradient/hessian sums. The primary implementation is [`XGBoostGain`],
//! which uses the exact gain formula from Chen & Guestrin (2016).

#[cfg(test)]
use alloc::vec::Vec;

/// Result of evaluating potential splits across histogram bins.
#[derive(Debug, Clone, Copy)]
pub struct SplitCandidate {
    /// Which bin index to split at (samples with bin <= index go left).
    pub bin_idx: usize,
    /// The gain from this split.
    pub gain: f64,
    /// Gradient sum for the left child.
    pub left_grad: f64,
    /// Hessian sum for the left child.
    pub left_hess: f64,
    /// Gradient sum for the right child.
    pub right_grad: f64,
    /// Hessian sum for the right child.
    pub right_hess: f64,
}

/// Evaluates split quality from histogram gradient/hessian sums.
pub trait SplitCriterion: Send + Sync + 'static {
    /// Evaluate all possible split points across histogram bins.
    /// Returns the best split candidate, or `None` if no valid split exists.
    ///
    /// # Arguments
    ///
    /// * `grad_sums` - gradient sums per bin
    /// * `hess_sums` - hessian sums per bin
    /// * `total_grad` - total gradient across all bins
    /// * `total_hess` - total hessian across all bins
    /// * `gamma` - minimum split gain threshold
    /// * `lambda` - L2 regularization on leaf weights
    fn evaluate(
        &self,
        grad_sums: &[f64],
        hess_sums: &[f64],
        total_grad: f64,
        total_hess: f64,
        gamma: f64,
        lambda: f64,
    ) -> Option<SplitCandidate>;
}

/// XGBoost-style split gain criterion.
///
/// Gain = 0.5 * (G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)) - gamma
///
/// where G_L, H_L are left child gradient/hessian sums,
/// G_R, H_R are right child sums, G, H are total sums.
///
/// Reference: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016.
#[derive(Debug, Clone, Copy)]
pub struct XGBoostGain {
    /// Minimum hessian sum for a child to be valid.
    pub min_child_weight: f64,
}

impl Default for XGBoostGain {
    fn default() -> Self {
        Self {
            min_child_weight: 1.0,
        }
    }
}

impl XGBoostGain {
    /// Create a new `XGBoostGain` with the given minimum child weight.
    pub fn new(min_child_weight: f64) -> Self {
        Self { min_child_weight }
    }
}

impl SplitCriterion for XGBoostGain {
    fn evaluate(
        &self,
        grad_sums: &[f64],
        hess_sums: &[f64],
        total_grad: f64,
        total_hess: f64,
        gamma: f64,
        lambda: f64,
    ) -> Option<SplitCandidate> {
        let n_bins = grad_sums.len();
        debug_assert_eq!(
            n_bins,
            hess_sums.len(),
            "grad_sums and hess_sums must have the same length"
        );

        // Need at least 2 bins to form a left and right child.
        if n_bins < 2 {
            return None;
        }

        // Precompute the parent term, which is constant across all candidate splits.
        let parent_score = total_grad * total_grad / (total_hess + lambda);

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_bin = 0usize;
        let mut best_left_grad = 0.0;
        let mut best_left_hess = 0.0;
        let mut best_right_grad = 0.0;
        let mut best_right_hess = 0.0;

        // Running prefix sums for the left child.
        let mut left_grad = 0.0;
        let mut left_hess = 0.0;

        // Iterate through bins 0..n_bins-1. The split point at bin_idx=i means
        // bins [0..=i] go left and bins [i+1..n_bins-1] go right.
        // We stop at n_bins-2 so the right child always has at least 1 bin.
        for i in 0..n_bins - 1 {
            left_grad += grad_sums[i];
            left_hess += hess_sums[i];

            let right_grad = total_grad - left_grad;
            let right_hess = total_hess - left_hess;

            // Enforce minimum child weight on both children.
            if left_hess < self.min_child_weight || right_hess < self.min_child_weight {
                continue;
            }

            let left_score = left_grad * left_grad / (left_hess + lambda);
            let right_score = right_grad * right_grad / (right_hess + lambda);
            let gain = 0.5 * (left_score + right_score - parent_score) - gamma;

            if gain > best_gain {
                best_gain = gain;
                best_bin = i;
                best_left_grad = left_grad;
                best_left_hess = left_hess;
                best_right_grad = right_grad;
                best_right_hess = right_hess;
            }
        }

        // Only return a split if the best gain is strictly positive.
        if best_gain > 0.0 {
            Some(SplitCandidate {
                bin_idx: best_bin,
                gain: best_gain,
                left_grad: best_left_grad,
                left_hess: best_left_hess,
                right_grad: best_right_grad,
                right_hess: best_right_hess,
            })
        } else {
            None
        }
    }
}

/// Compute the optimal leaf weight: -G / (H + lambda).
///
/// This is the closed-form solution for the leaf value that minimizes
/// the regularized second-order Taylor expansion of the loss.
#[inline]
pub fn leaf_weight(grad_sum: f64, hess_sum: f64, lambda: f64) -> f64 {
    -grad_sum / (hess_sum + lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    /// Two clearly separable groups: bins [0,1] have negative gradients,
    /// bins [2,3] have positive gradients. The best split should be at bin 1.
    #[test]
    fn perfect_split() {
        let criterion = XGBoostGain::new(0.0);

        // Left group: negative gradient, right group: positive gradient.
        let grad_sums = [
            -5.0, -5.0, // left: total grad = -10
            5.0, 5.0, // right: total grad = +10
        ];
        let hess_sums = [
            2.0, 2.0, // left: total hess = 4
            2.0, 2.0, // right: total hess = 4
        ];
        let total_grad: f64 = grad_sums.iter().sum(); // 0.0
        let total_hess: f64 = hess_sums.iter().sum(); // 8.0

        let lambda = 1.0;
        let gamma = 0.0;

        let result = criterion
            .evaluate(
                &grad_sums, &hess_sums, total_grad, total_hess, gamma, lambda,
            )
            .expect("should find a valid split");

        // Best split at bin index 1 (bins 0,1 go left; bins 2,3 go right).
        assert_eq!(result.bin_idx, 1);

        // Verify gradient/hessian sums.
        assert!((result.left_grad - (-10.0)).abs() < EPSILON);
        assert!((result.left_hess - 4.0).abs() < EPSILON);
        assert!((result.right_grad - 10.0).abs() < EPSILON);
        assert!((result.right_hess - 4.0).abs() < EPSILON);

        // Verify gain calculation manually:
        // parent_score = 0^2 / (8+1) = 0
        // left_score = (-10)^2 / (4+1) = 100/5 = 20
        // right_score = (10)^2 / (4+1) = 100/5 = 20
        // gain = 0.5 * (20 + 20 - 0) - 0 = 20.0
        assert!((result.gain - 20.0).abs() < EPSILON);
        assert!(result.gain > 0.0);
    }

    /// All data concentrated in a single bin. No valid split possible because
    /// the other side would always be empty.
    #[test]
    fn no_valid_split_single_bin() {
        let criterion = XGBoostGain::new(0.0);

        // Only one bin: impossible to split.
        let grad_sums = [5.0];
        let hess_sums = [3.0];

        let result = criterion.evaluate(&grad_sums, &hess_sums, 5.0, 3.0, 0.0, 1.0);
        assert!(result.is_none());
    }

    /// All data in the first bin, rest empty. Even with multiple bins, the
    /// right child would have zero hessian so no split meets min_child_weight=0
    /// unless there is actual data on both sides. Here we use min_child_weight=0
    /// but right hessian is 0, so only the right side violates the check
    /// when min_child_weight > 0. With min_child_weight=0, gain should still
    /// be zero or negative since one side has no information.
    #[test]
    fn no_valid_split_all_data_one_side() {
        let criterion = XGBoostGain::new(1.0);

        let grad_sums = [5.0, 0.0, 0.0];
        let hess_sums = [3.0, 0.0, 0.0];

        let result = criterion.evaluate(&grad_sums, &hess_sums, 5.0, 3.0, 0.0, 1.0);
        // Right hessian = 0 at every candidate, which is below min_child_weight=1.0.
        assert!(result.is_none());
    }

    /// min_child_weight enforcement: a split that would otherwise be valid
    /// gets rejected because one child has insufficient hessian.
    #[test]
    fn min_child_weight_enforcement() {
        // Two bins, each with reasonable gradient signal but one has low hessian.
        let grad_sums = [10.0, 10.0];
        let hess_sums = [0.5, 5.0]; // left hess = 0.5, right hess = 5.0

        let total_grad = 20.0;
        let total_hess = 5.5;

        // With min_child_weight=1.0, left_hess=0.5 is too small.
        let strict = XGBoostGain::new(1.0);
        let result = strict.evaluate(&grad_sums, &hess_sums, total_grad, total_hess, 0.0, 1.0);
        assert!(
            result.is_none(),
            "split should be rejected: left hess 0.5 < min_child_weight 1.0"
        );

        // With min_child_weight=0.1, the split should be accepted.
        let lenient = XGBoostGain::new(0.1);
        let result = lenient.evaluate(&grad_sums, &hess_sums, total_grad, total_hess, 0.0, 1.0);
        assert!(
            result.is_some(),
            "split should be accepted with lower min_child_weight"
        );
    }

    /// Verify the leaf weight formula: w = -G / (H + lambda).
    #[test]
    fn leaf_weight_computation() {
        // Basic case.
        assert!((leaf_weight(10.0, 5.0, 1.0) - (-10.0 / 6.0)).abs() < EPSILON);

        // Zero gradient -> zero weight.
        assert!((leaf_weight(0.0, 5.0, 1.0) - 0.0).abs() < EPSILON);

        // Negative gradient -> positive weight.
        assert!((leaf_weight(-3.0, 2.0, 0.5) - (3.0 / 2.5)).abs() < EPSILON);

        // Lambda=0 edge case.
        assert!((leaf_weight(4.0, 2.0, 0.0) - (-2.0)).abs() < EPSILON);
    }

    /// Symmetry test: if we negate all gradients on a symmetric histogram,
    /// the gain should remain the same (gain depends on G^2 terms).
    #[test]
    fn gain_symmetry_under_gradient_sign_flip() {
        let criterion = XGBoostGain::new(0.0);
        let lambda = 1.0;
        let gamma = 0.0;

        let grad_sums = [-3.0, -2.0, 2.0, 3.0];
        let hess_sums = [1.0, 1.0, 1.0, 1.0];
        let total_grad: f64 = grad_sums.iter().sum(); // 0.0
        let total_hess: f64 = hess_sums.iter().sum(); // 4.0

        let result_pos = criterion
            .evaluate(
                &grad_sums, &hess_sums, total_grad, total_hess, gamma, lambda,
            )
            .expect("should find split");

        // Negate all gradients.
        let grad_sums_neg: Vec<f64> = grad_sums.iter().map(|g| -g).collect();
        let total_grad_neg: f64 = grad_sums_neg.iter().sum(); // still 0.0

        let result_neg = criterion
            .evaluate(
                &grad_sums_neg,
                &hess_sums,
                total_grad_neg,
                total_hess,
                gamma,
                lambda,
            )
            .expect("should find split with negated gradients");

        // Gain should be identical.
        assert!(
            (result_pos.gain - result_neg.gain).abs() < EPSILON,
            "gain should be invariant under gradient sign flip: {} vs {}",
            result_pos.gain,
            result_neg.gain
        );

        // Both should pick the same bin.
        assert_eq!(result_pos.bin_idx, result_neg.bin_idx);
    }

    /// Gamma threshold: a split that is marginally positive without gamma
    /// should be rejected when gamma is large enough.
    #[test]
    fn gamma_threshold_rejects_weak_split() {
        let criterion = XGBoostGain::new(0.0);
        let lambda = 1.0;

        let grad_sums = [-1.0, 1.0];
        let hess_sums = [5.0, 5.0];
        let total_grad = 0.0;
        let total_hess = 10.0;

        // With gamma=0, should find a split.
        let result =
            criterion.evaluate(&grad_sums, &hess_sums, total_grad, total_hess, 0.0, lambda);
        assert!(result.is_some(), "should find split with gamma=0");
        let gain_no_gamma = result.unwrap().gain;

        // With gamma larger than the raw gain, should reject.
        let result = criterion.evaluate(
            &grad_sums,
            &hess_sums,
            total_grad,
            total_hess,
            gain_no_gamma + 1.0,
            lambda,
        );
        assert!(
            result.is_none(),
            "split should be rejected when gamma exceeds raw gain"
        );
    }

    /// Lambda regularization: increasing lambda should reduce gain by
    /// penalizing the score terms.
    #[test]
    fn lambda_reduces_gain() {
        let criterion = XGBoostGain::new(0.0);
        let gamma = 0.0;

        let grad_sums = [-5.0, 5.0];
        let hess_sums = [2.0, 2.0];
        let total_grad = 0.0;
        let total_hess = 4.0;

        let result_low = criterion
            .evaluate(&grad_sums, &hess_sums, total_grad, total_hess, gamma, 0.1)
            .expect("should find split with low lambda");

        let result_high = criterion
            .evaluate(&grad_sums, &hess_sums, total_grad, total_hess, gamma, 100.0)
            .expect("should find split with high lambda");

        assert!(
            result_low.gain > result_high.gain,
            "higher lambda should reduce gain: {} vs {}",
            result_low.gain,
            result_high.gain
        );
    }

    /// Empty histogram (no bins) should return None.
    #[test]
    fn empty_histogram() {
        let criterion = XGBoostGain::new(0.0);
        let result = criterion.evaluate(&[], &[], 0.0, 0.0, 0.0, 1.0);
        assert!(result.is_none());
    }

    /// Best split is chosen among multiple valid candidates.
    #[test]
    fn selects_best_among_multiple_candidates() {
        let criterion = XGBoostGain::new(0.0);
        let lambda = 1.0;
        let gamma = 0.0;

        // Design: gradients are arranged so the best split is at bin 2.
        // Bins: [-1, -1, -8, 5, 5] (total_grad = 0)
        // Best split at bin 2: left_grad=-10, right_grad=10.
        let grad_sums = [-1.0, -1.0, -8.0, 5.0, 5.0];
        let hess_sums = [1.0, 1.0, 1.0, 1.0, 1.0];
        let total_grad: f64 = grad_sums.iter().sum();
        let total_hess: f64 = hess_sums.iter().sum();

        let result = criterion
            .evaluate(
                &grad_sums, &hess_sums, total_grad, total_hess, gamma, lambda,
            )
            .expect("should find a valid split");

        assert_eq!(result.bin_idx, 2, "best split should be at bin 2");
    }
}
