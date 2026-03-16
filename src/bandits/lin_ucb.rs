//! LinUCB — contextual bandit with linear reward models (Li et al., 2010).

use super::ContextualBandit;

/// LinUCB contextual bandit with per-arm linear reward models.
///
/// Each arm maintains a ridge regression model that estimates expected reward
/// as a linear function of context features. Arm selection maximizes an upper
/// confidence bound: `theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)`, where
/// `theta_a` is the estimated parameter vector and `alpha` controls
/// exploration.
///
/// The inverse covariance matrices are maintained incrementally via the
/// Sherman-Morrison formula, avoiding O(d^3) matrix inversion on each update.
///
/// # Reference
///
/// Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A
/// contextual-bandit approach to personalized news article recommendation.
/// *Proceedings of the 19th International Conference on World Wide Web*.
///
/// # Example
///
/// ```
/// use irithyll::bandits::{ContextualBandit, LinUCB};
///
/// let mut bandit = LinUCB::new(2, 3, 1.0); // 2 arms, 3 features
/// let ctx = [1.0, 0.0, 0.5];
/// let arm = bandit.select_arm(&ctx);
/// bandit.update(arm, &ctx, 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct LinUCB {
    n_arms: usize,
    n_features: usize,
    alpha: f64,
    lambda: f64,
    /// Per-arm d*d inverse matrices, each stored flat in row-major order.
    a_inv: Vec<Vec<f64>>,
    /// Per-arm d-dimensional reward accumulator vectors.
    b: Vec<Vec<f64>>,
    /// Per-arm pull counts.
    counts: Vec<u64>,
    total_pulls: u64,
}

impl LinUCB {
    /// Create a LinUCB bandit with default regularization `lambda = 1.0`.
    ///
    /// # Panics
    ///
    /// Panics if `n_arms == 0` or `n_features == 0`.
    pub fn new(n_arms: usize, n_features: usize, alpha: f64) -> Self {
        Self::with_lambda(n_arms, n_features, alpha, 1.0)
    }

    /// Create a LinUCB bandit with custom regularization parameter `lambda`.
    ///
    /// The initial inverse covariance matrix for each arm is `(1/lambda) * I_d`.
    ///
    /// # Panics
    ///
    /// Panics if `n_arms == 0`, `n_features == 0`, or `lambda <= 0`.
    pub fn with_lambda(n_arms: usize, n_features: usize, alpha: f64, lambda: f64) -> Self {
        assert!(n_arms > 0, "must have at least one arm");
        assert!(n_features > 0, "must have at least one feature");
        assert!(lambda > 0.0, "lambda must be positive, got {}", lambda);

        let inv_lambda = 1.0 / lambda;
        let a_inv = (0..n_arms)
            .map(|_| {
                let mut mat = vec![0.0; n_features * n_features];
                for i in 0..n_features {
                    mat[i * n_features + i] = inv_lambda;
                }
                mat
            })
            .collect();

        let b = vec![vec![0.0; n_features]; n_arms];
        let counts = vec![0u64; n_arms];

        Self {
            n_arms,
            n_features,
            alpha,
            lambda,
            a_inv,
            b,
            counts,
            total_pulls: 0,
        }
    }

    /// Compute the estimated parameter vector `theta_a = A_a^{-1} b_a` for
    /// the given arm.
    pub fn arm_theta(&self, arm: usize) -> Vec<f64> {
        assert!(arm < self.n_arms, "arm index {} out of bounds", arm);
        mat_vec_mul(&self.a_inv[arm], &self.b[arm], self.n_features)
    }

    /// Compute the UCB score for a given arm and context.
    ///
    /// Returns `theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)`.
    pub fn arm_ucb(&self, arm: usize, context: &[f64]) -> f64 {
        assert!(arm < self.n_arms, "arm index {} out of bounds", arm);
        assert_eq!(
            context.len(),
            self.n_features,
            "context length {} does not match n_features {}",
            context.len(),
            self.n_features
        );

        let theta = self.arm_theta(arm);
        let pred = dot(&theta, context);

        // x^T A^{-1} x
        let a_inv_x = mat_vec_mul(&self.a_inv[arm], context, self.n_features);
        let var = dot(context, &a_inv_x);

        // Clamp variance to zero in case of floating-point negativity.
        let exploration = self.alpha * var.max(0.0).sqrt();

        pred + exploration
    }
}

impl ContextualBandit for LinUCB {
    fn select_arm(&mut self, context: &[f64]) -> usize {
        assert_eq!(
            context.len(),
            self.n_features,
            "context length {} does not match n_features {}",
            context.len(),
            self.n_features
        );

        let mut best_arm = 0;
        let mut best_score = f64::NEG_INFINITY;

        for a in 0..self.n_arms {
            let score = if self.counts[a] == 0 {
                // Unvisited arms get infinite UCB to ensure initial exploration.
                f64::INFINITY
            } else {
                self.arm_ucb(a, context)
            };

            if score > best_score {
                best_score = score;
                best_arm = a;
            }
        }

        best_arm
    }

    fn update(&mut self, arm: usize, context: &[f64], reward: f64) {
        assert!(arm < self.n_arms, "arm index {} out of bounds", arm);
        assert_eq!(
            context.len(),
            self.n_features,
            "context length {} does not match n_features {}",
            context.len(),
            self.n_features
        );

        // Sherman-Morrison rank-1 downdate of A_inv.
        sherman_morrison_update(&mut self.a_inv[arm], context, self.n_features);

        // b_a += reward * x
        for (bi, &xi) in self.b[arm].iter_mut().zip(context.iter()) {
            *bi += reward * xi;
        }

        self.counts[arm] += 1;
        self.total_pulls += 1;
    }

    fn n_arms(&self) -> usize {
        self.n_arms
    }

    fn n_pulls(&self) -> u64 {
        self.total_pulls
    }

    fn reset(&mut self) {
        let inv_lambda = 1.0 / self.lambda;
        let d = self.n_features;

        for a in 0..self.n_arms {
            self.a_inv[a].fill(0.0);
            for i in 0..d {
                self.a_inv[a][i * d + i] = inv_lambda;
            }
            self.b[a].fill(0.0);
        }

        self.counts.fill(0);
        self.total_pulls = 0;
    }
}

// ---------------------------------------------------------------------------
// Private helper functions
// ---------------------------------------------------------------------------

/// Matrix-vector product for a row-major flat matrix.
///
/// `mat` is `d x d` stored in row-major order, `v` has length `d`.
fn mat_vec_mul(mat: &[f64], v: &[f64], d: usize) -> Vec<f64> {
    debug_assert_eq!(mat.len(), d * d);
    debug_assert_eq!(v.len(), d);

    let mut result = vec![0.0; d];
    for (i, ri) in result.iter_mut().enumerate() {
        let row_start = i * d;
        let mut sum = 0.0;
        for j in 0..d {
            sum += mat[row_start + j] * v[j];
        }
        *ri = sum;
    }
    result
}

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// In-place Sherman-Morrison rank-1 downdate of the inverse matrix.
///
/// Given current `A^{-1}` and an observation vector `x`, computes the
/// inverse of `A + x x^T` as:
///
/// ```text
/// A'^{-1} = A^{-1} - (A^{-1} x)(x^T A^{-1}) / (1 + x^T A^{-1} x)
/// ```
///
/// `a_inv` is a flat `d x d` row-major matrix, modified in place.
fn sherman_morrison_update(a_inv: &mut [f64], x: &[f64], d: usize) {
    debug_assert_eq!(a_inv.len(), d * d);
    debug_assert_eq!(x.len(), d);

    // Compute A^{-1} x
    let a_inv_x = mat_vec_mul(a_inv, x, d);

    // Compute denominator: 1 + x^T A^{-1} x
    let denom = 1.0 + dot(x, &a_inv_x);

    // Update: A^{-1} -= (A^{-1} x)(A^{-1} x)^T / denom
    // Note: x^T A^{-1} = (A^{-1} x)^T when A^{-1} is symmetric.
    let inv_denom = 1.0 / denom;
    for i in 0..d {
        for j in 0..d {
            a_inv[i * d + j] -= a_inv_x[i] * a_inv_x[j] * inv_denom;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_best_arm_with_context() {
        // Arm 0 gets reward 1.0 when feature[0] > 0.5, arm 1 gets reward 1.0
        // otherwise. After training, the bandit should learn to select the
        // correct arm for each context region.
        let mut bandit = LinUCB::new(2, 2, 1.0);

        // Training phase: explicit round-robin to build up model.
        for _ in 0..200 {
            // Context where feature[0] > 0.5 -> arm 0 is best
            let ctx_high = [0.8, 0.2];
            bandit.update(0, &ctx_high, 1.0);
            bandit.update(1, &ctx_high, 0.0);

            // Context where feature[0] <= 0.5 -> arm 1 is best
            let ctx_low = [0.2, 0.8];
            bandit.update(0, &ctx_low, 0.0);
            bandit.update(1, &ctx_low, 1.0);
        }

        // Evaluation: bandit should pick correctly.
        let arm_high = bandit.select_arm(&[0.9, 0.1]);
        assert_eq!(
            arm_high, 0,
            "should select arm 0 for high feature[0] context, got arm {}",
            arm_high
        );

        let arm_low = bandit.select_arm(&[0.1, 0.9]);
        assert_eq!(
            arm_low, 1,
            "should select arm 1 for low feature[0] context, got arm {}",
            arm_low
        );
    }

    #[test]
    fn explores_all_arms_initially() {
        // With 0 pulls, every arm has infinite UCB. The first n_arms calls
        // should each pull a different arm (ties broken by lowest index, so
        // arms 0, 1, 2 in order).
        let n = 3;
        let mut bandit = LinUCB::new(n, 2, 1.0);

        let mut pulled = vec![false; n];
        for _ in 0..n {
            let ctx = [0.5, 0.5];
            let arm = bandit.select_arm(&ctx);
            pulled[arm] = true;
            // Give some reward so the arm is no longer unvisited.
            bandit.update(arm, &ctx, 0.5);
        }

        for (i, &was_pulled) in pulled.iter().enumerate() {
            assert!(
                was_pulled,
                "arm {} should have been explored in the first {} pulls",
                i, n
            );
        }
    }

    #[test]
    fn reset_clears_state() {
        let lambda = 2.0;
        let mut bandit = LinUCB::with_lambda(2, 3, 1.0, lambda);

        // Do some updates.
        bandit.update(0, &[1.0, 0.0, 0.5], 1.0);
        bandit.update(1, &[0.0, 1.0, 0.5], 0.5);
        bandit.update(0, &[0.3, 0.7, 0.1], 0.8);
        assert!(bandit.n_pulls() > 0, "should have pulls before reset");

        bandit.reset();

        assert_eq!(bandit.n_pulls(), 0, "total pulls should be 0 after reset");
        assert!(
            bandit.counts.iter().all(|&c| c == 0),
            "all arm counts should be 0 after reset"
        );

        let d = bandit.n_features;
        let inv_lambda = 1.0 / lambda;
        for a in 0..bandit.n_arms {
            // b vectors should be zeroed.
            assert!(
                bandit.b[a].iter().all(|&v| v == 0.0),
                "arm {} b-vector should be zeroed after reset",
                a
            );

            // A_inv should be (1/lambda) * I.
            for i in 0..d {
                for j in 0..d {
                    let val = bandit.a_inv[a][i * d + j];
                    let expected = if i == j { inv_lambda } else { 0.0 };
                    assert!(
                        (val - expected).abs() < 1e-12,
                        "arm {} A_inv[{}][{}] should be {}, got {} after reset",
                        a,
                        i,
                        j,
                        expected,
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn theta_converges() {
        // After many updates of arm 0 with x=[1, 0] and reward=1.0,
        // theta should converge to approximately [1, 0].
        let mut bandit = LinUCB::new(2, 2, 1.0);

        for _ in 0..500 {
            bandit.update(0, &[1.0, 0.0], 1.0);
        }

        let theta = bandit.arm_theta(0);
        assert!(
            (theta[0] - 1.0).abs() < 0.05,
            "theta[0] should be close to 1.0, got {}",
            theta[0]
        );
        assert!(
            theta[1].abs() < 0.05,
            "theta[1] should be close to 0.0, got {}",
            theta[1]
        );
    }

    #[test]
    fn single_feature_linear_reward() {
        // 2 arms, 2 features (value + bias) so the model can represent affine
        // reward functions.  Arm 0 reward = x, arm 1 reward = 1 - x.
        // Should learn arm 0 for large x, arm 1 for small x.
        let mut bandit = LinUCB::new(2, 2, 0.5);

        // Training: sweep feature values with a bias term.
        for _ in 0..300 {
            for &x in &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
                let ctx = [x, 1.0]; // [value, bias]
                bandit.update(0, &ctx, x);
                bandit.update(1, &ctx, 1.0 - x);
            }
        }

        // For large x, arm 0 should be preferred.
        let arm_large = bandit.select_arm(&[0.9, 1.0]);
        assert_eq!(
            arm_large, 0,
            "should select arm 0 for x=0.9, got arm {}",
            arm_large
        );

        // For small x, arm 1 should be preferred.
        let arm_small = bandit.select_arm(&[0.1, 1.0]);
        assert_eq!(
            arm_small, 1,
            "should select arm 1 for x=0.1, got arm {}",
            arm_small
        );
    }

    #[test]
    fn ucb_score_finite() {
        let mut bandit = LinUCB::new(3, 4, 2.0);

        // Pull each arm once so counts > 0.
        for a in 0..3 {
            let ctx = vec![0.25; 4];
            bandit.update(a, &ctx, 0.5);
        }

        // UCB scores should all be finite.
        let ctx = [0.5, 0.3, 0.2, 0.1];
        for a in 0..3 {
            let score = bandit.arm_ucb(a, &ctx);
            assert!(
                score.is_finite(),
                "UCB score for arm {} should be finite, got {}",
                a,
                score
            );
        }
    }

    #[test]
    fn sherman_morrison_preserves_symmetry() {
        // After multiple updates, A_inv should remain approximately symmetric.
        let mut bandit = LinUCB::new(1, 4, 1.0);

        let contexts: &[&[f64]] = &[
            &[1.0, 0.0, 0.5, 0.3],
            &[0.0, 1.0, 0.2, 0.8],
            &[0.3, 0.7, 0.1, 0.4],
            &[0.5, 0.5, 0.5, 0.5],
            &[0.9, 0.1, 0.0, 0.6],
            &[0.2, 0.3, 0.8, 0.1],
        ];

        for ctx in contexts {
            bandit.update(0, ctx, 0.5);
        }

        let d = bandit.n_features;
        let mat = &bandit.a_inv[0];
        for i in 0..d {
            for j in (i + 1)..d {
                let a_ij = mat[i * d + j];
                let a_ji = mat[j * d + i];
                let diff = (a_ij - a_ji).abs();
                assert!(
                    diff < 1e-10,
                    "A_inv should be symmetric: A_inv[{}][{}]={} vs A_inv[{}][{}]={}, diff={}",
                    i,
                    j,
                    a_ij,
                    j,
                    i,
                    a_ji,
                    diff
                );
            }
        }
    }
}
