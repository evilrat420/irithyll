//! Kernel Recursive Least Squares (KRLS) with ALD sparsification.
//!
//! Implements the algorithm from Engel, Mannor & Meir (2004):
//! *"The Kernel Recursive Least-Squares Algorithm"*, IEEE Trans. Signal Processing.
//!
//! KRLS extends Recursive Least Squares to non-linear regression by operating
//! in a reproducing kernel Hilbert space (RKHS). Instead of maintaining a fixed
//! d-dimensional weight vector, KRLS builds a *dictionary* of support vectors
//! and learns dual weights in kernel space.
//!
//! # Sparsification
//!
//! Not every sample is added to the dictionary. The Approximate Linear
//! Dependency (ALD) criterion tests whether a new sample can be well-approximated
//! by existing dictionary entries in RKHS. If the approximation error is below
//! `ald_threshold`, the sample is *absorbed* (weights updated, dictionary unchanged).
//! Otherwise it is *added* to the dictionary.
//!
//! A `budget` parameter caps dictionary size. Once the budget is reached, no
//! new entries are added -- subsequent samples use weight-only updates.
//!
//! # Complexity
//!
//! - Training: O(N²) per sample, where N is dictionary size (N ≤ budget)
//! - Prediction: O(N·d) per sample
//! - Memory: O(N² + N·d)
//!
//! # Example
//!
//! ```
//! use irithyll::learners::krls::{KRLS, RBFKernel};
//! use irithyll::learner::StreamingLearner;
//!
//! // Learn y = sin(x) with an RBF kernel, budget=50
//! let mut krls = KRLS::new(Box::new(RBFKernel::new(1.0)), 50, 1e-4);
//!
//! for i in 0..500 {
//!     let x = i as f64 * 0.02;
//!     krls.train(&[x], x.sin());
//! }
//!
//! let pred = krls.predict(&[1.0]);
//! assert!((pred - 1.0_f64.sin()).abs() < 0.3);
//! ```

use std::fmt;

use crate::learner::StreamingLearner;

// ===========================================================================
// Kernel trait and implementations
// ===========================================================================

/// Object-safe kernel function for KRLS and other kernel methods.
///
/// A kernel computes an inner product in a (possibly infinite-dimensional)
/// feature space without explicitly computing the feature map.
pub trait Kernel: Send + Sync {
    /// Evaluate the kernel function k(a, b).
    fn eval(&self, a: &[f64], b: &[f64]) -> f64;

    /// Human-readable kernel name (for Debug output).
    fn name(&self) -> &str {
        "Kernel"
    }
}

/// Radial Basis Function (Gaussian) kernel.
///
/// k(a, b) = exp(-gamma * ||a - b||²)
///
/// The most common kernel for non-linear regression and classification.
/// `gamma` controls the "reach" of each support vector: larger gamma means
/// each point has influence over a smaller region.
#[derive(Clone, Debug)]
pub struct RBFKernel {
    /// Width parameter. Larger = narrower influence.
    pub gamma: f64,
}

impl RBFKernel {
    /// Create a new RBF kernel with the given gamma.
    pub fn new(gamma: f64) -> Self {
        assert!(gamma > 0.0, "RBF gamma must be > 0, got {gamma}");
        Self { gamma }
    }
}

impl Kernel for RBFKernel {
    #[inline]
    fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let sq_dist: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum();
        (-self.gamma * sq_dist).exp()
    }

    fn name(&self) -> &str {
        "RBF"
    }
}

/// Polynomial kernel.
///
/// k(a, b) = (a · b + coef0)^degree
#[derive(Clone, Debug)]
pub struct PolynomialKernel {
    /// Polynomial degree.
    pub degree: usize,
    /// Independent term.
    pub coef0: f64,
}

impl PolynomialKernel {
    /// Create a new polynomial kernel.
    pub fn new(degree: usize, coef0: f64) -> Self {
        assert!(degree >= 1, "polynomial degree must be >= 1, got {degree}");
        Self { degree, coef0 }
    }
}

impl Kernel for PolynomialKernel {
    #[inline]
    fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
        (dot + self.coef0).powi(self.degree as i32)
    }

    fn name(&self) -> &str {
        "Polynomial"
    }
}

/// Linear kernel (identity in input space).
///
/// k(a, b) = a · b
///
/// Equivalent to standard RLS but through the kernel interface. Useful as a
/// baseline for comparing kernel methods against linear models.
#[derive(Clone, Debug, Default)]
pub struct LinearKernel;

impl Kernel for LinearKernel {
    #[inline]
    fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
    }

    fn name(&self) -> &str {
        "Linear"
    }
}

// ===========================================================================
// KRLS
// ===========================================================================

/// Kernel Recursive Least Squares with ALD sparsification.
///
/// Maintains a dictionary of support vectors and learns dual weights in the
/// kernel space defined by the provided [`Kernel`] function.
///
/// # Sparsification (ALD)
///
/// The Approximate Linear Dependency test measures whether a new sample lies
/// within the span of existing dictionary entries (in RKHS). The test computes:
///
/// ```text
/// delta = k(x,x) - k_t^T * P * k_t
/// ```
///
/// where `k_t` is the vector of kernel evaluations between the new sample and
/// all dictionary entries, and P is the inverse kernel matrix. If `delta < ald_threshold`,
/// the sample is linearly dependent and only the weights are updated.
///
/// # Budget
///
/// When the dictionary reaches `budget` entries, no new entries are added.
/// Subsequent independent samples fall through to weight-only updates.
///
/// # Example
///
/// ```
/// use irithyll::learners::krls::{KRLS, RBFKernel};
/// use irithyll::learner::StreamingLearner;
///
/// let mut model = KRLS::new(Box::new(RBFKernel::new(0.5)), 100, 1e-4);
/// model.train(&[1.0], 1.0_f64.sin());
/// model.train(&[2.0], 2.0_f64.sin());
/// let pred = model.predict(&[1.5]);
/// assert!(pred.is_finite());
/// ```
pub struct KRLS {
    /// Kernel function.
    kernel: Box<dyn Kernel>,
    /// Dictionary of support vectors.
    dictionary: Vec<Vec<f64>>,
    /// Dual weight vector (alpha). Length = dictionary size.
    weights: Vec<f64>,
    /// Inverse kernel matrix (P), N×N row-major. Length = N*N.
    p_matrix: Vec<f64>,
    /// Maximum dictionary size.
    budget: usize,
    /// ALD threshold (nu). Smaller = more selective (larger dictionary).
    ald_threshold: f64,
    /// Forgetting factor lambda in (0, 1]. 1.0 = no forgetting.
    forgetting_factor: f64,
    /// Total samples trained on.
    samples_seen: u64,
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl KRLS {
    /// Create a new KRLS learner with no forgetting.
    ///
    /// # Arguments
    ///
    /// * `kernel` -- kernel function (e.g., `RBFKernel`, `PolynomialKernel`)
    /// * `budget` -- maximum dictionary size (caps memory usage)
    /// * `ald_threshold` -- ALD sparsification threshold. Typical values: 1e-4 to 1e-2.
    ///   Smaller = more dictionary entries (closer approximation).
    pub fn new(kernel: Box<dyn Kernel>, budget: usize, ald_threshold: f64) -> Self {
        Self::with_forgetting(kernel, budget, ald_threshold, 1.0)
    }

    /// Create a KRLS learner with exponential forgetting.
    ///
    /// `forgetting_factor` in (0, 1]. Values like 0.99 introduce exponential
    /// discounting of older data for non-stationary environments.
    pub fn with_forgetting(
        kernel: Box<dyn Kernel>,
        budget: usize,
        ald_threshold: f64,
        forgetting_factor: f64,
    ) -> Self {
        assert!(budget > 0, "KRLS budget must be > 0, got {budget}");
        assert!(
            ald_threshold > 0.0,
            "ALD threshold must be > 0, got {ald_threshold}"
        );
        assert!(
            (0.0..=1.0).contains(&forgetting_factor),
            "forgetting_factor must be in (0, 1], got {forgetting_factor}"
        );
        Self {
            kernel,
            dictionary: Vec::new(),
            weights: Vec::new(),
            p_matrix: Vec::new(),
            budget,
            ald_threshold,
            forgetting_factor,
            samples_seen: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current dictionary size (number of support vectors).
    #[inline]
    pub fn dict_size(&self) -> usize {
        self.dictionary.len()
    }

    /// Maximum dictionary size.
    #[inline]
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// The ALD threshold.
    #[inline]
    pub fn ald_threshold(&self) -> f64 {
        self.ald_threshold
    }

    /// The forgetting factor.
    #[inline]
    pub fn forgetting_factor(&self) -> f64 {
        self.forgetting_factor
    }

    /// Immutable access to the dictionary (support vectors).
    pub fn dictionary(&self) -> &[Vec<f64>] {
        &self.dictionary
    }

    /// Immutable access to the dual weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Compute kernel vector: [k(x, dict_0), k(x, dict_1), ..., k(x, dict_{N-1})]
    fn kernel_vector(&self, x: &[f64]) -> Vec<f64> {
        self.dictionary
            .iter()
            .map(|di| self.kernel.eval(x, di))
            .collect()
    }

    /// Matrix-vector multiply: P * v, where P is N×N row-major.
    fn p_times_vec(&self, v: &[f64]) -> Vec<f64> {
        let n = self.dictionary.len();
        let mut result = vec![0.0; n];
        for (i, ri) in result.iter_mut().enumerate() {
            let row_start = i * n;
            for (j, &vj) in v.iter().enumerate() {
                *ri += self.p_matrix[row_start + j] * vj;
            }
        }
        result
    }

    /// Dot product of two slices.
    #[inline]
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
    }

    /// Check if the dictionary is at capacity.
    #[inline]
    fn at_budget(&self) -> bool {
        self.dictionary.len() >= self.budget
    }

    /// Add a new sample to the dictionary (ALD-independent).
    fn add_to_dictionary(&mut self, x: Vec<f64>, k_t: &[f64], delta: f64, target: f64) {
        let n = self.dictionary.len();
        let pred = Self::dot(&self.weights, k_t);
        let error = target - pred;

        // Compute a_t = P * k_t (for existing dictionary)
        let a_t = if n > 0 {
            self.p_times_vec(k_t)
        } else {
            Vec::new()
        };

        // Expand P matrix: new dimension is n+1
        let new_n = n + 1;
        let mut new_p = vec![0.0; new_n * new_n];

        let inv_delta = 1.0 / delta;

        // Top-left: P + (1/delta) * a_t * a_t^T
        for i in 0..n {
            for j in 0..n {
                new_p[i * new_n + j] = self.p_matrix[i * n + j] + inv_delta * a_t[i] * a_t[j];
            }
        }

        // Right column and bottom row: -(1/delta) * a_t
        for i in 0..n {
            new_p[i * new_n + n] = -inv_delta * a_t[i];
            new_p[n * new_n + i] = -inv_delta * a_t[i];
        }

        // Bottom-right: 1/delta
        new_p[n * new_n + n] = inv_delta;

        self.p_matrix = new_p;

        // Update weights: existing weights adjusted, new weight added
        // alpha_new = (1/delta) * error
        // alpha_old -= (1/delta) * a_t * error
        for (wi, &ai) in self.weights.iter_mut().zip(a_t.iter()) {
            *wi -= inv_delta * ai * error;
        }
        self.weights.push(inv_delta * error);

        // Add to dictionary
        self.dictionary.push(x);
    }

    /// Update weights without adding to dictionary (ALD-dependent).
    fn update_weights_only(&mut self, k_t: &[f64], target: f64) {
        let n = self.dictionary.len();
        if n == 0 {
            return;
        }

        // a_t = P * k_t
        let a_t = self.p_times_vec(k_t);

        // prediction and error
        let pred = Self::dot(&self.weights, k_t);
        let error = target - pred;

        // Compute gain: q = a_t / (lambda + k_t^T * a_t)
        let denom = self.forgetting_factor + Self::dot(k_t, &a_t);
        let inv_denom = 1.0 / denom;

        // Update weights: alpha += q * error
        for (wi, &ai) in self.weights.iter_mut().zip(a_t.iter()) {
            *wi += ai * inv_denom * error;
        }

        // Update P: P = (P - q * a_t^T) / lambda
        let inv_lambda = 1.0 / self.forgetting_factor;
        for (i, &a_i) in a_t.iter().enumerate() {
            let qi = a_i * inv_denom;
            let row_start = i * n;
            for (j, &a_j) in a_t.iter().enumerate() {
                self.p_matrix[row_start + j] =
                    (self.p_matrix[row_start + j] - qi * a_j) * inv_lambda;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner
// ---------------------------------------------------------------------------

impl StreamingLearner for KRLS {
    fn train_one(&mut self, features: &[f64], target: f64, _weight: f64) {
        self.samples_seen += 1;

        // Kernel evaluation k(x_t, x_t)
        let k_tt = self.kernel.eval(features, features);

        let n = self.dictionary.len();

        if n == 0 {
            // First sample always goes into the dictionary.
            self.dictionary.push(features.to_vec());
            self.weights.push(target / k_tt.max(1e-15));
            self.p_matrix.push(1.0 / k_tt.max(1e-15));
            return;
        }

        // Compute kernel vector k_t = [k(x_t, d_0), ..., k(x_t, d_{n-1})]
        let k_t = self.kernel_vector(features);

        // ALD test: delta = k_tt - k_t^T * P * k_t
        let p_k = self.p_times_vec(&k_t);
        let delta = k_tt - Self::dot(&k_t, &p_k);

        if delta > self.ald_threshold && !self.at_budget() {
            // Sample is linearly independent and budget has room -- expand dictionary.
            self.add_to_dictionary(features.to_vec(), &k_t, delta, target);
        } else {
            // Either dependent (ALD) or budget full -- update weights only.
            self.update_weights_only(&k_t, target);
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        if self.dictionary.is_empty() {
            return 0.0;
        }
        let k_t = self.kernel_vector(features);
        Self::dot(&self.weights, &k_t)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        self.dictionary.clear();
        self.weights.clear();
        self.p_matrix.clear();
        self.samples_seen = 0;
    }

    fn diagnostics_array(&self) -> [f64; 5] {
        let budget = self.budget.max(1) as f64;
        [
            0.0,                                   // residual_alignment
            1.0 - self.forgetting_factor,          // reg_sensitivity
            0.0,                                   // depth_sufficiency
            self.dictionary.len() as f64,          // effective_dof
            self.dictionary.len() as f64 / budget, // uncertainty
        ]
    }

    fn adjust_config(&mut self, lr_multiplier: f64, _lambda_delta: f64) {
        // Scale the forgetting factor toward/away from 1.0.
        // lr_multiplier > 1 => more aggressive forgetting (smaller factor).
        // Clamp to (0, 1].
        self.forgetting_factor = (self.forgetting_factor * lr_multiplier).clamp(1e-6, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl fmt::Debug for KRLS {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KRLS")
            .field("kernel", &self.kernel.name())
            .field("dict_size", &self.dictionary.len())
            .field("budget", &self.budget)
            .field("ald_threshold", &self.ald_threshold)
            .field("forgetting_factor", &self.forgetting_factor)
            .field("samples_seen", &self.samples_seen)
            .finish()
    }
}

// ===========================================================================
// DiagnosticSource impl
// ===========================================================================

impl crate::automl::DiagnosticSource for KRLS {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        let budget = self.budget().max(1) as f64;
        Some(crate::automl::ConfigDiagnostics {
            effective_dof: self.dict_size() as f64,
            regularization_sensitivity: 1.0 - self.forgetting_factor(),
            // Dictionary utilization: 1.0 = full, model capacity maxed out.
            uncertainty: self.dict_size() as f64 / budget,
            ..Default::default()
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::StreamingLearner;

    #[test]
    fn construction_and_initial_state() {
        let krls = KRLS::new(Box::new(RBFKernel::new(1.0)), 50, 1e-4);
        assert_eq!(krls.dict_size(), 0);
        assert_eq!(krls.budget(), 50);
        assert_eq!(krls.n_samples_seen(), 0);
        assert!((krls.predict(&[1.0]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn learn_sine_rbf() {
        // gamma=0.5 gives a wider kernel, better coverage with budget=100
        let mut krls = KRLS::new(Box::new(RBFKernel::new(0.5)), 100, 1e-4);

        for i in 0..500 {
            let x = i as f64 * 0.02;
            krls.train(&[x], x.sin());
        }

        // Test at points well within training range [0, 10].
        let test_points = [1.0, 2.0, 3.0, 5.0];
        let mut max_err = 0.0_f64;
        for &x in &test_points {
            let pred = krls.predict(&[x]);
            let err = (pred - x.sin()).abs();
            max_err = max_err.max(err);
        }
        assert!(
            max_err < 1.0,
            "KRLS should learn sin(x), max error = {}",
            max_err
        );
    }

    #[test]
    fn dictionary_respects_budget() {
        let budget = 20;
        let mut krls = KRLS::new(Box::new(RBFKernel::new(0.5)), budget, 1e-6);

        for i in 0..200 {
            let x = i as f64 * 0.1;
            krls.train(&[x], x.sin());
        }

        assert!(
            krls.dict_size() <= budget,
            "dict_size={} exceeds budget={}",
            krls.dict_size(),
            budget
        );
    }

    #[test]
    fn ald_sparsifies_dictionary() {
        // With a reasonable threshold, dictionary should be much smaller than samples.
        let mut krls = KRLS::new(Box::new(RBFKernel::new(1.0)), 500, 0.01);

        for i in 0..200 {
            let x = i as f64 * 0.05;
            krls.train(&[x], x.sin());
        }

        assert!(
            krls.dict_size() < 200,
            "ALD should sparsify: dict_size={}, samples=200",
            krls.dict_size()
        );
    }

    #[test]
    fn forgetting_adapts_to_shift() {
        let mut krls = KRLS::with_forgetting(Box::new(RBFKernel::new(1.0)), 50, 1e-4, 0.98);

        // Phase 1: y = x
        for i in 0..200 {
            let x = i as f64 * 0.01;
            krls.train(&[x], x);
        }

        // Phase 2: y = -x (distribution shift)
        for i in 0..200 {
            let x = i as f64 * 0.01;
            krls.train(&[x], -x);
        }

        // After shift, prediction at x=1.0 should be closer to -1.0 than 1.0.
        let pred = krls.predict(&[1.0]);
        assert!(
            pred < 0.5,
            "forgetting KRLS should adapt to shift, pred at 1.0 = {}",
            pred
        );
    }

    #[test]
    fn reset_clears_all_state() {
        let mut krls = KRLS::new(Box::new(RBFKernel::new(1.0)), 50, 1e-4);
        krls.train(&[1.0], 1.0);
        krls.train(&[2.0], 4.0);
        assert!(krls.dict_size() > 0);
        assert_eq!(krls.n_samples_seen(), 2);

        krls.reset();
        assert_eq!(krls.dict_size(), 0);
        assert_eq!(krls.n_samples_seen(), 0);
        assert!(krls.weights().is_empty());
    }

    #[test]
    fn trait_object_works() {
        let krls = KRLS::new(Box::new(RBFKernel::new(1.0)), 50, 1e-4);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(krls);

        boxed.train(&[1.0], 2.0);
        boxed.train(&[2.0], 4.0);
        assert_eq!(boxed.n_samples_seen(), 2);

        let pred = boxed.predict(&[1.5]);
        assert!(pred.is_finite());

        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }

    #[test]
    fn polynomial_kernel_works() {
        let mut krls = KRLS::new(Box::new(PolynomialKernel::new(2, 1.0)), 100, 1e-4);

        // Learn y = x^2
        for i in 0..300 {
            let x = (i as f64 - 150.0) * 0.02;
            krls.train(&[x], x * x);
        }

        let pred = krls.predict(&[2.0]);
        assert!(
            (pred - 4.0).abs() < 2.0,
            "poly kernel should approximate x^2, pred(2.0) = {}",
            pred
        );
    }

    #[test]
    fn linear_kernel_matches_linear() {
        let mut krls = KRLS::new(Box::new(LinearKernel), 100, 1e-4);

        // Learn y = 3*x
        for i in 0..200 {
            let x = i as f64 * 0.05;
            krls.train(&[x], 3.0 * x);
        }

        let pred = krls.predict(&[5.0]);
        assert!(
            (pred - 15.0).abs() < 2.0,
            "linear kernel KRLS should learn y=3x, pred(5.0) = {}",
            pred
        );
    }

    #[test]
    fn debug_format_works() {
        let krls = KRLS::new(Box::new(RBFKernel::new(1.0)), 50, 1e-4);
        let debug = format!("{:?}", krls);
        assert!(debug.contains("KRLS"));
        assert!(debug.contains("RBF"));
    }
}
