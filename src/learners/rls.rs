//! Recursive Least Squares (RLS) and kernel-based streaming regressors.
//!
//! This module provides three streaming linear/nonlinear regression models that
//! implement [`StreamingLearner`]:
//!
//! - **[`RecursiveLeastSquares`]** -- Exact streaming OLS via the Sherman-Morrison
//!   matrix inversion lemma. O(d^2) per sample, where d is the feature dimension.
//!   Supports exponential forgetting for non-stationary environments.
//!
//! - **[`StreamingPolynomialRegression`]** -- Wraps RLS with online polynomial
//!   feature expansion, enabling streaming nonlinear regression without manual
//!   feature engineering. Supports arbitrary polynomial degree.
//!
//! - **[`LocallyWeightedRegression`]** -- Nadaraya-Watson kernel regression over
//!   a fixed-capacity circular buffer. Predictions are Gaussian-kernel-weighted
//!   averages of nearby training targets, providing adaptive local fits without
//!   parametric assumptions.
//!
//! # When to use which
//!
//! | Model | Best for | Cost per sample |
//! |-------|----------|----------------|
//! | RLS | Streaming linear regression, high-dimensional | O(d^2) |
//! | Polynomial | Streaming polynomial trends, low-dimensional | O(d_expanded^2) |
//! | LWR | Non-stationary local patterns, low-dimensional | O(buffer_len) |

use std::fmt;

use crate::learner::StreamingLearner;

// ===========================================================================
// Private linear algebra helpers
// ===========================================================================

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Multiply an n x n row-major matrix by an n-vector.
#[inline]
fn mat_vec(mat: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(mat.len(), n * n);
    debug_assert_eq!(v.len(), n);
    let mut result = vec![0.0; n];
    for (i, res) in result.iter_mut().enumerate() {
        let row_start = i * n;
        let mut sum = 0.0;
        for (j, &vj) in v.iter().enumerate() {
            sum += mat[row_start + j] * vj;
        }
        *res = sum;
    }
    result
}

/// P = (P - k * px^T) / lambda, applied in-place.
///
/// `k` is the gain vector (n), `px` is P*x (n), `lambda` is the forgetting
/// factor. The outer product `k * px^T` is subtracted from P, then the entire
/// matrix is divided by lambda.
#[inline]
fn outer_subtract_scaled(p: &mut [f64], k: &[f64], px: &[f64], lambda: f64, n: usize) {
    debug_assert_eq!(p.len(), n * n);
    debug_assert_eq!(k.len(), n);
    debug_assert_eq!(px.len(), n);
    let inv_lambda = 1.0 / lambda;
    for (i, &ki) in k.iter().enumerate() {
        let row_start = i * n;
        for (j, &pxj) in px.iter().enumerate() {
            p[row_start + j] = (p[row_start + j] - ki * pxj) * inv_lambda;
        }
    }
}

// ===========================================================================
// RecursiveLeastSquares
// ===========================================================================

/// Exact streaming ordinary least squares via the Sherman-Morrison formula.
///
/// Maintains a d-dimensional weight vector and a d x d inverse covariance
/// matrix (the "P matrix"). Each new observation updates both in O(d^2) time
/// using the matrix inversion lemma, avoiding any explicit matrix inversion.
///
/// # Forgetting factor
///
/// The forgetting factor `lambda` controls how much weight is given to older
/// observations. `lambda = 1.0` is standard (no forgetting) RLS, equivalent
/// to exact streaming OLS. Values like `0.99` or `0.995` introduce exponential
/// discounting of older data, enabling adaptation to non-stationary targets.
///
/// # Lazy initialisation
///
/// The weight vector and P matrix are allocated on the first call to
/// [`train_one`](StreamingLearner::train_one), when the feature dimensionality
/// becomes known.
///
/// # Examples
///
/// ```
/// use irithyll::learners::rls::RecursiveLeastSquares;
/// use irithyll::learner::StreamingLearner;
///
/// let mut rls = RecursiveLeastSquares::new(1.0);
///
/// // Learn y = 3*x (no intercept -- RLS has no bias term)
/// for i in 0..200 {
///     let x = i as f64 * 0.1;
///     rls.train(&[x], 3.0 * x);
/// }
///
/// let pred = rls.predict(&[5.0]);
/// assert!((pred - 15.0).abs() < 0.1);
/// ```
pub struct RecursiveLeastSquares {
    /// d-dimensional weight vector (learned coefficients).
    weights: Vec<f64>,
    /// d x d inverse covariance matrix, flattened row-major.
    p_matrix: Vec<f64>,
    /// Forgetting factor lambda in (0, 1]. 1.0 = no forgetting.
    forgetting_factor: f64,
    /// Initial P = delta * I. Default 100.0.
    delta: f64,
    /// Feature dimensionality, `None` until first sample.
    n_features: Option<usize>,
    /// Total samples trained on.
    samples_seen: u64,
    /// Exponentially weighted moving average of squared residuals (EWMA alpha = 0.01).
    running_mse: f64,
    /// Whether to use error-driven adaptive forgetting factor.
    adaptive_forgetting: bool,
    /// EWMA of |residual| (alpha = 0.01) for comparing current error to baseline.
    baseline_error: f64,
    /// The actual lambda used in the last P matrix update (for diagnostics).
    effective_forgetting_factor: f64,
}

// ---------------------------------------------------------------------------
// Constructors and accessors
// ---------------------------------------------------------------------------

impl RecursiveLeastSquares {
    /// Create a new RLS learner with the given forgetting factor and default
    /// `delta = 100.0`. Adaptive forgetting is enabled by default.
    ///
    /// # Arguments
    ///
    /// * `forgetting_factor` -- lambda in (0, 1]. Use 1.0 for standard RLS.
    pub fn new(forgetting_factor: f64) -> Self {
        Self::with_delta(forgetting_factor, 100.0)
    }

    /// Create a new RLS learner with explicit forgetting factor and delta.
    /// Adaptive forgetting is enabled by default.
    ///
    /// `delta` controls the initial uncertainty: P is initialised to `delta * I`.
    /// Larger delta means faster initial adaptation but potentially noisier
    /// early predictions.
    pub fn with_delta(forgetting_factor: f64, delta: f64) -> Self {
        Self {
            weights: Vec::new(),
            p_matrix: Vec::new(),
            forgetting_factor,
            delta,
            n_features: None,
            samples_seen: 0,
            running_mse: 0.0,
            adaptive_forgetting: true,
            baseline_error: 0.0,
            effective_forgetting_factor: forgetting_factor,
        }
    }

    /// Create a new RLS learner with fixed (non-adaptive) forgetting factor.
    ///
    /// This disables the error-driven adaptive forgetting mechanism, using
    /// a static lambda for all P matrix updates. Useful when you need
    /// deterministic forgetting behaviour or are benchmarking against
    /// classical RLS.
    pub fn with_fixed_forgetting(forgetting_factor: f64, delta: f64) -> Self {
        Self {
            weights: Vec::new(),
            p_matrix: Vec::new(),
            forgetting_factor,
            delta,
            n_features: None,
            samples_seen: 0,
            running_mse: 0.0,
            adaptive_forgetting: false,
            baseline_error: 0.0,
            effective_forgetting_factor: forgetting_factor,
        }
    }

    /// Current weight vector. Empty before the first training sample.
    #[inline]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The base forgetting factor (lambda).
    #[inline]
    pub fn forgetting_factor(&self) -> f64 {
        self.forgetting_factor
    }

    /// The effective forgetting factor used in the last P matrix update.
    ///
    /// When adaptive forgetting is enabled, this may differ from the base
    /// forgetting factor. When adaptive forgetting is disabled, this always
    /// equals the base forgetting factor.
    #[inline]
    pub fn effective_forgetting_factor(&self) -> f64 {
        self.effective_forgetting_factor
    }

    /// Dynamically set the base forgetting factor, clamped to `[0.95, 1.0]`.
    ///
    /// This allows external callers (e.g. [`StreamingTTT`](crate::ttt::StreamingTTT))
    /// to modulate the forgetting rate based on upstream uncertainty signals,
    /// enabling error-driven adaptation of the RLS readout.
    #[inline]
    pub fn set_forgetting_factor(&mut self, ff: f64) {
        self.forgetting_factor = ff.clamp(0.95, 1.0);
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Lazily initialise weights and P matrix for `d` features.
    fn init(&mut self, d: usize) {
        self.n_features = Some(d);
        self.weights = vec![0.0; d];
        self.p_matrix = vec![0.0; d * d];
        // P = delta * I
        for i in 0..d {
            self.p_matrix[i * d + i] = self.delta;
        }
    }

    /// Compute x^T * P * x (quadratic form with inverse covariance matrix).
    fn quadratic_form_p(&self, features: &[f64]) -> f64 {
        let d = self.weights.len();
        // P * x
        let mut px = vec![0.0; d];
        for (i, px_i) in px.iter_mut().enumerate() {
            let row_start = i * d;
            for (j, &fj) in features.iter().enumerate() {
                *px_i += self.p_matrix[row_start + j] * fj;
            }
        }
        // x^T * (P * x)
        features.iter().zip(px.iter()).map(|(xi, pi)| xi * pi).sum()
    }

    // -----------------------------------------------------------------------
    // Prediction confidence intervals
    // -----------------------------------------------------------------------

    /// Prediction variance at a given point: sigma^2 * (1 + x^T P x).
    ///
    /// This gives the variance of the predictive distribution, combining
    /// noise variance with parameter uncertainty through the P matrix.
    pub fn prediction_variance(&self, features: &[f64]) -> f64 {
        if self.weights.is_empty() {
            return f64::INFINITY;
        }
        let sigma2 = self.noise_variance();
        let x_p_x = self.quadratic_form_p(features);
        sigma2 * (1.0 + x_p_x)
    }

    /// Prediction standard deviation: sqrt(prediction_variance).
    pub fn prediction_std(&self, features: &[f64]) -> f64 {
        self.prediction_variance(features).sqrt()
    }

    /// Predict with confidence interval: returns (mean, lower, upper).
    ///
    /// `z` is the number of standard deviations (e.g. 1.96 for ~95% CI).
    pub fn predict_interval(&self, features: &[f64], z: f64) -> (f64, f64, f64) {
        let mean = self.predict(features);
        let std = self.prediction_std(features);
        (mean, mean - z * std, mean + z * std)
    }

    /// Estimated noise variance from EWMA of squared residuals.
    pub fn noise_variance(&self) -> f64 {
        self.running_mse
    }

    /// Re-regularize the covariance matrix to `delta * I` and reset weights.
    ///
    /// Called internally when the P matrix becomes ill-conditioned (extreme
    /// diagonal values or NaN). This is a numerical safety guard that prevents
    /// the RLS readout from exploding after sudden distribution shifts.
    fn reset_covariance(&mut self) {
        if let Some(d) = self.n_features {
            // Reset P to delta * I
            self.p_matrix.fill(0.0);
            for i in 0..d {
                self.p_matrix[i * d + i] = self.delta;
            }
            // Reset weights to zero
            self.weights.fill(0.0);
            self.running_mse = 0.0;
            self.baseline_error = 0.0;
        }
    }

    /// Check P matrix diagonal health and re-regularize if needed.
    ///
    /// Returns `true` if a reset was performed.
    fn check_covariance_health(&mut self) -> bool {
        if let Some(d) = self.n_features {
            let mut max_diag: f64 = 0.0;
            let mut has_nan = false;
            for i in 0..d {
                let diag = self.p_matrix[i * d + i];
                if diag.is_nan() || !diag.is_finite() {
                    has_nan = true;
                    break;
                }
                if diag.abs() > max_diag {
                    max_diag = diag.abs();
                }
                // Negative diagonal is also pathological
                if diag < 0.0 {
                    has_nan = true;
                    break;
                }
            }

            // Also check if any weight is NaN
            if !has_nan {
                has_nan = self.weights.iter().any(|w| !w.is_finite());
            }

            if has_nan || max_diag > 1e10 {
                self.reset_covariance();
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner
// ---------------------------------------------------------------------------

impl StreamingLearner for RecursiveLeastSquares {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let d = features.len();

        // Lazy init on first sample.
        if self.n_features.is_none() {
            self.init(d);
        }
        let n = self.n_features.unwrap();
        debug_assert_eq!(d, n);

        // Update running MSE (EWMA of squared residuals) BEFORE weight update
        // so it measures the pre-update prediction error.
        let residual = target - self.predict(features);
        self.running_mse = 0.99 * self.running_mse + 0.01 * residual * residual;

        // Compute effective forgetting factor (adaptive or fixed).
        let effective_ff = if self.adaptive_forgetting {
            // Compute error ratio: current |error| vs baseline
            let abs_error = residual.abs();

            // Update baseline with slow EWMA
            self.baseline_error = 0.99 * self.baseline_error + 0.01 * abs_error;

            // Error ratio: >1 means error is above baseline (possible drift)
            let ratio = if self.baseline_error > 1e-15 {
                (abs_error / self.baseline_error).clamp(0.1, 10.0)
            } else {
                1.0
            };

            // Modulate: high error -> lower lambda (forget faster)
            // ratio=1 -> ff unchanged. ratio=2 -> ff * 0.999. ratio=0.5 -> ff * 1.0005
            // The key: when error spikes, we forget old covariance faster
            let adjustment = 1.0 - 0.001 * (ratio - 1.0);
            let adaptive = self.forgetting_factor * adjustment.clamp(0.99, 1.001);
            adaptive.clamp(0.95, 1.0) // Never go below 0.95 (forget at most 5% per step)
        } else {
            self.forgetting_factor
        };
        self.effective_forgetting_factor = effective_ff;

        // Prediction error, scaled by sqrt(weight).
        let prediction = dot(&self.weights, features);
        let alpha = (target - prediction) * weight.sqrt();

        // Gain vector: k = (P * x) / (lambda + x^T * P * x)
        let px = mat_vec(&self.p_matrix, features, n);
        let denom = (effective_ff + dot(features, &px)).max(1e-8);
        let mut k = vec![0.0; n];
        let inv_denom = 1.0 / denom;
        for (ki, &pxi) in k.iter_mut().zip(px.iter()) {
            *ki = pxi * inv_denom;
        }

        // Update weights: w = w + alpha * k
        for (wi, &ki) in self.weights.iter_mut().zip(k.iter()) {
            *wi += alpha * ki;
        }

        // Update P: P = (P - k * (P*x)^T) / lambda
        outer_subtract_scaled(&mut self.p_matrix, &k, &px, effective_ff, n);

        // Numerical stability guard: re-regularize if P becomes ill-conditioned.
        // This prevents the covariance matrix from exploding after sudden
        // distribution shifts (concept drift).
        self.check_covariance_health();

        self.samples_seen += 1;
    }

    #[inline]
    fn predict(&self, features: &[f64]) -> f64 {
        if self.weights.is_empty() {
            return 0.0;
        }
        dot(&self.weights, features)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        self.weights.clear();
        self.p_matrix.clear();
        self.n_features = None;
        self.samples_seen = 0;
        self.running_mse = 0.0;
        self.baseline_error = 0.0;
        self.effective_forgetting_factor = self.forgetting_factor;
    }

    fn diagnostics_array(&self) -> [f64; 5] {
        [
            0.0,                          // residual_alignment
            1.0 - self.forgetting_factor, // reg_sensitivity
            0.0,                          // depth_sufficiency
            self.weights.len() as f64,    // effective_dof
            self.running_mse.sqrt(),      // uncertainty
        ]
    }

    fn adjust_config(&mut self, lr_multiplier: f64, _lambda_delta: f64) {
        // Scale the forgetting factor. Clamp to (0, 1].
        self.forgetting_factor = (self.forgetting_factor * lr_multiplier).clamp(1e-6, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Clone impl
// ---------------------------------------------------------------------------

impl Clone for RecursiveLeastSquares {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            p_matrix: self.p_matrix.clone(),
            forgetting_factor: self.forgetting_factor,
            delta: self.delta,
            n_features: self.n_features,
            samples_seen: self.samples_seen,
            running_mse: self.running_mse,
            adaptive_forgetting: self.adaptive_forgetting,
            baseline_error: self.baseline_error,
            effective_forgetting_factor: self.effective_forgetting_factor,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

impl fmt::Debug for RecursiveLeastSquares {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecursiveLeastSquares")
            .field("n_features", &self.n_features)
            .field("forgetting_factor", &self.forgetting_factor)
            .field(
                "effective_forgetting_factor",
                &self.effective_forgetting_factor,
            )
            .field("adaptive_forgetting", &self.adaptive_forgetting)
            .field("delta", &self.delta)
            .field("samples_seen", &self.samples_seen)
            .field("running_mse", &self.running_mse)
            .field("baseline_error", &self.baseline_error)
            .field("weights", &self.weights)
            .finish()
    }
}

// ===========================================================================
// StreamingPolynomialRegression
// ===========================================================================

/// Streaming polynomial regression via RLS with online feature expansion.
///
/// For an input vector `[x1, x2, ..., xd]` and polynomial degree `p`, the
/// feature expansion generates all monomials of total degree 1 through `p`:
///
/// - Degree 1: `x1, x2, ..., xd` (original features)
/// - Degree 2: `x1^2, x1*x2, ..., xd^2` (all pairs)
/// - Degree 3: `x1^3, x1^2*x2, ..., xd^3` (all triples)
///
/// The expanded feature vector is passed to an inner [`RecursiveLeastSquares`]
/// learner, enabling streaming nonlinear regression without manual feature
/// engineering.
///
/// # Examples
///
/// ```
/// use irithyll::learners::rls::StreamingPolynomialRegression;
/// use irithyll::learner::StreamingLearner;
///
/// let mut poly = StreamingPolynomialRegression::new(2, 1.0);
///
/// // Learn y = x^2
/// for i in 0..300 {
///     let x = (i as f64 - 150.0) * 0.02;
///     poly.train(&[x], x * x);
/// }
///
/// let pred = poly.predict(&[2.0]);
/// assert!((pred - 4.0).abs() < 1.0);
/// ```
pub struct StreamingPolynomialRegression {
    /// Inner RLS learner operating on expanded features.
    rls: RecursiveLeastSquares,
    /// Polynomial degree (2 or 3 typically).
    degree: usize,
    /// Total samples trained on.
    samples_seen: u64,
}

// ---------------------------------------------------------------------------
// Constructors and accessors
// ---------------------------------------------------------------------------

impl StreamingPolynomialRegression {
    /// Create a new streaming polynomial regressor.
    ///
    /// # Arguments
    ///
    /// * `degree` -- maximum polynomial degree (must be >= 1)
    /// * `forgetting_factor` -- lambda for the inner RLS
    pub fn new(degree: usize, forgetting_factor: f64) -> Self {
        assert!(degree >= 1, "polynomial degree must be >= 1, got {degree}");
        Self {
            rls: RecursiveLeastSquares::new(forgetting_factor),
            degree,
            samples_seen: 0,
        }
    }

    /// The polynomial degree.
    #[inline]
    pub fn degree(&self) -> usize {
        self.degree
    }

    // -----------------------------------------------------------------------
    // Feature expansion
    // -----------------------------------------------------------------------

    /// Expand features to include all monomials up to `self.degree`.
    ///
    /// For input `[x1, x2]` with degree=2, produces:
    /// `[x1, x2, x1^2, x1*x2, x2^2]`
    fn expand_features(&self, features: &[f64]) -> Vec<f64> {
        let d = features.len();
        let mut expanded = Vec::new();

        // Degree 1: original features.
        expanded.extend_from_slice(features);

        // Degrees 2..=self.degree: enumerate monomials via index tuples.
        for deg in 2..=self.degree {
            Self::enumerate_monomials(features, d, deg, &mut expanded);
        }

        expanded
    }

    /// Enumerate all monomials of exactly `degree` over `d` variables and
    /// append their values to `out`.
    ///
    /// Uses the "stars and bars" enumeration: indices `[i0, i1, ..., i_{deg-1}]`
    /// with `0 <= i0 <= i1 <= ... <= i_{deg-1} < d`.
    fn enumerate_monomials(features: &[f64], d: usize, degree: usize, out: &mut Vec<f64>) {
        let mut indices = vec![0usize; degree];
        loop {
            // Compute the monomial value for current index tuple.
            let mut val = 1.0;
            for &idx in &indices {
                val *= features[idx];
            }
            out.push(val);

            // Advance indices (lexicographic order with non-decreasing constraint).
            let mut pos = degree - 1;
            loop {
                indices[pos] += 1;
                if indices[pos] < d {
                    // Fill all subsequent positions with the same value (non-decreasing).
                    let v = indices[pos];
                    for idx in indices.iter_mut().take(degree).skip(pos + 1) {
                        *idx = v;
                    }
                    break;
                }
                if pos == 0 {
                    return; // All monomials exhausted.
                }
                pos -= 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner
// ---------------------------------------------------------------------------

impl StreamingLearner for StreamingPolynomialRegression {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let expanded = self.expand_features(features);
        self.rls.train_one(&expanded, target, weight);
        self.samples_seen += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let expanded = self.expand_features(features);
        self.rls.predict(&expanded)
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        self.rls.reset();
        self.samples_seen = 0;
    }

    fn diagnostics_array(&self) -> [f64; 5] {
        [0.0; 5]
    }
}

// ---------------------------------------------------------------------------
// Clone impl
// ---------------------------------------------------------------------------

impl Clone for StreamingPolynomialRegression {
    fn clone(&self) -> Self {
        Self {
            rls: self.rls.clone(),
            degree: self.degree,
            samples_seen: self.samples_seen,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

impl fmt::Debug for StreamingPolynomialRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamingPolynomialRegression")
            .field("degree", &self.degree)
            .field("samples_seen", &self.samples_seen)
            .field("rls", &self.rls)
            .finish()
    }
}

// ===========================================================================
// LocallyWeightedRegression
// ===========================================================================

/// Nadaraya-Watson kernel regression over a fixed-capacity circular buffer.
///
/// Training simply stores observations in a circular buffer. At prediction
/// time, all buffered samples are weighted by a Gaussian kernel centred on
/// the query point, and the prediction is the weighted average of targets:
///
/// ```text
/// y_hat(x) = sum_i K(x, x_i) * w_i * y_i  /  sum_i K(x, x_i) * w_i
/// K(x, x_i) = exp(-||x - x_i||^2 / (2 * bandwidth^2))
/// ```
///
/// This is a non-parametric, lazy learner: it makes no assumptions about the
/// underlying function form and adapts entirely at prediction time.
///
/// # Bandwidth
///
/// The `bandwidth` parameter (sigma) controls locality:
/// - Small bandwidth: predictions are dominated by the nearest neighbours
/// - Large bandwidth: predictions approach a global weighted average
///
/// # Examples
///
/// ```
/// use irithyll::learners::rls::LocallyWeightedRegression;
/// use irithyll::learner::StreamingLearner;
///
/// let mut lwr = LocallyWeightedRegression::new(500, 1.0);
///
/// // Train on a sine wave
/// for i in 0..500 {
///     let x = i as f64 * 0.02;
///     lwr.train(&[x], x.sin());
/// }
///
/// // Predict near a training point
/// let pred = lwr.predict(&[1.0]);
/// assert!((pred - 1.0_f64.sin()).abs() < 0.2);
/// ```
pub struct LocallyWeightedRegression {
    /// Circular buffer of feature vectors.
    buffer_features: Vec<Vec<f64>>,
    /// Circular buffer of target values.
    buffer_targets: Vec<f64>,
    /// Circular buffer of sample weights.
    buffer_weights: Vec<f64>,
    /// Maximum buffer size.
    capacity: usize,
    /// Next write position (wraps around).
    head: usize,
    /// Current number of valid entries.
    len: usize,
    /// Gaussian kernel bandwidth (sigma).
    bandwidth: f64,
    /// Total samples ever trained on (including those evicted from the buffer).
    samples_seen: u64,
}

// ---------------------------------------------------------------------------
// Constructors and accessors
// ---------------------------------------------------------------------------

impl LocallyWeightedRegression {
    /// Create a new locally weighted regressor.
    ///
    /// # Arguments
    ///
    /// * `capacity` -- maximum number of samples to store in the buffer
    /// * `bandwidth` -- Gaussian kernel bandwidth (sigma); must be > 0
    pub fn new(capacity: usize, bandwidth: f64) -> Self {
        assert!(capacity > 0, "capacity must be > 0, got {capacity}");
        assert!(bandwidth > 0.0, "bandwidth must be > 0.0, got {bandwidth}");
        Self {
            buffer_features: Vec::with_capacity(capacity),
            buffer_targets: Vec::with_capacity(capacity),
            buffer_weights: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            len: 0,
            bandwidth,
            samples_seen: 0,
        }
    }

    /// Maximum buffer capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The Gaussian kernel bandwidth.
    #[inline]
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Current number of samples in the buffer.
    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.len
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Squared Euclidean distance between two vectors.
    #[inline]
    fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let mut sum = 0.0;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner
// ---------------------------------------------------------------------------

impl StreamingLearner for LocallyWeightedRegression {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        if self.len < self.capacity {
            // Buffer not yet full -- push to the end.
            self.buffer_features.push(features.to_vec());
            self.buffer_targets.push(target);
            self.buffer_weights.push(weight);
            self.len += 1;
            self.head = self.len % self.capacity;
        } else {
            // Buffer full -- overwrite at head.
            self.buffer_features[self.head] = features.to_vec();
            self.buffer_targets[self.head] = target;
            self.buffer_weights[self.head] = weight;
            self.head = (self.head + 1) % self.capacity;
        }
        self.samples_seen += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        if self.len == 0 {
            return 0.0;
        }

        let two_bw_sq = 2.0 * self.bandwidth * self.bandwidth;
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for i in 0..self.len {
            let sq_d = Self::sq_dist(features, &self.buffer_features[i]);
            let kernel_w = (-sq_d / two_bw_sq).exp();
            let w = kernel_w * self.buffer_weights[i];
            weighted_sum += w * self.buffer_targets[i];
            weight_total += w;
        }

        if weight_total.abs() < 1e-15 {
            return 0.0;
        }
        weighted_sum / weight_total
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        self.buffer_features.clear();
        self.buffer_targets.clear();
        self.buffer_weights.clear();
        self.head = 0;
        self.len = 0;
        self.samples_seen = 0;
    }

    fn diagnostics_array(&self) -> [f64; 5] {
        [0.0; 5]
    }
}

// ---------------------------------------------------------------------------
// Clone impl
// ---------------------------------------------------------------------------

impl Clone for LocallyWeightedRegression {
    fn clone(&self) -> Self {
        Self {
            buffer_features: self.buffer_features.clone(),
            buffer_targets: self.buffer_targets.clone(),
            buffer_weights: self.buffer_weights.clone(),
            capacity: self.capacity,
            head: self.head,
            len: self.len,
            bandwidth: self.bandwidth,
            samples_seen: self.samples_seen,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug impl
// ---------------------------------------------------------------------------

impl fmt::Debug for LocallyWeightedRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocallyWeightedRegression")
            .field("capacity", &self.capacity)
            .field("len", &self.len)
            .field("bandwidth", &self.bandwidth)
            .field("samples_seen", &self.samples_seen)
            .finish()
    }
}

// ===========================================================================
// DiagnosticSource impls
// ===========================================================================

impl crate::automl::DiagnosticSource for RecursiveLeastSquares {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        Some(crate::automl::ConfigDiagnostics {
            // Dimension not known until first sample.
            effective_dof: self.weights().len() as f64,
            regularization_sensitivity: 1.0 - self.forgetting_factor(),
            // Prediction uncertainty: std dev of residuals.
            uncertainty: self.noise_variance().sqrt(),
            ..Default::default()
        })
    }
}

impl crate::automl::DiagnosticSource for StreamingPolynomialRegression {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        None
    }
}

impl crate::automl::DiagnosticSource for LocallyWeightedRegression {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        None
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::StreamingLearner;

    // ===================================================================
    // RLS tests
    // ===================================================================

    #[test]
    fn test_rls_creation() {
        let rls = RecursiveLeastSquares::new(0.99);
        assert_eq!(rls.n_samples_seen(), 0);
        assert!(rls.weights().is_empty());
        assert!((rls.forgetting_factor() - 0.99).abs() < 1e-15);
        // Untrained model predicts zero.
        assert!((rls.predict(&[1.0, 2.0]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_rls_simple_linear() {
        // Learn y = 2*x + 1.
        let mut rls = RecursiveLeastSquares::new(1.0);

        for i in 0..500 {
            let x = i as f64 * 0.01;
            let y = 2.0 * x + 1.0;
            // Use two features: [x, 1.0] (bias trick).
            rls.train(&[x, 1.0], y);
        }

        assert_eq!(rls.n_samples_seen(), 500);

        // Weights should be close to [2.0, 1.0].
        let w = rls.weights();
        assert_eq!(w.len(), 2);
        assert!(
            (w[0] - 2.0).abs() < 0.05,
            "expected w[0] ~ 2.0, got {}",
            w[0]
        );
        assert!(
            (w[1] - 1.0).abs() < 0.05,
            "expected w[1] ~ 1.0, got {}",
            w[1]
        );

        // Prediction check.
        let pred = rls.predict(&[5.0, 1.0]);
        assert!(
            (pred - 11.0).abs() < 0.5,
            "expected pred ~ 11.0, got {}",
            pred
        );
    }

    #[test]
    fn test_rls_multivariate() {
        // Learn y = x1 + 2*x2.
        let mut rls = RecursiveLeastSquares::new(1.0);

        for i in 0..800 {
            let x1 = (i as f64 * 0.037).sin();
            let x2 = (i as f64 * 0.053).cos();
            let y = x1 + 2.0 * x2;
            rls.train(&[x1, x2], y);
        }

        let w = rls.weights();
        assert!(
            (w[0] - 1.0).abs() < 0.1,
            "expected w[0] ~ 1.0, got {}",
            w[0]
        );
        assert!(
            (w[1] - 2.0).abs() < 0.1,
            "expected w[1] ~ 2.0, got {}",
            w[1]
        );
    }

    #[test]
    fn test_rls_forgetting() {
        // With forgetting, RLS should adapt to a distribution shift.
        let mut rls = RecursiveLeastSquares::new(0.98);

        // Phase 1: y = 1*x (with bias trick).
        for i in 0..500 {
            let x = i as f64 * 0.01;
            rls.train(&[x, 1.0], 1.0 * x + 0.0);
        }

        // Phase 2: y = 5*x + 10 -- drastic shift.
        for i in 0..500 {
            let x = i as f64 * 0.01;
            rls.train(&[x, 1.0], 5.0 * x + 10.0);
        }

        // With forgetting, weights should have adapted toward [5.0, 10.0].
        let w = rls.weights();
        assert!(
            (w[0] - 5.0).abs() < 1.5,
            "forgetting RLS should adapt slope toward 5.0, got {}",
            w[0]
        );
    }

    #[test]
    fn test_rls_reset() {
        let mut rls = RecursiveLeastSquares::new(1.0);
        rls.train(&[1.0, 2.0], 5.0);
        rls.train(&[3.0, 4.0], 7.0);
        assert_eq!(rls.n_samples_seen(), 2);
        assert!(!rls.weights().is_empty());

        rls.reset();
        assert_eq!(rls.n_samples_seen(), 0);
        assert!(rls.weights().is_empty());
        assert!((rls.predict(&[1.0, 2.0]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_rls_trait_object() {
        let rls = RecursiveLeastSquares::new(1.0);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(rls);

        boxed.train(&[1.0, 1.0], 3.0);
        boxed.train(&[2.0, 1.0], 5.0);
        assert_eq!(boxed.n_samples_seen(), 2);

        let pred = boxed.predict(&[1.0, 1.0]);
        assert!(pred.is_finite());

        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }

    // ===================================================================
    // Polynomial tests
    // ===================================================================

    #[test]
    fn test_poly_quadratic() {
        // Learn y = x^2. Polynomial degree 2 should capture this well;
        // linear RLS would fail.
        let mut poly = StreamingPolynomialRegression::new(2, 1.0);

        for i in 0..600 {
            let x = (i as f64 - 300.0) * 0.01;
            poly.train(&[x], x * x);
        }

        // Test at a few points.
        let pred_at_2 = poly.predict(&[2.0]);
        assert!(
            (pred_at_2 - 4.0).abs() < 1.0,
            "expected pred(2.0) ~ 4.0, got {}",
            pred_at_2
        );

        let pred_at_neg1 = poly.predict(&[-1.0]);
        assert!(
            (pred_at_neg1 - 1.0).abs() < 1.0,
            "expected pred(-1.0) ~ 1.0, got {}",
            pred_at_neg1
        );
    }

    #[test]
    fn test_poly_expansion() {
        // Verify the expanded feature dimension for degree 2 with 2 input features.
        // Input [x1, x2] -> expanded [x1, x2, x1^2, x1*x2, x2^2] = 5 features.
        let poly = StreamingPolynomialRegression::new(2, 1.0);
        let expanded = poly.expand_features(&[3.0, 4.0]);
        assert_eq!(
            expanded.len(),
            5,
            "degree-2 expansion of 2 features should give 5"
        );
        // Check values: [3, 4, 9, 12, 16]
        assert!((expanded[0] - 3.0).abs() < 1e-12);
        assert!((expanded[1] - 4.0).abs() < 1e-12);
        assert!((expanded[2] - 9.0).abs() < 1e-12); // 3^2
        assert!((expanded[3] - 12.0).abs() < 1e-12); // 3*4
        assert!((expanded[4] - 16.0).abs() < 1e-12); // 4^2

        // Degree 3 with 2 features: 2 + 3 + 4 = 9 features.
        let poly3 = StreamingPolynomialRegression::new(3, 1.0);
        let expanded3 = poly3.expand_features(&[2.0, 3.0]);
        assert_eq!(
            expanded3.len(),
            9,
            "degree-3 expansion of 2 features should give 9"
        );
    }

    #[test]
    fn test_poly_reset() {
        let mut poly = StreamingPolynomialRegression::new(2, 1.0);
        poly.train(&[1.0], 1.0);
        poly.train(&[2.0], 4.0);
        assert_eq!(poly.n_samples_seen(), 2);

        poly.reset();
        assert_eq!(poly.n_samples_seen(), 0);
        assert!((poly.predict(&[1.0]) - 0.0).abs() < 1e-15);
    }

    // ===================================================================
    // LWR tests
    // ===================================================================

    #[test]
    fn test_lwr_creation() {
        let lwr = LocallyWeightedRegression::new(100, 0.5);
        assert_eq!(lwr.n_samples_seen(), 0);
        assert_eq!(lwr.capacity(), 100);
        assert!((lwr.bandwidth() - 0.5).abs() < 1e-15);
        assert_eq!(lwr.buffer_len(), 0);
        // Empty buffer predicts zero.
        assert!((lwr.predict(&[1.0]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_lwr_constant() {
        // All training targets are 42.0 -- prediction everywhere should be ~42.
        let mut lwr = LocallyWeightedRegression::new(200, 1.0);
        for i in 0..200 {
            let x = i as f64 * 0.1;
            lwr.train(&[x], 42.0);
        }

        let pred = lwr.predict(&[5.0]);
        assert!(
            (pred - 42.0).abs() < 1e-6,
            "constant target should predict ~42.0, got {}",
            pred
        );
    }

    #[test]
    fn test_lwr_local() {
        // Train two clusters with different targets.
        let mut lwr = LocallyWeightedRegression::new(200, 0.5);

        // Cluster A around x=0, y=10.
        for i in 0..100 {
            let x = (i as f64 - 50.0) * 0.01;
            lwr.train(&[x], 10.0);
        }

        // Cluster B around x=10, y=50.
        for i in 0..100 {
            let x = 10.0 + (i as f64 - 50.0) * 0.01;
            lwr.train(&[x], 50.0);
        }

        // Prediction near cluster A should be close to 10.
        let pred_a = lwr.predict(&[0.0]);
        assert!(
            (pred_a - 10.0).abs() < 5.0,
            "prediction near cluster A should be ~10, got {}",
            pred_a
        );

        // Prediction near cluster B should be close to 50.
        let pred_b = lwr.predict(&[10.0]);
        assert!(
            (pred_b - 50.0).abs() < 5.0,
            "prediction near cluster B should be ~50, got {}",
            pred_b
        );
    }

    #[test]
    fn test_lwr_buffer_capacity() {
        let mut lwr = LocallyWeightedRegression::new(5, 1.0);

        // Insert 8 samples into a buffer of capacity 5.
        for i in 0..8 {
            lwr.train(&[i as f64], i as f64 * 10.0);
        }

        assert_eq!(lwr.n_samples_seen(), 8);
        assert_eq!(lwr.buffer_len(), 5);

        // The buffer should contain the last 5 samples (indices 3,4,5,6,7).
        // Prediction at x=7 should be dominated by target=70.
        let pred = lwr.predict(&[7.0]);
        assert!(
            (pred - 70.0).abs() < 20.0,
            "prediction at x=7 should be near 70, got {}",
            pred
        );
    }

    #[test]
    fn test_lwr_reset() {
        let mut lwr = LocallyWeightedRegression::new(100, 1.0);
        for i in 0..50 {
            lwr.train(&[i as f64], i as f64);
        }
        assert_eq!(lwr.n_samples_seen(), 50);
        assert_eq!(lwr.buffer_len(), 50);

        lwr.reset();
        assert_eq!(lwr.n_samples_seen(), 0);
        assert_eq!(lwr.buffer_len(), 0);
        assert!((lwr.predict(&[1.0]) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_lwr_trait_object() {
        let lwr = LocallyWeightedRegression::new(100, 1.0);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(lwr);

        boxed.train(&[1.0], 10.0);
        boxed.train(&[2.0], 20.0);
        assert_eq!(boxed.n_samples_seen(), 2);

        let pred = boxed.predict(&[1.5]);
        assert!(pred.is_finite());

        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }

    // ===================================================================
    // RLS confidence interval tests
    // ===================================================================

    #[test]
    fn confidence_intervals_narrow_with_data() {
        // More data -> lower prediction variance (P shrinks)
        let mut rls = RecursiveLeastSquares::new(0.01);
        // Train on y = 2x + 1
        for i in 0..100 {
            let x = i as f64 * 0.1;
            rls.train(&[x], 2.0 * x + 1.0);
        }
        let var_100 = rls.prediction_variance(&[5.0]);

        for i in 100..1000 {
            let x = i as f64 * 0.1;
            rls.train(&[x], 2.0 * x + 1.0);
        }
        let var_1000 = rls.prediction_variance(&[5.0]);

        assert!(
            var_1000 < var_100,
            "variance should decrease with more data: {} vs {}",
            var_1000,
            var_100
        );
    }

    #[test]
    fn predict_interval_z_scaling() {
        let mut rls = RecursiveLeastSquares::new(0.01);
        for i in 0..200 {
            let x = i as f64 * 0.05;
            rls.train(&[x], x * x + 0.1); // slight noise-like pattern
        }
        let (mean1, lo1, hi1) = rls.predict_interval(&[5.0], 1.0);
        let (mean2, lo2, hi2) = rls.predict_interval(&[5.0], 2.0);
        assert!((mean1 - mean2).abs() < 1e-12); // same mean
        let width1 = hi1 - lo1;
        let width2 = hi2 - lo2;
        assert!(
            (width2 / width1 - 2.0).abs() < 0.01,
            "width should scale with z: w1={}, w2={}",
            width1,
            width2
        );
    }

    #[test]
    fn noise_variance_reflects_residuals() {
        let mut rls = RecursiveLeastSquares::new(0.01);
        // Perfect linear data: noise variance should be small
        for i in 0..500 {
            let x = i as f64 * 0.01;
            rls.train(&[x], 3.0 * x);
        }
        let nv = rls.noise_variance();
        assert!(
            nv < 1.0,
            "noise variance should be small for perfect data: {}",
            nv
        );
    }

    #[test]
    fn prediction_bounds_are_finite() {
        let mut rls = RecursiveLeastSquares::new(0.01);
        rls.train(&[1.0], 2.0);
        let (mean, lo, hi) = rls.predict_interval(&[1.0], 1.96);
        assert!(mean.is_finite());
        assert!(lo.is_finite());
        assert!(hi.is_finite());
        assert!(lo <= mean);
        assert!(mean <= hi);
    }

    // ===================================================================
    // Adaptive forgetting tests
    // ===================================================================

    #[test]
    fn adaptive_ff_stable_data_stays_near_base() {
        // Feed stable linear data: effective_ff should stay near the base forgetting factor.
        let base_ff = 0.99;
        let mut rls = RecursiveLeastSquares::new(base_ff);

        // Train on a clean linear relationship y = 2x + 1 with bias trick.
        for i in 0..500 {
            let x = i as f64 * 0.01;
            rls.train(&[x, 1.0], 2.0 * x + 1.0);
        }

        let eff = rls.effective_forgetting_factor();
        let diff = (eff - base_ff).abs();
        assert!(
            diff < 0.005,
            "stable data: effective_ff should stay near base {}, got {} (diff={})",
            base_ff,
            eff,
            diff
        );
    }

    #[test]
    fn adaptive_ff_drops_on_sudden_shift() {
        // Feed stable data then a sudden distribution shift.
        // The effective_ff should drop below the base during the shift.
        let base_ff = 0.99;
        let mut rls = RecursiveLeastSquares::new(base_ff);

        // Phase 1: stable y = x.
        for i in 0..300 {
            let x = i as f64 * 0.01;
            rls.train(&[x, 1.0], x);
        }

        let ff_before_shift = rls.effective_forgetting_factor();

        // Phase 2: abrupt shift to y = 10x + 50.
        // Train a few samples from the new distribution and capture ff.
        let mut min_ff_during_shift = base_ff;
        for i in 0..50 {
            let x = i as f64 * 0.01;
            rls.train(&[x, 1.0], 10.0 * x + 50.0);
            let eff = rls.effective_forgetting_factor();
            if eff < min_ff_during_shift {
                min_ff_during_shift = eff;
            }
        }

        assert!(
            min_ff_during_shift < ff_before_shift,
            "effective_ff should drop during distribution shift: before={}, min_during={}",
            ff_before_shift,
            min_ff_during_shift
        );
    }

    #[test]
    fn adaptive_ff_recovers_after_shift() {
        // After a shift stabilizes, effective_ff should recover toward the base.
        let base_ff = 0.99;
        let mut rls = RecursiveLeastSquares::new(base_ff);

        // Phase 1: stable y = x.
        for i in 0..300 {
            let x = i as f64 * 0.01;
            rls.train(&[x, 1.0], x);
        }

        // Phase 2: abrupt shift to y = 10x + 50.
        for i in 0..50 {
            let x = i as f64 * 0.01;
            rls.train(&[x, 1.0], 10.0 * x + 50.0);
        }

        let ff_after_shift = rls.effective_forgetting_factor();

        // Phase 3: continue with the new distribution (now stable).
        for i in 0..500 {
            let x = i as f64 * 0.01;
            rls.train(&[x, 1.0], 10.0 * x + 50.0);
        }

        let ff_recovered = rls.effective_forgetting_factor();
        let diff_from_base = (ff_recovered - base_ff).abs();

        assert!(
            ff_recovered > ff_after_shift || diff_from_base < 0.005,
            "effective_ff should recover after shift stabilizes: after_shift={}, recovered={}, base={}",
            ff_after_shift, ff_recovered, base_ff
        );
    }

    #[test]
    fn fixed_forgetting_does_not_adapt() {
        // with_fixed_forgetting should keep effective_ff equal to base at all times.
        let base_ff = 0.99;
        let mut rls = RecursiveLeastSquares::with_fixed_forgetting(base_ff, 100.0);

        // Phase 1: stable.
        for i in 0..200 {
            let x = i as f64 * 0.01;
            rls.train(&[x, 1.0], x);
        }
        assert!(
            (rls.effective_forgetting_factor() - base_ff).abs() < 1e-15,
            "fixed forgetting: effective_ff must equal base"
        );

        // Phase 2: shift.
        for i in 0..100 {
            let x = i as f64 * 0.01;
            rls.train(&[x, 1.0], 10.0 * x + 50.0);
        }
        assert!(
            (rls.effective_forgetting_factor() - base_ff).abs() < 1e-15,
            "fixed forgetting: effective_ff must equal base even after shift"
        );
    }
}
