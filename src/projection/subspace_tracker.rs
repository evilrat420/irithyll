//! PAST (Projection Approximation Subspace Tracking) algorithm.
//!
//! Implements the streaming subspace tracker from Yang (1995). Given a stream
//! of d-dimensional inputs, PAST maintains an r-dimensional projection that
//! tracks the dominant subspace using RLS-style rank-one updates.
//!
//! # Complexity
//!
//! - **Time:** O(d_in * rank) per sample
//! - **Memory:** O(d_in * rank + rank^2)
//!
//! # References
//!
//! Yang, B. (1995). Projection approximation subspace tracking.
//! *IEEE Transactions on Signal Processing*, 43(1), 95--107.

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

/// Multiply a row-major n x n matrix by an n-vector.
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

/// P = (P - g * h^T) / lambda, applied in-place.
///
/// `g` is the gain vector (n), `h` is P*y (n), `lambda` is the forgetting
/// factor. The outer product `g * h^T` is subtracted from P, then the entire
/// matrix is divided by lambda.
#[inline]
fn outer_subtract_scaled(p: &mut [f64], g: &[f64], h: &[f64], lambda: f64, n: usize) {
    debug_assert_eq!(p.len(), n * n);
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(h.len(), n);
    let inv_lambda = 1.0 / lambda;
    for (i, &gi) in g.iter().enumerate() {
        let row_start = i * n;
        for (j, &hj) in h.iter().enumerate() {
            p[row_start + j] = (p[row_start + j] - gi * hj) * inv_lambda;
        }
    }
}

/// Multiply a column-major (rows x cols) matrix transposed by a vector.
///
/// Computes W^T * x where W is d_in x rank stored column-major.
/// Result has `rank` elements.
#[inline]
fn mat_t_vec_col_major(w: &[f64], x: &[f64], d_in: usize, rank: usize) -> Vec<f64> {
    debug_assert_eq!(w.len(), d_in * rank);
    debug_assert_eq!(x.len(), d_in);
    let mut result = vec![0.0; rank];
    for (col, res) in result.iter_mut().enumerate() {
        let col_start = col * d_in;
        let mut sum = 0.0;
        for row in 0..d_in {
            sum += w[col_start + row] * x[row];
        }
        *res = sum;
    }
    result
}

/// Multiply a column-major (rows x cols) matrix by a vector.
///
/// Computes W * y where W is d_in x rank stored column-major.
/// Result has `d_in` elements.
#[inline]
fn mat_vec_col_major(w: &[f64], y: &[f64], d_in: usize, rank: usize) -> Vec<f64> {
    debug_assert_eq!(w.len(), d_in * rank);
    debug_assert_eq!(y.len(), rank);
    let mut result = vec![0.0; d_in];
    for (col, &yc) in y.iter().enumerate() {
        let col_start = col * d_in;
        for row in 0..d_in {
            result[row] += w[col_start + row] * yc;
        }
    }
    result
}

// ===========================================================================
// RNG helpers (xorshift64, same pattern as the rest of irithyll)
// ===========================================================================

/// Advance xorshift64 state and return raw u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

/// Uniform f64 in [0, 1) from xorshift64.
#[inline]
fn xorshift64_f64(state: &mut u64) -> f64 {
    // Use 53-bit mantissa for full f64 precision.
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

// ===========================================================================
// SubspaceTracker
// ===========================================================================

/// Streaming subspace tracker using the PAST algorithm (Yang, 1995).
///
/// Maintains an r-dimensional projection of R^d that adapts online via
/// RLS-style rank-one updates. The projection matrix W (d_in x rank) and
/// an inverse correlation matrix P (rank x rank) are updated for every
/// new sample in O(d_in * rank) time.
///
/// # Usage
///
/// ```
/// use irithyll::projection::SubspaceTracker;
///
/// let mut tracker = SubspaceTracker::new(10, 3, 0.998, 100.0, 42);
///
/// // Feed streaming data
/// let x = vec![1.0; 10];
/// let projected = tracker.project(&x);
/// assert_eq!(projected.len(), 3);
///
/// // Update subspace (pure reconstruction-based PAST)
/// tracker.update(&x, 0.0);
/// assert_eq!(tracker.n_samples(), 1);
/// ```
pub struct SubspaceTracker {
    /// Projection matrix W: d_in x rank (column-major, so w[col * d_in + row]).
    w: Vec<f64>,
    /// Inverse correlation matrix P: rank x rank (row-major).
    p: Vec<f64>,
    /// Forgetting factor (0.995--0.9999 typical).
    lambda: f64,
    /// Initial P diagonal value, kept for reset.
    delta: f64,
    /// Input dimension.
    d_in: usize,
    /// Projection rank (output dimension).
    rank: usize,
    /// Samples processed.
    n_samples: u64,
    /// RNG state for initialization.
    rng_state: u64,
    /// Original seed, kept for reset.
    seed: u64,
}

// ---------------------------------------------------------------------------
// Constructors and accessors
// ---------------------------------------------------------------------------

impl SubspaceTracker {
    /// Create a new PAST subspace tracker.
    ///
    /// # Arguments
    ///
    /// * `d_in` -- Input dimensionality.
    /// * `rank` -- Projection rank (output dimensionality). Must be <= d_in.
    /// * `lambda` -- Forgetting factor in (0, 1]. Typical range: 0.995--0.9999.
    /// * `delta` -- Initial P matrix diagonal (P = delta * I). Typical: 100.0.
    /// * `seed` -- PRNG seed for Xavier initialization. Must be non-zero.
    ///
    /// # Panics
    ///
    /// Panics if `rank > d_in`, `rank == 0`, `d_in == 0`, or `seed == 0`.
    pub fn new(d_in: usize, rank: usize, lambda: f64, delta: f64, seed: u64) -> Self {
        assert!(d_in > 0, "d_in must be positive");
        assert!(rank > 0, "rank must be positive");
        assert!(rank <= d_in, "rank must be <= d_in");
        assert!(seed != 0, "seed must be non-zero for xorshift64");

        let mut rng_state = seed;
        let w = xavier_init_col_major(d_in, rank, &mut rng_state);
        let p = delta_identity(rank, delta);

        Self {
            w,
            p,
            lambda,
            delta,
            d_in,
            rank,
            n_samples: 0,
            rng_state,
            seed,
        }
    }

    /// Project an input vector through the current subspace.
    ///
    /// Computes y = W^T * x, returning a `rank`-dimensional vector.
    ///
    /// # Panics
    ///
    /// Panics (in debug) if `x.len() != d_in`.
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.d_in, "input dimension mismatch");
        mat_t_vec_col_major(&self.w, x, self.d_in, self.rank)
    }

    /// Update the subspace estimate using the PAST algorithm.
    ///
    /// Performs a pure reconstruction-based update: the subspace W is adjusted
    /// to minimize ||x - W * W^T * x||^2 (streaming PCA). The scalar residual
    /// parameter is accepted for API compatibility but the core update uses
    /// reconstruction error, not prediction error.
    ///
    /// The inner model's readout (e.g. RLS) handles supervised adaptation.
    /// PAST handles unsupervised subspace discovery -- finding the directions
    /// of maximum variance in the input stream.
    ///
    /// # PAST update rule
    ///
    /// ```text
    /// y = W^T * x
    /// h = P * y
    /// g = h / (lambda + y^T * h)
    /// P = (P - g * h^T) / lambda
    /// e = x - W * y                  (reconstruction error)
    /// W = W + e * g^T                (rank-one subspace update)
    /// ```
    pub fn update(&mut self, x: &[f64], _residual: f64) {
        debug_assert_eq!(x.len(), self.d_in, "input dimension mismatch");
        self.past_update(x);
    }

    /// Update the subspace using a full error vector.
    ///
    /// This variant accepts a d_in-dimensional error signal for API
    /// compatibility, but performs the same pure reconstruction-based
    /// PAST update as [`update()`](Self::update). The subspace tracks
    /// directions of maximum variance, not prediction error.
    ///
    /// For use cases that require error-weighted subspace adaptation,
    /// the caller should implement custom scaling externally.
    pub fn update_with_error(&mut self, x: &[f64], _error_signal: &[f64]) {
        debug_assert_eq!(x.len(), self.d_in, "input dimension mismatch");
        self.past_update(x);
    }

    /// Reset the tracker to its initial state.
    ///
    /// Re-initializes W with Xavier random weights (same seed), resets
    /// P = delta * I, and clears the sample counter.
    pub fn reset(&mut self) {
        self.rng_state = self.seed;
        self.w = xavier_init_col_major(self.d_in, self.rank, &mut self.rng_state);
        self.p = delta_identity(self.rank, self.delta);
        self.n_samples = 0;
    }

    /// Return the projection rank (output dimensionality).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Return the input dimensionality.
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Return the number of samples processed.
    pub fn n_samples(&self) -> u64 {
        self.n_samples
    }
}

// ---------------------------------------------------------------------------
// Private PAST core
// ---------------------------------------------------------------------------

impl SubspaceTracker {
    /// Core PAST update on a (possibly scaled) input vector.
    fn past_update(&mut self, x: &[f64]) {
        let d = self.d_in;
        let r = self.rank;

        // y = W^T * x  (rank-dim projection)
        let y = mat_t_vec_col_major(&self.w, x, d, r);

        // h = P * y  (rank-dim intermediate)
        let h = mat_vec(&self.p, &y, r);

        // g = h / (lambda + y^T * h)  (gain vector)
        let denom = self.lambda + dot(&y, &h);
        let g: Vec<f64> = h.iter().map(|&hi| hi / denom).collect();

        // P = (P - g * h^T) / lambda
        outer_subtract_scaled(&mut self.p, &g, &h, self.lambda, r);

        // e = x - W * y  (reconstruction error in d_in-space)
        let wy = mat_vec_col_major(&self.w, &y, d, r);
        // e[i] = x[i] - wy[i]

        // W = W + e * g^T  (rank-one update, column-major)
        // W[row, col] += e[row] * g[col]
        for (col, &gc) in g.iter().enumerate() {
            let col_start = col * d;
            for row in 0..d {
                let e_row = x[row] - wy[row];
                self.w[col_start + row] += e_row * gc;
            }
        }

        self.n_samples += 1;
    }
}

// ---------------------------------------------------------------------------
// Supervised projection update
// ---------------------------------------------------------------------------

impl SubspaceTracker {
    /// Supervised rank-one update using prediction gradient.
    ///
    /// Updates W in the direction that reduces squared prediction error,
    /// given the inner model's readout weights (beta). The gradient is:
    ///
    /// ```text
    /// dL/dW = -2 * residual * outer(x, beta)
    /// W += lr * residual * outer(x, beta)
    /// ```
    ///
    /// Periodically re-orthogonalizes W columns via modified Gram-Schmidt
    /// (every 64 samples) to prevent column collapse.
    ///
    /// # Arguments
    ///
    /// * `x` -- normalized input vector (d_in dimensions)
    /// * `residual` -- prediction residual (target - prediction)
    /// * `beta` -- inner model's readout weight vector (rank dimensions)
    /// * `lr` -- learning rate for the gradient step
    ///
    /// # Complexity
    ///
    /// O(d_in * rank) per call; O(d_in * rank^2) amortized due to periodic
    /// re-orthogonalization.
    pub fn supervised_update(&mut self, x: &[f64], residual: f64, beta: &[f64], lr: f64) {
        debug_assert_eq!(x.len(), self.d_in);
        debug_assert_eq!(beta.len(), self.rank);

        let d = self.d_in;
        let r = self.rank;

        // W[:, j] += lr * residual * x[i] * beta[j]
        // W is column-major: w[col * d_in + row]
        for (j, &bj) in beta.iter().enumerate().take(r) {
            let scale = lr * residual * bj;
            let col_start = j * d;
            for (i, &xi) in x.iter().enumerate().take(d) {
                self.w[col_start + i] += scale * xi;
            }
        }

        self.n_samples += 1;

        // Re-orthogonalize every 64 samples to prevent column collapse.
        if self.n_samples % 64 == 0 {
            self.reorthogonalize();
        }
    }

    /// Modified Gram-Schmidt orthonormalization on W columns.
    ///
    /// Ensures the projection matrix columns remain orthonormal after
    /// repeated rank-one gradient updates. Without periodic re-orthogonalization,
    /// gradient updates can cause columns to collapse onto each other,
    /// reducing the effective rank of the projection.
    fn reorthogonalize(&mut self) {
        let d = self.d_in;
        let r = self.rank;
        for j in 0..r {
            // Subtract projections onto previous (already orthonormalized) columns.
            for k in 0..j {
                let mut dot_val = 0.0;
                for i in 0..d {
                    dot_val += self.w[j * d + i] * self.w[k * d + i];
                }
                for i in 0..d {
                    self.w[j * d + i] -= dot_val * self.w[k * d + i];
                }
            }
            // Normalize column j.
            let mut norm_sq = 0.0;
            for i in 0..d {
                norm_sq += self.w[j * d + i] * self.w[j * d + i];
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-12 {
                let inv_norm = 1.0 / norm;
                for i in 0..d {
                    self.w[j * d + i] *= inv_norm;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Initialization helpers
// ---------------------------------------------------------------------------

/// Xavier-uniform initialization for a d_in x rank column-major matrix.
///
/// scale = sqrt(2.0 / (d_in + rank)), values in [-scale, scale].
fn xavier_init_col_major(d_in: usize, rank: usize, rng: &mut u64) -> Vec<f64> {
    let scale = (2.0 / (d_in + rank) as f64).sqrt();
    let n = d_in * rank;
    let mut w = vec![0.0; n];
    for val in w.iter_mut() {
        // Map [0, 1) to [-scale, scale)
        let u = xorshift64_f64(rng);
        *val = (2.0 * u - 1.0) * scale;
    }
    w
}

/// Create a rank x rank identity matrix scaled by delta, row-major.
fn delta_identity(rank: usize, delta: f64) -> Vec<f64> {
    let mut p = vec![0.0; rank * rank];
    for i in 0..rank {
        p[i * rank + i] = delta;
    }
    p
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism() {
        let tracker_a = SubspaceTracker::new(8, 3, 0.998, 100.0, 42);
        let tracker_b = SubspaceTracker::new(8, 3, 0.998, 100.0, 42);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ya = tracker_a.project(&x);
        let yb = tracker_b.project(&x);

        assert_eq!(ya, yb, "same seed must produce identical projections");
    }

    #[test]
    fn convergence_on_known_subspace() {
        // Create 10-dim inputs that live in a 3-dim subspace.
        // The true subspace is spanned by 3 basis vectors; inputs are
        // random linear combinations of those basis vectors.
        let d_in = 10;
        let true_rank = 3;
        let mut tracker = SubspaceTracker::new(d_in, true_rank, 0.998, 100.0, 123);

        // Fixed basis vectors (3 vectors of length 10).
        let basis: [[f64; 10]; 3] = [
            [1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.0],
        ];

        let mut rng: u64 = 777;

        // Measure initial reconstruction error.
        let mut initial_error = 0.0;
        for _ in 0..50 {
            let x = random_subspace_sample(&basis, &mut rng, d_in);
            let y = tracker.project(&x);
            let recon = mat_vec_col_major(&tracker.w, &y, d_in, true_rank);
            let err: f64 = x
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            initial_error += err;
        }
        initial_error /= 50.0;

        // Train on 1000 samples.
        rng = 777; // reset to same sequence for fair comparison
        for _ in 0..1000 {
            let x = random_subspace_sample(&basis, &mut rng, d_in);
            tracker.update(&x, 0.0); // residual ignored (pure reconstruction-based PAST)
        }

        // Measure post-training reconstruction error.
        rng = 999; // different test samples
        let mut final_error = 0.0;
        for _ in 0..50 {
            let x = random_subspace_sample(&basis, &mut rng, d_in);
            let y = tracker.project(&x);
            let recon = mat_vec_col_major(&tracker.w, &y, d_in, true_rank);
            let err: f64 = x
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            final_error += err;
        }
        final_error /= 50.0;

        assert!(
            final_error < initial_error * 0.5,
            "reconstruction error should drop significantly after training: \
             initial={initial_error:.4}, final={final_error:.4}"
        );
    }

    #[test]
    fn reset_restores_initial_state() {
        let fresh = SubspaceTracker::new(6, 2, 0.995, 50.0, 99);

        let mut tracker = SubspaceTracker::new(6, 2, 0.995, 50.0, 99);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for _ in 0..20 {
            tracker.update(&x, 0.5);
        }
        assert!(tracker.n_samples() > 0);

        tracker.reset();
        assert_eq!(tracker.n_samples(), 0, "reset should clear sample count");

        let y_fresh = fresh.project(&x);
        let y_reset = tracker.project(&x);
        assert_eq!(
            y_fresh, y_reset,
            "reset tracker must match a fresh tracker with the same seed"
        );
    }

    #[test]
    fn dimensions_correct() {
        let d_in = 12;
        let rank = 4;
        let tracker = SubspaceTracker::new(d_in, rank, 0.999, 100.0, 7);

        let x = vec![0.5; d_in];
        let y = tracker.project(&x);
        assert_eq!(y.len(), rank, "projection output must have rank elements");
        assert_eq!(
            tracker.w.len(),
            d_in * rank,
            "W must have d_in * rank elements"
        );
        assert_eq!(tracker.p.len(), rank * rank, "P must have rank^2 elements");
        assert_eq!(tracker.d_in(), d_in);
        assert_eq!(tracker.rank(), rank);
    }

    #[test]
    fn update_increments_samples() {
        let mut tracker = SubspaceTracker::new(5, 2, 0.99, 10.0, 1);
        assert_eq!(tracker.n_samples(), 0);

        let x = vec![1.0; 5];
        tracker.update(&x, 1.0);
        assert_eq!(tracker.n_samples(), 1);

        tracker.update(&x, -0.5);
        assert_eq!(tracker.n_samples(), 2);

        tracker.update_with_error(&x, &[0.1, 0.2]);
        assert_eq!(tracker.n_samples(), 3);
    }

    #[test]
    fn supervised_update_finds_signal_direction() {
        // Data: y = x[0] + x[1], with noise dims x[2..6].
        // After supervised updates the subspace should capture the signal
        // direction [1, 1, 0, 0, 0, 0] / sqrt(2).
        //
        // We measure how well the *full* subspace captures the signal
        // direction (projection energy ratio) rather than checking a single
        // column, since the gradient can route signal into any column.
        let d_in = 6;
        let rank = 2;
        let mut tracker = SubspaceTracker::new(d_in, rank, 0.998, 100.0, 42);

        let mut rng: u64 = 1234;

        let signal_dir = [
            1.0 / 2.0_f64.sqrt(),
            1.0 / 2.0_f64.sqrt(),
            0.0,
            0.0,
            0.0,
            0.0,
        ];

        // Measure initial subspace capture: ||W * W^T * s||^2 / ||s||^2.
        let initial_capture = subspace_capture(&tracker, &signal_dir);

        // Run supervised updates: y = x[0] + x[1].
        // beta = [1.0, 0.0] means the readout only uses the first projected dim.
        let beta = [1.0, 0.0];
        for _ in 0..2000 {
            let mut x = vec![0.0; d_in];
            for xi in x.iter_mut() {
                *xi = xorshift64_f64(&mut rng) * 2.0 - 1.0;
            }
            let y = x[0] + x[1];
            let projected = tracker.project(&x);
            let pred = projected[0]; // beta = [1, 0] -> pred = projected[0]
            let residual = y - pred;
            tracker.supervised_update(&x, residual, &beta, 0.01);
        }

        let final_capture = subspace_capture(&tracker, &signal_dir);

        assert!(
            final_capture > initial_capture,
            "supervised update should improve signal capture: \
             initial={:.4}, final={:.4}",
            initial_capture,
            final_capture
        );
        assert!(
            final_capture > 0.5,
            "subspace should capture at least half the signal energy: {:.4}",
            final_capture
        );
    }

    #[test]
    fn supervised_update_increments_samples() {
        let mut tracker = SubspaceTracker::new(4, 2, 0.99, 10.0, 1);
        assert_eq!(tracker.n_samples(), 0);

        let x = vec![1.0; 4];
        let beta = vec![0.5, -0.3];
        tracker.supervised_update(&x, 1.0, &beta, 0.01);
        assert_eq!(tracker.n_samples(), 1);

        tracker.supervised_update(&x, -0.5, &beta, 0.01);
        assert_eq!(tracker.n_samples(), 2);
    }

    #[test]
    fn reorthogonalize_preserves_unit_columns() {
        // After 64 supervised updates, columns should still be approximately
        // unit-norm and orthogonal due to automatic re-orthogonalization.
        let d_in = 8;
        let rank = 3;
        let mut tracker = SubspaceTracker::new(d_in, rank, 0.998, 100.0, 77);

        let mut rng: u64 = 555;
        let beta = vec![1.0, 0.5, -0.3];
        for _ in 0..128 {
            let x: Vec<f64> = (0..d_in)
                .map(|_| xorshift64_f64(&mut rng) * 2.0 - 1.0)
                .collect();
            tracker.supervised_update(&x, 0.5, &beta, 0.01);
        }

        // Check that columns are approximately unit-norm.
        for j in 0..rank {
            let mut norm_sq = 0.0;
            for i in 0..d_in {
                let val = tracker.w[j * d_in + i];
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "column {} norm should be ~1.0, got {:.6}",
                j,
                norm
            );
        }

        // Check that columns are approximately orthogonal.
        for j in 0..rank {
            for k in (j + 1)..rank {
                let mut dot_val = 0.0;
                for i in 0..d_in {
                    dot_val += tracker.w[j * d_in + i] * tracker.w[k * d_in + i];
                }
                assert!(
                    dot_val.abs() < 0.05,
                    "columns {} and {} should be orthogonal, dot={:.6}",
                    j,
                    k,
                    dot_val
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Generate a random point that lies within the span of the given basis.
    fn random_subspace_sample(basis: &[[f64; 10]; 3], rng: &mut u64, d_in: usize) -> Vec<f64> {
        let mut x = vec![0.0; d_in];
        for b in basis.iter() {
            let coeff = xorshift64_f64(rng) * 2.0 - 1.0; // [-1, 1)
            for (xi, &bi) in x.iter_mut().zip(b.iter()) {
                *xi += coeff * bi;
            }
        }
        x
    }

    /// Fraction of signal energy captured by the subspace: ||W W^T s||^2 / ||s||^2.
    ///
    /// Returns a value in [0, 1]. A value of 1.0 means the signal direction
    /// lies entirely within the column space of W.
    fn subspace_capture(tracker: &SubspaceTracker, signal: &[f64]) -> f64 {
        let d = tracker.d_in;
        let r = tracker.rank;
        // y = W^T * s  (project into subspace)
        let y = mat_t_vec_col_major(&tracker.w, signal, d, r);
        // recon = W * y  (reconstruct back to d_in-space)
        let recon = mat_vec_col_major(&tracker.w, &y, d, r);
        let recon_sq: f64 = recon.iter().map(|v| v * v).sum();
        let signal_sq: f64 = signal.iter().map(|v| v * v).sum();
        if signal_sq < 1e-15 {
            0.0
        } else {
            recon_sq / signal_sq
        }
    }
}
