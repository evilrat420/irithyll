//! Online input normalizer based on Welford's algorithm.
//!
//! Maintains per-feature running mean and variance in a single pass with
//! numerically stable updates. Produces standardized inputs `(x − μ) / σ`.

#[cfg(feature = "alloc")]
use crate::math;

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Per-dimension Welford online normalizer.
///
/// Applies Welford's update rule to track mean and M2 (the unnormalized
/// sum of squared deviations) per input dimension. Variance uses Bessel
/// correction: `s² = M2 / (n − 1)`, which is unbiased for the true
/// population variance.
///
/// The normalized output is `(x − μ) / σ` where `σ = sqrt(s² + ε)` and
/// `ε = 1e-8` prevents division by zero on cold starts.
///
/// # no_std
///
/// Requires the `alloc` feature. All allocations happen in `new`.
#[cfg(feature = "alloc")]
pub struct WelfordNormalizer {
    /// Per-feature running mean.
    mean: Vec<f64>,
    /// Per-feature running M2 (unnormalized sum of squared deviations).
    m2: Vec<f64>,
    /// Total sample count (shared across dims — all dims updated together).
    count: u64,
}

#[cfg(feature = "alloc")]
impl WelfordNormalizer {
    /// Create a new normalizer for inputs of dimension `d`.
    ///
    /// All accumulators are initialized to zero. The first call to
    /// [`update`](Self::update) or [`update_and_normalize`](Self::update_and_normalize)
    /// seeds the mean from the first sample, which avoids early zero-mean bias.
    pub fn new(d: usize) -> Self {
        Self {
            mean: vec![0.0; d],
            m2: vec![0.0; d],
            count: 0,
        }
    }

    /// Number of samples seen so far.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Update the running statistics with a new input vector `x`.
    ///
    /// Uses the Welford one-pass formula:
    /// ```text
    /// delta  = x_i − mean_i
    /// mean_i += delta / n
    /// M2_i  += delta * (x_i − new_mean_i)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `x.len() != self.mean.len()`.
    pub fn update(&mut self, x: &[f64]) {
        debug_assert_eq!(
            x.len(),
            self.mean.len(),
            "WelfordNormalizer: input length {} != expected {}",
            x.len(),
            self.mean.len()
        );
        self.count += 1;
        let n = self.count as f64;
        for ((m, m2), &xi) in self.mean.iter_mut().zip(self.m2.iter_mut()).zip(x.iter()) {
            let delta = xi - *m;
            *m += delta / n;
            let delta2 = xi - *m;
            *m2 += delta * delta2;
        }
    }

    /// Write the normalized version of `x` into `out`.
    ///
    /// If fewer than 2 samples have been seen, variance is zero and
    /// `out` is filled with zeros (cold-start convention: centre at origin,
    /// no scaling).
    ///
    /// # Panics
    ///
    /// Panics in debug mode if lengths mismatch.
    pub fn normalize(&self, x: &[f64], out: &mut [f64]) {
        debug_assert_eq!(x.len(), self.mean.len());
        debug_assert_eq!(out.len(), self.mean.len());
        const EPS: f64 = 1e-8;
        if self.count < 2 {
            for o in out.iter_mut() {
                *o = 0.0;
            }
            return;
        }
        let n_minus_1 = (self.count - 1) as f64;
        for (((m, m2), &xi), o) in self
            .mean
            .iter()
            .zip(self.m2.iter())
            .zip(x.iter())
            .zip(out.iter_mut())
        {
            // Bessel-corrected variance then std.
            let var = *m2 / n_minus_1;
            let std = math::sqrt(var + EPS);
            *o = (xi - m) / std;
        }
    }

    /// Update statistics with `x`, then write the normalized `x` into `out`.
    ///
    /// Equivalent to calling [`update`](Self::update) followed by
    /// [`normalize`](Self::normalize), but avoids re-iterating the vectors.
    pub fn update_and_normalize(&mut self, x: &[f64], out: &mut [f64]) {
        debug_assert_eq!(x.len(), self.mean.len());
        debug_assert_eq!(out.len(), self.mean.len());
        const EPS: f64 = 1e-8;
        self.count += 1;
        let n = self.count as f64;
        for (((m, m2), &xi), o) in self
            .mean
            .iter_mut()
            .zip(self.m2.iter_mut())
            .zip(x.iter())
            .zip(out.iter_mut())
        {
            let delta = xi - *m;
            *m += delta / n;
            let delta2 = xi - *m;
            *m2 += delta * delta2;

            if self.count < 2 {
                *o = 0.0;
            } else {
                let n_minus_1 = (self.count - 1) as f64;
                let var = *m2 / n_minus_1;
                let std = math::sqrt(var + EPS);
                *o = (xi - *m) / std;
            }
        }
    }

    /// Reset all accumulators to zero. Equivalent to `*self = WelfordNormalizer::new(d)`.
    pub fn reset(&mut self) {
        for m in self.mean.iter_mut() {
            *m = 0.0;
        }
        for m2 in self.m2.iter_mut() {
            *m2 = 0.0;
        }
        self.count = 0;
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    /// Zero-mean, unit-variance data normalizes close to N(0,1) parameters.
    #[test]
    fn welford_known_mean_and_variance() {
        // Feed xs = [1, 2, 3, 4, 5] — population: mean=3, var=2, std=sqrt(2).
        // Welford (Bessel) sample var = 10/4 = 2.5, sample std = sqrt(2.5).
        let xs: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let mut norm = WelfordNormalizer::new(1);
        for &x in xs {
            norm.update(&[x]);
        }
        // After 5 samples: mean = 3.0, M2 = 10, Bessel var = 2.5.
        assert!((norm.mean[0] - 3.0).abs() < 1e-12, "mean should be 3.0");
        let bessel_var = norm.m2[0] / 4.0;
        assert!((bessel_var - 2.5).abs() < 1e-12, "Bessel var should be 2.5");
    }

    /// Normalized output has mean ~ 0 and std ~ 1 over the training set.
    #[test]
    fn normalized_output_is_standardized() {
        let xs: &[f64] = &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut norm = WelfordNormalizer::new(1);
        for &x in xs {
            norm.update(&[x]);
        }
        // Check normalized mean is close to 0.
        let mut out = [0.0f64; 1];
        let mut sum = 0.0;
        for &x in xs {
            norm.normalize(&[x], &mut out);
            sum += out[0];
        }
        let normalized_mean = sum / xs.len() as f64;
        assert!(
            normalized_mean.abs() < 0.1,
            "normalized mean {normalized_mean} should be near 0"
        );
    }

    /// Cold start (< 2 samples) returns zeros, not NaN or panic.
    #[test]
    fn cold_start_returns_zeros() {
        let mut norm = WelfordNormalizer::new(3);
        let mut out = [0.0f64; 3];
        norm.normalize(&[1.0, 2.0, 3.0], &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0]);

        norm.update(&[1.0, 2.0, 3.0]); // count = 1, still < 2
        norm.normalize(&[1.0, 2.0, 3.0], &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    /// Reset clears state back to cold-start.
    #[test]
    fn reset_clears_all_state() {
        let mut norm = WelfordNormalizer::new(2);
        for i in 0..10 {
            norm.update(&[i as f64, i as f64 * 2.0]);
        }
        norm.reset();
        assert_eq!(norm.count(), 0);
        assert_eq!(norm.mean[0], 0.0);
        assert_eq!(norm.m2[0], 0.0);
    }

    /// update_and_normalize gives same result as separate update + normalize.
    #[test]
    fn update_and_normalize_matches_separate_calls() {
        let inputs: &[f64] = &[10.0, 20.0, 30.0, 40.0, 50.0];
        let mut norm_a = WelfordNormalizer::new(1);
        let mut norm_b = WelfordNormalizer::new(1);

        let mut out_a = [0.0f64; 1];
        let mut out_b = [0.0f64; 1];

        for &x in inputs {
            norm_a.update_and_normalize(&[x], &mut out_a);

            norm_b.update(&[x]);
            norm_b.normalize(&[x], &mut out_b);

            assert!(
                (out_a[0] - out_b[0]).abs() < 1e-10,
                "combined and separate paths diverged at x={x}: combined={}, separate={}",
                out_a[0],
                out_b[0]
            );
        }
    }
}
