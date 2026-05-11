//! Bounded-readout RLS wrapper: clips predictions to a caller-specified range.
//!
//! Every model that uses an RLS readout head needs to clip wild predictions
//! during early streaming (before the P matrix has converged). Today this is
//! open-coded in each model. This wrapper centralises the pattern.
//!
//! # Usage
//!
//! ```rust
//! use irithyll::learners::rls::RecursiveLeastSquares;
//! use irithyll::learners::bounded_rls::BoundedRls;
//!
//! let rls = RecursiveLeastSquares::new(1.0);
//! let bounded = BoundedRls::new(rls);
//!
//! // Predictions outside [lo, hi] are clipped to the boundary.
//! let _pred = bounded.predict_clipped(&[1.0, 2.0], -10.0, 10.0);
//! ```

use crate::learner::StreamingLearner;
use crate::learners::rls::RecursiveLeastSquares;

/// Thin wrapper around [`RecursiveLeastSquares`] that exposes a
/// `predict_clipped(x, lo, hi)` helper.
///
/// The wrapper delegates all `StreamingLearner` methods to the inner RLS —
/// it is a pure readout-ergonomics adapter, not a separate learner.
pub struct BoundedRls {
    inner: RecursiveLeastSquares,
}

impl BoundedRls {
    /// Create a new `BoundedRls` wrapping the given `RecursiveLeastSquares`.
    pub fn new(rls: RecursiveLeastSquares) -> Self {
        Self { inner: rls }
    }

    /// Consume the wrapper and return the inner [`RecursiveLeastSquares`].
    pub fn into_inner(self) -> RecursiveLeastSquares {
        self.inner
    }

    /// Borrow the inner [`RecursiveLeastSquares`].
    pub fn inner(&self) -> &RecursiveLeastSquares {
        &self.inner
    }

    /// Mutable borrow of the inner [`RecursiveLeastSquares`].
    pub fn inner_mut(&mut self) -> &mut RecursiveLeastSquares {
        &mut self.inner
    }

    /// Predict and clip the result to `[lo, hi]`.
    ///
    /// Equivalent to `rls.predict(features).clamp(lo, hi)`, but named
    /// explicitly so call sites communicate "this is a deliberate bound step."
    ///
    /// If the raw prediction is within `[lo, hi]`, it passes through unchanged.
    ///
    /// # Arguments
    ///
    /// - `features` — input feature vector for the current step.
    /// - `lo` — lower bound (inclusive).
    /// - `hi` — upper bound (inclusive).
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `lo > hi`.
    #[inline]
    pub fn predict_clipped(&self, features: &[f64], lo: f64, hi: f64) -> f64 {
        debug_assert!(lo <= hi, "BoundedRls: lo ({lo}) must be <= hi ({hi})");
        let raw = self.inner.predict(features);
        raw.clamp(lo, hi)
    }
}

// Delegate all StreamingLearner methods so BoundedRls can be used wherever
// RecursiveLeastSquares is expected.
impl StreamingLearner for BoundedRls {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        self.inner.train_one(features, target, weight);
    }

    fn predict(&self, features: &[f64]) -> f64 {
        self.inner.predict(features)
    }

    fn n_samples_seen(&self) -> u64 {
        self.inner.n_samples_seen()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Predictions within bounds pass through unchanged.
    #[test]
    fn in_range_prediction_passes_through() {
        let mut bounded = BoundedRls::new(RecursiveLeastSquares::new(1.0));
        // Train on y = 2*x.
        for i in 0..100 {
            let x = i as f64 * 0.1;
            bounded.train(&[x], 2.0 * x);
        }
        // Predict at x=1.0 → ~2.0, well within [-10, 10].
        let pred = bounded.predict_clipped(&[1.0], -10.0, 10.0);
        assert!(
            (pred - 2.0).abs() < 0.5,
            "in-range prediction should be near 2.0, got {pred}"
        );
        assert!(
            (-10.0..=10.0).contains(&pred),
            "prediction {pred} must be within [-10, 10]"
        );
    }

    /// Predictions exceeding hi are clipped to hi.
    #[test]
    fn exceeds_hi_clipped_to_hi() {
        let mut bounded = BoundedRls::new(RecursiveLeastSquares::new(1.0));
        for i in 0..50 {
            let x = i as f64;
            bounded.train(&[x], 1000.0 * x);
        }
        let pred = bounded.predict_clipped(&[100.0], -1.0, 1.0);
        assert!(
            (pred - 1.0).abs() < 1e-9,
            "expected hi clip = 1.0, got {pred}"
        );
    }

    /// Predictions below lo are clipped to lo.
    #[test]
    fn below_lo_clipped_to_lo() {
        let mut bounded = BoundedRls::new(RecursiveLeastSquares::new(1.0));
        for i in 0..50 {
            let x = i as f64;
            bounded.train(&[x], -1000.0 * x);
        }
        let pred = bounded.predict_clipped(&[100.0], -1.0, 1.0);
        assert!(
            (pred - (-1.0)).abs() < 1e-9,
            "expected lo clip = -1.0, got {pred}"
        );
    }

    /// Cold-start (no training) returns 0.0, within any symmetric bound.
    #[test]
    fn cold_start_predict_clipped_returns_zero_within_bounds() {
        let bounded = BoundedRls::new(RecursiveLeastSquares::new(1.0));
        let pred = bounded.predict_clipped(&[1.0, 2.0], -5.0, 5.0);
        assert_eq!(pred, 0.0, "cold-start RLS predicts 0.0");
    }

    /// n_samples_seen reflects inner RLS training count.
    #[test]
    fn n_samples_seen_delegates_to_inner() {
        let mut bounded = BoundedRls::new(RecursiveLeastSquares::new(1.0));
        assert_eq!(bounded.n_samples_seen(), 0);
        for i in 0..7 {
            bounded.train(&[i as f64], i as f64);
        }
        assert_eq!(bounded.n_samples_seen(), 7);
    }
}
