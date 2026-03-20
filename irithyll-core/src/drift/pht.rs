//! Page-Hinkley Test drift detector.
//!
//! The Page-Hinkley Test (PHT) is a sequential cumulative sum test that
//! monitors the cumulative deviation of observed values from the running mean.
//! It performs two-sided detection, signaling drift on both upward and downward
//! mean shifts.
//!
//! # Algorithm
//!
//! Two cumulative sums are tracked:
//!
//! **Upward shift detection:**
//! - `sum_up += x_i - mean_T - delta`
//! - `min_sum_up = min(min_sum_up, sum_up)`
//! - Drift when `sum_up - min_sum_up > lambda`
//!
//! **Downward shift detection:**
//! - `sum_down += mean_T - delta - x_i`
//! - `min_sum_down = min(min_sum_down, sum_down)`
//! - Drift when `sum_down - min_sum_down > lambda`
//!
//! # Parameters
//!
//! - `delta` -- magnitude tolerance; small values increase sensitivity (default 0.005)
//! - `lambda` -- detection threshold; smaller values trigger faster (default 50.0)
//!
//! # Example
//!
//! ```
//! use irithyll_core::drift::pht::PageHinkleyTest;
//! use irithyll_core::drift::DriftSignal;
//!
//! let mut pht = PageHinkleyTest::new();
//!
//! // Feed stationary data -- should remain stable.
//! for _ in 0..500 {
//!     assert_ne!(pht.update(1.0), DriftSignal::Drift);
//! }
//!
//! // Abrupt upward shift -- should eventually trigger drift.
//! let mut detected = false;
//! for _ in 0..500 {
//!     if pht.update(10.0) == DriftSignal::Drift {
//!         detected = true;
//!         break;
//!     }
//! }
//! assert!(detected);
//! ```

use super::DriftSignal;

/// Page-Hinkley Test for concept drift detection.
///
/// Monitors the cumulative deviation of values from the running mean.
/// Two-sided: detects both upward and downward shifts.
///
/// # Parameters
///
/// - `delta`: magnitude tolerance (allowable deviation, default 0.005)
/// - `lambda`: detection threshold (default 50.0)
///
/// A warning is signaled at `lambda * 0.5` before full drift is confirmed.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PageHinkleyTest {
    /// Allowed deviation before accumulation (default 0.005).
    delta: f64,
    /// Detection threshold (default 50.0).
    lambda: f64,
    /// Warning threshold (default lambda * 0.5).
    lambda_warning: f64,
    /// Running mean of observed values.
    running_mean: f64,
    /// Cumulative sum for upward shift detection.
    sum_up: f64,
    /// Minimum cumulative sum (up).
    min_sum_up: f64,
    /// Cumulative sum for downward shift detection.
    sum_down: f64,
    /// Minimum cumulative sum (down).
    min_sum_down: f64,
    /// Number of observations processed.
    count: u64,
}

impl PageHinkleyTest {
    /// Create a new `PageHinkleyTest` with default parameters.
    ///
    /// Defaults: `delta = 0.005`, `lambda = 50.0`.
    pub fn new() -> Self {
        Self::with_params(0.005, 50.0)
    }

    /// Create a new `PageHinkleyTest` with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `delta` - Magnitude tolerance. Smaller values increase sensitivity.
    /// * `lambda` - Detection threshold. Smaller values trigger drift sooner.
    pub fn with_params(delta: f64, lambda: f64) -> Self {
        Self {
            delta,
            lambda,
            lambda_warning: lambda * 0.5,
            running_mean: 0.0,
            sum_up: 0.0,
            min_sum_up: f64::MAX,
            sum_down: 0.0,
            min_sum_down: f64::MAX,
            count: 0,
        }
    }

    /// Returns the configured `delta` (magnitude tolerance).
    #[inline]
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Returns the configured `lambda` (detection threshold).
    #[inline]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Returns the number of observations processed so far.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Compute the upward Page-Hinkley statistic: `sum_up - min_sum_up`.
    #[inline]
    fn ph_up(&self) -> f64 {
        self.sum_up - self.min_sum_up
    }

    /// Compute the downward Page-Hinkley statistic: `sum_down - min_sum_down`.
    #[inline]
    fn ph_down(&self) -> f64 {
        self.sum_down - self.min_sum_down
    }

    /// Reset only the cumulative sums, preserving the running mean and count.
    ///
    /// Called after drift is detected so the detector can resume monitoring
    /// from the new distribution without losing the mean estimate.
    fn reset_sums(&mut self) {
        self.sum_up = 0.0;
        self.min_sum_up = f64::MAX;
        self.sum_down = 0.0;
        self.min_sum_down = f64::MAX;
    }

    /// Feed a new value and get the current drift signal.
    ///
    /// This is the core algorithm, usable without the `alloc` feature.
    /// When `alloc` is enabled, the `DriftDetector::update` trait method delegates here.
    pub fn update(&mut self, value: f64) -> DriftSignal {
        // Step 1: Increment observation count.
        self.count += 1;

        // Step 2: Update running mean incrementally (Welford-style).
        self.running_mean += (value - self.running_mean) / self.count as f64;

        // Step 3: Update cumulative sums.
        // Upward shift accumulator: positive when values exceed mean + delta.
        self.sum_up += value - self.running_mean - self.delta;
        self.min_sum_up = crate::math::fmin(self.min_sum_up, self.sum_up);

        // Downward shift accumulator: positive when values fall below mean - delta.
        self.sum_down += self.running_mean - self.delta - value;
        self.min_sum_down = crate::math::fmin(self.min_sum_down, self.sum_down);

        // Step 4: Check for drift (either direction exceeds lambda).
        let ph_up = self.ph_up();
        let ph_down = self.ph_down();

        if ph_up > self.lambda || ph_down > self.lambda {
            // Drift confirmed -- reset cumulative sums but keep the running mean
            // so monitoring can continue against the new distribution.
            self.reset_sums();
            return DriftSignal::Drift;
        }

        // Step 5: Check for warning (either direction exceeds warning threshold).
        if ph_up > self.lambda_warning || ph_down > self.lambda_warning {
            return DriftSignal::Warning;
        }

        // Step 6: No significant change.
        DriftSignal::Stable
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.running_mean = 0.0;
        self.sum_up = 0.0;
        self.min_sum_up = f64::MAX;
        self.sum_down = 0.0;
        self.min_sum_down = f64::MAX;
        self.count = 0;
    }

    /// Current estimated mean of the monitored stream.
    pub fn estimated_mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.running_mean
        }
    }
}

impl Default for PageHinkleyTest {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DriftDetector trait impl (requires alloc for Box)
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl super::DriftDetector for PageHinkleyTest {
    fn update(&mut self, value: f64) -> DriftSignal {
        // Delegate to the inherent method so the algorithm is usable without alloc.
        PageHinkleyTest::update(self, value)
    }

    fn reset(&mut self) {
        PageHinkleyTest::reset(self);
    }

    fn clone_fresh(&self) -> alloc::boxed::Box<dyn super::DriftDetector> {
        alloc::boxed::Box::new(Self::with_params(self.delta, self.lambda))
    }

    fn clone_boxed(&self) -> alloc::boxed::Box<dyn super::DriftDetector> {
        alloc::boxed::Box::new(self.clone())
    }

    fn estimated_mean(&self) -> f64 {
        PageHinkleyTest::estimated_mean(self)
    }

    fn serialize_state(&self) -> Option<super::DriftDetectorState> {
        Some(super::DriftDetectorState::PageHinkley {
            running_mean: self.running_mean,
            sum_up: self.sum_up,
            min_sum_up: self.min_sum_up,
            sum_down: self.sum_down,
            min_sum_down: self.min_sum_down,
            count: self.count,
        })
    }

    fn restore_state(&mut self, state: &super::DriftDetectorState) -> bool {
        // Allow irrefutable pattern: more variants (Adwin, Ddm) will be added.
        #[allow(irrefutable_let_patterns)]
        if let super::DriftDetectorState::PageHinkley {
            running_mean,
            sum_up,
            min_sum_up,
            sum_down,
            min_sum_down,
            count,
        } = state
        {
            self.running_mean = *running_mean;
            self.sum_up = *sum_up;
            self.min_sum_up = *min_sum_up;
            self.sum_down = *sum_down;
            self.min_sum_down = *min_sum_down;
            self.count = *count;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: produce a deterministic "noisy" value around a center.
    /// Uses a simple linear congruential generator seeded per call index
    /// to avoid pulling in rand as a dev-dependency.
    fn noisy_stream(center: f64, amplitude: f64, count: usize, seed: u64) -> alloc::vec::Vec<f64> {
        let mut values = alloc::vec::Vec::with_capacity(count);
        let mut state = seed;
        for _ in 0..count {
            // LCG: Numerical Recipes constants
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // Map to [-1.0, 1.0)
            let frac = ((state >> 33) as f64) / (u32::MAX as f64 / 2.0) - 1.0;
            values.push(center + amplitude * frac);
        }
        values
    }

    #[test]
    fn stationary_stream_no_drift() {
        let mut pht = PageHinkleyTest::new();
        let values = noisy_stream(5.0, 0.1, 5000, 42);

        for v in &values {
            let signal = pht.update(*v);
            assert_ne!(
                signal,
                DriftSignal::Drift,
                "stationary stream should never trigger Drift"
            );
        }
    }

    #[test]
    fn abrupt_upward_shift_detected() {
        let mut pht = PageHinkleyTest::new();

        // Phase 1: stationary near 0.
        let stable = noisy_stream(0.0, 0.01, 1000, 100);
        for v in &stable {
            pht.update(*v);
        }

        // Phase 2: abrupt shift to 10.
        let shifted = noisy_stream(10.0, 0.01, 1000, 200);
        let mut drift_detected = false;
        for v in &shifted {
            if pht.update(*v) == DriftSignal::Drift {
                drift_detected = true;
                break;
            }
        }

        assert!(
            drift_detected,
            "upward shift from 0 to 10 must trigger Drift"
        );
    }

    #[test]
    fn abrupt_downward_shift_detected() {
        let mut pht = PageHinkleyTest::new();

        // Phase 1: stationary near 10.
        let stable = noisy_stream(10.0, 0.01, 1000, 300);
        for v in &stable {
            pht.update(*v);
        }

        // Phase 2: abrupt shift to 0.
        let shifted = noisy_stream(0.0, 0.01, 1000, 400);
        let mut drift_detected = false;
        for v in &shifted {
            if pht.update(*v) == DriftSignal::Drift {
                drift_detected = true;
                break;
            }
        }

        assert!(
            drift_detected,
            "downward shift from 10 to 0 must trigger Drift"
        );
    }

    #[test]
    fn estimated_mean_tracks_true_mean() {
        let mut pht = PageHinkleyTest::new();
        let center = 7.5;
        let values = noisy_stream(center, 0.01, 2000, 500);

        for v in &values {
            pht.update(*v);
        }

        let estimated = pht.estimated_mean();
        let error = crate::math::abs(estimated - center);
        assert!(
            error < 0.1,
            "estimated mean {estimated} should be close to true mean {center}, error={error}"
        );
    }

    #[test]
    fn estimated_mean_returns_zero_when_empty() {
        let pht = PageHinkleyTest::new();
        assert_eq!(
            pht.estimated_mean(),
            0.0,
            "empty detector should report mean=0.0"
        );
    }

    #[test]
    fn reset_clears_all_state() {
        let mut pht = PageHinkleyTest::new();

        // Feed some data.
        let values = noisy_stream(5.0, 0.1, 500, 600);
        for v in &values {
            pht.update(*v);
        }
        assert!(pht.count() > 0);
        assert!(pht.estimated_mean() != 0.0);

        // Reset.
        pht.reset();

        assert_eq!(pht.count(), 0);
        assert_eq!(pht.estimated_mean(), 0.0);
        assert_eq!(pht.sum_up, 0.0);
        assert_eq!(pht.sum_down, 0.0);
        assert_eq!(pht.min_sum_up, f64::MAX);
        assert_eq!(pht.min_sum_down, f64::MAX);
        assert_eq!(pht.running_mean, 0.0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn clone_fresh_returns_clean_detector_with_same_params() {
        use super::super::DriftDetector;

        let mut pht = PageHinkleyTest::with_params(0.01, 100.0);

        // Pollute state.
        for v in noisy_stream(3.0, 0.5, 200, 700) {
            pht.update(v);
        }

        let fresh = DriftDetector::clone_fresh(&pht);

        // Fresh detector should have zero state.
        assert_eq!(fresh.estimated_mean(), 0.0);

        // But same sensitivity: feed same data, should produce same signals.
        // We just verify it compiles and returns a functional detector.
        let mut fresh = DriftDetector::clone_fresh(&pht);
        let signal = fresh.update(1.0);
        assert_eq!(
            signal,
            DriftSignal::Stable,
            "single sample on fresh detector should be Stable"
        );
    }

    #[test]
    fn custom_params_stored_correctly() {
        let pht = PageHinkleyTest::with_params(0.1, 200.0);
        assert_eq!(pht.delta(), 0.1);
        assert_eq!(pht.lambda(), 200.0);
        assert_eq!(pht.lambda_warning, 100.0); // lambda * 0.5
    }

    #[test]
    fn default_params_match_new() {
        let from_new = PageHinkleyTest::new();
        let from_default = PageHinkleyTest::default();

        assert_eq!(from_new.delta(), from_default.delta());
        assert_eq!(from_new.lambda(), from_default.lambda());
        assert_eq!(from_new.count(), from_default.count());
    }

    #[test]
    fn warmup_no_drift_on_extreme_early_values() {
        // A constant stream of extreme (but identical) values should never
        // trigger drift regardless of magnitude, because there is no actual
        // shift. The cumulative sums stay near zero when every sample equals
        // the running mean.
        let mut pht = PageHinkleyTest::new();

        // Constant extreme value -- large but stable.
        for i in 0..20 {
            let signal = pht.update(1_000_000.0);
            assert_ne!(
                signal,
                DriftSignal::Drift,
                "constant extreme stream should not trigger Drift at sample {i}"
            );
        }

        // Also verify a small number of moderately noisy early samples
        // do not immediately trigger drift (genuine warmup scenario).
        let mut pht2 = PageHinkleyTest::new();
        let early_values = [100.0, 102.0, 98.0, 105.0, 95.0, 101.0, 99.0, 103.0];
        for (i, &v) in early_values.iter().enumerate() {
            let signal = pht2.update(v);
            assert_ne!(
                signal,
                DriftSignal::Drift,
                "early noisy samples should not trigger Drift at sample {i}"
            );
        }
    }

    #[test]
    fn warning_fires_before_drift() {
        let mut pht = PageHinkleyTest::with_params(0.005, 50.0);

        // Establish baseline.
        for v in noisy_stream(0.0, 0.001, 500, 800) {
            pht.update(v);
        }

        // Shift upward -- we expect Warning before Drift.
        let mut saw_warning = false;
        let mut saw_drift = false;
        for v in noisy_stream(5.0, 0.001, 1000, 900) {
            match pht.update(v) {
                DriftSignal::Warning => {
                    if !saw_drift {
                        saw_warning = true;
                    }
                }
                DriftSignal::Drift => {
                    saw_drift = true;
                    break;
                }
                DriftSignal::Stable => {}
            }
        }

        assert!(saw_warning, "Warning should fire before Drift");
        assert!(saw_drift, "Drift should eventually fire");
    }

    #[test]
    fn drift_resets_sums_not_mean() {
        let mut pht = PageHinkleyTest::new();

        // Build up mean around 0.
        for v in noisy_stream(0.0, 0.01, 500, 1000) {
            pht.update(v);
        }
        let count_before = pht.count();

        // Shift to trigger drift.
        let shifted = noisy_stream(20.0, 0.01, 500, 1100);
        let mut drifted = false;
        for v in &shifted {
            if pht.update(*v) == DriftSignal::Drift {
                drifted = true;
                break;
            }
        }
        assert!(drifted, "should detect drift");

        // After drift, count should still be incrementing (mean preserved).
        assert!(pht.count() > count_before);
        // Sums should be reset to initial state.
        assert_eq!(pht.sum_up, 0.0);
        assert_eq!(pht.sum_down, 0.0);
        assert_eq!(pht.min_sum_up, f64::MAX);
        assert_eq!(pht.min_sum_down, f64::MAX);
    }

    #[test]
    fn higher_lambda_needs_bigger_shift() {
        // With a very high lambda, small shifts should NOT trigger drift.
        let mut pht = PageHinkleyTest::with_params(0.005, 5000.0);

        for v in noisy_stream(0.0, 0.01, 500, 1200) {
            pht.update(v);
        }

        // Moderate shift: 0 -> 2. With lambda=5000, this should NOT drift
        // in a reasonable number of samples.
        let mut drifted = false;
        for v in noisy_stream(2.0, 0.01, 500, 1300) {
            if pht.update(v) == DriftSignal::Drift {
                drifted = true;
                break;
            }
        }

        assert!(
            !drifted,
            "high lambda (5000) should not trigger on small shift within 500 samples"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn serialize_restore_roundtrip() {
        use super::super::DriftDetector;

        let mut pht = PageHinkleyTest::new();

        // Feed data to build up state.
        for v in noisy_stream(5.0, 0.1, 200, 1400) {
            pht.update(v);
        }

        // Serialize.
        let state = DriftDetector::serialize_state(&pht);
        assert!(state.is_some(), "PHT should support state serialization");

        // Restore into a fresh detector.
        let mut pht2 = PageHinkleyTest::new();
        let restored = DriftDetector::restore_state(&mut pht2, state.as_ref().unwrap());
        assert!(restored, "restore should succeed for PageHinkley state");

        // Verify state matches.
        assert_eq!(pht.count(), pht2.count());
        assert_eq!(pht.estimated_mean(), pht2.estimated_mean());
    }
}
