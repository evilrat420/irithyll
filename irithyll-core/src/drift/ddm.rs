//! DDM (Drift Detection Method) detector.
//!
//! DDM monitors the running error rate and standard deviation of a stream of
//! error values. When the current `mean + std` exceeds the historical minimum
//! by a configurable number of standard deviations, the detector signals
//! [`DriftSignal::Warning`] (at `warning_level * min_std`) or
//! [`DriftSignal::Drift`] (at `drift_level * min_std`).
//!
//! Originally designed for binary error indicators (0/1), this implementation
//! generalises to continuous error magnitudes by using Welford's online
//! algorithm for running mean and population standard deviation.
//!
//! # References
//!
//! Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004).
//! *Learning with drift detection.* In Advances in Artificial Intelligence --
//! SBIA 2004, pp. 286--295.

use super::{DriftDetector, DriftSignal};

/// DDM (Drift Detection Method) for concept drift detection.
///
/// Monitors running error rate and standard deviation.
/// Signals warning when error exceeds minimum by 2 sigma, drift at 3 sigma.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Ddm {
    /// Warning threshold multiplier (default 2.0).
    warning_level: f64,
    /// Drift threshold multiplier (default 3.0).
    drift_level: f64,
    /// Minimum observations before detection activates (default 30).
    min_instances: u64,

    // --- Welford running statistics ---
    /// Running mean of observed values.
    mean: f64,
    /// Running M2 aggregate for variance (Welford).
    m2: f64,
    /// Number of observations seen.
    count: u64,

    // --- Minimum tracking ---
    /// Minimum `mean + std` observed so far.
    min_p_plus_s: f64,
    /// Standard deviation at the point where `min_p_plus_s` was recorded.
    min_s: f64,
}

impl Ddm {
    /// Create a new DDM detector with default parameters.
    ///
    /// Defaults: `warning_level = 2.0`, `drift_level = 3.0`,
    /// `min_instances = 30`.
    pub fn new() -> Self {
        Self::with_params(2.0, 3.0, 30)
    }

    /// Create a DDM detector with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `warning_level` -- multiplier of `min_s` above `min_p_plus_s` to
    ///   trigger a warning (typically 2.0).
    /// * `drift_level` -- multiplier of `min_s` above `min_p_plus_s` to
    ///   trigger drift (typically 3.0).
    /// * `min_instances` -- number of observations required before the
    ///   detector can emit Warning or Drift signals.
    pub fn with_params(warning_level: f64, drift_level: f64, min_instances: u64) -> Self {
        Self {
            warning_level,
            drift_level,
            min_instances,
            mean: 0.0,
            m2: 0.0,
            count: 0,
            min_p_plus_s: f64::MAX,
            min_s: f64::MAX,
        }
    }

    /// Return the current warning-level multiplier.
    #[inline]
    pub fn warning_level(&self) -> f64 {
        self.warning_level
    }

    /// Return the current drift-level multiplier.
    #[inline]
    pub fn drift_level(&self) -> f64 {
        self.drift_level
    }

    /// Return the minimum-instances warmup threshold.
    #[inline]
    pub fn min_instances(&self) -> u64 {
        self.min_instances
    }

    /// Current population standard deviation of the observed stream.
    #[inline]
    pub fn std_dev(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            crate::math::sqrt(self.m2 / self.count as f64)
        }
    }

    /// Minimum `mean + std` observed so far.
    #[inline]
    pub fn min_p_plus_s(&self) -> f64 {
        self.min_p_plus_s
    }

    /// Reset the running statistics (mean, m2, count) while preserving the
    /// minimum tracking state. Called internally after drift is confirmed.
    fn reset_running_stats(&mut self) {
        self.mean = 0.0;
        self.m2 = 0.0;
        self.count = 0;
    }
}

impl Default for Ddm {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for Ddm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Ddm(warn={}, drift={}, min_inst={}, count={})",
            self.warning_level, self.drift_level, self.min_instances, self.count
        )
    }
}

impl DriftDetector for Ddm {
    fn update(&mut self, value: f64) -> DriftSignal {
        // 1. Increment count.
        self.count += 1;
        let n = self.count as f64;

        // 2. Welford online update.
        let delta = value - self.mean;
        self.mean += delta / n;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        // 3. Population standard deviation.
        let std = crate::math::sqrt(self.m2 / n);

        // 4. Current metric.
        let p_plus_s = self.mean + std;

        // 5. Warmup guard -- return Stable and skip minimum / threshold
        //    updates until running statistics have stabilised.
        if self.count <= self.min_instances {
            return DriftSignal::Stable;
        }

        // 6. Update minimums (only after warmup so the baseline is stable).
        if p_plus_s < self.min_p_plus_s {
            self.min_p_plus_s = p_plus_s;
            self.min_s = std;
        }

        // 7. Drift check (must come before warning -- drift is more severe).
        if p_plus_s >= self.min_p_plus_s + self.drift_level * self.min_s {
            self.reset_running_stats();
            return DriftSignal::Drift;
        }

        // 8. Warning check.
        if p_plus_s >= self.min_p_plus_s + self.warning_level * self.min_s {
            return DriftSignal::Warning;
        }

        // 9. No change.
        DriftSignal::Stable
    }

    fn reset(&mut self) {
        self.mean = 0.0;
        self.m2 = 0.0;
        self.count = 0;
        self.min_p_plus_s = f64::MAX;
        self.min_s = f64::MAX;
    }

    #[cfg(feature = "alloc")]
    fn clone_fresh(&self) -> alloc::boxed::Box<dyn DriftDetector> {
        alloc::boxed::Box::new(Self::with_params(
            self.warning_level,
            self.drift_level,
            self.min_instances,
        ))
    }

    #[cfg(feature = "alloc")]
    fn clone_boxed(&self) -> alloc::boxed::Box<dyn DriftDetector> {
        alloc::boxed::Box::new(self.clone())
    }

    fn estimated_mean(&self) -> f64 {
        self.mean
    }

    #[cfg(feature = "alloc")]
    fn serialize_state(&self) -> Option<super::DriftDetectorState> {
        Some(super::DriftDetectorState::Ddm {
            mean: self.mean,
            m2: self.m2,
            count: self.count,
            min_p_plus_s: self.min_p_plus_s,
            min_s: self.min_s,
        })
    }

    #[cfg(feature = "alloc")]
    fn restore_state(&mut self, state: &super::DriftDetectorState) -> bool {
        if let super::DriftDetectorState::Ddm {
            mean,
            m2,
            count,
            min_p_plus_s,
            min_s,
        } = state
        {
            self.mean = *mean;
            self.m2 = *m2;
            self.count = *count;
            self.min_p_plus_s = *min_p_plus_s;
            self.min_s = *min_s;
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
    extern crate alloc;
    use alloc::vec::Vec;

    use super::super::DriftSignal;
    use super::*;

    /// Helper: feed a slice of values, collect all signals.
    fn feed(ddm: &mut Ddm, values: &[f64]) -> Vec<DriftSignal> {
        values.iter().map(|&v| ddm.update(v)).collect()
    }

    /// Helper: generate `n` values centred around `centre` with deterministic
    /// jitter derived from the index (no randomness needed).
    fn generate_values(centre: f64, jitter: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                // Deterministic oscillation: sin-based spread.
                let t = i as f64;
                centre + jitter * crate::math::sin(t * 0.7)
            })
            .collect()
    }

    // 1. Stationary low error -- no drift expected.
    #[test]
    fn stationary_low_error_no_drift() {
        let mut ddm = Ddm::new();
        let values = generate_values(0.1, 0.02, 5000);
        let signals = feed(&mut ddm, &values);

        let drift_count = signals.iter().filter(|&&s| s == DriftSignal::Drift).count();
        assert_eq!(
            drift_count, 0,
            "stationary low error should produce no drift"
        );
    }

    // 2. Error rate increase -- drift detected.
    #[test]
    fn error_rate_increase_detects_drift() {
        let mut ddm = Ddm::new();

        // Phase 1: low error rate.
        let low = generate_values(0.1, 0.01, 2000);
        feed(&mut ddm, &low);

        // Phase 2: high error rate -- should trigger drift.
        let high = generate_values(0.8, 0.01, 2000);
        let signals = feed(&mut ddm, &high);

        let drift_count = signals.iter().filter(|&&s| s == DriftSignal::Drift).count();
        assert!(
            drift_count >= 1,
            "sudden error increase should trigger at least one drift, got {}",
            drift_count
        );
    }

    // 3. Warning fires before drift on gradual increase.
    #[test]
    fn warning_before_drift() {
        let mut ddm = Ddm::new();

        // Stable baseline.
        let baseline = generate_values(0.05, 0.005, 200);
        feed(&mut ddm, &baseline);

        // Gradual increase toward high error.
        let ramp: Vec<f64> = (0..3000)
            .map(|i| {
                let t = i as f64 / 3000.0;
                // Ramp from 0.05 to 0.9 linearly.
                0.05 + 0.85 * t
            })
            .collect();

        let signals = feed(&mut ddm, &ramp);

        let first_warning = signals.iter().position(|&s| s == DriftSignal::Warning);
        let first_drift = signals.iter().position(|&s| s == DriftSignal::Drift);

        assert!(
            first_warning.is_some(),
            "gradual increase should trigger at least one warning"
        );
        assert!(
            first_drift.is_some(),
            "gradual increase should eventually trigger drift"
        );
        assert!(
            first_warning.unwrap() < first_drift.unwrap(),
            "warning (idx {}) should fire before drift (idx {})",
            first_warning.unwrap(),
            first_drift.unwrap()
        );
    }

    // 4. Minimum tracking -- min_p_plus_s decreases as low errors flow in.
    #[test]
    fn minimum_tracking() {
        let mut ddm = Ddm::new();

        // Feed a few high-ish values first.
        for _ in 0..5 {
            ddm.update(0.5);
        }
        let early_min = ddm.min_p_plus_s();

        // Feed many low values -- min should decrease.
        let low = generate_values(0.05, 0.005, 200);
        feed(&mut ddm, &low);
        let later_min = ddm.min_p_plus_s();

        assert!(
            later_min < early_min,
            "min_p_plus_s should decrease with low errors: early={}, later={}",
            early_min,
            later_min
        );
    }

    // 5. estimated_mean tracks running mean correctly.
    #[test]
    fn estimated_mean_tracks_correctly() {
        // Use a very long warmup so drift never fires during our test.
        let mut ddm = Ddm::with_params(2.0, 3.0, 100_000);

        // Feed 1000 values of exactly 0.3.
        for _ in 0..1000 {
            ddm.update(0.3);
        }
        let mean = ddm.estimated_mean();
        assert!(
            (mean - 0.3).abs() < 1e-10,
            "mean should be ~0.3, got {}",
            mean
        );

        // Feed 1000 more of exactly 0.7 => overall mean = 0.5.
        for _ in 0..1000 {
            ddm.update(0.7);
        }
        let mean2 = ddm.estimated_mean();
        assert!(
            (mean2 - 0.5).abs() < 1e-10,
            "mean should be ~0.5, got {}",
            mean2
        );

        // Feed 1000 more of exactly 0.2 => overall mean = (300+700+200)/3000 = 0.4.
        for _ in 0..1000 {
            ddm.update(0.2);
        }
        let mean3 = ddm.estimated_mean();
        assert!(
            (mean3 - 0.4).abs() < 1e-10,
            "mean should be ~0.4, got {}",
            mean3
        );
    }

    // 6. Reset clears all state.
    #[test]
    fn reset_clears_state() {
        let mut ddm = Ddm::new();

        // Feed some data.
        let vals = generate_values(0.4, 0.05, 500);
        feed(&mut ddm, &vals);
        assert!(ddm.count > 0);

        ddm.reset();

        assert_eq!(ddm.count, 0);
        assert_eq!(ddm.mean, 0.0);
        assert_eq!(ddm.m2, 0.0);
        assert_eq!(ddm.min_p_plus_s, f64::MAX);
        assert_eq!(ddm.min_s, f64::MAX);
    }

    // 7. clone_fresh produces a fresh instance with same params.
    #[cfg(feature = "alloc")]
    #[test]
    fn clone_fresh_same_params() {
        let ddm = Ddm::with_params(1.5, 2.5, 50);

        // Feed some data to dirty the original.
        let mut dirty = ddm.clone();
        let vals = generate_values(0.3, 0.02, 200);
        feed(&mut dirty, &vals);

        let mut fresh = dirty.clone_fresh();

        // Fresh detector should have zero mean (no data seen yet).
        assert_eq!(fresh.estimated_mean(), 0.0);

        // Compare behaviour: feed identical streams to the clone_fresh result
        // and a manually-constructed fresh Ddm with the same params.
        let mut manual_fresh = Ddm::with_params(1.5, 2.5, 50);
        let test_vals = generate_values(0.2, 0.01, 100);

        let signals_a: Vec<DriftSignal> = test_vals.iter().map(|&v| fresh.update(v)).collect();
        let signals_b: Vec<DriftSignal> =
            test_vals.iter().map(|&v| manual_fresh.update(v)).collect();

        assert_eq!(
            signals_a, signals_b,
            "clone_fresh should behave identically to a new instance with same params"
        );

        // Means should converge to the same value.
        assert!(
            (fresh.estimated_mean() - manual_fresh.estimated_mean()).abs() < 1e-12,
            "means should match: {} vs {}",
            fresh.estimated_mean(),
            manual_fresh.estimated_mean()
        );
    }

    // 8. Warmup -- no drift during min_instances period.
    #[test]
    fn warmup_no_drift() {
        let min_inst = 50u64;
        let mut ddm = Ddm::with_params(2.0, 3.0, min_inst);

        // Feed wildly changing values during warmup -- should still be Stable.
        for i in 0..min_inst {
            let value = if i % 2 == 0 { 0.0 } else { 1.0 };
            let signal = ddm.update(value);
            assert_eq!(
                signal,
                DriftSignal::Stable,
                "during warmup (i={}), signal should be Stable, got {:?}",
                i,
                signal
            );
        }
    }

    // 9. Custom params stored correctly.
    #[test]
    fn custom_params() {
        let ddm = Ddm::with_params(1.0, 4.0, 100);
        assert_eq!(ddm.warning_level(), 1.0);
        assert_eq!(ddm.drift_level(), 4.0);
        assert_eq!(ddm.min_instances(), 100);
    }

    // Extra: Default trait works.
    #[test]
    fn default_matches_new() {
        let a = Ddm::new();
        let b = Ddm::default();
        assert_eq!(a.warning_level, b.warning_level);
        assert_eq!(a.drift_level, b.drift_level);
        assert_eq!(a.min_instances, b.min_instances);
    }

    // Extra: std_dev is zero with no data.
    #[test]
    fn std_dev_zero_initially() {
        let ddm = Ddm::new();
        assert_eq!(ddm.std_dev(), 0.0);
    }

    // Extra: std_dev is zero for constant stream.
    #[test]
    fn std_dev_zero_for_constant() {
        let mut ddm = Ddm::new();
        for _ in 0..100 {
            ddm.update(0.5);
        }
        assert!(
            ddm.std_dev().abs() < 1e-12,
            "std of constant stream should be ~0, got {}",
            ddm.std_dev()
        );
    }

    // Extra: after drift reset, running stats are cleared but mins are kept.
    #[test]
    fn drift_resets_running_keeps_mins() {
        let mut ddm = Ddm::with_params(2.0, 3.0, 10);

        // Build up a low baseline.
        for _ in 0..100 {
            ddm.update(0.05);
        }
        let min_before = ddm.min_p_plus_s();

        // Trigger drift with sudden high error.
        let mut got_drift = false;
        for _ in 0..500 {
            if ddm.update(0.95) == DriftSignal::Drift {
                got_drift = true;
                break;
            }
        }
        assert!(got_drift, "should have triggered drift");

        // After drift: count is reset, but min_p_plus_s is preserved.
        assert_eq!(ddm.count, 0, "count should be 0 after drift reset");
        assert_eq!(ddm.mean, 0.0, "mean should be 0 after drift reset");
        assert_eq!(
            ddm.min_p_plus_s(),
            min_before,
            "min_p_plus_s should be preserved after drift reset"
        );
    }
}
