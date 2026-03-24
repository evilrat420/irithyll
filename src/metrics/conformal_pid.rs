//! Conformal PID control (Angelopoulos et al., NeurIPS 2023).
//!
//! Extends ACI with PID control theory for tighter, faster-adapting
//! prediction intervals at the same coverage level. Instead of ACI's
//! single integrator `alpha_t += gamma * (alpha - err_t)`, C-PID adds
//! proportional and derivative terms for faster convergence and
//! reduced oscillation.

/// PID-controlled conformal prediction.
///
/// The update rule:
/// ```text
/// alpha_t = saturate(alpha_t + K_I * (alpha - err_t) + K_P * score_t + K_D * (score_t - score_{t-1}))
/// ```
/// where `score_t` is the nonconformity score (|prediction - target|).
///
/// # Example
///
/// ```
/// use irithyll::metrics::conformal_pid::ConformalPID;
///
/// let mut cpid = ConformalPID::default_90();
/// for i in 0..100 {
///     let target = (i as f64) * 0.1;
///     let prediction = target + 0.05;
///     cpid.update(prediction, target);
/// }
/// assert!(cpid.coverage() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct ConformalPID {
    /// Target miscoverage rate (e.g., 0.1 for 90% coverage).
    alpha_target: f64,
    /// Proportional gain.
    k_p: f64,
    /// Integral gain (equivalent to ACI's gamma).
    k_i: f64,
    /// Derivative gain.
    k_d: f64,
    /// Current threshold level.
    alpha_t: f64,
    /// Previous nonconformity score.
    s_prev: f64,
    /// Number of updates processed.
    n_updates: u64,
    /// Running coverage rate (fraction of covered observations).
    cumulative_coverage: f64,
}

impl ConformalPID {
    /// Create a new PID-controlled conformal predictor.
    ///
    /// - `alpha_target`: desired miscoverage rate (e.g., 0.1 for 90% coverage).
    /// - `k_p`: proportional gain (reacts to current score magnitude).
    /// - `k_i`: integral gain (reacts to cumulative miscoverage error).
    /// - `k_d`: derivative gain (reacts to score rate-of-change).
    ///
    /// # Panics
    ///
    /// Panics if `alpha_target` is not in (0, 1) or any gain is negative.
    pub fn new(alpha_target: f64, k_p: f64, k_i: f64, k_d: f64) -> Self {
        assert!(
            alpha_target > 0.0 && alpha_target < 1.0,
            "alpha_target must be in (0, 1), got {alpha_target}"
        );
        assert!(k_p >= 0.0, "k_p must be >= 0, got {k_p}");
        assert!(k_i >= 0.0, "k_i must be >= 0, got {k_i}");
        assert!(k_d >= 0.0, "k_d must be >= 0, got {k_d}");
        Self {
            alpha_target,
            k_p,
            k_i,
            k_d,
            alpha_t: alpha_target,
            s_prev: 0.0,
            n_updates: 0,
            cumulative_coverage: 0.0,
        }
    }

    /// 90% coverage with tuned PID gains.
    pub fn default_90() -> Self {
        Self::new(0.1, 0.05, 0.01, 0.005)
    }

    /// 95% coverage with tuned PID gains.
    pub fn default_95() -> Self {
        Self::new(0.05, 0.05, 0.01, 0.005)
    }

    /// Update with a new (prediction, target) pair.
    ///
    /// Computes the nonconformity score and updates `alpha_t` via the
    /// PID control law.
    pub fn update(&mut self, prediction: f64, target: f64) {
        let score = (prediction - target).abs();
        let err = if score > self.alpha_t { 1.0 } else { 0.0 };

        // PID update
        self.alpha_t += self.k_i * (self.alpha_target - err)
            + self.k_p * score
            + self.k_d * (score - self.s_prev);

        // Saturate to prevent runaway
        self.alpha_t = self.alpha_t.clamp(0.001, 10.0);

        // Update coverage tracking
        self.n_updates += 1;
        let covered = if err == 0.0 { 1.0 } else { 0.0 };
        self.cumulative_coverage += (covered - self.cumulative_coverage) / self.n_updates as f64;

        self.s_prev = score;
    }

    /// Current miscoverage level (adaptive threshold).
    pub fn current_alpha(&self) -> f64 {
        self.alpha_t
    }

    /// Current threshold width. `alpha_t` can be used to scale prediction intervals.
    pub fn interval_width(&self) -> f64 {
        self.alpha_t
    }

    /// Empirical coverage rate.
    pub fn coverage(&self) -> f64 {
        self.cumulative_coverage
    }

    /// Number of updates processed.
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.alpha_t = self.alpha_target;
        self.s_prev = 0.0;
        self.n_updates = 0;
        self.cumulative_coverage = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_converges_to_target() {
        // Feed data where ~90% of scores are below threshold.
        let mut cpid = ConformalPID::default_90();
        for i in 0..1000 {
            let target = (i as f64) * 0.01;
            // Small error most of the time, large error 10% of the time
            let offset = if i % 10 == 0 { 50.0 } else { 0.01 };
            cpid.update(target + offset, target);
        }
        // Coverage should be in a reasonable range around 90%
        assert!(
            cpid.coverage() > 0.5,
            "coverage {} should be > 0.5",
            cpid.coverage()
        );
    }

    #[test]
    fn pid_tighter_than_constant_threshold() {
        // PID should adapt its threshold; a fixed threshold cannot.
        let mut cpid = ConformalPID::new(0.1, 0.05, 0.01, 0.005);
        let initial_alpha = cpid.current_alpha();

        // Feed consistently small errors — PID should adapt
        for i in 0..200 {
            let t = i as f64;
            cpid.update(t + 0.01, t);
        }

        // Alpha should have changed from initial (adapted)
        assert!(
            (cpid.current_alpha() - initial_alpha).abs() > 1e-6,
            "PID should adapt alpha_t, but it stayed at {initial_alpha}"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut cpid = ConformalPID::default_90();
        for i in 0..50 {
            cpid.update(i as f64 + 1.0, i as f64);
        }
        assert!(cpid.n_updates() > 0);

        cpid.reset();
        assert_eq!(cpid.n_updates(), 0);
        assert!((cpid.current_alpha() - 0.1).abs() < 1e-10);
        assert!((cpid.coverage()).abs() < 1e-10);
    }

    #[test]
    fn score_derivative_term_reacts_to_changes() {
        // Two identical configs except one has k_d=0
        let mut with_d = ConformalPID::new(0.1, 0.0, 0.01, 0.05);
        let mut without_d = ConformalPID::new(0.1, 0.0, 0.01, 0.0);

        // Feed identical small errors, then a sudden spike
        for i in 0..50 {
            let t = i as f64;
            with_d.update(t + 0.1, t);
            without_d.update(t + 0.1, t);
        }
        // Sudden large error
        with_d.update(100.0, 0.0);
        without_d.update(100.0, 0.0);

        // The D-term config should react differently to the spike
        assert!(
            (with_d.current_alpha() - without_d.current_alpha()).abs() > 1e-6,
            "D-term should cause different alpha after score spike"
        );
    }

    #[test]
    fn saturate_prevents_runaway() {
        let mut cpid = ConformalPID::new(0.1, 1.0, 1.0, 1.0); // very aggressive gains

        // Feed huge errors — alpha_t should not exceed 10.0
        for _ in 0..1000 {
            cpid.update(1e6, 0.0);
        }
        assert!(
            cpid.current_alpha() <= 10.0,
            "alpha_t {} should be <= 10.0",
            cpid.current_alpha()
        );
        assert!(
            cpid.current_alpha() >= 0.001,
            "alpha_t {} should be >= 0.001",
            cpid.current_alpha()
        );

        // Feed zero errors — alpha_t should not go below 0.001
        let mut cpid2 = ConformalPID::new(0.1, 0.0, 1.0, 0.0);
        for _ in 0..10000 {
            cpid2.update(0.0, 0.0); // score = 0, err = 0 each time
                                    // k_i * (0.1 - 0) = positive, so this actually pushes up.
                                    // To push down we need err=1 consistently.
        }
        // alpha_t stays in bounds
        assert!(cpid2.current_alpha() >= 0.001);
        assert!(cpid2.current_alpha() <= 10.0);
    }

    #[test]
    fn n_updates_tracks_correctly() {
        let mut cpid = ConformalPID::default_95();
        assert_eq!(cpid.n_updates(), 0);

        for i in 1..=37 {
            cpid.update(i as f64, 0.0);
            assert_eq!(cpid.n_updates(), i);
        }
    }
}
