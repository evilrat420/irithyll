//! Online Platt Scaling with Calibeating (Gupta & Ramdas, ICML 2023).
//!
//! Post-hoc calibration for streaming classifiers. Fits logistic
//! regression on model logits to produce calibrated probabilities.

/// Online Platt scaling for probability calibration.
///
/// Maintains two parameters (a, b) such that:
/// ```text
/// P(y=1 | z) = sigmoid(a * z + b)
/// ```
/// where z is the raw model output (logit/score).
///
/// Parameters are updated via online gradient descent on the
/// log-loss, one sample at a time.
///
/// # Example
///
/// ```
/// use irithyll::metrics::platt_scaling::OnlinePlattScaling;
///
/// let mut platt = OnlinePlattScaling::new(0.01);
/// // Feed (logit, label) pairs:
/// platt.update(2.0, 1.0);
/// platt.update(-1.0, 0.0);
///
/// // Get calibrated probability from a logit:
/// let prob = platt.calibrate(1.5);
/// assert!(prob > 0.0 && prob < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct OnlinePlattScaling {
    /// Scale parameter.
    a: f64,
    /// Shift parameter.
    b: f64,
    /// Learning rate for online gradient descent.
    lr: f64,
    /// Number of updates performed.
    n_updates: u64,
}

/// Sigmoid function: 1 / (1 + exp(-x)).
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl OnlinePlattScaling {
    /// Create a new online Platt scaler with the given learning rate.
    ///
    /// Initialises with `a = 1.0`, `b = 0.0` (identity scaling through sigmoid).
    ///
    /// # Panics
    ///
    /// Panics if `lr` is not positive.
    pub fn new(lr: f64) -> Self {
        assert!(lr > 0.0, "learning rate must be > 0, got {lr}");
        Self {
            a: 1.0,
            b: 0.0,
            lr,
            n_updates: 0,
        }
    }

    /// Return a calibrated probability for the given logit.
    ///
    /// Computes `sigmoid(a * logit + b)`.
    #[inline]
    pub fn calibrate(&self, logit: f64) -> f64 {
        sigmoid(self.a * logit + self.b)
    }

    /// Update parameters with one (logit, label) observation.
    ///
    /// `label` should be `0.0` (negative) or `1.0` (positive).
    /// Uses online gradient descent on the log-loss.
    pub fn update(&mut self, logit: f64, label: f64) {
        let p = sigmoid(self.a * logit + self.b);
        let error = p - label;
        self.a -= self.lr * error * logit;
        self.b -= self.lr * error;
        self.n_updates += 1;
    }

    /// Current (a, b) parameters.
    pub fn params(&self) -> (f64, f64) {
        (self.a, self.b)
    }

    /// Number of update steps performed.
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Reset to initial state (a=1.0, b=0.0, n_updates=0).
    pub fn reset(&mut self) {
        self.a = 1.0;
        self.b = 0.0;
        self.n_updates = 0;
    }
}

impl Default for OnlinePlattScaling {
    fn default() -> Self {
        Self::new(0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn calibrate_identity_by_default() {
        // With a=1, b=0: calibrate(x) == sigmoid(x)
        let platt = OnlinePlattScaling::new(0.01);
        let logit = 0.5;
        let expected = sigmoid(logit);
        assert!((platt.calibrate(logit) - expected).abs() < EPS);

        // Also check zero and negative
        assert!((platt.calibrate(0.0) - 0.5).abs() < EPS);
        assert!(platt.calibrate(-5.0) < 0.5);
        assert!(platt.calibrate(5.0) > 0.5);
    }

    #[test]
    fn update_moves_toward_correct_calibration() {
        let mut platt = OnlinePlattScaling::new(0.1);
        // Train on consistent positive signal: high logit => label 1
        for _ in 0..200 {
            platt.update(3.0, 1.0);
            platt.update(-3.0, 0.0);
        }
        // After training, high logits should give high probability
        let p_high = platt.calibrate(3.0);
        let p_low = platt.calibrate(-3.0);
        assert!(p_high > 0.9, "expected p_high > 0.9, got {p_high}");
        assert!(p_low < 0.1, "expected p_low < 0.1, got {p_low}");
    }

    #[test]
    fn perfect_classifier_stays_calibrated() {
        // If sigmoid(logit) already matches labels well, params shouldn't drift far
        let mut platt = OnlinePlattScaling::new(0.01);
        let (a0, b0) = platt.params();
        // Feed perfectly calibrated data (logit=2 => sigmoid~0.88, label=1 is reasonable)
        for _ in 0..50 {
            platt.update(2.0, 1.0);
            platt.update(-2.0, 0.0);
        }
        let (a1, b1) = platt.params();
        // Parameters should stay near initial values
        assert!((a1 - a0).abs() < 0.5, "a drifted too far: {a0} -> {a1}");
        assert!((b1 - b0).abs() < 0.5, "b drifted too far: {b0} -> {b1}");
    }

    #[test]
    fn reset_restores_default() {
        let mut platt = OnlinePlattScaling::new(0.05);
        for _ in 0..100 {
            platt.update(1.0, 1.0);
        }
        assert!(platt.n_updates() > 0);
        platt.reset();
        let (a, b) = platt.params();
        assert!(
            (a - 1.0).abs() < EPS,
            "a should be 1.0 after reset, got {a}"
        );
        assert!(
            (b - 0.0).abs() < EPS,
            "b should be 0.0 after reset, got {b}"
        );
        assert_eq!(platt.n_updates(), 0);
    }

    #[test]
    fn extreme_logits_dont_cause_nan() {
        let mut platt = OnlinePlattScaling::new(0.01);
        // Extreme positive
        let p = platt.calibrate(1000.0);
        assert!(p.is_finite(), "calibrate(1000) should be finite, got {p}");
        assert!((p - 1.0).abs() < EPS);
        // Extreme negative
        let p = platt.calibrate(-1000.0);
        assert!(p.is_finite(), "calibrate(-1000) should be finite, got {p}");
        assert!(p.abs() < EPS);
        // Update with extreme values shouldn't produce NaN
        platt.update(1000.0, 1.0);
        platt.update(-1000.0, 0.0);
        let (a, b) = platt.params();
        assert!(a.is_finite(), "a should be finite after extreme updates");
        assert!(b.is_finite(), "b should be finite after extreme updates");
    }

    #[test]
    fn n_updates_tracks() {
        let mut platt = OnlinePlattScaling::new(0.01);
        assert_eq!(platt.n_updates(), 0);
        platt.update(1.0, 1.0);
        assert_eq!(platt.n_updates(), 1);
        platt.update(-1.0, 0.0);
        assert_eq!(platt.n_updates(), 2);
        platt.update(0.5, 1.0);
        assert_eq!(platt.n_updates(), 3);
    }
}
