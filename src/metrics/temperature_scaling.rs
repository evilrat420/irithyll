//! Online temperature scaling for multiclass probability calibration.
//!
//! Temperature scaling divides logits by a learned scalar T > 0 before
//! softmax, sharpening (T < 1) or smoothing (T > 1) the distribution.
//! This module provides an online variant that updates T via gradient
//! descent on the negative log-likelihood, one sample at a time.
//!
//! # Theory
//!
//! Given logits z = [z_1, ..., z_K] and true class c:
//!
//! ```text
//! NLL = -log softmax(z / T)_c
//!     = -z_c/T + log sum_k exp(z_k / T)
//! ```
//!
//! The gradient with respect to T:
//!
//! ```text
//! dNLL/dT = (z_c / T^2) - (1/T^2) * sum_k z_k * softmax(z/T)_k
//!         = (1/T^2) * (z_c - sum_k z_k * p_k)
//! ```
//!
//! where p_k = softmax(z/T)_k. We do gradient descent: T -= lr * dNLL/dT.
//!
//! # Example
//!
//! ```
//! use irithyll::metrics::temperature_scaling::OnlineTemperatureScaling;
//!
//! let mut ts = OnlineTemperatureScaling::new(0.01);
//! assert!((ts.temperature() - 1.0).abs() < 1e-12);
//!
//! // Calibrate logits before softmax
//! let calibrated = ts.calibrate(&[2.0, 1.0, 0.5]);
//! assert_eq!(calibrated.len(), 3);
//! let sum: f64 = calibrated.iter().sum();
//! assert!((sum - 1.0).abs() < 1e-10);
//! ```

// ---------------------------------------------------------------------------
// OnlineTemperatureScaling
// ---------------------------------------------------------------------------

/// Online temperature scaling for multiclass probability calibration.
///
/// Maintains a single scalar parameter T > 0, initialized to 1.0 (no scaling).
/// At each step, divides logits by T and applies stable softmax. The temperature
/// is updated via gradient descent on the negative log-likelihood of the true
/// class, enabling continuous calibration on streaming data.
#[derive(Debug, Clone)]
pub struct OnlineTemperatureScaling {
    /// Temperature parameter T > 0. Initialized to 1.0.
    temperature: f64,
    /// Learning rate for gradient descent on T.
    lr: f64,
    /// Number of updates performed.
    n_updates: u64,
}

impl OnlineTemperatureScaling {
    /// Create a new online temperature scaler with the given learning rate.
    ///
    /// Temperature is initialized to 1.0 (identity scaling).
    ///
    /// # Panics
    ///
    /// Panics if `lr` is not positive.
    pub fn new(lr: f64) -> Self {
        assert!(lr > 0.0, "learning rate must be > 0, got {lr}");
        Self {
            temperature: 1.0,
            lr,
            n_updates: 0,
        }
    }

    /// Apply temperature scaling and return calibrated probabilities.
    ///
    /// Divides each logit by T, then applies numerically stable softmax.
    /// Returns a probability vector of the same length as `logits`.
    pub fn calibrate(&self, logits: &[f64]) -> Vec<f64> {
        let inv_t = 1.0 / self.temperature;
        let scaled: Vec<f64> = logits.iter().map(|&z| z * inv_t).collect();
        stable_softmax(&scaled)
    }

    /// Update the temperature parameter with one (logits, true_class) observation.
    ///
    /// Performs a single gradient descent step on the negative log-likelihood
    /// with respect to T.
    ///
    /// # Panics
    ///
    /// Debug-panics if `true_class >= logits.len()`.
    pub fn update(&mut self, logits: &[f64], true_class: usize) {
        debug_assert!(
            true_class < logits.len(),
            "true_class {} out of range for {} logits",
            true_class,
            logits.len(),
        );

        // Compute softmax(z / T)
        let proba = self.calibrate(logits);

        // Gradient: dNLL/dT = (1/T^2) * (z_c - sum_k z_k * p_k)
        let z_c = logits[true_class];
        let weighted_mean: f64 = logits.iter().zip(proba.iter()).map(|(&z, &p)| z * p).sum();

        let t_sq = self.temperature * self.temperature;
        let grad = (z_c - weighted_mean) / t_sq;

        // Gradient descent step, then clamp T to stay positive and bounded.
        self.temperature -= self.lr * grad;
        // Clamp to [0.01, 100.0] to prevent degenerate scaling
        self.temperature = self.temperature.clamp(0.01, 100.0);
        self.n_updates += 1;
    }

    /// Current temperature value.
    #[inline]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Number of update steps performed.
    #[inline]
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Reset to initial state (T = 1.0, n_updates = 0).
    pub fn reset(&mut self) {
        self.temperature = 1.0;
        self.n_updates = 0;
    }
}

impl Default for OnlineTemperatureScaling {
    fn default() -> Self {
        Self::new(0.01)
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Numerically stable softmax.
fn stable_softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&z| (z - max_logit).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn initial_temperature_is_one() {
        let ts = OnlineTemperatureScaling::new(0.01);
        assert!(
            (ts.temperature() - 1.0).abs() < EPS,
            "initial temperature should be 1.0, got {}",
            ts.temperature()
        );
        assert_eq!(ts.n_updates(), 0);
    }

    #[test]
    fn calibrate_with_t_equals_one_is_softmax() {
        let ts = OnlineTemperatureScaling::new(0.01);
        let logits = vec![2.0, 1.0, 0.5];
        let calibrated = ts.calibrate(&logits);
        let expected = stable_softmax(&logits);
        for (c, e) in calibrated.iter().zip(expected.iter()) {
            assert!(
                (c - e).abs() < EPS,
                "calibrate with T=1 should equal softmax: {c} vs {e}"
            );
        }
    }

    #[test]
    fn calibrate_sums_to_one() {
        let ts = OnlineTemperatureScaling::new(0.01);
        let logits = vec![3.0, 1.0, -2.0, 0.5];
        let calibrated = ts.calibrate(&logits);
        let sum: f64 = calibrated.iter().sum();
        assert!(
            (sum - 1.0).abs() < EPS,
            "calibrated probabilities should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn higher_temperature_flattens_distribution() {
        let mut ts = OnlineTemperatureScaling::new(0.01);
        let logits = vec![5.0, 1.0, 0.0];

        let sharp = ts.calibrate(&logits);
        // Manually set higher temperature
        ts.temperature = 5.0;
        let flat = ts.calibrate(&logits);

        // With higher T, the max probability should be lower (flatter distribution)
        let max_sharp = sharp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_flat = flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_sharp > max_flat,
            "higher T should flatten distribution: max_sharp={max_sharp} > max_flat={max_flat}"
        );
    }

    #[test]
    fn lower_temperature_sharpens_distribution() {
        let mut ts = OnlineTemperatureScaling::new(0.01);
        let logits = vec![2.0, 1.0, 0.5];

        let normal = ts.calibrate(&logits);
        ts.temperature = 0.1;
        let sharp = ts.calibrate(&logits);

        let max_normal = normal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_sharp = sharp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_sharp > max_normal,
            "lower T should sharpen distribution: max_sharp={max_sharp} > max_normal={max_normal}"
        );
    }

    #[test]
    fn update_adjusts_temperature() {
        let mut ts = OnlineTemperatureScaling::new(0.1);
        let t_before = ts.temperature();

        // Feed a confident-but-wrong prediction: logits favor class 0 but true is class 2
        let logits = vec![5.0, 1.0, 0.0];
        ts.update(&logits, 2);

        assert_ne!(
            ts.temperature(),
            t_before,
            "temperature should change after update"
        );
        assert_eq!(ts.n_updates(), 1);
    }

    #[test]
    fn overconfident_model_increases_temperature() {
        // When the model is overconfident (logits are very spread), temperature
        // should increase to smooth the distribution.
        let mut ts = OnlineTemperatureScaling::new(0.05);

        // Repeatedly: model is confident about class 0 but true class varies
        for _ in 0..100 {
            ts.update(&[10.0, 0.0, 0.0], 1); // wrong class
            ts.update(&[10.0, 0.0, 0.0], 2); // wrong class
        }

        assert!(
            ts.temperature() > 1.0,
            "overconfident wrong predictions should increase T, got {}",
            ts.temperature()
        );
    }

    #[test]
    fn temperature_stays_positive_after_many_updates() {
        let mut ts = OnlineTemperatureScaling::new(0.1);
        for i in 0..1000 {
            let true_class = i % 3;
            ts.update(&[1.0, 2.0, 3.0], true_class);
        }
        assert!(
            ts.temperature() > 0.0,
            "temperature must stay positive, got {}",
            ts.temperature()
        );
        assert!(
            ts.temperature().is_finite(),
            "temperature must be finite, got {}",
            ts.temperature()
        );
    }

    #[test]
    fn reset_restores_default_state() {
        let mut ts = OnlineTemperatureScaling::new(0.01);
        ts.update(&[1.0, 0.0], 0);
        ts.update(&[0.0, 1.0], 1);
        assert!(ts.n_updates() > 0);

        ts.reset();
        assert!(
            (ts.temperature() - 1.0).abs() < EPS,
            "temperature should be 1.0 after reset, got {}",
            ts.temperature()
        );
        assert_eq!(ts.n_updates(), 0, "n_updates should be 0 after reset");
    }

    #[test]
    fn default_uses_lr_0_01() {
        let ts = OnlineTemperatureScaling::default();
        assert!(
            (ts.temperature() - 1.0).abs() < EPS,
            "default temperature should be 1.0"
        );
    }

    #[test]
    #[should_panic(expected = "learning rate must be > 0")]
    fn panics_on_zero_lr() {
        let _ = OnlineTemperatureScaling::new(0.0);
    }

    #[test]
    #[should_panic(expected = "learning rate must be > 0")]
    fn panics_on_negative_lr() {
        let _ = OnlineTemperatureScaling::new(-0.01);
    }

    #[test]
    fn extreme_logits_dont_cause_nan() {
        let mut ts = OnlineTemperatureScaling::new(0.01);
        let logits = vec![1000.0, -1000.0, 0.0];
        let calibrated = ts.calibrate(&logits);
        assert!(
            calibrated.iter().all(|p| p.is_finite()),
            "calibrate should be finite for extreme logits"
        );

        // Update with extreme logits shouldn't break
        ts.update(&logits, 0);
        assert!(
            ts.temperature().is_finite(),
            "temperature should be finite after extreme update"
        );
    }
}
