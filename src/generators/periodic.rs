//! Periodic signal generator for frequency-domain benchmarking.
//!
//! Generates a multi-harmonic periodic signal and presents it as a streaming
//! regression task: given a window of past values as features, predict the
//! next value. Models that can learn frequency-domain structure (e.g., complex
//! SSMs, T-KAN, ESN) should outperform memoryless baselines on this task.
//!
//! # Signal model
//!
//! ```text
//! y(t) = sum_{k=1}^{harmonics} amplitude_k * sin(2π * k * t / period + phase_k)
//!      + noise * ε_t
//! ```
//!
//! where `amplitude_k = amplitude / k` (natural harmonic decay) and `phase_k`
//! are fixed offsets derived from the seed. The fundamental frequency is
//! `1 / period`.
//!
//! # Feature construction
//!
//! Features: a window of the last `window` signal values (oldest first).
//! Target: the signal value at the next timestep.

use super::{Rng, StreamGenerator, TaskType};

/// Periodic signal stream generator.
///
/// # Parameters
/// - `seed`: PRNG seed for phase offsets and noise
/// - `period`: fundamental period in samples (default: 20)
/// - `amplitude`: fundamental harmonic amplitude (default: 1.0)
/// - `harmonics`: number of harmonics to sum (default: 3)
/// - `noise_std`: additive Gaussian noise standard deviation (default: 0.05)
/// - `window`: number of past values used as features (default: 10)
#[derive(Debug, Clone)]
pub struct PeriodicStream {
    /// PRNG for noise.
    rng: Rng,
    /// Fundamental period in samples.
    period: f64,
    /// Amplitude of the fundamental harmonic.
    amplitude: f64,
    /// Number of harmonics to include.
    harmonics: usize,
    /// Additive noise standard deviation.
    noise_std: f64,
    /// Feature window size (number of past values).
    window: usize,
    /// Phase offsets for each harmonic (fixed from seed).
    phases: Vec<f64>,
    /// Rolling buffer of the last `window + 1` clean signal values.
    /// Index 0 is oldest.
    signal_buf: Vec<f64>,
    /// Current timestep.
    t: usize,
}

impl PeriodicStream {
    /// Default period in samples.
    pub const DEFAULT_PERIOD: usize = 20;
    /// Default amplitude.
    pub const DEFAULT_AMPLITUDE: f64 = 1.0;
    /// Default number of harmonics.
    pub const DEFAULT_HARMONICS: usize = 3;
    /// Default noise standard deviation.
    pub const DEFAULT_NOISE_STD: f64 = 0.05;
    /// Default feature window size.
    pub const DEFAULT_WINDOW: usize = 10;

    /// Create a periodic generator.
    ///
    /// - `seed`: PRNG seed.
    /// - `window`: feature window size (number of past signal values as features).
    /// - `period`: fundamental period in samples.
    pub fn new(seed: u64, window: usize, period: usize) -> Self {
        Self::with_config(
            seed,
            period,
            Self::DEFAULT_AMPLITUDE,
            Self::DEFAULT_HARMONICS,
            Self::DEFAULT_NOISE_STD,
            window,
        )
    }

    /// Create a periodic generator with custom parameters.
    ///
    /// # Panics
    ///
    /// Panics if `period == 0`, `harmonics == 0`, or `window == 0`.
    pub fn with_config(
        seed: u64,
        period: usize,
        amplitude: f64,
        harmonics: usize,
        noise_std: f64,
        window: usize,
    ) -> Self {
        assert!(period > 0, "period must be > 0");
        assert!(harmonics > 0, "harmonics must be > 0");
        assert!(window > 0, "window must be > 0");

        // Derive fixed phase offsets from seed (one per harmonic).
        let mut phase_rng = Rng::new(seed.wrapping_add(0xDEAD_BEEF));
        let phases: Vec<f64> = (0..harmonics)
            .map(|_| phase_rng.uniform_range(0.0, 2.0 * std::f64::consts::PI))
            .collect();

        // Pre-fill the buffer with clean signal values for t = 0..=window.
        // This allows immediate feature production without a warmup silent period.
        let mut signal_buf = Vec::with_capacity(window + 1);
        for step in 0..=window {
            signal_buf.push(Self::eval_signal(
                step,
                period as f64,
                amplitude,
                harmonics,
                &phases,
            ));
        }

        Self {
            rng: Rng::new(seed),
            period: period as f64,
            amplitude,
            harmonics,
            noise_std,
            window,
            phases,
            signal_buf,
            t: window, // next sample index (buffer holds 0..=window, features window = t-window..t, target = t+1)
        }
    }

    /// Evaluate the clean (no-noise) signal at integer timestep `t`.
    fn eval_signal(t: usize, period: f64, amplitude: f64, harmonics: usize, phases: &[f64]) -> f64 {
        let mut y = 0.0;
        for k in 1..=harmonics {
            let amp_k = amplitude / k as f64;
            let freq = 2.0 * std::f64::consts::PI * k as f64 / period;
            y += amp_k * (freq * t as f64 + phases[k - 1]).sin();
        }
        y
    }

    /// Evaluate the clean signal at the current timestep.
    fn eval_current(&self, t: usize) -> f64 {
        Self::eval_signal(t, self.period, self.amplitude, self.harmonics, &self.phases)
    }

    /// Fundamental period (in samples).
    pub fn period(&self) -> f64 {
        self.period
    }

    /// Feature window size.
    pub fn window(&self) -> usize {
        self.window
    }
}

impl StreamGenerator for PeriodicStream {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Features: last `window` noisy signal values (oldest first).
        // The buffer holds signal_buf[0..=window], where signal_buf[window] = clean value at t.
        // We add noise to the features to simulate realistic observation.
        let features: Vec<f64> = (0..self.window)
            .map(|i| {
                let clean = self.signal_buf[i];
                if self.noise_std > 0.0 {
                    clean + self.rng.normal(0.0, self.noise_std)
                } else {
                    clean
                }
            })
            .collect();

        // Target: clean signal value one step ahead (t + 1).
        let next_t = self.t + 1;
        let clean_next = self.eval_current(next_t);

        // Advance the buffer: drop oldest, append clean value at next_t.
        self.signal_buf.remove(0);
        self.signal_buf.push(clean_next);
        self.t = next_t;

        (features, clean_next)
    }

    fn n_features(&self) -> usize {
        self.window
    }

    fn task_type(&self) -> TaskType {
        TaskType::Regression
    }

    fn drift_occurred(&self) -> bool {
        false // Periodic signal is stationary (no concept drift).
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_produces_correct_n_features() {
        let window = PeriodicStream::DEFAULT_WINDOW;
        let period = PeriodicStream::DEFAULT_PERIOD;
        let mut gen = PeriodicStream::new(42, window, period);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            window,
            "features should have window={} dims, got {}",
            window,
            features.len()
        );
    }

    #[test]
    fn periodic_task_type_is_regression() {
        let gen = PeriodicStream::new(42, 10, 20);
        assert_eq!(gen.task_type(), TaskType::Regression);
    }

    #[test]
    fn periodic_no_drift() {
        let mut gen = PeriodicStream::new(42, 10, 20);
        for _ in 0..500 {
            gen.next_sample();
            assert!(!gen.drift_occurred(), "periodic signal should not drift");
        }
    }

    #[test]
    fn periodic_produces_finite_values() {
        let mut gen = PeriodicStream::new(13, 10, 20);
        for i in 0..1000 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(f.is_finite(), "feature {} at sample {} is not finite", j, i);
            }
            assert!(target.is_finite(), "target at sample {} is not finite", i);
        }
    }

    #[test]
    fn periodic_deterministic_with_same_seed() {
        let mut gen1 = PeriodicStream::new(99, 10, 20);
        let mut gen2 = PeriodicStream::new(99, 10, 20);
        for _ in 0..500 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(
                t1, t2,
                "same seed should produce identical targets: {} vs {}",
                t1, t2
            );
        }
    }

    #[test]
    fn periodic_signal_has_expected_variance() {
        // A periodic signal with amplitude=1 and 3 harmonics should have
        // non-trivial variance (not all zeros, not constant).
        let mut gen = PeriodicStream::with_config(7, 20, 1.0, 3, 0.0, 5);
        let mut targets: Vec<f64> = Vec::new();
        for _ in 0..200 {
            let (_, t) = gen.next_sample();
            targets.push(t);
        }
        let mean: f64 = targets.iter().sum::<f64>() / targets.len() as f64;
        let var: f64 =
            targets.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / targets.len() as f64;
        assert!(
            var > 0.01,
            "periodic signal should have substantial variance, got {}",
            var
        );
    }

    #[test]
    fn periodic_noiseless_target_is_deterministic() {
        // With noise_std=0, target is a pure function of t — identical across two instances.
        let mut gen1 = PeriodicStream::with_config(42, 10, 1.0, 2, 0.0, 5);
        let mut gen2 = PeriodicStream::with_config(42, 10, 1.0, 2, 0.0, 5);
        for _ in 0..100 {
            let (_, t1) = gen1.next_sample();
            let (_, t2) = gen2.next_sample();
            assert!(
                (t1 - t2).abs() < 1e-12,
                "noiseless targets must match: {} vs {}",
                t1,
                t2
            );
        }
    }

    #[test]
    fn periodic_custom_window_dimension() {
        let mut gen = PeriodicStream::with_config(1, 30, 2.0, 2, 0.1, 15);
        assert_eq!(gen.n_features(), 15);
        let (features, _) = gen.next_sample();
        assert_eq!(features.len(), 15);
    }
}
