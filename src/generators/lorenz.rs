//! Lorenz Attractor generator (Lorenz, 1963).
//!
//! Generates samples from the Lorenz system of ordinary differential equations:
//!
//! ```text
//! dx/dt = sigma * (y - x)
//! dy/dt = x * (rho - z) - y
//! dz/dt = x * y - beta * z
//! ```
//!
//! Default parameters: `sigma=10, rho=28, beta=8/3`.
//! Integration via 4th-order Runge-Kutta with `dt=0.01`, subsampled every 10 steps.
//!
//! Features: `(x, y, z)` at time `t`. Target: `x(t+1)`.

use super::{StreamGenerator, TaskType};

/// Lorenz Attractor stream generator.
///
/// # Parameters
/// - `sigma`: Prandtl number (default 10.0)
/// - `rho`: Rayleigh number (default 28.0)
/// - `beta`: geometric factor (default 8/3)
/// - `dt`: integration time step (default 0.01)
/// - `subsample`: take every N-th integrated step (default 10)
/// - `warmup`: integration steps to skip (default 1000)
#[derive(Debug, Clone)]
pub struct Lorenz {
    /// Pre-generated trajectory: list of (x, y, z) states.
    trajectory: Vec<[f64; 3]>,
    /// Current position in the trajectory.
    cursor: usize,
}

impl Lorenz {
    /// Create a Lorenz generator with default parameters.
    ///
    /// `sigma=10, rho=28, beta=8/3, dt=0.01, subsample=10, warmup=1000`.
    /// Pre-generates enough data for ~100K samples.
    pub fn new() -> Self {
        Self::with_config(10.0, 28.0, 8.0 / 3.0, 0.01, 10, 1000, 100_000)
    }

    /// Create with custom parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        sigma: f64,
        rho: f64,
        beta: f64,
        dt: f64,
        subsample: usize,
        warmup: usize,
        max_samples: usize,
    ) -> Self {
        let subsample = subsample.max(1);
        // Total integration steps needed
        let total_steps = warmup + (max_samples + 1) * subsample;

        // Integrate using RK4
        let mut x = 1.0_f64;
        let mut y = 1.0_f64;
        let mut z = 1.0_f64;

        // Collect subsampled points after warmup
        let mut trajectory = Vec::with_capacity(max_samples + 2);

        for step in 0..total_steps {
            // RK4 integration
            let (dx1, dy1, dz1) = lorenz_deriv(x, y, z, sigma, rho, beta);

            let (x2, y2, z2) = (x + 0.5 * dt * dx1, y + 0.5 * dt * dy1, z + 0.5 * dt * dz1);
            let (dx2, dy2, dz2) = lorenz_deriv(x2, y2, z2, sigma, rho, beta);

            let (x3, y3, z3) = (x + 0.5 * dt * dx2, y + 0.5 * dt * dy2, z + 0.5 * dt * dz2);
            let (dx3, dy3, dz3) = lorenz_deriv(x3, y3, z3, sigma, rho, beta);

            let (x4, y4, z4) = (x + dt * dx3, y + dt * dy3, z + dt * dz3);
            let (dx4, dy4, dz4) = lorenz_deriv(x4, y4, z4, sigma, rho, beta);

            x += dt / 6.0 * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
            y += dt / 6.0 * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4);
            z += dt / 6.0 * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4);

            // After warmup, collect subsampled points
            if step >= warmup && (step - warmup) % subsample == 0 {
                trajectory.push([x, y, z]);
            }
        }

        Self {
            trajectory,
            cursor: 0,
        }
    }
}

impl Default for Lorenz {
    fn default() -> Self {
        Self::new()
    }
}

/// Lorenz system derivatives.
#[inline]
fn lorenz_deriv(x: f64, y: f64, z: f64, sigma: f64, rho: f64, beta: f64) -> (f64, f64, f64) {
    (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)
}

impl StreamGenerator for Lorenz {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        let state = self.trajectory[self.cursor];
        let features = vec![state[0], state[1], state[2]];

        // Target: x at next time step
        let next_idx = self.cursor + 1;
        let target = if next_idx < self.trajectory.len() {
            self.trajectory[next_idx][0]
        } else {
            // Wrap around
            self.trajectory[0][0]
        };

        self.cursor += 1;
        if self.cursor >= self.trajectory.len() - 1 {
            self.cursor = 0;
        }

        (features, target)
    }

    fn n_features(&self) -> usize {
        3
    }

    fn task_type(&self) -> TaskType {
        TaskType::Regression
    }

    fn drift_occurred(&self) -> bool {
        false // Lorenz attractor is stationary (deterministic chaos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lorenz_produces_correct_n_features() {
        let mut gen = Lorenz::new();
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            3,
            "Lorenz should produce 3 features (x, y, z), got {}",
            features.len()
        );
    }

    #[test]
    fn lorenz_task_type_is_regression() {
        let gen = Lorenz::new();
        assert_eq!(gen.task_type(), TaskType::Regression);
    }

    #[test]
    fn lorenz_produces_finite_values() {
        let mut gen = Lorenz::new();
        for i in 0..5000 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(
                    f.is_finite(),
                    "feature {} at sample {} is not finite: {}",
                    j,
                    i,
                    f
                );
            }
            assert!(
                target.is_finite(),
                "target at sample {} is not finite: {}",
                i,
                target
            );
        }
    }

    #[test]
    fn lorenz_values_on_attractor() {
        // The Lorenz attractor has x roughly in [-20, 20], y in [-30, 30], z in [0, 50]
        let mut gen = Lorenz::new();
        for i in 0..5000 {
            let (features, _) = gen.next_sample();
            assert!(
                features[0].abs() < 50.0,
                "x at sample {} has unexpected magnitude: {}",
                i,
                features[0]
            );
            assert!(
                features[1].abs() < 60.0,
                "y at sample {} has unexpected magnitude: {}",
                i,
                features[1]
            );
            assert!(
                features[2] > -10.0 && features[2] < 70.0,
                "z at sample {} has unexpected range: {}",
                i,
                features[2]
            );
        }
    }

    #[test]
    fn lorenz_no_drift() {
        let mut gen = Lorenz::new();
        for _ in 0..500 {
            gen.next_sample();
            assert!(!gen.drift_occurred(), "Lorenz should never signal drift");
        }
    }

    #[test]
    fn lorenz_deterministic() {
        let mut gen1 = Lorenz::new();
        let mut gen2 = Lorenz::new();
        for _ in 0..200 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same params should produce identical features");
            assert_eq!(t1, t2, "same params should produce identical targets");
        }
    }

    #[test]
    fn lorenz_series_is_chaotic() {
        let mut gen = Lorenz::new();
        let mut x_vals = Vec::new();
        for _ in 0..200 {
            let (features, _) = gen.next_sample();
            x_vals.push(features[0]);
        }
        let mean: f64 = x_vals.iter().sum::<f64>() / x_vals.len() as f64;
        let var: f64 = x_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x_vals.len() as f64;
        assert!(
            var > 1.0,
            "Lorenz x should have substantial variance on the attractor, got {}",
            var
        );
    }

    #[test]
    fn lorenz_custom_parameters() {
        let gen = Lorenz::with_config(10.0, 28.0, 8.0 / 3.0, 0.005, 5, 500, 1000);
        assert_eq!(gen.n_features(), 3);
        assert_eq!(gen.task_type(), TaskType::Regression);
    }
}
