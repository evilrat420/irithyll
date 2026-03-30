//! Real-world synthetic benchmark harness for irithyll streaming algorithms.
//!
//! Evaluates 8 streaming algorithms across 14 synthetic datasets using the
//! prequential (test-then-train) protocol. All datasets are generated inline
//! with deterministic seeded RNG -- no file downloads needed for CI.
//!
//! Datasets:
//! - **Sine regression**: x -> sin(x) + noise, 10K samples
//! - **Rotating hyperplane**: concept drift classification, 50K samples
//! - **Sensor drift**: 6 features with gradual drift every 5K samples, 30K samples
//! - **Multi-class spiral**: 3 interlocking spirals, 15K samples
//! - **Sudden drift**: abrupt distribution change at sample 25K of 50K
//! - **High-dim nonlinear**: 50 features (5 signal + 45 noise), regression, 20K samples
//! - **Temporal pattern**: autoregressive 10-feature regression with 3-step lag, 30K samples
//! - **High-dim binary drift**: 30 features, binary classification with feature-level concept drift, 30K samples
//! - **Mackey-Glass**: chaotic delay-differential time series, 30K samples (reservoir computing standard)
//! - **NARMA10**: nonlinear autoregressive moving average order 10, 20K samples (reservoir computing standard)
//! - **Lorenz attractor**: chaotic 3D dynamical system, 30K samples (chaotic prediction benchmark)
//! - **Compositional function**: 10-feature compositional nonlinearity, 20K samples (KAN benchmark)
//! - **Non-stationary sequence**: 5-phase distribution shift, 40K samples (TTT benchmark)
//! - **Sparse temporal events**: 8-feature sparse binary events, 25K samples (SpikeNet benchmark)
//!
//! Algorithms:
//! - SGBT (default config)
//! - DistributionalSGBT
//! - StreamingKAN
//! - StreamingTTT
//! - EchoStateNetwork
//! - StreamingMamba
//! - RecursiveLeastSquares
//! - NeuralMoE (3 experts)
//!
//! Run: `cargo bench --bench real_world_bench`

use irithyll::*;
use std::time::Instant;

// ===========================================================================
// Deterministic PRNG (xorshift64 -- no external dependencies)
// ===========================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1)) // avoid zero state
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Uniform [lo, hi)
    fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.uniform() * (hi - lo)
    }

    /// Approximate standard normal via Box-Muller.
    fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.uniform().max(1e-15); // avoid log(0)
        let u2 = self.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }
}

// ===========================================================================
// BenchmarkDataset trait
// ===========================================================================

/// A dataset that produces (features_matrix, targets) for benchmarking.
trait BenchmarkDataset {
    fn name(&self) -> &str;
    fn n_features(&self) -> usize;
    fn task(&self) -> Task;
    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>);
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Task {
    Regression,
    BinaryClassification,
    MulticlassClassification { n_classes: usize },
}

// ===========================================================================
// Dataset implementations
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Sine regression: x -> sin(x) + noise, 10K samples
// ---------------------------------------------------------------------------

struct SineRegression;

impl BenchmarkDataset for SineRegression {
    fn name(&self) -> &str {
        "Sine Regression (10K)"
    }

    fn n_features(&self) -> usize {
        1
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0x51AE_0001);
        let n = 10_000;
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.uniform_range(-3.0 * std::f64::consts::PI, 3.0 * std::f64::consts::PI);
            let noise = rng.normal(0.0, 0.1);
            features.push(vec![x]);
            targets.push(x.sin() + noise);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 2. Rotating hyperplane: concept drift binary classification, 50K samples
// ---------------------------------------------------------------------------

struct RotatingHyperplane;

impl BenchmarkDataset for RotatingHyperplane {
    fn name(&self) -> &str {
        "Rotating Hyperplane (50K)"
    }

    fn n_features(&self) -> usize {
        10
    }

    fn task(&self) -> Task {
        Task::BinaryClassification
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xBEEF_0002);
        let n = 50_000;
        let d = 10;
        let rotation_speed = 0.001;

        let mut weights: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) / d as f64).collect();

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for i in 0..n {
            let x: Vec<f64> = (0..d).map(|_| rng.uniform()).collect();
            let dot: f64 = x.iter().zip(weights.iter()).map(|(a, b)| a * b).sum();
            let threshold: f64 = weights.iter().sum::<f64>() * 0.5;
            let noise = rng.normal(0.0, 0.05);
            let label = if dot + noise > threshold { 1.0 } else { 0.0 };

            features.push(x);
            targets.push(label);

            // Rotate weights gradually to introduce concept drift
            if i % 100 == 0 {
                for (j, w) in weights.iter_mut().enumerate() {
                    let angle = rotation_speed * (j as f64 + 1.0);
                    *w = (*w * angle.cos() + 0.5 * angle.sin()).abs();
                    if *w < 0.01 {
                        *w = 0.01;
                    }
                }
            }
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 3. Sensor drift: 6 features with gradual drift every 5K samples, regression
// ---------------------------------------------------------------------------

struct SensorDrift;

impl BenchmarkDataset for SensorDrift {
    fn name(&self) -> &str {
        "Sensor Drift (30K)"
    }

    fn n_features(&self) -> usize {
        6
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xD4F7_0003);
        let n = 30_000;
        let d = 6;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for i in 0..n {
            // Drift regime: coefficients shift every 5K samples
            let regime = i / 5_000;
            let drift_offset = regime as f64 * 0.5;

            let x: Vec<f64> = (0..d)
                .map(|j| {
                    let base = rng.normal(0.0, 1.0);
                    // Feature-specific drift: odd features drift up, even drift down
                    if j % 2 == 0 {
                        base + drift_offset
                    } else {
                        base - drift_offset * 0.3
                    }
                })
                .collect();

            // Target function changes with drift
            let y = x[0] * (1.0 + drift_offset * 0.2) + x[1] * 2.0
                - x[2] * (0.5 + drift_offset * 0.1)
                + x[3] * x[4] * 0.3
                + (x[5] * 0.5).sin()
                + rng.normal(0.0, 0.2);

            features.push(x);
            targets.push(y);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 4. Multi-class spiral: 3 interlocking spirals, 15K samples
// ---------------------------------------------------------------------------

struct MulticlassSpiral;

impl BenchmarkDataset for MulticlassSpiral {
    fn name(&self) -> &str {
        "Multi-class Spiral (15K)"
    }

    fn n_features(&self) -> usize {
        2
    }

    fn task(&self) -> Task {
        Task::MulticlassClassification { n_classes: 3 }
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0x5A1C_0004);
        let n = 15_000;
        let n_classes = 3;
        let samples_per_class = n / n_classes;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for class in 0..n_classes {
            let angle_offset = (class as f64) * 2.0 * std::f64::consts::PI / n_classes as f64;
            for j in 0..samples_per_class {
                let t = (j as f64) / (samples_per_class as f64) * 4.0 * std::f64::consts::PI;
                let r = t / (4.0 * std::f64::consts::PI) * 5.0;
                let noise_r = rng.normal(0.0, 0.25);
                let noise_theta = rng.normal(0.0, 0.1);
                let x = (r + noise_r) * (t + angle_offset + noise_theta).cos();
                let y = (r + noise_r) * (t + angle_offset + noise_theta).sin();
                features.push(vec![x, y]);
                targets.push(class as f64);
            }
        }

        // Shuffle deterministically using Fisher-Yates
        let total = features.len();
        for i in (1..total).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            features.swap(i, j);
            targets.swap(i, j);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 5. Sudden drift: abrupt distribution change at sample 25K of 50K
// ---------------------------------------------------------------------------

struct SuddenDrift;

impl BenchmarkDataset for SuddenDrift {
    fn name(&self) -> &str {
        "Sudden Drift (50K)"
    }

    fn n_features(&self) -> usize {
        5
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xABCD_0005);
        let n = 50_000;
        let switch_point = 25_000;
        let d = 5;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for i in 0..n {
            let x: Vec<f64> = (0..d).map(|_| rng.normal(0.0, 1.0)).collect();

            let y = if i < switch_point {
                // Regime 1: linear + mild nonlinearity
                x[0] * 3.0 + x[1] * 1.5 - x[2] * 2.0 + (x[3] * 0.5).sin() + rng.normal(0.0, 0.3)
            } else {
                // Regime 2: completely different function -- tests drift recovery
                -x[0] * 1.0 + x[2] * 4.0 + x[3] * x[4] * 2.0 - x[1].powi(2) * 0.5
                    + rng.normal(0.0, 0.3)
            };

            features.push(x);
            targets.push(y);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 6. High-dim nonlinear: 50 features (5 signal + 45 noise), regression, 20K
// ---------------------------------------------------------------------------

struct HighDimNonlinear;

impl BenchmarkDataset for HighDimNonlinear {
    fn name(&self) -> &str {
        "High-Dim Nonlinear (20K)"
    }

    fn n_features(&self) -> usize {
        50
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xAD10_0006);
        let n = 20_000;
        let d = 50;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for _ in 0..n {
            let x: Vec<f64> = (0..d).map(|_| rng.normal(0.0, 1.0)).collect();

            // Target = nonlinear combination of 5 key features + noise
            // f(x) = sin(x[0]*x[1]) + x[2]^2 - x[3]*x[4] + cos(x[0]+x[2]) + noise
            // Features 5-49 are noise dimensions
            let y = (x[0] * x[1]).sin() + x[2].powi(2) - x[3] * x[4]
                + (x[0] + x[2]).cos()
                + rng.normal(0.0, 0.3);

            features.push(x);
            targets.push(y);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 7. Temporal pattern: autoregressive 10-feature regression with 3-step lag
// ---------------------------------------------------------------------------

struct TemporalPattern;

impl BenchmarkDataset for TemporalPattern {
    fn name(&self) -> &str {
        "Temporal Pattern (30K)"
    }

    fn n_features(&self) -> usize {
        10
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0x7E3F_0007);
        let n = 30_000;
        let d = 10;
        let lag = 3;

        // Ring buffer for lagged features
        let mut history: Vec<Vec<f64>> = Vec::new();
        let lag_weights: [f64; 3] = [0.5, 0.3, 0.2]; // weights for lag 1,2,3

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        // State vector that evolves autoregressively
        let mut state: Vec<f64> = (0..d).map(|_| rng.normal(0.0, 1.0)).collect();

        for _ in 0..n {
            // Evolve state autoregressively: state[t] = 0.8*state[t-1] + noise
            for s in state.iter_mut() {
                *s = 0.8 * *s + rng.normal(0.0, 0.3);
            }

            let x = state.clone();

            // Target = weighted sum with 3-step lag
            // Current contribution
            let mut y = x[0] * 1.0 + x[1] * 0.5 - x[2] * 0.3;

            // Lagged contributions (if enough history)
            for (lag_idx, &w) in lag_weights.iter().enumerate() {
                let hist_idx = lag_idx + 1; // lag 1, 2, 3
                if history.len() >= hist_idx {
                    let past = &history[history.len() - hist_idx];
                    y += w * (past[3] + past[4] * 0.5);
                }
            }

            y += rng.normal(0.0, 0.2);

            features.push(x.clone());
            targets.push(y);

            // Maintain history ring buffer (keep only last `lag` entries)
            history.push(x);
            if history.len() > lag {
                history.remove(0);
            }
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 8. High-dim binary drift: 30 features, binary classification with
//    feature-level concept drift every 10K samples
// ---------------------------------------------------------------------------

struct HighDimBinaryDrift;

impl BenchmarkDataset for HighDimBinaryDrift {
    fn name(&self) -> &str {
        "High-Dim Binary Drift (30K)"
    }

    fn n_features(&self) -> usize {
        30
    }

    fn task(&self) -> Task {
        Task::BinaryClassification
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xBDF7_0008);
        let n = 30_000;
        let d = 30;

        // Three regimes of 10K samples each, with different relevant features
        // Regime 0: features 0-4 are relevant
        // Regime 1: features 10-14 are relevant
        // Regime 2: features 20-24 are relevant
        let regime_offsets = [0usize, 10, 20];

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for i in 0..n {
            let x: Vec<f64> = (0..d).map(|_| rng.normal(0.0, 1.0)).collect();

            // Select which 5 features are relevant based on regime
            let regime = (i / 10_000).min(2);
            let offset = regime_offsets[regime];

            // Nonlinear boundary using 5 relevant features
            let score = (x[offset] * x[offset + 1]).sin() + x[offset + 2].powi(2)
                - x[offset + 3] * x[offset + 4]
                + 0.5 * x[offset].cos();

            let noise = rng.normal(0.0, 0.3);
            let label = if score + noise > 0.5 { 1.0 } else { 0.0 };

            features.push(x);
            targets.push(label);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 9. Mackey-Glass: chaotic delay-differential time series, 30K samples
//    Standard reservoir computing benchmark (Jaeger 2001, Lukoševičius 2009).
//    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^10) - gamma * x(t)
//    Tests: temporal memory, nonlinear approximation
// ---------------------------------------------------------------------------

struct MackeyGlass;

impl BenchmarkDataset for MackeyGlass {
    fn name(&self) -> &str {
        "Mackey-Glass (30K)"
    }

    fn n_features(&self) -> usize {
        1
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xCA05_0009);
        let n = 30_000;
        let tau: usize = 17;
        let beta = 0.2;
        let gamma = 0.1;

        // Total time steps: tau warmup + n + 1 (for one-step-ahead target)
        let total = tau + n + 1;
        let mut x = vec![0.0; total];

        // Initialize first tau steps with small random values
        for i in 0..tau {
            x[i] = 0.9 + rng.uniform_range(-0.1, 0.1);
        }

        // Generate via discrete approximation of the delay-differential equation
        for t in tau..total {
            let x_tau = x[t - tau];
            x[t] = x[t - 1] + beta * x_tau / (1.0 + x_tau.powi(10)) - gamma * x[t - 1];
        }

        // Feature at time t = x(t), target = x(t+1) (one-step prediction)
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for i in 0..n {
            let t = tau + i;
            features.push(vec![x[t]]);
            targets.push(x[t + 1]);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 10. NARMA10: Nonlinear Auto-Regressive Moving Average order 10, 20K samples
//     Standard reservoir computing benchmark (Atiya & Parlos 2000).
//     y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i) for i=0..9) + 1.5*u(t-9)*u(t) + 0.1
//     Tests: long-range memory (10-step dependency), nonlinear mixing
// ---------------------------------------------------------------------------

struct Narma10;

impl BenchmarkDataset for Narma10 {
    fn name(&self) -> &str {
        "NARMA10 (20K)"
    }

    fn n_features(&self) -> usize {
        1
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xAA10_000A);
        let n = 20_000;
        let history_len = 10;

        // Pre-generate all input u(t) ~ uniform [0, 0.5]
        let total = n + history_len;
        let u: Vec<f64> = (0..total).map(|_| rng.uniform_range(0.0, 0.5)).collect();

        // y buffer, initialized to 0
        let mut y = vec![0.0; total];

        // Generate NARMA10 series starting from index history_len
        for t in history_len..total - 1 {
            let y_sum: f64 = (0..history_len).map(|i| y[t - i]).sum();
            y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y_sum + 1.5 * u[t - 9] * u[t] + 0.1;
            // Clamp to avoid divergence (NARMA10 can be unstable)
            y[t + 1] = y[t + 1].clamp(-10.0, 10.0);
        }

        // Feature = u(t), target = y(t), starting from history_len
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for i in 0..n {
            let t = history_len + i;
            features.push(vec![u[t]]);
            targets.push(y[t]);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 11. Lorenz Attractor: chaotic 3D dynamical system, 30K samples
//     dx = sigma*(y-x)*dt, dy = (x*(rho-z)-y)*dt, dz = (x*y-beta*z)*dt
//     Parameters: sigma=10, rho=28, beta=8/3, dt=0.01
//     Tests: high-dimensional chaotic prediction
// ---------------------------------------------------------------------------

struct LorenzAttractor;

impl BenchmarkDataset for LorenzAttractor {
    fn name(&self) -> &str {
        "Lorenz Attractor (30K)"
    }

    fn n_features(&self) -> usize {
        3
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = 30_000;
        let sigma = 10.0;
        let rho = 28.0;
        let beta_param = 8.0 / 3.0;
        let dt = 0.01;

        // Warmup of 1000 steps to reach the attractor, then n+1 for data
        let warmup = 1000;
        let total = warmup + n + 1;

        let mut xs = Vec::with_capacity(total);
        let mut ys_lorenz = Vec::with_capacity(total);
        let mut zs = Vec::with_capacity(total);

        // Initial conditions (standard starting point near attractor)
        xs.push(1.0);
        ys_lorenz.push(1.0);
        zs.push(1.0);

        // Integrate using 4th-order Runge-Kutta for accuracy
        for i in 0..total - 1 {
            let (xc, yc, zc) = (xs[i], ys_lorenz[i], zs[i]);

            // k1
            let dx1 = sigma * (yc - xc);
            let dy1 = xc * (rho - zc) - yc;
            let dz1 = xc * yc - beta_param * zc;

            // k2
            let (x2, y2, z2) = (
                xc + 0.5 * dt * dx1,
                yc + 0.5 * dt * dy1,
                zc + 0.5 * dt * dz1,
            );
            let dx2 = sigma * (y2 - x2);
            let dy2 = x2 * (rho - z2) - y2;
            let dz2 = x2 * y2 - beta_param * z2;

            // k3
            let (x3, y3, z3) = (
                xc + 0.5 * dt * dx2,
                yc + 0.5 * dt * dy2,
                zc + 0.5 * dt * dz2,
            );
            let dx3 = sigma * (y3 - x3);
            let dy3 = x3 * (rho - z3) - y3;
            let dz3 = x3 * y3 - beta_param * z3;

            // k4
            let (x4, y4, z4) = (xc + dt * dx3, yc + dt * dy3, zc + dt * dz3);
            let dx4 = sigma * (y4 - x4);
            let dy4 = x4 * (rho - z4) - y4;
            let dz4 = x4 * y4 - beta_param * z4;

            xs.push(xc + dt / 6.0 * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4));
            ys_lorenz.push(yc + dt / 6.0 * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4));
            zs.push(zc + dt / 6.0 * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4));
        }

        // Skip warmup, use [warmup..warmup+n] as features, predict x at t+1
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for i in 0..n {
            let t = warmup + i;
            features.push(vec![xs[t], ys_lorenz[t], zs[t]]);
            targets.push(xs[t + 1]);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 12. Compositional Function: 10-feature compositional nonlinearity, 20K samples
//     Designed for KAN -- tests compositional function learning, sparsity discovery.
//     target = sin(x[0]+x[1]) * exp(-x[2]^2) + sqrt(|x[3]*x[4]|) + cos(x[5])/(1+x[6]^2)
//     Features 7-9 are noise. Features 0-6 are uniform [-2, 2].
// ---------------------------------------------------------------------------

struct CompositionalFunction;

impl BenchmarkDataset for CompositionalFunction {
    fn name(&self) -> &str {
        "Compositional Function (20K)"
    }

    fn n_features(&self) -> usize {
        10
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xC0AF_000B);
        let n = 20_000;
        let d = 10;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for _ in 0..n {
            let x: Vec<f64> = (0..d)
                .map(|j| {
                    if j < 7 {
                        rng.uniform_range(-2.0, 2.0) // signal features
                    } else {
                        rng.normal(0.0, 1.0) // noise features
                    }
                })
                .collect();

            // Compositional target: each sub-expression uses a different nonlinearity
            let y = (x[0] + x[1]).sin() * (-x[2] * x[2]).exp()
                + (x[3] * x[4]).abs().sqrt()
                + x[5].cos() / (1.0 + x[6] * x[6]);

            features.push(x);
            targets.push(y);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 13. Non-Stationary Sequence: 5-phase distribution shift, 40K samples
//     Designed for TTT -- the generating function changes every 8K samples.
//     Phase 1: linear combination
//     Phase 2: nonlinear (sin/cos)
//     Phase 3: different linear combination
//     Phase 4: polynomial
//     Phase 5: back to phase 1 pattern (recurrent drift)
//     Tests: adaptation to distribution shift, recurrent patterns
// ---------------------------------------------------------------------------

struct NonStationarySequence;

impl BenchmarkDataset for NonStationarySequence {
    fn name(&self) -> &str {
        "Non-Stationary Sequence (40K)"
    }

    fn n_features(&self) -> usize {
        5
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0x7757_000C);
        let n = 40_000;
        let d = 5;
        let phase_len = 8_000;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for i in 0..n {
            let x: Vec<f64> = (0..d).map(|_| rng.normal(0.0, 1.0)).collect();
            let phase = i / phase_len;

            let y = match phase {
                0 | 4 => {
                    // Phase 1 & 5: linear combination (recurrent drift — same pattern returns)
                    2.0 * x[0] - 1.5 * x[1] + 0.8 * x[2] + 0.3 * x[3] - x[4] + rng.normal(0.0, 0.2)
                }
                1 => {
                    // Phase 2: nonlinear (sin/cos)
                    (x[0] * x[1]).sin() + (x[2] + x[3]).cos() + x[4].tanh() + rng.normal(0.0, 0.2)
                }
                2 => {
                    // Phase 3: different linear combination (orthogonal to phase 1)
                    -x[0] + 2.5 * x[2] - 0.5 * x[3] + 1.8 * x[4] + 0.1 * x[1] + rng.normal(0.0, 0.2)
                }
                _ => {
                    // Phase 4: polynomial
                    x[0].powi(2) - x[1].powi(2)
                        + x[2] * x[3]
                        + 0.5 * x[4].powi(3)
                        + rng.normal(0.0, 0.2)
                }
            };

            features.push(x);
            targets.push(y);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// 14. Sparse Temporal Events: 8-feature sparse binary events, 25K samples
//     Designed for SpikeNet -- sparse binary events with temporal patterns.
//     6/8 features are binary (0/1) with ~20% probability of being 1.
//     2/8 features are continuous.
//     Target depends on temporal patterns: specific binary feature combinations
//     in last 5 steps.
//     Tests: sparse event detection, temporal coincidence
// ---------------------------------------------------------------------------

struct SparseTemporalEvents;

impl BenchmarkDataset for SparseTemporalEvents {
    fn name(&self) -> &str {
        "Sparse Temporal Events (25K)"
    }

    fn n_features(&self) -> usize {
        8
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0x5B1E_000D);
        let n = 25_000;
        let lookback = 5;

        // Ring buffer of recent feature vectors for temporal patterns
        let mut history: Vec<Vec<f64>> = Vec::new();

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for _ in 0..n {
            // Generate 6 binary features (~20% active) + 2 continuous
            let x: Vec<f64> = (0..8)
                .map(|j| {
                    if j < 6 {
                        if rng.uniform() < 0.2 {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        rng.normal(0.0, 1.0)
                    }
                })
                .collect();

            // Target depends on temporal coincidence patterns in last 5 steps
            let mut y = 0.0;

            // Continuous features contribute a baseline
            y += 0.3 * x[6] + 0.2 * x[7];

            // Current binary features: simple contribution
            y += 0.5 * x[0] - 0.3 * x[1];

            // Temporal patterns: specific binary coincidences in history
            if history.len() >= 1 {
                let prev = &history[history.len() - 1];
                // Feature 0 active now AND feature 2 active 1 step ago
                if x[0] > 0.5 && prev[2] > 0.5 {
                    y += 2.0;
                }
                // Feature 3 active now AND feature 4 active 1 step ago
                if x[3] > 0.5 && prev[4] > 0.5 {
                    y -= 1.5;
                }
            }

            if history.len() >= 3 {
                let past3 = &history[history.len() - 3];
                // Feature 1 active now AND feature 5 active 3 steps ago
                if x[1] > 0.5 && past3[5] > 0.5 {
                    y += 1.8;
                }
            }

            if history.len() >= lookback {
                let past5 = &history[history.len() - lookback];
                // Feature 2 active now AND feature 0 active 5 steps ago
                if x[2] > 0.5 && past5[0] > 0.5 {
                    y += 1.2;
                }
                // Continuous feature modulates a historical binary event
                if past5[3] > 0.5 {
                    y += 0.5 * x[7];
                }
            }

            y += rng.normal(0.0, 0.1); // small noise

            features.push(x.clone());
            targets.push(y);

            // Maintain history ring buffer (keep last `lookback` entries)
            history.push(x);
            if history.len() > lookback {
                history.remove(0);
            }
        }

        (features, targets)
    }
}

// ===========================================================================
// Algorithm wrappers
// ===========================================================================

/// Wraps any StreamingLearner with a name for display purposes.
struct NamedModel {
    name: &'static str,
    model: Box<dyn StreamingLearner>,
    regression_only: bool,
}

fn build_algorithms(n_features: usize) -> Vec<NamedModel> {
    let mut algos = vec![
        NamedModel {
            name: "SGBT",
            model: Box::new(sgbt(50, 0.05)),
            regression_only: false,
        },
        NamedModel {
            name: "DistributionalSGBT",
            model: {
                let config = SGBTConfig::builder()
                    .n_steps(50)
                    .learning_rate(0.05)
                    .build()
                    .unwrap();
                Box::new(DistributionalSGBT::new(config))
            },
            regression_only: false,
        },
        NamedModel {
            name: "StreamingKAN",
            model: {
                // Input -> hidden -> output (20 hidden for more capacity)
                let layers = vec![n_features, 20, 1];
                Box::new(streaming_kan(&layers, 0.01))
            },
            regression_only: false,
        },
        NamedModel {
            name: "StreamingTTT",
            model: Box::new(streaming_ttt(32, 0.005)),
            regression_only: false,
        },
        NamedModel {
            name: "EchoStateNetwork",
            model: Box::new(esn(50, 0.9)),
            regression_only: false,
        },
    ];

    algos.push(NamedModel {
        name: "StreamingMamba",
        model: Box::new(mamba(n_features, 32)),
        regression_only: false,
    });

    algos.push(NamedModel {
        name: "RLS",
        model: Box::new(rls(0.999)),
        regression_only: false,
    });

    algos.push(NamedModel {
        name: "NeuralMoE(3)",
        model: {
            let moe = NeuralMoE::builder()
                .expert(sgbt(25, 0.05))
                .expert_with_warmup(esn(30, 0.9), 30)
                .expert(rls(0.999))
                .top_k(2)
                .build();
            Box::new(moe)
        },
        regression_only: false,
    });

    algos
}

// ===========================================================================
// Prequential evaluation
// ===========================================================================

#[allow(dead_code)]
struct BenchResult {
    algo_name: &'static str,
    metric_value: f64,
    metric_name: &'static str,
    throughput_sps: f64,
    peak_predict_us: f64,
    total_time_ms: f64,
    n_samples: usize,
}

/// Run prequential (test-then-train) evaluation for regression.
/// Returns RMSE and timing metrics.
fn prequential_regression(
    name: &'static str,
    model: &mut dyn StreamingLearner,
    features: &[Vec<f64>],
    targets: &[f64],
) -> BenchResult {
    let n = features.len();
    let mut sum_sq_error = 0.0;
    let mut count = 0u64;
    let mut peak_predict_ns: u64 = 0;

    let start = Instant::now();

    for i in 0..n {
        // TEST: predict before training
        let t0 = Instant::now();
        let pred = model.predict(&features[i]);
        let pred_ns = t0.elapsed().as_nanos() as u64;
        if pred_ns > peak_predict_ns {
            peak_predict_ns = pred_ns;
        }

        let err = targets[i] - pred;
        sum_sq_error += err * err;
        count += 1;

        // TRAIN: learn from this sample
        model.train(&features[i], targets[i]);
    }

    let elapsed = start.elapsed();
    let rmse = (sum_sq_error / count as f64).sqrt();

    BenchResult {
        algo_name: name,
        metric_value: rmse,
        metric_name: "RMSE",
        throughput_sps: n as f64 / elapsed.as_secs_f64(),
        peak_predict_us: peak_predict_ns as f64 / 1000.0,
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        n_samples: n,
    }
}

/// Run prequential (test-then-train) evaluation for binary classification.
/// Returns accuracy and timing metrics.
fn prequential_binary_classification(
    name: &'static str,
    model: &mut dyn StreamingLearner,
    features: &[Vec<f64>],
    targets: &[f64],
) -> BenchResult {
    let n = features.len();
    let mut correct = 0u64;
    let mut total = 0u64;
    let mut peak_predict_ns: u64 = 0;

    let start = Instant::now();

    for i in 0..n {
        // TEST: predict before training
        let t0 = Instant::now();
        let pred_raw = model.predict(&features[i]);
        let pred_ns = t0.elapsed().as_nanos() as u64;
        if pred_ns > peak_predict_ns {
            peak_predict_ns = pred_ns;
        }

        // Threshold at 0.5 for binary classification
        let pred_label = if pred_raw >= 0.5 { 1.0 } else { 0.0 };
        total += 1;
        if (pred_label - targets[i]).abs() < 0.5 {
            correct += 1;
        }

        // TRAIN: learn from this sample
        model.train(&features[i], targets[i]);
    }

    let elapsed = start.elapsed();
    let accuracy = correct as f64 / total as f64;

    BenchResult {
        algo_name: name,
        metric_value: accuracy,
        metric_name: "Accuracy",
        throughput_sps: n as f64 / elapsed.as_secs_f64(),
        peak_predict_us: peak_predict_ns as f64 / 1000.0,
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        n_samples: n,
    }
}

/// Run prequential (test-then-train) evaluation for multi-class classification.
/// Uses argmax over one-vs-rest models approximated by rounding predict() output.
/// Returns accuracy and timing metrics.
fn prequential_multiclass_classification(
    name: &'static str,
    model: &mut dyn StreamingLearner,
    features: &[Vec<f64>],
    targets: &[f64],
    n_classes: usize,
) -> BenchResult {
    let n = features.len();
    let mut correct = 0u64;
    let mut total = 0u64;
    let mut peak_predict_ns: u64 = 0;

    let start = Instant::now();

    for i in 0..n {
        // TEST: predict before training
        let t0 = Instant::now();
        let pred_raw = model.predict(&features[i]);
        let pred_ns = t0.elapsed().as_nanos() as u64;
        if pred_ns > peak_predict_ns {
            peak_predict_ns = pred_ns;
        }

        // Nearest-class: round and clamp to valid class range
        let pred_class = pred_raw.round().clamp(0.0, (n_classes - 1) as f64) as usize;
        let true_class = targets[i] as usize;
        total += 1;
        if pred_class == true_class {
            correct += 1;
        }

        // TRAIN: learn from this sample (class index as target)
        model.train(&features[i], targets[i]);
    }

    let elapsed = start.elapsed();
    let accuracy = correct as f64 / total as f64;

    BenchResult {
        algo_name: name,
        metric_value: accuracy,
        metric_name: "Accuracy",
        throughput_sps: n as f64 / elapsed.as_secs_f64(),
        peak_predict_us: peak_predict_ns as f64 / 1000.0,
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        n_samples: n,
    }
}

// ===========================================================================
// Table printing
// ===========================================================================

fn print_header(dataset_name: &str, n_samples: usize, n_features: usize, task: &Task) {
    let task_str = match task {
        Task::Regression => "regression".to_string(),
        Task::BinaryClassification => "binary classification".to_string(),
        Task::MulticlassClassification { n_classes } => {
            format!("{}-class classification", n_classes)
        }
    };
    eprintln!();
    eprintln!(
        "=== {} === ({} samples, {} features, {})",
        dataset_name, n_samples, n_features, task_str
    );
    eprintln!("Protocol: prequential (test-then-train)");
    eprintln!(
        "{:<22} {:>10} {:>12} {:>14} {:>12}",
        "Algorithm", "Metric", "Throughput", "Peak Pred", "Time(ms)"
    );
    eprintln!("{}", "-".repeat(74));
}

fn print_row(r: &BenchResult) {
    eprintln!(
        "{:<22} {:>10.4} {:>9.0} s/s {:>10.1} us {:>10.1}",
        r.algo_name, r.metric_value, r.throughput_sps, r.peak_predict_us, r.total_time_ms,
    );
}

// ===========================================================================
// Main benchmark runner
// ===========================================================================

fn run_dataset(dataset: &dyn BenchmarkDataset) {
    let (features, targets) = dataset.load();
    let n = features.len();
    let n_feat = dataset.n_features();
    let task = dataset.task();

    print_header(dataset.name(), n, n_feat, &task);

    let algorithms = build_algorithms(n_feat);

    for algo in algorithms {
        // Skip regression-only algorithms on classification tasks
        if algo.regression_only && task != Task::Regression {
            continue;
        }

        let mut model = algo.model;

        let result = match task {
            Task::Regression => {
                prequential_regression(algo.name, model.as_mut(), &features, &targets)
            }
            Task::BinaryClassification => {
                prequential_binary_classification(algo.name, model.as_mut(), &features, &targets)
            }
            Task::MulticlassClassification { n_classes } => prequential_multiclass_classification(
                algo.name,
                model.as_mut(),
                &features,
                &targets,
                n_classes,
            ),
        };

        print_row(&result);
    }
}

fn main() {
    eprintln!();
    eprintln!("============================================================");
    eprintln!("  irithyll real-world synthetic benchmark suite");
    eprintln!("  {} algorithms x 14 datasets, prequential evaluation", 13);
    eprintln!("============================================================");

    let datasets: Vec<Box<dyn BenchmarkDataset>> = vec![
        Box::new(SineRegression),
        Box::new(RotatingHyperplane),
        Box::new(SensorDrift),
        Box::new(MulticlassSpiral),
        Box::new(SuddenDrift),
        Box::new(HighDimNonlinear),
        Box::new(TemporalPattern),
        Box::new(HighDimBinaryDrift),
        Box::new(MackeyGlass),
        Box::new(Narma10),
        Box::new(LorenzAttractor),
        Box::new(CompositionalFunction),
        Box::new(NonStationarySequence),
        Box::new(SparseTemporalEvents),
    ];

    let overall_start = Instant::now();

    for dataset in &datasets {
        run_dataset(dataset.as_ref());
    }

    let total_elapsed = overall_start.elapsed();
    eprintln!();
    eprintln!("Total benchmark time: {:.1}s", total_elapsed.as_secs_f64());
    eprintln!();
}
