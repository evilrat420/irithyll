//! Expanded streaming ML benchmark suite for irithyll v9.8.5.
//!
//! Evaluates 8 streaming algorithms across 25 synthetic datasets using the
//! prequential (test-then-train) protocol. Datasets use the `irithyll::generators`
//! module where available, with inline deterministic generators for the rest.
//!
//! **Classification (binary):** 5 datasets
//! - SEA Concepts (100K, sudden drift)
//! - Rotating Hyperplane (50K, gradual drift)
//! - Agrawal (100K, sudden drift)
//! - Random RBF (50K, gradual nonlinear drift)
//! - Spike-Encoded Stream (50K, sparse binary)
//!
//! **Classification (multiclass):** 3 datasets
//! - LED (50K, 10-class, sudden drift)
//! - Waveform (50K, 3-class)
//! - Multi-class Spiral (15K, 3-class)
//!
//! **Regression:** 14 datasets
//! - Sine Regression (10K)
//! - Friedman #1 with drift (50K)
//! - Sensor Drift (30K)
//! - Mackey-Glass (30K, chaotic)
//! - Lorenz Attractor (30K, chaotic)
//! - NARMA10 (20K)
//! - Regime Shift (100K, 5 regimes) -- TTT showcase
//! - Continuous Drift (100K) -- TTT showcase
//! - Contextual Few-Shot (100K, blocks of 100) -- TTT showcase
//! - Contextual Few-Shot Short (100K, blocks of 50) -- TTT showcase
//! - Long-Seq Autoregressive (100K, 64 lags) -- Mamba showcase
//! - Compositional Physics (50K, 6 features) -- KAN showcase
//! - Feynman Physics (100K, 8 equations) -- KAN showcase
//! - Power Plant Regression (50K, 4 features)
//!
//! **Stress tests:** 3 datasets
//! - Sudden Drift (50K)
//! - High-Dim Nonlinear (20K, 50 features)
//! - Non-Stationary Sequence (40K)
//!
//! **Algorithms:**
//! - SGBT, Dist-SGBT (tree ensembles)
//! - ESN, Mamba, KAN, TTT, RLS, MoE (neural/linear models)
//! - Classification wrappers: binary_classifier() / multiclass_classifier()
//!
//! **Metrics:**
//! - Regression: RMSE
//! - Binary classification: Accuracy, Kappa, Kappa-T
//! - Multiclass classification: Accuracy, Kappa
//!
//! Run: `cargo bench --bench real_world_bench`

use irithyll::generators::{
    Agrawal, Friedman, Hyperplane, Lorenz, MackeyGlass, RandomRBF, StreamGenerator, Waveform, LED,
    SEA,
};
use irithyll::*;
use std::time::Instant;

// ===========================================================================
// Deterministic PRNG (xorshift64 -- for inline datasets only)
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
// BenchmarkDataset trait + Task enum
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
// Generator-backed datasets
// ===========================================================================

// ---------------------------------------------------------------------------
// SEA Concepts (100K, sudden drift)
// ---------------------------------------------------------------------------

struct SeaBenchmark;

impl BenchmarkDataset for SeaBenchmark {
    fn name(&self) -> &str {
        "SEA Concepts (100K)"
    }

    fn n_features(&self) -> usize {
        3
    }

    fn task(&self) -> Task {
        Task::BinaryClassification
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut gen = SEA::with_config(42, vec![25_000, 50_000, 75_000], 0.1);
        let n = 100_000;
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let (f, t) = gen.next_sample();
            features.push(f);
            targets.push(t);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Rotating Hyperplane (50K, gradual drift) -- generator replaces inline
// ---------------------------------------------------------------------------

struct HyperplaneBenchmark;

impl BenchmarkDataset for HyperplaneBenchmark {
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
        let mut gen = Hyperplane::new(0xBEEF_0002, 10, 3, 0.01, 0.05);
        let n = 50_000;
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let (f, t) = gen.next_sample();
            features.push(f);
            targets.push(t);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Agrawal (100K, sudden drift)
// ---------------------------------------------------------------------------

struct AgrawalBenchmark;

impl BenchmarkDataset for AgrawalBenchmark {
    fn name(&self) -> &str {
        "Agrawal (100K)"
    }

    fn n_features(&self) -> usize {
        9
    }

    fn task(&self) -> Task {
        Task::BinaryClassification
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        // Start with function 2, switch to function 5 at 50K
        let mut gen = Agrawal::with_config(42, 2, vec![(50_000, 5)], 0.05);
        let n = 100_000;
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let (f, t) = gen.next_sample();
            features.push(f);
            targets.push(t);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Random RBF (50K, gradual nonlinear drift)
// ---------------------------------------------------------------------------

struct RbfBenchmark;

impl BenchmarkDataset for RbfBenchmark {
    fn name(&self) -> &str {
        "Random RBF (50K)"
    }

    fn n_features(&self) -> usize {
        10
    }

    fn task(&self) -> Task {
        Task::BinaryClassification
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut gen = RandomRBF::with_config(42, 50, 10, 2, 0.0001);
        let n = 50_000;
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let (f, t) = gen.next_sample();
            features.push(f);
            targets.push(t);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// LED (50K, 10-class, sudden drift)
// ---------------------------------------------------------------------------

struct LedBenchmark;

impl BenchmarkDataset for LedBenchmark {
    fn name(&self) -> &str {
        "LED (50K)"
    }

    fn n_features(&self) -> usize {
        24
    }

    fn task(&self) -> Task {
        Task::MulticlassClassification { n_classes: 10 }
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut gen = LED::with_config(42, 0.1, vec![12_500, 25_000, 37_500]);
        let n = 50_000;
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let (f, t) = gen.next_sample();
            features.push(f);
            targets.push(t);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Waveform (50K, 3-class)
// ---------------------------------------------------------------------------

struct WaveformBenchmark;

impl BenchmarkDataset for WaveformBenchmark {
    fn name(&self) -> &str {
        "Waveform (50K)"
    }

    fn n_features(&self) -> usize {
        21
    }

    fn task(&self) -> Task {
        Task::MulticlassClassification { n_classes: 3 }
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut gen = Waveform::new(42);
        let n = 50_000;
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let (f, t) = gen.next_sample();
            features.push(f);
            targets.push(t);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Friedman #1 with drift (50K)
// ---------------------------------------------------------------------------

struct FriedmanBenchmark;

impl BenchmarkDataset for FriedmanBenchmark {
    fn name(&self) -> &str {
        "Friedman #1 Drift (50K)"
    }

    fn n_features(&self) -> usize {
        10
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut gen = Friedman::with_config(42, 1.0, 1.0);
        let n = 50_000;
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let (f, t) = gen.next_sample();
            features.push(f);
            targets.push(t);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Mackey-Glass (30K, chaotic) -- generator
// ---------------------------------------------------------------------------

struct MackeyGlassBenchmark;

impl BenchmarkDataset for MackeyGlassBenchmark {
    fn name(&self) -> &str {
        "Mackey-Glass (30K)"
    }

    fn n_features(&self) -> usize {
        6
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut gen = MackeyGlass::with_config(42, 17, 6, 500, 30_000);
        let n = 30_000;
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let (f, t) = gen.next_sample();
            features.push(f);
            targets.push(t);
        }
        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Lorenz (30K, chaotic) -- generator
// ---------------------------------------------------------------------------

struct LorenzBenchmark;

impl BenchmarkDataset for LorenzBenchmark {
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
        let mut gen = Lorenz::with_config(10.0, 28.0, 8.0 / 3.0, 0.01, 10, 1000, 30_000);
        let n = 30_000;
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);
        for _ in 0..n {
            let (f, t) = gen.next_sample();
            features.push(f);
            targets.push(t);
        }
        (features, targets)
    }
}

// ===========================================================================
// Inline datasets (kept from v9.0 bench)
// ===========================================================================

// ---------------------------------------------------------------------------
// Sine Regression (10K)
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
// Sensor Drift (30K)
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
            let regime = i / 5_000;
            let drift_offset = regime as f64 * 0.5;

            let x: Vec<f64> = (0..d)
                .map(|j| {
                    let base = rng.normal(0.0, 1.0);
                    if j % 2 == 0 {
                        base + drift_offset
                    } else {
                        base - drift_offset * 0.3
                    }
                })
                .collect();

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
// Multi-class Spiral (15K, 3-class)
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
// NARMA10 (20K)
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

        let total = n + history_len;
        let u: Vec<f64> = (0..total).map(|_| rng.uniform_range(0.0, 0.5)).collect();

        let mut y = vec![0.0; total];

        for t in history_len..total - 1 {
            let y_sum: f64 = (0..history_len).map(|i| y[t - i]).sum();
            y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y_sum + 1.5 * u[t - 9] * u[t] + 0.1;
            y[t + 1] = y[t + 1].clamp(-10.0, 10.0);
        }

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
// Sudden Drift (50K)
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
                x[0] * 3.0 + x[1] * 1.5 - x[2] * 2.0 + (x[3] * 0.5).sin() + rng.normal(0.0, 0.3)
            } else {
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
// High-Dim Nonlinear (20K, 50 features)
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
// Non-Stationary Sequence (40K)
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
                    2.0 * x[0] - 1.5 * x[1] + 0.8 * x[2] + 0.3 * x[3] - x[4] + rng.normal(0.0, 0.2)
                }
                1 => (x[0] * x[1]).sin() + (x[2] + x[3]).cos() + x[4].tanh() + rng.normal(0.0, 0.2),
                2 => {
                    -x[0] + 2.5 * x[2] - 0.5 * x[3] + 1.8 * x[4] + 0.1 * x[1] + rng.normal(0.0, 0.2)
                }
                _ => {
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

// ===========================================================================
// Neural architecture showcase datasets
// ===========================================================================

// ---------------------------------------------------------------------------
// Regime Shift Sequence (100K, 10 features, 5 regimes) -- TTT showcase
//
// 5 distinct regimes of 20K samples each with ABRUPT switches between
// completely different target functions. Tests rapid adaptation -- TTT's
// fast weights should adapt to each new regime faster than fixed models.
// ---------------------------------------------------------------------------

struct RegimeShiftSequence;

impl BenchmarkDataset for RegimeShiftSequence {
    fn name(&self) -> &str {
        "Regime Shift (100K)"
    }

    fn n_features(&self) -> usize {
        10
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xBEEF_1001);
        let n = 100_000;
        let d = 10;
        // 10 regime switches (5 functions cycling twice) — frequent enough
        // for TTT's fast weights to showcase rapid adaptation.
        let regime_len = 10_000;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for i in 0..n {
            let x: Vec<f64> = (0..d).map(|_| rng.uniform_range(-2.0, 2.0)).collect();
            let regime = (i / regime_len) % 5;

            let y = match regime {
                // Regime 0: smooth nonlinear
                0 => (x[0] * x[1]).sin() + x[2],
                // Regime 1: absolute value interaction
                1 => (x[3] - x[4]).abs() * x[5],
                // Regime 2: polynomial
                2 => x[0] * x[0] - x[1] * x[1] + x[2] * x[3],
                // Regime 3: saturating
                3 => (x[4] + x[5]).tanh() * x[6],
                // Regime 4: logarithmic
                _ => (1.0 + x[7].abs()).ln() + x[8] * x[9],
            };

            let noise = rng.normal(0.0, 0.1);
            features.push(x);
            targets.push(y + noise);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Continuous Feature Drift (100K, 10 features) -- TTT showcase #2
//
// The target depends on a WEIGHTED combination of features, where the weights
// ROTATE continuously. Every sample slightly shifts which features matter.
// TTT's per-sample fast weight adaptation should track the slowly rotating
// relevance vector better than fixed models. Unlike abrupt regime shift,
// this rewards smooth, continuous adaptation.
// ---------------------------------------------------------------------------

struct ContinuousFeatureDrift;

impl BenchmarkDataset for ContinuousFeatureDrift {
    fn name(&self) -> &str {
        "Continuous Drift (100K)"
    }

    fn n_features(&self) -> usize {
        10
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xBEEF_2001);
        let n = 100_000;
        let d = 10;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 / n as f64; // normalized time [0, 1)
            let x: Vec<f64> = (0..d).map(|_| rng.uniform_range(-2.0, 2.0)).collect();

            // Feature weights rotate continuously via sinusoidal modulation.
            // Each feature's weight has a different frequency and phase,
            // so the "important" features shift smoothly over time.
            let mut y = 0.0;
            for (j, xj) in x.iter().enumerate() {
                let freq = (j as f64 + 1.0) * 2.0; // frequencies: 2, 4, 6, ..., 20
                let phase = j as f64 * 0.7; // staggered phases
                let weight = (2.0 * std::f64::consts::PI * freq * t + phase).sin();
                y += weight * xj;
            }

            // Add a nonlinear interaction term that also drifts
            let interaction_weight = (4.0 * std::f64::consts::PI * t).cos();
            y += interaction_weight * x[0] * x[1];

            y += rng.normal(0.0, 0.1);
            features.push(x);
            targets.push(y);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Contextual Few-Shot (100K, 8 features) -- TTT showcase (few-shot adaptation)
//
// Samples come in BLOCKS of 100 (1000 blocks total). Within each block,
// the SAME random function (drawn from 4 types) with random weights applies
// to all 100 samples. Between blocks, the function and weights change.
// Feature[7] encodes position within the block (0.0 to 1.0).
//
// TTT should learn each block's function from the first ~10-20 samples
// via fast weight updates on the reconstruction loss. The correlated
// structure within each block is exactly what TTT was designed for.
// ---------------------------------------------------------------------------

struct ContextualFewShot;

impl BenchmarkDataset for ContextualFewShot {
    fn name(&self) -> &str {
        "Contextual Few-Shot (100K)"
    }

    fn n_features(&self) -> usize {
        8
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xBEEF_3001);
        let n = 100_000;
        let block_size = 100;
        let n_blocks = n / block_size;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for _block in 0..n_blocks {
            // Generate random weight vector for this block
            let w: Vec<f64> = (0..7).map(|_| rng.uniform_range(-2.0, 2.0)).collect();

            // Select function type for this block (0..4)
            let func_type = (rng.next_u64() % 4) as usize;

            for pos in 0..block_size {
                // Generate 7 input features, feature[7] = block position
                let mut x = Vec::with_capacity(8);
                for _ in 0..7 {
                    x.push(rng.uniform_range(-2.0, 2.0));
                }
                x.push(pos as f64 / (block_size - 1) as f64); // normalized position [0, 1]

                // Compute target based on function type and block weights
                let y = match func_type {
                    // Type 0: linear  y = w . x
                    0 => {
                        let mut s = 0.0;
                        for j in 0..7 {
                            s += w[j] * x[j];
                        }
                        s
                    }
                    // Type 1: quadratic  y = sum(w_i * x_i^2)
                    1 => {
                        let mut s = 0.0;
                        for j in 0..7 {
                            s += w[j] * x[j] * x[j];
                        }
                        s
                    }
                    // Type 2: sinusoidal  y = sum(w_i * sin(x_i))
                    2 => {
                        let mut s = 0.0;
                        for j in 0..7 {
                            s += w[j] * x[j].sin();
                        }
                        s
                    }
                    // Type 3: absolute value  y = sum(w_i * |x_i|)
                    _ => {
                        let mut s = 0.0;
                        for j in 0..7 {
                            s += w[j] * x[j].abs();
                        }
                        s
                    }
                };

                let noise = rng.normal(0.0, 0.1);
                features.push(x);
                targets.push(y + noise);
            }
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Contextual Few-Shot Long (100K, 8 features) -- TTT showcase (frequent switches)
//
// Same as ContextualFewShot but with blocks of 50 samples (2000 blocks).
// More frequent context switches = more TTT adaptation events.
// Shorter blocks mean the model has fewer samples to adapt, putting more
// pressure on fast few-shot learning.
// ---------------------------------------------------------------------------

struct ContextualFewShotLong;

impl BenchmarkDataset for ContextualFewShotLong {
    fn name(&self) -> &str {
        "Ctx Few-Shot Short (100K)"
    }

    fn n_features(&self) -> usize {
        8
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xBEEF_3002);
        let n = 100_000;
        let block_size = 50;
        let n_blocks = n / block_size;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for _block in 0..n_blocks {
            // Generate random weight vector for this block
            let w: Vec<f64> = (0..7).map(|_| rng.uniform_range(-2.0, 2.0)).collect();

            // Select function type for this block (0..4)
            let func_type = (rng.next_u64() % 4) as usize;

            for pos in 0..block_size {
                let mut x = Vec::with_capacity(8);
                for _ in 0..7 {
                    x.push(rng.uniform_range(-2.0, 2.0));
                }
                x.push(pos as f64 / (block_size - 1) as f64);

                let y = match func_type {
                    0 => {
                        let mut s = 0.0;
                        for j in 0..7 {
                            s += w[j] * x[j];
                        }
                        s
                    }
                    1 => {
                        let mut s = 0.0;
                        for j in 0..7 {
                            s += w[j] * x[j] * x[j];
                        }
                        s
                    }
                    2 => {
                        let mut s = 0.0;
                        for j in 0..7 {
                            s += w[j] * x[j].sin();
                        }
                        s
                    }
                    _ => {
                        let mut s = 0.0;
                        for j in 0..7 {
                            s += w[j] * x[j].abs();
                        }
                        s
                    }
                };

                let noise = rng.normal(0.0, 0.1);
                features.push(x);
                targets.push(y + noise);
            }
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Long-Sequence Autoregressive (100K, 64 lags) -- Mamba showcase
//
// Autoregressive time series with multi-scale temporal structure:
// sum of 5 sinusoids at different frequencies + AR(3) + slow amplitude
// modulation. Lag-embedded to 64 features. Mamba's SSM state should
// capture the multi-scale temporal patterns better than trees.
// ---------------------------------------------------------------------------

struct LongSequenceAutoregressive;

impl BenchmarkDataset for LongSequenceAutoregressive {
    fn name(&self) -> &str {
        "Long-Seq AR (100K)"
    }

    fn n_features(&self) -> usize {
        64
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xCAFE_2002);
        let n_output = 100_000;
        let n_lags: usize = 64;
        let total_series = n_output + n_lags + 3; // +3 for AR(3) warmup

        // Sinusoid parameters
        let freqs = [0.01, 0.03, 0.07, 0.13, 0.29];
        let amps = [1.0, 0.7, 0.5, 0.3, 0.2];
        let phases = [0.0, 1.2, 2.5, 0.8, 3.7];

        // Generate the full time series
        let mut series = vec![0.0; total_series];

        // Seed the first 3 values
        for val in series.iter_mut().take(3) {
            *val = rng.normal(0.0, 0.1);
        }

        let pi2 = 2.0 * std::f64::consts::PI;

        for t in 3..total_series {
            // Multi-scale sinusoidal base
            let mut base = 0.0;
            for k in 0..5 {
                base += amps[k] * (pi2 * freqs[k] * t as f64 + phases[k]).sin();
            }

            // AR(3) component
            let ar = 0.5 * series[t - 1] + 0.2 * series[t - 2] - 0.1 * series[t - 3];

            // Slow amplitude modulation
            let amp_mod = 1.0 + 0.5 * (pi2 * t as f64 / 5000.0).sin();

            let noise = rng.normal(0.0, 0.05);
            series[t] = amp_mod * base + ar + noise;

            // Clamp to prevent divergence from AR feedback
            series[t] = series[t].clamp(-20.0, 20.0);
        }

        // Create lag-embedded features: [x(t-64), x(t-63), ..., x(t-1)] -> target x(t)
        let offset = 3; // AR warmup offset
        let mut features = Vec::with_capacity(n_output);
        let mut targets = Vec::with_capacity(n_output);

        for i in 0..n_output {
            let t = offset + n_lags + i;
            let lags: Vec<f64> = (0..n_lags).map(|lag| series[t - n_lags + lag]).collect();
            features.push(lags);
            targets.push(series[t]);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Spike-Encoded Stream (50K, 20 binary features) -- SpikeNet showcase
//
// 5 signal channels converted to spikes via threshold crossing, plus
// 15 noise spike channels with low firing rate. Target classifies based
// on temporal coincidence patterns in the signal channels -- exactly what
// SNNs are built for.
// ---------------------------------------------------------------------------

struct SpikeEncodedStream;

impl BenchmarkDataset for SpikeEncodedStream {
    fn name(&self) -> &str {
        "Spike-Encoded (50K)"
    }

    fn n_features(&self) -> usize {
        20
    }

    fn task(&self) -> Task {
        Task::BinaryClassification
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xDEAD_3003);
        let n = 50_000;
        let n_signal = 5;
        let n_noise = 15;
        let spike_threshold = 0.15;
        let noise_fire_rate = 0.05;

        // Pre-generate underlying continuous signals (random walks)
        let mut signals = vec![vec![0.0f64; n]; n_signal];
        for signal in &mut signals {
            signal[0] = rng.normal(0.0, 0.5);
            for t in 1..n {
                signal[t] = signal[t - 1] + rng.normal(0.0, 0.1);
                // Mean-revert slowly to prevent drift to infinity
                signal[t] -= signal[t] * 0.001;
            }
        }

        // Convert to spike trains via threshold crossing
        let mut spike_trains = vec![vec![0.0f64; n]; n_signal];
        for ch in 0..n_signal {
            for t in 1..n {
                let delta = (signals[ch][t] - signals[ch][t - 1]).abs();
                if delta > spike_threshold {
                    spike_trains[ch][t] = 1.0;
                }
            }
        }

        // History buffers for temporal coincidence check
        // Class 1: if (spike[0] AND spike[1] within last 3 steps) AND (spike[2] within last 5 steps)
        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for t in 0..n {
            // Build feature vector: 5 signal spikes + 15 noise spikes
            let mut feat = Vec::with_capacity(n_signal + n_noise);

            // Signal channels
            for train in &spike_trains {
                feat.push(train[t]);
            }

            // Noise channels
            for _ in 0..n_noise {
                let spike = if rng.uniform() < noise_fire_rate {
                    1.0
                } else {
                    0.0
                };
                feat.push(spike);
            }

            // Determine target based on temporal coincidence
            let has_01_recent = {
                let lookback = 3usize;
                let start = t.saturating_sub(lookback);
                let mut found_0 = false;
                let mut found_1 = false;
                for s in start..=t {
                    if spike_trains[0][s] > 0.5 {
                        found_0 = true;
                    }
                    if spike_trains[1][s] > 0.5 {
                        found_1 = true;
                    }
                }
                found_0 && found_1
            };

            let has_2_recent = {
                let lookback = 5usize;
                let start = t.saturating_sub(lookback);
                let mut found = false;
                for s in start..=t {
                    if spike_trains[2][s] > 0.5 {
                        found = true;
                    }
                }
                found
            };

            let label = if has_01_recent && has_2_recent {
                1.0
            } else {
                0.0
            };

            features.push(feat);
            targets.push(label);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Compositional Physics (50K, 6 features) -- KAN showcase
//
// Target function with known compositional structure:
//   y = sin(x0 + x1^2) * exp(-x2/3) + sqrt(|x3*x4| + 0.1) * cos(x5)
//
// Composition of elementary operations -- exactly what KAN's B-spline
// edges decompose naturally. Trees approximate this poorly because it
// is smooth and compositional, not axis-aligned.
// ---------------------------------------------------------------------------

struct CompositionalPhysics;

impl BenchmarkDataset for CompositionalPhysics {
    fn name(&self) -> &str {
        "Compositional Physics (50K)"
    }

    fn n_features(&self) -> usize {
        6
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xFACE_4004);
        let n = 50_000;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for _ in 0..n {
            // x0-x2: uniform [0, 3], x3-x5: uniform [-2, 2]
            let x0 = rng.uniform_range(0.0, 3.0);
            let x1 = rng.uniform_range(0.0, 3.0);
            let x2 = rng.uniform_range(0.0, 3.0);
            let x3 = rng.uniform_range(-2.0, 2.0);
            let x4 = rng.uniform_range(-2.0, 2.0);
            let x5 = rng.uniform_range(-2.0, 2.0);

            let y = (x0 + x1 * x1).sin() * (-x2 / 3.0).exp()
                + ((x3 * x4).abs() + 0.1).sqrt() * x5.cos()
                + rng.normal(0.0, 0.05);

            features.push(vec![x0, x1, x2, x3, x4, x5]);
            targets.push(y);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Feynman Physics (100K, 6 features, 8 equations) -- KAN showcase
//
// 8 REAL Feynman symbolic regression equations cycled over 100K samples
// (~12.5K per equation). Features padded to 6 dimensions (max across
// equations). Creates implicit concept drift (function changes) AND tests
// compositional decomposition -- the exact domain KAN excels at.
// ---------------------------------------------------------------------------

struct FeynmanPhysics;

impl BenchmarkDataset for FeynmanPhysics {
    fn name(&self) -> &str {
        "Feynman Physics (100K)"
    }

    fn n_features(&self) -> usize {
        6
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xFE01_5001);
        let n = 100_000;
        let samples_per_eq = n / 8; // 12,500 each

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for i in 0..n {
            let eq_idx = i / samples_per_eq;

            let (feat, y) = match eq_idx.min(7) {
                // 1. Coulomb's law: F = q1*q2 / (4*pi*epsilon*r^2)
                0 => {
                    let q1 = rng.uniform_range(1.0, 5.0);
                    let q2 = rng.uniform_range(1.0, 5.0);
                    let epsilon = rng.uniform_range(1.0, 5.0);
                    let r = rng.uniform_range(1.0, 5.0);
                    let y = q1 * q2 / (4.0 * std::f64::consts::PI * epsilon * r * r);
                    (vec![q1, q2, epsilon, r, 0.0, 0.0], y)
                }
                // 2. Gravitational force: F = G*m1*m2/r^2
                1 => {
                    let g = rng.uniform_range(1.0, 5.0);
                    let m1 = rng.uniform_range(1.0, 5.0);
                    let m2 = rng.uniform_range(1.0, 5.0);
                    let r = rng.uniform_range(1.0, 5.0);
                    let y = g * m1 * m2 / (r * r);
                    (vec![g, m1, m2, r, 0.0, 0.0], y)
                }
                // 3. Kinetic energy: E = 0.5*m*v^2
                2 => {
                    let m = rng.uniform_range(1.0, 10.0);
                    let v = rng.uniform_range(1.0, 10.0);
                    let y = 0.5 * m * v * v;
                    (vec![m, v, 0.0, 0.0, 0.0, 0.0], y)
                }
                // 4. Spring potential: E = 0.5*k*x^2
                3 => {
                    let k = rng.uniform_range(1.0, 10.0);
                    let x = rng.uniform_range(-3.0, 3.0);
                    let y = 0.5 * k * x * x;
                    (vec![k, x, 0.0, 0.0, 0.0, 0.0], y)
                }
                // 5. Ideal gas law: P = nRT/V
                4 => {
                    let n_gas = rng.uniform_range(1.0, 5.0);
                    let r_const = rng.uniform_range(1.0, 5.0);
                    let t = rng.uniform_range(1.0, 5.0);
                    let v = rng.uniform_range(1.0, 5.0);
                    let y = n_gas * r_const * t / v;
                    (vec![n_gas, r_const, t, v, 0.0, 0.0], y)
                }
                // 6. Relativistic energy: E = m*c^2 / sqrt(1 - v^2/c^2)
                5 => {
                    let m = rng.uniform_range(1.0, 5.0);
                    let c = rng.uniform_range(5.0, 20.0);
                    let v = rng.uniform_range(1.0, 4.0); // v < c always
                    let ratio = v * v / (c * c);
                    let y = m * c * c / (1.0 - ratio).max(1e-10).sqrt();
                    (vec![m, c, v, 0.0, 0.0, 0.0], y)
                }
                // 7. Snell's law: theta2 = asin(n1*sin(theta1)/n2)
                6 => {
                    let n1 = rng.uniform_range(1.0, 3.0);
                    let theta1 = rng.uniform_range(0.1, 1.2);
                    let n2 = rng.uniform_range(1.0, 3.0);
                    let arg = n1 * theta1.sin() / n2;
                    // Clamp to valid asin domain
                    let y = arg.clamp(-1.0, 1.0).asin();
                    (vec![n1, theta1, n2, 0.0, 0.0, 0.0], y)
                }
                // 8. Planck radiation (simplified): E = 8*pi*freq^3 / (exp(freq/T) - 1)
                //    Using h=1, k=1, c=1 for simplicity
                _ => {
                    let freq = rng.uniform_range(0.1, 2.0);
                    let t = rng.uniform_range(100.0, 1000.0);
                    let exponent = (freq / t).min(500.0); // prevent overflow
                    let denom = exponent.exp() - 1.0;
                    let y = if denom.abs() > 1e-15 {
                        8.0 * std::f64::consts::PI * freq * freq * freq / denom
                    } else {
                        // Near-zero denominator: use first-order Taylor
                        8.0 * std::f64::consts::PI * freq * freq * t
                    };
                    (vec![freq, t, 0.0, 0.0, 0.0, 0.0], y)
                }
            };

            // Proportional noise: N(0, 0.01 * |y|)
            let noise = rng.normal(0.0, 0.01 * y.abs().max(1e-10));
            features.push(feat);
            targets.push(y + noise);
        }

        (features, targets)
    }
}

// ---------------------------------------------------------------------------
// Power Plant Regression (50K, 4 features) -- Engineering regression
//
// Based on UCI Combined Cycle Power Plant statistical properties.
// 4 features (temperature, ambient pressure, relative humidity, exhaust
// vacuum) predict net hourly electrical energy output. Includes gradual
// temperature drift simulating climate change over the stream.
// ---------------------------------------------------------------------------

struct PowerPlantRegression;

impl BenchmarkDataset for PowerPlantRegression {
    fn name(&self) -> &str {
        "Power Plant (50K)"
    }

    fn n_features(&self) -> usize {
        4
    }

    fn task(&self) -> Task {
        Task::Regression
    }

    fn load(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Rng::new(0xCC77_5001);
        let n = 50_000;

        let mut features = Vec::with_capacity(n);
        let mut targets = Vec::with_capacity(n);

        for i in 0..n {
            let progress = i as f64 / n as f64;

            // Temperature drifts upward over the stream (climate change simulation)
            let t_mean = 19.7 + 5.0 * progress;
            let t = rng.normal(t_mean, 7.5).clamp(1.8, 42.1);

            // Ambient pressure
            let ap = rng.normal(1013.0, 5.9);

            // Relative humidity
            let rh = rng.normal(73.3, 14.6).clamp(25.0, 100.0);

            // Exhaust vacuum
            let v = rng.normal(54.3, 12.7).clamp(25.0, 82.0);

            // Target: approximate real physics relationship
            // Higher T reduces efficiency, pressure helps, humidity hurts
            let pe = 480.0 - 2.0 * t + 0.05 * ap - 0.15 * rh - 0.5 * v + 0.01 * t * v
                - 0.003 * t * t
                + rng.normal(0.0, 3.0);

            features.push(vec![t, ap, rh, v]);
            targets.push(pe);
        }

        (features, targets)
    }
}

// ===========================================================================
// Algorithm construction
// ===========================================================================

/// Wraps any StreamingLearner with a name for display purposes.
struct NamedModel {
    name: &'static str,
    model: Box<dyn StreamingLearner>,
}

fn build_algorithms(n_features: usize, task: &Task) -> Vec<NamedModel> {
    let mut algos = vec![];

    // Tree models -- work natively with all tasks (thresholding-based classification)
    algos.push(NamedModel {
        name: "SGBT (50t)",
        model: Box::new(sgbt(50, 0.05)),
    });
    algos.push(NamedModel {
        name: "Dist-SGBT (50t)",
        model: {
            let config = SGBTConfig::builder()
                .n_steps(50)
                .learning_rate(0.05)
                .build()
                .unwrap();
            Box::new(DistributionalSGBT::new(config))
        },
    });

    // --- Helper closures for properly-configured neural models ---

    // ESN: 100 reservoir neurons (up from 50) for richer dynamics
    let make_esn = || esn(100, 0.95);

    // Mamba: n_state=64 (up from 32), forgetting 0.998 for longer memory
    let make_mamba = || {
        irithyll::ssm::StreamingMamba::new(
            irithyll::ssm::MambaConfig::builder()
                .d_in(n_features)
                .n_state(64)
                .forgetting_factor(0.998)
                .build()
                .expect("mamba config"),
        )
    };

    // KAN regression: shallow [n, 20, 1], high LR, no momentum (Hoang et al., 2026)
    let make_kan = || {
        irithyll::kan::StreamingKAN::new(
            irithyll::kan::KANConfig::builder()
                .layer_sizes(vec![n_features, 20, 1])
                .grid_size(8)
                .lr(0.1)
                .momentum(0.0)
                .build()
                .expect("kan config"),
        )
    };

    // KAN classification: shallow [n, 20, 1], lower LR for classification stability
    let make_kan_clf = || {
        irithyll::kan::StreamingKAN::new(
            irithyll::kan::KANConfig::builder()
                .layer_sizes(vec![n_features, 20, 1])
                .grid_size(8)
                .lr(0.05)
                .momentum(0.0)
                .build()
                .expect("kan clf config"),
        )
    };

    // TTT: d_model=64 (up from 32), eta=0.01, faster adaptation
    // TTT: d_model=64, Titans momentum=0.9 for gradient accumulation across
    // samples (the principled fix — single-sample rank-1 gradients are too noisy
    // without momentum to accumulate coherent direction).
    let make_ttt = || {
        irithyll::ttt::StreamingTTT::new(
            irithyll::ttt::TTTConfig::builder()
                .d_model(64)
                .eta(0.01)
                .alpha(0.01)
                .momentum(0.9)
                .forgetting_factor(0.995)
                .build()
                .expect("ttt config"),
        )
    };

    // MoE: bigger ESN expert, top-2 routing
    let make_moe = || {
        NeuralMoE::builder()
            .expert(sgbt(25, 0.05))
            .expert_with_warmup(esn(50, 0.95), 30)
            .expert(rls(0.999))
            .top_k(2)
            .build()
    };

    match task {
        Task::Regression => {
            algos.push(NamedModel {
                name: "ESN (100n)",
                model: Box::new(make_esn()),
            });
            algos.push(NamedModel {
                name: "Mamba (s=64)",
                model: Box::new(make_mamba()),
            });
            algos.push(NamedModel {
                name: "KAN [20]",
                model: Box::new(make_kan()),
            });
            algos.push(NamedModel {
                name: "TTT (d=64)",
                model: Box::new(make_ttt()),
            });
            algos.push(NamedModel {
                name: "RLS",
                model: Box::new(rls(0.999)),
            });
            algos.push(NamedModel {
                name: "MoE (3 experts)",
                model: Box::new(make_moe()),
            });
        }
        Task::BinaryClassification => {
            algos.push(NamedModel {
                name: "ESN-bin (100n)",
                model: Box::new(binary_classifier(make_esn())),
            });
            algos.push(NamedModel {
                name: "Mamba-bin (s=64)",
                model: Box::new(binary_classifier(make_mamba())),
            });
            algos.push(NamedModel {
                name: "KAN-bin [20]",
                model: Box::new(binary_classifier(make_kan_clf())),
            });
            algos.push(NamedModel {
                name: "TTT-bin (d=64)",
                model: Box::new(binary_classifier(make_ttt())),
            });
            algos.push(NamedModel {
                name: "RLS-bin",
                model: Box::new(binary_classifier(rls(0.999))),
            });
            algos.push(NamedModel {
                name: "MoE-bin (3 exp)",
                model: Box::new(binary_classifier(make_moe())),
            });
        }
        Task::MulticlassClassification { n_classes } => {
            let nc = *n_classes;
            algos.push(NamedModel {
                name: "ESN-mc (100n)",
                model: Box::new(multiclass_classifier(make_esn(), nc)),
            });
            algos.push(NamedModel {
                name: "Mamba-mc (s=64)",
                model: Box::new(multiclass_classifier(make_mamba(), nc)),
            });
            algos.push(NamedModel {
                name: "KAN-mc [20]",
                model: Box::new(multiclass_classifier(make_kan_clf(), nc)),
            });
            algos.push(NamedModel {
                name: "TTT-mc (d=64)",
                model: Box::new(multiclass_classifier(make_ttt(), nc)),
            });
            algos.push(NamedModel {
                name: "RLS-mc",
                model: Box::new(multiclass_classifier(rls(0.999), nc)),
            });
            algos.push(NamedModel {
                name: "MoE-mc (3 exp)",
                model: Box::new(multiclass_classifier(make_moe(), nc)),
            });
        }
    }

    algos
}

// ===========================================================================
// Metrics
// ===========================================================================

/// Cohen's Kappa: (accuracy - chance) / (1 - chance)
/// `class_counts_pred[c]` = number of predictions for class c
/// `class_counts_true[c]` = number of true labels for class c
fn cohens_kappa(
    correct: u64,
    total: u64,
    class_counts_pred: &[u64],
    class_counts_true: &[u64],
) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let acc = correct as f64 / total as f64;
    let n = total as f64;
    let chance: f64 = class_counts_pred
        .iter()
        .zip(class_counts_true.iter())
        .map(|(&p, &t)| (p as f64 / n) * (t as f64 / n))
        .sum();
    if (1.0 - chance).abs() < 1e-15 {
        return if acc >= 1.0 - 1e-15 { 1.0 } else { 0.0 };
    }
    (acc - chance) / (1.0 - chance)
}

/// Kappa-Temporal for binary classification:
/// kappa_t = (acc - acc_nochange) / (1 - acc_nochange)
/// acc_nochange = fraction of consecutive samples with same label
fn kappa_temporal(correct: u64, total: u64, same_as_prev_count: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let acc = correct as f64 / total as f64;
    // acc_nochange: if we predicted "same label as previous", how often would we be right?
    // = fraction of transitions where label didn't change
    let pairs = if total > 1 { total - 1 } else { 1 };
    let acc_nochange = same_as_prev_count as f64 / pairs as f64;
    if (1.0 - acc_nochange).abs() < 1e-15 {
        return if acc >= 1.0 - 1e-15 { 1.0 } else { 0.0 };
    }
    (acc - acc_nochange) / (1.0 - acc_nochange)
}

// ===========================================================================
// Prequential evaluation
// ===========================================================================

#[allow(dead_code)]
struct BenchResult {
    algo_name: &'static str,
    accuracy: f64,
    kappa: f64,
    kappa_t: f64,
    rmse: f64,
    throughput_sps: f64,
    peak_predict_us: f64,
    total_time_ms: f64,
    n_samples: usize,
    task: Task,
}

/// Run prequential (test-then-train) evaluation for regression.
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
        let t0 = Instant::now();
        let pred = model.predict(&features[i]);
        let pred_ns = t0.elapsed().as_nanos() as u64;
        if pred_ns > peak_predict_ns {
            peak_predict_ns = pred_ns;
        }

        let err = targets[i] - pred;
        sum_sq_error += err * err;
        count += 1;

        model.train(&features[i], targets[i]);
    }

    let elapsed = start.elapsed();
    let rmse = (sum_sq_error / count as f64).sqrt();

    BenchResult {
        algo_name: name,
        accuracy: f64::NAN,
        kappa: f64::NAN,
        kappa_t: f64::NAN,
        rmse,
        throughput_sps: n as f64 / elapsed.as_secs_f64(),
        peak_predict_us: peak_predict_ns as f64 / 1000.0,
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        n_samples: n,
        task: Task::Regression,
    }
}

/// Run prequential (test-then-train) evaluation for binary classification.
/// Computes Accuracy, Cohen's Kappa, and Kappa-Temporal.
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

    // For Kappa: class distribution tracking (2 classes)
    let mut pred_counts = [0u64; 2];
    let mut true_counts = [0u64; 2];

    // For Kappa-T: count consecutive same-label pairs
    let mut same_as_prev_count = 0u64;
    let mut prev_label: Option<usize> = None;

    let start = Instant::now();

    for i in 0..n {
        let t0 = Instant::now();
        let pred_raw = model.predict(&features[i]);
        let pred_ns = t0.elapsed().as_nanos() as u64;
        if pred_ns > peak_predict_ns {
            peak_predict_ns = pred_ns;
        }

        let pred_label = if pred_raw >= 0.5 { 1usize } else { 0usize };
        let true_label = if targets[i] >= 0.5 { 1usize } else { 0usize };

        total += 1;
        if pred_label == true_label {
            correct += 1;
        }
        pred_counts[pred_label] += 1;
        true_counts[true_label] += 1;

        // Kappa-T: track same-as-previous
        if let Some(prev) = prev_label {
            if true_label == prev {
                same_as_prev_count += 1;
            }
        }
        prev_label = Some(true_label);

        model.train(&features[i], targets[i]);
    }

    let elapsed = start.elapsed();
    let accuracy = correct as f64 / total as f64;
    let kappa = cohens_kappa(correct, total, &pred_counts, &true_counts);
    let kappa_t = kappa_temporal(correct, total, same_as_prev_count);

    BenchResult {
        algo_name: name,
        accuracy,
        kappa,
        kappa_t,
        rmse: f64::NAN,
        throughput_sps: n as f64 / elapsed.as_secs_f64(),
        peak_predict_us: peak_predict_ns as f64 / 1000.0,
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        n_samples: n,
        task: Task::BinaryClassification,
    }
}

/// Run prequential (test-then-train) evaluation for multi-class classification.
/// Computes Accuracy and Cohen's Kappa.
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

    // For Kappa: per-class distribution tracking
    let mut pred_counts = vec![0u64; n_classes];
    let mut true_counts = vec![0u64; n_classes];

    let start = Instant::now();

    for i in 0..n {
        let t0 = Instant::now();
        let pred_raw = model.predict(&features[i]);
        let pred_ns = t0.elapsed().as_nanos() as u64;
        if pred_ns > peak_predict_ns {
            peak_predict_ns = pred_ns;
        }

        let pred_class = pred_raw.round().clamp(0.0, (n_classes - 1) as f64) as usize;
        let true_class = targets[i] as usize;
        total += 1;
        if pred_class == true_class {
            correct += 1;
        }
        if pred_class < n_classes {
            pred_counts[pred_class] += 1;
        }
        if true_class < n_classes {
            true_counts[true_class] += 1;
        }

        model.train(&features[i], targets[i]);
    }

    let elapsed = start.elapsed();
    let accuracy = correct as f64 / total as f64;
    let kappa = cohens_kappa(correct, total, &pred_counts, &true_counts);

    BenchResult {
        algo_name: name,
        accuracy,
        kappa,
        kappa_t: f64::NAN,
        rmse: f64::NAN,
        throughput_sps: n as f64 / elapsed.as_secs_f64(),
        peak_predict_us: peak_predict_ns as f64 / 1000.0,
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        n_samples: n,
        task: Task::MulticlassClassification { n_classes },
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

    match task {
        Task::Regression => {
            eprintln!(
                "{:<22} {:>10} {:>12} {:>14} {:>12}",
                "Algorithm", "RMSE", "Throughput", "Peak Pred", "Time(ms)"
            );
        }
        Task::BinaryClassification => {
            eprintln!(
                "{:<22} {:>8} {:>8} {:>8} {:>12} {:>14} {:>12}",
                "Algorithm", "Acc", "Kappa", "Kappa-T", "Throughput", "Peak Pred", "Time(ms)"
            );
        }
        Task::MulticlassClassification { .. } => {
            eprintln!(
                "{:<22} {:>8} {:>8} {:>12} {:>14} {:>12}",
                "Algorithm", "Acc", "Kappa", "Throughput", "Peak Pred", "Time(ms)"
            );
        }
    }
    let width = match task {
        Task::Regression => 74,
        Task::BinaryClassification => 90,
        Task::MulticlassClassification { .. } => 82,
    };
    eprintln!("{}", "-".repeat(width));
}

fn print_row(r: &BenchResult) {
    match r.task {
        Task::Regression => {
            eprintln!(
                "{:<22} {:>10.4} {:>9.0} s/s {:>10.1} us {:>10.1}",
                r.algo_name, r.rmse, r.throughput_sps, r.peak_predict_us, r.total_time_ms,
            );
        }
        Task::BinaryClassification => {
            eprintln!(
                "{:<22} {:>8.4} {:>8.4} {:>8.4} {:>9.0} s/s {:>10.1} us {:>10.1}",
                r.algo_name,
                r.accuracy,
                r.kappa,
                r.kappa_t,
                r.throughput_sps,
                r.peak_predict_us,
                r.total_time_ms,
            );
        }
        Task::MulticlassClassification { .. } => {
            eprintln!(
                "{:<22} {:>8.4} {:>8.4} {:>9.0} s/s {:>10.1} us {:>10.1}",
                r.algo_name,
                r.accuracy,
                r.kappa,
                r.throughput_sps,
                r.peak_predict_us,
                r.total_time_ms,
            );
        }
    }
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

    let algorithms = build_algorithms(n_feat, &task);

    for algo in algorithms {
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
    eprintln!("  irithyll v9.8.5 expanded benchmark suite");
    eprintln!("  8+ algorithms x 25 datasets, prequential evaluation");
    eprintln!("  Classification wrappers: binary_classifier / multiclass_classifier");
    eprintln!("============================================================");

    let datasets: Vec<Box<dyn BenchmarkDataset>> = vec![
        // --- Binary Classification ---
        Box::new(SeaBenchmark),
        Box::new(HyperplaneBenchmark),
        Box::new(AgrawalBenchmark),
        Box::new(RbfBenchmark),
        Box::new(SpikeEncodedStream), // SpikeNet showcase
        // --- Multiclass Classification ---
        Box::new(LedBenchmark),
        Box::new(WaveformBenchmark),
        Box::new(MulticlassSpiral),
        // --- Regression ---
        Box::new(SineRegression),
        Box::new(FriedmanBenchmark),
        Box::new(SensorDrift),
        Box::new(MackeyGlassBenchmark),
        Box::new(LorenzBenchmark),
        Box::new(Narma10),
        Box::new(RegimeShiftSequence),        // TTT showcase (abrupt)
        Box::new(ContinuousFeatureDrift),     // TTT showcase (continuous)
        Box::new(ContextualFewShot),          // TTT showcase (few-shot)
        Box::new(ContextualFewShotLong),      // TTT showcase (few-shot, short blocks)
        Box::new(LongSequenceAutoregressive), // Mamba showcase
        Box::new(CompositionalPhysics),       // KAN showcase
        Box::new(FeynmanPhysics),             // KAN showcase (Feynman SR)
        Box::new(PowerPlantRegression),       // Engineering regression
        // --- Stress Tests ---
        Box::new(SuddenDrift),
        Box::new(HighDimNonlinear),
        Box::new(NonStationarySequence),
    ];

    let overall_start = Instant::now();

    for dataset in &datasets {
        run_dataset(dataset.as_ref());
    }

    let total_elapsed = overall_start.elapsed();
    eprintln!();
    eprintln!(
        "Total benchmark time: {:.1}s ({} datasets)",
        total_elapsed.as_secs_f64(),
        datasets.len()
    );
    eprintln!();
}
