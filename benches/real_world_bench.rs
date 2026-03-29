//! Real-world synthetic benchmark harness for irithyll streaming algorithms.
//!
//! Evaluates 8 streaming algorithms across 5 synthetic datasets using the
//! prequential (test-then-train) protocol. All datasets are generated inline
//! with deterministic seeded RNG -- no file downloads needed for CI.
//!
//! Datasets:
//! - **Sine regression**: x -> sin(x) + noise, 10K samples
//! - **Rotating hyperplane**: concept drift classification, 50K samples
//! - **Sensor drift**: 6 features with gradual drift every 5K samples, 30K samples
//! - **Multi-class spiral**: 3 interlocking spirals, 15K samples
//! - **Sudden drift**: abrupt distribution change at sample 25K of 50K
//!
//! Algorithms:
//! - SGBT (default config)
//! - DistributionalSGBT
//! - StreamingKAN
//! - StreamingTTT
//! - EchoStateNetwork
//! - StreamingMamba
//! - RecursiveLeastSquares (regression only)
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
                // Input -> hidden -> output
                let hidden = n_features.max(5);
                let layers = vec![n_features, hidden, 1];
                Box::new(streaming_kan(&layers, 0.01))
            },
            regression_only: false,
        },
        NamedModel {
            name: "StreamingTTT",
            model: Box::new(streaming_ttt(16, 0.005)),
            regression_only: false,
        },
        NamedModel {
            name: "EchoStateNetwork",
            model: Box::new(esn(50, 0.9)),
            regression_only: false,
        },
    ];

    // StreamingMamba requires d_in >= 2 (internal SIMD constraint)
    if n_features >= 2 {
        algos.push(NamedModel {
            name: "StreamingMamba",
            model: Box::new(mamba(n_features, 16)),
            regression_only: false,
        });
    }

    algos.push(NamedModel {
        name: "RLS",
        model: Box::new(rls(0.999)),
        regression_only: true,
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
    _n_classes: usize,
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

        // Round to nearest class
        let pred_class = pred_raw.round().max(0.0) as usize;
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
    eprintln!("  {} algorithms x 5 datasets, prequential evaluation", 8);
    eprintln!("============================================================");

    let datasets: Vec<Box<dyn BenchmarkDataset>> = vec![
        Box::new(SineRegression),
        Box::new(RotatingHyperplane),
        Box::new(SensorDrift),
        Box::new(MulticlassSpiral),
        Box::new(SuddenDrift),
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
