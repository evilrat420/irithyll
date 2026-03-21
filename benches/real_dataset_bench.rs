//! Real dataset prequential benchmarks for irithyll.
//!
//! Evaluates SGBT on standard streaming ML datasets using the prequential
//! (interleaved test-then-train) protocol. This is the gold standard evaluation
//! method for streaming classifiers (Gama et al. 2013).
//!
//! Datasets:
//! - Electricity (Elec2): 45,312 samples, 8 features, binary classification
//! - Airlines: 539,383 samples, 7 features, binary classification
//! - Covertype: 581,012 samples, 54 features, 7-class classification
//!
//! Run: `cargo bench --bench real_dataset_bench`
//! Detailed: `cargo bench --bench real_dataset_bench -- detailed`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use irithyll::loss::logistic::LogisticLoss;
use irithyll::{MulticlassSGBT, SGBTConfig, Sample, SGBT};
use std::io::Write;
use std::time::Instant;

// ---------------------------------------------------------------------------
// CSV dataset loading
// ---------------------------------------------------------------------------

struct DatasetRow {
    features: Vec<f64>,
    label: f64, // binary: 0.0/1.0, multiclass: 0-based class index
}

/// Load the Electricity dataset from CSV.
///
/// Format: date,day,period,nswprice,nswdemand,vicprice,vicdemand,transfer,class
/// All features are pre-normalized floats. Class is UP (1.0) or DOWN (0.0).
fn load_electricity() -> Vec<DatasetRow> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/electricity.csv");
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {}: {}. Run `python datasets/download.py` first.",
            path, e
        )
    });

    let mut rows = Vec::with_capacity(45_312);
    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 9 {
            continue;
        }
        let features: Vec<f64> = fields[..8]
            .iter()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        if features.len() != 8 {
            continue;
        }
        let label = if fields[8].trim() == "UP" { 1.0 } else { 0.0 };
        rows.push(DatasetRow { features, label });
    }
    rows
}

/// Load the Airlines dataset from CSV.
///
/// Format: Airline,Flight,AirportFrom,AirportTo,DayOfWeek,Time,Length,Delay
/// Features 0-6 are already numeric (categoricals pre-encoded). Delay is 0/1.
fn load_airlines() -> Vec<DatasetRow> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/airlines.csv");
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut rows = Vec::with_capacity(540_000);
    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 8 {
            continue;
        }
        let features: Vec<f64> = fields[..7]
            .iter()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        if features.len() != 7 {
            continue;
        }
        let label = match fields[7].trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        rows.push(DatasetRow { features, label });
    }
    rows
}

/// Load the Covertype dataset from CSV.
///
/// Format: 54 features + Cover_Type (1-7)
/// Label stored as 0-based class index (subtract 1 from original 1-7).
fn load_covertype() -> Vec<DatasetRow> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/covertype.csv");
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut rows = Vec::with_capacity(581_012);
    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 55 {
            continue;
        }
        let features: Vec<f64> = fields[..54]
            .iter()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        if features.len() != 54 {
            continue;
        }
        let class_1based: f64 = match fields[54].trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let label = class_1based - 1.0; // 0-based
        rows.push(DatasetRow { features, label });
    }
    rows
}

// ---------------------------------------------------------------------------
// Prequential evaluation — binary classification
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct PrequentialResult {
    accuracy: f64,
    kappa: f64,
    total_time_ms: f64,
    throughput_sps: f64,
    n_samples: usize,
    checkpoints: Vec<(usize, f64, f64)>, // (step, accuracy, kappa)
    peak_memory_estimate: usize,
}

/// Compute binary Cohen's Kappa from confusion matrix.
fn binary_kappa(tp: u64, tn: u64, fp: u64, fn_: u64) -> f64 {
    let n = (tp + tn + fp + fn_) as f64;
    if n == 0.0 {
        return 0.0;
    }
    let p_o = (tp + tn) as f64 / n;
    let p_yes = ((tp + fp) as f64 / n) * ((tp + fn_) as f64 / n);
    let p_no = ((tn + fn_) as f64 / n) * ((tn + fp) as f64 / n);
    let p_e = p_yes + p_no;
    if (1.0 - p_e).abs() < 1e-12 {
        0.0
    } else {
        (p_o - p_e) / (1.0 - p_e)
    }
}

/// Run prequential (test-then-train) evaluation on a binary classification dataset.
fn prequential_binary(
    data: &[DatasetRow],
    config: SGBTConfig,
    checkpoint_interval: usize,
) -> PrequentialResult {
    let mut model: SGBT<LogisticLoss> = SGBT::with_loss(config, LogisticLoss);

    let mut correct: u64 = 0;
    let mut total: u64 = 0;
    let mut tp: u64 = 0;
    let mut tn: u64 = 0;
    let mut fp: u64 = 0;
    let mut fn_: u64 = 0;

    let mut checkpoints = Vec::new();

    let start = Instant::now();

    for (i, row) in data.iter().enumerate() {
        // TEST: predict before training
        let pred_raw = model.predict(&row.features);
        let pred_label = if pred_raw >= 0.5 { 1.0 } else { 0.0 };

        // UPDATE metrics
        total += 1;
        if (pred_label - row.label).abs() < 0.5 {
            correct += 1;
        }
        match (pred_label as u8, row.label as u8) {
            (1, 1) => tp += 1,
            (0, 0) => tn += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }

        // TRAIN: learn from this sample
        let sample = Sample::new(row.features.clone(), row.label);
        model.train_one(&sample);

        // Checkpoint
        if (i + 1) % checkpoint_interval == 0 {
            let acc = correct as f64 / total as f64;
            let kappa = binary_kappa(tp, tn, fp, fn_);
            checkpoints.push((i + 1, acc, kappa));
        }
    }

    let elapsed = start.elapsed();
    let total_time_ms = elapsed.as_secs_f64() * 1000.0;
    let n = data.len();
    let throughput_sps = n as f64 / elapsed.as_secs_f64();

    let accuracy = correct as f64 / total as f64;
    let kappa = binary_kappa(tp, tn, fp, fn_);

    // Memory estimate: shallow struct size + ~64 bytes per leaf node
    let shallow = std::mem::size_of_val(&model);
    let leaf_estimate = model.total_leaves() * 64;
    let peak_memory_estimate = shallow + leaf_estimate;

    PrequentialResult {
        accuracy,
        kappa,
        total_time_ms,
        throughput_sps,
        n_samples: n,
        checkpoints,
        peak_memory_estimate,
    }
}

// ---------------------------------------------------------------------------
// Prequential evaluation — multi-class classification
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct PrequentialMulticlassResult {
    accuracy: f64,
    kappa: f64,
    macro_f1: f64,
    total_time_ms: f64,
    throughput_sps: f64,
    n_samples: usize,
    n_classes: usize,
    checkpoints: Vec<(usize, f64, f64, f64)>, // (step, accuracy, kappa, macro_f1)
    peak_memory_estimate: usize,
}

/// Compute accuracy, Cohen's Kappa, and macro-F1 from a confusion matrix.
fn multiclass_metrics(confusion: &[Vec<u64>], n_classes: usize, total: u64) -> (f64, f64, f64) {
    let n = total as f64;
    if n == 0.0 {
        return (0.0, 0.0, 0.0);
    }

    // Accuracy
    let correct: u64 = (0..n_classes).map(|i| confusion[i][i]).sum();
    let accuracy = correct as f64 / n;

    // Cohen's Kappa (multi-class)
    let mut p_e = 0.0;
    for k in 0..n_classes {
        let row_total: u64 = confusion[k].iter().sum();
        let col_total: u64 = (0..n_classes).map(|j| confusion[j][k]).sum();
        p_e += (row_total as f64 / n) * (col_total as f64 / n);
    }
    let kappa = if (1.0 - p_e).abs() < 1e-12 {
        0.0
    } else {
        (accuracy - p_e) / (1.0 - p_e)
    };

    // Macro F1
    let mut f1_sum = 0.0;
    for k in 0..n_classes {
        let tp_k = confusion[k][k] as f64;
        let fp_k: f64 = (0..n_classes).map(|j| confusion[j][k]).sum::<u64>() as f64 - tp_k;
        let fn_k: f64 = confusion[k].iter().sum::<u64>() as f64 - tp_k;
        let precision = if tp_k + fp_k > 0.0 {
            tp_k / (tp_k + fp_k)
        } else {
            0.0
        };
        let recall = if tp_k + fn_k > 0.0 {
            tp_k / (tp_k + fn_k)
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        f1_sum += f1;
    }
    let macro_f1 = f1_sum / n_classes as f64;

    (accuracy, kappa, macro_f1)
}

/// Run prequential (test-then-train) evaluation on a multi-class dataset.
fn prequential_multiclass(
    data: &[DatasetRow],
    config: SGBTConfig,
    n_classes: usize,
    checkpoint_interval: usize,
) -> PrequentialMulticlassResult {
    let mut model = MulticlassSGBT::new(config, n_classes).unwrap();

    let mut total: u64 = 0;
    let mut confusion: Vec<Vec<u64>> = vec![vec![0u64; n_classes]; n_classes];
    let mut checkpoints = Vec::new();

    let start = Instant::now();

    for (i, row) in data.iter().enumerate() {
        // TEST: predict before training
        let pred_class = model.predict(&row.features);
        let true_class = row.label as usize;

        // UPDATE metrics
        total += 1;
        if true_class < n_classes && pred_class < n_classes {
            confusion[true_class][pred_class] += 1;
        }

        // TRAIN: learn from this sample
        let sample = Sample::new(row.features.clone(), row.label);
        model.train_one(&sample);

        // Checkpoint
        if (i + 1) % checkpoint_interval == 0 {
            let (acc, kappa, macro_f1) = multiclass_metrics(&confusion, n_classes, total);
            checkpoints.push((i + 1, acc, kappa, macro_f1));
        }
    }

    let elapsed = start.elapsed();
    let total_time_ms = elapsed.as_secs_f64() * 1000.0;
    let n = data.len();
    let throughput_sps = n as f64 / elapsed.as_secs_f64();

    let (accuracy, kappa, macro_f1) = multiclass_metrics(&confusion, n_classes, total);

    // Memory estimate: shallow struct only (committee internals are private)
    let peak_memory_estimate = std::mem::size_of_val(&model);

    PrequentialMulticlassResult {
        accuracy,
        kappa,
        macro_f1,
        total_time_ms,
        throughput_sps,
        n_samples: n,
        n_classes,
        checkpoints,
        peak_memory_estimate,
    }
}

// ---------------------------------------------------------------------------
// Learning curve CSV writers
// ---------------------------------------------------------------------------

fn results_dir() -> String {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/results");
    std::fs::create_dir_all(dir).ok();
    dir.to_string()
}

fn write_binary_learning_curve(filename: &str, checkpoints: &[(usize, f64, f64)]) {
    let dir = results_dir();
    let path = format!("{}/{}", dir, filename);
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "step,accuracy,kappa").unwrap();
    for &(step, acc, kappa) in checkpoints {
        writeln!(f, "{},{:.6},{:.6}", step, acc, kappa).unwrap();
    }
    eprintln!("  Learning curve -> {}", path);
}

fn write_multiclass_learning_curve(filename: &str, checkpoints: &[(usize, f64, f64, f64)]) {
    let dir = results_dir();
    let path = format!("{}/{}", dir, filename);
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "step,accuracy,kappa,macro_f1").unwrap();
    for &(step, acc, kappa, f1) in checkpoints {
        writeln!(f, "{},{:.6},{:.6},{:.6}", step, acc, kappa, f1).unwrap();
    }
    eprintln!("  Learning curve -> {}", path);
}

// ---------------------------------------------------------------------------
// Criterion benchmarks (timing — Electricity only, small enough for iteration)
// ---------------------------------------------------------------------------

fn electricity_prequential(c: &mut Criterion) {
    let data = load_electricity();
    assert!(
        data.len() > 40_000,
        "Electricity dataset should have ~45K rows, got {}",
        data.len()
    );

    let mut group = c.benchmark_group("electricity_prequential");
    group.sample_size(10); // full dataset runs, fewer samples needed
    group.throughput(Throughput::Elements(data.len() as u64));

    let configs = vec![
        ("10t_d4", 10, 4),
        ("25t_d4", 25, 4),
        ("50t_d4", 50, 4),
        ("50t_d6", 50, 6),
        ("100t_d4", 100, 4),
    ];

    for (label, n_steps, max_depth) in &configs {
        let config = SGBTConfig::builder()
            .n_steps(*n_steps)
            .learning_rate(0.05)
            .grace_period(50)
            .max_depth(*max_depth)
            .n_bins(32)
            .initial_target_count(50)
            .build()
            .unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(label), &config, |b, cfg| {
            b.iter(|| prequential_binary(&data, cfg.clone(), 5000));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Detailed results — Electricity
// ---------------------------------------------------------------------------

fn electricity_detailed_results(c: &mut Criterion) {
    let _ = c;
    let data = load_electricity();

    let configs: Vec<(&str, usize, usize, f64)> = vec![
        ("sgbt_25t_d4", 25, 4, 0.05),
        ("sgbt_50t_d4", 50, 4, 0.05),
        ("sgbt_50t_d6", 50, 6, 0.05),
        ("sgbt_100t_d4", 100, 4, 0.05),
        ("sgbt_100t_d6", 100, 6, 0.05),
        ("sgbt_50t_d6_lr1", 50, 6, 0.1),
        ("sgbt_100t_d6_lr1", 100, 6, 0.1),
    ];

    eprintln!("\n=== Electricity Dataset Prequential Results ===");
    eprintln!(
        "Dataset: {} samples, 8 features, binary classification",
        data.len()
    );
    eprintln!("Protocol: prequential (test-then-train)");
    eprintln!(
        "{:<25} {:>8} {:>8} {:>10} {:>12} {:>10}",
        "Model", "Acc", "Kappa", "Time(ms)", "Throughput", "Mem(KB)"
    );
    eprintln!("{}", "-".repeat(80));

    for (label, n_steps, max_depth, lr) in &configs {
        let config = SGBTConfig::builder()
            .n_steps(*n_steps)
            .learning_rate(*lr)
            .grace_period(50)
            .max_depth(*max_depth)
            .n_bins(32)
            .initial_target_count(50)
            .build()
            .unwrap();

        let result = prequential_binary(&data, config, 5000);

        eprintln!(
            "{:<25} {:>7.4} {:>7.4} {:>9.1} {:>10.0} s/s {:>8.1}",
            label,
            result.accuracy,
            result.kappa,
            result.total_time_ms,
            result.throughput_sps,
            result.peak_memory_estimate as f64 / 1024.0,
        );

        // Write learning curve for best config
        if *label == "sgbt_100t_d6_lr1" {
            write_binary_learning_curve("electricity_learning_curve.csv", &result.checkpoints);
        }
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Detailed results — Airlines
// ---------------------------------------------------------------------------

fn airlines_detailed_results(c: &mut Criterion) {
    let _ = c;
    let data = load_airlines();
    if data.is_empty() {
        eprintln!("\n[SKIP] Airlines dataset not found. Run `python datasets/download.py`");
        return;
    }

    let configs: Vec<(&str, usize, usize, f64)> = vec![
        ("sgbt_25t_d4", 25, 4, 0.05),
        ("sgbt_50t_d4", 50, 4, 0.05),
        ("sgbt_50t_d6", 50, 6, 0.05),
        ("sgbt_100t_d4", 100, 4, 0.05),
        ("sgbt_100t_d6", 100, 6, 0.05),
        ("sgbt_50t_d6_lr1", 50, 6, 0.1),
        ("sgbt_100t_d6_lr1", 100, 6, 0.1),
    ];

    eprintln!("\n=== Airlines Dataset Prequential Results ===");
    eprintln!(
        "Dataset: {} samples, 7 features, binary classification",
        data.len()
    );
    eprintln!("Protocol: prequential (test-then-train)");
    eprintln!(
        "{:<25} {:>8} {:>8} {:>10} {:>12} {:>10}",
        "Model", "Acc", "Kappa", "Time(ms)", "Throughput", "Mem(KB)"
    );
    eprintln!("{}", "-".repeat(80));

    for (label, n_steps, max_depth, lr) in &configs {
        let config = SGBTConfig::builder()
            .n_steps(*n_steps)
            .learning_rate(*lr)
            .grace_period(50)
            .max_depth(*max_depth)
            .n_bins(32)
            .initial_target_count(50)
            .build()
            .unwrap();

        let result = prequential_binary(&data, config, 50_000);

        eprintln!(
            "{:<25} {:>7.4} {:>7.4} {:>9.1} {:>10.0} s/s {:>8.1}",
            label,
            result.accuracy,
            result.kappa,
            result.total_time_ms,
            result.throughput_sps,
            result.peak_memory_estimate as f64 / 1024.0,
        );

        // Write learning curve for best config
        if *label == "sgbt_100t_d6_lr1" {
            write_binary_learning_curve("airlines_learning_curve.csv", &result.checkpoints);
        }
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Detailed results — Covertype (multi-class)
// ---------------------------------------------------------------------------

fn covertype_detailed_results(c: &mut Criterion) {
    let _ = c;
    let data = load_covertype();
    if data.is_empty() {
        eprintln!("\n[SKIP] Covertype dataset not found. Run `python datasets/download.py`");
        return;
    }

    let configs: Vec<(&str, usize, usize, f64)> = vec![
        ("sgbt_25t_d4", 25, 4, 0.05),
        ("sgbt_50t_d6", 50, 6, 0.05),
        ("sgbt_50t_d6_lr1", 50, 6, 0.1),
        ("sgbt_100t_d6_lr1", 100, 6, 0.1),
    ];

    eprintln!("\n=== Covertype Dataset Prequential Results ===");
    eprintln!(
        "Dataset: {} samples, 54 features, 7-class classification",
        data.len()
    );
    eprintln!("Protocol: prequential (test-then-train)");
    eprintln!(
        "{:<25} {:>8} {:>8} {:>8} {:>10} {:>12} {:>10}",
        "Model", "Acc", "Kappa", "F1", "Time(ms)", "Throughput", "Mem(KB)"
    );
    eprintln!("{}", "-".repeat(90));

    for (label, n_steps, max_depth, lr) in &configs {
        let config = SGBTConfig::builder()
            .n_steps(*n_steps)
            .learning_rate(*lr)
            .grace_period(50)
            .max_depth(*max_depth)
            .n_bins(32)
            .initial_target_count(50)
            .build()
            .unwrap();

        let result = prequential_multiclass(&data, config, 7, 50_000);

        eprintln!(
            "{:<25} {:>7.4} {:>7.4} {:>7.4} {:>9.1} {:>10.0} s/s {:>8.1}",
            label,
            result.accuracy,
            result.kappa,
            result.macro_f1,
            result.total_time_ms,
            result.throughput_sps,
            result.peak_memory_estimate as f64 / 1024.0,
        );

        // Write learning curve for best config
        if *label == "sgbt_100t_d6_lr1" {
            write_multiclass_learning_curve("covertype_learning_curve.csv", &result.checkpoints);
        }
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    real_datasets,
    electricity_prequential,
    electricity_detailed_results,
    airlines_detailed_results,
    covertype_detailed_results,
);
criterion_main!(real_datasets);
