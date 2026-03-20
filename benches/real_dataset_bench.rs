//! Real dataset prequential benchmarks for irithyll.
//!
//! Evaluates SGBT on standard streaming ML datasets using the prequential
//! (interleaved test-then-train) protocol. This is the gold standard evaluation
//! method for streaming classifiers (Gama et al. 2013).
//!
//! Datasets:
//! - Electricity (Elec2): 45,312 samples, 8 features, binary classification
//!   Source: Australian NSW Electricity Market (Harries 1999)
//!
//! Run: `cargo bench --bench real_dataset_bench`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use irithyll::loss::logistic::LogisticLoss;
use irithyll::{SGBTConfig, Sample, SGBT};
use std::time::Instant;

// ---------------------------------------------------------------------------
// CSV dataset loading
// ---------------------------------------------------------------------------

struct DatasetRow {
    features: Vec<f64>,
    label: f64, // 1.0 or 0.0 for binary classification
}

/// Load the Electricity dataset from CSV.
///
/// Format: date,day,period,nswprice,nswdemand,vicprice,vicdemand,transfer,class
/// All features are pre-normalized floats. Class is UP (1.0) or DOWN (0.0).
fn load_electricity() -> Vec<DatasetRow> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/electricity.csv");
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {}: {}. Run the dataset download script first.",
            path, e
        )
    });

    let mut rows = Vec::with_capacity(45_312);
    for line in content.lines().skip(1) {
        // skip header
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

// ---------------------------------------------------------------------------
// Prequential evaluation
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct PrequentialResult {
    accuracy: f64,
    kappa: f64,
    total_time_ms: f64,
    throughput_sps: f64,
    n_samples: usize,
    checkpoints: Vec<(usize, f64)>,
}

/// Run prequential (test-then-train) evaluation on a binary classification dataset.
///
/// Protocol:
/// 1. Predict on sample (threshold 0.5)
/// 2. Update metrics
/// 3. Train on sample
fn prequential_binary(
    data: &[DatasetRow],
    config: SGBTConfig,
    checkpoint_interval: usize,
) -> PrequentialResult {
    let mut model: SGBT<LogisticLoss> = SGBT::with_loss(config, LogisticLoss);

    let mut correct: u64 = 0;
    let mut total: u64 = 0;
    // For Cohen's Kappa: confusion matrix
    let mut tp: u64 = 0;
    let mut tn: u64 = 0;
    let mut fp: u64 = 0;
    let mut r#fn: u64 = 0;

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
        // Confusion matrix
        match (pred_label as u8, row.label as u8) {
            (1, 1) => tp += 1,
            (0, 0) => tn += 1,
            (1, 0) => fp += 1,
            (0, 1) => r#fn += 1,
            _ => {}
        }

        // TRAIN: learn from this sample
        let sample = Sample::new(row.features.clone(), row.label);
        model.train_one(&sample);

        // Checkpoint
        if (i + 1) % checkpoint_interval == 0 {
            let acc = correct as f64 / total as f64;
            checkpoints.push((i + 1, acc));
        }
    }

    let elapsed = start.elapsed();
    let total_time_ms = elapsed.as_secs_f64() * 1000.0;
    let n = data.len();
    let throughput_sps = n as f64 / elapsed.as_secs_f64();

    let accuracy = correct as f64 / total as f64;

    // Cohen's Kappa: kappa = (p_o - p_e) / (1 - p_e)
    let n_f = total as f64;
    let p_o = accuracy;
    let p_yes = ((tp + fp) as f64 / n_f) * ((tp + r#fn) as f64 / n_f);
    let p_no = ((tn + r#fn) as f64 / n_f) * ((tn + fp) as f64 / n_f);
    let p_e = p_yes + p_no;
    let kappa = if (1.0 - p_e).abs() < 1e-12 {
        0.0
    } else {
        (p_o - p_e) / (1.0 - p_e)
    };

    PrequentialResult {
        accuracy,
        kappa,
        total_time_ms,
        throughput_sps,
        n_samples: n,
        checkpoints,
    }
}

// ---------------------------------------------------------------------------
// Criterion benchmarks
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

    // Test multiple SGBT configurations
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

/// Non-Criterion benchmark that prints detailed results to stdout.
/// Run with: `cargo bench --bench real_dataset_bench -- --ignored`
/// Or directly: `cargo test --bench real_dataset_bench -- --ignored --nocapture`
fn electricity_detailed_results(c: &mut Criterion) {
    // This benchmark exists to print results, not for Criterion timing
    let _ = c;

    let data = load_electricity();

    let configs = vec![
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
        "{:<25} {:>8} {:>8} {:>10} {:>12}",
        "Model", "Acc", "Kappa", "Time(ms)", "Throughput"
    );
    eprintln!("{}", "-".repeat(70));

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
            "{:<25} {:>7.4} {:>7.4} {:>9.1} {:>10.0} s/s",
            label, result.accuracy, result.kappa, result.total_time_ms, result.throughput_sps,
        );
    }
    eprintln!();
}

criterion_group!(
    real_datasets,
    electricity_prequential,
    electricity_detailed_results,
);
criterion_main!(real_datasets);
