//! Cross-crate prediction latency comparison benchmarks.
//!
//! Compares irithyll's three inference paths (SGBT f64, packed f32, quantized i16)
//! against gbdt-rs (the closest pure-Rust GBDT implementation) on equivalent models.
//!
//! All models are trained on the same deterministic synthetic data.
//!
//! Run: `cd comparison && cargo bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use irithyll::export_embedded::{export_packed, export_packed_i16};
use irithyll::{SGBTConfig, Sample, SGBT};
use irithyll_core::{EnsembleView, QuantizedEnsembleView};

use gbdt::config::Config as GBDTConfig;
use gbdt::decision_tree::Data;
use gbdt::gradient_boost::GBDT;

// ---------------------------------------------------------------------------
// Deterministic PRNG (no rand dependency)
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const N_TREES: usize = 50;
const MAX_DEPTH: usize = 4;
const N_FEATURES: usize = 10;
const N_TRAIN: usize = 1000;
const BATCH_SIZE: usize = 10_000;

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

fn generate_train_data(
    n: usize,
    n_features: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = seed;
    let mut features = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);
    for _ in 0..n {
        let f: Vec<f64> = (0..n_features)
            .map(|_| xorshift64(&mut rng) * 10.0 - 5.0)
            .collect();
        let t = f
            .iter()
            .enumerate()
            .fold(0.0, |acc, (i, &v)| acc + (i as f64 + 1.0) * v);
        features.push(f);
        targets.push(t);
    }
    (features, targets)
}

fn generate_f64_samples(n: usize, n_features: usize) -> Vec<Vec<f64>> {
    let mut rng: u64 = 0xDEAD_1337;
    (0..n)
        .map(|_| {
            (0..n_features)
                .map(|_| xorshift64(&mut rng) * 10.0 - 5.0)
                .collect()
        })
        .collect()
}

fn generate_f32_samples(n: usize, n_features: usize) -> Vec<Vec<f32>> {
    let mut rng: u64 = 0xDEAD_1337;
    (0..n)
        .map(|_| {
            (0..n_features)
                .map(|_| (xorshift64(&mut rng) * 10.0 - 5.0) as f32)
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Model training helpers (not timed)
// ---------------------------------------------------------------------------

fn train_irithyll(features: &[Vec<f64>], targets: &[f64]) -> SGBT {
    let config = SGBTConfig::builder()
        .n_steps(N_TREES)
        .learning_rate(0.01)
        .max_depth(MAX_DEPTH)
        .n_bins(32)
        .grace_period(20)
        .build()
        .unwrap();
    let mut model = SGBT::new(config);
    for (f, &t) in features.iter().zip(targets.iter()) {
        model.train_one(&Sample::new(f.clone(), t));
    }
    model
}

fn train_gbdt(features: &[Vec<f64>], targets: &[f64]) -> GBDT {
    let mut cfg = GBDTConfig::new();
    cfg.set_feature_size(N_FEATURES);
    cfg.set_max_depth(MAX_DEPTH as u32);
    cfg.set_iterations(N_TREES);
    cfg.set_shrinkage(0.01);
    cfg.set_loss("SquaredError");
    cfg.set_debug(false);
    cfg.set_min_leaf_size(1);

    let mut train_data: Vec<Data> = features
        .iter()
        .zip(targets.iter())
        .map(|(f, &t)| {
            Data::new_training_data(f.iter().map(|&v| v as f32).collect(), 1.0, t as f32, None)
        })
        .collect();

    let mut gbdt_model = GBDT::new(&cfg);
    gbdt_model.fit(&mut train_data);
    gbdt_model
}

// ---------------------------------------------------------------------------
// Group 1: Single prediction latency comparison
// ---------------------------------------------------------------------------

fn predict_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict_latency");

    let (train_features, train_targets) = generate_train_data(N_TRAIN, N_FEATURES, 0xBEEF_CAFE);

    let irithyll_model = train_irithyll(&train_features, &train_targets);
    let gbdt_model = train_gbdt(&train_features, &train_targets);

    let packed_f32_bytes = export_packed(&irithyll_model, N_FEATURES);
    let view_f32 = EnsembleView::from_bytes(&packed_f32_bytes).unwrap();

    let packed_i16_bytes = export_packed_i16(&irithyll_model, N_FEATURES);
    let view_i16 = QuantizedEnsembleView::from_bytes(&packed_i16_bytes).unwrap();

    let features_f64: Vec<f64> = (0..N_FEATURES).map(|i| (i as f64) * 0.3 - 1.5).collect();
    let features_f32: Vec<f32> = features_f64.iter().map(|&v| v as f32).collect();
    let gbdt_single = vec![Data::new_test_data(features_f32.clone(), None)];

    group.bench_function("irithyll_sgbt_f64", |b| {
        b.iter(|| irithyll_model.predict(black_box(&features_f64)))
    });

    group.bench_function("irithyll_packed_f32", |b| {
        b.iter(|| view_f32.predict(black_box(&features_f32)))
    });

    group.bench_function("irithyll_quantized_i16", |b| {
        b.iter(|| view_i16.predict(black_box(&features_f32)))
    });

    group.bench_function("gbdt_rs_predict", |b| {
        b.iter(|| gbdt_model.predict(black_box(&gbdt_single)))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: Batch prediction throughput (10,000 samples)
// ---------------------------------------------------------------------------

fn batch_predict_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_predict_10k");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let (train_features, train_targets) = generate_train_data(N_TRAIN, N_FEATURES, 0xBEEF_CAFE);

    let irithyll_model = train_irithyll(&train_features, &train_targets);
    let gbdt_model = train_gbdt(&train_features, &train_targets);

    let packed_f32_bytes = export_packed(&irithyll_model, N_FEATURES);
    let view_f32 = EnsembleView::from_bytes(&packed_f32_bytes).unwrap();

    let packed_i16_bytes = export_packed_i16(&irithyll_model, N_FEATURES);
    let view_i16 = QuantizedEnsembleView::from_bytes(&packed_i16_bytes).unwrap();

    let samples_f64 = generate_f64_samples(BATCH_SIZE, N_FEATURES);
    let samples_f32 = generate_f32_samples(BATCH_SIZE, N_FEATURES);
    let sample_refs_f32: Vec<&[f32]> = samples_f32.iter().map(|s| s.as_slice()).collect();

    let gbdt_batch: Vec<Data> = samples_f32
        .iter()
        .map(|f| Data::new_test_data(f.clone(), None))
        .collect();

    let mut out_f32 = vec![0.0f32; BATCH_SIZE];
    let mut out_i16 = vec![0.0f32; BATCH_SIZE];

    group.bench_function("irithyll_sgbt_f64", |b| {
        b.iter(|| {
            for sample in &samples_f64 {
                black_box(irithyll_model.predict(black_box(sample.as_slice())));
            }
        })
    });

    group.bench_function("irithyll_packed_f32", |b| {
        b.iter(|| {
            view_f32.predict_batch(black_box(&sample_refs_f32), &mut out_f32);
            black_box(&out_f32);
        })
    });

    group.bench_function("irithyll_quantized_i16", |b| {
        b.iter(|| {
            view_i16.predict_batch(black_box(&sample_refs_f32), &mut out_i16);
            black_box(&out_i16);
        })
    });

    group.bench_function("gbdt_rs_predict", |b| {
        b.iter(|| gbdt_model.predict(black_box(&gbdt_batch)))
    });

    group.finish();
}

criterion_group!(comparison_benches, predict_latency, batch_predict_10k);
criterion_main!(comparison_benches);
