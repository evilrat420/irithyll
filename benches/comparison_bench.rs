//! Self-comparison benchmarks: irithyll inference paths head-to-head.
//!
//! Compares SGBT (f64 tree walk), packed EnsembleView (f32 branch-free),
//! and quantized QuantizedEnsembleView (i16) on the same trained model.
//! No external crates — pure irithyll self-comparison for Reddit/marketing.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use irithyll::export_embedded::{export_packed, export_packed_i16};
use irithyll::{SGBTConfig, Sample, SGBT};
use irithyll_core::{EnsembleView, QuantizedEnsembleView};

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
// Helpers
// ---------------------------------------------------------------------------

/// Train a deterministic model for benchmarking.
///
/// Uses xorshift64 for reproducible data with a non-trivial target function:
/// `y = sum_i (i+1) * x_i` to ensure the trees actually learn splits.
fn train_bench_model(
    n_steps: usize,
    n_features: usize,
    max_depth: usize,
    n_samples: usize,
) -> SGBT {
    let config = SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(0.01)
        .grace_period(20)
        .max_depth(max_depth)
        .n_bins(32)
        .build()
        .unwrap();
    let mut model = SGBT::new(config);
    let mut rng: u64 = 0xBEEF_CAFE;
    for _ in 0..n_samples {
        let features: Vec<f64> = (0..n_features)
            .map(|_| xorshift64(&mut rng) * 10.0 - 5.0)
            .collect();
        let target = features
            .iter()
            .enumerate()
            .fold(0.0, |acc, (i, &f)| acc + (i as f64 + 1.0) * f);
        model.train_one(&Sample::new(features, target));
    }
    model
}

/// Generate deterministic f64 feature vectors for SGBT predict.
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

/// Generate deterministic f32 feature vectors for packed/quantized predict.
fn generate_f32_samples(n: usize, n_features: usize) -> Vec<Vec<f32>> {
    let mut rng: u64 = 0xDEAD_1337;
    (0..n)
        .map(|_| {
            (0..n_features)
                .map(|_| {
                    let v = xorshift64(&mut rng) * 10.0 - 5.0;
                    v as f32
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Group 1: Single prediction latency comparison
// ---------------------------------------------------------------------------

/// Compare single-sample predict latency across all three inference paths.
///
/// All three use the same underlying model (50 trees, depth 4, 10 features):
/// - `irithyll_sgbt_f64`:       Standard SGBT predict (f64 tree walk)
/// - `irithyll_packed_f32`:     Packed EnsembleView predict (f32 branch-free)
/// - `irithyll_quantized_i16`:  QuantizedEnsembleView predict (i16)
fn comparison_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_predict");
    let n_features = 10;

    // Train the shared model
    let model = train_bench_model(50, n_features, 4, 1000);

    // Export to both packed formats
    let packed_f32 = export_packed(&model, n_features);
    let view_f32 = EnsembleView::from_bytes(&packed_f32).unwrap();

    let packed_i16 = export_packed_i16(&model, n_features);
    let view_i16 = QuantizedEnsembleView::from_bytes(&packed_i16).unwrap();

    // Prepare feature vectors (same values, different types)
    let features_f64: Vec<f64> = (0..n_features).map(|i| (i as f64) * 0.1).collect();
    let features_f32: Vec<f32> = (0..n_features).map(|i| (i as f32) * 0.1).collect();

    // Bench: SGBT f64 predict (standard tree walk)
    group.bench_function("irithyll_sgbt_f64", |b| {
        b.iter(|| model.predict(black_box(&features_f64)))
    });

    // Bench: Packed f32 predict (branch-free, cache-optimized)
    group.bench_function("irithyll_packed_f32", |b| {
        b.iter(|| view_f32.predict(black_box(&features_f32)))
    });

    // Bench: Quantized i16 predict (integer-only traversal)
    group.bench_function("irithyll_quantized_i16", |b| {
        b.iter(|| view_i16.predict(black_box(&features_f32)))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: Batch prediction throughput comparison
// ---------------------------------------------------------------------------

/// Compare batch prediction throughput (10,000 samples) across all three paths.
///
/// Uses `Throughput::Elements(10_000)` so criterion reports samples/sec.
/// - `sgbt_batch_f64`:       SGBT predict loop over f64 samples
/// - `packed_batch_f32`:     EnsembleView.predict_batch (f32, x4 interleaved)
/// - `quantized_batch_i16`:  QuantizedEnsembleView.predict_batch (i16 inline)
fn comparison_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_batch_throughput");
    let n_features = 10;
    let batch_size: u64 = 10_000;

    group.throughput(Throughput::Elements(batch_size));

    // Train the shared model
    let model = train_bench_model(50, n_features, 4, 1000);

    // Export to both packed formats
    let packed_f32 = export_packed(&model, n_features);
    let view_f32 = EnsembleView::from_bytes(&packed_f32).unwrap();

    let packed_i16 = export_packed_i16(&model, n_features);
    let view_i16 = QuantizedEnsembleView::from_bytes(&packed_i16).unwrap();

    // Pre-generate sample data (same underlying values for fairness)
    let samples_f64 = generate_f64_samples(batch_size as usize, n_features);
    let samples_f32 = generate_f32_samples(batch_size as usize, n_features);
    let sample_refs_f32: Vec<&[f32]> = samples_f32.iter().map(|s| s.as_slice()).collect();

    // Output buffers
    let mut out_f32 = vec![0.0f32; batch_size as usize];
    let mut out_i16 = vec![0.0f32; batch_size as usize];

    // Bench: SGBT batch f64 (predict loop)
    group.bench_function("sgbt_batch_f64", |b| {
        b.iter(|| {
            for sample in &samples_f64 {
                black_box(model.predict(black_box(sample.as_slice())));
            }
        })
    });

    // Bench: Packed batch f32 (predict_batch with x4 interleaving)
    group.bench_function("packed_batch_f32", |b| {
        b.iter(|| {
            view_f32.predict_batch(black_box(&sample_refs_f32), &mut out_f32);
            black_box(&out_f32);
        })
    });

    // Bench: Quantized batch i16 (predict_batch with inline quantization)
    group.bench_function("quantized_batch_i16", |b| {
        b.iter(|| {
            view_i16.predict_batch(black_box(&sample_refs_f32), &mut out_i16);
            black_box(&out_i16);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    comparison_benches,
    comparison_predict,
    comparison_batch_throughput
);
criterion_main!(comparison_benches);
