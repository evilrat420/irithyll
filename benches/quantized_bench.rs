//! Benchmarks for int16 quantized inference vs f32 packed inference.
//!
//! Measures latency, throughput, accuracy degradation, and binary size for the
//! quantized export path introduced in v8.1.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use irithyll::ensemble::config::SGBTConfig;
use irithyll::ensemble::SGBT;
use irithyll::export_embedded::{export_packed, export_packed_i16};
use irithyll::sample::Sample;
use irithyll_core::{EnsembleView, QuantizedEnsembleView};

// ---------------------------------------------------------------------------
// Deterministic helpers (no rand dependency)
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

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
// Group 1: quantized_predict_vs_f32
//
// Same model exported to both f32 packed and int16 quantized formats.
// Sweep n_trees in [10, 50, 100, 500].
// Hold max_depth=4, n_features=10.
// ---------------------------------------------------------------------------

fn quantized_predict_vs_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_vs_f32");
    let n_features = 10;
    let max_depth = 4;
    let n_samples = 1000;

    for n_trees in [10, 50, 100, 500] {
        let model = train_bench_model(n_trees, n_features, max_depth, n_samples);

        let packed_f32 = export_packed(&model, n_features);
        let view_f32 = EnsembleView::from_bytes(&packed_f32).unwrap();

        let packed_i16 = export_packed_i16(&model, n_features);
        let view_i16 = QuantizedEnsembleView::from_bytes(&packed_i16).unwrap();

        let features: Vec<f32> = (0..n_features).map(|i| (i as f32) * 0.1).collect();

        group.bench_with_input(BenchmarkId::new("f32_packed", n_trees), &n_trees, |b, _| {
            b.iter(|| view_f32.predict(black_box(&features)))
        });
        group.bench_with_input(
            BenchmarkId::new("i16_quantized", n_trees),
            &n_trees,
            |b, _| b.iter(|| view_i16.predict(black_box(&features))),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: quantized_batch
//
// Batch prediction comparison f32 vs i16.
// Batch sizes: [1_000, 10_000, 100_000].
// Hold n_trees=50, max_depth=4, n_features=10.
// ---------------------------------------------------------------------------

fn quantized_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_batch");
    let n_trees = 50;
    let n_features = 10;
    let max_depth = 4;
    let n_samples = 1000;

    let model = train_bench_model(n_trees, n_features, max_depth, n_samples);

    let packed_f32 = export_packed(&model, n_features);
    let view_f32 = EnsembleView::from_bytes(&packed_f32).unwrap();

    let packed_i16 = export_packed_i16(&model, n_features);
    let view_i16 = QuantizedEnsembleView::from_bytes(&packed_i16).unwrap();

    for batch_size in [1_000u64, 10_000, 100_000] {
        let samples = generate_f32_samples(batch_size as usize, n_features);
        let sample_refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();
        let mut out_f32 = vec![0.0f32; batch_size as usize];
        let mut out_i16 = vec![0.0f32; batch_size as usize];

        group.throughput(Throughput::Elements(batch_size));

        group.bench_with_input(
            BenchmarkId::new("f32_packed", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    view_f32.predict_batch(black_box(&sample_refs), &mut out_f32);
                    black_box(&out_f32);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("i16_quantized", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    view_i16.predict_batch(black_box(&sample_refs), &mut out_i16);
                    black_box(&out_i16);
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: quantized_accuracy
//
// Measure max |delta| between f32 and i16 predictions across 10K samples.
// Sweep n_trees in [10, 50, 100, 500].
// Not primarily a latency bench -- a correctness validation bench.
// ---------------------------------------------------------------------------

fn quantized_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_accuracy");
    let n_features = 10;
    let max_depth = 4;
    let n_train_samples = 1000;
    let n_test_samples = 10_000;

    for n_trees in [10, 50, 100, 500] {
        let model = train_bench_model(n_trees, n_features, max_depth, n_train_samples);

        let packed_f32 = export_packed(&model, n_features);
        let view_f32 = EnsembleView::from_bytes(&packed_f32).unwrap();

        let packed_i16 = export_packed_i16(&model, n_features);
        let view_i16 = QuantizedEnsembleView::from_bytes(&packed_i16).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n_trees), &n_trees, |b, _| {
            b.iter(|| {
                let mut max_delta: f32 = 0.0;
                let mut rng: u64 = 0xACE0_FACE;
                for _ in 0..n_test_samples {
                    let features: Vec<f32> = (0..n_features)
                        .map(|_| {
                            let v = xorshift64(&mut rng);
                            v as f32 * 10.0 - 5.0
                        })
                        .collect();
                    let f32_pred = view_f32.predict(&features);
                    let i16_pred = view_i16.predict(&features);
                    let delta = (f32_pred - i16_pred).abs();
                    if delta > max_delta {
                        max_delta = delta;
                    }
                }
                black_box(max_delta)
            });
        });

        // Print the accuracy result for each tree count (visible in criterion output)
        let mut max_delta: f32 = 0.0;
        let mut sum_delta: f64 = 0.0;
        let mut rng: u64 = 0xACE0_FACE;
        for _ in 0..n_test_samples {
            let features: Vec<f32> = (0..n_features)
                .map(|_| {
                    let v = xorshift64(&mut rng);
                    v as f32 * 10.0 - 5.0
                })
                .collect();
            let f32_pred = view_f32.predict(&features);
            let i16_pred = view_i16.predict(&features);
            let delta = (f32_pred - i16_pred).abs();
            sum_delta += delta as f64;
            if delta > max_delta {
                max_delta = delta;
            }
        }
        let mean_delta = sum_delta / n_test_samples as f64;
        eprintln!(
            "[ACCURACY] {}t: max_delta={:.6}, mean_delta={:.6}",
            n_trees, max_delta, mean_delta
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: quantized_model_size
//
// Export speed + binary size comparison for both formats.
// Sweep n_trees in [10, 50, 100, 500].
// Hold max_depth=4, n_features=10.
// ---------------------------------------------------------------------------

fn quantized_model_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_model_size");
    let n_features = 10;
    let max_depth = 4;
    let n_samples = 1000;

    for n_trees in [10, 50, 100, 500] {
        let model = train_bench_model(n_trees, n_features, max_depth, n_samples);

        let packed_f32 = export_packed(&model, n_features);
        let packed_i16 = export_packed_i16(&model, n_features);

        let f32_nodes = EnsembleView::from_bytes(&packed_f32).unwrap().total_nodes();
        let i16_nodes = QuantizedEnsembleView::from_bytes(&packed_i16)
            .unwrap()
            .total_nodes();

        eprintln!(
            "[SIZE] {}t: f32={} bytes ({} nodes), i16={} bytes ({} nodes), ratio={:.2}x",
            n_trees,
            packed_f32.len(),
            f32_nodes,
            packed_i16.len(),
            i16_nodes,
            packed_f32.len() as f64 / packed_i16.len() as f64
        );

        // Bench export speed for f32
        group.bench_with_input(BenchmarkId::new("export_f32", n_trees), &n_trees, |b, _| {
            b.iter(|| export_packed(black_box(&model), n_features))
        });

        // Bench export speed for i16
        group.bench_with_input(BenchmarkId::new("export_i16", n_trees), &n_trees, |b, _| {
            b.iter(|| export_packed_i16(black_box(&model), n_features))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------------

criterion_group!(
    quantized_benches,
    quantized_predict_vs_f32,
    quantized_batch,
    quantized_accuracy,
    quantized_model_size,
);
criterion_main!(quantized_benches);
