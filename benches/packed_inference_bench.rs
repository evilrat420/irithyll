//! Comprehensive packed inference benchmarks for irithyll.
//!
//! Six benchmark groups measuring packed `EnsembleView` performance across
//! tree count, depth, feature count, batch size, vs standard SGBT, and
//! export speed / model size.
//!
//! This is the Reddit headline benchmark suite.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use irithyll::export_embedded::export_packed;
use irithyll::{SGBTConfig, Sample, SGBT};
use irithyll_core::EnsembleView;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic xorshift64 PRNG. No external RNG dependency.
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Train a deterministic SGBT model for benchmarking.
///
/// Uses xorshift64 for reproducible feature generation and a weighted-sum
/// target function: `target = sum((i+1) * f_i)`.
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

/// Generate deterministic f32 feature vectors for inference benchmarking.
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
// Group 1: packed_predict_trees — Scaling with tree count
// ---------------------------------------------------------------------------

/// Sweep n_trees in [10, 50, 100, 500] with max_depth=4, n_features=10.
///
/// Measures single-sample packed predict latency as tree count grows.
/// Expected: near-linear scaling (each tree is an independent traversal).
fn packed_predict_trees(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_predict_trees");
    let n_features = 10;
    let max_depth = 4;
    let n_samples = 1000;

    for n_trees in [10, 50, 100, 500] {
        let model = train_bench_model(n_trees, n_features, max_depth, n_samples);
        let packed = export_packed(&model, n_features);
        let view = EnsembleView::from_bytes(&packed).unwrap();
        let features: Vec<f32> = (0..n_features).map(|i| (i as f32) * 0.1).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n_trees), &n_trees, |b, _| {
            b.iter(|| view.predict(black_box(&features)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: packed_predict_depth — Scaling with tree depth
// ---------------------------------------------------------------------------

/// Sweep max_depth in [2, 4, 6, 8] with n_trees=50, n_features=10.
///
/// Measures how deeper trees affect traversal latency. Deeper trees have
/// more nodes but each traversal is still O(depth).
fn packed_predict_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_predict_depth");
    let n_trees = 50;
    let n_features = 10;
    let n_samples = 1000;

    for max_depth in [2, 4, 6, 8] {
        let model = train_bench_model(n_trees, n_features, max_depth, n_samples);
        let packed = export_packed(&model, n_features);
        let view = EnsembleView::from_bytes(&packed).unwrap();
        let features: Vec<f32> = (0..n_features).map(|i| (i as f32) * 0.1).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(max_depth),
            &max_depth,
            |b, _| {
                b.iter(|| view.predict(black_box(&features)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: packed_predict_features — Scaling with feature count
// ---------------------------------------------------------------------------

/// Sweep n_features in [5, 10, 50, 100] with n_trees=50, max_depth=4.
///
/// Feature count affects model size (more split candidates) and potentially
/// cache pressure, but traversal cost per node is constant.
fn packed_predict_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_predict_features");
    let n_trees = 50;
    let max_depth = 4;
    let n_samples = 1000;

    for n_features in [5, 10, 50, 100] {
        let model = train_bench_model(n_trees, n_features, max_depth, n_samples);
        let packed = export_packed(&model, n_features);
        let view = EnsembleView::from_bytes(&packed).unwrap();
        let features: Vec<f32> = (0..n_features).map(|i| (i as f32) * 0.1).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(n_features),
            &n_features,
            |b, _| {
                b.iter(|| view.predict(black_box(&features)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: packed_batch_scaling — Batch size scaling with throughput
// ---------------------------------------------------------------------------

/// Sweep batch in [1, 1_000, 10_000, 100_000] with Throughput::Elements.
/// Holds n_trees=50, max_depth=4, n_features=10.
///
/// Measures samples/sec to show ILP benefits of x4 interleaved batch predict.
fn packed_batch_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_batch_scaling");
    let n_trees = 50;
    let n_features = 10;
    let max_depth = 4;
    let n_samples = 1000;

    let model = train_bench_model(n_trees, n_features, max_depth, n_samples);
    let packed = export_packed(&model, n_features);
    let view = EnsembleView::from_bytes(&packed).unwrap();

    for batch_size in [1u64, 1_000, 10_000, 100_000] {
        let samples = generate_f32_samples(batch_size as usize, n_features);
        let sample_refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();
        let mut out = vec![0.0f32; batch_size as usize];

        group.throughput(Throughput::Elements(batch_size));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    if batch_size == 1 {
                        black_box(view.predict(black_box(&samples[0])));
                    } else {
                        view.predict_batch(black_box(&sample_refs), &mut out);
                        black_box(&out);
                    }
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 5: packed_vs_sgbt — f32 packed vs f64 SGBT predict
// ---------------------------------------------------------------------------

/// Head-to-head: packed f32 EnsembleView vs standard f64 SGBT predict.
/// Same model (50 trees, 10 features, depth 4).
///
/// Shows the speedup from the cache-optimized packed format over the
/// standard tree-walking SGBT implementation.
fn packed_vs_sgbt(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_vs_sgbt");
    let n_trees = 50;
    let n_features = 10;
    let max_depth = 4;
    let n_samples = 1000;

    let model = train_bench_model(n_trees, n_features, max_depth, n_samples);
    let packed = export_packed(&model, n_features);
    let view = EnsembleView::from_bytes(&packed).unwrap();

    let features_f64: Vec<f64> = (0..n_features).map(|i| (i as f64) * 0.1).collect();
    let features_f32: Vec<f32> = (0..n_features).map(|i| (i as f32) * 0.1).collect();

    group.bench_function("sgbt_f64_predict", |b| {
        b.iter(|| model.predict(black_box(&features_f64)));
    });

    group.bench_function("packed_f32_predict", |b| {
        b.iter(|| view.predict(black_box(&features_f32)));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 6: packed_model_size — Export speed + binary size reporting
// ---------------------------------------------------------------------------

/// Benchmark export (serialization) speed for various model configurations
/// and print packed binary sizes to stderr for reference.
///
/// Not a latency benchmark per se — measures how fast `export_packed`
/// serializes the model. Binary sizes are printed as `[SIZE]` lines in
/// criterion output for offline analysis.
fn packed_model_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_model_size");
    let n_samples = 1000;

    for (n_trees, max_depth, n_features) in [
        (10, 4, 10),
        (50, 4, 10),
        (100, 4, 10),
        (500, 4, 10),
        (50, 2, 10),
        (50, 6, 10),
        (50, 8, 10),
    ] {
        let model = train_bench_model(n_trees, n_features, max_depth, n_samples);
        let packed = export_packed(&model, n_features);
        let label = format!("{}t_d{}_{}f", n_trees, max_depth, n_features);

        // Print size for reference (visible in criterion stderr output)
        let view = EnsembleView::from_bytes(&packed).unwrap();
        eprintln!(
            "[SIZE] {} = {} bytes ({} nodes)",
            label,
            packed.len(),
            view.total_nodes()
        );

        // Bench the export itself (serialization speed)
        group.bench_function(&label, |b| {
            b.iter(|| export_packed(black_box(&model), n_features));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    packed_inference,
    packed_predict_trees,
    packed_predict_depth,
    packed_predict_features,
    packed_batch_scaling,
    packed_vs_sgbt,
    packed_model_size,
);
criterion_main!(packed_inference);
