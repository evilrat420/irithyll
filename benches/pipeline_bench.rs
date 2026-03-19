//! End-to-end pipeline benchmarks: train → export → predict roundtrip,
//! export latency scaling, and accuracy preservation across the packed pipeline.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use irithyll::export_embedded::{export_packed, validate_export};
use irithyll::{SGBTConfig, Sample, SGBT};
use irithyll_core::EnsembleView;

// ---------------------------------------------------------------------------
// Helpers (deterministic, zero-dependency)
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Train a model with the given parameters using deterministic synthetic data.
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

/// Generate a `Sample` with `n_features` features from xorshift state.
fn gen_sample_n(rng: &mut u64, n_features: usize) -> Sample {
    let features: Vec<f64> = (0..n_features)
        .map(|_| xorshift64(rng) * 10.0 - 5.0)
        .collect();
    let target = features
        .iter()
        .enumerate()
        .fold(0.0, |acc, (i, &f)| acc + (i as f64 + 1.0) * f);
    Sample::new(features, target)
}

/// Generate `n` test-feature vectors of `n_features` f64 values.
fn generate_f64_samples(n: usize, n_features: usize) -> Vec<Vec<f64>> {
    let mut rng: u64 = 0xACE0_FACE;
    (0..n)
        .map(|_| {
            (0..n_features)
                .map(|_| xorshift64(&mut rng) * 10.0 - 5.0)
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Group 1: train_export_predict — full roundtrip
// ---------------------------------------------------------------------------

/// Measures the complete pipeline: train model → export packed → load
/// EnsembleView → predict.  Each iteration starts from scratch so we
/// capture the true end-to-end cost.
fn train_export_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("train_export_predict");

    for &(n_steps, n_features, n_train_samples) in
        &[(10usize, 5usize, 200usize), (50, 10, 500), (100, 10, 1000)]
    {
        group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "{}steps_{}feat_{}samples",
                    n_steps, n_features, n_train_samples
                ),
                n_train_samples,
            ),
            &(n_steps, n_features, n_train_samples),
            |b, &(ns, nf, ntr)| {
                b.iter(|| {
                    // Phase 1: Train
                    let config = SGBTConfig::builder()
                        .n_steps(ns)
                        .learning_rate(0.1)
                        .max_depth(4)
                        .n_bins(32)
                        .grace_period(50)
                        .build()
                        .unwrap();
                    let mut model = SGBT::new(config);
                    let mut rng: u64 = 0x1234_5678;
                    for _ in 0..ntr {
                        model.train_one(&gen_sample_n(&mut rng, nf));
                    }

                    // Phase 2: Export
                    let packed = export_packed(&model, nf);

                    // Phase 3: Load + predict
                    let view = EnsembleView::from_bytes(&packed).unwrap();
                    let features: Vec<f32> = (0..nf).map(|i| i as f32 * 0.1).collect();
                    black_box(view.predict(black_box(&features)));
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: export_time — export latency scaling
// ---------------------------------------------------------------------------

/// Pre-trains a model, then measures only the `export_packed` call.
/// Sweeps `n_steps` to show how export cost scales with tree count.
fn export_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("export_time");

    for n_steps in [10, 50, 100, 500] {
        let model = train_bench_model(n_steps, 10, 4, 1000);
        group.bench_with_input(BenchmarkId::from_parameter(n_steps), &n_steps, |b, _| {
            b.iter(|| export_packed(black_box(&model), 10));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: accuracy_preservation — max |delta| f64 vs f32 packed
// ---------------------------------------------------------------------------

/// Measures the cost of `validate_export` (which computes max |delta|
/// between SGBT f64 predictions and packed f32 predictions) across 10K
/// test samples.  The result itself is the accuracy metric — inspect
/// criterion's reported values to see the f64→f32 precision loss.
fn accuracy_preservation(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_preservation");

    for n_steps in [10, 50, 100, 500] {
        let model = train_bench_model(n_steps, 10, 4, 1000);
        let packed = export_packed(&model, 10);
        let test_features: Vec<Vec<f64>> = generate_f64_samples(10_000, 10);

        group.bench_with_input(BenchmarkId::from_parameter(n_steps), &n_steps, |b, _| {
            b.iter(|| {
                validate_export(
                    black_box(&model),
                    black_box(&packed),
                    black_box(&test_features),
                )
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    pipeline_benches,
    train_export_predict,
    export_time,
    accuracy_preservation,
);
criterion_main!(pipeline_benches);
