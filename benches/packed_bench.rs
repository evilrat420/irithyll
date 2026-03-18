//! Benchmarks for packed inference vs standard SGBT prediction.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use irithyll::ensemble::config::SGBTConfig;
use irithyll::ensemble::SGBT;
use irithyll::export_embedded::export_packed;
use irithyll::sample::Sample;
use irithyll_core::EnsembleView;

/// Train a model for benchmarking.
fn train_bench_model(n_steps: usize, n_features: usize, n_samples: usize) -> SGBT {
    let config = SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(0.01)
        .grace_period(20)
        .max_depth(4)
        .n_bins(32)
        .build()
        .unwrap();
    let mut model = SGBT::new(config);

    for i in 0..n_samples {
        let features: Vec<f64> = (0..n_features)
            .map(|j| ((i * n_features + j) as f64) * 0.01)
            .collect();
        let target = features.iter().sum::<f64>() * 0.5;
        model.train_one(&Sample::new(features, target));
    }
    model
}

fn bench_packed_single_predict(c: &mut Criterion) {
    let n_features = 10;
    let model = train_bench_model(50, n_features, 500);
    let packed = export_packed(&model, n_features);
    let view = EnsembleView::from_bytes(&packed).unwrap();

    let features: Vec<f32> = (0..n_features).map(|i| (i as f32) * 0.1).collect();

    c.bench_function("packed_single_predict (50 trees)", |b| {
        b.iter(|| view.predict(black_box(&features)))
    });
}

fn bench_sgbt_single_predict(c: &mut Criterion) {
    let n_features = 10;
    let model = train_bench_model(50, n_features, 500);
    let features: Vec<f64> = (0..n_features).map(|i| (i as f64) * 0.1).collect();

    c.bench_function("sgbt_single_predict (50 trees)", |b| {
        b.iter(|| model.predict(black_box(&features)))
    });
}

fn bench_packed_batch_predict(c: &mut Criterion) {
    let n_features = 10;
    let model = train_bench_model(50, n_features, 500);
    let packed = export_packed(&model, n_features);
    let view = EnsembleView::from_bytes(&packed).unwrap();

    let samples: Vec<Vec<f32>> = (0..1000)
        .map(|i| {
            (0..n_features)
                .map(|j| ((i * n_features + j) as f32) * 0.01)
                .collect()
        })
        .collect();
    let sample_refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();
    let mut out = vec![0.0f32; 1000];

    c.bench_function("packed_batch_1000 (50 trees)", |b| {
        b.iter(|| view.predict_batch(black_box(&sample_refs), &mut out))
    });
}

fn bench_packed_model_size(c: &mut Criterion) {
    let n_features = 10;
    let model = train_bench_model(50, n_features, 500);

    c.bench_function("export_packed (50 trees)", |b| {
        b.iter(|| export_packed(black_box(&model), n_features))
    });
}

criterion_group!(
    packed_benches,
    bench_packed_single_predict,
    bench_sgbt_single_predict,
    bench_packed_batch_predict,
    bench_packed_model_size,
);
criterion_main!(packed_benches);
