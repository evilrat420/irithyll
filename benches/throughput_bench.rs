//! Throughput benchmarks for v2.0.0 baseline.
//!
//! Measures sustained samples/sec across different configurations.
//! Run before and after performance changes to verify improvements.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use irithyll::{SGBTConfig, Sample, SGBT};

fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

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

fn bench_config(n_steps: usize, n_features: usize) -> SGBTConfig {
    let grace = if n_features > 50 { 100 } else { 50 };
    SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(0.1)
        .max_depth(4)
        .n_bins(32)
        .grace_period(grace)
        .build()
        .expect("valid bench config")
}

/// Training throughput: batch of 1000 samples, varying steps and features.
fn train_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("train_throughput");
    let batch_size = 1000u64;

    for &(n_steps, n_features) in &[
        (10, 3),
        (10, 20),
        (10, 100),
        (50, 3),
        (50, 20),
        (100, 3),
        (100, 20),
    ] {
        group.throughput(Throughput::Elements(batch_size));
        group.bench_with_input(
            BenchmarkId::new(format!("{}steps_{}feat", n_steps, n_features), batch_size),
            &(n_steps, n_features),
            |b, &(ns, nf)| {
                let config = bench_config(ns, nf);
                let mut model = SGBT::new(config);
                let mut rng: u64 = 0xBEEF_CAFE_1234;

                // Warmup: build tree structure.
                for _ in 0..200 {
                    model.train_one(&gen_sample_n(&mut rng, nf));
                }

                b.iter(|| {
                    for _ in 0..batch_size {
                        let sample = gen_sample_n(&mut rng, nf);
                        model.train_one(black_box(&sample));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Prediction throughput: batch of 10000 predictions, varying steps.
fn predict_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict_throughput");
    let batch_size = 10_000u64;

    for &(n_steps, n_features) in &[(10, 3), (50, 3), (100, 3), (50, 20), (50, 100)] {
        group.throughput(Throughput::Elements(batch_size));
        group.bench_with_input(
            BenchmarkId::new(format!("{}steps_{}feat", n_steps, n_features), batch_size),
            &(n_steps, n_features),
            |b, &(ns, nf)| {
                let config = bench_config(ns, nf);
                let mut model = SGBT::new(config);
                let mut rng: u64 = 0xDEAD_1337;

                // Train enough to build full tree structure.
                for _ in 0..1000 {
                    model.train_one(&gen_sample_n(&mut rng, nf));
                }

                b.iter(|| {
                    for _ in 0..batch_size {
                        let features: Vec<f64> =
                            (0..nf).map(|_| xorshift64(&mut rng) * 10.0 - 5.0).collect();
                        black_box(model.predict(black_box(&features)));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Train+predict interleaved (prequential pattern).
fn prequential_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("prequential_throughput");
    let batch_size = 1000u64;

    for &n_steps in &[10, 50, 100] {
        let n_features = 3;
        group.throughput(Throughput::Elements(batch_size));
        group.bench_with_input(
            BenchmarkId::new(format!("{}steps", n_steps), batch_size),
            &n_steps,
            |b, &ns| {
                let config = bench_config(ns, n_features);
                let mut model = SGBT::new(config);
                let mut rng: u64 = 0x7331_ABCD;

                for _ in 0..200 {
                    model.train_one(&gen_sample_n(&mut rng, n_features));
                }

                b.iter(|| {
                    for _ in 0..batch_size {
                        let sample = gen_sample_n(&mut rng, n_features);
                        // Predict-then-train (prequential)
                        black_box(model.predict(sample.features.as_slice()));
                        model.train_one(black_box(&sample));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    train_throughput,
    predict_throughput,
    prequential_throughput
);
criterion_main!(benches);
