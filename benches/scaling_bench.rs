//! Scaling benchmarks for v2.0.0 baseline.
//!
//! Measures how latency scales with n_steps, n_features, max_depth, and n_bins.
//! Each dimension is varied independently to isolate its contribution.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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

/// Scaling with n_steps (3 features, depth 4, 32 bins).
fn scale_n_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_n_steps");

    for n_steps in [5, 10, 25, 50, 100, 200] {
        let config = SGBTConfig::builder()
            .n_steps(n_steps)
            .learning_rate(0.1)
            .max_depth(4)
            .n_bins(32)
            .grace_period(50)
            .build()
            .expect("valid config");

        let mut model = SGBT::new(config);
        let mut rng: u64 = 0x1234_5678;

        for _ in 0..300 {
            model.train_one(&gen_sample_n(&mut rng, 3));
        }

        group.bench_with_input(BenchmarkId::from_parameter(n_steps), &n_steps, |b, _| {
            b.iter(|| {
                let sample = gen_sample_n(&mut rng, 3);
                model.train_one(black_box(&sample));
            });
        });
    }

    group.finish();
}

/// Scaling with n_features (50 steps, depth 4, 32 bins).
fn scale_n_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_n_features");

    for n_features in [3, 10, 20, 50, 100, 200] {
        let config = SGBTConfig::builder()
            .n_steps(50)
            .learning_rate(0.1)
            .max_depth(4)
            .n_bins(32)
            .grace_period(50)
            .build()
            .expect("valid config");

        let mut model = SGBT::new(config);
        let mut rng: u64 = 0xABCD_EF01;

        for _ in 0..300 {
            model.train_one(&gen_sample_n(&mut rng, n_features));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(n_features),
            &n_features,
            |b, &nf| {
                b.iter(|| {
                    let sample = gen_sample_n(&mut rng, nf);
                    model.train_one(black_box(&sample));
                });
            },
        );
    }

    group.finish();
}

/// Scaling with max_depth (50 steps, 3 features, 32 bins).
fn scale_max_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_max_depth");

    for max_depth in [2usize, 4, 6, 8, 10] {
        let config = SGBTConfig::builder()
            .n_steps(50)
            .learning_rate(0.1)
            .max_depth(max_depth)
            .n_bins(32)
            .grace_period(50)
            .build()
            .expect("valid config");

        let mut model = SGBT::new(config);
        let mut rng: u64 = 0xFEDC_BA98;

        for _ in 0..300 {
            model.train_one(&gen_sample_n(&mut rng, 3));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(max_depth),
            &max_depth,
            |b, _| {
                b.iter(|| {
                    let sample = gen_sample_n(&mut rng, 3);
                    model.train_one(black_box(&sample));
                });
            },
        );
    }

    group.finish();
}

/// Scaling with n_bins (50 steps, 3 features, depth 4).
fn scale_n_bins(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_n_bins");

    for n_bins in [8, 16, 32, 64, 128, 256] {
        let config = SGBTConfig::builder()
            .n_steps(50)
            .learning_rate(0.1)
            .max_depth(4)
            .n_bins(n_bins)
            .grace_period(50)
            .build()
            .expect("valid config");

        let mut model = SGBT::new(config);
        let mut rng: u64 = 0x9876_5432;

        for _ in 0..300 {
            model.train_one(&gen_sample_n(&mut rng, 3));
        }

        group.bench_with_input(BenchmarkId::from_parameter(n_bins), &n_bins, |b, _| {
            b.iter(|| {
                let sample = gen_sample_n(&mut rng, 3);
                model.train_one(black_box(&sample));
            });
        });
    }

    group.finish();
}

/// Predict latency scaling with n_steps.
fn scale_predict_n_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_predict_n_steps");

    for n_steps in [5, 10, 25, 50, 100, 200] {
        let config = SGBTConfig::builder()
            .n_steps(n_steps)
            .learning_rate(0.1)
            .max_depth(4)
            .n_bins(32)
            .grace_period(50)
            .build()
            .expect("valid config");

        let mut model = SGBT::new(config);
        let mut rng: u64 = 0x3456_7890;

        for _ in 0..1000 {
            model.train_one(&gen_sample_n(&mut rng, 3));
        }

        group.bench_with_input(BenchmarkId::from_parameter(n_steps), &n_steps, |b, _| {
            b.iter(|| {
                let f0 = xorshift64(&mut rng) * 10.0 - 5.0;
                let f1 = xorshift64(&mut rng) * 10.0 - 5.0;
                let f2 = xorshift64(&mut rng) * 10.0 - 5.0;
                black_box(model.predict(black_box(&[f0, f1, f2])));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    scale_n_steps,
    scale_n_features,
    scale_max_depth,
    scale_n_bins,
    scale_predict_n_steps
);
criterion_main!(benches);
