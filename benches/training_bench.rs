//! Training benchmarks for DistributionalSGBT and MoEDistributionalSGBT.
//!
//! Measures train and predict throughput across different configurations
//! for distributional and mixture-of-experts distributional ensembles.
//! Complements `throughput_bench.rs` (which covers plain SGBT).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use irithyll::ensemble::distributional::DistributionalSGBT;
use irithyll::ensemble::moe_distributional::MoEDistributionalSGBT;
use irithyll::{SGBTConfig, Sample};

// ---------------------------------------------------------------------------
// Deterministic PRNG (no rand dependency)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Config helper
// ---------------------------------------------------------------------------

fn bench_config(n_steps: usize, n_features: usize) -> SGBTConfig {
    let grace = if n_features > 10 { 100 } else { 50 };
    SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(0.1)
        .max_depth(4)
        .n_bins(32)
        .grace_period(grace)
        .build()
        .expect("valid bench config")
}

// ---------------------------------------------------------------------------
// Group 1: DistributionalSGBT train throughput
// ---------------------------------------------------------------------------

/// Training throughput for DistributionalSGBT: batch of 1000 samples across
/// different step/feature configurations.
fn distributional_train_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributional_train_throughput");
    let batch_size = 1000u64;

    for &(n_steps, n_features) in &[(10, 3), (50, 3), (50, 20), (100, 3)] {
        group.throughput(Throughput::Elements(batch_size));
        group.bench_with_input(
            BenchmarkId::new(format!("{}steps_{}feat", n_steps, n_features), batch_size),
            &(n_steps, n_features),
            |b, &(ns, nf)| {
                let config = bench_config(ns, nf);
                let mut model = DistributionalSGBT::new(config);
                let mut rng: u64 = 0xFACE_BEEF;

                // Warmup: 200 samples to build tree structure.
                for _ in 0..200 {
                    model.train_one(&gen_sample_n(&mut rng, nf));
                }

                b.iter(|| {
                    for _ in 0..batch_size {
                        model.train_one(black_box(&gen_sample_n(&mut rng, nf)));
                    }
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: DistributionalSGBT predict throughput
// ---------------------------------------------------------------------------

/// Prediction throughput for DistributionalSGBT: batch of 10000 predictions
/// across different step/feature configurations.
fn distributional_predict_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributional_predict_throughput");
    let batch_size = 10_000u64;

    for &(n_steps, n_features) in &[(10, 3), (50, 3), (50, 20), (100, 3)] {
        group.throughput(Throughput::Elements(batch_size));
        group.bench_with_input(
            BenchmarkId::new(format!("{}steps_{}feat", n_steps, n_features), batch_size),
            &(n_steps, n_features),
            |b, &(ns, nf)| {
                let config = bench_config(ns, nf);
                let mut model = DistributionalSGBT::new(config);
                let mut rng: u64 = 0xDEAD_1337;

                // Train enough to build full tree structure.
                for _ in 0..200 {
                    model.train_one(&gen_sample_n(&mut rng, nf));
                }

                b.iter(|| {
                    for _ in 0..batch_size {
                        let features: Vec<f64> =
                            (0..nf).map(|_| xorshift64(&mut rng) * 10.0 - 5.0).collect();
                        let pred = model.predict(black_box(&features));
                        black_box(pred.mu);
                    }
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: MoEDistributionalSGBT train throughput
// ---------------------------------------------------------------------------

/// Training throughput for MoEDistributionalSGBT: batch of 500 samples
/// across different expert/step configurations (3 features fixed).
fn moe_train_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_train_throughput");
    let batch_size = 500u64;
    let n_features = 3;

    for &(n_experts, n_steps) in &[(2, 10), (3, 10), (3, 50), (5, 10)] {
        group.throughput(Throughput::Elements(batch_size));
        group.bench_with_input(
            BenchmarkId::new(format!("{}exp_{}steps", n_experts, n_steps), batch_size),
            &(n_experts, n_steps),
            |b, &(ne, ns)| {
                let config = bench_config(ns, n_features);
                let mut model = MoEDistributionalSGBT::new(config, ne);
                let mut rng: u64 = 0xDEAD_CAFE;

                // Warmup: 200 samples to initialize gate and expert structures.
                for _ in 0..200 {
                    model.train_one(&gen_sample_n(&mut rng, n_features));
                }

                b.iter(|| {
                    for _ in 0..batch_size {
                        model.train_one(black_box(&gen_sample_n(&mut rng, n_features)));
                    }
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: MoEDistributionalSGBT predict throughput
// ---------------------------------------------------------------------------

/// Prediction throughput for MoEDistributionalSGBT: batch of 10000
/// predictions across different expert/step configurations (3 features fixed).
fn moe_predict_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_predict_throughput");
    let batch_size = 10_000u64;
    let n_features = 3;

    for &(n_experts, n_steps) in &[(2, 10), (3, 10), (3, 50), (5, 10)] {
        group.throughput(Throughput::Elements(batch_size));
        group.bench_with_input(
            BenchmarkId::new(format!("{}exp_{}steps", n_experts, n_steps), batch_size),
            &(n_experts, n_steps),
            |b, &(ne, ns)| {
                let config = bench_config(ns, n_features);
                let mut model = MoEDistributionalSGBT::new(config, ne);
                let mut rng: u64 = 0x7331_ABCD;

                // Warmup: 200 samples to initialize gate and expert structures.
                for _ in 0..200 {
                    model.train_one(&gen_sample_n(&mut rng, n_features));
                }

                b.iter(|| {
                    for _ in 0..batch_size {
                        let features: Vec<f64> = (0..n_features)
                            .map(|_| xorshift64(&mut rng) * 10.0 - 5.0)
                            .collect();
                        let pred = model.predict(black_box(&features));
                        black_box(pred.mu);
                    }
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    distributional_train_throughput,
    distributional_predict_throughput,
    moe_train_throughput,
    moe_predict_throughput
);
criterion_main!(benches);
