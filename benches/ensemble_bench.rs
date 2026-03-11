use criterion::{black_box, criterion_group, criterion_main, Criterion};
use irithyll::{SGBTConfig, Sample, SGBT};

/// Deterministic xorshift64 PRNG returning f64 in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Build an SGBTConfig with the given number of steps and sensible bench defaults.
fn bench_config(n_steps: usize) -> SGBTConfig {
    SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(0.1)
        .max_depth(4)
        .n_bins(32)
        .grace_period(50)
        .build()
        .expect("valid bench config")
}

/// Generate a deterministic sample from the RNG state.
fn gen_sample(rng: &mut u64) -> Sample {
    let f0 = xorshift64(rng) * 10.0 - 5.0;
    let f1 = xorshift64(rng) * 10.0 - 5.0;
    let f2 = xorshift64(rng) * 10.0 - 5.0;
    let target = 2.0 * f0 + f1 - 0.5 * f2;
    Sample::new(vec![f0, f1, f2], target)
}

fn ensemble_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble");

    // -----------------------------------------------------------------------
    // 1. sgbt_train_one_10_steps: single-sample latency with 10 boosting steps
    // -----------------------------------------------------------------------
    group.bench_function("sgbt_train_one_10_steps", |b| {
        let config = bench_config(10);
        let mut model = SGBT::new(config);
        let mut rng: u64 = 0xAAAA_BBBB;

        // Warmup: 100 samples to initialize base prediction and tree structure.
        for _ in 0..100 {
            let sample = gen_sample(&mut rng);
            model.train_one(&sample);
        }

        b.iter(|| {
            let sample = gen_sample(&mut rng);
            model.train_one(black_box(&sample));
        });
    });

    // -----------------------------------------------------------------------
    // 2. sgbt_train_one_50_steps: single-sample latency with 50 boosting steps
    // -----------------------------------------------------------------------
    group.bench_function("sgbt_train_one_50_steps", |b| {
        let config = bench_config(50);
        let mut model = SGBT::new(config);
        let mut rng: u64 = 0xCCCC_DDDD;

        // Warmup: 100 samples.
        for _ in 0..100 {
            let sample = gen_sample(&mut rng);
            model.train_one(&sample);
        }

        b.iter(|| {
            let sample = gen_sample(&mut rng);
            model.train_one(black_box(&sample));
        });
    });

    // -----------------------------------------------------------------------
    // 3. sgbt_predict_10_steps: prediction latency with 10 boosting steps
    // -----------------------------------------------------------------------
    group.bench_function("sgbt_predict_10_steps", |b| {
        let config = bench_config(10);
        let mut model = SGBT::new(config);
        let mut rng: u64 = 0xEEEE_FFFF;

        // Train 500 samples to build up ensemble structure.
        for _ in 0..500 {
            let sample = gen_sample(&mut rng);
            model.train_one(&sample);
        }

        b.iter(|| {
            let f0 = xorshift64(&mut rng) * 10.0 - 5.0;
            let f1 = xorshift64(&mut rng) * 10.0 - 5.0;
            let f2 = xorshift64(&mut rng) * 10.0 - 5.0;
            let pred = model.predict(black_box(&[f0, f1, f2]));
            black_box(pred);
        });
    });

    // -----------------------------------------------------------------------
    // 4. sgbt_predict_50_steps: prediction latency with 50 boosting steps
    // -----------------------------------------------------------------------
    group.bench_function("sgbt_predict_50_steps", |b| {
        let config = bench_config(50);
        let mut model = SGBT::new(config);
        let mut rng: u64 = 0x1111_2222;

        // Train 500 samples.
        for _ in 0..500 {
            let sample = gen_sample(&mut rng);
            model.train_one(&sample);
        }

        b.iter(|| {
            let f0 = xorshift64(&mut rng) * 10.0 - 5.0;
            let f1 = xorshift64(&mut rng) * 10.0 - 5.0;
            let f2 = xorshift64(&mut rng) * 10.0 - 5.0;
            let pred = model.predict(black_box(&[f0, f1, f2]));
            black_box(pred);
        });
    });

    group.finish();
}

criterion_group!(benches, ensemble_benchmarks);
criterion_main!(benches);
