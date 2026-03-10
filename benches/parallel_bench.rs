use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use irithyll::{SGBTConfig, Sample};

/// Deterministic xorshift64 PRNG returning f64 in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Generate a deterministic sample from the RNG state.
fn gen_sample(rng: &mut u64) -> Sample {
    let f0 = xorshift64(rng) * 10.0 - 5.0;
    let f1 = xorshift64(rng) * 10.0 - 5.0;
    let f2 = xorshift64(rng) * 10.0 - 5.0;
    let target = 2.0 * f0 + f1 - 0.5 * f2;
    Sample::new(vec![f0, f1, f2], target)
}

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

fn parallel_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");

    // Compare sequential SGBT vs ParallelSGBT at different ensemble sizes.
    for n_steps in [10, 25, 50, 100] {
        // --- Sequential SGBT ---
        group.bench_with_input(
            BenchmarkId::new("sequential_train", n_steps),
            &n_steps,
            |b, &n| {
                let config = bench_config(n);
                let mut model = irithyll::SGBT::new(config);
                let mut rng: u64 = 0xDEAD_BEEF;

                // Warmup to initialize base prediction.
                for _ in 0..100 {
                    model.train_one(&gen_sample(&mut rng));
                }

                b.iter(|| {
                    let sample = gen_sample(&mut rng);
                    model.train_one(black_box(&sample));
                });
            },
        );

        // --- Parallel SGBT ---
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel_train", n_steps),
            &n_steps,
            |b, &n| {
                let config = bench_config(n);
                let mut model = irithyll::ParallelSGBT::new(config);
                let mut rng: u64 = 0xDEAD_BEEF;

                for _ in 0..100 {
                    model.train_one(&gen_sample(&mut rng));
                }

                b.iter(|| {
                    let sample = gen_sample(&mut rng);
                    model.train_one(black_box(&sample));
                });
            },
        );

        // --- Sequential predict ---
        group.bench_with_input(
            BenchmarkId::new("sequential_predict", n_steps),
            &n_steps,
            |b, &n| {
                let config = bench_config(n);
                let mut model = irithyll::SGBT::new(config);
                let mut rng: u64 = 0xCAFE_BABE;

                for _ in 0..500 {
                    model.train_one(&gen_sample(&mut rng));
                }

                b.iter(|| {
                    let f0 = xorshift64(&mut rng) * 10.0 - 5.0;
                    let f1 = xorshift64(&mut rng) * 10.0 - 5.0;
                    let f2 = xorshift64(&mut rng) * 10.0 - 5.0;
                    black_box(model.predict(black_box(&[f0, f1, f2])));
                });
            },
        );

        // --- Parallel predict (should be identical speed — prediction is sequential) ---
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel_predict", n_steps),
            &n_steps,
            |b, &n| {
                let config = bench_config(n);
                let mut model = irithyll::ParallelSGBT::new(config);
                let mut rng: u64 = 0xCAFE_BABE;

                for _ in 0..500 {
                    model.train_one(&gen_sample(&mut rng));
                }

                b.iter(|| {
                    let f0 = xorshift64(&mut rng) * 10.0 - 5.0;
                    let f1 = xorshift64(&mut rng) * 10.0 - 5.0;
                    let f2 = xorshift64(&mut rng) * 10.0 - 5.0;
                    black_box(model.predict(black_box(&[f0, f1, f2])));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, parallel_benchmarks);
criterion_main!(benches);
