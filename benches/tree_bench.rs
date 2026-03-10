use criterion::{black_box, criterion_group, criterion_main, Criterion};
use irithyll::tree::builder::TreeConfig;
use irithyll::tree::hoeffding::HoeffdingTree;
use irithyll::tree::StreamingTree;

/// Deterministic xorshift64 PRNG returning f64 in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn tree_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree");

    let config = TreeConfig::new()
        .max_depth(6)
        .n_bins(64)
        .grace_period(200)
        .lambda(1.0);

    // -----------------------------------------------------------------------
    // 1. tree_train_one: single-sample latency (after 100-sample warmup)
    // -----------------------------------------------------------------------
    group.bench_function("tree_train_one", |b| {
        let mut tree = HoeffdingTree::new(config.clone());
        let mut rng: u64 = 0xABCD_1234;

        // Warmup: train 100 samples so the tree has structure.
        for _ in 0..100 {
            let f0 = xorshift64(&mut rng) * 10.0;
            let f1 = xorshift64(&mut rng) * 10.0;
            let f2 = xorshift64(&mut rng) * 10.0;
            let target = 2.0 * f0 + f1 - 0.5 * f2;
            let pred = tree.predict(&[f0, f1, f2]);
            tree.train_one(&[f0, f1, f2], pred - target, 1.0);
        }

        b.iter(|| {
            let f0 = xorshift64(&mut rng) * 10.0;
            let f1 = xorshift64(&mut rng) * 10.0;
            let f2 = xorshift64(&mut rng) * 10.0;
            let target = 2.0 * f0 + f1 - 0.5 * f2;
            let pred = tree.predict(&[f0, f1, f2]);
            tree.train_one(
                black_box(&[f0, f1, f2]),
                black_box(pred - target),
                black_box(1.0),
            );
        });
    });

    // -----------------------------------------------------------------------
    // 2. tree_predict: prediction latency from a trained tree
    // -----------------------------------------------------------------------
    group.bench_function("tree_predict", |b| {
        let mut tree = HoeffdingTree::new(config.clone());
        let mut rng: u64 = 0xBEEF_CAFE;

        // Train 500 samples to build up tree structure.
        for _ in 0..500 {
            let f0 = xorshift64(&mut rng) * 10.0;
            let f1 = xorshift64(&mut rng) * 10.0;
            let f2 = xorshift64(&mut rng) * 10.0;
            let target = 2.0 * f0 + f1 - 0.5 * f2;
            let pred = tree.predict(&[f0, f1, f2]);
            tree.train_one(&[f0, f1, f2], pred - target, 1.0);
        }

        b.iter(|| {
            let f0 = xorshift64(&mut rng) * 10.0;
            let f1 = xorshift64(&mut rng) * 10.0;
            let f2 = xorshift64(&mut rng) * 10.0;
            let pred = tree.predict(black_box(&[f0, f1, f2]));
            black_box(pred);
        });
    });

    // -----------------------------------------------------------------------
    // 3. tree_train_500_samples: throughput for training 500 samples
    // -----------------------------------------------------------------------
    group.bench_function("tree_train_500_samples", |b| {
        b.iter(|| {
            let mut tree = HoeffdingTree::new(config.clone());
            let mut rng: u64 = 0x1234_5678;

            for _ in 0..500 {
                let f0 = xorshift64(&mut rng) * 10.0;
                let f1 = xorshift64(&mut rng) * 10.0;
                let f2 = xorshift64(&mut rng) * 10.0;
                let target = 2.0 * f0 + f1 - 0.5 * f2;
                let pred = tree.predict(&[f0, f1, f2]);
                tree.train_one(
                    black_box(&[f0, f1, f2]),
                    black_box(pred - target),
                    black_box(1.0),
                );
            }
            black_box(&tree);
        });
    });

    group.finish();
}

criterion_group!(benches, tree_benchmarks);
criterion_main!(benches);
