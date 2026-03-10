use criterion::{black_box, criterion_group, criterion_main, Criterion};
use irithyll::histogram::bins::FeatureHistogram;
use irithyll::histogram::uniform::UniformBinning;
use irithyll::histogram::{BinEdges, BinningStrategy};

/// Deterministic xorshift64 PRNG returning f64 in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn histogram_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram");

    // -----------------------------------------------------------------------
    // 1. accumulate_64_bins: accumulate 10000 samples into a 64-bin histogram
    // -----------------------------------------------------------------------
    group.bench_function("accumulate_64_bins", |b| {
        let n_bins = 64;
        let edges = BinEdges {
            edges: (1..n_bins).map(|i| i as f64).collect(),
        };

        b.iter(|| {
            let mut hist = FeatureHistogram::new(edges.clone());
            let mut rng: u64 = 0xDEAD_BEEF;
            for _ in 0..10_000 {
                let value = xorshift64(&mut rng) * (n_bins as f64);
                let grad = xorshift64(&mut rng) - 0.5;
                let hess = xorshift64(&mut rng).abs();
                hist.accumulate(black_box(value), black_box(grad), black_box(hess));
            }
            black_box(&hist);
        });
    });

    // -----------------------------------------------------------------------
    // 2. accumulate_256_bins: same but 256 bins
    // -----------------------------------------------------------------------
    group.bench_function("accumulate_256_bins", |b| {
        let n_bins = 256;
        let edges = BinEdges {
            edges: (1..n_bins).map(|i| i as f64).collect(),
        };

        b.iter(|| {
            let mut hist = FeatureHistogram::new(edges.clone());
            let mut rng: u64 = 0xDEAD_BEEF;
            for _ in 0..10_000 {
                let value = xorshift64(&mut rng) * (n_bins as f64);
                let grad = xorshift64(&mut rng) - 0.5;
                let hess = xorshift64(&mut rng).abs();
                hist.accumulate(black_box(value), black_box(grad), black_box(hess));
            }
            black_box(&hist);
        });
    });

    // -----------------------------------------------------------------------
    // 3. uniform_binning_observe_and_compute: observe 1000 values, compute 64 edges
    // -----------------------------------------------------------------------
    group.bench_function("uniform_binning_observe_and_compute", |b| {
        b.iter(|| {
            let mut binner = UniformBinning::new();
            let mut rng: u64 = 0xCAFE_BABE;
            for _ in 0..1_000 {
                let value = xorshift64(&mut rng) * 100.0;
                binner.observe(black_box(value));
            }
            let edges = binner.compute_edges(black_box(64));
            black_box(&edges);
        });
    });

    // -----------------------------------------------------------------------
    // 4. histogram_subtraction: subtract one histogram from another
    // -----------------------------------------------------------------------
    group.bench_function("histogram_subtraction", |b| {
        let n_bins = 64;
        let edges = BinEdges {
            edges: (1..n_bins).map(|i| i as f64).collect(),
        };

        // Pre-populate parent and child histograms.
        let mut parent = FeatureHistogram::new(edges.clone());
        let mut child = FeatureHistogram::new(edges);
        let mut rng: u64 = 0xFACE_FEED;

        for _ in 0..5_000 {
            let value = xorshift64(&mut rng) * (n_bins as f64);
            let grad = xorshift64(&mut rng) - 0.5;
            let hess = xorshift64(&mut rng).abs();
            parent.accumulate(value, grad, hess);
        }
        for _ in 0..2_000 {
            let value = xorshift64(&mut rng) * (n_bins as f64);
            let grad = xorshift64(&mut rng) - 0.5;
            let hess = xorshift64(&mut rng).abs();
            child.accumulate(value, grad, hess);
        }

        b.iter(|| {
            let sibling = black_box(&parent).subtract(black_box(&child));
            black_box(&sibling);
        });
    });

    group.finish();
}

criterion_group!(benches, histogram_benchmarks);
criterion_main!(benches);
