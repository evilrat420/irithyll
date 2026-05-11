//! MOA-standard generator throughput bench.
//!
//! Measures prequential RMSE and throughput (samples/sec) for the MOA-standard
//! synthetic generators (`MqarStream`, `NeedleStream`, `PeriodicStream`,
//! `ParityStream`) across four model families:
//!
//! - **SGBT** — streaming gradient-boosted trees (tree ensemble baseline)
//! - **Mamba-3 V3Exp** — exp-trapezoidal complex SSM
//! - **Log-Linear Attention** — O(log T) Fenwick hierarchy (v10 headline)
//! - **StreamingKAN** — B-spline KAN (symbolic regression baseline)
//!
//! # Protocol
//!
//! Prequential (test-then-train). Warmup is excluded from RMSE accumulation.
//! Throughput measured as wall-clock samples-per-second using `std::time::Instant`.
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench moa_generators_bench
//! ```
//!
//! Build only:
//!
//! ```bash
//! cargo bench --bench moa_generators_bench --no-run
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use irithyll::{
    attention::{AttentionMode, GatedDeltaMode, StreamingAttentionConfig, StreamingAttentionModel},
    generators::{MqarStream, NeedleStream, ParityStream, PeriodicStream, StreamGenerator},
    kan::{KANConfig, StreamingKAN},
    sgbt,
    ssm::{MambaConfig, MambaVersion, StreamingMamba},
    StreamingLearner,
};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Generator parameters
// ---------------------------------------------------------------------------

/// MqarStream: d_key=8, n_pairs=32 (smaller than default 128 for bench speed).
const MQAR_D: usize = 8;
const MQAR_N_PAIRS: usize = 32;

/// NeedleStream: 8 features, haystack_size=64.
const NEEDLE_D: usize = 8;
const NEEDLE_HAYSTACK: usize = 64;

/// PeriodicStream: window=10, period=20.
const PERIODIC_WINDOW: usize = 10;
const PERIODIC_PERIOD: usize = 20;

/// ParityStream: 8 bits, 4 parity bits.
const PARITY_N_BITS: usize = 8;
const PARITY_COUNT: usize = 4;

// ---------------------------------------------------------------------------
// Model factories
// ---------------------------------------------------------------------------

fn make_sgbt() -> impl StreamingLearner {
    sgbt(20, 0.05)
}

fn make_mamba_v3exp(d_in: usize) -> StreamingMamba {
    let n_groups = if d_in >= 4 { 2 } else { 1 };
    StreamingMamba::new(
        MambaConfig::builder()
            .d_in(d_in)
            .n_state(16)
            .version(MambaVersion::V3Exp { use_bcnorm: true })
            .n_groups(n_groups)
            .seed(0xCAFE_BABE)
            .build()
            .expect("moa_generators_bench: mamba v3exp config"),
    )
}

fn make_log_linear(d_model: usize) -> StreamingAttentionModel {
    let n_heads = if d_model >= 4 { 2 } else { 1 };
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(d_model)
            .n_heads(n_heads)
            .mode(AttentionMode::LogLinear {
                inner: Box::new(AttentionMode::GatedDeltaNet {
                    beta_scale: 1.0,
                    gate_mode_delta: GatedDeltaMode::Static,
                }),
                max_levels: 16,
                lambda_init: 1.0 / 16.0,
            })
            .seed(0xDEAD_BEEF)
            .build()
            .expect("moa_generators_bench: log_linear config"),
    )
}

fn make_kan(d_in: usize) -> StreamingKAN {
    StreamingKAN::new(
        KANConfig::builder()
            .layer_sizes(vec![d_in, 16, 1])
            .grid_size(8)
            .learning_rate(0.05)
            .build()
            .expect("moa_generators_bench: kan config"),
    )
}

// ---------------------------------------------------------------------------
// Prequential harness
// ---------------------------------------------------------------------------

fn prequential<G, L>(gen: &mut G, model: &mut L, n_steps: usize, warmup: usize) -> (f64, f64)
where
    G: StreamGenerator,
    L: StreamingLearner,
{
    let mut sse = 0.0_f64;
    let mut count = 0usize;
    let start = Instant::now();
    for t in 0..n_steps {
        let (features, target) = gen.next_sample();
        let pred = model.predict(&features);
        model.train(&features, target);
        if t >= warmup {
            let err = pred - target;
            sse += err * err;
            count += 1;
        }
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let rmse = if count > 0 {
        (sse / count as f64).sqrt()
    } else {
        f64::NAN
    };
    (rmse, n_steps as f64 / elapsed)
}

// ---------------------------------------------------------------------------
// MQAR stream
// ---------------------------------------------------------------------------

fn bench_mqar_stream(c: &mut Criterion) {
    let mut group = c.benchmark_group("moa_mqar_stream");

    group.bench_function("sgbt", |b| {
        b.iter(|| {
            let mut gen = MqarStream::new(42, MQAR_D, MQAR_N_PAIRS);
            let mut model = make_sgbt();
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("mamba_v3exp", |b| {
        b.iter(|| {
            let mut gen = MqarStream::new(42, MQAR_D, MQAR_N_PAIRS);
            let mut model = make_mamba_v3exp(MQAR_D);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("log_linear", |b| {
        b.iter(|| {
            let mut gen = MqarStream::new(42, MQAR_D, MQAR_N_PAIRS);
            let mut model = make_log_linear(MQAR_D);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("kan", |b| {
        b.iter(|| {
            let mut gen = MqarStream::new(42, MQAR_D, MQAR_N_PAIRS);
            let mut model = make_kan(MQAR_D);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Needle stream
// ---------------------------------------------------------------------------

fn bench_needle_stream(c: &mut Criterion) {
    let mut group = c.benchmark_group("moa_needle_stream");

    group.bench_function("sgbt", |b| {
        b.iter(|| {
            let mut gen = NeedleStream::new(42, NEEDLE_D, NEEDLE_HAYSTACK);
            let mut model = make_sgbt();
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("mamba_v3exp", |b| {
        b.iter(|| {
            let mut gen = NeedleStream::new(42, NEEDLE_D, NEEDLE_HAYSTACK);
            let mut model = make_mamba_v3exp(NEEDLE_D);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("log_linear", |b| {
        b.iter(|| {
            let mut gen = NeedleStream::new(42, NEEDLE_D, NEEDLE_HAYSTACK);
            let mut model = make_log_linear(NEEDLE_D);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("kan", |b| {
        b.iter(|| {
            let mut gen = NeedleStream::new(42, NEEDLE_D, NEEDLE_HAYSTACK);
            let mut model = make_kan(NEEDLE_D);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Periodic stream
// ---------------------------------------------------------------------------

fn bench_periodic_stream(c: &mut Criterion) {
    let mut group = c.benchmark_group("moa_periodic_stream");

    group.bench_function("sgbt", |b| {
        b.iter(|| {
            let mut gen = PeriodicStream::new(42, PERIODIC_WINDOW, PERIODIC_PERIOD);
            let mut model = make_sgbt();
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("mamba_v3exp", |b| {
        b.iter(|| {
            let mut gen = PeriodicStream::new(42, PERIODIC_WINDOW, PERIODIC_PERIOD);
            let mut model = make_mamba_v3exp(PERIODIC_WINDOW);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("log_linear", |b| {
        b.iter(|| {
            let mut gen = PeriodicStream::new(42, PERIODIC_WINDOW, PERIODIC_PERIOD);
            let mut model = make_log_linear(PERIODIC_WINDOW);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("kan", |b| {
        b.iter(|| {
            let mut gen = PeriodicStream::new(42, PERIODIC_WINDOW, PERIODIC_PERIOD);
            let mut model = make_kan(PERIODIC_WINDOW);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Parity stream
// ---------------------------------------------------------------------------

fn bench_parity_stream(c: &mut Criterion) {
    let mut group = c.benchmark_group("moa_parity_stream");

    group.bench_function("sgbt", |b| {
        b.iter(|| {
            let mut gen = ParityStream::new(42, PARITY_N_BITS, PARITY_COUNT);
            let mut model = make_sgbt();
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("mamba_v3exp", |b| {
        b.iter(|| {
            let mut gen = ParityStream::new(42, PARITY_N_BITS, PARITY_COUNT);
            let mut model = make_mamba_v3exp(PARITY_N_BITS);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("log_linear", |b| {
        b.iter(|| {
            let mut gen = ParityStream::new(42, PARITY_N_BITS, PARITY_COUNT);
            let mut model = make_log_linear(PARITY_N_BITS);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.bench_function("kan", |b| {
        b.iter(|| {
            let mut gen = ParityStream::new(42, PARITY_N_BITS, PARITY_COUNT);
            let mut model = make_kan(PARITY_N_BITS);
            black_box(prequential(&mut gen, &mut model, 1000, 100));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_mqar_stream,
    bench_needle_stream,
    bench_periodic_stream,
    bench_parity_stream,
);
criterion_main!(benches);
