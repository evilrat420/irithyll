//! Delta-family head-to-head gauntlet.
//!
//! Benchmarks all five delta-family attention variants on a fixed associative
//! recall dataset. Reports both throughput (samples/sec) and prequential RMSE
//! so the results surface both speed and quality regressions in CI.
//!
//! # Delta family covered (5 variants)
//!
//! 1. **DeltaNet** — original error-corrective delta rule (Schlag et al., ICML 2021)
//! 2. **GatedDeltaNet (Static beta)** — GLA gate + delta rule, static β (Yang et al., ICLR 2025)
//! 3. **GatedDeltaNet (PerToken beta)** — paper-canonical data-dependent β_t
//! 4. **DeltaProduct (2 compositions)** — product of Householder delta rules (Siems et al., NeurIPS 2025)
//! 5. **RWKV-7** — vector-gated delta rule with DPLR transitions (Peng et al., 2025)
//!
//! # Dataset
//!
//! Fixed: `MqarStream` (associative recall, d_key=8, n_pairs=32).
//! All variants see the same 4000-step stream; first 400 steps are warmup (excluded
//! from RMSE). Using the same dataset across all five isolates architectural signal
//! from stream-structure confounders.
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench delta_family_gauntlet
//! ```
//!
//! Build only:
//!
//! ```bash
//! cargo bench --bench delta_family_gauntlet --no-run
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use irithyll::{
    attention::{AttentionMode, GatedDeltaMode, StreamingAttentionConfig, StreamingAttentionModel},
    generators::{MqarStream, StreamGenerator},
    StreamingLearner,
};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Input feature dimension. Must match `d_key` passed to MqarStream.
const D: usize = 8;
/// Number of (key, value) pairs per MQAR epoch.
const N_PAIRS: usize = 32;
/// Number of attention heads per model.
const HEADS: usize = 2;
/// Total prequential steps.
const N_STEPS: usize = 4_000;
/// Warmup excluded from RMSE accumulation.
const WARMUP: usize = 400;
/// PRNG seed for deterministic stream.
const STREAM_SEED: u64 = 0xBEEF_CAFE;
/// PRNG seed for model weight init — same across all for reproducibility.
const MODEL_SEED: u64 = 0xC0DE_F00D;

// ---------------------------------------------------------------------------
// Model factories — all parametrised identically for head-to-head fairness
// ---------------------------------------------------------------------------

fn make(mode: AttentionMode) -> StreamingAttentionModel {
    StreamingAttentionModel::new(
        StreamingAttentionConfig::builder()
            .d_model(D)
            .n_heads(HEADS)
            .mode(mode)
            .seed(MODEL_SEED)
            .build()
            .expect("delta_family_gauntlet: config build failed"),
    )
}

fn delta_net() -> StreamingAttentionModel {
    make(AttentionMode::DeltaNet)
}

fn gated_delta_static() -> StreamingAttentionModel {
    make(AttentionMode::GatedDeltaNet {
        beta_scale: 1.0,
        gate_mode_delta: GatedDeltaMode::Static,
    })
}

fn gated_delta_per_token() -> StreamingAttentionModel {
    make(AttentionMode::GatedDeltaNet {
        beta_scale: 1.0,
        gate_mode_delta: GatedDeltaMode::PerToken,
    })
}

fn delta_product_2() -> StreamingAttentionModel {
    make(AttentionMode::DeltaProduct {
        n_compositions: 2,
        reflections: false,
    })
}

fn rwkv7() -> StreamingAttentionModel {
    make(AttentionMode::RWKV7)
}

// ---------------------------------------------------------------------------
// Prequential harness
// ---------------------------------------------------------------------------

fn prequential(
    model: &mut StreamingAttentionModel,
    gen: &mut MqarStream,
    n_steps: usize,
    warmup: usize,
) -> (f64, f64) {
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
// Individual variant benchmarks
// ---------------------------------------------------------------------------

fn bench_delta_net(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_family");
    group.bench_function("delta_net", |b| {
        b.iter(|| {
            let mut model = delta_net();
            let mut gen = MqarStream::new(STREAM_SEED, D, N_PAIRS);
            black_box(prequential(&mut model, &mut gen, N_STEPS, WARMUP));
        });
    });
    group.finish();
}

fn bench_gated_delta_static(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_family");
    group.bench_function("gated_delta_static_beta", |b| {
        b.iter(|| {
            let mut model = gated_delta_static();
            let mut gen = MqarStream::new(STREAM_SEED, D, N_PAIRS);
            black_box(prequential(&mut model, &mut gen, N_STEPS, WARMUP));
        });
    });
    group.finish();
}

fn bench_gated_delta_per_token(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_family");
    group.bench_function("gated_delta_per_token_beta", |b| {
        b.iter(|| {
            let mut model = gated_delta_per_token();
            let mut gen = MqarStream::new(STREAM_SEED, D, N_PAIRS);
            black_box(prequential(&mut model, &mut gen, N_STEPS, WARMUP));
        });
    });
    group.finish();
}

fn bench_delta_product_2(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_family");
    group.bench_function("delta_product_2_compositions", |b| {
        b.iter(|| {
            let mut model = delta_product_2();
            let mut gen = MqarStream::new(STREAM_SEED, D, N_PAIRS);
            black_box(prequential(&mut model, &mut gen, N_STEPS, WARMUP));
        });
    });
    group.finish();
}

fn bench_rwkv7(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_family");
    group.bench_function("rwkv7", |b| {
        b.iter(|| {
            let mut model = rwkv7();
            let mut gen = MqarStream::new(STREAM_SEED, D, N_PAIRS);
            black_box(prequential(&mut model, &mut gen, N_STEPS, WARMUP));
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Summary: all five in one group, emit RMSE + throughput to stderr
// ---------------------------------------------------------------------------

fn delta_family_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_family_summary");

    type ModelFactory = fn() -> StreamingAttentionModel;
    let variants: &[(&str, ModelFactory)] = &[
        ("delta_net", delta_net),
        ("gated_delta_static", gated_delta_static),
        ("gated_delta_per_token", gated_delta_per_token),
        ("delta_product_2", delta_product_2),
        ("rwkv7", rwkv7),
    ];

    for (name, make_fn) in variants {
        let name = *name;
        let make_fn = *make_fn;
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut model = make_fn();
                let mut gen = MqarStream::new(STREAM_SEED, D, N_PAIRS);
                black_box(prequential(&mut model, &mut gen, N_STEPS, WARMUP));
            });
        });
    }

    // One-shot RMSE+tput report to stderr for CI log parsing.
    eprintln!("\n[delta_family_gauntlet] head-to-head on MQAR d={D} n_pairs={N_PAIRS}:");
    for (name, make_fn) in variants {
        let mut model = make_fn();
        let mut gen = MqarStream::new(STREAM_SEED, D, N_PAIRS);
        let (rmse, tput) = prequential(&mut model, &mut gen, N_STEPS, WARMUP);
        eprintln!("  {name:<32} rmse={rmse:.4}  tput={tput:.0} samples/sec");
    }
    eprintln!();

    group.finish();
}

criterion_group!(
    benches,
    bench_delta_net,
    bench_gated_delta_static,
    bench_gated_delta_per_token,
    bench_delta_product_2,
    bench_rwkv7,
    delta_family_summary,
);
criterion_main!(benches);
