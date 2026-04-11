//! Self-comparison benchmarks: irithyll inference paths head-to-head.
//!
//! Compares SGBT (f64 tree walk), packed EnsembleView (f32 branch-free),
//! and quantized QuantizedEnsembleView (i16) on the same trained model.
//! No external crates — pure irithyll self-comparison for Reddit/marketing.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use irithyll::export_embedded::{export_packed, export_packed_i16};
use irithyll::{SGBTConfig, Sample, SGBT};
use irithyll_core::turbo_quant::{quantize, QuantMode, TurboQuantizedView};
use irithyll_core::{EnsembleView, QuantizedEnsembleView};

// ---------------------------------------------------------------------------
// Deterministic PRNG (no rand dependency)
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Train a deterministic model for benchmarking.
///
/// Uses xorshift64 for reproducible data with a non-trivial target function:
/// `y = sum_i (i+1) * x_i` to ensure the trees actually learn splits.
fn train_bench_model(
    n_steps: usize,
    n_features: usize,
    max_depth: usize,
    n_samples: usize,
) -> SGBT {
    let config = SGBTConfig::builder()
        .n_steps(n_steps)
        .learning_rate(0.01)
        .grace_period(20)
        .max_depth(max_depth)
        .n_bins(32)
        .build()
        .unwrap();
    let mut model = SGBT::new(config);
    let mut rng: u64 = 0xBEEF_CAFE;
    for _ in 0..n_samples {
        let features: Vec<f64> = (0..n_features)
            .map(|_| xorshift64(&mut rng) * 10.0 - 5.0)
            .collect();
        let target = features
            .iter()
            .enumerate()
            .fold(0.0, |acc, (i, &f)| acc + (i as f64 + 1.0) * f);
        model.train_one(&Sample::new(features, target));
    }
    model
}

/// Generate deterministic f64 feature vectors for SGBT predict.
fn generate_f64_samples(n: usize, n_features: usize) -> Vec<Vec<f64>> {
    let mut rng: u64 = 0xDEAD_1337;
    (0..n)
        .map(|_| {
            (0..n_features)
                .map(|_| xorshift64(&mut rng) * 10.0 - 5.0)
                .collect()
        })
        .collect()
}

/// Generate deterministic f32 feature vectors for packed/quantized predict.
fn generate_f32_samples(n: usize, n_features: usize) -> Vec<Vec<f32>> {
    let mut rng: u64 = 0xDEAD_1337;
    (0..n)
        .map(|_| {
            (0..n_features)
                .map(|_| {
                    let v = xorshift64(&mut rng) * 10.0 - 5.0;
                    v as f32
                })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Group 1: Single prediction latency comparison
// ---------------------------------------------------------------------------

/// Compare single-sample predict latency across all three inference paths.
///
/// All three use the same underlying model (50 trees, depth 4, 10 features):
/// - `irithyll_sgbt_f64`:       Standard SGBT predict (f64 tree walk)
/// - `irithyll_packed_f32`:     Packed EnsembleView predict (f32 branch-free)
/// - `irithyll_quantized_i16`:  QuantizedEnsembleView predict (i16)
fn comparison_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_predict");
    let n_features = 10;

    // Train the shared model
    let model = train_bench_model(50, n_features, 4, 1000);

    // Export to both packed formats
    let packed_f32 = export_packed(&model, n_features);
    let view_f32 = EnsembleView::from_bytes(&packed_f32).unwrap();

    let packed_i16 = export_packed_i16(&model, n_features);
    let view_i16 = QuantizedEnsembleView::from_bytes(&packed_i16).unwrap();

    // Prepare feature vectors (same values, different types)
    let features_f64: Vec<f64> = (0..n_features).map(|i| (i as f64) * 0.1).collect();
    let features_f32: Vec<f32> = (0..n_features).map(|i| (i as f32) * 0.1).collect();

    // Bench: SGBT f64 predict (standard tree walk)
    group.bench_function("irithyll_sgbt_f64", |b| {
        b.iter(|| model.predict(black_box(&features_f64)))
    });

    // Bench: Packed f32 predict (branch-free, cache-optimized)
    group.bench_function("irithyll_packed_f32", |b| {
        b.iter(|| view_f32.predict(black_box(&features_f32)))
    });

    // Bench: Quantized i16 predict (integer-only traversal)
    group.bench_function("irithyll_quantized_i16", |b| {
        b.iter(|| view_i16.predict(black_box(&features_f32)))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: Batch prediction throughput comparison
// ---------------------------------------------------------------------------

/// Compare batch prediction throughput (10,000 samples) across all three paths.
///
/// Uses `Throughput::Elements(10_000)` so criterion reports samples/sec.
/// - `sgbt_batch_f64`:       SGBT predict loop over f64 samples
/// - `packed_batch_f32`:     EnsembleView.predict_batch (f32, x4 interleaved)
/// - `quantized_batch_i16`:  QuantizedEnsembleView.predict_batch (i16 inline)
fn comparison_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_batch_throughput");
    let n_features = 10;
    let batch_size: u64 = 10_000;

    group.throughput(Throughput::Elements(batch_size));

    // Train the shared model
    let model = train_bench_model(50, n_features, 4, 1000);

    // Export to both packed formats
    let packed_f32 = export_packed(&model, n_features);
    let view_f32 = EnsembleView::from_bytes(&packed_f32).unwrap();

    let packed_i16 = export_packed_i16(&model, n_features);
    let view_i16 = QuantizedEnsembleView::from_bytes(&packed_i16).unwrap();

    // Pre-generate sample data (same underlying values for fairness)
    let samples_f64 = generate_f64_samples(batch_size as usize, n_features);
    let samples_f32 = generate_f32_samples(batch_size as usize, n_features);
    let sample_refs_f32: Vec<&[f32]> = samples_f32.iter().map(|s| s.as_slice()).collect();

    // Output buffers
    let mut out_f32 = vec![0.0f32; batch_size as usize];
    let mut out_i16 = vec![0.0f32; batch_size as usize];

    // Bench: SGBT batch f64 (predict loop)
    group.bench_function("sgbt_batch_f64", |b| {
        b.iter(|| {
            for sample in &samples_f64 {
                black_box(model.predict(black_box(sample.as_slice())));
            }
        })
    });

    // Bench: Packed batch f32 (predict_batch with x4 interleaving)
    group.bench_function("packed_batch_f32", |b| {
        b.iter(|| {
            view_f32.predict_batch(black_box(&sample_refs_f32), &mut out_f32);
            black_box(&out_f32);
        })
    });

    // Bench: Quantized batch i16 (predict_batch with inline quantization)
    group.bench_function("quantized_batch_i16", |b| {
        b.iter(|| {
            view_i16.predict_batch(black_box(&sample_refs_f32), &mut out_i16);
            black_box(&out_i16);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: TurboQuant weight vector inference throughput
//
// Benchmarks the TurboQuant quantized dot-product kernel in three modes:
//   - 8-bit (256 levels, ~8x compression, near-lossless)
//   - 3.5-bit (11 levels, ~14x compression, aggressive)
//   - 2.5-bit (5 levels, ~21x compression, ultra-aggressive)
//
// Compares against a raw f64 dot-product over the same weight vector to
// quantify the compression-vs-throughput tradeoff.
//
// Weight vectors represent typical RLS readout weights from sLSTM/TTT models
// (d_model=64 hidden state -> 1 output = 64 weights).
// ---------------------------------------------------------------------------

/// Compute a raw f64 dot product (baseline for TurboQuant comparison).
#[inline]
fn dot_f64(weights: &[f64], features: &[f64]) -> f64 {
    weights
        .iter()
        .zip(features.iter())
        .map(|(&w, &f)| w * f)
        .sum()
}

fn turbo_quant_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbo_quant_vs_f64");
    let n_weights: usize = 64; // d_model=64 RLS readout
    let batch_size: u64 = 10_000;

    group.throughput(Throughput::Elements(batch_size));

    // Build a deterministic weight vector (simulate trained RLS readout)
    let mut rng: u64 = 0xFACE_CAFE;
    let weights_f64: Vec<f64> = (0..n_weights)
        .map(|_| {
            let v = xorshift64(&mut rng);
            (v - 0.5) * 0.2 // centered around 0, typical RLS scale
        })
        .collect();

    // Quantize once in each mode
    let tq_8bit = quantize(&weights_f64, QuantMode::Bits8, 0xBEEF_0001);
    let tq_3_5bit = quantize(&weights_f64, QuantMode::Bits3_5, 0xBEEF_0001);
    let tq_2_5bit = quantize(&weights_f64, QuantMode::Bits2_5, 0xBEEF_0001);

    // Serialize then parse as views (zero-copy, embedded-style access)
    let packed_8bit = tq_8bit.to_bytes();
    let packed_3_5bit = tq_3_5bit.to_bytes();
    let packed_2_5bit = tq_2_5bit.to_bytes();
    let view_8bit = TurboQuantizedView::from_bytes(&packed_8bit).unwrap();
    let view_3_5bit = TurboQuantizedView::from_bytes(&packed_3_5bit).unwrap();
    let view_2_5bit = TurboQuantizedView::from_bytes(&packed_2_5bit).unwrap();

    // Generate deterministic feature vectors for inference loop
    let mut rng2: u64 = 0xDEAD_7777;
    let feature_batches: Vec<Vec<f64>> = (0..batch_size as usize)
        .map(|_| {
            (0..n_weights)
                .map(|_| (xorshift64(&mut rng2) - 0.5) * 2.0)
                .collect()
        })
        .collect();

    // Baseline: raw f64 dot product (no quantization overhead)
    group.bench_function("raw_f64_dot", |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for feat in &feature_batches {
                sum += dot_f64(black_box(&weights_f64), black_box(feat));
            }
            black_box(sum)
        })
    });

    // TurboQuant 8-bit: 256 levels, near-lossless
    group.bench_function("turbo_quant_8bit", |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for feat in &feature_batches {
                sum += view_8bit.predict(black_box(feat));
            }
            black_box(sum)
        })
    });

    // TurboQuant 3.5-bit: 11 levels, ~14x compression
    group.bench_function("turbo_quant_3_5bit", |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for feat in &feature_batches {
                sum += view_3_5bit.predict(black_box(feat));
            }
            black_box(sum)
        })
    });

    // TurboQuant 2.5-bit: 5 levels, ~21x compression (ultra-aggressive)
    group.bench_function("turbo_quant_2_5bit", |b| {
        b.iter(|| {
            let mut sum = 0.0f64;
            for feat in &feature_batches {
                sum += view_2_5bit.predict(black_box(feat));
            }
            black_box(sum)
        })
    });

    // TurboQuant 3.5-bit with zero-alloc scratch buffer (embedded-style)
    // predict_with_scratch avoids allocation per call -- critical for embedded
    let padded_len = n_weights.next_power_of_two();
    group.bench_function("turbo_quant_3_5bit_scratch", |b| {
        let mut scratch = vec![0.0f64; padded_len];
        b.iter(|| {
            let mut sum = 0.0f64;
            for feat in &feature_batches {
                sum += view_3_5bit.predict_with_scratch(black_box(feat), &mut scratch);
            }
            black_box(sum)
        })
    });

    group.finish();

    // Compression ratio summary (printed once, not measured by criterion)
    let f64_bytes = n_weights * 8;
    let quant_8_bytes = packed_8bit.len();
    let quant_3_5_bytes = packed_3_5bit.len();
    let quant_2_5_bytes = packed_2_5bit.len();
    eprintln!(
        "[turbo_quant] {n_weights} weights: f64={f64_bytes}B | \
         8-bit={quant_8_bytes}B ({:.1}x) | \
         3.5-bit={quant_3_5_bytes}B ({:.1}x) | \
         2.5-bit={quant_2_5_bytes}B ({:.1}x)",
        f64_bytes as f64 / quant_8_bytes as f64,
        f64_bytes as f64 / quant_3_5_bytes as f64,
        f64_bytes as f64 / quant_2_5_bytes as f64,
    );
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    comparison_benches,
    comparison_predict,
    comparison_batch_throughput,
    turbo_quant_throughput
);
criterion_main!(comparison_benches);
