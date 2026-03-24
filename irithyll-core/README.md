# irithyll-core

Standalone streaming ML engine for `no_std` + `alloc` targets. Train and deploy
gradient boosted trees on anything with a heap allocator — microcontrollers,
WASM, embedded Linux, or full desktops.

## What's in the box

- **SGBT ensemble** — streaming gradient boosted trees with 10 variants
  (distributional, MoE, multiclass, quantile, bagged, parallel, ARF, adaptive)
- **Hoeffding trees** — statistically sound split decisions via Hoeffding bound
- **Histogram binning** — uniform, categorical, quantile sketch, k-means, SIMD
- **Drift detection** — ADWIN, Page-Hinkley, DDM with automatic tree replacement
- **Reservoir computing** — NG-RC (time-delay polynomial) and ESN (cycle reservoir) with RLS readout
- **State space models** — selective SSM with diagonal A, ZOH discretization, input-dependent gating
- **Spiking neural networks** — `SpikeNetFixed` with Q1.14 integer LIF neurons, e-prop learning, delta encoding (64 neurons in 22KB)
- **Streaming linear attention** — 7 modes (RetNet, Hawk, GLA, DeltaNet, GatedDeltaNet, RWKV, mLSTM) via unified recurrence engine
- **Continual learning** — EWC (elastic weight consolidation), neuron regeneration (Dohare et al. 2024), drift-triggered parameter isolation
- **SIMD primitives** — AVX2-accelerated `simd_dot` and `simd_mat_vec` with automatic runtime detection and scalar fallback
- **Loss functions** — squared, logistic, Huber, softmax, expectile, quantile
- **Packed inference** — 12-byte f32 nodes (66ns predict on Cortex-M0+) and
  8-byte int16 nodes (integer-only traversal, zero float ops)
- **Zero-copy views** — `EnsembleView::from_bytes(&[u8])`, no allocation after validation

## Feature flags

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `alloc` | No | Training: histograms, trees, ensembles, drift detection, reservoir, SSM, SNN |
| `std` | No | Implies `alloc`. HashMap-based named features, SIMD runtime detection |
| `serde` | No | Serialize/deserialize configs and model state |
| `parallel` | No | Rayon-based parallel tree training |
| `kmeans-binning` | No | K-means histogram binning strategy |
| `simd` | No | AVX2 histogram acceleration (requires `std`) |

Without any features, irithyll-core provides packed inference only — runs on
bare metal with zero dependencies beyond `libm`.

## Quick start

Training (requires `alloc`):

```rust
use irithyll_core::ensemble::config::SGBTConfig;
use irithyll_core::ensemble::SGBT;
use irithyll_core::loss::squared::SquaredLoss;
use irithyll_core::sample::SampleRef;

let config = SGBTConfig::builder()
    .n_steps(50)
    .learning_rate(0.05)
    .build()
    .unwrap();

let mut model = SGBT::with_loss(config, SquaredLoss);

// Train one sample at a time
let sample = SampleRef::new(&[1.0, 2.0, 3.0], 0.5);
model.train_one(&sample);
let prediction = model.predict(&[1.0, 2.0, 3.0]);
```

Inference on embedded (no features needed):

```rust
use irithyll_core::EnsembleView;

// Load packed binary exported from a trained model
let packed_bytes: &[u8] = include_bytes!("model.bin");
let view = EnsembleView::from_bytes(packed_bytes).unwrap();
let prediction = view.predict(&[1.0f32, 2.0, 3.0]);
```

## License

MIT OR Apache-2.0
