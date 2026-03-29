```
.__       .__  __  .__           .__  .__
│__│______│__│╱  │_│  │__ ___.__.│  │ │  │
│  ╲_  __ ╲  ╲   __╲  │  <   │  ││  │ │  │
│  ││  │ ╲╱  ││  │ │   Y  ╲___  ││  │_│  │__
│__││__│  │__││__│ │___│  ╱ ____││____╱____╱
                        ╲╱╲╱
```

[![Crates.io](https://img.shields.io/crates/v/irithyll.svg)](https://crates.io/crates/irithyll)
[![Documentation](https://docs.rs/irithyll/badge.svg)](https://docs.rs/irithyll)
[![CI](https://github.com/evilrat420/irithyll/actions/workflows/ci.yml/badge.svg)](https://github.com/evilrat420/irithyll/actions)
[![License](https://img.shields.io/crates/l/irithyll.svg)](https://github.com/evilrat420/irithyll)
[![MSRV](https://img.shields.io/badge/MSRV-1.75-blue.svg)](https://blog.rust-lang.org/2023/12/28/Rust-1.75.0.html)
[![GitHub stars](https://img.shields.io/github/stars/evilrat420/irithyll?style=social)](https://github.com/evilrat420/irithyll)

**Streaming machine learning in Rust** -- gradient boosted trees, neural streaming architectures (reservoir computing, state space models, spiking networks), kernel methods, linear models, and composable pipelines, all learning one sample at a time.

```rust
use irithyll::{pipe, normalizer, sgbt, StreamingLearner};

let mut model = pipe(normalizer()).learner(sgbt(50, 0.01));
model.train(&[100.0, 0.5], 42.0);
let prediction = model.predict(&[100.0, 0.5]);
```

## Workspace

irithyll is structured as a Cargo workspace with three crates:

| Crate | Description | `no_std` | Allocator |
|-------|-------------|----------|-----------|
| **`irithyll`** | Training, streaming algorithms, pipelines, I/O, async | No | std |
| **`irithyll-core`** | Packed inference engine (12-byte nodes, branch-free traversal) | Yes | Zero-alloc |
| **`irithyll-python`** | PyO3 Python bindings for `irithyll` | No | std |

`irithyll-core` cross-compiles for bare-metal targets (verified: `cargo check --target thumbv6m-none-eabi`) and has zero dependencies.

## Why irithyll?

- **30+ streaming algorithms** under one unified [`StreamingLearner`](https://docs.rs/irithyll/latest/irithyll/trait.StreamingLearner.html) trait
- **One sample at a time** -- O(1) memory per model, no batches, no windows, no retraining
- **Embedded deployment** -- train with `irithyll`, export to packed binary, infer with `irithyll-core` on bare metal
- **Composable pipelines** -- chain preprocessors and learners with a builder API
- **Concept drift adaptation** -- automatic model replacement when the data distribution shifts
- **Confidence intervals** -- prediction uncertainty from RLS and conformal methods
- **Production-grade** -- async streaming, SIMD acceleration, Arrow/Parquet I/O, ONNX export
- **Neural streaming architectures** -- reservoir computing, SSMs, SNNs, TTT, KAN, streaming attention (7 modes)
- **Neural MoE** -- polymorphic experts with top-k routing and load balancing
- **Streaming AutoML** -- tournament racing, complexity-adjusted elimination, drift re-racing
- **Principled adaptation** -- honest_sigma, adaptive tree replacement, proactive pruning, soft routing
- **Pure Rust** -- zero unsafe in `irithyll`, deterministic, serializable, 2,500+ tests

## Algorithms

Every algorithm implements `StreamingLearner` -- train and predict with the same two-method interface.

| Algorithm | Type | Use Case | Per-Sample Cost |
|-----------|------|----------|-----------------|
| [`SGBT`](https://docs.rs/irithyll/latest/irithyll/struct.SGBT.html) | Gradient boosted trees | General regression/classification | O(n_steps * depth) |
| [`AdaptiveSGBT`](https://docs.rs/irithyll/latest/irithyll/struct.AdaptiveSGBT.html) | SGBT + LR scheduling | Decaying/cycling learning rates | O(n_steps * depth) |
| [`MulticlassSGBT`](https://docs.rs/irithyll/latest/irithyll/struct.MulticlassSGBT.html) | One-vs-rest SGBT | Multi-class classification | O(classes * n_steps * depth) |
| [`MultiTargetSGBT`](https://docs.rs/irithyll/latest/irithyll/struct.MultiTargetSGBT.html) | Independent SGBTs | Multi-output regression | O(targets * n_steps * depth) |
| [`DistributionalSGBT`](https://docs.rs/irithyll/latest/irithyll/struct.DistributionalSGBT.html) | Mean + variance SGBT | Prediction uncertainty | O(2 * n_steps * depth) |
| [`KRLS`](https://docs.rs/irithyll/latest/irithyll/struct.KRLS.html) | Kernel recursive LS | Nonlinear regression (sin, exp, ...) | O(budget^2) |
| [`RecursiveLeastSquares`](https://docs.rs/irithyll/latest/irithyll/struct.RecursiveLeastSquares.html) | RLS with confidence | Linear regression + uncertainty | O(d^2) |
| [`StreamingLinearModel`](https://docs.rs/irithyll/latest/irithyll/struct.StreamingLinearModel.html) | SGD linear model | Fast linear baseline | O(d) |
| [`StreamingPolynomialRegression`](https://docs.rs/irithyll/latest/irithyll/struct.StreamingPolynomialRegression.html) | Polynomial SGD | Polynomial curve fitting | O(d * degree) |
| [`GaussianNB`](https://docs.rs/irithyll/latest/irithyll/struct.GaussianNB.html) | Naive Bayes | Text/categorical classification | O(d * classes) |
| [`MondrianForest`](https://docs.rs/irithyll/latest/irithyll/struct.MondrianForest.html) | Random forest variant | Streaming ensemble regression | O(n_trees * depth) |
| [`LocallyWeightedRegression`](https://docs.rs/irithyll/latest/irithyll/struct.LocallyWeightedRegression.html) | Memory-based | Locally varying relationships | O(window) |

**Preprocessing** (implements `StreamingPreprocessor`):

| Preprocessor | Description |
|-------------|-------------|
| [`IncrementalNormalizer`](https://docs.rs/irithyll/latest/irithyll/struct.IncrementalNormalizer.html) | Welford's online standardization |
| [`OnlineFeatureSelector`](https://docs.rs/irithyll/latest/irithyll/struct.OnlineFeatureSelector.html) | Streaming mutual-information feature selection |
| [`CCIPCA`](https://docs.rs/irithyll/latest/irithyll/struct.CCIPCA.html) | O(kd) streaming PCA without covariance matrices |

## Neural Streaming Architectures

Three families of neural architectures, all implementing `StreamingLearner` -- train and predict one sample at a time, compose in pipelines, no batching required.

### Reservoir Computing

| Model | Description | Per-Sample Cost |
|-------|-------------|-----------------|
| [`NextGenRC`](https://docs.rs/irithyll/latest/irithyll/reservoir/struct.NextGenRC.html) | Polynomial features of time-delayed observations + RLS readout. No random matrices. Trains in <10 samples. Based on Gauthier et al. 2021 (*Nature Communications*). | O(k * s * d^degree) |
| [`EchoStateNetwork`](https://docs.rs/irithyll/latest/irithyll/reservoir/struct.EchoStateNetwork.html) | Deterministic cycle/ring reservoir (O(N) weights, not O(N^2)) with leaky integration + RLS readout. Based on Rodan & Tino 2010, Martinuzzi 2025. | O(N^2) readout |
| [`ESNPreprocessor`](https://docs.rs/irithyll/latest/irithyll/reservoir/struct.ESNPreprocessor.html) | Use ESN as a pipeline preprocessor feeding any downstream learner. | O(N) reservoir step |

### State Space Models (SSM / Mamba)

| Model | Description | Per-Sample Cost |
|-------|-------------|-----------------|
| [`StreamingMamba`](https://docs.rs/irithyll/latest/irithyll/ssm/struct.StreamingMamba.html) | Selective state space model with input-dependent B, C, Delta. ZOH discretization, diagonal A matrix. First streaming SSM implementation in Rust. Based on Gu & Dao 2023. | O(D * N) |
| [`MambaPreprocessor`](https://docs.rs/irithyll/latest/irithyll/ssm/struct.MambaPreprocessor.html) | SSM temporal features feeding SGBT or other learners in a pipeline. | O(D * N) |

### Spiking Neural Networks (SNN / e-prop)

| Model | Description | Per-Sample Cost |
|-------|-------------|-----------------|
| [`SpikeNet`](https://docs.rs/irithyll/latest/irithyll/snn/struct.SpikeNet.html) | LIF neurons with e-prop online learning rule. Delta spike encoding for continuous features. Based on Bellec et al. 2020 (*Nature Communications*), Neftci et al. 2019. | O(N_hidden^2) |
| [`SpikeNetFixed`](https://docs.rs/irithyll/latest/irithyll/irithyll_core/snn/struct.SpikeNetFixed.html) | Full `no_std` training in Q1.14 integer arithmetic. 64 neurons fits in 22KB (Cortex-M0+ 32KB SRAM). Lives in `irithyll-core`. | O(N_hidden^2) |

### Test-Time Training (TTT)

| Model | Description | Per-Sample Cost |
|-------|-------------|-----------------|
| [`StreamingTTT`](https://docs.rs/irithyll/latest/irithyll/ttt/struct.StreamingTTT.html) | Hidden state IS a linear model updated by gradient descent on self-supervised reconstruction every step. Titans-style weight decay (forgetting) and momentum for non-stationary streaming. Based on Sun et al. 2024, Behrouz et al. 2025. | O(d\_state^2) |

### Kolmogorov-Arnold Networks (KAN)

| Model | Description | Per-Sample Cost |
|-------|-------------|-----------------|
| [`StreamingKAN`](https://docs.rs/irithyll/latest/irithyll/kan/struct.StreamingKAN.html) | Learnable B-spline activations on edges. Sparse updates: only 4 coefficients per edge per sample (cubic splines). Bounded gradients prevent explosion. Based on Liu et al. 2024, Hoang et al. 2026. | O(n\_edges * 4) |

All neural models also have preprocessor variants (`ESNPreprocessor`, `MambaPreprocessor`, `SpikePreprocessor`) that implement `StreamingPreprocessor` for pipeline composition.

## Principled Streaming Adaptation

Principled streaming adaptation in gradient boosted ensembles:

| Feature | Description |
|---------|-------------|
| `honest_sigma` | Tree contribution variance — instant epistemic uncertainty from ensemble disagreement. Zero hyperparameters, reacts in one sample. |
| `adaptive_mts` | Sigma-modulated tree replacement speed. High uncertainty → faster cycling. Low uncertainty → more accumulation. |
| `proactive_prune` | Percentile-based worst-tree replacement. Maintains plasticity (Dohare+2024 Nature). |
| `soft_routing` | Per-node auto-bandwidth soft decision trees. Continuous weighted blends, no step discontinuities. |

## Streaming AutoML

Online hyperparameter optimization via champion-challenger tournament racing -- the first streaming AutoML framework in Rust.

| Component | Description |
|-----------|-------------|
| [`AutoTuner`](https://docs.rs/irithyll/latest/irithyll/automl/struct.AutoTuner.html) | Tournament successive halving with champion-challenger racing. Implements `StreamingLearner`. |
| [`Factory`](https://docs.rs/irithyll/latest/irithyll/automl/struct.Factory.html) | Unified model factory. `Factory::sgbt()`, `Factory::esn()`, `Factory::distributional()`, `Factory::mamba()`, `Factory::attention()`, `Factory::spike_net()`. |
| Complexity-adjusted elimination | MDL-inspired penalty: `score = error + complexity/n_seen`. Favors simple models on sparse data, relaxes with evidence. |
| Drift-triggered re-racing | ADWIN detects champion error drift, aborts tournament, starts fresh with expanded bracket. |
| Warmup-aware racing | Neural models with cold-start phases are protected from premature elimination. |
| Statistical early stopping | Welford z-test kills clearly-bad candidates before round budget expires. |

```rust
use irithyll::{auto_tune, automl::Factory, StreamingLearner};

// One-liner: auto-tune SGBT
let mut tuner = auto_tune(Factory::sgbt(5));

// Multi-factory: race trees vs neural architectures
let mut tuner = AutoTuner::builder()
    .add_factory(Factory::sgbt(5))
    .add_factory(Factory::esn())
    .add_factory(Factory::mamba(5))
    .use_drift_rerace(true)
    .build();

tuner.train(&[1.0, 2.0, 3.0, 4.0, 5.0], 10.0);
let pred = tuner.predict(&[1.0, 2.0, 3.0, 4.0, 5.0]);
```

Based on Wu et al. (2021) ChaCha, Qi et al. (2023) Discounted TS, Yamanishi (2018) MDL.

## Neural Mixture of Experts

Polymorphic MoE where each expert can be **any** `StreamingLearner` -- mix ESN, Mamba, SpikeNet, SGBT, and attention models in one ensemble.

| Component | Description |
|-----------|-------------|
| [`NeuralMoE`](https://docs.rs/irithyll/latest/irithyll/moe/struct.NeuralMoE.html) | Polymorphic MoE with top-k sparse routing. Implements `StreamingLearner`. |
| Linear softmax router | Learned online via SGD on cross-entropy against best expert. |
| Load balancing | Per-expert routing bias prevents collapse (DeepSeek-v3 style). |
| Dead expert reset | EWMA utilization tracking; experts below threshold are auto-reset. |

```rust
use irithyll::{sgbt, esn, spikenet, moe::NeuralMoE, StreamingLearner};

let mut moe = NeuralMoE::builder()
    .expert(sgbt(50, 0.01))
    .expert(sgbt(100, 0.005))
    .expert_with_warmup(esn(50, 0.9), 50)
    .expert_with_warmup(spikenet(32), 20)
    .top_k(2)
    .build();

moe.train(&[1.0, 2.0, 3.0], 4.0);
let pred = moe.predict(&[1.0, 2.0, 3.0]);
```

Based on Jacobs et al. (1991), Shazeer et al. (2017), Wang et al. (2024), Aspis et al. (2025).

## Quick Start

```sh
cargo add irithyll
```

### Factory Functions

The fastest way to get started -- one-liner construction for every algorithm:

```rust
use irithyll::{sgbt, rls, krls, gaussian_nb, mondrian, linear, StreamingLearner};

let mut trees  = sgbt(50, 0.01);        // 50 boosting steps, lr=0.01
let mut kernel = krls(1.0, 100, 1e-4);  // RBF gamma=1.0, budget=100
let mut bayes  = gaussian_nb();          // Gaussian Naive Bayes
let mut forest = mondrian(10);           // 10 Mondrian trees
let mut lin    = linear(0.01);           // SGD linear model, lr=0.01
let mut rls_m  = rls(0.99);             // RLS, forgetting factor=0.99

// All share the same interface
trees.train(&[1.0, 2.0], 3.0);
let pred = trees.predict(&[1.0, 2.0]);
```

### Neural Architectures

Reservoir computing, state space models, and spiking networks -- same `StreamingLearner` interface:

```rust
use irithyll::*;

// NG-RC: time-delay + polynomial features
let mut model = ngrc(2, 1, 2);
model.train(&[1.0], 2.0);
let pred = model.predict(&[1.0]);

// Echo State Network
let mut model = esn(100, 0.9);

// SSM as feature extractor -> gradient boosted trees
let mut model = pipe(mamba_preprocessor(5, 16)).learner(sgbt(50, 0.01));

// Spiking neural network
let mut model = spikenet(64);
model.train(&[0.5, -0.3], 1.0);
let pred = model.predict(&[0.5, -0.3]);
```

### Composable Pipelines

Chain preprocessors and learners with zero boilerplate:

```rust
use irithyll::{pipe, normalizer, sgbt, ccipca, StreamingLearner};

// Normalize → reduce to 5 components → gradient boosted trees
let mut model = pipe(normalizer())
    .pipe(ccipca(5))
    .learner(sgbt(50, 0.01));

model.train(&[100.0, 0.001, 50_000.0, 0.5, 1e-6, 42.0, 7.7, 0.3], 3.14);
let pred = model.predict(&[100.0, 0.001, 50_000.0, 0.5, 1e-6, 42.0, 7.7, 0.3]);
```

### Kernel Methods (KRLS)

Learn nonlinear functions with automatic dictionary sparsification:

```rust
use irithyll::{krls, StreamingLearner};

let mut model = krls(1.0, 100, 1e-4);  // RBF kernel, budget=100

for i in 0..500 {
    let x = i as f64 * 0.01;
    model.train(&[x], x.sin());
}

let pred = model.predict(&[1.5708]);  // sin(pi/2) ~ 1.0
```

### Prediction Intervals (RLS)

Get calibrated confidence intervals that narrow as data arrives:

```rust
use irithyll::{rls, StreamingLearner, RecursiveLeastSquares};

let mut model = rls(0.99);

for i in 0..1000 {
    let x = i as f64 * 0.01;
    model.train(&[x], 2.0 * x + 1.0);
}

let (pred, lo, hi) = model.predict_interval(&[5.0], 1.96);
// 95% CI: prediction is between lo and hi
```

### Full Builder Pattern

For complete control over SGBT hyperparameters:

```rust
use irithyll::{SGBTConfig, SGBT, Sample};

let config = SGBTConfig::builder()
    .n_steps(100)
    .learning_rate(0.0125)
    .max_depth(6)
    .n_bins(64)
    .lambda(1.0)
    .grace_period(200)
    .feature_names(vec!["price".into(), "volume".into()])
    .build()
    .expect("valid config");

let mut model = SGBT::new(config);

for i in 0..500 {
    let x = i as f64 * 0.01;
    model.train_one(&Sample::new(vec![x], 2.0 * x + 1.0));
}

// TreeSHAP explanations
let shap = model.explain(&[3.0]);
if let Some(named) = model.explain_named(&[3.0]) {
    for (name, value) in &named.values {
        println!("{}: {:.4}", name, value);
    }
}
```

### Concept Drift Detection

Automatic adaptation when the data distribution shifts:

```rust
use irithyll::{SGBTConfig, SGBT, Sample};
use irithyll::drift::adwin::Adwin;

let config = SGBTConfig::builder()
    .n_steps(50)
    .learning_rate(0.1)
    .drift_detector(Adwin::default())
    .build()
    .expect("valid config");

let mut model = SGBT::new(config);
// When drift is detected, trees are automatically replaced
```

### Async Streaming

Tokio-native with bounded channels and concurrent prediction:

```rust
use irithyll::{SGBTConfig, Sample};
use irithyll::stream::AsyncSGBT;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = SGBTConfig::builder()
        .n_steps(30)
        .learning_rate(0.1)
        .build()?;

    let mut runner = AsyncSGBT::new(config);
    let sender = runner.sender();
    let predictor = runner.predictor();

    let handle = tokio::spawn(async move { runner.run().await });

    for i in 0..500 {
        let x = i as f64 * 0.01;
        sender.send(Sample::new(vec![x], 2.0 * x)).await?;
    }

    let pred = predictor.predict(&[3.0]);
    drop(sender);
    handle.await??;
    Ok(())
}
```

### Python

```python
import numpy as np
from irithyll_python import StreamingGBTConfig, StreamingGBT

config = StreamingGBTConfig().n_steps(50).learning_rate(0.1)
model = StreamingGBT(config)

for i in range(500):
    x = np.array([i * 0.01])
    model.train_one(x, 2.0 * x[0] + 1.0)

pred = model.predict(np.array([3.0]))
shap = model.explain(np.array([3.0]))
```

## Packed Inference (`irithyll-core`)

Train with the full `irithyll` crate, export to a compact binary, and run inference on embedded targets with zero allocation.

### Node Format

Each `PackedNode` is 12 bytes (5 nodes per 64-byte cache line):

| Field | Size | Description |
|-------|------|-------------|
| `value` | 4B | Split threshold (internal) or prediction with learning rate baked in (leaf) |
| `children` | 4B | Packed left/right child u16 indices |
| `feature_flags` | 2B | Bit 15 = is_leaf, bits 14:0 = feature index |
| `_reserved` | 2B | Padding for future use |

### Export and Deploy

```rust
use irithyll::{SGBTConfig, SGBT, Sample};
use irithyll::export_embedded::export_packed;

// 1. Train on a host machine
let config = SGBTConfig::builder()
    .n_steps(50)
    .learning_rate(0.01)
    .max_depth(4)
    .build()
    .unwrap();
let mut model = SGBT::new(config);
for sample in training_data {
    model.train_one(&sample);
}

// 2. Export to packed binary (learning rate baked into leaf values)
let packed: Vec<u8> = export_packed(&model, n_features);
std::fs::write("model.bin", &packed).unwrap();
```

```rust
// 3. Load on embedded target (no_std, zero-alloc)
use irithyll_core::EnsembleView;

// Zero-copy: borrows the buffer, no heap allocation
let model_bytes: &[u8] = include_bytes!("model.bin");
let view = EnsembleView::from_bytes(model_bytes).unwrap();

let prediction: f32 = view.predict(&[1.0, 2.0, 3.0]);
```

### Performance

Benchmarked on x86-64 (single core, 50 trees, max depth 4, 10 features):

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Packed single predict (`irithyll-core`) | **66 ns** | 15.2M pred/s |
| Packed batch predict (x4 interleave) | -- | **5.3M pred/s** |
| Training-time predict (`irithyll` SoA) | 533 ns | 1.9M pred/s |

The 8x speedup comes from the 12-byte AoS node layout (vs training-time SoA vectors), branch-free child selection (compiles to `cmov` on x86), and f32 arithmetic with pre-baked learning rate.

## The SGBT Algorithm

The core gradient boosting engine is based on [Gunasekara et al., 2024](https://doi.org/10.1007/s10994-024-06517-y). The ensemble maintains `n_steps` boosting stages, each owning a streaming Hoeffding tree and a drift detector. For each sample *(x, y)*:

1. Compute the ensemble prediction *F(x) = base + lr * sum(tree_s(x))*
2. For each boosting step, compute gradient/hessian of the loss at the residual
3. Update the tree's histogram accumulators and evaluate splits via Hoeffding bound
4. Feed the standardized error to the drift detector
5. If drift is detected, replace the tree with a fresh alternate

Beyond the paper, irithyll adds EWMA leaf decay, lazy O(1) histogram decay, proactive tree replacement, and EFDT-style split re-evaluation for long-running non-stationary systems.

## Architecture

```
irithyll/                 Workspace root
  src/
    ensemble/             SGBT variants, config, multi-class/target, parallel, adaptive, distributional
    learners/             KRLS, RLS, Gaussian NB, Mondrian forests, linear/polynomial models
    pipeline/             Composable preprocessor + learner chains (StreamingPreprocessor trait)
    preprocessing/        IncrementalNormalizer, OnlineFeatureSelector, CCIPCA
    tree/                 Hoeffding-bound streaming decision trees
    histogram/            Streaming histogram binning (uniform, quantile, k-means)
    drift/                Concept drift detectors (Page-Hinkley, ADWIN, DDM)
    loss/                 Differentiable loss functions (squared, logistic, softmax, Huber)
    explain/              TreeSHAP, StreamingShap, importance drift monitoring
    stream/               Async tokio-based training runner and predictor handles
    metrics/              Online regression/classification metrics, conformal intervals, EWMA
    anomaly/              Half-space trees for streaming anomaly detection
    automl/               Champion-challenger racing, config space, model factories, reward normalization
    moe/                  Neural Mixture of Experts (polymorphic experts, top-k routing, load balancing)
    ttt/                  Test-Time Training layers (fast weights, self-supervised reconstruction, Titans)
    kan/                  Kolmogorov-Arnold Networks (B-spline edge activations, sparse SGD)
    reservoir/            NG-RC (time-delay polynomial) and ESN (cycle reservoir) + preprocessors
    ssm/                  StreamingMamba (selective SSM) + MambaPreprocessor
    snn/                  SpikeNet (f64 wrapper), SpikePreprocessor
    serde_support/        Model checkpoint/restore (JSON, bincode)
    export_embedded.rs    SGBT -> packed binary export for irithyll-core

  irithyll-core/          #![no_std] training engine + zero-alloc inference
    packed.rs             12-byte PackedNode, EnsembleHeader, TreeEntry
    traverse.rs           Branch-free tree traversal (single + x4 batch)
    view.rs               EnsembleView<'a> -- zero-copy inference from &[u8]
    quantize.rs           f64 -> f32 quantization utilities
    error.rs              FormatError (no_std compatible)
    reservoir/            NG-RC delay buffer, polynomial features, cycle reservoir, PRNG (alloc)
    ssm/                  Selective SSM: diagonal state, ZOH discretization, projections (alloc)
    snn/                  SpikeNetFixed: Q1.14 LIF neurons, e-prop, delta encoding (alloc)

  irithyll-python/        PyO3 Python bindings
```

## Configuration

| Parameter                | Default                    | Description                                   |
|--------------------------|----------------------------|-----------------------------------------------|
| `n_steps`                | 100                        | Number of boosting steps (trees in ensemble)   |
| `learning_rate`          | 0.0125                     | Shrinkage factor applied to each tree output   |
| `feature_subsample_rate` | 0.75                       | Fraction of features sampled per tree          |
| `max_depth`              | 6                          | Maximum depth of each streaming tree           |
| `n_bins`                 | 64                         | Number of histogram bins per feature           |
| `lambda`                 | 1.0                        | L2 regularization on leaf weights              |
| `gamma`                  | 0.0                        | Minimum gain required to make a split          |
| `grace_period`           | 200                        | Minimum samples before evaluating splits       |
| `delta`                  | 1e-7                       | Hoeffding bound confidence parameter           |
| `drift_detector`         | PageHinkley(0.005, 50.0)   | Drift detection algorithm for tree replacement |
| `variant`                | Standard                   | Computational variant (Standard, Skip, MI)     |
| `leaf_half_life`         | None (disabled)            | EWMA decay half-life for leaf statistics       |
| `max_tree_samples`       | None (disabled)            | Proactive tree replacement threshold           |
| `split_reeval_interval`  | None (disabled)            | Re-evaluation interval for max-depth leaves    |

## Feature Flags

These flags apply to the `irithyll` crate. `irithyll-core` has no required features (it is `no_std` with zero dependencies by default; an optional `std` feature is available).

| Feature | Default | Description |
|---------|---------|-------------|
| `serde-json` | Yes | JSON model serialization |
| `serde-bincode` | No | Compact binary serialization (bincode) |
| `parallel` | No | Rayon-based parallel tree training (`ParallelSGBT`) |
| `simd` | No | AVX2 histogram acceleration |
| `kmeans-binning` | No | K-means histogram binning strategy |
| `arrow` | No | Apache Arrow RecordBatch integration |
| `parquet` | No | Parquet file I/O |
| `onnx` | No | ONNX model export |
| `neural-leaves` | No | Experimental MLP leaf models |
| `full` | No | Enable all features |

Neural streaming modules (`reservoir`, `ssm`, `snn`) compile unconditionally -- no feature flag required.

## Examples

Run any example with `cargo run --example <name>`:

| Example | Description |
|---------|-------------|
| `basic_regression` | Linear regression with RMSE tracking |
| `classification` | Binary classification with logistic loss |
| `async_ingestion` | Tokio-native async training with concurrent prediction |
| `custom_loss` | Implementing a custom loss function |
| `drift_detection` | Abrupt concept drift with recovery analysis |
| `model_checkpointing` | Save/restore models with prediction verification |
| `streaming_metrics` | Prequential evaluation with windowed metrics |
| `krls_nonlinear` | Kernel regression on sin(x) with ALD sparsification |
| `ccipca_reduction` | Streaming PCA dimensionality reduction |
| `rls_confidence` | RLS prediction intervals narrowing over time |
| `pipeline_composition` | Normalizer + SGBT composable pipeline |

## Documentation

- [**API Reference**](https://docs.rs/irithyll) -- full docs on docs.rs (all features enabled)
- [**Rustdoc (GitHub Pages)**](https://evilrat420.github.io/irithyll/) -- latest from master
- [**Responsible Use**](RESPONSIBLE_USE.md) -- ethical use policy and research applications
- [**Contributing**](CONTRIBUTING.md) -- how to contribute, code standards, community values

## Minimum Supported Rust Version

The MSRV is **1.75**. This is checked in CI and will only be raised in minor version bumps.

## References

> Gunasekara, N., Pfahringer, B., Gomes, H. M., & Bifet, A. (2024). *Gradient boosted trees for evolving data streams.* Machine Learning, 113, 3325-3352.

> Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Banber, N., & Lee, S.-I. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2, 56-67.

> Weng, J., Zhang, Y., & Hwang, W.-S. (2003). *Candid covariance-free incremental principal component analysis.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 25(8), 1034-1040.

> Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021). *Next generation reservoir computing.* Nature Communications, 12, 5564.

> Rodan, A., & Tino, P. (2010). *Minimum complexity echo state network.* IEEE Transactions on Neural Networks, 23(1), 131-144.

> Gu, A., & Dao, T. (2023). *Mamba: Linear-time sequence modeling with selective state spaces.* arXiv preprint arXiv:2312.00752.

> Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). *A solution to the learning dilemma for recurrent networks of spiking neurons.* Nature Communications, 11, 3625.

> Neftci, E. O., Mostafa, H., & Zenke, F. (2019). *Surrogate gradient learning in spiking neural networks.* IEEE Signal Processing Magazine, 36(6), 51-63.

> Jaeger, H. (2001). *The "echo state" approach to analysing and training recurrent neural networks.* GMD Report 148.

> Lukoševičius, M., & Jaeger, H. (2009). *Reservoir computing approaches to recurrent neural network training.* Computer Science Review, 3(3), 127-149.

> Martinuzzi, F. (2025). *Minimal deterministic echo state networks outperform random reservoirs in learning chaotic dynamics.* Chaos, 35.

> Sussillo, D., & Abbott, L. F. (2009). *Generating coherent patterns of activity from chaotic neural networks.* Neuron, 63(4), 544-557.

> Yan, M., Huang, C., Bienstman, P., Tino, P., Lin, W., & Sun, J. (2024). *Emerging opportunities and challenges for the future of reservoir computing.* Nature Communications, 15, 2056.

> Gu, A., Goel, K., & Ré, C. (2021). *Efficiently modeling long sequences with structured state spaces.* arXiv preprint arXiv:2111.00396.

> Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). *On the parameterization and initialization of diagonal state space models.* NeurIPS 2022.

> Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality.* arXiv preprint arXiv:2405.21060.

> Liu, B., Wang, R., Wu, L., Feng, Y., Stone, P., & Liu, Q. (2024). *Longhorn: State space models are amortized online learners.* NeurIPS 2024.

> Kleyko, D., Frady, E. P., Kheffache, M., & Osipov, E. (2020). *Integer echo state networks: Efficient reservoir computing for digital hardware.* IEEE TNNLS.

> Zenke, F., & Ganguli, S. (2018). *SuperSpike: Supervised learning in multilayer spiking neural networks.* Neural Computation, 30(6), 1514-1541.

> Eshraghian, J. K., Ward, M., Neftci, E. O., et al. (2023). *Training spiking neural networks using lessons from deep learning.* Proceedings of the IEEE, 111(9), 1016-1054.

> Frenkel, C., & Indiveri, G. (2022). *ReckOn: A 28nm sub-mm² task-agnostic spiking recurrent neural network processor.* ISSCC 2022.

> Meyer, S. M., et al. (2024). *Diagonal state space model on Loihi 2 for efficient streaming.* arXiv preprint arXiv:2409.15022.

> Jaeger, H., Lukoševičius, M., Popovici, D., & Siewert, U. (2007). *Optimization and applications of echo state networks with leaky-integrator neurons.* Neural Networks, 20(3), 335-352.

> Javed, K., Shah, H., Sutton, R., & White, M. (2023). *Scalable real-time recurrent learning.* arXiv preprint arXiv:2302.05326.

> Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2023). *Gated linear attention transformers with hardware-efficient training.* arXiv preprint arXiv:2312.06635.

> Yang, S., et al. (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv preprint arXiv:2412.06464.

> Peng, B., et al. (2024). *Eagle and Finch: RWKV with matrix-valued states and dynamic recurrence.* arXiv preprint arXiv:2404.05892.

> Beck, M., et al. (2024). *xLSTM: Extended long short-term memory.* NeurIPS 2024.

> Sun, Y., et al. (2023). *Retentive network: A successor to transformer for large language models.* arXiv preprint arXiv:2307.08621.

> De, S., Smith, S. L., et al. (2024). *Griffin: Mixing gated linear recurrences with local attention for efficient language models.* arXiv preprint arXiv:2402.19427.

> Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS, 114(13), 3521-3526.

> Dohare, S., et al. (2024). *Loss of plasticity in deep continual learning.* Nature, 632, 768-774.

> Angelopoulos, A. N., Candes, E. J., & Tibshirani, R. J. (2023). *Conformal PID control for time series prediction.* NeurIPS 2023.

> Bhatnagar, A., Wang, H., Xiong, C., & Bai, Y. (2023). *Improved online conformal prediction via strongly adaptive online learning.* ICML 2023.

> Gupta, C., & Ramdas, A. (2023). *Online Platt scaling with calibeating.* ICML 2023.

> Wu, Q., Iyer, C., & Wang, C. (2021). *ChaCha for online AutoML.* ICML 2021.

> Qi, Y., et al. (2023). *Discounted Thompson Sampling for non-stationary bandits.* arXiv preprint arXiv:2305.10718.

> Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). *Adaptive mixtures of local experts.* Neural Computation, 3(1), 79-87.

> Shazeer, N., et al. (2017). *Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.* ICLR 2017.

> Wang, B., et al. (2024). *Auxiliary-loss-free load balancing strategy for mixture-of-experts.* arXiv preprint arXiv:2408.15664.

> Aspis, M., et al. (2025). *DriftMoE: Mixture of experts for streaming classification with concept drift.* ECMLPKDD 2025.

> Sun, Y., et al. (2024). *Learning to (Learn at Test Time): RNNs with expressive hidden states.* ICML 2025.

> Behrouz, A., Zhong, P., & Mirrokni, V. (2025). *Titans: Learning to memorize at test time.* arXiv preprint arXiv:2501.00663.

> Dohare, S., et al. (2024). *Loss of plasticity in deep continual learning.* Nature, 632, 768-774.

> Liu, Z., et al. (2024). *KAN: Kolmogorov-Arnold Networks.* ICLR 2025.

> Hoang, T. T., et al. (2026). *Ultrafast on-chip online learning via Kolmogorov-Arnold Networks.* arXiv preprint arXiv:2602.02056.

> Yamanishi, K. (2018). *Stochastic complexity for online learning with finite-state models.* IEEE TIT.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
