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

**Streaming machine learning in Rust** -- gradient boosted trees, kernel methods, linear models, and composable pipelines, all learning one sample at a time. Train on a server, deploy to a microcontroller.

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

- **12+ streaming algorithms** under one unified [`StreamingLearner`](https://docs.rs/irithyll/latest/irithyll/trait.StreamingLearner.html) trait
- **One sample at a time** -- O(1) memory per model, no batches, no windows, no retraining
- **Embedded deployment** -- train with `irithyll`, export to packed binary, infer with `irithyll-core` on bare metal
- **Composable pipelines** -- chain preprocessors and learners with a builder API
- **Concept drift adaptation** -- automatic model replacement when the data distribution shifts
- **Confidence intervals** -- prediction uncertainty from RLS and conformal methods
- **Production-grade** -- async streaming, SIMD acceleration, Arrow/Parquet I/O, ONNX export
- **Pure Rust** -- zero unsafe in `irithyll`, deterministic, serializable, 1100+ tests

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
    serde_support/        Model checkpoint/restore (JSON, bincode)
    export_embedded.rs    SGBT -> packed binary export for irithyll-core

  irithyll-core/          #![no_std] zero-alloc inference engine
    packed.rs             12-byte PackedNode, EnsembleHeader, TreeEntry
    traverse.rs           Branch-free tree traversal (single + x4 batch)
    view.rs               EnsembleView<'a> -- zero-copy inference from &[u8]
    quantize.rs           f64 -> f32 quantization utilities
    error.rs              FormatError (no_std compatible)

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

## Minimum Supported Rust Version

The MSRV is **1.75**. This is checked in CI and will only be raised in minor version bumps.

## References

> Gunasekara, N., Pfahringer, B., Gomes, H. M., & Bifet, A. (2024). *Gradient boosted trees for evolving data streams.* Machine Learning, 113, 3325-3352.

> Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Banber, N., & Lee, S.-I. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2, 56-67.

> Weng, J., Zhang, Y., & Hwang, W.-S. (2003). *Candid covariance-free incremental principal component analysis.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 25(8), 1034-1040.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
