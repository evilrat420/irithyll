# irithyll

[![Crates.io](https://img.shields.io/crates/v/irithyll.svg)](https://crates.io/crates/irithyll)
[![Documentation](https://docs.rs/irithyll/badge.svg)](https://docs.rs/irithyll)
[![CI](https://github.com/evilrat420/irithyll/actions/workflows/ci.yml/badge.svg)](https://github.com/evilrat420/irithyll/actions)
[![License](https://img.shields.io/crates/l/irithyll.svg)](https://github.com/evilrat420/irithyll)
[![MSRV](https://img.shields.io/badge/MSRV-1.75-blue.svg)](https://blog.rust-lang.org/2023/12/28/Rust-1.75.0.html)

**Streaming Gradient Boosted Trees for evolving data streams.**

Irithyll implements the SGBT algorithm ([Gunasekara et al., 2024](https://doi.org/10.1007/s10994-024-06517-y)) in pure Rust, providing incremental gradient boosted tree ensembles that learn one sample at a time. Trees use Hoeffding-bound split decisions and are automatically replaced when concept drift is detected, making the model suitable for non-stationary environments where the data distribution shifts over time.

Built for systems where data never stops — algorithmic trading, IoT telemetry, real-time anomaly detection — and extends the original paper with continuous adaptation mechanisms for production use.

## Features

### Core Algorithm
- **True online learning** -- train one sample at a time with `train_one()`, no batching required
- **Concept drift detection** -- automatic tree replacement via Page-Hinkley, ADWIN, or DDM detectors
- **Multi-class support** -- `MulticlassSGBT` with one-vs-rest committees and softmax normalization
- **Three SGBT variants** -- Standard, Skip (SGBT-SK), and MultipleIterations (SGBT-MI) per the paper
- **Pluggable loss functions** -- squared, logistic, softmax, Huber, or bring your own via the `Loss` trait
- **Hoeffding tree splitting** -- statistically-grounded split decisions with configurable confidence
- **XGBoost-style regularization** -- L2 (`lambda`) and minimum gain (`gamma`) on leaf weights

### Streaming Adaptation (beyond the paper)
- **EWMA leaf decay** -- exponential moving average on leaf statistics via `leaf_half_life`, enabling continuous adaptation without tree replacement
- **Lazy histogram decay** -- O(1) amortized forward decay per sample (not O(n_bins)), mathematically exact with automatic renormalization
- **Proactive tree replacement** -- time-based tree cycling via `max_tree_samples`, independent of drift detectors
- **Split re-evaluation** -- EFDT-inspired re-evaluation of max-depth leaves via `split_reeval_interval`

### Production Infrastructure
- **Async tokio-native streaming** -- `AsyncSGBT` with bounded channels, concurrent `Predictor` handles, and backpressure
- **Model checkpointing** -- `save_model()` / `load_model()` for JSON checkpoint/restore with backward-compatible deserialization
- **Online metrics** -- incremental MAE, MSE, RMSE, R-squared, accuracy, precision, recall, F1, and log loss with O(1) state
- **Feature importance** -- accumulated split gain per feature across the ensemble
- **Deterministic seeding** -- reproducible results via `SGBTConfig::seed`

### Optional Accelerators
- **Parallel training** (`parallel`) -- Rayon-based data-parallel tree training
- **SIMD histograms** (`simd`) -- AVX2 intrinsics for histogram gradient summation
- **Arrow integration** (`arrow`) -- train from `RecordBatch`, predict to arrays
- **Parquet I/O** (`parquet`) -- bulk training directly from Parquet files
- **ONNX export** (`onnx`) -- export trained models for cross-platform inference

## Quick Start

```sh
cargo add irithyll
```

### Regression

```rust
use irithyll::{SGBTConfig, SGBT, Sample};

fn main() {
    let config = SGBTConfig::builder()
        .n_steps(50)
        .learning_rate(0.1)
        .build()
        .expect("valid config");

    let mut model = SGBT::new(config);

    // Stream samples one at a time
    for i in 0..500 {
        let x = i as f64 * 0.01;
        let target = 2.0 * x + 1.0;
        model.train_one(&Sample::new(vec![x], target));
    }

    let prediction = model.predict(&[3.0]);
    println!("predict(3.0) = {:.4}", prediction);
}
```

### Binary Classification

```rust
use irithyll::{SGBTConfig, SGBT, Sample};
use irithyll::loss::logistic::LogisticLoss;

fn main() {
    let config = SGBTConfig::builder()
        .n_steps(30)
        .learning_rate(0.1)
        .build()
        .expect("valid config");

    let mut model = SGBT::with_loss(config, Box::new(LogisticLoss));

    // Class 0 near (-2, -2), class 1 near (2, 2)
    for _ in 0..500 {
        model.train_one(&Sample::new(vec![-2.0, -2.0], 0.0));
        model.train_one(&Sample::new(vec![ 2.0,  2.0], 1.0));
    }

    let prob = model.predict_proba(&[1.5, 1.5]);
    println!("P(class=1 | [1.5, 1.5]) = {:.4}", prob);
}
```

### Async Streaming

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

    // Spawn the training loop
    let handle = tokio::spawn(async move { runner.run().await });

    // Feed samples from any async context
    for i in 0..500 {
        let x = i as f64 * 0.01;
        sender.send(Sample::new(vec![x], 2.0 * x)).await?;
    }

    // Predict concurrently while training proceeds
    let pred = predictor.predict(&[3.0]);
    println!("prediction = {:.4}", pred);

    // Drop sender to signal shutdown
    drop(sender);
    handle.await??;
    Ok(())
}
```

### Streaming Adaptation

```rust
use irithyll::{SGBTConfig, SGBT, Sample};

fn main() {
    let config = SGBTConfig::builder()
        .n_steps(50)
        .learning_rate(0.1)
        // EWMA: leaf statistics half-life of 500 samples
        .leaf_half_life(500)
        // Replace trees after 10K samples regardless of drift
        .max_tree_samples(10_000)
        // Re-evaluate max-depth leaves every 1000 samples
        .split_reeval_interval(1000)
        .build()
        .expect("valid config");

    let mut model = SGBT::new(config);

    // The model now continuously adapts through three mechanisms:
    // 1. Leaf statistics decay exponentially (recent data weighted more)
    // 2. Trees are proactively replaced to prevent staleness
    // 3. Max-depth leaves re-evaluate whether splitting would help
    for i in 0..10_000 {
        let x = i as f64 * 0.001;
        let target = if i < 5000 { 2.0 * x } else { -x + 10.0 };
        model.train_one(&Sample::new(vec![x], target));
    }
}
```

## Architecture

```
irithyll/
  loss/          Differentiable loss functions (squared, logistic, softmax, huber)
  histogram/     Streaming histogram binning (uniform, quantile, optional k-means)
  tree/          Hoeffding-bound streaming decision trees
  drift/         Concept drift detectors (Page-Hinkley, ADWIN, DDM)
  ensemble/      SGBT boosting loop, config, variants, multi-class, parallel training
  stream/        Async tokio channel-based training runner and predictor handles
  metrics/       Online regression and classification metric trackers
  serde_support/ Model checkpoint/restore serialization
```

## Configuration

All hyperparameters are set via the builder pattern with validation on `build()`:

```rust
use irithyll::SGBTConfig;

let config = SGBTConfig::builder()
    .n_steps(100)              // Number of boosting steps (trees)
    .learning_rate(0.0125)     // Shrinkage factor
    .feature_subsample_rate(0.75) // Fraction of features per tree
    .max_depth(6)              // Maximum tree depth
    .n_bins(64)                // Histogram bins per feature
    .lambda(1.0)               // L2 regularization
    .gamma(0.0)                // Minimum split gain
    .grace_period(200)         // Samples before evaluating splits
    .delta(1e-7)               // Hoeffding bound confidence
    .build()
    .expect("valid config");
```

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

| Feature | Dependencies | Description |
|---------|-------------|-------------|
| `serde-json` (default) | `serde_json` | JSON model serialization |
| `serde-bincode` | `bincode` | Compact binary serialization |
| `parallel` | `rayon` | Parallel tree training |
| `simd` | -- | AVX2 histogram acceleration |
| `kmeans-binning` | -- | K-means histogram binning |
| `arrow` | `arrow` | Apache Arrow integration |
| `parquet` | `parquet` | Parquet file I/O |
| `onnx` | `prost` | ONNX model export |
| `neural-leaves` | -- | Experimental MLP leaf models |
| `full` | all above | Enable everything |

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

## Minimum Supported Rust Version

The MSRV is **1.75**. This is checked in CI and will only be raised in minor version bumps.

## References

> Gunasekara, N., Pfahringer, B., Gomes, H. M., & Bifet, A. (2024). *Gradient boosted trees for evolving data streams.* Machine Learning, 113, 3325--3352.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
