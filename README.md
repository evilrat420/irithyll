# irithyll

[![Crates.io](https://img.shields.io/crates/v/irithyll.svg)](https://crates.io/crates/irithyll)
[![Documentation](https://docs.rs/irithyll/badge.svg)](https://docs.rs/irithyll)
[![CI](https://github.com/evilrat420/irithyll/actions/workflows/ci.yml/badge.svg)](https://github.com/evilrat420/irithyll/actions)
[![License](https://img.shields.io/crates/l/irithyll.svg)](https://github.com/evilrat420/irithyll)
[![MSRV](https://img.shields.io/badge/MSRV-1.75-blue.svg)](https://blog.rust-lang.org/2023/12/28/Rust-1.75.0.html)

**Streaming Gradient Boosted Trees for evolving data streams.**

Irithyll is a pure Rust implementation of the SGBT algorithm ([Gunasekara et al., 2024](https://doi.org/10.1007/s10994-024-06517-y)). It learns one sample at a time. No batches, no windows, no retraining. Each tree in the ensemble uses Hoeffding-bound split decisions to grow incrementally, and when the data distribution shifts, concept drift detectors trigger automatic tree replacement so the model stays current.

The paper laid the foundation, but deploying streaming trees in long-running systems required going further. Irithyll adds EWMA leaf decay for continuous forgetting, lazy O(1) histogram decay (because decaying every bin on every sample doesn't scale), proactive tree replacement on a timer, and EFDT-style split re-evaluation at max-depth leaves. Together these close the gap between the research algorithm and a system you can run indefinitely on non-stationary data.

## Features

### Core Algorithm
- **True online learning** with `train_one()`, one sample at a time
- **Concept drift detection** via Page-Hinkley, ADWIN, or DDM, with automatic tree replacement
- **Multi-class support** through `MulticlassSGBT` with one-vs-rest committees
- **Multi-target regression** via `MultiTargetSGBT` with T independent models
- **Three SGBT variants** from the paper: Standard, Skip (SGBT-SK), and MultipleIterations (SGBT-MI)
- **Pluggable loss functions**: squared, logistic, softmax, Huber, or implement the `Loss` trait yourself
- **Hoeffding tree splitting** with configurable confidence bounds
- **XGBoost-style regularization**: L2 (`lambda`) and minimum gain (`gamma`)

### Interpretability
- **TreeSHAP explanations** via `explain()` with path-dependent SHAP values (Lundberg et al., 2020)
- **Named features** with `explain_named()` for human-readable per-feature contributions
- **StreamingShap** for online running-mean |SHAP| feature importance without storing past data
- **Feature importance** from accumulated split gain across the ensemble

### Streaming Adaptation
These go beyond the original paper to handle the realities of long-running, non-stationary systems:

- **EWMA leaf decay** (`leaf_half_life`): exponential moving average on leaf statistics so the model gradually forgets old data without needing to replace entire trees
- **Lazy histogram decay**: the decay math is O(1) per sample instead of O(n_bins), with exact results. The trick is storing samples in un-decayed coordinates and only materializing the decay when bins are actually read at split evaluation time
- **Proactive tree replacement** (`max_tree_samples`): cycle trees on a timer, independent of drift detectors. Useful when drift is gradual and detectors don't fire
- **Split re-evaluation** (`split_reeval_interval`): EFDT-inspired re-checking of max-depth leaves to see if splitting would now help

### Production Infrastructure
- **Async streaming** via `AsyncSGBT` with tokio channels, concurrent `Predictor` handles, and backpressure
- **Model checkpointing** with `save_model()` / `load_model()` — drift detector state is preserved across save/load
- **Online metrics**: incremental MAE, MSE, RMSE, R-squared, accuracy, precision, recall, F1, log loss
- **Deterministic seeding** for reproducible results
- **Python bindings** via the `irithyll-python` crate (PyO3 + numpy, GIL-released train/predict)

### Optional Accelerators
- **Parallel training** (`parallel`): Rayon-based data-parallel tree training
- **SIMD histograms** (`simd`): AVX2 intrinsics for histogram gradient summation
- **Arrow integration** (`arrow`): train from `RecordBatch`, predict to arrays
- **Parquet I/O** (`parquet`): bulk training directly from Parquet files
- **ONNX export** (`onnx`): export trained models for cross-platform inference

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

    let mut model = SGBT::with_loss(config, LogisticLoss);

    // Class 0 near (-2, -2), class 1 near (2, 2)
    for _ in 0..500 {
        model.train_one(&Sample::new(vec![-2.0, -2.0], 0.0));
        model.train_one(&Sample::new(vec![ 2.0,  2.0], 1.0));
    }

    let prob = model.predict_proba(&[1.5, 1.5]);
    println!("P(class=1 | [1.5, 1.5]) = {:.4}", prob);
}
```

### Explanations

```rust
use irithyll::{SGBTConfig, SGBT, Sample};

fn main() {
    let config = SGBTConfig::builder()
        .n_steps(20)
        .learning_rate(0.1)
        .feature_names(vec!["price".into(), "volume".into()])
        .build()
        .expect("valid config");

    let mut model = SGBT::new(config);

    for i in 0..500 {
        let price = i as f64 * 0.1;
        let volume = 100.0 - price;
        model.train_one(&Sample::new(vec![price, volume], price * 2.0));
    }

    // TreeSHAP: per-feature contributions
    let shap = model.explain(&[5.0, 50.0]);
    // Invariant: shap.base_value + sum(shap.values) == model.predict(&[5.0, 50.0])

    // Named explanations (sorted by |contribution|)
    if let Some(named) = model.explain_named(&[5.0, 50.0]) {
        for (name, value) in &named.values {
            println!("{}: {:.4}", name, value);
        }
    }
}
```

### Multi-Target Regression

```rust
use irithyll::{SGBTConfig, MultiTargetSGBT};

fn main() {
    let config = SGBTConfig::builder()
        .n_steps(20)
        .learning_rate(0.1)
        .build()
        .expect("valid config");

    let mut model = MultiTargetSGBT::new(config, 3).unwrap();

    for i in 0..500 {
        let x = i as f64 * 0.01;
        model.train_one(&[x, x * 2.0], &[x * 3.0, x * -1.0, x + 1.0]);
    }

    let preds = model.predict(&[1.0, 2.0]);
    assert_eq!(preds.len(), 3);
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

# SHAP explanations
shap = model.explain(np.array([3.0]))
print(shap.values, shap.base_value)

# Save/load
model.save("model.json")
restored = StreamingGBT.load("model.json")
```

## Architecture

```
irithyll/
  loss/          Differentiable loss functions (squared, logistic, softmax, huber)
  histogram/     Streaming histogram binning (uniform, quantile, optional k-means)
  tree/          Hoeffding-bound streaming decision trees
  drift/         Concept drift detectors (Page-Hinkley, ADWIN, DDM) with serializable state
  ensemble/      SGBT boosting loop, config, variants, multi-class, multi-target, parallel
  explain/       TreeSHAP explanations and StreamingShap online importance
  stream/        Async tokio channel-based training runner and predictor handles
  metrics/       Online regression and classification metric trackers
  serde_support/ Model checkpoint/restore serialization

irithyll-python/   PyO3 Python bindings (StreamingGBT, MultiTargetGBT, ShapExplanation)
```

## Configuration

All hyperparameters go through the builder pattern, validated on `build()`:

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
    .feature_names(vec!["price".into(), "volume".into()])
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
| `feature_names`          | None                       | Optional feature names for named explanations  |
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

> Gunasekara, N., Pfahringer, B., Gomes, H. M., & Bifet, A. (2024). *Gradient boosted trees for evolving data streams.* Machine Learning, 113, 3325-3352.

> Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Banber, N., & Lee, S.-I. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2, 56-67.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
