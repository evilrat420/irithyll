# irithyll

Streaming Gradient Boosted Trees for evolving data streams.

Irithyll implements the SGBT algorithm (Gunasekara et al., 2024) in pure Rust, providing
incremental gradient boosted tree ensembles that learn one sample at a time. Trees are
built using Hoeffding-bound split decisions and automatically replaced when concept drift
is detected, making the model suitable for non-stationary environments where the
underlying data distribution changes over time.

## Features

- **True online learning** -- train one sample at a time with `train_one()`, no batching required
- **Concept drift detection** -- automatic tree replacement via Page-Hinkley, ADWIN, or DDM detectors
- **Async tokio-native streaming** -- `AsyncSGBT` with bounded channels, concurrent `Predictor` handles, and backpressure
- **Pluggable loss functions** -- squared (regression), logistic (binary classification), softmax (multi-class), Huber (robust regression), or bring your own via the `Loss` trait
- **Three SGBT variants** -- Standard, Skip (SGBT-SK), and MultipleIterations (SGBT-MI) for accuracy/compute trade-offs
- **Multi-class support** -- `MulticlassSGBT` with one-vs-rest committees and softmax normalization
- **Online metrics** -- incremental MAE, MSE, RMSE, R-squared, accuracy, precision, recall, F1, and log loss with O(1) state
- **Hoeffding tree splitting** -- statistically-grounded split decisions using histogram-based gradient aggregation
- **Feature subsampling** -- per-tree random feature selection for diversity and reduced compute
- **Serialization** -- `serde` support with optional JSON and bincode backends

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

## Algorithm

Irithyll implements **Streaming Gradient Boosted Trees** as described by Gunasekara et al.
(2024). The ensemble maintains a sequence of boosting steps, each owning a streaming
decision tree and a drift detector. For each incoming sample `(x, y)`, the ensemble
computes `F(x) = base + lr * sum(tree_s(x))`, then sequentially updates each tree by
computing the gradient and hessian of the loss at the current prediction, fitting the
tree on the pseudo-residuals. Trees use Hoeffding-bound confidence intervals over
histogram-binned gradient statistics to decide when and where to split, enabling
statistically sound decisions without storing past data.

When a drift detector signals that the error distribution of a boosting step has changed,
the corresponding tree is replaced with a fresh one, allowing the ensemble to adapt to
concept drift without full retraining. Three computational variants are supported:
**Standard** (every tree processes every instance), **Skip / SGBT-SK** (randomly skips
1/k of instances per tree for reduced cost), and **MultipleIterations / SGBT-MI** (trains
each tree `ceil(|hessian| * multiplier)` times per instance, emphasizing hard-to-fit
regions).

> Gunasekara, N., Pfahringer, B., Gomes, H. M., & Bifet, A. (2024). *Streaming
> Gradient Boosted Trees.*

## Architecture

```
irithyll/
  loss/          Differentiable loss functions (squared, logistic, softmax, huber)
  histogram/     Streaming histogram binning (uniform, quantile, optional k-means)
  tree/          Hoeffding-bound streaming decision trees
  drift/         Concept drift detectors (Page-Hinkley, ADWIN, DDM)
  ensemble/      SGBT boosting loop, config, variants, multi-class committees
  stream/        Async tokio channel-based training runner and predictor handles
  metrics/       Online regression and classification metric trackers
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

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
