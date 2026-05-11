# 02 Essentials

The patterns you reach for in a real production streaming system. Each example
demonstrates one orthogonal concept cleanly.

## Examples (suggested order)

### 1. `streaming_metrics`

Prequential evaluation with windowed RMSE and R-squared. This is the standard
correctness protocol for streaming ML — always test before training. Read this
before drawing conclusions from any other example.

```sh
cargo run --example streaming_metrics
```

### 2. `pipeline_composition`

Chain a feature normalizer in front of an SGBT with `pipe()`. Shows why
preprocessing matters when feature scales differ by orders of magnitude.

```sh
cargo run --example pipeline_composition
```

### 3. `classification`

Binary classification with a logistic-loss SGBT. Covers `ClassificationMetrics`
(accuracy, F1) and the same prequential protocol applied to a discrete target.

```sh
cargo run --example classification
```

### 4. `rls_confidence`

Recursive Least Squares with online confidence intervals. Useful when you need
a point prediction AND an uncertainty band.

```sh
cargo run --example rls_confidence
```

### 5. `drift_detection`

Concept-drift detection with ADWIN. Shows how to integrate a drift detector
into the training loop and respond to detected shifts.

```sh
cargo run --example drift_detection
```

### 6. `async_ingestion`

Tokio-native async training API: `AsyncSGBT` runner, sender/predictor handles,
async task dispatch. Use this when samples arrive from an async I/O source
(network, message queue).

```sh
cargo run --example async_ingestion
```

## What you learn here

- Prequential evaluation as the correctness protocol.
- `pipe()` for composable preprocessing.
- `ClassificationMetrics` vs `RegressionMetrics`.
- Confidence intervals from RLS.
- Drift detection and response.
- The async streaming API.

## Where to go next

`03_neural/` for specialized neural architectures, or `04_advanced/` for
AutoML and kernel methods.
