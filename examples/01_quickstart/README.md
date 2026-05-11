# 01 Quickstart

Zero to prediction in under 5 minutes. No setup, no data files, no tuning.

These examples are deliberately minimal. Every line is load-bearing; nothing
is included to show off the API.

## Examples (suggested order)

### 1. `hello_streaming`

The absolute minimum irithyll program: create a model, stream samples, predict.
~20 lines. Read this first.

```sh
cargo run --example hello_streaming
```

### 2. `basic_regression`

The same idea with more depth: SGBTConfig, RegressionMetrics, prequential
evaluation, and a final test-point table. Read this when hello_streaming makes
sense and you want to see the full training loop.

```sh
cargo run --example basic_regression
```

## What you learn here

- The `sgbt()` shorthand vs `SGBTConfig::builder()` for explicit configuration.
- The `train()` / `predict()` call pattern on `StreamingLearner`.
- Prequential (test-then-train) evaluation — predict first, then update.

## Where to go next

`02_essentials/` — pipeline composition, drift detection, classification,
async ingestion.
