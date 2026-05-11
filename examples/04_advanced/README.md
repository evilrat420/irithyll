# 04 Advanced

Power-user patterns: hyperparameter search, kernel methods, dimensionality
reduction, custom loss functions, and model checkpointing. Assumes comfort
with the essentials tier.

## Examples (suggested order)

### 1. `automl`

One-liner AutoML with `auto_tune()`. Wraps tournament racing behind the
simplest possible API. Good entry point to understand what the racing
examples are doing under the hood.

```sh
cargo run --example automl
```

### 2. `factory_racing`

Champion-challenger tournament with explicit `Factory` and racing config.
Shows how to control the racing window, promotion threshold, and candidate
pool. Use this when `auto_tune()` defaults are not enough.

```sh
cargo run --example factory_racing
```

### 3. `custom_loss`

Implementing a custom `Loss` trait and plugging it into SGBT. Use when
asymmetric costs (e.g. false-negative penalty >> false-positive) matter.

```sh
cargo run --example custom_loss
```

### 4. `model_checkpointing`

Serialize a trained model to disk, restore it, and verify prediction
continuity. Covers the `Snapshot` API and serde integration.

```sh
cargo run --example model_checkpointing
```

### 5. `projected_learner`

Online dimensionality reduction with PAST-based `ProjectedLearner`. Wraps any
`StreamingLearner` in a projection that discards noise dimensions. Useful when
input dimensionality is high and most features carry no signal.

```sh
cargo run --example projected_learner
```

### 6. `ccipca_reduction`

Candid Covariance-free Incremental PCA for feature compression. Alternative to
ProjectedLearner when you want an explicit principal-component basis that
updates as the data distribution evolves.

```sh
cargo run --example ccipca_reduction
```

### 7. `krls_nonlinear`

Kernel Recursive Least Squares with RBF kernel. Learns non-linear functions
exactly (sin/cos) while maintaining a sparse dictionary via ALD. Use when the
target function is smooth and non-linear but feature dimensionality is low.

```sh
cargo run --example krls_nonlinear
```

## What you learn here

- AutoML: from one-liner to explicit racing configuration.
- Custom loss: the `Loss` trait contract.
- Checkpointing: serialize, restore, verify.
- Online dimensionality reduction: two complementary approaches.
- Kernel methods: exact non-linear function approximation with sparsification.
