# 03 Neural

Specialized neural streaming architectures. Each implements the same
`StreamingLearner` interface as SGBT — the training loop is identical, the
internal dynamics differ significantly.

All neural models use an online RLS readout layer. The recurrent cell updates
on every sample; no batches, no backward passes.

## Examples (suggested order)

### 1. `kan_regression`

Kolmogorov-Arnold Network with fixed spline activations. Best starting point
for neural examples — the simplest cell, clear diagnostic output.

```sh
cargo run --example kan_regression
```

### 2. `streaming_kan`

KAN with online basis adaptation. Shows the adaptive variant where spline
knots move with the data distribution.

```sh
cargo run --example streaming_kan
```

### 3. `mamba3_temporal`

Mamba-3 selective state-space model. Excels on temporal sequences with
long-range dependencies. Shows the gating behavior on a regime-shift signal.

```sh
cargo run --example mamba3_temporal
```

### 4. `slstm_regression`

sLSTM (scalar LSTM) with exponential gating. Lower parameter count than
standard LSTM, competitive on smooth nonlinear targets.

```sh
cargo run --example slstm_regression
```

### 5. `streaming_ttt`

Test-Time Training (TTT) attention layer. Updates its attention weights
on each incoming token — genuinely adaptive self-attention in a streaming
setting.

```sh
cargo run --example streaming_ttt
```

### 6. `rwkv7_attention`

RWKV-7 linear attention. Constant memory, constant compute per token.
Use when sequence length is unbounded and you need strict throughput bounds.

```sh
cargo run --example rwkv7_attention
```

### 7. `neural_moe`

Mixture-of-Experts gating over multiple streaming neural cells. Demonstrates
routing, per-expert load stats, and the gating dynamics over time.

```sh
cargo run --example neural_moe
```

## What you learn here

- How the recurrent cell + RLS readout pattern unifies all neural models.
- When to prefer each architecture (temporal structure, regime shift, throughput).
- How to read the diagnostic output each model exposes.

## Where to go next

`04_advanced/` for AutoML, kernel methods, projection, and custom loss.
