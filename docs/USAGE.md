# irithyll Usage Guide

Extended ergonomics, composition patterns, AutoML tournaments, streaming evaluation,
drift adaptation, embedded deployment, and Python bindings.

For a model-by-model reference — architectures, math summaries, config tables, and
benchmark notes — see [MODELS.md](../MODELS.md).

---

## Table of Contents

1. [Pipeline Composition](#pipeline-composition)
2. [AutoML Tournaments](#automl-tournaments)
3. [Streaming Metrics and Evaluation](#streaming-metrics-and-evaluation)
4. [Concept Drift Adaptation](#concept-drift-adaptation)
5. [Embedded Deployment (no_std)](#embedded-deployment-no_std)
6. [Python Bindings](#python-bindings)
7. [Common Patterns](#common-patterns)

---

## Pipeline Composition

A `Pipeline` chains one or more `StreamingPreprocessor` steps with a terminal
`StreamingLearner`. The resulting `Pipeline` itself implements `StreamingLearner`,
so it composes recursively — pipelines can be boxed, stacked in ensembles, or
passed to `AutoTuner` as-is.

The core contract: during `train`, each preprocessor calls `update_and_transform`
(updating its running statistics, then transforming). During `predict`, each
preprocessor calls `transform` only — statistics are frozen. This means you never
pollute the normalizer's mean/variance with test-time data.

### Basic builder walkthrough

```rust
use irithyll::{pipe, normalizer, sgbt, StreamingLearner};

let mut model = pipe(normalizer()).learner(sgbt(50, 0.01));

model.train(&[100.0, 0.001, 50_000.0], 42.0);
let pred = model.predict(&[100.0, 0.001, 50_000.0]);
```

`pipe(normalizer())` returns a `PipelineBuilder`. Calling `.learner(sgbt(50, 0.01))`
terminates the builder and returns the `Pipeline`. The free functions `pipe` and
`normalizer` are convenience wrappers — identical to writing
`Pipeline::builder().pipe(IncrementalNormalizer::new())`.

### Multi-preprocessor chains

Chain as many preprocessors as needed with repeated `.pipe(...)` calls. Each step
receives the output of the previous one:

```rust
use irithyll::{pipe, normalizer, ccipca, sgbt, StreamingLearner};

// Normalize -> reduce to 5 principal components -> gradient boosted trees
let mut model = pipe(normalizer())
    .pipe(ccipca(5))
    .learner(sgbt(50, 0.01));

// Features: 8-dimensional raw input
model.train(&[100.0, 0.001, 50_000.0, 0.5, 1e-6, 42.0, 7.7, 0.3], 3.14);
let pred = model.predict(&[100.0, 0.001, 50_000.0, 0.5, 1e-6, 42.0, 7.7, 0.3]);
```

`CCIPCA` is an O(kd) streaming PCA. It passes `k` principal components downstream
instead of the full `d`-dimensional input, compressing high-dimensional inputs
without storing a covariance matrix.

For the full runnable version see `examples/ccipca_reduction.rs`.

### Target preprocessing

When the regression target has a very different scale from the model's internal
representation (log-prices, sensor counts in the millions), add a target
preprocessor. It transforms the target before the learner sees it during training
and inverts the transformation on the raw prediction output — so `predict()` always
returns a value in the original target scale:

```rust
use irithyll::pipeline::PipelineBuilder;
use irithyll::preprocessing::{IncrementalNormalizer, TargetScaler};
use irithyll::{sgbt, StreamingLearner};

let mut model = PipelineBuilder::new()
    .pipe(IncrementalNormalizer::new())
    .target_preprocessor(TargetScaler::new())
    .learner(sgbt(50, 0.01));

model.train(&[1.0, 2.0], 1_000_000.0);
let pred = model.predict(&[1.0, 2.0]);
// pred is in the original scale (millions), not z-scored
```

`TargetScaler` is a Welford online z-scorer for the target. It wires into the
pipeline's `reset()` call, so resetting the pipeline resets the scaler too.

### Boxed learners and runtime polymorphism

`learner_boxed` accepts a `Box<dyn StreamingLearner>` when the concrete type is
not known at compile time:

```rust
use irithyll::pipeline::PipelineBuilder;
use irithyll::preprocessing::IncrementalNormalizer;
use irithyll::{esn, StreamingLearner};

let learner: Box<dyn StreamingLearner> = Box::new(esn(100, 0.9));
let mut model = PipelineBuilder::new()
    .pipe(IncrementalNormalizer::new())
    .learner_boxed(learner);

model.train(&[0.3, -0.1], 1.7);
```

The `make_pipeline!` macro offers a terser multi-stage form:
`make_pipeline!(PreprocessorA::new() => PreprocessorB::new() => Learner::new())`.

For a full pipeline demonstration see `examples/pipeline_composition.rs`.

---

## AutoML Tournaments

`AutoTuner` runs successive halving tournaments across one or more `Factory`
instances. A champion always provides predictions; challengers compete in
the background. The tournament cycle is:

1. Sample `n_initial` candidates (half warm-started from champion config,
   half random from bandit-guided factory selection).
2. Every `round_budget` samples, eliminate the bottom half by adjusted score.
3. Promote the finalist to champion if its Bernstein-verified error is better.
4. Start a new tournament immediately.

Complexity-adjusted elimination means simpler models are preferred on sparse data
and must prove themselves with evidence. Warmup protection prevents neural
architectures from being killed before their transients settle.

### One-liner: auto-tune a single factory

```rust
use irithyll::{auto_tune, automl::Factory, StreamingLearner};

let mut tuner = auto_tune(Factory::sgbt(5)); // 5 input features

for i in 0..1000 {
    let x = [i as f64 * 0.01, (i as f64 * 0.1).sin(), 1.0, 0.5, 0.2];
    let y = x[0] * 2.0 + x[1] * 3.0 + 1.0;
    tuner.train(&x, y);
}

println!("Promotions: {}", tuner.promotions());
println!("Tournaments: {}", tuner.tournaments_completed());
let pred = tuner.predict(&[5.0, 0.0, 1.0, 0.5, 0.2]);
```

`auto_tune(factory)` is a shorthand for `AutoTuner::builder().factory(factory).build()`.

### Multi-factory: race trees vs neural architectures

```rust
use irithyll::automl::{AutoTuner, Factory};
use irithyll::StreamingLearner;

let mut tuner = AutoTuner::builder()
    .factory(Factory::sgbt(5))
    .add_factory(Factory::esn())
    .add_factory(Factory::kan(5))
    .n_initial(6)         // 6 candidates per tournament
    .round_budget(100)    // eliminate every 100 samples
    .build()
    .expect("valid AutoTuner config");

for i in 0..3000_usize {
    let x: Vec<f64> = (0..5).map(|j| (i + j) as f64 * 0.01).collect();
    let y = x[0] * 2.0 + x[2].sin();
    tuner.train(&x, y);
}

println!("Factory names: {:?}", tuner.factory_names());
let pred = tuner.predict(&[1.0, 2.0, 3.0, 4.0, 5.0]);
```

The factory-level bandit (Discounted Thompson Sampling) tracks which factory type
has historically won tournaments and biases candidate sampling toward it.

### Drift re-racing

When the champion's error drifts (detected via ADWIN), the tournament is aborted and
restarted with an expanded bracket. Enable it with `.use_drift_rerace(true)`:

```rust
use irithyll::automl::{AutoTuner, Factory};

let mut tuner = AutoTuner::builder()
    .factory(Factory::sgbt(3))
    .add_factory(Factory::mamba(3))
    .use_drift_rerace(true)
    .build()
    .expect("valid config");
```

After a drift event, `tuner.promotions()` increments and the new champion is
chosen from a fresh tournament rather than from the old one's state.

For a full multi-factory race see `examples/factory_racing.rs`. For a minimal
single-factory example see `examples/automl.rs`.

### Constructors on `Factory`

`Factory` covers all model families. Key constructors:

| Constructor | Description |
|---|---|
| `Factory::sgbt(n_features)` | SGBT with SPSA auto-builder |
| `Factory::esn()` | Echo State Network |
| `Factory::mamba(n_features)` | Streaming Mamba (selective SSM) |
| `Factory::kan(n_features)` | Streaming KAN (B-spline activations) |
| `Factory::ttt()` | Test-Time Training fast weights |
| `Factory::projected_mamba(rank, n_features)` | Mamba wrapped in online projection |

All `Factory` instances implement `ModelFactory`. To race any model family, pass
its `Factory` to `AutoTuner::builder()`.

---

## Streaming Metrics and Evaluation

### `StreamingMetric` trait

Every metric type implements `StreamingMetric`: update in O(1), get current value,
reset. Metrics compose via `+` into a `MetricUnion`:

```rust
use irithyll::metrics::streaming_metric::{MAE, RMSE, StreamingMetric};

let mut m = MAE::new() + RMSE::new();

m.update(3.2, 3.0);   // (pred, actual)
m.update(2.8, 3.0);

println!("MAE:  {:.4}", m.a.get());
println!("RMSE: {:.4}", m.b.get());

m.reset(); // reset both
```

Available metric types: `MAE`, `MSE`, `RMSE`, `R2`, `Pinball<TAU>`, `LogLoss`,
`Accuracy`. All are O(1) space — no past samples stored.

### Prequential (test-then-train) evaluation

The canonical protocol for streaming models is prequential evaluation: predict
first, update metrics with the error, then train. This gives unbiased generalization
estimates because the model never trains before predicting each sample:

```rust
use irithyll::{RegressionMetrics, SGBTConfig, Sample, SGBT};

let config = SGBTConfig::builder()
    .n_steps(50)
    .learning_rate(0.1)
    .grace_period(20)
    .build()
    .expect("valid config");

let mut model = SGBT::new(config);
let mut metrics = RegressionMetrics::new();

for i in 0..2000_usize {
    let x = i as f64 * 0.01;
    let y = 2.0 * x + 1.0;

    // Predict BEFORE training (prequential)
    let pred = model.predict(&[x]);
    metrics.update(y, pred);

    model.train_one(&Sample::new(vec![x], y));
}

println!("RMSE: {:.4}", metrics.rmse());
println!("MAE:  {:.4}", metrics.mae());
println!("R2:   {:.4}", metrics.r_squared());
```

### Progressive windowed validation

Reset the metrics accumulator at regular intervals to track how error evolves
through the stream — a spike indicates concept drift, a downward trend confirms
the model is still learning:

```rust
use irithyll::{RegressionMetrics, sgbt, StreamingLearner};

let mut model = sgbt(50, 0.1);
let mut window = RegressionMetrics::new();
let window_size = 500;

for i in 0..3000_usize {
    let x = (i as f64) * 0.01;
    let y = 2.0 * x + 1.0;

    let pred = model.predict(&[x]);
    window.update(y, pred);
    model.train(&[x], y);

    if (i + 1) % window_size == 0 {
        println!("Window RMSE at {}: {:.4}", i + 1, window.rmse());
        window.reset(); // start a fresh window
    }
}
```

### Classification evaluation

`ClassificationMetrics` covers accuracy, precision, recall, F1, and log-loss.
Update with `(true_class, predicted_class, predicted_probability)`:

```rust
use irithyll::ClassificationMetrics;

let mut cls = ClassificationMetrics::new();
// cls.update(true_label, predicted_label, predicted_probability)
cls.update(1, 1, 0.85);
cls.update(0, 1, 0.60);

println!("Acc:    {:.4}", cls.accuracy());
println!("F1:     {:.4}", cls.f1());
println!("LogLoss:{:.4}", cls.log_loss());
```

For a full demonstration of windowed regression and classification metrics see
`examples/streaming_metrics.rs`.

---

## Concept Drift Adaptation

### Drift detectors

Drift detectors monitor a stream of scalar error values (typically per-sample
squared error or absolute error) and signal when the distribution has shifted.
Three detectors are available:

| Type | Description |
|---|---|
| `adwin::Adwin` | Adaptive windowing — statistically principled, recommended default |
| `pht::PageHinkley` | Page-Hinkley test — low-latency cumulative sum detector |
| `ddm::Ddm` | Drift Detection Method — for classification error streams |

Wire a detector into `SGBTConfig` to enable automatic tree replacement:

```rust
use irithyll::{SGBTConfig, SGBT};
use irithyll::drift::adwin::Adwin;

let config = SGBTConfig::builder()
    .n_steps(50)
    .learning_rate(0.1)
    .drift_detector(Adwin::default())
    .build()
    .expect("valid config");

let mut model = SGBT::new(config);
// When Adwin signals drift on a tree's error stream, that tree is
// replaced with a fresh Hoeffding tree in the next sample.
```

Each of the `n_steps` boosting stages owns its own detector instance. When the
standardized error on a stage crosses the detection threshold, that stage's tree
is replaced in-place. The ensemble continues predicting throughout — there is no
reset, no full retrain, no pause.

### Adaptive forest pattern

For more aggressive drift response, combine drift detection with adaptive
hyperparameters. The `AdaptiveSGBT` uses exponential learning-rate scheduling
(linear ramp-up, cosine decay, hard restart on drift):

```rust
use irithyll::{AdaptiveSGBT, SGBTConfig, Sample};

let config = SGBTConfig::builder()
    .n_steps(50)
    .learning_rate(0.05)
    .build()
    .expect("valid config");

let mut model = AdaptiveSGBT::new(config);

// Adapts lr schedule automatically; drift triggers a restart cycle
for i in 0..2000_usize {
    let x = (i as f64) * 0.01;
    let y = if i < 1000 { 2.0 * x } else { -3.0 * x + 5.0 };
    model.train_one(&Sample::new(vec![x], y));
}
```

### Principled SGBT adaptation flags

Beyond the drift detector, three flags on `SGBTConfig` control ensemble-level
adaptation behavior. They are independent and compose:

| Flag | Effect |
|---|---|
| `honest_sigma` | Instant epistemic uncertainty from ensemble disagreement. Zero hyperparameters. |
| `adaptive_mts` | Sigma-modulated tree replacement speed. High uncertainty -> faster cycling. |
| `proactive_prune` | Percentile-based worst-tree replacement. Maintains plasticity. |

```rust
use irithyll::{SGBTConfig, SGBT};

let config = SGBTConfig::builder()
    .n_steps(100)
    .learning_rate(0.01)
    .honest_sigma(true)
    .adaptive_mts(true)
    .proactive_prune(true)
    .build()
    .expect("valid config");
let mut model = SGBT::new(config);
```

For a full drift detection scenario with pre/post-drift RMSE analysis see
`examples/drift_detection.rs`.

---

## Embedded Deployment (no_std)

irithyll's deployment story is a two-crate split:

- **`irithyll`** (std) — full training, streaming algorithms, pipelines, I/O.
- **`irithyll-core`** (`no_std`, zero-alloc) — packed inference engine. Zero
  dependencies. Cross-compiles for Cortex-M0+, M3, M4 and any bare-metal target
  with a stable Rust toolchain.

The workflow is: train on a host machine with `irithyll`, export to a compact
binary, ship the binary to the embedded target, and infer with `irithyll-core`.

### Step 1: Train and export (host, `irithyll`)

```rust
use irithyll::{SGBTConfig, SGBT, Sample};
use irithyll::export_embedded::export_packed;

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

// Export: learning rate is baked into leaf values during packing
let packed: Vec<u8> = export_packed(&model, n_features);
std::fs::write("model.bin", &packed).unwrap();
```

`export_packed` produces 12-byte-per-node AoS layout (f32). Each node packs a
split threshold or a leaf prediction, two child indices, and a feature index into
12 bytes — 5 nodes fit in a single 64-byte cache line.

### Step 2: Validate on host before shipping

```rust
use irithyll::export_embedded::validate_export;

let max_error = validate_export(&model, &packed, &test_samples);
// f64->f32 quantization: expect max_error < 1e-4 for well-trained models
assert!(max_error < 0.1, "export error too large: {}", max_error);
```

### Step 3: Infer on embedded target (`irithyll-core`, no_std)

```rust
#![no_std]
use irithyll_core::EnsembleView;

// Zero-copy: borrows the flash-resident buffer, no heap allocation
static MODEL_BYTES: &[u8] = include_bytes!("model.bin");

let view = EnsembleView::from_bytes(MODEL_BYTES).unwrap();
let prediction: f32 = view.predict(&[1.0_f32, 2.0, 3.0]);
```

`EnsembleView::from_bytes` validates the entire binary (magic, version, bounds,
feature indices) on construction. After that, `predict` is pure pointer arithmetic
with `get_unchecked` — no allocation, no dynamic dispatch, no branches beyond the
tree traversal itself.

Verified cross-compilation targets: `thumbv6m-none-eabi` (Cortex-M0+),
`thumbv7m-none-eabi` (Cortex-M3), `thumbv7em-none-eabi` (Cortex-M4 soft-float).

For QEMU-based testing of embedded inference, see the `irithyll-core` test suite
under `tests/qemu_*` which runs predictions through `xpack` QEMU emulation.

### Integer-only export for FPU-less targets

For Cortex-M0+ targets without an FPU, use `export_packed_i16`, which quantizes
all thresholds and leaf values to i16. The inference hot loop becomes pure integer
arithmetic:

```rust
use irithyll::export_embedded::export_packed_i16;
use irithyll_core::QuantizedEnsembleView;

// Host: export with per-feature quantization scales
let packed_i16 = export_packed_i16(&model, n_features);
std::fs::write("model_i16.bin", &packed_i16).unwrap();

// Target: zero-alloc integer-only inference
let view = QuantizedEnsembleView::from_bytes(MODEL_I16_BYTES).unwrap();
let pred: f32 = view.predict(&[1.0_f32, 2.0, 3.0]);
```

Quantization error is typically < 0.5 for well-trained models. Use
`validate_export_i16` on the host before shipping to confirm.

`SpikeNetFixed` is a separate `no_std` model in `irithyll-core` for truly
event-driven targets: Q1.14 LIF neurons, e-prop learning, 64 neurons in 22KB SRAM.
It fits the Cortex-M0+ 32KB SRAM constraint with room to spare.

---

## Python Bindings

The `irithyll-python` crate provides PyO3 bindings for `StreamingGBT` and
`MultiTargetGBT`. Build with `maturin develop --release` from `irithyll-python/`.

```python
import numpy as np
from irithyll_python import StreamingGBTConfig, StreamingGBT

# Builder-style config: each setter returns self for chaining
config = StreamingGBTConfig().n_steps(50).learning_rate(0.1).max_depth(4)
model = StreamingGBT(config)

for i in range(500):
    x = np.array([i * 0.01, (i * 0.1) % 1.0])
    y = 2.0 * x[0] + 3.0 * x[1] + 0.1
    model.train_one(x, y)

pred = model.predict(np.array([3.0, 0.5]))
shap = model.explain(np.array([3.0, 0.5]))   # TreeSHAP values
```

`train_one` accepts a 1D numpy array for features and a scalar target. `predict`
and `explain` accept 1D numpy arrays. All operations are synchronous — no GIL
release, no async. The model state lives in Rust; Python holds a reference via
PyO3's ownership model.

For save/restore, the Python bindings expose `to_json()` and `from_json()` methods
that round-trip through irithyll's JSON checkpoint format.

---

## Common Patterns

### Reset and re-train (epoch pattern)

`reset()` returns a model to its initial untrained state while preserving the
configuration. Use it to re-train from scratch on a new epoch without
constructing a new model:

```rust
use irithyll::{sgbt, StreamingLearner};

let mut model = sgbt(50, 0.01);

// Epoch 1
for &(x, y) in &epoch1_data {
    model.train(&[x], y);
}

model.reset(); // wipes all learned state, keeps config

// Epoch 2 — starts cold
for &(x, y) in &epoch2_data {
    model.train(&[x], y);
}
```

For `Pipeline`, `reset()` propagates through all preprocessors and the terminal
learner — the normalizer's running mean/variance is also cleared.

### Save and load checkpoints

Use `save_model` / `load_model` from `irithyll::serde_support` for JSON
checkpointing. The restored model continues training from exactly where it left
off (config, tree structure, leaf statistics, drift detector state):

```rust
use irithyll::serde_support::{save_model, load_model};
use irithyll::{SGBT, SGBTConfig, Sample};

// Save
let json = save_model(&model).expect("serialization failed");
std::fs::write("checkpoint.json", &json).unwrap();

// Restore and continue training
let json = std::fs::read_to_string("checkpoint.json").unwrap();
let mut restored = load_model(&json).expect("deserialization failed");
restored.train_one(&Sample::new(vec![1.0, 2.0], 3.0));
```

For compact binary checkpoints (smaller file, faster I/O), use
`save_model_bincode` / `load_model_bincode` (requires the `serde-bincode`
feature):

```toml
# Cargo.toml
irithyll = { version = "...", features = ["serde-bincode"] }
```

For a full save/restore roundtrip with prediction verification see
`examples/model_checkpointing.rs`.

### ONNX export

Export a trained SGBT to an ONNX `TreeEnsembleRegressor` for inference in
Python (onnxruntime), C++, JavaScript, or any ONNX-compatible runtime. Requires
the `onnx` feature:

```toml
irithyll = { version = "...", features = ["onnx"] }
```

```rust
use irithyll::onnx_export::save_onnx;
use std::path::Path;

save_onnx(&model, n_features, Path::new("model.onnx"))
    .expect("ONNX export failed");
```

The exported model is a standard ONNX `TreeEnsembleRegressor` operator. Load
it in Python with `onnxruntime.InferenceSession("model.onnx")`.

### Drift-aware retraining

The most robust production pattern for long-running streams: prequential
evaluation detects performance degradation, then a controlled reset re-trains
on a recent sliding buffer. Pair this with `AutoTuner` if the optimal model
family is unknown:

```rust
use irithyll::{RegressionMetrics, sgbt, StreamingLearner};

let mut model = sgbt(50, 0.01);
let mut metrics = RegressionMetrics::new();
let mut recent_buffer: std::collections::VecDeque<([f64; 3], f64)> =
    std::collections::VecDeque::with_capacity(500);

for (x, y) in live_stream {
    // Prequential evaluation
    let pred = model.predict(&x);
    metrics.update(y, pred);

    model.train(&x, y);
    recent_buffer.push_back((x, y));
    if recent_buffer.len() > 500 {
        recent_buffer.pop_front();
    }

    // If last 200-sample window RMSE exceeds threshold, reset and retrain
    if metrics.n_samples() % 200 == 0 && metrics.rmse() > acceptable_rmse {
        model.reset();
        for &(bx, by) in &recent_buffer {
            model.train(&bx, by);
        }
        metrics.reset();
    }
}
```

For the stateless alternative — where the model adapts continuously without
manual resets — configure `honest_sigma`, `adaptive_mts`, and a drift detector
directly on `SGBTConfig` as described in the [Concept Drift Adaptation](#concept-drift-adaptation)
section.
