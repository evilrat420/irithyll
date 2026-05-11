# Model Specifications

This document is the canonical reference for every learning algorithm, neural architecture, and
preprocessing primitive in irithyll. Each section answers the same four questions: what is it,
when should you use it, what math drives it, and how do you configure and run it.

**How to read this document.** Start with the tier table to locate the model class, then jump
to the relevant section. The [README](README.md) gives a high-level overview and quick-start
snippets. [REFERENCES.md](REFERENCES.md) contains the full bibliography; this document links
to specific anchors there. [BENCHMARKS.md](BENCHMARKS.md) has quantitative comparisons across
datasets.

Every model implements [`StreamingLearner`](https://docs.rs/irithyll/latest/irithyll/trait.StreamingLearner.html)
unless noted otherwise. Train and predict with `model.train(features, target)` /
`model.predict(features)`. Preprocessors implement `StreamingPreprocessor` and slot into
pipelines via `pipe(preprocessor).learner(model)`.

---

## Tier Summary

| Tier | Count | Models |
|------|-------|--------|
| [Production](#production-tier) | 10 | SGBT family (5 variants), RLS, KRLS, StreamingLinearModel, MondrianForest, LocallyWeightedRegression |
| [Neural — Reservoir](#neural-tier--reservoir-computing) | 2 | NextGenRC, EchoStateNetwork |
| [Neural — State Space Models](#neural-tier--state-space-models) | 3 | StreamingMamba, StreamingMambaV3, BDLRUMamba |
| [Neural — Spiking](#neural-tier--spiking-neural-networks) | 2 | SpikeNet, SpikeNetFixed |
| [Neural — TTT](#neural-tier--test-time-training) | 1 | StreamingTTT |
| [Neural — KAN](#neural-tier--kolmogorov-arnold-networks) | 2 | StreamingKAN, TKAN |
| [Neural — Online Extensions](#neural-tier--online-learning-extensions) | 4 | StreamingAGMP, StreamingMGrade, StreamingHGRN2, StreamingLSTM |
| [Neural — Attention](#neural-tier--streaming-linear-attention) | 11 modes | GLA, GatedDeltaNet, DeltaNet, DeltaProduct, RWKV-7, HGRN2, Hawk, RetNet, mLSTM, RWKV, Log-Linear |
| [Ensemble](#ensemble-tier) | 5 | NeuralMoE, AdaptiveRandomForest, MulticlassSGBT, MultiTargetSGBT, DistributionalSGBT |
| [Specialized](#specialized-tier) | 5 | GaussianNB, HalfSpaceTree, ContinualLearner, ClassificationWrapper, ProjectedLearner |
| [Preprocessing](#preprocessing) | 9 | IncrementalNormalizer, CCIPCA, OnlineFeatureSelector, TargetEncoder, TargetScaler, TargetLog1pTransform, FeatureHasher, OneHotEncoder, PolynomialExpander |
| [Clustering](#clustering) | 3 | StreamingKMeans, CluStream, DBStream |
| [AutoML](#streaming-automl) | 2 | AutoTuner, Factory |

---

## Production Tier

The production tier contains the algorithms that have been most extensively validated across
diverse streaming benchmarks. Use these by default; reach into the neural tier when temporal
structure is the primary signal.

---

### SGBT

[`SGBT`](https://docs.rs/irithyll/latest/irithyll/struct.SGBT.html) — Streaming Gradient Boosted Trees

The core algorithm of irithyll. An online ensemble of Hoeffding-bound streaming decision trees
arranged in boosting stages. Each stage owns a tree and a drift detector; drift replaces the
tree with a fresh alternate. This is the default choice for tabular streaming regression and
binary classification with concept drift.

**Paper / origin.** Gunasekara et al. (2024) — see [REFERENCES.md §Core Gradient Boosting](REFERENCES.md).

**Math summary.**

```
F(x) = base + lr * sum_{s=1}^{n_steps} tree_s(x)
h_s  = hessian of loss at residual (sample-weight)
tree_s updates histogram accumulators for its split candidates
Hoeffding bound: split when gain_best - gain_second >= sqrt(R^2 * ln(1/delta) / (2n))
```

Leaf predictions use EWMA decay (half-life configurable) and graduated sibling interpolation
for continuous output between discrete split points.

**Config reference.** [`SGBTConfig::builder()`](https://docs.rs/irithyll/latest/irithyll/struct.SGBTConfigBuilder.html)

Key parameters:
- `n_steps(usize)` — number of boosting stages (default: 100)
- `learning_rate(f64)` — shrinkage applied to each tree (default: 0.0125)
- `max_depth(usize)` — max depth per tree (default: 6)
- `grace_period(usize)` — min samples before split evaluation (default: 200)
- `drift_detector(DriftDetectorType)` — `PageHinkley`, `ADWIN`, or `DDM` (default: `PageHinkley`)
- `proactive_prune` — percentile-based worst-tree replacement for plasticity

**Example.**
```rust
use irithyll::{sgbt, StreamingLearner};

let mut model = sgbt(50, 0.01);  // 50 boosting steps, lr=0.01
model.train(&[1.0, 2.0], 3.0);
let pred = model.predict(&[1.0, 2.0]);

// Full builder for SHAP + drift config
use irithyll::{SGBTConfig, SGBT};
let config = SGBTConfig::builder()
    .n_steps(100)
    .learning_rate(0.0125)
    .max_depth(6)
    .grace_period(200)
    .build()
    .expect("valid config");
let mut model = SGBT::new(config);
let shap = model.explain(&[1.0, 2.0]);
```

**Benchmarks.** See [BENCHMARKS.md](BENCHMARKS.md) — Electricity (88.5% accuracy), Airlines, Covertype.

---

### AdaptiveSGBT

[`AdaptiveSGBT`](https://docs.rs/irithyll/latest/irithyll/struct.AdaptiveSGBT.html) — SGBT with online learning rate scheduling

Extends SGBT with configurable LR schedules (step decay, exponential, cosine) applied across
the stream. Use when the learning rate should decay or cycle over time rather than remain constant.

**Paper / origin.** Built on Gunasekara et al. (2024). LR schedule design draws from standard
SGD scheduling literature.

**Math summary.** Learning rate at sample t follows the selected schedule:
```
step:        lr_t = lr_0 * factor^floor(t / step_size)
exponential: lr_t = lr_0 * exp(-decay * t)
cosine:      lr_t = lr_min + 0.5*(lr_0 - lr_min)*(1 + cos(pi * t / T_max))
```

**Config reference.** [`LRSchedule`](https://docs.rs/irithyll/latest/irithyll/enum.LRSchedule.html) + `SGBTConfig`

Key parameters:
- `schedule(LRSchedule)` — `Step`, `Exponential`, `Cosine`, or `Constant`
- All base SGBT parameters apply

**Example.**
```rust
use irithyll::ensemble::AdaptiveSGBT;
use irithyll::ensemble::lr_schedule::LRSchedule;

let mut model = AdaptiveSGBT::new(base_config, LRSchedule::Cosine { lr_min: 0.001, t_max: 10_000 });
model.train(&[1.0, 2.0], 3.0);
```

**Benchmarks.** <!-- TODO: spec — no standalone benchmark row in BENCHMARKS.md yet. -->

---

### DistributionalSGBT

[`DistributionalSGBT`](https://docs.rs/irithyll/latest/irithyll/struct.DistributionalSGBT.html) — mean + variance SGBT for prediction uncertainty

Trains two parallel SGBT heads — one for the mean, one for the variance — producing a
Gaussian prediction `(mu, sigma)` per sample. Also exposes `honest_sigma`, the tree
contribution variance across ensemble stages, as a zero-hyperparameter epistemic uncertainty
signal. Use when you need calibrated prediction intervals or downstream uncertainty weighting.

**Paper / origin.** Gunasekara et al. (2024) for the core. `honest_sigma` and `adaptive_mts`
are irithyll extensions.

**Math summary.**
```
mu    = SGBT_mean(x)
sigma = SGBT_var(x)   (trains on squared residuals from mu head)
honest_sigma = std( { tree_s(x) : s = 1..n_steps } )
               — instant epistemic signal, one sample reaction time
adaptive_mts: tree_lifetime = base_mts * (1 - sigma_ratio * honest_sigma / sigma_threshold)
```

**Config reference.** Same as `SGBTConfig` plus:
- `honest_sigma()` method — zero-config epistemic signal
- `adaptive_mts(sigma_ratio)` — modulates tree lifetime with uncertainty

**Example.**
```rust
use irithyll::ensemble::DistributionalSGBT;
use irithyll::{SGBTConfig, StreamingLearner};

let config = SGBTConfig::builder().n_steps(50).build().unwrap();
let mut model = DistributionalSGBT::new(config);
model.train(&[1.0, 2.0], 3.0);
let (mu, sigma) = model.predict_distributional(&[1.0, 2.0]);
let epistemic = model.honest_sigma(&[1.0, 2.0]);
```

**Benchmarks.** Refer to [BENCHMARKS.md](BENCHMARKS.md) uncertainty evaluation rows.

---

### MulticlassSGBT

[`MulticlassSGBT`](https://docs.rs/irithyll/latest/irithyll/struct.MulticlassSGBT.html) — one-vs-rest SGBT for multi-class classification

Maintains one SGBT per class, each trained with softmax-normalized pseudo-residuals. At
prediction time, the class with the highest raw score wins. Per-sample cost scales linearly
with class count, so prefer this for up to ~20 classes; beyond that consider `NeuralMoE`
with classification wrappers.

**Paper / origin.** One-vs-rest strategy is classical; Gunasekara et al. (2024) for the
underlying SGBT. Softmax normalization follows Friedman (2001) multi-class gradient boosting.

**Math summary.**
```
p_k(x) = softmax( { F_k(x) : k = 1..K } )_k
residual_k = 1[y == k] - p_k(x)
Each F_k is an independent SGBT trained on its residuals.
Prediction: argmax_k F_k(x)
```

**Config reference.** `SGBTConfig` + number of classes at construction.

**Example.**
```rust
use irithyll::ensemble::MulticlassSGBT;
use irithyll::SGBTConfig;

let config = SGBTConfig::builder().n_steps(50).build().unwrap();
let mut model = MulticlassSGBT::new(config, 3);
model.train(&[1.0, 2.0], 0.0);  // class label as f64
let pred = model.predict(&[1.0, 2.0]);  // returns predicted class as f64
let proba = model.predict_proba(&[1.0, 2.0]);
```

**Benchmarks.** Covertype (7-class), see [BENCHMARKS.md](BENCHMARKS.md).

---

### MultiTargetSGBT

[`MultiTargetSGBT`](https://docs.rs/irithyll/latest/irithyll/struct.MultiTargetSGBT.html) — independent SGBTs for multi-output regression

One SGBT per output target, trained independently. Use when multiple continuous targets
share the same input features. Outputs are not modeled jointly; for correlated targets
consider a custom loss.

**Paper / origin.** Gunasekara et al. (2024). Independent multi-output extension.

**Math summary.**
```
F_j(x) = SGBT_j(x),  j = 1..n_targets
Per-sample cost: n_targets * O(n_steps * depth)
```

**Config reference.** `SGBTConfig` + `n_targets` at construction.

**Example.**
```rust
use irithyll::ensemble::MultiTargetSGBT;
use irithyll::SGBTConfig;

let config = SGBTConfig::builder().n_steps(30).build().unwrap();
let mut model = MultiTargetSGBT::new(config, 2);
model.train(&[1.0, 2.0], &[3.0, 4.0]);
let preds: Vec<f64> = model.predict_multi(&[1.0, 2.0]);
```

**Benchmarks.** <!-- TODO: spec — no dedicated multi-target benchmark row in BENCHMARKS.md yet. -->

---

### RecursiveLeastSquares (RLS)

[`RecursiveLeastSquares`](https://docs.rs/irithyll/latest/irithyll/struct.RecursiveLeastSquares.html) — exact online linear regression with confidence intervals

Exact online OLS via Sherman-Morrison rank-1 covariance update. Supports exponential
forgetting (`forgetting_factor < 1`), adaptive forgetting driven by error magnitude, and
calibrated prediction intervals. This is the readout layer inside every neural model in
irithyll; it is also a first-class StreamingLearner for linear regression tasks.

**Paper / origin.** Classical adaptive filtering; see Haykin (1996) *Adaptive Filter Theory*.
Adaptive forgetting is an irithyll extension.

**Math summary.**
```
Gain:      k_t  = P_{t-1} x_t / (lambda + x_t^T P_{t-1} x_t)
Weights:   w_t  = w_{t-1} + k_t * (y_t - x_t^T w_{t-1})
Covariance:P_t  = (P_{t-1} - k_t x_t^T P_{t-1}) / lambda
Interval:  CI   = w^T x +/- z * sqrt(sigma_n^2 * x^T P x)
```

**Config reference.** `rls(forgetting_factor)` factory or `RecursiveLeastSquares::new(ff)`

Key parameters:
- `forgetting_factor` — lambda in [0, 1]; lower = faster adaptation (default: 0.998)
- `delta` — initial P matrix diagonal (default: 100.0)

**Example.**
```rust
use irithyll::{rls, StreamingLearner, RecursiveLeastSquares};

let mut model = rls(0.99);
for i in 0..1000 {
    let x = i as f64 * 0.01;
    model.train(&[x], 2.0 * x + 1.0);
}
let (pred, lo, hi) = model.predict_interval(&[5.0], 1.96);
```

**Benchmarks.** Used as baseline in all neural model comparisons. See [BENCHMARKS.md](BENCHMARKS.md).

---

### KRLS

[`KRLS`](https://docs.rs/irithyll/latest/irithyll/struct.KRLS.html) — Kernel Recursive Least Squares

Nonlinear RLS in a kernel-induced feature space. Maintains a sparse dictionary of support
vectors via Approximate Linear Dependence (ALD) — new samples are added to the dictionary
only if they cannot be approximately represented by the existing set. Use for nonlinear
regression (sin, exp, interactions) where SGBT is not the right fit.

**Paper / origin.** Engel, Mannor & Meir (2004), "The Kernel Recursive Least Squares
Algorithm." IEEE TSP. — [citation needed for REFERENCES.md anchor]

**Math summary.**
```
ALD test: ||x - K_t^{-1} K_{x,D_t}||^2 > threshold  ->  add to dictionary
Prediction:  f(x) = sum_{i in D} alpha_i * k(x_i, x)
Update:      alpha += k_t * (y - f(x)),  K^{-1} rank-1 updated
|D| <= budget  (oldest entry evicted when full)
```

**Config reference.** `krls(gamma, budget, ald_threshold)` factory

Key parameters:
- `gamma` — RBF kernel bandwidth (default: 1.0)
- `budget` — max dictionary size; controls O(budget^2) cost (default: 100)
- `ald_threshold` — ALD novelty threshold for dictionary admission (default: 1e-4)
- Kernel: `RBFKernel`, `PolynomialKernel`, or `LinearKernel` via `KRLS::new(kernel, budget, thr)`

**Example.**
```rust
use irithyll::{krls, StreamingLearner};

let mut model = krls(1.0, 100, 1e-4);
for i in 0..500 {
    let x = i as f64 * 0.01;
    model.train(&[x], x.sin());
}
let pred = model.predict(&[1.5708]);  // ~1.0
```

**Benchmarks.** See [BENCHMARKS.md](BENCHMARKS.md) KRLS rows on nonlinear regression benchmarks.

---

### StreamingLinearModel

[`StreamingLinearModel`](https://docs.rs/irithyll/latest/irithyll/struct.StreamingLinearModel.html) — SGD linear model (ridge, lasso, elastic net)

Stochastic gradient descent linear regression. Fastest possible per-sample cost: O(d).
Supports L1, L2, and elastic net regularization via factory variants. Use as a fast baseline
before trying tree or neural models.

**Paper / origin.** Classical online SGD; no single paper. L1/L2 regularization from
Tibshirani (1996) / Hoerl & Kennard (1970).

**Math summary.**
```
w_{t+1} = w_t - lr * grad_loss(w_t, x_t, y_t)
           - lr * lambda_l2 * w_t           (ridge)
           - lr * lambda_l1 * sign(w_t)     (lasso)
grad_squared_loss = -2 * (y_t - w_t^T x_t) * x_t
```

**Config reference.** `linear(lr)` / `StreamingLinearModel::ridge(lr, lambda)` /
`StreamingLinearModel::lasso(lr, lambda)` / `StreamingLinearModel::elastic_net(lr, l1, l2)`

Key parameters:
- `learning_rate` — SGD step size (default: 0.01)
- `lambda` — regularization strength (context-dependent default)

**Example.**
```rust
use irithyll::{linear, StreamingLearner};

let mut model = linear(0.01);
model.train(&[1.0, 2.0], 3.0);
let pred = model.predict(&[1.0, 2.0]);
```

**Benchmarks.** Fastest baseline. See [BENCHMARKS.md](BENCHMARKS.md) throughput section.

---

### MondrianForest

[`MondrianForest`](https://docs.rs/irithyll/latest/irithyll/struct.MondrianForest.html) — streaming random forest with Mondrian process splits

An ensemble of Mondrian trees: each tree samples axis-aligned cuts from the Mondrian process,
giving theoretical guarantees on online learning rates. Unlike Hoeffding trees, Mondrian
trees can refine existing splits, not just add new ones. Use for streaming ensemble regression
where feature space structure is unknown.

**Paper / origin.** Lakshminarayanan, Roy & Teh (2014), "Mondrian Forests: Efficient Online
Random Forests." NIPS 2014. — [citation needed for REFERENCES.md anchor]

**Math summary.**
```
Mondrian process: cut time E(x, y) = Expo( sum_j (max_j - min_j) )
Leaf prediction:  weighted mean of samples at leaf
Tree lifetime lambda: controls expected tree depth (lower = shallower)
```

**Config reference.** `mondrian(n_trees)` factory or `MondrianForestConfig::builder()`

Key parameters:
- `n_trees` — ensemble size (default: 10)
- `max_depth` — cap on tree depth
- `lifetime(f64)` — Mondrian process parameter (controls cut rate)
- `seed(u64)` — RNG seed for reproducibility

**Example.**
```rust
use irithyll::{mondrian, StreamingLearner};

let mut model = mondrian(10);
model.train(&[1.0, 2.0], 3.0);
let pred = model.predict(&[1.0, 2.0]);
```

**Benchmarks.** See [BENCHMARKS.md](BENCHMARKS.md) Mondrian rows.

---

### LocallyWeightedRegression

[`LocallyWeightedRegression`](https://docs.rs/irithyll/latest/irithyll/struct.LocallyWeightedRegression.html) — memory-based locally adaptive regression

Stores a sliding window of recent samples. At prediction time, fits a weighted least squares
model where weights decay exponentially with Euclidean distance from the query. Adapts
automatically to locally varying linear relationships. Cost is O(window) per prediction.

**Paper / origin.** Locally Weighted Learning — Atkeson, Moore & Schaal (1997). — [citation needed for REFERENCES.md anchor]

**Math summary.**
```
w_i = exp(-||x_query - x_i||^2 / (2 * bandwidth^2))
pred = (X^T W X)^{-1} X^T W y   (weighted OLS on window)
```

**Config reference.** `LocallyWeightedRegression::new(capacity, bandwidth)`

Key parameters:
- `capacity` — sliding window size
- `bandwidth` — Gaussian kernel bandwidth for distance weighting

**Example.**
```rust
use irithyll::learners::LocallyWeightedRegression;
use irithyll::StreamingLearner;

let mut model = LocallyWeightedRegression::new(200, 1.0);
for i in 0..300 {
    let x = i as f64 * 0.05;
    model.train(&[x], x.sin());
}
let pred = model.predict(&[1.5]);
```

**Benchmarks.** Suited for locally smooth nonlinear functions. No dedicated BENCHMARKS.md row yet. <!-- TODO: spec -->

---

## Neural Tier — Reservoir Computing

Reservoir computing maps inputs through a fixed (or structured) nonlinear dynamical system
and trains only the linear readout. No backpropagation; converges in under 10 samples in the
NG-RC case.

---

### NextGenRC (NG-RC)

[`NextGenRC`](https://docs.rs/irithyll/latest/irithyll/reservoir/struct.NextGenRC.html) — time-delay + polynomial features + RLS readout

No random matrices. Input is embedded via a delay buffer (k lags, stride s), then lifted
through degree-d polynomial features, then fit by RLS readout. Designed for time series
forecasting where the signal has low-dimensional attractor dynamics.

**Paper / origin.** Gauthier, Bollt, Griffith & Barbosa (2021), "Next generation reservoir
computing." *Nature Communications*, 12, 5564. → [REFERENCES.md §Reservoir Computing](REFERENCES.md).

**Math summary.**
```
Delay embedding:  z_t = [x_t, x_{t-s}, x_{t-2s}, ..., x_{t-(k-1)s}]
Polynomial lift:  phi(z) = {z_i^a * z_j^b : a+b <= degree}
Readout:          y_t = W * phi(z_t),  W trained via RLS
```

**Config reference.** `ngrc(k, s, degree)` factory or `NGRCConfig::builder()`

Key parameters:
- `k` — number of time-delay taps
- `s` — stride between taps
- `degree` — polynomial degree (2–5)
- `forgetting_factor` — RLS lambda (default: 0.998)

**Example.**
```rust
use irithyll::{ngrc, StreamingLearner};

let mut model = ngrc(2, 1, 2);  // 2 taps, stride 1, degree 2
model.train(&[1.0], 2.0);
let pred = model.predict(&[1.0]);
```

**Benchmarks.** Strong on Mackey-Glass and Lorenz. See [BENCHMARKS.md](BENCHMARKS.md) neural-specific rows.

---

### EchoStateNetwork (ESN)

[`EchoStateNetwork`](https://docs.rs/irithyll/latest/irithyll/reservoir/struct.EchoStateNetwork.html) — cycle/ring reservoir + leaky integration + RLS readout

A deterministic ESN using the cycle/ring reservoir topology (Rodan & Tino 2010): reservoir
weights form a single ring, giving O(N) weight storage vs O(N^2) for random reservoirs. State
evolves via leaky integration. For reservoirs larger than 200 units, JL random projection
reduces the RLS readout to a manageable dimension.

**Paper / origin.** Rodan & Tino (2010) for the cycle topology; Martinuzzi (2025) for
deterministic ESN superiority over random reservoirs. → [REFERENCES.md §Reservoir Computing](REFERENCES.md).

**Math summary.**
```
h_t = (1 - alpha) * h_{t-1} + alpha * tanh(W_res * h_{t-1} + W_in * x_t)
      where W_res is a single cycle (h_i -> h_{(i+1) mod N})
      spectral_radius scales W_res so rho(W_res) = spectral_radius
readout: y_t = W_out * h_t,  W_out trained by RLS
JL projection: if N > 200, project h_t to 64-dim before RLS
```

**Config reference.** `esn(n_reservoir, spectral_radius)` factory or `ESNConfig::builder()`

Key parameters:
- `n_reservoir` — reservoir size (default: 100)
- `spectral_radius` — echo state property control (default: 0.9)
- `leaky_rate` — alpha in leaky integration (default: 1.0 = no leak)
- `forgetting_factor` — RLS readout lambda

**Example.**
```rust
use irithyll::{esn, StreamingLearner};

let mut model = esn(100, 0.9);
for i in 0..500 {
    let x = i as f64 * 0.01;
    model.train(&[x], x.sin());
}
let pred = model.predict(&[1.0]);
```

**Benchmarks.** See [BENCHMARKS.md](BENCHMARKS.md) NARMA-10 and Mackey-Glass rows.

---

## Neural Tier — State Space Models

SSMs model sequence dynamics via a latent state `h` governed by structured linear recurrence.
All irithyll SSMs are discretized for sample-by-sample streaming via zero-order hold.

---

### StreamingMamba

[`StreamingMamba`](https://docs.rs/irithyll/latest/irithyll/ssm/struct.StreamingMamba.html) — selective SSM (Mamba V1) with input-dependent parameters

The Mamba architecture selectively focuses state updates based on the current input: B, C,
and Delta are functions of x rather than fixed matrices. S4D-Inv A-matrix initialization
ensures a bounded spectrum across all state sizes. Per-channel L2-norm state energy feeds
the RLS readout, giving a feature vector of dimension d_in * n_state.

**Paper / origin.** Gu & Dao (2023), "Mamba: Linear-time sequence modeling with selective
state spaces." arXiv:2312.00752; Gu et al. (2022) for S4D-Inv initialization. →
[REFERENCES.md §State Space Models](REFERENCES.md).

**Math summary.**
```
ZOH discretization: A_bar = exp(Delta * A),  B_bar = (A_bar - I) * A^{-1} * B
State update:        h_t  = A_bar * h_{t-1} + B_bar(x_t) * x_t
Output:              y_t  = C(x_t) * h_t + D * x_t
Readout features:    f_j  = ||h_t[j]||_2  (per-channel energy, bounded)
```

**Config reference.** `mamba(d_in, n_state)` factory or `MambaConfig::builder()`

Key parameters:
- `d_in` — input/output dimension
- `n_state` — SSM state size per channel (default: 32)
- `forgetting_factor` — RLS readout lambda
- `skip_connection` — add x_t directly to output (default: enabled)

**Example.**
```rust
use irithyll::{mamba, StreamingLearner};

let mut model = mamba(8, 16);
model.train(&[0.5; 8], 1.0);
let pred = model.predict(&[0.5; 8]);

// As preprocessor: SSM features -> SGBT
use irithyll::{pipe, mamba_preprocessor, sgbt};
let mut pipeline = pipe(mamba_preprocessor(5, 16)).learner(sgbt(50, 0.01));
```

**Benchmarks.** See [BENCHMARKS.md](BENCHMARKS.md) Mackey-Glass rows; larger n_state improves after readout fix.

---

### StreamingMambaV3

[`StreamingMambaV3`](https://docs.rs/irithyll/latest/irithyll/ssm/struct.StreamingMambaV3.html) — SSM duality (Mamba2) architecture

Implements the Mamba2 / structured state space duality view (Dao & Gu 2024): the recurrence
is rewritten as a semiseparable matrix multiplication, and the state update becomes a
rank-1 outer product per head. More scalable than V1 for large d_model.

**Paper / origin.** Dao & Gu (2024), "Transformers are SSMs: Generalized models and
efficient algorithms through structured state space duality." arXiv:2405.21060. →
[REFERENCES.md §State Space Models](REFERENCES.md).

**Math summary.**
```
State update:  S_t = Lambda * S_{t-1} + v_t k_t^T   (rank-1 outer product)
Query:         o_t = S_t * q_t
Lambda:        diagonal, learned per channel (structured state)
```

**Config reference.** `MambaV3Config::builder()` (no dedicated factory function; use `StreamingMambaV3::new(config)`).

Key parameters:
- `d_model` — model dimension
- `n_heads` — number of SSM heads (block-diagonal structure)
- `forgetting_factor` — RLS readout lambda

**Example.**
```rust
use irithyll::ssm::{StreamingMambaV3, MambaV3Config};
use irithyll::StreamingLearner;

let config = MambaV3Config::builder().d_model(16).n_heads(4).build().unwrap();
let mut model = StreamingMambaV3::new(config);
model.train(&[0.5; 16], 1.0);
```

**Benchmarks.** <!-- TODO: spec — no standalone V3 benchmark row yet. -->

---

### BDLRUMamba

[`BDLRUMamba`](https://docs.rs/irithyll/latest/irithyll/ssm/struct.BDLRUMamba.html) — block-diagonal LRU state matrix

A hardware-efficient SSM variant using a block-diagonal state transition matrix where each
block is a 2×2 rotation (LRU: Linear Recurrent Unit). The block structure provides selective
per-block forgetting and is well-suited to deployment targets where matrix operations are
parallelized at the block level.

**Paper / origin.** LRU: Orvieto et al. (2023) "Resurrecting Recurrent Neural Networks for
Long Sequences." ICML 2023. — [citation needed for REFERENCES.md anchor]

**Math summary.**
```
h_t^{(b)} = A^{(b)} h_{t-1}^{(b)} + B^{(b)} x_t
A^{(b)}: 2x2 rotation matrix (block of the block-diagonal A)
Diagonal magnitude: |lambda^{(b)}| in (0, 1) controls forgetting per block
```

**Config reference.** `BDLRUConfig::builder()`

Key parameters:
- `d_model` — total state dimension
- `n_blocks` — number of 2×2 blocks in the diagonal
- `forgetting_factor` — RLS readout lambda

**Example.**
```rust
use irithyll::ssm::{BDLRUMamba, BDLRUConfig};
use irithyll::StreamingLearner;

let config = BDLRUConfig::builder().d_model(16).n_blocks(4).build().unwrap();
let mut model = BDLRUMamba::new(config);
model.train(&[0.5; 16], 1.0);
```

**Benchmarks.** <!-- TODO: spec -->

---

## Neural Tier — Spiking Neural Networks

Event-driven computation with biologically inspired LIF (Leaky Integrate-and-Fire) neurons.
E-prop replaces backpropagation through time with an online local learning rule.

---

### SpikeNet

[`SpikeNet`](https://docs.rs/irithyll/latest/irithyll/snn/struct.SpikeNet.html) — LIF neurons with e-prop online learning

A recurrent SNN where synaptic weights are updated by e-prop: a biologically plausible
approximation to BPTT that uses only locally available signals. Delta spike encoding converts
the continuous input to spike events. Surrogate gradients (piecewise linear) handle
the non-differentiable threshold. The f64 interface wraps the fixed-point core.

**Paper / origin.** Bellec et al. (2020) for e-prop; Neftci et al. (2019) for surrogate
gradients. → [REFERENCES.md §Spiking Neural Networks](REFERENCES.md).

**Math summary.**
```
Membrane: U_t^i = beta * U_{t-1}^i + sum_j W_ji * z_t^j - z_t^i * U_thr
Spike:    z_t^i = 1 if U_t^i > U_thr, else 0
Delta enc: spike when |x_t - x_{t-1}| > threshold (sparse events)
e-prop:   delta_W_ji = e_t^ji * L_t  (eligibility trace × learning signal, no BPTT)
```

**Config reference.** `spikenet(n_hidden)` factory or `SpikeNetConfig::builder()`

Key parameters:
- `n_hidden` — number of LIF neurons
- `beta` — membrane time constant (default: 0.9)
- `threshold` — spike threshold (default: 1.0)
- `learning_rate` — e-prop step size

**Example.**
```rust
use irithyll::{spikenet, StreamingLearner};

let mut model = spikenet(64);
model.train(&[0.5, -0.3], 1.0);
let pred = model.predict(&[0.5, -0.3]);
```

**Benchmarks.** See [BENCHMARKS.md](BENCHMARKS.md) sparse temporal event dataset rows.

---

### SpikeNetFixed (`no_std`)

[`SpikeNetFixed`](https://docs.rs/irithyll/latest/irithyll/irithyll_core/snn/struct.SpikeNetFixed.html) — Q1.14 integer arithmetic throughout, `no_std`

Full training in fixed-point arithmetic — no floats, no heap beyond `alloc`. 64 hidden
neurons fit in 22KB on a Cortex-M0+ with 32KB SRAM. This is the embedded-first SNN:
deploy it with `irithyll-core` on bare metal, no OS required.

**Paper / origin.** Same e-prop foundation as `SpikeNet`. Fixed-point design informed by
Kleyko et al. (2020) integer ESN work. → [REFERENCES.md §Related Work](REFERENCES.md).

**Math summary.** Same e-prop equations as `SpikeNet` but all intermediate values in Q1.14
format (1 sign bit, 1 integer bit, 14 fractional bits). Saturation arithmetic prevents overflow.

**Config reference.** `SpikeNetFixed::new(n_in, n_hidden, n_out)` — no builder, construction is explicit.

**Example.**
```rust
// In no_std context with irithyll-core
use irithyll_core::snn::SpikeNetFixed;

let mut model = SpikeNetFixed::new(4, 64, 1);
// features and targets passed as Q1.14 scaled i16 values
```

**Benchmarks.** Verified on Cortex-M0+ (22KB), M3, M4 via QEMU. See README embedded section.

---

## Neural Tier — Test-Time Training

---

### StreamingTTT

[`StreamingTTT`](https://docs.rs/irithyll/latest/irithyll/ttt/struct.StreamingTTT.html) — fast weights updated by gradient descent per sample (Titans)

The hidden state IS a small linear model (fast weight matrix W). On each sample, W is updated
by gradient descent on a self-supervised reconstruction loss, then the prediction is read
from the updated W. Titans extensions (momentum + weight decay on W) from Behrouz et al.
(2025) improve stability. Outperforms RLS on regime-shift benchmarks by ~9%.

**Paper / origin.** Sun et al. (2024) for TTT; Behrouz et al. (2025) for Titans extensions.
→ [REFERENCES.md §Test-Time Training](REFERENCES.md).

**Math summary.**
```
Fast weights:  W_t = (1 - alpha) * W_{t-1} + momentum_term
               Updated by: W_t = W_{t-1} - eta * grad_L_reconstruction(W_{t-1}, x_t)
Prediction:    y_hat = W_t * K(x_t)
Momentum:      m_t = beta * m_{t-1} + eta * grad
Weight decay:  W_t -= alpha * W_t  (Titans)
```

**Config reference.** `streaming_ttt(d_model, learning_rate)` factory or `TTTConfig::builder()`

Key parameters:
- `d_model` — fast weight matrix size
- `learning_rate` (eta) — fast weight SGD step (default: 0.1)
- `alpha` — weight decay coefficient (default: 0.001)
- `momentum` — momentum coefficient for fast weight updates

**Example.**
```rust
use irithyll::{streaming_ttt, StreamingLearner};

let mut model = streaming_ttt(8, 0.01);
model.train(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4.5);
let pred = model.predict(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
```

**Benchmarks.** Beats RLS by 9% on regime-shift benchmarks. See [BENCHMARKS.md](BENCHMARKS.md) TTT rows.

---

## Neural Tier — Kolmogorov-Arnold Networks

---

### StreamingKAN

[`StreamingKAN`](https://docs.rs/irithyll/latest/irithyll/kan/struct.StreamingKAN.html) — B-spline edge activations with Adagrad sparse updates

KAN replaces fixed activation functions with learnable univariate functions on each edge,
parameterized as B-splines. Adagrad per-coefficient adaptive learning rates keep convergence
stable. Only 4 spline coefficients (the active basis functions) update per edge per sample,
giving sparse updates. Outperforms RLS on the Friedman benchmark by ~12%.

**Paper / origin.** Liu et al. (2024) for KAN; Hoang et al. (2026) for online convergence
analysis. → [REFERENCES.md §Kolmogorov-Arnold Networks](REFERENCES.md).

**Math summary.**
```
KAN layer: phi_{i,j}(x) = w_b * b(x) + w_s * spline(x)
           spline(x) = sum_k c_k * B_k(x)  (B-spline basis, order p, grid_size G)
           Only 4 active B_k (De Boor's algorithm) update per sample
Adagrad:   lr_{i,j,k} = lr_0 / sqrt(G_{i,j,k} + eps)
Output:    y = sum_j phi_{i,j}(x_i)  (across hidden sizes)
```

**Config reference.** `streaming_kan(n_in, n_out)` factory or `KANConfig::builder()`

Key parameters:
- `n_in` — input dimension
- `hidden_sizes` — list of hidden layer sizes
- `spline_order` — B-spline polynomial order (default: 3)
- `grid_size` — number of grid intervals (default: 8)
- `learning_rate` — base Adagrad learning rate
- `w_b`, `w_s` — blend weights for base vs spline components

**Example.**
```rust
use irithyll::{streaming_kan, StreamingLearner};

let mut model = streaming_kan(4, 1);
for i in 0..200 {
    let x = i as f64 * 0.05;
    model.train(&[x, x.sin(), x.cos(), x * x], x.sin() + x.cos());
}
let pred = model.predict(&[1.0, 1.0_f64.sin(), 1.0_f64.cos(), 1.0]);
```

**Benchmarks.** +12% over RLS on Friedman. See [BENCHMARKS.md](BENCHMARKS.md) compositional function rows.

---

### TKAN

[`TKAN`](https://docs.rs/irithyll/latest/irithyll/kan/struct.TKAN.html) — Temporal KAN with recurrent B-spline state

Extends `StreamingKAN` with a recurrent connection: hidden state from the previous step is
fed back through an additional B-spline edge, giving the network a memory of prior inputs
without a separate reservoir. Use for time series with compositional nonlinear dynamics.

**Paper / origin.** <!-- TODO: spec — no direct TKAN paper in REFERENCES.md. Temporal extension developed for irithyll. -->

**Math summary.**
```
State:   s_t = phi_{recurrent}(s_{t-1}) + phi_{input}(x_t)
Output:  y_t = W_out * s_t
Both phi are KAN layers with B-spline activations.
```

**Config reference.** `TKANConfig::builder()`

Key parameters:
- `n_in`, `hidden_sizes`, `n_out` — same as KAN
- `state_size` — dimension of recurrent state

**Example.**
```rust
use irithyll::kan::{TKAN, TKANConfig};
use irithyll::StreamingLearner;

let config = TKANConfig::builder().n_in(4).state_size(8).build().unwrap();
let mut model = TKAN::new(config);
model.train(&[0.5; 4], 1.0);
```

**Benchmarks.** <!-- TODO: spec -->

---

## Neural Tier — Online Learning Extensions

---

### StreamingLSTM (sLSTM)

[`StreamingLSTM`](https://docs.rs/irithyll/latest/irithyll/struct.StreamingLSTM.html) — exponential-gated LSTM, xLSTM scalar variant

The sLSTM cell from Beck et al. (2024) xLSTM: exponential gate activations replace sigmoid,
providing a numerically stable alternative to standard LSTM gating. The SLOTS mechanism
(block-diagonal recurrent weights when `n_heads > 1`) mixes only within head subspaces while
input projections remain dense. Optional neuron regeneration (Dohare et al. 2024 Nature)
prevents dead units on long streams.

**Paper / origin.** Beck et al. (2024), "xLSTM: Extended long short-term memory." NeurIPS 2024;
Dohare et al. (2024) for plasticity. → [REFERENCES.md §Streaming Linear Attention](REFERENCES.md)
(mLSTM / sLSTM entry).

**Math summary.**
```
sLSTM cell:
  f_t = exp(W_f x_t + R_f h_{t-1} + b_f)   (exponential forget gate)
  i_t = exp(W_i x_t + R_i h_{t-1} + b_i)   (exponential input gate)
  z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)
  o_t = sigma(W_o x_t + R_o h_{t-1} + b_o)
  c_t = f_t * c_{t-1} / max(f_t, 1) + i_t * z_t
  h_t = o_t * tanh(c_t / max(|c_t|, 1))
Readout: y_t = W_out * h_t, trained by RLS
```

**Config reference.** `SLSTMConfig::builder()`

Key parameters:
- `d_model` — hidden dimension (default: 32)
- `n_heads` — SLOTS block-diagonal heads (default: 1)
- `forgetting_factor` — RLS readout lambda (default: 0.998)
- `forget_bias` — optional `linspace(3, 6)` initialization from Beck et al.
- `plasticity` — `PlasticityConfig` for neuron regeneration

**Example.**
```rust
use irithyll::lstm::{SLSTMConfig, StreamingLSTM};
use irithyll::StreamingLearner;

let config = SLSTMConfig::builder().d_model(32).build().unwrap();
let mut model = StreamingLSTM::new(config);
model.train(&[0.5; 8], 1.0);
```

**Benchmarks.** See [BENCHMARKS.md](BENCHMARKS.md) sLSTM rows on non-stationary sequence datasets.

---

### StreamingMGrade (mGRADE)

[`StreamingMGrade`](https://docs.rs/irithyll/latest/irithyll/struct.StreamingMGrade.html) — modulated gradient recurrent architecture

mGRADE modulates the gradient signal through a depthwise convolution over a short temporal
kernel before applying the recurrent update. This soft memory of recent gradients helps
adapt quickly to local regime changes without the full overhead of meta-learning.

**Paper / origin.** <!-- TODO: spec — mGRADE paper not listed in REFERENCES.md. Add citation when identified. -->

**Math summary.**
```
Conv feature: c_t = depthwise_conv(x_{t-kernel_size:t})
Hidden:       h_t = tanh(W_h h_{t-1} + W_c c_t)
Readout:      y_t = W_out * h_t, trained by RLS
```

**Config reference.** `MGradeConfig::builder()`

Key parameters:
- `d_in` — input dimension
- `d_hidden` — recurrent hidden size
- `kernel_size` — depthwise conv kernel length
- `forgetting_factor` — RLS readout lambda

**Example.**
```rust
use irithyll::mgrade::{MGradeConfig, StreamingMGrade};
use irithyll::StreamingLearner;

let config = MGradeConfig::builder().d_in(4).d_hidden(16).kernel_size(3).build().unwrap();
let mut model = StreamingMGrade::new(config);
model.train(&[0.5; 4], 1.0);
```

**Benchmarks.** <!-- TODO: spec -->

---

### StreamingAGMP

[`StreamingAGMP`](https://docs.rs/irithyll/latest/irithyll/struct.StreamingAGMP.html) — adaptive gradient meta-predictor

AGMP maintains a second-order estimate of the gradient curvature online, using it to scale
the primary model's update step adaptively. Designed for non-stationary streams where the
curvature of the loss surface changes over time.

**Paper / origin.** <!-- TODO: spec — AGMP paper not in REFERENCES.md. Add citation when identified. -->

**Math summary.** <!-- TODO: spec — exact recurrence for AGMP not confirmed from source. -->

**Config reference.** `AGMPConfig::builder()`

**Example.**
```rust
use irithyll::StreamingAGMP;
use irithyll::StreamingLearner;

let mut model = StreamingAGMP::new(agmp_config);
model.train(&[1.0; 4], 2.0);
```

**Benchmarks.** <!-- TODO: spec -->

---

### StreamingHGRN2

[`StreamingHGRN2`](https://docs.rs/irithyll/latest/irithyll/struct.StreamingHGRN2.html) — hierarchical gated recurrent network (attention variant)

HGRN2 uses per-dimension forget gates with a configurable lower bound, ensuring a minimum
level of memory retention at each gate position. The outer-product state update (`v k^T`)
provides expressivity comparable to GLA. Use for sequences with long-range dependencies where
GLA's data-dependent gating is over-parameterized.

**Paper / origin.** Qin et al. (2024) "HGRN2: Gated Linear RNNs with State Expansion." — [citation
needed for REFERENCES.md anchor; HGRN2 is available via the `hgrn2()` factory in the attention module]

**Math summary.**
```
Gate:    f_t = lower_bound + (1 - lower_bound) * sigmoid(W_f x_t + b_f)
State:   S_t = f_t * S_{t-1} + (1 - f_t) * v_t k_t^T   (outer product update)
Output:  o_t = S_t q_t
```

**Config reference.** `hgrn2(d_model, n_heads, lower_bound)` factory or `StreamingAttentionConfig::builder().mode(AttentionMode::HGRN2 { lower_bound })`

Key parameters:
- `d_model` — model dimension
- `n_heads` — number of heads
- `lower_bound` — minimum forget gate value (prevents catastrophic forgetting)

**Example.**
```rust
use irithyll::attention::hgrn2;
use irithyll::StreamingLearner;

let mut model = hgrn2(8, 2, 0.9);
model.train(&[1.0; 8], 0.5);
```

**Benchmarks.** See [BENCHMARKS.md](BENCHMARKS.md) attention family rows.

---

## Neural Tier — Streaming Linear Attention

All modes share the [`StreamingAttentionModel`](https://docs.rs/irithyll/latest/irithyll/attention/struct.StreamingAttentionModel.html)
wrapper: multi-head recurrent attention + RLS readout. Select the mode via `AttentionMode` enum or
the dedicated factory function. The canonical streaming readout is `o_t = q(x_t) · S_{t-1}` (pre-update
features, never stale, label-independent — see AGENTS.md streaming readout principle).

| Mode | Factory | Architecture | Recommended for |
|------|---------|-------------|-----------------|
| GLA | `gla(d, h)` | Data-dependent scalar gates per channel | General sequential tasks, stable convergence |
| GatedDeltaNet | `delta_net(d, h)` | Delta rule + gating (NVIDIA) | Strongest associative recall |
| DeltaNet | via `AttentionMode::DeltaNet` | Pure delta rule (no extra gate) | Associative lookup, lighter than GatedDeltaNet |
| DeltaProduct | `delta_product(d, h, n)` | n sequential delta steps + Householder | Tunable expressivity |
| RWKV-7 | `rwkv7(d, h)` | Vector decay, vector in-context LR | In-context learning, DPLR transitions |
| HGRN2 | `hgrn2(d, h, lb)` | Lower-bounded forget gates, outer product | Long-range dependencies |
| Hawk | `hawk(d)` | Vector state, single head | Resource-constrained deployments |
| RetNet | `ret_net(d, gamma)` | Fixed exponential decay | Predictable, simple baseline |
| mLSTM | via `AttentionMode::MLSTM` | Matrix LSTM, exponential gating | xLSTM-style streaming |
| RWKV | via `AttentionMode::RWKV` | Time-mixing with scalar decay | Lightweight temporal mixing |
| Log-Linear | `log_linear(d, h, inner, levels)` | Hierarchical Fenwick state, O(log T) | Bridging linear and softmax expressivity |

**Paper citations for all modes.** → [REFERENCES.md §Streaming Linear Attention](REFERENCES.md).

---

### Log-Linear Attention (detail)

[`log_linear()`](https://docs.rs/irithyll/latest/irithyll/attention/fn.log_linear.html) — Han Guo et al. (ICLR 2026) hierarchical Fenwick state

The headline architecture. Wraps any inner linear-attention rule with an O(log T) hierarchical
state indexed by a Fenwick tree. At each level l, the state S^l aggregates the past 2^l tokens
with interpolated lambda weights. This bridges linear-attention efficiency (O(1) per token
update) and softmax expressivity (log-T effective context).

**Paper / origin.** Han Guo, Songlin Yang, Tarushii Goel, Eric P. Xing, Tri Dao, Yoon Kim.
"Log-Linear Attention." ICLR 2026. arXiv:2506.04761. — [citation needed for REFERENCES.md anchor]

**Math summary.**
```
Fenwick state: S_t^l = lambda^l * S_{t-1}^l + k_t v_t^T  for l = 0..max_levels
Query:         o_t   = sum_l weight_l * S_t^l * q_t
               weight_l = lambda_init[l] (learned interpolation)
State memory:  max_levels * d_k * d_v * n_heads  (constant, independent of T)
MQAR headline: recall >= 0.9 at 128 associative pairs
```

**Config reference.** `log_linear(d_model, n_heads, inner, max_levels)` factory.

Key parameters:
- `inner` — any `AttentionMode` (recommend `GatedDeltaNet` for recall, `GLA` for stability)
- `max_levels` — Fenwick depth cap (32 covers ~4G tokens)
- `lambda_init` — per-level interpolation weights (default via `default_lambda_init()`)

**Example.**
```rust
use irithyll::attention::{log_linear, AttentionMode, GatedDeltaMode};
use irithyll::StreamingLearner;

let mut model = log_linear(
    8, 2,
    AttentionMode::GatedDeltaNet { beta_scale: 1.0, gate_mode_delta: GatedDeltaMode::Static },
    32,
);
model.train(&[1.0; 8], 0.5);
let pred = model.predict(&[1.0; 8]);
```

**Benchmarks.** MQAR recall >= 0.9 at 128 pairs required (craft audit gate). See [BENCHMARKS.md](BENCHMARKS.md).

---

## Ensemble Tier

---

### NeuralMoE

[`NeuralMoE`](https://docs.rs/irithyll/latest/irithyll/moe/struct.NeuralMoE.html) — polymorphic mixture of experts with top-k routing

Any `StreamingLearner` can be an expert — mix ESN, Mamba, SpikeNet, SGBT, and attention
models in one ensemble. A linear softmax router, trained online by SGD on cross-entropy
against the best expert, assigns each sample to the top-k experts. DeepSeek-v3-style load
balancing prevents routing collapse. EWMA utilization tracking auto-resets dead experts.

**Paper / origin.** Jacobs et al. (1991) for adaptive mixtures; Shazeer et al. (2017) for
top-k routing; Wang et al. (2024) for auxiliary-loss-free load balancing; Aspis et al. (2025)
for streaming classification with drift. → [REFERENCES.md §Neural MoE](REFERENCES.md).

**Math summary.**
```
Gate:     g_t = softmax(W_router * x_t + bias)
Top-k:    route to k experts with highest g_t[i]
Expert i: y_i = expert_i.predict(x_t)
Output:   y_t = sum_{i in top-k} g_t[i] * y_i / sum_{i in top-k} g_t[i]
Load:     bias[i] -= eps * (utilization[i] - 1/n_experts)  (DeepSeek-v3)
```

**Config reference.** `NeuralMoE::builder()`

Key parameters:
- `expert(model)` — add any `StreamingLearner` as an expert
- `expert_with_warmup(model, n)` — protect expert from elimination during first n samples
- `top_k(k)` — number of experts activated per sample (default: 2)

**Example.**
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

**Benchmarks.** Strongest on heterogeneous or regime-shifting data. See [BENCHMARKS.md](BENCHMARKS.md).

---

### AdaptiveRandomForest

[`AdaptiveRandomForest`](https://docs.rs/irithyll/latest/irithyll/ensemble/struct.AdaptiveRandomForest.html) — online random forest with concept drift adaptation

Maintains an ensemble of online decision trees built on bootstrap samples (Poisson lambda=6
weighting). When a drift warning is detected on a background tree, a new candidate tree starts
training in parallel. On confirmed drift, the old foreground tree is replaced. Each tree uses
feature subsampling for decorrelation.

**Paper / origin.** Gomes et al. (2017) "Adaptive Random Forests for Evolving Data Stream
Classification." ECML-PKDD 2017. — [citation needed for REFERENCES.md anchor]

**Math summary.**
```
Bootstrap: sample weight w ~ Poisson(lambda=6) per tree per sample
Subsampling: feature fraction f per tree
Drift: ADWIN on per-tree error -> warning -> background tree -> replacement
```

**Config reference.** `ARFConfig::builder(n_trees)`

Key parameters:
- `n_trees` — forest size
- `lambda` — Poisson bootstrap parameter (default: 6.0)
- `feature_fraction` — feature subsampling rate per tree
- `drift_delta`, `warning_delta` — ADWIN thresholds for drift and warning

**Example.**
```rust
use irithyll::ensemble::{AdaptiveRandomForest, ARFConfig};

let config = ARFConfig::builder(10).feature_fraction(0.7).build().unwrap();
let mut model = AdaptiveRandomForest::new(config, || sgbt(10, 0.05));
model.train_one(&[1.0, 2.0], 1.0);
```

**Benchmarks.** <!-- TODO: spec — no dedicated ARF rows in BENCHMARKS.md yet. -->

---

### QuantileRegressorSGBT

[`QuantileRegressorSGBT`](https://docs.rs/irithyll/latest/irithyll/ensemble/struct.QuantileRegressorSGBT.html) — quantile regression via pinball loss

SGBT trained with pinball loss to predict a target quantile (e.g. 0.1 for p10, 0.9 for p90).
Train multiple instances with different quantiles to build a prediction interval. Use for
asymmetric risk or when the full conditional distribution matters.

**Paper / origin.** Koenker & Bassett (1978) for quantile regression; pinball loss is the
gradient of the check function. Built on Gunasekara et al. (2024).

**Math summary.**
```
Pinball loss: L_q(y, f) = q * max(y - f, 0) + (1 - q) * max(f - y, 0)
grad L_q     = -(q * 1[y > f] - (1-q) * 1[y <= f])
```

**Config reference.** `QuantileRegressorSGBT::new(config, quantile)`

**Example.**
```rust
use irithyll::ensemble::QuantileRegressorSGBT;
use irithyll::SGBTConfig;

let config = SGBTConfig::builder().n_steps(50).build().unwrap();
let mut p10 = QuantileRegressorSGBT::new(config.clone(), 0.1);
let mut p90 = QuantileRegressorSGBT::new(config, 0.9);
p10.train(&[1.0], 2.5);
p90.train(&[1.0], 2.5);
```

**Benchmarks.** <!-- TODO: spec -->

---

## Specialized Tier

---

### GaussianNB

[`GaussianNB`](https://docs.rs/irithyll/latest/irithyll/struct.GaussianNB.html) — streaming Gaussian Naive Bayes classifier

Tracks per-class, per-feature Gaussian statistics (mean, variance) incrementally via Welford's
algorithm. At prediction time, computes log-posterior = log-likelihood + log-prior. Use for
text classification, categorical features, or as a fast baseline classifier.

**Paper / origin.** Classical Bayesian classification. Online moment updates via Welford (1962).

**Math summary.**
```
p(x_j | class=k) = N(x_j; mu_{k,j}, sigma_{k,j}^2)
log P(class=k | x) = sum_j log N(x_j; mu_{k,j}, sigma^2_{k,j}) + log pi_k
prediction: argmax_k P(class=k | x)
```

**Config reference.** `gaussian_nb()` factory or `GaussianNB::new()`

Key parameters:
- `var_smoothing` — add to all variances for numerical stability (default: 1e-9)

**Example.**
```rust
use irithyll::{gaussian_nb, StreamingLearner};

let mut model = gaussian_nb();
model.train(&[0.1, 0.9], 1.0);  // class as float
model.train(&[0.9, 0.1], 0.0);
let pred = model.predict(&[0.2, 0.8]);
```

**Benchmarks.** Fastest classifier. See [BENCHMARKS.md](BENCHMARKS.md) binary classification baseline rows.

---

### HalfSpaceTree

[`HalfSpaceTree`](https://docs.rs/irithyll/latest/irithyll/anomaly/struct.HalfSpaceTree.html) — streaming anomaly detection via mass estimation

An ensemble of binary trees where each node randomly selects a feature and a split value
from the observed data range. Anomaly score is the mass estimate: how often a sample lands
in a large leaf (normal) vs a small leaf (anomalous). Scores are windowed; mass estimates
are periodically re-initialized to handle concept drift.

**Paper / origin.** Tan et al. (2011) "Fast Anomaly Detection for Streaming Data." IJCAI 2011.
— [citation needed for REFERENCES.md anchor]

**Math summary.**
```
score(x) = sum_{t=1}^{n_trees} mass_t(x) / n_trees
mass_t(x): number of samples in the leaf where x falls in tree t
Anomaly threshold: samples with score < threshold are flagged
```

**Config reference.** `HSTConfig::builder(n_features)`

Key parameters:
- `n_trees` — ensemble size (default: 25)
- `max_depth` — tree depth
- `window_size` — samples per mass estimation window
- `threshold` — anomaly score threshold

**Example.**
```rust
use irithyll::anomaly::{HalfSpaceTree, HSTConfig};

let config = HSTConfig::new(4).n_trees(25).max_depth(15).window_size(250).build();
let mut detector = HalfSpaceTree::new(config);
detector.update(&[1.0, 2.0, 1.5, 3.0]);
let score = detector.score(&[100.0, 200.0, 150.0, 300.0]);
println!("is_anomaly: {}", score.is_anomaly());
```

**Benchmarks.** <!-- TODO: spec -->

---

### ContinualLearner

[`ContinualLearner`](https://docs.rs/irithyll/latest/irithyll/continual/struct.ContinualLearner.html) — drift-aware wrapper with pluggable continual strategies

Wraps any `StreamingLearner` with an optional drift detector and continual learning strategy.
Strategies include Elastic Weight Consolidation (EWC — regularize against forgetting important
weights), neuron regeneration (periodic random re-initialization of dead units), and parameter
isolation (freeze high-importance parameters). On detected drift, can optionally reset the
inner model or trigger a strategy update.

**Paper / origin.** Kirkpatrick et al. (2017) for EWC; Dohare et al. (2024) for neuron
regeneration / plasticity. → [REFERENCES.md §Plasticity and Continual Learning](REFERENCES.md).

**Config reference.** `ContinualLearner::new(inner_model)` + builder methods

Key parameters:
- `with_drift_detector(detector)` — plug in ADWIN, DDM, or PageHinkley
- `with_reset_on_drift(bool)` — reset inner model on confirmed drift
- EWC config via `ContinualStrategy::EWC { lambda: f64 }`

**Example.**
```rust
use irithyll::continual::ContinualLearner;
use irithyll::drift::adwin::Adwin;
use irithyll::{sgbt, StreamingLearner};

let mut model = ContinualLearner::new(sgbt(50, 0.01))
    .with_drift_detector(Adwin::default())
    .with_reset_on_drift(true);
model.train(&[1.0, 2.0], 3.0);
println!("drift count: {}", model.drift_count());
```

**Benchmarks.** Evaluated on drift recovery benchmarks. See [BENCHMARKS.md](BENCHMARKS.md).

---

### ClassificationWrapper

[`ClassificationWrapper`](https://docs.rs/irithyll/latest/irithyll/struct.ClassificationWrapper.html) — binary/multiclass classifier from any StreamingLearner

Wraps any regression-outputting `StreamingLearner` for classification. Uses bipolar {-1, +1}
targets internally for proper MSE-based discrimination (standard in ESN/reservoir literature).
Binary: threshold output at 0.0. Multiclass: K one-vs-rest heads with stable softmax.

**Config reference.** `binary_classifier(model)` / `multiclass_classifier(model, n_classes)` factories.

**Example.**
```rust
use irithyll::{binary_classifier, esn, StreamingLearner};

let mut clf = binary_classifier(esn(50, 0.9));
clf.train(&[1.0, 2.0], 1.0);  // 1.0 = positive class
clf.train(&[0.1, 0.2], 0.0);  // 0.0 = negative class
let pred = clf.predict(&[1.0, 2.0]);  // 0.0 or 1.0
```

**Benchmarks.** Used in all classification benchmarks wrapping neural models. See [BENCHMARKS.md](BENCHMARKS.md).

---

### ProjectedLearner

[`ProjectedLearner`](https://docs.rs/irithyll/latest/irithyll/projection/struct.ProjectedLearner.html) — online subspace projection + supervised gradient

Wraps any `StreamingLearner` with a PAST-based `SubspaceTracker` that learns a low-rank
projection of the input space online. When the inner model exposes RLS readout weights
(via `readout_weights()`), the projection gradient is supervised by the prediction residual.
Otherwise it falls back to unsupervised PCA (pure PAST). Use for high-dimensional or noisy
inputs where dimensionality reduction should adapt to the target.

**Paper / origin.** Yang (1995) for PAST. → [REFERENCES.md §Projection Learning](REFERENCES.md).

**Math summary.**
```
PAST unsupervised: W_{t+1} = W_t + (x_t - W_t z_t)(z_t / ||z_t||^2)^T
                   where z_t = W_t^T x_t  (projection onto rank-r subspace)
Supervised path:   additional gradient step using prediction residual * readout_weights
Output:            inner_model.train(z_t, y_t)  (trains on projected features)
```

**Config reference.** `ProjectionConfig::builder()`

Key parameters:
- `rank` — projection rank (number of principal components to track)
- `lambda` — PAST forgetting factor (default: 0.998)
- `supervised_lr` — learning rate for supervised projection gradient
- `warmup` — samples to collect before activating projection

Also available via `Factory::projected_mamba(d_in, rank)`, `Factory::projected_ttt(...)`, etc.

**Example.**
```rust
use irithyll::{mamba, StreamingLearner};
use irithyll::projection::{ProjectedLearner, ProjectionConfig};

let config = ProjectionConfig::builder().rank(4).warmup(50).build().unwrap();
let mut model = ProjectedLearner::from_learner(mamba(4, 16), 16, config);
model.train(&[1.0; 16], 0.5);
let pred = model.predict(&[1.0; 16]);
```

**Benchmarks.** See [BENCHMARKS.md](BENCHMARKS.md) high-dimensional input rows.

---

## Preprocessing

All preprocessors implement `StreamingPreprocessor` and compose via `pipe(preprocessor).learner(model)`
or `pipe(preprocessor).pipe(preprocessor2).learner(model)`.

---

### IncrementalNormalizer

[`IncrementalNormalizer`](https://docs.rs/irithyll/latest/irithyll/struct.IncrementalNormalizer.html) — Welford's online standardization

Tracks per-feature mean and variance using Welford's one-pass algorithm. During warmup,
passes features through unchanged. After warmup, outputs `(x - mu) / sigma` per feature.
Variance floor prevents division by zero on constant features. Use as the first stage of
any pipeline where features have heterogeneous scales.

**Math summary.**
```
Welford: delta = x - mean;  mean += delta / n;  M2 += delta * (x - mean)
sigma^2 = M2 / (n - 1)  (or variance_floor if too small)
output  = (x - mean) / sigma
```

**Example.**
```rust
use irithyll::{pipe, normalizer, sgbt, StreamingLearner};

let mut model = pipe(normalizer()).learner(sgbt(50, 0.01));
model.train(&[100.0, 0.001], 42.0);
```

---

### CCIPCA

[`CCIPCA`](https://docs.rs/irithyll/latest/irithyll/struct.CCIPCA.html) — candid covariance-free incremental PCA

O(kd) streaming PCA without computing or storing the covariance matrix. Tracks the top-k
principal components via a sequence of deflated rank-1 updates. Use for streaming
dimensionality reduction before a downstream learner.

**Paper / origin.** Weng, Zhang & Hwang (2003). → [REFERENCES.md §Core Gradient Boosting](REFERENCES.md)
(CCIPCA entry).

**Math summary.**
```
For each component i = 1..k:
  v_i_{t+1} = (1 - l/n) * v_i_t + (l/n) * x_t * (x_t^T v_i_t / ||v_i_t||)
  deflate:    x_t -= (x_t^T u_i) * u_i   where u_i = v_i / ||v_i||
Output: k-dimensional projection z_t = U^T x_t
```

**Example.**
```rust
use irithyll::{pipe, ccipca, normalizer, sgbt, StreamingLearner};

let mut model = pipe(normalizer()).pipe(ccipca(5)).learner(sgbt(50, 0.01));
model.train(&[1.0; 20], 3.0);
```

---

### OnlineFeatureSelector

[`OnlineFeatureSelector`](https://docs.rs/irithyll/latest/irithyll/struct.OnlineFeatureSelector.html) — streaming mutual-information feature selection

Estimates mutual information between each feature and the target online, using a histogram-based
discretization updated incrementally. Selects the top-k features by MI estimate for downstream
learning.

**Config reference.** `OnlineFeatureSelector::new(k_features)`

**Example.**
```rust
use irithyll::preprocessing::OnlineFeatureSelector;
use irithyll::{pipe, sgbt, StreamingLearner};

let mut model = pipe(OnlineFeatureSelector::new(5)).learner(sgbt(50, 0.01));
model.train(&[1.0; 20], 3.0);
```

---

### TargetEncoder

[`TargetEncoder`](https://docs.rs/irithyll/latest/irithyll/preprocessing/struct.TargetEncoder.html) — online target encoding for categorical features

Replaces categorical feature values with their running mean target value, smoothed towards
the global target mean. Prevents high-cardinality categoricals from blowing up feature space.
Operates directly on f64-encoded category indices in the feature vector.

**Math summary.**
```
encoded(cat) = (n_cat * mean_cat + smoothing * global_mean) / (n_cat + smoothing)
```

**Config reference.** `TargetEncoder::new(categorical_indices)` or `TargetEncoder::with_smoothing(indices, s)`

---

### TargetScaler / TargetLog1pTransform

These are target-side preprocessors that normalize the regression target before passing it
to the inner model, then invert the transform on prediction.

- [`TargetScaler`](https://docs.rs/irithyll/latest/irithyll/preprocessing/struct.TargetScaler.html) — online Welford standardization of the target
- [`TargetLog1pTransform`](https://docs.rs/irithyll/latest/irithyll/preprocessing/struct.TargetLog1pTransform.html) — `log1p` / `expm1` transform for heavy-tailed targets

---

## Clustering

Clustering algorithms do not implement `StreamingLearner`; they expose their own `train_one` /
`predict` API returning cluster assignments.

---

### StreamingKMeans

[`StreamingKMeans`](https://docs.rs/irithyll/latest/irithyll/clustering/struct.StreamingKMeans.html) — online k-means with exponential forgetting

MacQueen online k-means updated with exponential forgetting: centroids are pulled towards
new samples weighted by the forgetting factor. Assignment to nearest centroid, then winner
centroid update.

**Math summary.**
```
assignment: c* = argmin_c ||x - centroid_c||^2
update:     centroid_{c*} = (1 - lr) * centroid_{c*} + lr * x
            where lr = 1 - forgetting_factor per sample
```

**Config reference.** `StreamingKMeansConfig::builder(k)`

---

### CluStream

[`CluStream`](https://docs.rs/irithyll/latest/irithyll/clustering/struct.CluStream.html) — micro-cluster streaming with temporal extensions

Aggarwal et al. (2003) CluStream: maintains a fixed number of micro-clusters as online
summaries. Each micro-cluster is a (N, LS, SS, LST, SST) tuple — the cluster feature extended
with temporal statistics. Assignment is to nearest micro-cluster by radius; merging when
capacity is exceeded.

**Paper / origin.** Aggarwal et al. (2003) "A Framework for Clustering Evolving Data Streams."
VLDB 2003. — [citation needed for REFERENCES.md anchor]

**Config reference.** `CluStreamConfig::builder(max_micro_clusters)`

---

### DBStream

[`DBStream`](https://docs.rs/irithyll/latest/irithyll/clustering/struct.DBStream.html) — density-based streaming clustering with decay

Density-based clustering adapted for streams: micro-clusters grow when new points arrive
within `radius`. Shared density between micro-clusters is tracked and decays over time.
Macro-cluster connectivity is derived from shared density above a threshold. Handles
arbitrary cluster shapes and naturally identifies noise points.

**Paper / origin.** Hahsler & Dunham (2010) "Density-Based Projected Clustering of Data
Streams." SIAM SDM 2010. — [citation needed for REFERENCES.md anchor]

**Config reference.** `DBStreamConfig::builder(radius)`

---

## Streaming AutoML

---

### AutoTuner

[`AutoTuner`](https://docs.rs/irithyll/latest/irithyll/automl/struct.AutoTuner.html) — streaming hyperparameter optimization via tournament racing

Implements `StreamingLearner`: trains and predicts like any model, internally running
champion-challenger tournament successive halving in the background. The champion is always
the live predictor; challengers are evaluated prequentially in parallel. ADWIN-triggered
re-racing restarts the tournament with an expanded bracket on detected drift.

**Paper / origin.** Wu et al. (2021) ChaCha for tournament racing; Qi et al. (2023) Discounted
Thompson Sampling for factory selection; Yamanishi (2018) for MDL-inspired complexity
adjustment. → [REFERENCES.md §Streaming AutoML](REFERENCES.md).

**Math summary.**
```
Complexity-adjusted score: score = ewma_error + complexity / n_seen
                           complexity = n_params * log(n_params) (MDL-inspired)
Successive halving:        round_budget samples -> eliminate bottom half -> repeat
Statistical early stopping: Welford z-test (z > 2.0, min 30 samples)
```

**Config reference.** `auto_tune(factory)` / `AutoTuner::builder()`

Key parameters:
- `add_factory(Factory)` — register a model family for the race
- `use_drift_rerace(bool)` — restart tournament on ADWIN drift signal
- `round_budget`, `n_initial` — halving tournament parameters

**Example.**
```rust
use irithyll::{auto_tune, automl::Factory, StreamingLearner};

// Single factory: auto-tune SGBT hyperparameters
let mut tuner = auto_tune(Factory::sgbt(5));

// Multi-factory: race trees vs neural
let mut tuner = AutoTuner::builder()
    .add_factory(Factory::sgbt(5))
    .add_factory(Factory::esn())
    .add_factory(Factory::mamba(5))
    .use_drift_rerace(true)
    .build();
tuner.train(&[1.0, 2.0, 3.0], 4.0);
let pred = tuner.predict(&[1.0, 2.0, 3.0]);
```

---

### Factory

[`Factory`](https://docs.rs/irithyll/latest/irithyll/automl/struct.Factory.html) — unified model factory with hyperparameter search spaces

Provides a hyperparameter search space for each model family and constructs sampled instances
for AutoTuner racing. Also exposes convenience constructors for projected variants.

| Factory method | Model family |
|---------------|-------------|
| `Factory::sgbt(n_rounds)` | SGBT family |
| `Factory::distributional(n_rounds)` | DistributionalSGBT |
| `Factory::esn()` | EchoStateNetwork |
| `Factory::mamba(n_rounds)` | StreamingMamba |
| `Factory::ttt()` | StreamingTTT |
| `Factory::kan()` | StreamingKAN |
| `Factory::attention(mode)` | StreamingAttentionModel |
| `Factory::spike_net()` | SpikeNet |
| `Factory::projected_mamba(d_in, rank)` | ProjectedLearner wrapping Mamba |
| `Factory::projected_ttt(d_in, rank)` | ProjectedLearner wrapping TTT |
| `Factory::projected_esn(d_in, rank)` | ProjectedLearner wrapping ESN |

---

## Drift Detectors

Drift detectors are not `StreamingLearner`s; they are embedded in `SGBTConfig`, `ContinualLearner`,
and `AutoTuner`. All implement a `DriftDetector` trait with `update(error) -> DriftSignal`.

| Detector | Type | Key parameter |
|----------|------|---------------|
| `PageHinkley` | Sequential change-point | `delta` (sensitivity), `threshold` |
| `Adwin` | Adaptive windowing | `delta` (confidence) |
| `DDM` | Drift Detection Method | `warning_level`, `drift_level` |

**Paper citations.** → [REFERENCES.md §Algorithmic Foundations](REFERENCES.md).

---

## Conformal Prediction

Conformal prediction modules augment any point predictor with coverage guarantees. They are
not `StreamingLearner`s; they wrap a fitted model and maintain a nonconformity score buffer.

| Component | Description | Paper |
|-----------|-------------|-------|
| `ConformalPID` | PID-controlled adaptive coverage | Angelopoulos et al. (2023) |
| `OnlinePlattScaling` | Online probability calibration | Gupta & Ramdas (2023) |
| `OnlineTemperatureScaling` | Single-parameter multiclass calibration | [citation needed] |

**Paper citations.** → [REFERENCES.md §Conformal Prediction](REFERENCES.md).

---

## Packed Inference (`irithyll-core`)

After training with the full `irithyll` crate, export to a compact packed binary for
zero-alloc inference on embedded targets.

```rust
use irithyll::export_embedded::export_packed;
use irithyll_core::EnsembleView;

// Export (host)
let packed: Vec<u8> = export_packed(&model, n_features);
std::fs::write("model.bin", &packed).unwrap();

// Infer (embedded, no_std)
let model_bytes: &[u8] = include_bytes!("model.bin");
let view = EnsembleView::from_bytes(model_bytes).unwrap();
let pred: f32 = view.predict(&[1.0, 2.0, 3.0]);
```

Each `PackedNode` is 12 bytes (5 nodes per 64-byte cache line). Single-predict latency:
66 ns at 15.2M predictions/second on x86-64 with a 50-tree depth-4 ensemble.

`SpikeNetFixed` (`no_std`, Q1.14) also available in `irithyll-core` for spiking networks on
bare metal — 64 neurons in 22KB, verified on Cortex-M0+, M3, M4.
