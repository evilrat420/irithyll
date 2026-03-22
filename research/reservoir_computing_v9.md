# Reservoir Computing Research for irithyll v9

## Compiled 2026-03-22

---

## Table of Contents

1. [Paper Summaries](#1-paper-summaries)
2. [ESN State Update Equations](#2-esn-state-update-equations)
3. [RLS Readout Update Equations](#3-rls-readout-update-equations)
4. [FORCE Learning Equations](#4-force-learning-equations)
5. [Reservoir Initialization Algorithm](#5-reservoir-initialization-algorithm)
6. [Next Generation RC Equations](#6-next-generation-rc-equations)
7. [Pseudocode](#7-pseudocode)
8. [Hyperparameter Guide](#8-hyperparameter-guide)
9. [Architectural Recommendations for irithyll v9](#9-architectural-recommendations-for-irithyll-v9)
10. [Open Questions and Risks](#10-open-questions-and-risks)

---

## 1. Paper Summaries

### 1.1 Foundational Papers

**Jaeger (2001) -- "The 'echo state' approach to analysing and training recurrent neural networks"**
- Authors: Herbert Jaeger
- Year: 2001 | Citations: 3,090
- Key findings: Introduced the Echo State Network concept. Core idea: construct a large random recurrent neural network (the "reservoir"), drive it with input, and train only the output weights via linear regression. The recurrent weights are fixed after random initialization. Introduced the "echo state property" (ESP) -- the requirement that the effect of initial states washes out over time, so reservoir state depends only on input history.
- Relevance: THE foundational paper. Defines the entire paradigm we are implementing.

**Jaeger & Haas (2004) -- "Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication"**
- Authors: Herbert Jaeger, Harald Haas
- Year: 2004 | Published in Science | Citations: ~3,000
- Key findings: Demonstrated ESNs on practical tasks including chaotic time series prediction and wireless channel equalization. Showed ESN could predict Mackey-Glass chaotic time series with orders of magnitude better accuracy than prior methods.
- Relevance: Proved ESNs work on real tasks. The chaotic time series prediction benchmark is directly relevant to orderflow prediction.

**Lukosevicius & Jaeger (2009) -- "Reservoir computing approaches to recurrent neural network training"**
- Authors: Mantas Lukosevicius, Herbert Jaeger
- Year: 2009 | Computer Science Review | Citations: 2,840
- Key findings: Comprehensive tutorial on reservoir computing. Unified ESNs with Liquid State Machines under the RC umbrella. Detailed practical guidelines for reservoir design, spectral radius selection, input scaling, sparsity.
- Relevance: THE practical reference for implementation. Contains all the design rules we need.

**Lukosevicius (2012) -- "A Practical Guide to Applying Echo State Networks"**
- Authors: Mantas Lukosevicius
- Year: 2012 | Lecture Notes in Computer Science | Citations: 840
- Key findings: Step-by-step practical guide. Recommends: start with reservoir size ~10x number of independent real-valued signals to be learned; typical sparsity 10-20% connectivity; spectral radius close to 1 for tasks requiring long memory; input scaling controls nonlinearity degree; use ridge regression (Tikhonov regularization) for readout training.
- Relevance: Direct implementation guide. Hyperparameter recipes.

### 1.2 Survey Papers

**Yan et al. (2024) -- "Emerging opportunities and challenges for the future of reservoir computing"**
- Authors: Min Yan, Can Huang, Peter Bienstman, Peter Tino, Wei Lin, Jie Sun
- Year: 2024 | Nature Communications | Citations: 249
- Key findings:
  - RC achieves competitive performance in chaotic system prediction and weather forecasting
  - RC deployable on edge devices (wearables, phones) where deep learning cannot fit due to memory constraints
  - RC possesses universal approximation capabilities despite randomly assigned non-trainable weights
  - Key design principles: operate near edge-of-chaos for coupling parameters, hierarchical/deep architectures, input-space transformation (NG-RC)
  - Critical barriers to adoption: lack of unified theory-algorithm-implementation integration, absence of general design guidelines, missing "killer applications" beyond synthetic benchmarks
  - Future: domain-specific architecture design with Bayesian optimization and neural architecture search
  - Online training scheme: periodically insert target data, retrain output weights -- enables longer prediction and prevents divergence
- Relevance: State-of-the-art survey. Validates RC as viable. Identifies exactly the gaps irithyll can fill: lightweight, streaming, well-designed.

**Tanaka et al. (2019) -- "Recent advances in physical reservoir computing: A review"**
- Authors: Gouhei Tanaka et al.
- Year: 2019 | Neural Networks | Citations: 1,948
- Key findings: Comprehensive review of physical RC implementations (optical, electronic, mechanical). Confirmed that RC's computational efficiency and simple training make it attractive for hardware implementation.
- Relevance: Validates that RC maps well to resource-constrained environments -- analogous to our no_std target.

### 1.3 Next Generation RC

**Gauthier et al. (2021) -- "Next generation reservoir computing"**
- Authors: Daniel J. Gauthier, Erik Bollt, Aaron Griffith, Wendson A. S. Barbosa
- Year: 2021 | Nature Communications | Citations: 502
- Key findings:
  - NVAR (nonlinear vector autoregression) is mathematically equivalent to a linear reservoir with polynomial nonlinear readout
  - No random matrices required, no reservoir needed -- feature vector built directly from time-delayed observations + polynomial functions
  - Trains on only 400 data points in <10ms
  - Warm-up period of just 2-3 steps (vs 10^3-10^5 in traditional RC)
  - 33-162x less costly than typical efficient traditional RC
  - Low-order polynomials (quadratic/cubic) provide sufficient nonlinear expressiveness
  - Feature vector: O_total = c + O_lin + O_nonlin where O_lin = time-delayed observations, O_nonlin = polynomial monomials of O_lin
  - Uses Tikhonov regularization for training (same as ridge regression)
  - Successfully forecasts Lorenz63 ~5 Lyapunov times ahead
  - Works on non-polynomial systems (demonstrated with sinh nonlinearity)
- Relevance: CRITICAL. This is potentially simpler than ESN for irithyll. No random matrices, fewer hyperparameters, deterministic, faster. irithyll already has PolynomialFeatures + RLS -- NG-RC might be buildable from existing components.

### 1.4 Advanced Techniques

**Jaeger et al. (2007) -- "Optimization and applications of echo state networks with leaky-integrator neurons"**
- Authors: Herbert Jaeger, Mantas Lukosevicius, Dan Popovici, Udo Siewert
- Year: 2007 | Neural Networks | Citations: 930
- Key findings: Introduced Leaky Integrator ESN (LI-ESN). Adds a leak rate alpha that controls the speed of reservoir dynamics. State update becomes: x(t) = (1-alpha)*x(t-1) + alpha*f(W*x(t-1) + W_in*u(t)). Enables learning very slow dynamics. Can replay learned systems at different speeds.
- Relevance: Leaky integration is essential for time series with multiple timescales. Must include in irithyll ESN.

**Sussillo & Abbott (2009) -- "Generating Coherent Patterns of Activity from Chaotic Neural Networks"**
- Authors: David Sussillo, L.F. Abbott
- Year: 2009 | Neuron | Citations: ~1,000
- Key findings: FORCE learning -- a method to train recurrent networks (including reservoirs) in real-time using RLS. Unlike standard ESN training (batch ridge regression on collected states), FORCE updates readout weights on every time step. Uses RLS with an inverse correlation matrix P that is updated recursively. Enables real-time learning even in chaotic networks. The error is guaranteed to decrease with each update.
- Relevance: FORCE = online RLS applied to reservoir readout. irithyll's RLS already does this. FORCE learning is literally "drive reservoir + run RLS on reservoir states" -- we already have the readout part.

**Jaeger (2014) -- "Conceptors: an easy introduction"**
- Authors: Herbert Jaeger
- Year: 2014 | arXiv:1406.2671
- Key findings: Conceptors are soft projection matrices C that capture the "concept" of a particular input pattern in the reservoir state space. C = R(R + alpha^-2 * I)^-1 where R is the state correlation matrix. Conceptors can be combined using Boolean logic (AND, OR, NOT). Enables: storing multiple patterns, switching between regimes, interpolating between patterns, filtering noise.
- Relevance: Directly applicable to regime switching in financial markets. Different market regimes (trending, ranging, volatile) could have different conceptors. Could be a v10 feature.

**Gallicchio & Micheli (2017/2019) -- "Deep Echo State Networks"**
- Authors: Claudio Gallicchio, Alessio Micheli
- Year: 2017 | Cognitive Computation | Citations: 160
- Key findings: Stacked reservoir layers where the output of one reservoir feeds into the next. Each layer captures different temporal features. Deep reservoirs achieve better representation than single-layer reservoirs. Echo state property extends to deep architectures with proper spectral radius conditions per layer.
- Relevance: Future extension -- deep ESN could be irithyll v10+. For v9, single-layer is sufficient.

**Rodan & Tino (2010) -- "Minimum Complexity Echo State Network"**
- Authors: Ali Rodan, Peter Tino
- Year: 2010 | IEEE Trans Neural Networks | Citations: 695
- Key findings: Simple cycle/ring reservoir topology (each neuron connected to next in a ring) achieves comparable performance to random reservoirs with far fewer parameters. Deterministic construction. All linear ESNs with simple loop reservoir have the same memory capacity, arbitrarily converging to optimal value.
- Relevance: CRITICAL for no_std. Deterministic, reproducible, minimal memory. A cycle reservoir with N neurons needs only N weights (one per edge) instead of N^2.

**Martinuzzi (2025) -- "Minimal Deterministic Echo State Networks Outperform Random Reservoirs in Learning Chaotic Dynamics"**
- Authors: Francesco Martinuzzi
- Year: 2025 | Chaos | Citations: 2
- Key findings: ESNs with deterministic topologies (cycle/ring) outperform standard random ESNs in chaotic attractor reconstruction. Structured simplicity beats stochastic complexity. Confirms Rodan & Tino (2010) with modern benchmarks.
- Relevance: Strong validation for deterministic reservoir construction -- exactly what we want for reproducibility and no_std.

**Viehweg et al. (2025) -- "Deterministic reservoir computing for chaotic time series prediction"**
- Authors: Johannes Viehweg, Constanze Poll, Patrick Mader
- Year: 2025 | Scientific Reports | Citations: 4
- Key findings: Fully deterministic reservoir outperforms classical ESNs by up to 99.99% using the Lobachevsky function as activation (alternative to tanh). Novel deterministic construction eliminates sensitivity to random seed.
- Relevance: Confirms deterministic construction is viable and superior. The Lobachevsky function is an interesting alternative activation to explore.

### 1.5 Quantized/Integer ESN

**Kleyko et al. (2020) -- "Integer Echo State Networks: Efficient Reservoir Computing for Digital Hardware"**
- Authors: Denis Kleyko, Edward Paxon Frady, Mansour Kheffache, Evgeny Osipov
- Year: 2020 | IEEE TNNLS | Citations: 33
- Key findings: Reservoir weights can be quantized to n-bit integers (n<8 typically sufficient). Int8 implementation is 3.9x faster than float32 ESN. Training time for convergence is shorter. Based on hyperdimensional computing mathematics. Minimal accuracy loss.
- Relevance: CRITICAL for embedded/no_std targets. Reservoir weights can be i8, only the readout needs f32/f64.

### 1.6 Financial Time Series

**Wang et al. (2021) -- "Stock market index prediction based on reservoir computing models"**
- Authors: Wei-Jia Wang, Yong Tang, J. Xiong, Yi-Cheng Zhang
- Year: 2021 | Expert Systems with Applications | Citations: 46-58
- Key findings: RC outperformed LSTM and vanilla RNN in most stock market index prediction cases. Demonstrated RC's effectiveness for financial time series.
- Relevance: Direct validation of RC for financial applications.

**RC vs LSTM benchmark comparisons (2024):**
- RC model runtime: 1.11 seconds vs LSTM: 423.55 seconds for crude oil price prediction
- RC outperformed ARIMA, LSTM, and GRU with superior accuracy and computational efficiency
- RC combined with genetic algorithm outperformed LSTM, Transformers, Elastic-Net, and XGBoost for high-dimensional time series

### 1.7 Online Adaptation

**Steil (2007) -- "Online reservoir adaptation by intrinsic plasticity"**
- Key findings: Intrinsic plasticity (IP) adapts reservoir neurons' activation functions online to maximize information transmission. Each neuron's gain and bias are adjusted to produce a desired output distribution (typically exponential). This is an unsupervised, local learning rule that can run alongside readout training.
- Relevance: Online reservoir adaptation without backpropagation. Could be a v10 feature for concept drift handling.

**Schubert & Gros (2021) -- "Local homeostatic regulation of the spectral radius of echo-state networks"**
- Key findings: Spectral radius can be regulated online using local homeostatic mechanisms at individual neurons, without global knowledge of the network. Each neuron adjusts its gain to maintain a target activation variance.
- Relevance: Online spectral radius adaptation -- enables the reservoir to self-tune to changing input statistics.

**Yildiz, Jaeger & Kiebel (2012) -- "Re-visiting the echo state property"**
- Year: 2012 | Neural Networks | Citations: 461
- Key findings: The commonly used sufficient condition (spectral radius < 1) is neither necessary nor sufficient in general. Tighter conditions depend on the input and the specific weight matrix structure. For zero input, spectral radius < 1 guarantees ESP. For nonzero input, ESP can hold even with spectral radius > 1.
- Relevance: Important nuance for implementation. Default to spectral radius < 1 but allow user override.

---

## 2. ESN State Update Equations

### 2.1 Standard ESN (Discrete-Time)

```
x(t+1) = f(W_in * u(t+1) + W * x(t) + W_fb * y(t) + b)
y(t+1) = W_out * x(t+1)
```

Where:
- `x(t)` : N-dimensional reservoir state vector at time t
- `u(t)` : K-dimensional input vector at time t
- `y(t)` : L-dimensional output vector at time t
- `W_in` : N x K input weight matrix (fixed, random)
- `W`    : N x N reservoir weight matrix (fixed, random, scaled to desired spectral radius)
- `W_fb` : N x L output feedback matrix (optional, fixed, random)
- `W_out`: L x N output weight matrix (TRAINABLE -- the only learned weights)
- `b`    : N-dimensional bias vector (fixed)
- `f`    : element-wise activation function (typically tanh)

### 2.2 Leaky Integrator ESN (LI-ESN)

```
x_tilde(t+1) = f(W_in * u(t+1) + W * x(t) + b)
x(t+1) = (1 - alpha) * x(t) + alpha * x_tilde(t+1)
y(t+1) = W_out * x(t+1)
```

Where:
- `alpha` : leak rate in (0, 1]. alpha=1 gives standard ESN (no leaking).
- Small alpha = slow dynamics (smoothing, long memory)
- Large alpha = fast dynamics (responsive, short memory)

### 2.3 Simplified (No Feedback) -- Recommended for irithyll v9

```
x_tilde(t) = tanh(W_in * u(t) + W * x(t-1) + b)
x(t) = (1 - alpha) * x(t-1) + alpha * x_tilde(t)
y(t) = W_out * [x(t); u(t)]   // optionally concatenate input to state
```

Note: Output feedback (W_fb) is omitted for simplicity and stability. It can cause instability during free-running prediction and is not needed for the streaming one-step-ahead prediction use case.

### 2.4 State Update Cost

For a reservoir of size N with K inputs:
- Dense W: `O(N^2 + N*K)` per time step (matrix-vector multiply)
- Sparse W with nnz non-zeros: `O(nnz + N*K)` per time step
- Cycle/ring topology: `O(N + N*K)` = `O(N*K)` per time step (only N edges)
- With fixed fanout c per neuron: `O(c*N + N*K)` per time step -- linear in N

---

## 3. RLS Readout Update Equations

irithyll already implements RLS in `src/learners/rls.rs`. The equations match what is needed for ESN readout training:

### 3.1 Standard RLS (Sherman-Morrison)

At each time step, given reservoir state x(t) as the "feature vector" and target y_target(t):

```
// Prediction error
e(t) = y_target(t) - W_out * x(t)

// Gain vector (P is the inverse correlation matrix, lambda is forgetting factor)
k(t) = P(t-1) * x(t) / (lambda + x(t)^T * P(t-1) * x(t))

// Update output weights
W_out(t) = W_out(t-1) + e(t) * k(t)^T

// Update inverse correlation matrix
P(t) = (P(t-1) - k(t) * x(t)^T * P(t-1)) / lambda
```

Where:
- `P(t)` : N x N inverse correlation matrix (initialized to delta * I)
- `lambda`: forgetting factor in (0, 1]. lambda=1 means no forgetting.
- `k(t)` : N-dimensional gain vector
- `delta` : initial uncertainty (typically 100.0)

### 3.2 Multi-Output Extension

For L outputs, maintain L separate weight vectors and a single shared P matrix:
```
for each output l = 1..L:
    e_l(t) = y_target_l(t) - w_l^T * x(t)
    w_l(t) = w_l(t-1) + e_l(t) * k(t)
```

The P matrix and gain vector k(t) are shared across outputs since they depend only on x(t).

### 3.3 irithyll Alignment

irithyll's `RecursiveLeastSquares`:
- Has forgetting factor (lambda) -- CHECK
- Has P matrix (inverse covariance) -- CHECK
- Has Sherman-Morrison update -- CHECK
- Has lazy initialization -- CHECK
- Has prediction confidence intervals -- BONUS

For ESN readout, we feed `x(t)` (reservoir state) as the feature vector to RLS. The existing RLS implementation works out of the box.

---

## 4. FORCE Learning Equations

FORCE learning (Sussillo & Abbott 2009) is essentially RLS applied to the readout of a reservoir network, with the key insight that it works even when the reservoir is chaotic.

### 4.1 Network Dynamics

```
// Reservoir update (same as standard ESN, but often with output feedback)
x(t+1) = tanh(W * x(t) + W_in * u(t) + W_fb * z(t))

// Readout
z(t) = W_out^T * x(t)
```

### 4.2 Weight Update (RLS)

```
// Error
e(t) = z(t) - y_target(t)

// Gain
k(t) = P(t-1) * x(t) / (1 + x(t)^T * P(t-1) * x(t))

// Weight update
W_out(t) = W_out(t-1) - e(t) * k(t)

// Inverse correlation matrix update
P(t) = P(t-1) - k(t) * x(t)^T * P(t-1)
```

Note: FORCE uses lambda=1 (no forgetting) and updates happen at every time step during training.

### 4.3 Difference from Standard ESN Training

| Aspect | Standard ESN | FORCE Learning |
|--------|-------------|----------------|
| Training | Batch (collect states, then solve) | Online (update every step) |
| Method | Ridge regression (Tikhonov) | Recursive Least Squares |
| Reservoir | Usually non-chaotic (spectral radius < 1) | Can be chaotic (spectral radius > 1) |
| Feedback | Optional | Typically uses output feedback |
| irithyll | Collect states -> RLS batch | RLS.train_one() per step |

### 4.4 Recommendation for irithyll

Use standard RLS readout (already implemented). FORCE learning is a strict superset -- it adds output feedback which complicates implementation and can cause instability. For v9, standard online RLS readout is sufficient and simpler.

---

## 5. Reservoir Initialization Algorithm

### 5.1 Standard Random Sparse Initialization

```
function initialize_reservoir(N, K, spectral_radius, sparsity, input_scaling, seed):
    rng = PRNG(seed)

    // 1. Generate sparse reservoir weight matrix W (N x N)
    W = zeros(N, N)
    for each (i, j) in N x N:
        if rng.random() < sparsity:   // sparsity = fraction of non-zero connections
            W[i][j] = rng.normal(0, 1)  // or rng.uniform(-1, 1)

    // 2. Compute spectral radius of W
    rho = max_eigenvalue_magnitude(W)

    // 3. Scale W to desired spectral radius
    W = W * (spectral_radius / rho)

    // 4. Generate input weight matrix W_in (N x K)
    W_in = zeros(N, K)
    for each (i, j) in N x K:
        W_in[i][j] = rng.uniform(-input_scaling, input_scaling)

    // 5. Generate bias vector b (N)
    b = zeros(N)  // or small random values

    // 6. Initialize reservoir state
    x = zeros(N)

    return (W, W_in, b, x)
```

### 5.2 Deterministic Cycle/Ring Reservoir (Recommended for no_std)

From Rodan & Tino (2010) and Martinuzzi (2025):

```
function initialize_cycle_reservoir(N, K, spectral_radius, input_scaling, seed):
    rng = PRNG(seed)

    // 1. Cycle reservoir: each neuron i connects to neuron (i+1) % N
    // Only N weights instead of N^2
    w_cycle = array of N values, each = spectral_radius
    // (For a cycle, the spectral radius equals the absolute value of the weight)
    // Alternative: random signs
    for i in 0..N:
        w_cycle[i] = spectral_radius * (if rng.random() < 0.5 then 1.0 else -1.0)

    // 2. Input weights (N x K)
    W_in = zeros(N, K)
    for each (i, j) in N x K:
        W_in[i][j] = rng.uniform(-input_scaling, input_scaling)
    // Alternative: each neuron samples ONE input randomly (sparse input)

    // 3. Bias
    b = zeros(N)

    // 4. State
    x = zeros(N)

    return (w_cycle, W_in, b, x)
```

Cycle state update (replaces dense matrix-vector multiply):
```
x_tilde[0] = tanh(w_cycle[N-1] * x[N-1] + W_in[0] * u + b[0])
for i in 1..N:
    x_tilde[i] = tanh(w_cycle[i-1] * x[i-1] + W_in[i] * u + b[i])
```

This is O(N*K) instead of O(N^2).

### 5.3 Spectral Radius Computation

For a general sparse matrix, computing the spectral radius requires finding the largest eigenvalue magnitude. For the no_std case, use power iteration:

```
function spectral_radius(W, N, max_iter=100, tol=1e-6):
    v = random_unit_vector(N)
    lambda_old = 0.0
    for iter in 0..max_iter:
        w = W * v           // matrix-vector multiply
        lambda = ||w||      // L2 norm
        v = w / lambda      // normalize
        if |lambda - lambda_old| < tol:
            break
        lambda_old = lambda
    return lambda
```

For cycle reservoir: spectral radius = |w| (the single weight magnitude). No eigenvalue computation needed.

---

## 6. Next Generation RC Equations

### 6.1 Feature Vector Construction

Given d-dimensional input time series X(t) = [x_1(t), x_2(t), ..., x_d(t)]:

**Linear features (time-delay embedding):**
```
O_lin(t) = [X(t), X(t-s), X(t-2s), ..., X(t-(k-1)*s)]
```
- k = number of delays (typically 2-4)
- s = skip between delays (typically 1-5)
- O_lin has d*k components

**Nonlinear features (polynomial):**
```
O_nonlin(t) = unique_monomials(O_lin(t), degree=p)
```
- For p=2: all unique products x_i * x_j where i <= j
- Number of degree-2 monomials: dk*(dk+1)/2
- For p=3: all unique products x_i * x_j * x_k where i <= j <= k

**Total feature vector:**
```
O_total(t) = [c, O_lin(t), O_nonlin(t)]
```
- c = constant (bias term, typically 1.0)
- Total size: 1 + dk + C(dk+p-1, p) where C is "choose"

### 6.2 Prediction

```
X(t+1) = X(t) + W_out * O_total(t)    // Euler-like: learn the delta
```
or equivalently:
```
X(t+1) = W_out * O_total(t)            // Direct prediction
```

The delta form (Eq. 11 in Gauthier 2021) is preferred for forecasting as it lets the NG-RC focus on the correction term.

### 6.3 Training (Batch, Tikhonov Regularization)

```
W_out = Y_d * O_total^T * (O_total * O_total^T + alpha * I)^(-1)
```

Or equivalently using the normal equations:
```
W_out = (O_total^T * O_total + alpha * I)^(-1) * O_total^T * Y_d
```

Where alpha is the ridge/Tikhonov regularization parameter.

### 6.4 Online Training with RLS

Instead of batch ridge regression, use RLS (already in irithyll):
```
At each time step t:
    features = O_total(t)
    target = X(t+1) - X(t)   // or X(t+1) for direct prediction
    rls.train_one(features, target)
```

### 6.5 Key Advantages over Traditional RC

| Aspect | Traditional RC | NG-RC |
|--------|---------------|-------|
| Random matrices | Yes (W, W_in) | None |
| Metaparameters | ~7 (N, spectral_radius, sparsity, input_scaling, leak_rate, alpha, seed) | ~3 (k, s, p) |
| Warm-up period | 10^3 - 10^5 steps | s*k steps (e.g., 2) |
| Training data needed | Large | Very small (400 points) |
| Training time | O(N^2 * T) for state collection + O(N^3) for regression | O(d_feat^2 * T) |
| Deterministic | No (random seed dependent) | Yes |
| Feature vector size | N (reservoir size, 100-1000) | dk + C(dk+p-1,p) (typically 20-100) |

### 6.6 irithyll Alignment

irithyll already has:
- `PolynomialFeatures` in `src/preprocessing/polynomial.rs` (degree-2)
- `RecursiveLeastSquares` in `src/learners/rls.rs` (with forgetting)

An NG-RC can be built by:
1. Maintaining a delay buffer of last k*s observations
2. Building O_lin from the buffer
3. Applying PolynomialFeatures to get O_nonlin
4. Feeding [c, O_lin, O_nonlin] as features to RLS

This requires NO new math -- just plumbing existing components.

---

## 7. Pseudocode

### 7.1 ESN: Reservoir Init

```rust
struct EchoStateNetwork {
    // Reservoir
    w_reservoir: Vec<f64>,   // N*N (dense) or N (cycle) or CSR sparse
    w_input: Vec<f64>,       // N*K input weights
    bias: Vec<f64>,          // N bias values
    state: Vec<f64>,         // N reservoir state

    // Parameters
    n_reservoir: usize,      // N
    n_inputs: usize,         // K
    leak_rate: f64,          // alpha in (0, 1]
    spectral_radius: f64,
    input_scaling: f64,

    // Readout
    readout: RecursiveLeastSquares,  // existing irithyll RLS

    // Topology
    topology: ReservoirTopology,     // Dense, Sparse, Cycle
}

enum ReservoirTopology {
    Dense,                           // N*N matrix
    Sparse { density: f64 },         // CSR format
    Cycle,                           // N weights (ring)
}
```

### 7.2 ESN: State Update

```rust
fn update_state(&mut self, input: &[f64]) {
    let n = self.n_reservoir;
    let mut x_tilde = vec![0.0; n];

    match self.topology {
        Cycle => {
            // Cycle: neuron i receives from neuron (i-1) mod N
            for i in 0..n {
                let prev = if i == 0 { n - 1 } else { i - 1 };
                let mut sum = self.w_reservoir[prev] * self.state[prev];
                // Add input contribution
                for (j, &u_j) in input.iter().enumerate() {
                    sum += self.w_input[i * self.n_inputs + j] * u_j;
                }
                sum += self.bias[i];
                x_tilde[i] = sum.tanh();
            }
        }
        Dense => {
            // Dense: full matrix-vector multiply
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += self.w_reservoir[i * n + j] * self.state[j];
                }
                for (j, &u_j) in input.iter().enumerate() {
                    sum += self.w_input[i * self.n_inputs + j] * u_j;
                }
                sum += self.bias[i];
                x_tilde[i] = sum.tanh();
            }
        }
        // ... Sparse similar with CSR iteration
    }

    // Leaky integration
    for i in 0..n {
        self.state[i] = (1.0 - self.leak_rate) * self.state[i]
                       + self.leak_rate * x_tilde[i];
    }
}
```

### 7.3 ESN: Predict

```rust
fn predict(&self) -> f64 {
    self.readout.predict(&self.state)
}
```

### 7.4 ESN: Train (Online)

```rust
fn train_one(&mut self, input: &[f64], target: f64) {
    // 1. Update reservoir state with new input
    self.update_state(input);

    // 2. Train readout on current state
    self.readout.train_one(&self.state, target, 1.0);
}
```

### 7.5 NG-RC: Full Pipeline

```rust
struct NextGenRC {
    // Delay buffer
    buffer: VecDeque<Vec<f64>>,  // stores last k*s observations
    k: usize,                    // number of delays
    s: usize,                    // skip between delays
    p: usize,                    // polynomial degree (2 or 3)
    d: usize,                    // input dimension

    // Readout
    readout: RecursiveLeastSquares,

    // State
    last_prediction: Option<Vec<f64>>,
}

fn build_features(&self) -> Vec<f64> {
    let mut features = Vec::new();

    // Constant term
    features.push(1.0);

    // O_lin: current + delayed observations
    let mut o_lin = Vec::new();
    for delay_idx in 0..self.k {
        let t_offset = delay_idx * self.s;
        if let Some(obs) = self.buffer.get(self.buffer.len() - 1 - t_offset) {
            o_lin.extend_from_slice(obs);
        }
    }
    features.extend_from_slice(&o_lin);

    // O_nonlin: polynomial monomials of o_lin
    let dk = o_lin.len();
    if self.p >= 2 {
        for i in 0..dk {
            for j in i..dk {
                features.push(o_lin[i] * o_lin[j]);
            }
        }
    }
    if self.p >= 3 {
        for i in 0..dk {
            for j in i..dk {
                for k in j..dk {
                    features.push(o_lin[i] * o_lin[j] * o_lin[k]);
                }
            }
        }
    }

    features
}

fn train_one(&mut self, observation: &[f64], target: &[f64]) {
    self.buffer.push_back(observation.to_vec());
    if self.buffer.len() > self.k * self.s + 1 {
        self.buffer.pop_front();
    }

    if self.buffer.len() >= self.k * self.s + 1 {
        let features = self.build_features();
        // For each target dimension
        for (l, &t) in target.iter().enumerate() {
            self.readout[l].train_one(&features, t, 1.0);
        }
    }
}
```

---

## 8. Hyperparameter Guide

### 8.1 ESN Hyperparameters

| Parameter | Symbol | Typical Range | Effect | Recommendation |
|-----------|--------|---------------|--------|----------------|
| Reservoir size | N | 50 - 5000 | More neurons = more capacity but more compute. 10x the number of independent signals to learn. | Start with 100-500 for time series |
| Spectral radius | rho | 0.1 - 1.5 | Controls memory length. Higher = longer memory but risk of instability. | 0.9 for general use. 0.99 for long-memory tasks. |
| Leak rate | alpha | 0.01 - 1.0 | Controls speed of dynamics. Small = slow, smooth. Large = fast, responsive. | 0.3 for smooth series. 1.0 for fast-changing. |
| Input scaling | sigma_in | 0.01 - 10.0 | Controls nonlinearity. Small = nearly linear. Large = saturated tanh. | 0.1-1.0. Start with 1.0. |
| Sparsity | density | 0.01 - 0.2 | Fraction of non-zero reservoir connections. | 0.1 (10%) for large reservoirs. Use cycle topology instead. |
| Forgetting factor | lambda | 0.99 - 1.0 | RLS forgetting for readout. Lower = adapts faster, forgets more. | 0.999 for stationary. 0.99 for non-stationary. |
| Ridge parameter | alpha_ridge | 1e-8 - 1e-2 | Regularization for batch readout training. | 1e-6 for noise-free. 1e-3 for noisy data. |
| Bias scaling | sigma_b | 0.0 - 1.0 | Scale of random bias values. | 0.0 (no bias) or 0.1 |

### 8.2 NG-RC Hyperparameters

| Parameter | Symbol | Typical Range | Effect | Recommendation |
|-----------|--------|---------------|--------|----------------|
| Number of delays | k | 2 - 10 | Number of time-delay observations. More = richer features but larger feature vector. | 2-4 |
| Skip | s | 1 - 10 | Steps between consecutive delays. Larger = captures slower dynamics. | 1 for fast dynamics. 5 for slow. |
| Polynomial degree | p | 2 - 3 | Degree of polynomial features. 2 = quadratic. 3 = cubic. | 2 for most systems. 3 for odd-symmetry systems. |
| Ridge parameter | alpha | 1e-8 - 1e-1 | Regularization strength. | 2.5e-6 (Lorenz63 optimal from Gauthier 2021) |
| dt | - | task-dependent | Sampling interval. Must be small enough to capture dynamics. | Match data rate |

### 8.3 Feature Vector Sizes

For d-dimensional input, k delays, polynomial degree p:
- Linear features: d*k
- Degree-2 monomials: dk*(dk+1)/2
- Degree-3 monomials: dk*(dk+1)*(dk+2)/6
- Total (p=2): 1 + dk + dk*(dk+1)/2

Examples:
- d=3, k=2, p=2: 1 + 6 + 21 = 28 features
- d=3, k=2, p=3: 6 + 56 = 62 features (no constant for odd-symmetry)
- d=3, k=4, p=2: 1 + 12 + 78 = 91 features
- d=10, k=2, p=2: 1 + 20 + 210 = 231 features

### 8.4 Computational Cost Comparison

| Model | Per-sample cost | Memory | Training |
|-------|----------------|--------|----------|
| ESN (dense, N=100) | O(N^2) = 10,000 FLOPs | O(N^2) = 40KB | O(N^2) per step (RLS) |
| ESN (cycle, N=100) | O(N*K) ~= 300 FLOPs | O(N*K) = 3.2KB | O(N^2) per step (RLS) |
| ESN (dense, N=1000) | O(N^2) = 1M FLOPs | O(N^2) = 4MB | O(N^2) per step (RLS) |
| ESN (cycle, N=1000) | O(N*K) ~= 3,000 FLOPs | O(N*K) = 32KB | O(N^2) per step (RLS) |
| NG-RC (d=3, k=2, p=2) | O(F^2) = 784 FLOPs | O(F^2) = 6.3KB | O(F^2) per step (RLS) |
| NG-RC (d=10, k=2, p=2) | O(F^2) = 53,361 FLOPs | O(F^2) = 427KB | O(F^2) per step (RLS) |

Where F = total feature vector size. The RLS P matrix is F x F.

Key insight: NG-RC feature vector is typically 20-100, giving P matrix of 400-10,000 elements. ESN with cycle reservoir has state size N (100-1000) giving P matrix of 10,000-1,000,000 elements. **For large input dimensions (d>5), NG-RC P matrix can be larger than cycle ESN P matrix.**

---

## 9. Architectural Recommendations for irithyll v9

### 9.1 Two-Track Strategy

Implement BOTH Echo State Networks and Next-Generation RC as separate modules. They serve different use cases:

**Track A: Echo State Network (`EchoStateNetwork`)**
- Best for: high-dimensional inputs, long-memory tasks, when reservoir dynamics are beneficial
- Architecture: Leaky integrator ESN with cycle reservoir topology (default) + optional dense/sparse
- Readout: irithyll's existing RLS with forgetting factor
- no_std compatible: cycle topology + RLS requires only Vec<f64> operations + tanh

**Track B: Next Generation RC (`NextGenRC`)**
- Best for: low-dimensional inputs (d < 10), short training data, deterministic behavior, interpretability
- Architecture: Time-delay embedding + polynomial features + RLS
- Readout: irithyll's existing RLS
- no_std compatible: delay buffer + polynomial expansion + RLS
- Can potentially be built from existing `PolynomialFeatures` + `RecursiveLeastSquares`

### 9.2 Module Structure

```
src/
  reservoir/
    mod.rs              // Re-exports, ReservoirConfig
    esn.rs              // EchoStateNetwork struct
    topology.rs         // Cycle, Dense, Sparse reservoir topologies
    ngrc.rs             // NextGenRC struct
    delay_buffer.rs     // Circular delay buffer for NG-RC
    activation.rs       // tanh, Lobachevsky, custom activations
```

### 9.3 Core Traits

```rust
/// A streaming reservoir computer.
pub trait ReservoirComputer {
    /// Feed one input sample, update internal state.
    fn update(&mut self, input: &[f64]);

    /// Get current feature vector for readout.
    fn features(&self) -> &[f64];

    /// Feed input and predict output (update + readout.predict).
    fn predict(&mut self, input: &[f64]) -> f64;

    /// Feed input, observe target, update readout (update + readout.train_one).
    fn train_one(&mut self, input: &[f64], target: f64);

    /// Reset internal state (reservoir state or delay buffer).
    fn reset(&mut self);

    /// Number of features produced by the reservoir.
    fn n_features(&self) -> usize;
}
```

### 9.4 What Already Exists in irithyll (Reuse)

| Component | Existing | Location |
|-----------|----------|----------|
| RLS readout | YES | `src/learners/rls.rs` - RecursiveLeastSquares |
| Polynomial features | YES (degree-2) | `src/preprocessing/polynomial.rs` |
| Forgetting factor | YES | RLS has lambda parameter |
| Streaming learner trait | YES | `src/learner.rs` - StreamingLearner |
| Preprocessing trait | YES | `src/pipeline.rs` - StreamingPreprocessor |
| Multi-target | YES | `src/ensemble/multi_target.rs` |
| EWMA metrics | YES | `src/metrics/ewma.rs` |
| Concept drift detection | YES | `src/drift/mod.rs` |

### 9.5 Minimum Viable ESN (Streaming)

Parameters for a minimal but functional ESN:
- Reservoir size: 100 (cycle topology)
- Spectral radius: 0.9
- Leak rate: 0.3
- Input scaling: 1.0
- No bias
- No output feedback
- RLS readout with lambda=0.999
- Total memory: ~100KB (P matrix dominates at 100*100*8 = 80KB)

### 9.6 Concept Drift Handling

Three approaches, from simplest to most complex:

1. **RLS forgetting factor** (already have): Set lambda < 1.0 (e.g., 0.995). Readout naturally adapts. Cheapest option.

2. **Reservoir reset**: When drift detected (using irithyll's drift detectors), reset reservoir state to zeros and reset P matrix. Forces fresh start.

3. **Conceptors** (future v10): Learn a conceptor matrix C for each regime. When regime changes, switch conceptor. Requires detecting the regime change.

### 9.7 no_std Compatibility

Both ESN and NG-RC can be no_std:
- Core operations: matrix-vector multiply, tanh, dot product, RLS update
- No heap allocation needed if sizes are fixed at compile time (const generics)
- Cycle topology ESN needs only: N f64s (state) + N f64s (weights) + N*K f64s (input weights) + N^2 f64s (P matrix)
- tanh can be approximated with a polynomial for hard-float-free targets
- Integer reservoir (Kleyko 2020): reservoir weights can be i8, only readout needs f64

### 9.8 Activation Function Options

| Function | Formula | Properties | no_std |
|----------|---------|------------|--------|
| tanh | tanh(x) | Standard. Bounded [-1,1]. Smooth. | libm or polynomial approx |
| Lobachevsky | L(x) = integral of (sin(t)/t)^n from 0 to x | Deterministic RC paper showed 99.99% improvement. | Complex to implement |
| Sigmoid | 1/(1+exp(-x)) | Bounded [0,1]. Equivalent to shifted tanh. | libm |
| Identity | x | Linear reservoir (for NG-RC theoretical equivalence). | Trivial |

Recommendation: Use tanh as default. It is the universal standard for ESN and is well-supported even in no_std via libm.

---

## 10. Open Questions and Risks

### 10.1 Open Questions

1. **ESN vs NG-RC: which first?** NG-RC is simpler (no random matrices, fewer parameters) and can be built almost entirely from existing irithyll components. ESN provides richer dynamics for complex tasks. Recommendation: **NG-RC first** as it validates the architecture with minimal new code, then ESN.

2. **Multi-output RLS**: irithyll's current RLS is single-output. For multi-dimensional time series (e.g., OHLCV), we need either L separate RLS instances (sharing state but independent P matrices) or a multi-output RLS with a shared P matrix. The shared-P approach is more memory-efficient (one N*N P matrix instead of L). Worth implementing.

3. **Optimal reservoir size**: No closed-form answer. Rule of thumb is 10x the number of independent signals. For financial orderflow with 20 streams, this suggests N=200 minimum. Need empirical testing.

4. **Spectral radius > 1**: Yildiz et al. (2012) showed ESP can hold with spectral radius > 1 for certain inputs. For financial data which is always nonzero, we might be able to use spectral radius slightly above 1 for better memory. Needs careful testing.

5. **Polynomial degree for NG-RC**: Gauthier used p=2 for symmetric and p=3 for odd-symmetry systems. Financial time series have no known symmetry. Start with p=2 and benchmark.

6. **Warm-up period**: ESN needs ~1000 steps to "wash out" initial state. For 1-second candles, this is 17 minutes. For tick-by-tick data, this is near-instantaneous. NG-RC needs only k*s steps (typically 2-20).

7. **Tanh approximation for embedded**: For Cortex-M targets without FPU, tanh needs a polynomial approximation. A degree-7 Chebyshev approximation gives ~1e-5 accuracy. Already solved in irithyll-core's embedded support.

### 10.2 Risks

1. **Financial time series are not chaotic systems**: Most RC benchmarks use Lorenz, Mackey-Glass, etc. Financial data is noisy, non-stationary, and partially stochastic. RC's advantages over LSTM may not transfer directly. Mitigation: RLS forgetting factor handles non-stationarity; irithyll's drift detection can trigger resets.

2. **Feature vector explosion in NG-RC**: For d=20 (20 orderflow streams), k=2, p=2: feature vector is 1 + 40 + 820 = 861. P matrix is 861^2 * 8 = 5.9MB. This is large. Mitigation: reduce k or use only a subset of inputs. Or use ESN with cycle reservoir instead.

3. **Memory for P matrix**: The RLS P matrix is N^2 (ESN) or F^2 (NG-RC) floats. For N=1000, that is 8MB. For embedded targets, this may be prohibitive. Mitigation: use smaller reservoirs (N=50-100) or use batch training instead of online RLS for embedded.

4. **No established Rust reservoir computing crates**: We are building from scratch. No existing Rust implementations to reference. Mitigation: the math is simple (matrix-vector multiply + tanh + RLS). Python ReservoirPy can serve as a reference implementation.

5. **Reproducibility across platforms**: Floating-point arithmetic varies across architectures. Deterministic cycle reservoirs help, but tanh implementations may differ. Mitigation: provide integer reservoir option for bit-exact reproducibility.

### 10.3 Implementation Priority

Phase 1 (v9.0): NG-RC
- Delay buffer
- Polynomial feature expansion (extend existing to degree p)
- Wire to existing RLS
- Multi-output support
- Tests on Lorenz63 and Mackey-Glass benchmarks
- Estimated: ~500 lines of new code

Phase 2 (v9.1): ESN
- Cycle reservoir topology
- Leaky integrator state update
- Wire to existing RLS
- Dense and sparse topologies as optional features
- Reservoir initialization from seed
- Tests on same benchmarks
- Estimated: ~800 lines of new code

Phase 3 (v9.2): Advanced Features
- Deep ESN (stacked reservoirs)
- Ensemble of reservoirs
- Integer/quantized reservoir weights (i8)
- no_std support via irithyll-core
- Conceptor support (experimental)

---

## Appendix A: Key References

### Foundational
1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Report 148.
2. Jaeger, H., Haas, H. (2004). Harnessing nonlinearity: predicting chaotic systems and saving energy in wireless communication. Science, 304, 78-80.
3. Lukosevicius, M., Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. Computer Science Review, 3, 127-149.
4. Lukosevicius, M. (2012). A practical guide to applying echo state networks. Springer LNCS 7700, 659-686.

### Advanced
5. Jaeger, H. et al. (2007). Optimization and applications of echo state networks with leaky-integrator neurons. Neural Networks, 20, 335-352.
6. Sussillo, D., Abbott, L.F. (2009). Generating coherent patterns of activity from chaotic neural networks. Neuron, 63, 544-557.
7. Jaeger, H. (2014). Conceptors: an easy introduction. arXiv:1406.2671.
8. Rodan, A., Tino, P. (2010). Minimum complexity echo state network. IEEE Trans Neural Networks, 22, 131-144.

### Next Generation RC
9. Gauthier, D.J. et al. (2021). Next generation reservoir computing. Nature Communications, 12, 5564.
10. Barbosa, W.A.S., Gauthier, D.J. (2022). Learning spatiotemporal chaos using next-generation reservoir computing. Chaos, 32, 093137.

### Surveys
11. Yan, M. et al. (2024). Emerging opportunities and challenges for the future of reservoir computing. Nature Communications, 15, 2056.
12. Tanaka, G. et al. (2019). Recent advances in physical reservoir computing: A review. Neural Networks, 115, 100-123.

### Quantized/Deterministic
13. Kleyko, D. et al. (2020). Integer Echo State Networks: Efficient Reservoir Computing for Digital Hardware. IEEE TNNLS.
14. Martinuzzi, F. (2025). Minimal Deterministic Echo State Networks Outperform Random Reservoirs. Chaos.
15. Viehweg, J. et al. (2025). Deterministic reservoir computing for chaotic time series prediction. Scientific Reports.

### Echo State Property
16. Yildiz, I.B., Jaeger, H., Kiebel, S.J. (2012). Re-visiting the echo state property. Neural Networks, 35, 1-9.
17. Caluwaerts, K. et al. (2013). The spectral radius remains a valid indicator of the echo state property for large reservoirs. IJCNN.

### Financial
18. Wang, W.J. et al. (2021). Stock market index prediction based on reservoir computing models. Expert Systems with Applications.

### Deep ESN
19. Gallicchio, C., Micheli, A. (2017). Echo state property of deep reservoir computing networks. Cognitive Computation.

### Online Adaptation
20. Steil, J.J. (2007). Online reservoir adaptation by intrinsic plasticity. Neural Networks.
21. Schubert, F., Gros, C. (2021). Local homeostatic regulation of the spectral radius of echo-state networks. Frontiers Comp Neurosci.

---

## Appendix B: Code Available

- NG-RC reference implementation (Python): https://github.com/quantinfo/ng-rc-paper-code
- ReservoirPy (Python, comprehensive): https://github.com/reservoirpy/reservoirpy
- Awesome Reservoir Computing (curated list): https://github.com/reservoirpy/awesome-reservoir-computing
- Scholarpedia ESN article (equations reference): http://www.scholarpedia.org/article/Echo_state_network

---

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| ESN | Echo State Network -- reservoir with tanh neurons + linear readout |
| ESP | Echo State Property -- initial state washes out, state depends only on input history |
| RC | Reservoir Computing -- general framework encompassing ESN, LSM, etc. |
| LSM | Liquid State Machine -- spiking neuron variant of RC |
| NG-RC | Next Generation Reservoir Computing -- polynomial features of delayed inputs |
| NVAR | Nonlinear Vector Autoregression -- mathematical framework equivalent to NG-RC |
| RLS | Recursive Least Squares -- online linear regression with O(d^2) updates |
| FORCE | First-Order Reduced and Controlled Error -- RLS-based online training for recurrent nets |
| LI-ESN | Leaky Integrator ESN -- adds exponential smoothing to reservoir dynamics |
| Spectral radius | Largest absolute eigenvalue of the reservoir weight matrix |
| Conceptor | Soft projection matrix capturing a pattern's "concept" in reservoir state space |
| Forgetting factor | Lambda in RLS; exponentially discounts older data for adaptation |
