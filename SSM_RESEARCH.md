# SSM/Mamba Research for irithyll v9 -- Streaming State Space Models

**Date:** 2026-03-22
**Purpose:** Comprehensive research document for implementing streaming SSMs in irithyll
**Target:** TRUE streaming (one sample at a time, online learning), no_std compatible, Rust

---

## Table of Contents

1. [Paper Summaries](#1-paper-summaries)
2. [The Exact Math](#2-the-exact-math)
3. [Pseudocode for Streaming SSM](#3-pseudocode-for-streaming-ssm)
4. [Architectural Recommendations for irithyll v9](#4-architectural-recommendations-for-irithyll-v9)
5. [Open Questions and Risks](#5-open-questions-and-risks)
6. [Comparison with Alternatives](#6-comparison-with-alternatives)

---

## 1. Paper Summaries

### 1.1 HiPPO: Recurrent Memory with Optimal Polynomial Projections
**Authors:** Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Re (2020)
**arXiv:** 2008.07669

**Key findings:** Introduces the HiPPO framework for online compression of continuous signals by projection onto polynomial bases. Given a measure family, HiPPO derives an ODE that optimally maintains a compressed representation of the input history. The key matrix variants are:
- **HiPPO-LegS** (Legendre Scaled): Sliding window, uniform weighting over recent history
- **HiPPO-LegT** (Legendre Translated): Exponential decay weighting
- **HiPPO-LagT** (Laguerre Translated): Another exponential decay variant

**Relevance to irithyll:** HiPPO provides the theoretically grounded initialization for the A matrix. For streaming, HiPPO-LegS is the most natural choice -- it maintains a compressed polynomial approximation of recent history that can be updated incrementally.

---

### 1.2 S4: Efficiently Modeling Long Sequences with Structured State Spaces
**Authors:** Albert Gu, Karan Goel, Christopher Re (2021)
**arXiv:** 2111.00396

**Key findings:** Introduced the Structured State Space sequence model (S4), which parameterizes the continuous-time state space model:
```
h'(t) = A h(t) + B x(t)
y(t) = C h(t) + D x(t)
```
Key innovation: Uses the HiPPO matrix as A, combined with a DPLR (Diagonal Plus Low-Rank) structure for efficient computation. Can be computed as a convolution during training (parallel) or as a recurrence during inference (sequential/streaming).

**Relevance to irithyll:** S4 proves SSMs can handle extremely long-range dependencies (up to 16K+ steps). The dual computation mode (convolution vs recurrence) is important context, though we only need the recurrence mode for streaming.

---

### 1.3 S4D: On the Parameterization and Initialization of Diagonal State Space Models
**Authors:** Albert Gu, Ankit Gupta, Karan Goel, Christopher Re (NeurIPS 2022)
**arXiv:** 2206.11893

**Key findings:** Simplifies S4 dramatically by showing that a fully diagonal A matrix works nearly as well as the DPLR structure. Key results:
- Diagonal restriction of S4's HiPPO matrix recovers the same kernel in the limit of infinite state dimension
- Kernel computation requires just 2 lines of code
- **S4D-Lin initialization:** A_n = -1/2 + i * pi * n (complex diagonal)
- **S4D-Inv initialization:** A_n = -1/2 + i * N/(n+1) (approximates HiPPO-LegS)
- B is initialized to all ones
- Uses Vandermonde matrix multiplication for efficient kernel computation

**Relevance to irithyll:** THIS IS THE KEY SIMPLIFICATION. A diagonal A matrix means:
- Each state dimension is independent (embarrassingly parallel)
- State update is element-wise multiply + add, not matrix multiply
- Complex arithmetic can be done with paired f32 values
- Massively simpler to implement in no_std Rust
- Memory: only N complex scalars for state, N for A, N for B, N for C per channel

---

### 1.4 Mamba: Linear-Time Sequence Modeling with Selective State Spaces
**Authors:** Albert Gu, Tri Dao (December 2023)
**arXiv:** 2312.00752 | **Citations:** 6190+

**Key findings:** The breakthrough paper. Identifies that prior SSMs (S4, S4D) have a critical weakness: time-invariant parameters (A, B, C) cannot perform content-based reasoning. Mamba makes B, C, and the step size Delta INPUT-DEPENDENT:

- **Selection mechanism:** B_t = Linear_N(x_t), C_t = Linear_N(x_t), Delta_t = softplus(Linear_1(x_t))
- **A remains fixed** (initialized as S4D diagonal, then log-parameterized)
- **Discretization:** Uses Zero-Order Hold (ZOH)
  - A_bar_t = exp(Delta_t * A)
  - B_bar_t = (exp(Delta_t * A) - I) * A^{-1} * B_t  (simplifies for diagonal A)
- **Recurrence:** h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
- **Output:** y_t = C_t^T * h_t

**Default hyperparameters:**
- d_model = 768 (model dimension)
- d_state (N) = 16 (state expansion factor)
- d_conv = 4 (local convolution width)
- expand = 2 (block expansion factor, so inner dimension = 2 * d_model)

**Architecture of a Mamba block:**
1. Input x -> Linear projection to 2 * expand * d_model (two branches: main + gate)
2. Main branch: Conv1d(d_conv) -> SiLU -> SSM -> output
3. Gate branch: SiLU activation
4. Element-wise multiply of main and gate branches
5. Linear projection back to d_model

**Relevance to irithyll:** The selective mechanism is what makes Mamba powerful for content-aware processing. For streaming:
- Delta controls the "forget rate" -- small Delta means ignore input, large Delta means reset state
- This is EXACTLY what we need for regime detection: the model learns when to forget old regimes
- The recurrence mode IS the streaming mode -- O(1) per step

---

### 1.5 Mamba-2/SSD: Transformers are SSMs
**Authors:** Tri Dao, Albert Gu (May 2024)
**arXiv:** 2405.21060 | **Citations:** 66+

**Key findings:** Shows deep connection between SSMs and attention (State Space Duality). Key architectural changes:
- **Scalar A:** A is now a scalar per head (not diagonal), simplifying further
- **Multi-head structure:** Groups state dimensions into heads, similar to multi-head attention
- **d_state = 64 or 128** (much larger than Mamba-1's 16)
- **SSD algorithm:** Can compute SSM as matrix multiplication chunks, enabling tensor core usage

**Relevance to irithyll:** For streaming, Mamba-2's scalar A per head is even simpler to implement. However, the SSD chunked algorithm is only useful for batch training, not streaming. The recurrence mode is identical in complexity to Mamba-1. The larger state size (64-128) may give better representation at modest memory cost.

---

### 1.6 Mamba-3: Improved Sequence Modeling using State Space Principles
**Authors:** Gu, Dao et al. (ICLR 2026, March 2026)
**arXiv:** 2603.15569

**Key findings:** Three axes of improvement:
1. **Exponential-trapezoidal discretization:** More expressive than ZOH. Changes 2-term recurrence to 3-term (uses both x_t and x_{t-1}). Second-order accurate approximation of the state-input integral.
2. **Complex-valued SSM:** Equivalent to real-valued SSM with data-dependent RoPE. Recovers state-tracking ability lost in previous linear models. Keeps speed of real-valued recurrence.
3. **MIMO (Multi-Input Multi-Output):** Improves model performance without increasing decode latency.

**Performance:** At 1.5B scale, comparable perplexity to Mamba-2 with HALF the state size. 2x smaller states = 2x less memory for streaming.

**Relevance to irithyll:** The trapezoidal discretization is interesting for streaming because it uses x_{t-1} (which we already have) for better accuracy. Complex-valued SSM adds expressiveness. HOWEVER, for a v9 initial implementation, starting with Mamba-1 style ZOH is simpler and proven.

---

### 1.7 Longhorn: State Space Models are Amortized Online Learners
**Authors:** Bo Liu, Rui Wang, Lemeng Wu, Yihao Feng, Peter Stone, Qian Liu (NeurIPS 2024)
**arXiv:** 2407.14207 | **Citations:** 34

**THIS IS THE MOST CRITICAL PAPER FOR OUR USE CASE.**

**Key findings:** Reframes SSMs as meta-modules for online learning. The key insight:
- The recurrent state update of an SSM can be interpreted as a proximal update step or closed-form solution to an online learning objective
- Specifically, for ONLINE ASSOCIATIVE RECALL: given key-value pairs arriving in a stream, maintain a weight matrix W that maps keys to values
- The optimal online update rule naturally produces a STABLE RECURRENT FORM without manually designed gating

**Longhorn's update rule:**
```
W_t = (1 - alpha_t) * W_{t-1} + alpha_t * v_t * k_t^T
```
Where:
- k_t, v_t are key/value projections of the input
- alpha_t is a learned gating/learning rate (analogous to Delta in Mamba)
- W_t is the state matrix (replaces h_t vector with a matrix)

**Key advantages:**
- No separate forget gate needed (saves parameters for large state)
- 1.8x improvement in sample efficiency vs Mamba
- Can extrapolate over 16x longer contexts at inference
- Near-perfect recall on associative tasks
- NATURALLY SUITED TO ONLINE LEARNING because it IS an online learner

**Relevance to irithyll:** This is the theoretical bridge we need. If we view our SSM state as an online learner:
- The forward pass IS the online update
- No separate "training" needed during streaming -- the state evolution IS learning
- For regime detection: the state matrix naturally tracks key-value associations (e.g., "when orderflow pattern X appears, regime Y follows")
- This means we can potentially skip backpropagation entirely for the SSM core, using the Longhorn-style closed-form update

---

### 1.8 S6MOD: Enhancing Online Continual Learning with Plug-and-Play SSMs
**Authors:** Sihao Liu, Yibo Yang, Xiaojie Li, David A. Clifton, Bernard Ghanem (CVPR 2025)
**arXiv:** 2412.18177

**Key findings:** Proposes a plug-and-play SSM module (S6MOD) for online continual learning:
- Integrates into existing methods to improve adaptability
- Uses Mixture of Experts (MoE) for discretization -- different discretization strategies for different input patterns
- Class-conditional routing for dynamic, uncertainty-based adjustment
- Contrastive discretization loss

**Relevance to irithyll:** Demonstrates SSMs as online continual learning modules. The MoE discretization idea is interesting for regime detection -- different regimes might benefit from different discretization parameters.

---

### 1.9 Mamba-FSCIL: Dynamic Adaptation with Selective State Space Model
**Authors:** Xiaojie Li, Yibo Yang et al. (2024)
**arXiv:** 2407.06136

**Key findings:** Applies Mamba to Few-Shot Class-Incremental Learning (FSCIL). Uses dual selective SSM projector for dynamic adaptation. Class-sensitive selective scan mechanism guides adaptation.

**Relevance to irithyll:** Shows Mamba working in continual learning settings where new classes appear with limited samples -- analogous to new market regimes appearing.

---

### 1.10 Chimera: 2D State Space Models for Multivariate Time Series
**Authors:** Ali Behrouz, Michele Santacatterina, Ramin Zabih (NeurIPS 2024)
**arXiv:** 2406.04320

**Key findings:** Extends SSMs to 2D for multivariate time series. Two SSM heads with different discretization processes. Can learn long-term progression, seasonal patterns, and dynamic autoregressive processes. Superior on ECG, speech, long/short-term forecasting, and anomaly detection benchmarks.

**Relevance to irithyll:** Directly applicable to multivariate financial data (multiple streams: price, volume, delta, etc.). The 2D approach models both temporal and cross-variate dependencies.

---

### 1.11 "Is Mamba Effective for Time Series Forecasting?"
**Authors:** Zihan Wang et al. (Neurocomputing 2024)
**DOI:** 10.1016/j.neucom.2024.129178 | **Citations:** 143

**Key findings:** Comprehensive benchmark of Mamba for time series forecasting. Results show Mamba is competitive with Transformers on many benchmarks, with significantly lower compute cost. Particularly strong on long-term dependencies. Some limitations on short-term local pattern capture.

---

### 1.12 Diagonal SSM on Loihi 2 for Efficient Streaming
**Authors:** Svea Marie Meyer et al. (NICE 2024/2025)
**arXiv:** 2409.15022 | **Citations:** 16

**Key findings:** First demonstration of SSM on neuromorphic hardware (Loihi 2) for streaming. Uses diagonal S4D variant. Achieves substantial efficiency improvement over GPU (Jetson Orin Nano). Demonstrates SSMs are naturally suited to streaming on resource-constrained hardware.

**Relevance to irithyll:** Validates that diagonal SSMs work well in streaming mode on constrained hardware. Directly analogous to our no_std embedded target.

---

### 1.13 KOSS: Kalman-Optimal Selective State Spaces
**Authors:** Lei Wang et al. (2025)
**arXiv:** 2512.16723

**Key findings:** Formulates selection mechanism as Kalman-optimal latent state uncertainty minimization. Provides theoretical grounding for WHY selectivity works -- it minimizes state estimation error.

**Relevance to irithyll:** Could inform a principled approach to the Delta/gating parameter. If we treat the state as a Kalman filter estimate, the optimal gating balances process noise vs. measurement noise.

---

### 1.14 RTRL: Scalable Real-Time Recurrent Learning
**Authors:** Khurram Javed, Haseeb Shah, Rich Sutton, Martha White (2023)
**arXiv:** 2302.05326

**Key findings:** RTRL computes exact gradients for recurrent networks online (without BPTT). Standard RTRL is O(n^3) per step for hidden size n. This paper proposes columnar-constructive networks that make RTRL O(n) per step by restricting connectivity patterns.

**Relevance to irithyll:** If we need gradient-based online training (not just the Longhorn closed-form approach), RTRL is the gold standard for streaming. For diagonal SSMs, RTRL simplifies significantly because the Jacobian of the state transition is diagonal.

---

## 2. The Exact Math

### 2.1 Continuous-Time State Space Model

The foundation of all SSMs:

```
h'(t) = A * h(t) + B * x(t)       (state equation)
y(t)  = C * h(t) + D * x(t)       (output equation)
```

Where:
- h(t) in R^N is the hidden state
- x(t) in R^1 is the input (scalar, one channel)
- y(t) in R^1 is the output
- A in R^{N x N}, B in R^{N x 1}, C in R^{1 x N}, D in R^{1 x 1}

### 2.2 Discretization (Zero-Order Hold)

Given step size Delta > 0, the ZOH discretization produces:

```
A_bar = exp(Delta * A)
B_bar = (exp(Delta * A) - I) * A^{-1} * B
```

For **diagonal A** (S4D/Mamba), this simplifies element-wise:

```
A_bar_n = exp(Delta * A_n)                          for n = 0..N-1
B_bar_n = (exp(Delta * A_n) - 1) / A_n * B_n        for n = 0..N-1
```

Where A_n is the n-th diagonal element.

### 2.3 Discrete-Time Recurrence (The Streaming Equation)

This is what we implement for streaming:

```
h_t = A_bar * h_{t-1} + B_bar * x_t     (element-wise for diagonal)
y_t = C^T * h_t                          (dot product)
```

**Per-step cost:** O(N) multiplies + O(N) adds for state update, O(N) for output. Total: O(N).

### 2.4 Mamba's Selective Mechanism (Input-Dependent Parameters)

In Mamba, B, C, and Delta are functions of the current input:

```
// Given input x_t (scalar or projected to d_inner dimension)
Delta_t = softplus(W_delta * x_t + b_delta)    // step size, shape: (1,) or (D,)
B_t     = W_B * x_t                            // input projection, shape: (N,)
C_t     = W_C * x_t                            // output projection, shape: (N,)

// A is FIXED, learned, stored as log for stability
A = -exp(log_A)                                // shape: (N,), always negative real

// Discretize with input-dependent Delta
A_bar_t = exp(Delta_t * A)                     // element-wise, shape: (N,)
B_bar_t = (A_bar_t - 1) / A * B_t             // element-wise, shape: (N,)

// State update
h_t = A_bar_t .* h_{t-1} + B_bar_t * x_t      // element-wise, shape: (N,)

// Output
y_t = real(C_t^T * h_t)                        // dot product, scalar
```

### 2.5 HiPPO/S4D Initialization for A

**S4D-Lin (recommended for simplicity):**
```
A_n = -1/2 + i * pi * n       for n = 0, 1, ..., N-1
```

**S4D-Inv (closer to HiPPO-LegS):**
```
A_n = -1/2 + i * N / (n + 1)  for n = 0, 1, ..., N-1
```

**Mamba's practical initialization (real-valued, simpler):**
```
A_n = -(n + 1)                 for n = 0, 1, ..., N-1
```
Then stored as log_A_n = log(n + 1), and A_n = -exp(log_A_n) during forward pass.

Note: Mamba uses REAL-valued A (not complex) for simplicity. The imaginary component from S4D encodes oscillatory behavior, but Mamba found that the selectivity mechanism compensates.

### 2.6 Bilinear Discretization (Alternative to ZOH)

```
A_bar = (I - Delta/2 * A)^{-1} * (I + Delta/2 * A)
B_bar = (I - Delta/2 * A)^{-1} * Delta * B
```

For diagonal A, element-wise:
```
A_bar_n = (1 + Delta/2 * A_n) / (1 - Delta/2 * A_n)
B_bar_n = Delta / (1 - Delta/2 * A_n) * B_n
```

Bilinear preserves stability (maps left-half plane to unit disk) but ZOH is more commonly used in Mamba.

### 2.7 Mamba-3 Trapezoidal Discretization (Advanced)

Changes the 2-term recurrence to 3-term:
```
h_t = A_bar * h_{t-1} + B_bar_1 * x_t + B_bar_0 * x_{t-1}
```
Where B_bar_0 and B_bar_1 are derived from the trapezoidal rule applied to the state-input integral. This gives second-order accuracy. Requires storing x_{t-1} alongside h_{t-1}.

### 2.8 Longhorn Online Update Rule

The Longhorn update is a matrix-valued state:
```
W_t = (1 - alpha_t) * W_{t-1} + alpha_t * v_t * k_t^T
```

Where:
- W_t in R^{d_v x d_k} is the state matrix
- k_t = projection(x_t), shape (d_k,)
- v_t = projection(x_t), shape (d_v,)
- alpha_t = sigmoid(w_alpha^T * x_t), scalar learning rate

Output: y_t = W_t * q_t, where q_t = projection(x_t)

This is equivalent to an SSM where:
- A = (1 - alpha_t) * I (scalar forget gate)
- The "state" is a key-value store updated via outer products

### 2.9 Cost Summary Per Sample

For a single Mamba-style SSM layer with D input channels, N state dimensions:

| Operation | FLOPs | Memory |
|-----------|-------|--------|
| Delta projection | O(D) | W_delta: D params |
| B projection | O(D*N) | W_B: D*N params |
| C projection | O(D*N) | W_C: D*N params |
| Discretize A_bar | O(D*N) | log_A: D*N params |
| Discretize B_bar | O(D*N) | (reuses above) |
| State update | O(D*N) | h: D*N state |
| Output dot product | O(D*N) | - |
| **Total per sample** | **O(D*N)** | **State: D*N, Params: ~3*D*N** |

For D=16, N=16: ~768 FLOPs per sample, ~256 floats of state, ~768 params.
For D=64, N=16: ~3072 FLOPs per sample, ~1024 floats of state, ~3072 params.

---

## 3. Pseudocode for Streaming SSM

### 3.1 Minimal Streaming SSM Forward Pass (S4D-style, non-selective)

```rust
/// Non-selective diagonal SSM -- simplest possible streaming SSM
/// Good baseline, equivalent to a fixed-parameter linear recurrence
struct DiagonalSSM {
    // Parameters (learned or fixed)
    a_real: [f32; N],       // Real part of diagonal A (negative values)
    a_imag: [f32; N],       // Imaginary part of diagonal A
    b_real: [f32; N],       // Real part of B
    b_imag: [f32; N],       // Imaginary part of B
    c_real: [f32; N],       // Real part of C
    c_imag: [f32; N],       // Imaginary part of C
    delta: f32,             // Fixed step size
    d: f32,                 // Skip connection

    // State
    h_real: [f32; N],       // Real part of hidden state
    h_imag: [f32; N],       // Imaginary part of hidden state
}

fn forward(&mut self, x: f32) -> f32 {
    // Discretize A and B using ZOH
    for n in 0..N {
        let da_r = self.delta * self.a_real[n];
        let da_i = self.delta * self.a_imag[n];

        // exp(delta * A) for complex diagonal
        // exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
        let exp_mag = da_r.exp();
        let a_bar_r = exp_mag * da_i.cos();
        let a_bar_i = exp_mag * da_i.sin();

        // B_bar = (A_bar - I) / A * B
        // For complex: (exp(dA) - 1) / A * B
        // Simplified: delta * B (first-order approx, used in practice for small delta)
        let b_bar_r = self.delta * self.b_real[n];
        let b_bar_i = self.delta * self.b_imag[n];

        // State update: h_t = A_bar * h_{t-1} + B_bar * x_t
        let h_old_r = self.h_real[n];
        let h_old_i = self.h_imag[n];

        self.h_real[n] = a_bar_r * h_old_r - a_bar_i * h_old_i + b_bar_r * x;
        self.h_imag[n] = a_bar_r * h_old_i + a_bar_i * h_old_r + b_bar_i * x;
    }

    // Output: y = Re(C^H * h) + D * x
    let mut y = self.d * x;
    for n in 0..N {
        y += self.c_real[n] * self.h_real[n] + self.c_imag[n] * self.h_imag[n];
    }
    y
}
```

### 3.2 Selective Streaming SSM Forward Pass (Mamba-style)

```rust
/// Selective SSM -- input-dependent B, C, Delta
/// This is the core of Mamba's power
struct SelectiveSSM<const D: usize, const N: usize> {
    // Fixed parameters
    log_a: [f32; N],                    // log(-A), A is always negative real in Mamba

    // Learnable projection weights
    w_delta: [[f32; D]; 1],             // Delta projection (D -> 1)
    b_delta: f32,                       // Delta bias
    w_b: [[f32; D]; N],                // B projection (D -> N)
    w_c: [[f32; D]; N],                // C projection (D -> N)
    d: [f32; D],                        // Skip connection per channel

    // State: D independent SSMs, each with N state dims
    h: [[f32; N]; D],                   // Hidden state (real-valued for Mamba)
}

fn forward(&mut self, x: &[f32; D]) -> [f32; D] {
    // 1. Compute input-dependent parameters
    let delta_raw: f32 = dot(&self.w_delta[0], x) + self.b_delta;
    let delta: f32 = softplus(delta_raw);  // softplus(x) = ln(1 + exp(x))

    let b: [f32; N] = matvec(&self.w_b, x);   // Input-to-state projection
    let c: [f32; N] = matvec(&self.w_c, x);   // State-to-output projection

    // 2. Compute A (fixed, negative real diagonal)
    let a: [f32; N] = self.log_a.map(|la| -la.exp());

    // 3. Discretize
    let a_bar: [f32; N] = a.map(|a_n| (delta * a_n).exp());
    let b_bar: [f32; N] = array_zip(a_bar, a, b, |ab, a_n, b_n| {
        (ab - 1.0) / a_n * b_n
    });

    // 4. State update + output for each channel
    let mut y = [0.0f32; D];
    for d in 0..D {
        for n in 0..N {
            self.h[d][n] = a_bar[n] * self.h[d][n] + b_bar[n] * x[d];
        }
        // Output for this channel
        y[d] = dot_n(&c, &self.h[d]) + self.d[d] * x[d];
    }
    y
}
```

### 3.3 Online Parameter Update (Longhorn-style, no backprop)

```rust
/// Online learning SSM -- the state update IS the learning
/// Based on Longhorn's amortized online learning framework
struct OnlineSSM<const D_K: usize, const D_V: usize, const D_IN: usize> {
    // Key/Value/Query projections (fixed after initial training, or slowly adapted)
    w_key: [[f32; D_IN]; D_K],
    w_value: [[f32; D_IN]; D_V],
    w_query: [[f32; D_IN]; D_K],
    w_alpha: [f32; D_IN],              // Learning rate projection

    // State: the associative memory matrix
    w_state: [[f32; D_K]; D_V],        // Online learner weight matrix
}

fn forward(&mut self, x: &[f32; D_IN]) -> [f32; D_V] {
    // 1. Project input to key, value, query
    let k: [f32; D_K] = matvec(&self.w_key, x);
    let v: [f32; D_V] = matvec(&self.w_value, x);
    let q: [f32; D_K] = matvec(&self.w_query, x);

    // 2. Compute learning rate (analogous to Delta in Mamba)
    let alpha: f32 = sigmoid(dot(&self.w_alpha, x));

    // 3. Update associative memory (this IS the online learning step)
    for i in 0..D_V {
        for j in 0..D_K {
            self.w_state[i][j] = (1.0 - alpha) * self.w_state[i][j]
                                + alpha * v[i] * k[j];
        }
    }

    // 4. Query the memory
    let y: [f32; D_V] = matvec(&self.w_state, &q);
    y
}
```

### 3.4 Online Parameter Update via Streaming SGD (for projection weights)

```rust
/// For updating the projection weights (w_delta, w_b, w_c) online
/// Uses RTRL-style gradient for the diagonal SSM
fn online_update_projections(
    ssm: &mut SelectiveSSM,
    x: &[f32; D],
    target: &[f32; D],
    lr: f32,
) {
    // For diagonal SSM, the gradient through the state is simple:
    // dh_t/dtheta = A_bar * dh_{t-1}/dtheta + dB_bar/dtheta * x_t
    // Because A is diagonal, this is element-wise

    // Forward pass (get prediction)
    let y_pred = ssm.forward(x);

    // Compute loss gradient
    let dy: [f32; D] = array_sub(y_pred, target);  // dL/dy for MSE

    // For the C projection (output weights):
    // dL/dw_c = dy * h (outer product)
    // This is the simplest gradient -- doesn't require BPTT

    // For B and Delta projections:
    // Need to maintain running gradient through state (RTRL)
    // dh_t/dw_b = A_bar * dh_{t-1}/dw_b + x_t * dx/dw_b
    // For diagonal A, this is O(N * D * |w_b|) per step

    // PRACTICAL APPROACH: Use truncated RTRL with k=1 (only current step)
    // This ignores gradient through previous states but is O(D*N) per step
    // Works well in practice for slowly-varying systems (like financial data)

    // Update C weights using current-step gradient only
    for n in 0..N {
        for d in 0..D {
            ssm.w_c[n][d] -= lr * dy[d] * ssm.h[d][n];
        }
    }
}
```

---

## 4. Architectural Recommendations for irithyll v9

### 4.1 Minimum Viable Streaming SSM (Start Here)

**Recommendation: Diagonal S4D with real-valued A (Mamba-1 style selectivity)**

```
Architecture: SelectiveSSM
A: Real diagonal, initialized as A_n = -(n+1), log-parameterized
B: Input-dependent via linear projection
C: Input-dependent via linear projection
Delta: Input-dependent via linear projection + softplus
Discretization: ZOH (simplest, proven)
State dimension N: 16 (Mamba-1 default, tiny memory footprint)
```

**Why this over alternatives:**
- Real A avoids complex arithmetic (simpler, no_std friendly)
- Diagonal A means O(N) state update (not O(N^2))
- Input-dependent B/C/Delta gives content-aware selectivity
- N=16 means only 16 floats of state per channel
- ZOH is the simplest discretization that preserves stability

### 4.2 Proposed Module Hierarchy

```
irithyll-core/
  src/
    ssm/
      mod.rs              // SSM trait definitions
      diagonal.rs         // Non-selective diagonal SSM (S4D baseline)
      selective.rs        // Selective diagonal SSM (Mamba-style)
      online.rs           // Longhorn-style online learning SSM
      discretize.rs       // ZOH, bilinear, trapezoidal discretization
      init.rs             // HiPPO, S4D-Lin, S4D-Inv, Mamba initializations
      block.rs            // Full Mamba block (projection + SSM + gating)
```

### 4.3 Key Hyperparameters and Typical Ranges

| Parameter | Description | Typical Range | irithyll Default |
|-----------|-------------|---------------|------------------|
| N (d_state) | State dimension | 4-128 | 16 |
| D (d_model) | Model/input dimension | 1-768 | User-defined |
| expand | Expansion factor | 1-4 | 2 |
| d_conv | Local conv width | 0-8 | 4 (or 0 for pure SSM) |
| Delta init | Step size initialization | 0.001-0.1 | 0.01 |
| A init | State decay rate | S4D-Lin or -(n+1) | -(n+1) |
| lr (online) | Online learning rate | 1e-5 to 1e-2 | 1e-3 |

### 4.4 Memory Budget for Streaming

For a single Mamba-style SSM layer with D=16 input channels, N=16 state dims:

```
State:        D * N = 256 floats = 1 KB (f32) or 512 bytes (f16)
Parameters:   ~3 * D * N + D + N + D*N = ~1280 floats = 5 KB
Total:        ~6 KB per layer
```

For 4 layers: ~24 KB total. Easily fits in no_std / embedded contexts.

### 4.5 Online Training Strategy

**Recommended: Hybrid approach**

1. **SSM core (state transition):** Longhorn-style closed-form update. The state evolution IS the learning. No backprop needed for the hidden state dynamics.

2. **Projection weights (W_B, W_C, W_Delta):** Two options:
   - **Frozen after pre-training:** Train on historical data, then freeze. State dynamics handle adaptation.
   - **Slow online SGD:** Use truncated RTRL (k=1) for simple gradient estimates. Update at a much slower rate than the SSM state evolves.

3. **A (decay rate):** Keep FIXED after initialization. This is the "structural" parameter of the SSM -- it defines the time-scale of memory. Don't adapt online.

4. **Output head:** If needed, use irithyll's existing online tree/ensemble methods on top of SSM features.

### 4.6 Integration with Existing irithyll

The SSM layer should produce FEATURES, not final predictions:

```
Raw input (e.g., price, volume, delta)
  -> SelectiveSSM (extracts temporal features, regime-aware)
  -> Feature vector of D dimensions
  -> irithyll GradientBoostedTree (or ensemble) for final prediction
```

This leverages irithyll's proven tree-based online learners while adding temporal state via the SSM.

### 4.7 Quantization Considerations

Research findings on SSM quantization:
- **INT8 works for weights:** <3% accuracy degradation in Mamba models up to 2.8B params
- **Activations are harder:** SSM activations have outliers that require special handling
- **State needs higher precision:** The recurrent state accumulates over time; quantization error compounds
- **Recommendation for irithyll:** Use f32 for state, allow i16/i8 for weights and projections. The state is small (N*D values) so f32 cost is negligible.

For irithyll's existing i16 quantization infrastructure:
- Projection weights (W_B, W_C, W_Delta): Can be i16 quantized
- A parameters: Should stay f32 (exponential is sensitive to precision)
- Hidden state h: Must stay f32 (recurrence amplifies quantization error)

---

## 5. Open Questions and Risks

### 5.1 Open Questions

1. **How to initialize for financial data?** HiPPO assumptions (polynomial approximation of history) may not match financial time series characteristics. May need domain-specific initialization experiments.

2. **Optimal N for time series?** N=16 works for language (Mamba-1), but financial data has different structure. May need experiments with N=4, 8, 16, 32.

3. **Complex vs real A?** Mamba uses real A, but complex A (S4D-style) gives oscillatory components that might better capture market cycles. Worth experimenting.

4. **Multi-scale time horizons?** Financial data has multiple regimes at different time scales (tick, minute, hour, day). May need multiple SSM layers with different Delta initializations or a Chimera-style multi-head approach.

5. **How to evaluate streaming SSM quality?** Need online metrics -- can't use batch train/test split. Prequential evaluation (test-then-train) is appropriate.

6. **Gradient-free online learning viability?** Longhorn's closed-form update works for associative recall. Is this sufficient for regime detection, or do we need gradient-based updates for the projections?

### 5.2 Risks

1. **Catastrophic forgetting in streaming:** The SSM state can be overwritten by new data, losing memory of past regimes. Mitigation: Multiple SSM heads with different decay rates (fast/slow memory).

2. **Numerical stability:** Repeated multiplication by A_bar can cause overflow/underflow. Mitigation: A is always negative (decaying), and exp(Delta * A) is always in (0, 1) for negative A. Log-parameterization of A prevents positive values.

3. **Non-stationarity:** Financial data is non-stationary. Fixed A may not adapt. Mitigation: The selective mechanism (input-dependent Delta) handles this -- it effectively resets the state when the model detects a regime change.

4. **No existing Rust SSM crate:** We're building from scratch. This is actually an advantage for no_std/irithyll integration but means more implementation work.

5. **Training data requirements:** SSM projections need initial training. How much historical financial data is needed? Literature suggests SSMs are sample-efficient (Longhorn shows 1.8x vs Mamba), but this needs validation for our domain.

6. **Precision under i16:** If the projection weights use i16 quantization, the softplus(Delta) computation may lose precision. Need to validate numerically.

---

## 6. Comparison with Alternatives

### 6.1 SSM vs Transformer for Streaming

| Aspect | SSM (Mamba) | Transformer |
|--------|-------------|-------------|
| Per-step cost | O(D*N) | O(L*D) where L is context window |
| Memory | O(D*N) fixed | O(L*D) growing |
| Streaming native | YES (recurrence) | NO (needs full context window) |
| Long-range deps | Excellent (HiPPO theory) | Excellent (attention) |
| Content-aware | YES (selective) | YES (attention) |
| Online training | Natural (Longhorn) | Difficult (needs BPTT over window) |
| no_std compatible | YES (simple arithmetic) | Difficult (softmax, large matrices) |
| **Winner for streaming** | **SSM** | - |

### 6.2 SSM vs Traditional RNN (LSTM/GRU)

| Aspect | SSM (Mamba) | LSTM/GRU |
|--------|-------------|----------|
| Per-step cost | O(D*N) | O(D^2) (hidden size = D typically) |
| Gradient flow | Stable (diagonal A, HiPPO) | Vanishing/exploding gradient risk |
| Long-range deps | Excellent | Moderate (despite gating) |
| Theoretical grounding | HiPPO optimal compression | Empirical gating design |
| Online training (RTRL) | O(D*N) for diagonal | O(D^3) standard |
| Parallelizable (batch) | YES (scan/convolution) | NO |
| **Winner for streaming** | **SSM** | - |

### 6.3 SSM vs irithyll's Current Trees/Ensembles

| Aspect | SSM | Gradient Boosted Trees |
|--------|-----|----------------------|
| Temporal state | Native (hidden state) | None (memoryless per sample) |
| Feature engineering needed | Minimal (learns temporal features) | Heavy (must engineer lag features) |
| Online learning | Natural | Native (irithyll's strength) |
| Interpretability | Low (black box state) | High (tree structure) |
| Compute per sample | O(D*N) | O(trees * depth) |
| no_std | Yes | Yes (irithyll proven) |
| **Recommendation** | Use as feature extractor | Use as predictor on SSM features |

### 6.4 SSM Variant Comparison for irithyll

| Variant | Complexity | Implementation | Quality | Recommendation |
|---------|-----------|----------------|---------|----------------|
| S4D (fixed params) | Simplest | ~200 LOC | Baseline | v9.0 baseline |
| Mamba-1 (selective) | Moderate | ~400 LOC | Best proven | v9.0 primary |
| Longhorn (online) | Moderate | ~350 LOC | Best for online | v9.1 addition |
| Mamba-2 (SSD) | Complex | ~600 LOC | Marginal gain | Skip for now |
| Mamba-3 (trapezoidal) | Complex | ~500 LOC | Latest SOTA | v9.2+ if needed |
| Chimera (2D) | Complex | ~700 LOC | Best for multivariate | v9.2+ for multi-stream |

---

## Appendix A: Key Source Links

### Papers
- [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Mamba-2/SSD (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060)
- [Mamba-3 (ICLR 2026)](https://openreview.net/pdf?id=HwCvaJOiCj)
- [S4D (Gu et al., NeurIPS 2022)](https://arxiv.org/abs/2206.11893)
- [HiPPO (Gu et al., 2020)](https://arxiv.org/abs/2008.07669)
- [Longhorn (Liu et al., NeurIPS 2024)](https://arxiv.org/abs/2407.14207)
- [S6MOD (Liu et al., CVPR 2025)](https://arxiv.org/abs/2412.18177)
- [Mamba-FSCIL (Li et al., 2024)](https://arxiv.org/abs/2407.06136)
- [Chimera (Behrouz et al., NeurIPS 2024)](https://arxiv.org/abs/2406.04320)
- [Is Mamba Effective for Time Series? (Wang et al., 2024)](https://doi.org/10.1016/j.neucom.2024.129178)
- [Diagonal SSM on Loihi 2 (Meyer et al., 2024)](https://arxiv.org/abs/2409.15022)
- [KOSS Kalman-Optimal SSM (Wang et al., 2025)](https://arxiv.org/abs/2512.16723)
- [Quamba SSM Quantization (2024)](https://arxiv.org/abs/2410.13229)
- [Scalable RTRL (Javed et al., 2023)](https://arxiv.org/abs/2302.05326)

### Implementations
- [Official Mamba (Python/CUDA)](https://github.com/state-spaces/mamba)
- [mamba.py (Pure PyTorch)](https://github.com/alxndrTL/mamba.py)
- [mamba-minimal (Single file)](https://github.com/johnma2006/mamba-minimal)
- [mamba3-minimal (PyTorch)](https://github.com/VikramKarLex/mamba3-minimal)
- [Longhorn (Official)](https://github.com/Cranial-XIX/longhorn)
- [S6MOD (Official)](https://github.com/MyToumaKazusa/S6MOD)

### Blog Posts and Tutorials
- [Visual Guide to Mamba (Maarten Grootendorst)](https://www.maartengrootendorst.com/blog/mamba/)
- [Mamba-2 Part I: The Model (Goomba Lab)](https://goombalab.github.io/blog/2024/mamba2-part1-model/)
- [Mamba-2 Part II: The Theory (Goomba Lab)](https://goombalab.github.io/blog/2024/mamba2-part2-theory/)
- [Mamba-2 Part III: The Algorithm (Goomba Lab)](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/)
- [HuggingFace SSM Introduction](https://huggingface.co/blog/lbourdois/get-on-the-ssm-train)
- [SSM History 2022 (HuggingFace)](https://huggingface.co/blog/lbourdois/ssm-2022)
- [Mamba SSM Theory and Implementation (TDS)](https://towardsdatascience.com/mamba-ssm-theory-and-implementation-in-keras-and-tensorflow-32d6d4b32546/)

### No Rust SSM Crates Exist
As of March 2026, no dedicated Rust crate for SSMs/Mamba was found. The closest Rust ML frameworks are:
- [Burn](https://burn.dev/) - General neural network framework (not no_std)
- [tch-rs](https://github.com/LaurentMazare/tch-rs) - PyTorch bindings (not no_std)
- irithyll would be the FIRST no_std Rust SSM implementation

---

## Appendix B: Recommended Implementation Order

### Phase 1: v9.0 -- Core Streaming SSM
1. `discretize.rs` -- ZOH and bilinear discretization functions
2. `init.rs` -- Mamba-style A initialization (log-parameterized negative reals)
3. `diagonal.rs` -- Non-selective diagonal SSM (fixed B, C, Delta) as baseline
4. `selective.rs` -- Selective diagonal SSM (Mamba-1 style input-dependent B, C, Delta)
5. Tests: verify numerical stability, compare outputs with reference Python implementation

### Phase 2: v9.1 -- Online Learning
6. `online.rs` -- Longhorn-style online learning SSM
7. Integration with existing irithyll tree learners (SSM features -> tree prediction)
8. Truncated RTRL for slow online adaptation of projection weights

### Phase 3: v9.2 -- Advanced Features
9. `block.rs` -- Full Mamba block with gating and projection layers
10. Complex-valued A option (S4D-Lin initialization)
11. Trapezoidal discretization (Mamba-3 style)
12. Multi-head/multi-scale SSM for different time horizons

### Phase 4: v9.3+ -- Domain Optimization
13. Financial-specific initialization experiments
14. Regime detection evaluation framework
15. Chimera-style 2D SSM for multivariate orderflow data
16. i16 quantized projection weights
