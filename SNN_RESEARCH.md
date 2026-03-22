# Spiking Neural Networks for irithyll v9.1: Comprehensive Research

**Date:** 2026-03-22
**Target:** irithyll v9.1.0 — streaming SNN training with no_std support
**Author:** Research compilation for Jono (evilrat420)

---

## Table of Contents

1. [Paper Summaries](#1-paper-summaries)
2. [Neuron Dynamics Equations](#2-neuron-dynamics-equations-discrete-time-lif)
3. [Online Learning Rule Equations](#3-online-learning-rule-equations-e-prop)
4. [Spike Encoding Recommendation](#4-spike-encoding-recommendation)
5. [Pseudocode](#5-pseudocode)
6. [Memory Layout Proposal](#6-memory-layout-proposal)
7. [Architectural Recommendations](#7-architectural-recommendations-for-irithyll-v9)
8. [Open Questions and Risks](#8-open-questions-and-risks)

---

## 1. Paper Summaries

### 1.1 Core SNN Architecture

**"A solution to the learning dilemma for recurrent networks of spiking neurons"**
- Authors: Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass
- Year: 2020, Nature Communications (470 citations)
- DOI: 10.1038/s41467-020-17236-y
- Key findings: Introduces e-prop (eligibility propagation), a biologically plausible
  online learning rule for recurrent spiking neural networks. Factorizes BPTT gradients
  into a product of eligibility traces (local, forward-computed) and learning signals
  (top-down error). Demonstrated on speech recognition, evidence accumulation, and
  Atari games. Uses 600 recurrently connected LIF neurons with 3 linear readout neurons.
  The adaptive LIF (ALIF) variant includes threshold adaptation.
- **Relevance:** THE foundational paper for irithyll's SNN learning rule. E-prop is
  inherently online (no storage of past states), local (each synapse computes its own
  eligibility trace), and forward-only. Maps directly to streaming architecture.

**"Surrogate Gradient Learning in Spiking Neural Networks"**
- Authors: Neftci, Mostafa, Zenke
- Year: 2019, IEEE Signal Processing Magazine (1300 citations)
- DOI: 10.1109/msp.2019.2931595
- Key findings: Comprehensive tutorial on using surrogate gradients to train SNNs.
  The spike function is non-differentiable (Heaviside step), so a smooth
  pseudo-derivative substitutes during backpropagation. Multiple surrogate choices
  work: piecewise linear, sigmoid derivative, exponential. Remarkably robust to the
  specific choice of surrogate function. Key insight: SNNs can be trained to the
  same accuracy as ReLU networks on standard benchmarks.
- **Relevance:** The surrogate gradient is the critical bridge that lets us compute
  eligibility traces for LIF neurons. Without it, no gradient-based learning is
  possible in SNNs. We need to choose one for irithyll (piecewise linear recommended
  for integer arithmetic).

**"SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"**
- Authors: Zenke, Ganguli
- Year: 2018, Neural Computation (464 citations)
- DOI: 10.1162/neco_a_01086
- Key findings: Introduces SuperSpike, a three-factor learning rule for multilayer
  SNNs. Uses van Rossum distance as loss function and derives online weight updates
  using eligibility traces filtered with exponential kernels. Demonstrates that
  multilayer SNNs CAN learn complex temporal spike patterns with purely local rules.
- **Relevance:** Validates that eligibility trace + learning signal approach scales
  to multiple layers. The exponential filtering of traces maps to simple multiply-decay
  operations in integer arithmetic.

**"Training Spiking Neural Networks Using Lessons From Deep Learning"**
- Authors: Eshraghian, Ward, Neftci, Wang, Lenz, Dwivedi, Bennamoun, Jeong, Lu
- Year: 2023, Proceedings of the IEEE (607 citations)
- DOI: 10.1109/jproc.2023.3308088
- Key findings: Comprehensive tutorial bridging deep learning and SNN training.
  Covers neuron models (LIF, Lapicque, Synaptic, Recurrent), surrogate gradients,
  encoding methods, and practical implementation with snnTorch. Key takeaway:
  the discrete-time LIF equation U[t+1] = beta*U[t] + W*X[t+1] - S[t]*U_thr
  is the workhorse of modern SNN research.
- **Relevance:** Best single reference for implementation details. The snnTorch
  formulation maps directly to integer arithmetic.

### 1.2 Online Learning Rules

**"Three-factor learning in spiking neural networks: An overview of methods and trends"**
- Authors: (Multiple, survey paper)
- Year: 2025, Patterns (Cell Press)
- DOI: 10.1016/j.patter.2025.101414
- Key findings: Comprehensive survey of three-factor learning rules: eligibility
  trace x learning signal x modulation factor. Categories: (1) SuperSpike-style
  (eligibility trace + error signal), (2) R-STDP (eligibility trace + reward signal),
  (3) E-prop (eligibility trace + learning signal from readout). E-prop performs
  best for supervised tasks. S-TLLR (STDP-inspired Temporal Local Learning Rule)
  is highlighted for low memory and time complexity independent of timesteps.
- **Relevance:** Confirms e-prop as the right choice for supervised regression/
  classification in streaming mode. S-TLLR is worth watching for future versions.

**"A Learning Theory for Reward-Modulated STDP"**
- Authors: Legenstein, Pecevski, Maass
- Year: 2008, PLoS Computational Biology (267 citations)
- DOI: 10.1371/journal.pcbi.1000180
- Key findings: Formal learning theory for R-STDP. Weight update is:
  dW = eta * eligibility_trace * reward_signal. The eligibility trace captures
  spike timing correlations; the reward signal provides the "what to learn" signal.
  Proven to converge under mild conditions.
- **Relevance:** R-STDP is the reinforcement learning variant of three-factor rules.
  Could be relevant for trading signal optimization where reward is delayed P&L.
  But for v9.1, supervised e-prop is the priority.

**"Including STDP to eligibility propagation in multi-layer recurrent SNNs"**
- Authors: van der Veen
- Year: 2022, arXiv: 2201.07602
- Key findings: Combines classical STDP with e-prop eligibility traces. Shows that
  adding an STDP component to the eligibility trace computation can improve learning
  in certain tasks. The STDP term biases weight updates toward coincident pre/post
  spike patterns.
- **Relevance:** Potential enhancement for v9.2+ once basic e-prop is working.

### 1.3 Time Series and Financial Applications

**"Brain-Inspired Spiking Neural Network for Online Unsupervised Time Series Prediction"**
- Authors: Chakraborty, Mukhopadhyay
- Year: 2023, IJCNN / arXiv: 2304.04697
- Key findings: Demonstrates SNN for online time series prediction. Uses STDP-based
  unsupervised feature extraction + supervised readout. Achieves competitive
  performance with much less energy than DNN-based online learners. Key insight:
  SNN naturally handles non-stationary time series because STDP continuously adapts.
- **Relevance:** Directly validates our use case — streaming time series prediction
  with SNNs. The architecture (unsupervised hidden + supervised readout) is close
  to what we'd build.

**"Predicting Price Movements in High-Frequency Financial Data with SNNs"**
- Year: 2024, arXiv: 2512.05868
- Key findings: Applies SNNs to financial price prediction. Uses Poisson rate
  encoding for features (returns, volatility, volume). Compares with MLP and LSTM.
  SNN competitive on accuracy with 10-34% energy savings.
- **Relevance:** Validates SNN applicability to financial data. However, Poisson
  rate encoding is wasteful for streaming — delta encoding is better.

**"Online time-series forecasting using spiking reservoir"**
- Year: 2022, Neurocomputing
- Key findings: Spiking reservoir achieves up to 8% higher R2 score vs SARIMA,
  Online ARIMA, and Stacked LSTM while using negligible buffer memory. The reservoir
  approach (fixed random recurrent + trainable readout) needs no backpropagation.
- **Relevance:** Bridges our v9.0 (reservoir) and v9.1 (SNN) plans. A spiking
  reservoir is conceptually the simplest SNN architecture — random LIF recurrent
  layer + linear readout.

### 1.4 Efficient Implementation and Edge Deployment

**"ReckOn: A 28nm Sub-mm2 Task-Agnostic Spiking RNN Processor"**
- Authors: Frenkel, Indiveri
- Year: 2022, ISSCC
- DOI: 10.1109/isscc42614.2022.9731734
- Key findings: 0.45mm2 chip implementing e-prop with only 0.8% memory overhead
  vs inference-only design. <50uW power at 0.5V. Task-agnostic: demonstrated on
  vision, audition, and navigation. Key engineering insight: e-prop's memory
  overhead is tiny because eligibility traces are scalar per synapse, updated
  in-place.
- **Relevance:** PROOF that e-prop can be implemented in extreme resource constraints.
  If it fits on a 0.45mm2 chip, it fits on Cortex-M0+. The 0.8% memory overhead
  figure is critical for our memory budget analysis.

**"NeuroCoreX: An Open-Source FPGA-Based SNN Emulator with On-Chip Learning"**
- Authors: Gautam, Date, Kulkarni, Patton, Potok
- Year: 2025, arXiv: 2506.14138
- Key findings: Open-source FPGA SNN with on-chip STDP learning. Uses fixed-point
  arithmetic throughout. Demonstrates that integer-only SNN implementations are
  practical and performant.
- **Relevance:** Validates integer-only SNN approach. Their FPGA architecture
  decisions (fixed-point membrane potential, integer weights) map to our i16 target.

**"Energy-Efficient Neuromorphic Computing for Edge AI"**
- Authors: Imanov et al.
- Year: 2026, arXiv: 2602.02439
- Key findings: Framework for SNN deployment on resource-constrained edge devices.
  Addresses training difficulty, hardware-mapping overhead, and practical deployment
  barriers. Proposes hardware-aware optimization for SNN inference.
- **Relevance:** Most recent (2026) paper on SNN edge deployment. Confirms the
  space is active and underserved in software tools.

**"A Brief Review of Deep Neural Network Implementations for ARM Cortex-M"**
- Authors: Orasan, Seiculescu, Caleanu
- Year: 2022, Electronics
- DOI: 10.3390/electronics11162545
- Key findings: Reviews DNN deployment on Cortex-M processors. Key constraint:
  M0+ has no FPU, 32KB-256KB SRAM, 16-bit multiply. Fixed-point and quantized
  representations are mandatory. CMSIS-NN provides optimized kernels.
- **Relevance:** Confirms our target constraints. irithyll's existing i16 quantized
  format is well-aligned with Cortex-M capabilities.

### 1.5 Spike Encoding

**"Spike encoding techniques for IoT time-varying signals"**
- Year: 2022, Frontiers in Neuroscience
- DOI: 10.3389/fnins.2022.999029
- Key findings: Benchmarks multiple encoding schemes on neuromorphic classification
  task. Multi-threshold delta modulation showed best robustness (0.7% accuracy drop
  at 0.1 noisy spike rate). Rate encoding requires many timesteps; temporal/latency
  encoding is more efficient but less robust. Delta modulation is optimal for
  time-varying signals because it only generates spikes on change.
- **Relevance:** Confirms delta encoding as the right choice for streaming financial
  data where features change incrementally.

**"A PyTorch-Compatible Spike Encoding Framework"**
- Authors: Vasilache et al.
- Year: 2025, arXiv: 2504.11026
- Key findings: Systematic framework for spike encoding. Compares rate, temporal,
  and delta-based schemes. Delta-based encoding produces the sparsest spike trains,
  which translates directly to computational savings.
- **Relevance:** Implementation reference for encoding schemes.

### 1.6 Quantization

**"Quantization Framework for Fast Spiking Neural Networks"**
- Year: 2022, Frontiers in Neuroscience
- DOI: 10.3389/fnins.2022.918793
- Key findings: 16-bit quantization preserves accuracy fully (0% loss). 8-bit
  quantization drops accuracy by up to 1.07%. Mixed (8,8,16) combination drops
  only 0.3% with 58% size reduction. 4-bit weights are insufficient — too few
  levels to modulate input spikes.
- **Relevance:** CRITICAL finding — i16 (16-bit) quantization has zero accuracy
  loss for SNNs. This is better than for conventional DNNs. irithyll's existing
  i16 infrastructure is perfect.

### 1.7 Existing Rust SNN Implementations

**spiking_neural_networks crate** (crates.io)
- Features: LIF, Hodgkin-Huxley, Izhikevich neurons. STDP learning. Lattice
  connectivity. Trait-based extensibility.
- Limitations: Requires std (uses HashMap, random number generation). No online/
  streaming learning. No e-prop. Simulation-focused, not ML-focused.
- Assessment: Not competition for irithyll's use case. Neuroscience simulation tool,
  not a streaming ML library.

**spiking-neural-net** (github.com/michaelmelanson)
- Uses Specs ECS framework. ~1800 neurons, 100k synapses in real-time on laptop.
- Limitations: Requires std. ECS architecture is overhead for embedded. No e-prop.
- Assessment: Interesting architecture but wrong design goals.

**neural-rs** (github.com/fomorians)
- Minimal spiking library.
- Limitations: Appears unmaintained. Basic functionality only.
- Assessment: Not relevant.

**Summary:** No existing Rust SNN crate offers: no_std + integer arithmetic +
online learning + e-prop + streaming interface. This is the gap irithyll fills.

---

## 2. Neuron Dynamics Equations (Discrete-Time LIF)

### 2.1 Leaky Integrate-and-Fire (LIF) Neuron

The LIF neuron is the standard model used in all modern SNN training frameworks
(snnTorch, Norse, NEST). It is computationally minimal while retaining the essential
spike-integrate-leak dynamics.

**Continuous-time ODE:**
```
tau_m * dV/dt = -(V - V_rest) + R * I(t)
```

**Discrete-time update (the form we implement):**

```
V_j[t] = alpha * V_j[t-1] + sum_i(W_ji * z_i[t-1]) + sum_i(W_ji_in * x_i[t]) - z_j[t-1] * V_thr
```

Where:
- V_j[t]    = membrane potential of neuron j at timestep t
- alpha     = exp(-dt/tau_m) = leak/decay factor, typically 0.9-0.99
- W_ji      = recurrent weight from neuron i to neuron j
- W_ji_in   = input weight from input i to neuron j
- z_i[t-1]  = spike output of neuron i at previous timestep (0 or 1)
- x_i[t]    = external input i at timestep t
- V_thr     = firing threshold
- The last term is the reset: subtracts V_thr when the neuron spiked

**Spike generation (Heaviside step function):**

```
z_j[t] = H(V_j[t] - V_thr) = 1 if V_j[t] >= V_thr, else 0
```

**Reset mechanism (subtract reset, preferred over hard reset):**

After spike emission, the membrane potential is reduced by V_thr rather than
clamped to V_rest. This preserves information about how much V exceeded threshold:

```
V_j[t] = V_j[t] - z_j[t] * V_thr    (subtract reset)
V_j[t] = V_j[t] * (1 - z_j[t])       (hard/zero reset, more lossy)
```

**Refractory period (optional):**

After spiking, the neuron is clamped to V_rest for t_ref timesteps.
For streaming ML, we likely skip this to keep things simple.

### 2.2 Simplified Form (snnTorch convention)

The cleanest discrete formulation used by snnTorch:

```
U[t+1] = beta * U[t] + W * X[t+1] - S[t] * U_thr
S[t]   = 1 if U[t] >= U_thr, else 0
```

Where:
- beta = decay rate (=alpha above), range [0, 1)
- W    = weight matrix (input or recurrent)
- X    = input (spikes or encoded continuous values)
- S    = spike output
- U    = membrane potential

### 2.3 Integer Arithmetic LIF

For i16 implementation on Cortex-M0+:

```
// All values in Q1.14 fixed-point (1 sign bit, 1 integer bit, 14 fractional bits)
// alpha is stored as i16: alpha_i16 = (alpha * 16384) as i16

// Membrane potential update (i32 intermediate to avoid overflow):
let decay: i32 = (V_j as i32 * alpha_i16 as i32) >> 14;
let input: i32 = sum_i(W_ji as i32 * z_i as i32);  // weights * spikes
let V_new: i32 = decay + input;

// Spike check:
let spike: bool = V_new >= V_thr_i16 as i32;

// Reset (subtract):
let V_j: i16 = if spike { (V_new - V_thr_i16 as i32) as i16 } else { V_new as i16 };

// Output:
let z_j: u8 = spike as u8;  // 0 or 1
```

Key insight: LIF neuron update is **3 multiplications + 2 additions + 1 comparison**
per neuron per timestep. On M0+ with single-cycle 16x16->32 multiply, this is
~10-15 cycles per neuron.

---

## 3. Online Learning Rule Equations (e-prop)

### 3.1 The Three-Factor Rule

E-prop factorizes the BPTT gradient into three locally available factors:

```
delta_W_ji = eta * e_ji[t] * L_j[t]
```

Where:
- delta_W_ji = weight change for synapse from neuron i to neuron j
- eta        = learning rate
- e_ji[t]    = eligibility trace (local to the synapse)
- L_j[t]     = learning signal (error-dependent, from readout)

### 3.2 Eligibility Trace for LIF Neurons

The eligibility trace captures how much a weight change would have affected
the postsynaptic neuron's spiking:

```
e_ji[t] = psi_j[t] * z_bar_i[t-1]
```

Where:
- psi_j[t]     = pseudo-derivative (surrogate gradient) of spike function
- z_bar_i[t-1] = low-pass filtered presynaptic spike train

**Filtered presynaptic trace:**
```
z_bar_i[t] = alpha * z_bar_i[t-1] + z_i[t]
```
(Same decay as membrane potential, or can use a separate time constant kappa)

**Pseudo-derivative (surrogate gradient):**

The spike function z = H(V - V_thr) has zero derivative everywhere except at
threshold. The pseudo-derivative substitutes a smooth function:

Option 1 — Piecewise linear (best for integer arithmetic):
```
psi_j[t] = (gamma / V_thr) * max(0, 1 - |V_j[t] - V_thr| / V_thr)
```

Option 2 — Fast sigmoid derivative:
```
psi_j[t] = gamma / (1 + beta * |V_j[t] - V_thr|)^2
```

Option 3 — Exponential:
```
psi_j[t] = gamma * exp(-beta * |V_j[t] - V_thr|)
```

gamma is a dampening factor (typically 0.3-0.5).

**For integer arithmetic, Option 1 (piecewise linear) is strongly recommended:**
it requires only subtraction, absolute value, and comparison — no division or
transcendentals.

### 3.3 Filtered Eligibility Trace

The raw eligibility trace is smoothed for stability:

```
e_bar_ji[t] = kappa * e_bar_ji[t-1] + (1 - kappa) * e_ji[t]
```

Where kappa = exp(-dt/tau_e), typically 0.9-0.999 (long time constant for
temporal credit assignment).

### 3.4 Learning Signal

For supervised regression (our primary use case), the learning signal is
the error at the readout:

```
L_j[t] = sum_k(B_kj * (y*_k[t] - y_k[t]))
```

Where:
- B_kj     = feedback weight from readout k to hidden neuron j
- y*_k[t]  = target output k at time t
- y_k[t]   = actual output k at time t (readout neuron membrane potential)

**Critical insight from Lillicrap et al. (2016):** B does NOT need to be the
transpose of the output weights. Random fixed feedback weights work (feedback
alignment). This eliminates the weight transport problem and simplifies
implementation enormously.

### 3.5 Readout Neuron (for regression)

The readout is a non-spiking leaky integrator:

```
y_k[t] = kappa_out * y_k[t-1] + sum_j(W_kj_out * z_j[t])
```

Where:
- y_k[t]    = readout value (continuous, this IS the prediction)
- kappa_out = readout decay factor
- W_kj_out  = output weight from hidden neuron j to readout k

The readout neuron does NOT spike — it integrates incoming spikes from the
hidden layer through a leaky membrane, producing a continuous-valued output
suitable for regression.

### 3.6 Complete Weight Update

**Hidden-to-readout weights (output layer):**
```
delta_W_kj_out = eta_out * (y*_k[t] - y_k[t]) * z_j[t]
```
(Simple delta rule — error times presynaptic spike)

**Input-to-hidden and recurrent weights (hidden layer):**
```
delta_W_ji = eta * e_bar_ji[t] * L_j[t]
```
(Three-factor rule with filtered eligibility trace and learning signal)

### 3.7 Integer e-prop

All e-prop computations map to integer arithmetic:

- Eligibility trace: multiply-accumulate with decay (i16 * i16 >> 14)
- Learning signal: weighted error sum (i16 * i16 >> 14)
- Weight update: eligibility * learning signal * lr (i16 * i16 * i16 >> 28, accumulated in i32)
- Pseudo-derivative: piecewise linear requires only subtract + abs + compare

The only precision concern is the weight update accumulation — it should use
i32 intermediate arithmetic to avoid overflow, then truncate back to i16.

---

## 4. Spike Encoding Recommendation

### 4.1 Delta Encoding (RECOMMENDED for irithyll)

For streaming continuous-valued features (like financial data), **delta encoding**
is the optimal choice:

```
// For each input feature x_i:
delta_i[t] = x_i[t] - x_i[t-1]

// Generate positive spike if delta exceeds threshold:
spike_pos_i[t] = 1 if delta_i[t] > threshold_i, else 0

// Generate negative spike if delta below negative threshold:
spike_neg_i[t] = 1 if delta_i[t] < -threshold_i, else 0

// Update reference after spike:
if spike_pos_i[t]: x_ref_i = x_ref_i + threshold_i
if spike_neg_i[t]: x_ref_i = x_ref_i - threshold_i
```

**Advantages for streaming:**
1. Zero spikes when data is stable (energy efficient, sparse)
2. Natural match for change detection (which IS orderflow analysis)
3. No need to accumulate data over time windows
4. One computation per feature per timestep
5. Each feature maps to 2 input neurons (positive change, negative change)
6. Threshold is per-feature, can be adaptive

**For financial data specifically:**
- Price features: threshold = 1 tick (minimum price movement)
- Volume features: threshold = adaptive, based on EWMA of volume changes
- Delta (buy-sell imbalance): threshold = N lots
- Natural sparsity: most features don't change on every market tick

### 4.2 Why NOT Rate Encoding

Rate encoding (Poisson spike trains) requires many timesteps per sample to
encode a value through firing rate. For streaming ML where we process ONE
sample at a time, this introduces unnecessary latency. We'd need 50-100
timesteps per sample just for encoding, which defeats the purpose.

### 4.3 Why NOT Latency Encoding

Latency encoding (time-to-first-spike) requires a synchronization signal
and a defined time window. This adds complexity and doesn't naturally map
to streaming data where samples arrive asynchronously.

### 4.4 Integer Delta Encoding

```
// Features arrive as i16 (already quantized by irithyll's feature pipeline)
let delta: i16 = x_i_new - x_i_prev;
let spike_pos: bool = delta > threshold_i;
let spike_neg: bool = delta < -threshold_i;
// That's it. Two comparisons per feature.
```

---

## 5. Pseudocode

### 5.1 Neuron Update (Single Timestep, All Neurons)

```
fn update_neurons(
    membrane: &mut [i16; N_HIDDEN],     // membrane potentials
    spikes: &mut [u8; N_HIDDEN],         // spike outputs (0/1)
    input_spikes: &[u8; N_INPUT],        // encoded input (from delta encoding)
    weights_in: &[[i16; N_INPUT]; N_HIDDEN],    // input weights
    weights_rec: &[[i16; N_HIDDEN]; N_HIDDEN],  // recurrent weights
    prev_spikes: &[u8; N_HIDDEN],        // previous hidden spikes
    alpha_i16: i16,                       // decay factor in Q1.14
    v_thr_i16: i16,                       // threshold in Q1.14
) {
    for j in 0..N_HIDDEN {
        // 1. Decay
        let mut v: i32 = (membrane[j] as i32 * alpha_i16 as i32) >> 14;

        // 2. Input current
        for i in 0..N_INPUT {
            if input_spikes[i] != 0 {
                v += weights_in[j][i] as i32;
            }
        }

        // 3. Recurrent current
        for i in 0..N_HIDDEN {
            if prev_spikes[i] != 0 {
                v += weights_rec[j][i] as i32;
            }
        }

        // 4. Spike check
        let spike = v >= v_thr_i16 as i32;

        // 5. Reset (subtract)
        if spike {
            v -= v_thr_i16 as i32;
        }

        // 6. Clamp and store
        membrane[j] = v.clamp(-32768, 32767) as i16;
        spikes[j] = spike as u8;
    }
}
```

### 5.2 Learning Step (e-prop, Single Timestep)

```
fn learning_step(
    // Neuron state
    membrane: &[i16; N_HIDDEN],
    spikes: &[u8; N_HIDDEN],
    input_spikes: &[u8; N_INPUT],

    // Traces (persistent state, updated in-place)
    pre_trace: &mut [i16; N_INPUT],      // filtered presynaptic traces (input)
    pre_trace_rec: &mut [i16; N_HIDDEN], // filtered presynaptic traces (recurrent)
    elig_in: &mut [[i16; N_INPUT]; N_HIDDEN],    // eligibility traces (input)
    elig_rec: &mut [[i16; N_HIDDEN]; N_HIDDEN],  // eligibility traces (recurrent)

    // Weights (updated in-place)
    weights_in: &mut [[i16; N_INPUT]; N_HIDDEN],
    weights_rec: &mut [[i16; N_HIDDEN]; N_HIDDEN],
    weights_out: &mut [[i16; N_HIDDEN]; N_OUTPUT],
    feedback: &[[i16; N_OUTPUT]; N_HIDDEN],  // random fixed feedback weights

    // Readout state
    readout: &mut [i32; N_OUTPUT],       // readout membrane (i32 for precision)
    target: &[i16; N_OUTPUT],            // target values

    // Parameters (all Q1.14 fixed-point)
    alpha_i16: i16,
    kappa_i16: i16,                      // eligibility trace decay
    kappa_out_i16: i16,                  // readout decay
    eta_i16: i16,                        // learning rate
    v_thr_i16: i16,
    gamma_i16: i16,                      // surrogate gradient dampening
) {
    // === Step 1: Update presynaptic traces ===
    for i in 0..N_INPUT {
        pre_trace[i] = ((pre_trace[i] as i32 * alpha_i16 as i32) >> 14) as i16
                       + (input_spikes[i] as i16) * 16384; // scale spike to Q1.14
    }
    for i in 0..N_HIDDEN {
        pre_trace_rec[i] = ((pre_trace_rec[i] as i32 * alpha_i16 as i32) >> 14) as i16
                           + (spikes[i] as i16) * 16384;
    }

    // === Step 2: Compute pseudo-derivative for each hidden neuron ===
    let mut psi = [0i16; N_HIDDEN];
    for j in 0..N_HIDDEN {
        let dist = abs_i16(membrane[j] - v_thr_i16);
        if dist < v_thr_i16 {
            // Piecewise linear: gamma * (1 - |V - V_thr| / V_thr)
            psi[j] = ((gamma_i16 as i32 * (v_thr_i16 - dist) as i32) / v_thr_i16 as i32) as i16;
        }
        // else: psi[j] = 0 (already initialized)
    }

    // === Step 3: Update eligibility traces ===
    for j in 0..N_HIDDEN {
        for i in 0..N_INPUT {
            let raw_elig = ((psi[j] as i32 * pre_trace[i] as i32) >> 14) as i16;
            elig_in[j][i] = ((elig_in[j][i] as i32 * kappa_i16 as i32) >> 14) as i16
                           + ((16384 - kappa_i16) as i32 * raw_elig as i32 >> 14) as i16;
        }
        for i in 0..N_HIDDEN {
            let raw_elig = ((psi[j] as i32 * pre_trace_rec[i] as i32) >> 14) as i16;
            elig_rec[j][i] = ((elig_rec[j][i] as i32 * kappa_i16 as i32) >> 14) as i16
                            + ((16384 - kappa_i16) as i32 * raw_elig as i32 >> 14) as i16;
        }
    }

    // === Step 4: Update readout (leaky integrator) ===
    for k in 0..N_OUTPUT {
        readout[k] = ((readout[k] as i64 * kappa_out_i16 as i64) >> 14) as i32;
        for j in 0..N_HIDDEN {
            if spikes[j] != 0 {
                readout[k] += weights_out[k][j] as i32;
            }
        }
    }

    // === Step 5: Compute learning signal ===
    let mut learning_signal = [0i32; N_HIDDEN];
    for j in 0..N_HIDDEN {
        for k in 0..N_OUTPUT {
            let error = target[k] as i32 - (readout[k] >> 14) as i32; // scale readout to i16 range
            learning_signal[j] += feedback[j][k] as i32 * error;
        }
        learning_signal[j] >>= 14; // normalize
    }

    // === Step 6: Update weights ===
    // Output weights (delta rule)
    for k in 0..N_OUTPUT {
        let error = target[k] as i32 - (readout[k] >> 14) as i32;
        for j in 0..N_HIDDEN {
            if spikes[j] != 0 {
                let dw = (eta_i16 as i32 * error) >> 14;
                weights_out[k][j] = (weights_out[k][j] as i32 + dw).clamp(-32768, 32767) as i16;
            }
        }
    }

    // Hidden weights (three-factor rule)
    for j in 0..N_HIDDEN {
        let ls = learning_signal[j].clamp(-32768, 32767) as i16;
        for i in 0..N_INPUT {
            let dw = (eta_i16 as i32 * ((elig_in[j][i] as i32 * ls as i32) >> 14)) >> 14;
            weights_in[j][i] = (weights_in[j][i] as i32 + dw).clamp(-32768, 32767) as i16;
        }
        for i in 0..N_HIDDEN {
            let dw = (eta_i16 as i32 * ((elig_rec[j][i] as i32 * ls as i32) >> 14)) >> 14;
            weights_rec[j][i] = (weights_rec[j][i] as i32 + dw).clamp(-32768, 32767) as i16;
        }
    }
}
```

### 5.3 Prediction Readout

```
fn predict(readout: &[i32; N_OUTPUT], leaf_scale: f32) -> f32 {
    // Dequantize: single float operation at the very end
    (readout[0] as f32) / (16384.0 * leaf_scale)
}

// For classification: argmax over readout neurons
fn predict_class(readout: &[i32; N_OUTPUT]) -> usize {
    readout.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0
}
```

---

## 6. Memory Layout Proposal

### 6.1 Per-Neuron State

For a single hidden LIF neuron with e-prop learning:

| Field | Type | Bytes | Description |
|-------|------|-------|-------------|
| membrane | i16 | 2 | Current membrane potential |
| spike | u8 | 1 | Current spike state (0/1) |
| pre_trace | i16 | 2 | Filtered presynaptic trace (for this neuron as presynaptic) |
| **Total per neuron** | | **5** | Core neuron state |

### 6.2 Per-Synapse State (the bulk of memory)

| Field | Type | Bytes | Description |
|-------|------|-------|-------------|
| weight | i16 | 2 | Synaptic weight |
| eligibility | i16 | 2 | Filtered eligibility trace |
| **Total per synapse** | | **4** | |

### 6.3 Memory Budget Calculation

For a network with:
- N_IN = 20 input features (10 features x 2 for delta +/- encoding)
- N_HIDDEN = 64 hidden neurons
- N_OUT = 1 readout neuron (regression)
- Full recurrent connectivity in hidden layer

| Component | Count | Bytes Each | Total |
|-----------|-------|-----------|-------|
| Hidden neuron state | 64 | 5 | 320 |
| Input synapses (weight + elig) | 64 x 20 = 1,280 | 4 | 5,120 |
| Recurrent synapses (weight + elig) | 64 x 64 = 4,096 | 4 | 16,384 |
| Output weights | 1 x 64 = 64 | 2 | 128 |
| Feedback weights (fixed) | 64 x 1 = 64 | 2 | 128 |
| Readout state | 1 | 4 (i32) | 4 |
| Input state (prev values) | 10 | 2 | 20 |
| Config params | 8 | 2 | 16 |
| **TOTAL** | | | **22,120 bytes (~21.6 KB)** |

**This fits in 32KB SRAM** on Cortex-M0+ with ~10KB headroom for stack and
other application state.

### 6.4 Scaling Options

| Hidden neurons | Input features | Memory (KB) | Fits M0+ (32KB)? |
|---------------|---------------|-------------|-------------------|
| 32 | 10 (20 enc) | ~6.9 | Yes |
| 64 | 10 (20 enc) | ~21.6 | Yes (tight) |
| 64 | 20 (40 enc) | ~27.3 | Barely |
| 128 | 10 (20 enc) | ~78.5 | No (needs M3+) |
| 128 | 20 (40 enc) | ~88.2 | No |

**The recurrent connections dominate memory: N_HIDDEN^2 * 4 bytes.**
With 64 neurons: 16KB just for recurrent synapses.

### 6.5 Sparse Connectivity Option

If we use sparse connectivity (e.g., 10% connection probability) instead of
full recurrent, stored in CSR format:

| Component | Dense (64 neurons) | Sparse 10% (64 neurons) |
|-----------|-------------------|------------------------|
| Recurrent synapses | 16,384 B | ~1,640 B + ~820 B index = ~2,460 B |
| **Total network** | ~21.6 KB | ~7.7 KB |

Sparse connectivity dramatically reduces memory and is biologically realistic
(cortical connection probability is 10-20%).

### 6.6 Packed Struct Layout

```rust
/// Packed neuron state — 6 bytes (padded to 8 for alignment)
#[repr(C, align(4))]
pub struct PackedNeuronState {
    pub membrane: i16,     // 2 bytes
    pub pre_trace: i16,    // 2 bytes
    pub spike: u8,         // 1 byte
    pub _pad: u8,          // 1 byte padding
}

/// Packed synapse — 4 bytes
#[repr(C, align(4))]
pub struct PackedSynapse {
    pub weight: i16,       // 2 bytes
    pub eligibility: i16,  // 2 bytes
}
```

---

## 7. Architectural Recommendations for irithyll v9

### 7.1 Decision: Neuron Model

**Recommendation: LIF (Leaky Integrate-and-Fire)**

Rationale:
- LIF is 3 multiplications + 2 additions + 1 compare per neuron per timestep
- Maps perfectly to i16 arithmetic (multiply-accumulate with shift)
- Used by e-prop paper, snnTorch, Norse, NEST, ReckOn chip, NeuroCoreX
- More complex models (AdEx, Izhikevich, HH) add parameters and operations
  without proportional benefit for ML tasks
- ALIF (adaptive LIF with threshold adaptation) is worth adding as a variant
  in v9.2+ but LIF is the right starting point

Decision: **LIF for v9.1, ALIF option in v9.2**

### 7.2 Decision: Learning Rule

**Recommendation: e-prop (eligibility propagation)**

Rationale:
- Only learning rule proven to work for supervised regression/classification
  in recurrent SNNs without BPTT
- Forward-only computation — no storage of past states
- 0.8% memory overhead demonstrated on silicon (ReckOn chip)
- Eligibility trace = simple multiply-decay per synapse per timestep
- Learning signal via random feedback alignment eliminates weight transport
- Three-factor rule is inherently local = embarrassingly streamable

Compared alternatives:
- **STDP:** Unsupervised only. Good for feature extraction but cannot do
  supervised regression without readout modification.
- **R-STDP:** Reinforcement variant. Needs reward signal, not targets. Better
  for v9.3+ trading applications with P&L reward.
- **Surrogate gradient BPTT:** Requires storing all intermediate states for
  backward pass. Incompatible with streaming (O(T) memory for T timesteps).
- **S-TLLR (2025):** Promising new rule with O(1) memory per timestep.
  Not yet validated on regression tasks. Watch for v9.3+.

Decision: **e-prop for v9.1, R-STDP as optional variant in v9.2+**

### 7.3 Decision: Spike Encoding

**Recommendation: Delta encoding with adaptive thresholds**

Rationale:
- Produces sparsest spike trains (fewest operations)
- Natural for streaming data (only fires on change)
- Per-feature thresholds adapt to feature scale
- Two neurons per feature (positive change, negative change)
- Trivially maps to integer comparison

Decision: **Delta encoding for v9.1**

### 7.4 Decision: Network Architecture

**Recommendation for v9.1:**

```
Input layer:  N_features x 2 delta-encoded neurons (no computation)
     |
     v (input weights: N_hidden x N_input synapses)
Hidden layer: 32-64 LIF neurons, recurrent connections
     |        (recurrent weights: N_hidden x N_hidden synapses)
     v (output weights: N_output x N_hidden synapses)
Readout:      1-K leaky integrator neurons (non-spiking)
```

- **32 hidden neurons** for embedded targets (fits in ~7KB)
- **64 hidden neurons** for PC/server targets (fits in ~22KB)
- Full recurrent connectivity for small networks (32 neurons)
- 10-20% sparse connectivity for larger networks (64+)
- 1 readout neuron for regression, K for K-class classification

### 7.5 Decision: Prediction Readout

**Recommendation: Leaky integrator membrane potential**

The readout neuron does NOT spike. Its membrane potential IS the prediction.
For regression: raw membrane potential (dequantized). For classification:
argmax over K readout neurons' membrane potentials.

This avoids the spike rate counting problem entirely — no need to accumulate
spikes over a time window.

### 7.6 Decision: Quantization

**Recommendation: i16 throughout (Q1.14 fixed-point)**

- 16-bit quantization has ZERO accuracy loss for SNNs (validated by
  multiple papers)
- Q1.14 gives range [-2.0, +2.0) with 14 bits of fractional precision
  (~0.00006 resolution)
- Intermediate computations in i32 to prevent overflow
- Single f32 dequantization at final readout
- Matches irithyll's existing i16 quantized infrastructure

### 7.7 Per-Sample Computational Cost

For a 64-hidden-neuron, 20-input network:

| Operation | Per neuron | Per timestep (64 neurons) |
|-----------|-----------|--------------------------|
| Membrane decay | 1 mul + 1 shift | 64 |
| Input current (sparse) | ~3 add (avg 3 active inputs) | 192 |
| Recurrent current (sparse) | ~5 add (avg 5 active recurrents) | 320 |
| Spike check + reset | 1 cmp + 1 sub | 128 |
| Presynaptic trace update | 1 mul + 1 shift + 1 add | 192 |
| Pseudo-derivative | 1 sub + 1 abs + 1 cmp + 1 mul | 256 |
| Eligibility update (input) | 20 * (2 mul + 1 add) | 3,840 |
| Eligibility update (recur) | 64 * (2 mul + 1 add) | 12,288 |
| Learning signal | N_OUT * (1 mul + 1 add) | 64 |
| Weight update (input) | 20 * (2 mul + 1 add) | 3,840 |
| Weight update (recur) | 64 * (2 mul + 1 add) | 12,288 |
| **Total** | | **~33,500 operations** |

On Cortex-M0+ at 48MHz with ~2 cycles per operation: ~1.4ms per sample.
On a modern x86 at 4GHz: ~8.4us per sample (~120k samples/second).

**With sparse 10% recurrent connectivity, the recurrent-related operations
drop by 10x, bringing total to ~7,500 ops/sample = ~0.3ms on M0+.**

### 7.8 Rust API Sketch

```rust
/// Configuration for a SpikeNet
pub struct SpikeNetConfig {
    pub n_input: u16,           // number of raw input features
    pub n_hidden: u16,          // number of hidden LIF neurons
    pub n_output: u16,          // number of readout neurons
    pub alpha: i16,             // membrane decay (Q1.14)
    pub kappa: i16,             // eligibility trace decay (Q1.14)
    pub kappa_out: i16,         // readout decay (Q1.14)
    pub v_threshold: i16,       // firing threshold (Q1.14)
    pub eta: i16,               // learning rate (Q1.14)
    pub gamma: i16,             // surrogate gradient dampening (Q1.14)
    pub delta_threshold: i16,   // spike encoding threshold (Q1.14)
    pub sparse_p: Option<u8>,   // connection probability (0-100), None = dense
}

/// A streaming spiking neural network with online learning
pub struct SpikeNet {
    config: SpikeNetConfig,
    // Neuron state
    membrane: Vec<i16>,
    spikes: Vec<u8>,
    pre_traces: Vec<i16>,
    // Synapse state
    weights_in: Vec<i16>,       // flattened [n_hidden x n_input_encoded]
    weights_rec: Vec<i16>,      // flattened [n_hidden x n_hidden] or CSR
    weights_out: Vec<i16>,      // flattened [n_output x n_hidden]
    elig_in: Vec<i16>,          // same shape as weights_in
    elig_rec: Vec<i16>,         // same shape as weights_rec
    feedback: Vec<i16>,         // random fixed [n_hidden x n_output]
    // Readout state
    readout: Vec<i32>,
    // Input state
    prev_input: Vec<i16>,       // for delta encoding
    // Bookkeeping
    n_samples: u64,
}

impl StreamingLearner for SpikeNet {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // 1. Quantize features to i16
        // 2. Delta-encode to spikes
        // 3. Update neurons (forward pass)
        // 4. E-prop learning step
    }

    fn predict(&self, features: &[f64]) -> f64 {
        // 1. Quantize features
        // 2. Delta-encode to spikes
        // 3. Forward pass (no learning)
        // 4. Dequantize readout
    }

    fn n_samples_seen(&self) -> u64 { self.n_samples }
    fn reset(&mut self) { /* zero all state */ }
}
```

### 7.9 no_std Variant

For embedded targets, provide a fixed-size `SpikeNetView`-style struct:

```rust
/// Fixed-size SNN for no_std embedded targets
/// N_IN, N_HIDDEN, N_OUT are const generics
pub struct SpikeNetFixed<const N_IN: usize, const N_HID: usize, const N_OUT: usize> {
    membrane: [i16; N_HID],
    spikes: [u8; N_HID],
    pre_traces: [i16; N_HID],
    weights_in: [[i16; N_IN * 2]; N_HID],     // *2 for delta +/- encoding
    weights_rec: [[i16; N_HID]; N_HID],
    weights_out: [[i16; N_HID]; N_OUT],
    elig_in: [[i16; N_IN * 2]; N_HID],
    elig_rec: [[i16; N_HID]; N_HID],
    feedback: [[i16; N_OUT]; N_HID],
    readout: [i32; N_OUT],
    prev_input: [i16; N_IN],
    config: SpikeNetConfigFixed,
}
```

All state is stack-allocated. No Vec, no Box, no alloc. Const generics determine
network size at compile time.

---

## 8. Open Questions and Risks

### 8.1 Open Questions

1. **Optimal hidden layer size for financial time series?**
   Literature suggests 50-128 neurons can achieve useful classification. Needs
   empirical testing on our streaming benchmarks (Electricity, Airlines, Covertype).

2. **Sparse vs dense recurrent connectivity?**
   Sparse saves massive memory but may hurt learning for small networks (32 neurons).
   Need to benchmark both.

3. **Multi-timestep per sample?**
   e-prop paper uses 1 timestep per input token. For streaming ML with 1 sample
   at a time, do we want 1 timestep per sample, or should we run the network for
   T timesteps per sample to let recurrent dynamics settle? Literature suggests
   T=1 is fine for simple prediction tasks, T=5-10 for temporal pattern recognition.

4. **Weight initialization?**
   Random normal initialization works, but the scale matters. Too large = instant
   saturation. Too small = no spikes. Need to calibrate for i16 range.
   Recommendation: Initialize weights in [-1000, 1000] (i16), threshold at 8192
   (0.5 in Q1.14).

5. **Readout time constant vs instantaneous?**
   The readout leaky integrator adds memory (kappa_out controls timescale).
   For instantaneous prediction, set kappa_out=0. For predictions that depend
   on recent history, set kappa_out=0.9-0.99.

6. **Learning rate scheduling?**
   E-prop literature uses fixed learning rate. But irithyll's drift detection
   could modulate eta: increase on drift, decrease on stability.

7. **Can SpikeNet compose with SGBT in a stacking ensemble?**
   If SpikeNet implements StreamingLearner, it can be a base learner in
   StackedSGBT or MoE. This is a compelling hybrid architecture.

### 8.2 Known Risks

1. **Accuracy gap vs SGBT on tabular data.**
   SNNs are NOT expected to beat gradient boosted trees on standard tabular
   benchmarks. The value proposition is: (a) temporal pattern learning that
   trees cannot do, (b) embedded deployment with online training, (c) energy
   efficiency for always-on inference. Market the strengths, not false claims.

2. **Integer precision accumulation error.**
   Repeated i16 multiply-shift operations accumulate rounding error. Over
   thousands of training samples, this could cause drift. Mitigation: periodic
   re-normalization of weights and traces (every 1000 samples).

3. **Feedback alignment convergence.**
   Random fixed feedback weights (instead of transposed output weights) work
   for shallow networks but can struggle with deeper architectures. For our
   1-hidden-layer design, this is well within the validated regime.

4. **Spike encoding sensitivity.**
   Delta encoding threshold is a hyperparameter. Too small = too many spikes
   (wasted computation). Too large = missed signal changes. Adaptive thresholds
   (EWMA of feature variance) mitigate this.

5. **Recurrent instability.**
   Without careful weight initialization, recurrent SNNs can enter all-silent
   or all-firing states. Homeostatic mechanisms (adaptive threshold, intrinsic
   plasticity) may be needed. Start with LIF + e-prop and add homeostasis in
   v9.2 if instability is observed.

6. **Limited validation in financial domain.**
   Most SNN research is on MNIST, speech, and neuroscience datasets. Financial
   time series validation is thin (only 2-3 papers). We're entering somewhat
   uncharted territory for financial streaming prediction with SNNs.

### 8.3 Competitive Landscape

| Tool | Language | no_std | Online Training | Neuron Models | Learning Rules |
|------|----------|--------|----------------|---------------|----------------|
| snnTorch | Python/PyTorch | No | BPTT (offline) | LIF, Synaptic, Recurrent | Surrogate gradient BPTT |
| Norse | Python/PyTorch | No | BPTT (offline) | LIF, LI, LSNN | Surrogate gradient BPTT |
| Brian2 | Python | No | Simulation | Any (equation-based) | Any (equation-based) |
| BindsNET | Python/PyTorch | No | Online (STDP) | LIF, IF, SRM | STDP, R-STDP |
| spiking_neural_networks (Rust) | Rust | No | STDP (basic) | LIF, HH, Izhikevich | STDP |
| Lava (Intel) | Python | No | Online | LIF (Loihi) | Three-factor |
| **irithyll v9.1** | **Rust** | **Yes** | **Online (e-prop)** | **LIF (i16)** | **e-prop, delta rule** |

**irithyll would be the ONLY tool offering:** Rust + no_std + i16 integer arithmetic +
online e-prop training + streaming interface + embedded deployment.

---

## Sources

### Papers
- [Bellec et al. 2020 - e-prop (Nature Communications)](https://www.nature.com/articles/s41467-020-17236-y)
- [Neftci et al. 2019 - Surrogate Gradient Learning (IEEE SPM)](https://arxiv.org/abs/1901.09948)
- [Zenke & Ganguli 2018 - SuperSpike](https://doi.org/10.1162/neco_a_01086)
- [Eshraghian et al. 2023 - Training SNNs from Deep Learning (Proc. IEEE)](https://doi.org/10.1109/jproc.2023.3308088)
- [Three-factor learning survey 2025 (Patterns)](https://www.cell.com/patterns/fulltext/S2666-3899(25)00262-4)
- [Legenstein et al. 2008 - R-STDP Theory](https://doi.org/10.1371/journal.pcbi.1000180)
- [Frenkel & Indiveri 2022 - ReckOn Chip (ISSCC)](https://doi.org/10.1109/isscc42614.2022.9731734)
- [Gautam et al. 2025 - NeuroCoreX (arXiv)](https://arxiv.org/abs/2506.14138)
- [Chakraborty & Mukhopadhyay 2023 - Online SNN Time Series](https://arxiv.org/abs/2304.04697)
- [Spike encoding for IoT signals 2022 (Frontiers)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.999029/full)
- [SNN Quantization Framework 2022 (Frontiers)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.918793/full)
- [Lillicrap et al. 2016 - Random Feedback Alignment](https://doi.org/10.1038/ncomms13276)
- [van der Veen 2022 - STDP + e-prop](https://arxiv.org/abs/2201.07602)
- [Imanov et al. 2026 - Energy-Efficient Edge SNN](https://arxiv.org/abs/2602.02439)
- [Orasan et al. 2022 - DNN on Cortex-M (Electronics)](https://doi.org/10.3390/electronics11162545)
- [SNN for nonlinear regression (Royal Society)](https://royalsocietypublishing.org/doi/10.1098/rsos.231606)
- [Financial price prediction with SNNs](https://arxiv.org/html/2512.05868)
- [Vasilache et al. 2025 - Spike Encoding Framework](https://arxiv.org/abs/2504.11026)
- [van der Veen 2022 - STDP + e-prop thesis](https://fse.studenttheses.ub.rug.nl/24344/1/mAI_2021_VanDerVeenWK.pdf)

### Implementation References
- [ChFrenkel/eprop-PyTorch (GitHub)](https://github.com/ChFrenkel/eprop-PyTorch)
- [IGITUGraz/eligibility_propagation (GitHub)](https://github.com/IGITUGraz/eligibility_propagation)
- [ChFrenkel/ReckOn HDL (GitHub)](https://github.com/ChFrenkel/ReckOn)
- [NEST eprop_iaf documentation](https://nest-simulator.readthedocs.io/en/v3.9/models/eprop_iaf.html)
- [NEST eprop_iaf_psc_delta documentation](https://nest-simulator.readthedocs.io/en/latest/models/eprop_iaf_psc_delta.html)
- [NEST eprop_readout documentation](https://nest-simulator.readthedocs.io/en/latest/models/eprop_readout.html)
- [snnTorch Tutorial 1 - Spike Encoding](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)
- [snnTorch Tutorial 2 - LIF Neuron](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html)
- [spiking_neural_networks crate (docs.rs)](https://docs.rs/spiking_neural_networks/latest/spiking_neural_networks/)
- [spiking_neural_networks crate (crates.io)](https://crates.io/crates/spiking_neural_networks)
- [Rust SNNs - Open Neuromorphic](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/rust-snns/)
- [Open Neuromorphic - SNN Frameworks](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/)
- [Spiking Neurons Digital Hardware (Open Neuromorphic)](https://open-neuromorphic.org/blog/spiking-neurons-digital-hardware-implementation/)
- [Loihi Architecture (WikiChip)](https://en.wikichip.org/wiki/intel/loihi)
- [TrueNorth Deep Dive (Open Neuromorphic)](https://open-neuromorphic.org/blog/truenorth-deep-dive-ibm-neuromorphic-chip-design/)
- [Integer LIF FPGA Implementation](https://arxiv.org/html/2411.01628v1)
- [CSR for SNN Simulation](https://arxiv.org/abs/2304.05587)
- [SNN Simulator Comparison (GitHub)](https://github.com/nasiryahm/SNNSimulatorComparison)
- [snntorch Population Coding Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_pop.html)
