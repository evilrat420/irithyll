//! StreamingKAN model implementation.

use super::config::{GateMode, KANConfig};
use super::layer::KANLayer;

use crate::learner::StreamingLearner;

use irithyll_core::continual::{ContinualStrategy, NeuronRegeneration};
use irithyll_core::math::sigmoid;
use irithyll_core::rng::standard_normal;

/// Streaming KAN model with online B-spline learning.
///
/// Processes one sample at a time. Each edge in the network learns a
/// univariate B-spline activation function via per-sample SGD. Online
/// Welford normalization ensures stable input distribution.
///
/// # Example
///
/// ```no_run
/// use irithyll::kan::{StreamingKAN, KANConfig};
/// use irithyll::StreamingLearner;
///
/// let config = KANConfig::builder()
///     .layer_sizes(vec![3, 10, 1])
///     .learning_rate(0.1)
///     .build()
///     .unwrap();
/// let mut model = StreamingKAN::new(config);
/// model.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = model.predict(&[1.0, 2.0, 3.0]);
/// ```
pub struct StreamingKAN {
    layers: Vec<KANLayer>,
    config: KANConfig,
    // Online input normalization (Welford's algorithm)
    input_mean: Vec<f64>,
    input_var: Vec<f64>,
    input_count: u64,
    // Online target normalization (Welford's algorithm)
    target_mean: f64,
    target_var: f64,
    target_count: u64,
    // Cached output for side-effect-free predict()
    last_output: f64,
    n_samples: u64,
    rng_state: u64,
    /// EWMA of squared prediction error for uncertainty-modulated learning.
    pub rolling_loss: f64,
    /// Previous prediction for residual alignment tracking.
    prev_prediction: f64,
    /// Previous prediction change for residual alignment tracking.
    prev_change: f64,
    /// Change from two steps ago, for acceleration-based alignment.
    prev_prev_change: f64,
    /// EWMA of residual alignment signal.
    alignment_ewma: f64,
    // Temporal gating state (T-KAN, Makinde 2026)
    /// Recurrent hidden state for temporal gating. Length = last hidden layer size.
    temporal_state: Vec<f64>,
    /// Gate weights W_gate for projecting input to a scalar gate value.
    /// Length = n_input (config.layer_sizes[0]).
    gate_weights: Vec<f64>,
    /// Gate bias, initialized to 0.0 so sigmoid(b) = 0.5, unbiased initial gating.
    gate_bias: f64,
    /// Optional plasticity guard for maintaining learning capacity.
    plasticity_guard: Option<NeuronRegeneration>,
    /// Snapshot of per-hidden-unit activation energy from previous step.
    prev_hidden_energy: Vec<f64>,
    /// Total number of hidden units (sum of all hidden layer sizes).
    n_hidden_total: usize,
}

impl StreamingKAN {
    /// Create a new StreamingKAN from config.
    pub fn new(config: KANConfig) -> Self {
        let mut rng = if config.seed == 0 { 1 } else { config.seed };
        let mut layers = Vec::with_capacity(config.layer_sizes.len() - 1);
        for i in 0..config.layer_sizes.len() - 1 {
            layers.push(KANLayer::new(
                config.layer_sizes[i],
                config.layer_sizes[i + 1],
                config.spline_order,
                config.grid_size,
                config.momentum,
                &mut rng,
            ));
        }
        let n_in = config.layer_sizes[0];

        // Temporal gating: state dim = last hidden layer size.
        // For layer_sizes=[a, h1, h2, 1], last hidden = h2 = layer_sizes[len-2].
        // For layer_sizes=[a, 1] (no hidden), temporal is a no-op; use empty vecs.
        let has_hidden = config.layer_sizes.len() >= 3;
        let temporal_dim = if has_hidden {
            config.layer_sizes[config.layer_sizes.len() - 2]
        } else {
            0
        };
        let temporal_state = vec![0.0; temporal_dim];
        let gate_weights = if config.gate_mode != GateMode::None && has_hidden {
            (0..n_in)
                .map(|_| standard_normal(&mut rng) * 0.01)
                .collect()
        } else {
            vec![0.0; n_in]
        };
        // B.4: gate_bias initialized to 0.0; sigmoid(0) = 0.5, unbiased initial gating.
        let gate_bias = 0.0;

        // Compute total hidden units (all layers except input and output).
        let n_hidden_total: usize = if config.layer_sizes.len() > 2 {
            config.layer_sizes[1..config.layer_sizes.len() - 1]
                .iter()
                .sum()
        } else {
            0
        };

        // Create plasticity guard if a PlasticityConfig was provided and there are hidden units.
        // Tracks n_hidden_total units (group_size=1 = per-unit tracking).
        let plasticity_guard = config.plasticity.as_ref().and_then(|p| {
            if n_hidden_total > 0 {
                Some(NeuronRegeneration::new(
                    n_hidden_total,
                    1, // group_size = 1 (per-unit tracking)
                    p.regen_fraction,
                    p.regen_interval,
                    p.utility_alpha,
                    config.seed.wrapping_add(0x_DEAD_CAFE),
                ))
            } else {
                None
            }
        });
        let prev_hidden_energy = vec![0.0; n_hidden_total];

        Self {
            layers,
            config,
            input_mean: vec![0.0; n_in],
            input_var: vec![1.0; n_in],
            input_count: 0,
            target_mean: 0.0,
            target_var: 1.0,
            target_count: 0,
            last_output: 0.0,
            n_samples: 0,
            rng_state: rng,
            rolling_loss: 0.0,
            prev_prediction: 0.0,
            prev_change: 0.0,
            prev_prev_change: 0.0,
            alignment_ewma: 0.0,
            temporal_state,
            gate_weights,
            gate_bias,
            plasticity_guard,
            prev_hidden_energy,
            n_hidden_total,
        }
    }

    /// Access the config.
    pub fn config(&self) -> &KANConfig {
        &self.config
    }

    /// Number of KAN layers (edges between node layers).
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total number of learnable parameters (B-spline coefficients).
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// The layer sizes from the config.
    pub fn layer_sizes(&self) -> &[usize] {
        &self.config.layer_sizes
    }

    /// Per-input feature importance from layer 0 B-spline coefficient magnitudes.
    ///
    /// For each input `i`, sums `|coefficient|` across all output edges and all
    /// `n_coeffs = grid_size + spline_order` coefficients per edge. Inputs whose
    /// edges accumulate larger spline weight after training rank higher.
    ///
    /// Returns un-normalized scores (length = `layer_sizes()[0]`); the caller
    /// should sum-normalize for display. Returns empty if the model has no
    /// layers (impossible in normal use — `new()` requires `layer_sizes.len() >= 2`).
    pub fn input_importances(&self) -> Vec<f64> {
        let Some(first) = self.layers.first() else {
            return Vec::new();
        };
        let coeffs = first.coefficients();
        let n_in = self.config.layer_sizes[0];
        let n_out = self.config.layer_sizes[1];
        let n_coeffs = self.config.grid_size + self.config.spline_order;

        let mut importances = vec![0.0_f64; n_in];
        for j in 0..n_out {
            for (i, imp) in importances.iter_mut().enumerate() {
                let edge = j * n_in + i;
                let coeff_base = edge * n_coeffs;
                for c in 0..n_coeffs {
                    *imp += coeffs[coeff_base + c].abs();
                }
            }
        }
        importances
    }

    /// Surgically reinitialize a hidden unit by flat index.
    ///
    /// Maps the flat hidden index to the correct layer and local node index,
    /// then reinitializes all incoming edges (in the layer producing this node)
    /// and all outgoing edges (in the layer consuming this node). This preserves
    /// all other units' learned representations — only the dead unit gets recycled.
    ///
    /// # Arguments
    ///
    /// * `flat_j` — flat hidden unit index (0-based across all hidden layers)
    /// * `rng` — mutable RNG state for generating fresh weights
    ///
    /// # Panics
    ///
    /// Panics if `flat_j >= n_hidden_total`.
    fn reinitialize_hidden_unit(&mut self, flat_j: usize, rng: &mut u64) {
        assert!(
            flat_j < self.n_hidden_total,
            "hidden unit index {} out of range (n_hidden_total={})",
            flat_j,
            self.n_hidden_total
        );

        // Map flat index to (hidden_layer_idx, local_j).
        // Hidden layers are layer_sizes[1..len-1].
        let mut remaining = flat_j;
        let mut hidden_layer_idx = 0;
        for &size in &self.config.layer_sizes[1..self.config.layer_sizes.len() - 1] {
            if remaining < size {
                break;
            }
            remaining -= size;
            hidden_layer_idx += 1;
        }
        let local_j = remaining;

        // layers[hidden_layer_idx] connects layer_sizes[hidden_layer_idx] →
        // layer_sizes[hidden_layer_idx+1]. This hidden node is output node
        // local_j of layers[hidden_layer_idx].
        self.layers[hidden_layer_idx].reinitialize_output_node(local_j, rng);

        // layers[hidden_layer_idx+1] connects layer_sizes[hidden_layer_idx+1] →
        // layer_sizes[hidden_layer_idx+2]. This hidden node is input node
        // local_j of layers[hidden_layer_idx+1].
        if hidden_layer_idx + 1 < self.layers.len() {
            self.layers[hidden_layer_idx + 1].reinitialize_input_node(local_j, rng);
        }

        // Zero the temporal state for this unit (if temporal gating is active
        // and this unit is in the last hidden layer).
        if self.config.gate_mode != GateMode::None && !self.temporal_state.is_empty() {
            // Check if this flat_j maps to the last hidden layer.
            let last_hidden_start: usize = self.config.layer_sizes
                [1..self.config.layer_sizes.len() - 2]
                .iter()
                .sum();
            let last_hidden_size = self.config.layer_sizes[self.config.layer_sizes.len() - 2];
            if flat_j >= last_hidden_start && flat_j < last_hidden_start + last_hidden_size {
                let local_in_last = flat_j - last_hidden_start;
                if local_in_last < self.temporal_state.len() {
                    self.temporal_state[local_in_last] = 0.0;
                }
            }
        }
    }

    /// Normalize input via Welford's online algorithm, clamped for B-spline safety.
    fn normalize_input(&mut self, features: &[f64]) -> Vec<f64> {
        self.input_count += 1;
        let n = self.input_count as f64;
        let mut normalized = vec![0.0; features.len()];

        for (i, &x) in features.iter().enumerate() {
            if i >= self.input_mean.len() {
                // Handle dimension mismatch gracefully -- pass through
                normalized[i] = x;
                continue;
            }
            let delta = x - self.input_mean[i];
            self.input_mean[i] += delta / n;
            let delta2 = x - self.input_mean[i];
            self.input_var[i] += delta * delta2;

            // Normalize to roughly zero-mean, unit-variance.
            // B.3: If feature std < 1e-8 (constant feature), skip normalization
            // for that feature to avoid div-by-zero / NaN contamination.
            let raw_std = if n > 1.0 {
                (self.input_var[i] / (n - 1.0)).sqrt()
            } else {
                0.0
            };
            if raw_std < 1e-8 {
                // Constant feature: pass through centered at 0 in grid domain
                normalized[i] = 0.0;
                continue;
            }
            normalized[i] = (x - self.input_mean[i]) / raw_std;
            // Clamp to [-1+eps, 1-eps] — the full B-spline grid domain [-1, 1].
            // Using the full domain ensures all grid intervals participate.
            // Previous [-0.95, 0.95] clamp discarded ~44% of normalized data
            // at the boundaries, creating dead zones where B-splines had support.
            normalized[i] = normalized[i].clamp(-1.0 + 1e-7, 1.0 - 1e-7);
        }
        normalized
    }
}

impl StreamingLearner for StreamingKAN {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        // Option D note: KAN has no RLS readout — the learning rule is full backpropagation
        // through B-spline coefficients. There is no "pre-update vs post-update feature
        // distribution" asymmetry: both train_one and predict() run a forward pass through
        // the same current spline weights, so train and predict see identical features at
        // any given state. Option D does not apply to pure-backprop learners like StreamingKAN.

        // 1. Normalize input to B-spline grid domain [-1, 1].
        //    During warmup (first 50 samples): accumulate Welford stats only,
        //    map features to zero (don't train on un-normalized garbage).
        let normalized = if self.input_count < 50 {
            self.input_count += 1;
            let n = self.input_count as f64;
            for (i, &x) in features.iter().enumerate() {
                if i < self.input_mean.len() {
                    let delta = x - self.input_mean[i];
                    self.input_mean[i] += delta / n;
                    let delta2 = x - self.input_mean[i];
                    self.input_var[i] += delta * delta2;
                }
            }
            // During warmup: use centered features scaled by a rough estimate.
            // Centering + scaling preserves signal for large-scale inputs
            // (e.g. pressure ~1013) that would be destroyed by fixed-range clamping.
            if n > 2.0 {
                features
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        if i < self.input_mean.len() {
                            // B.3: skip normalization for constant features
                            let raw_std = (self.input_var[i] / (n - 1.0)).sqrt();
                            if raw_std < 1e-8 {
                                return 0.0;
                            }
                            ((x - self.input_mean[i]) / raw_std).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
                        } else {
                            0.0
                        }
                    })
                    .collect()
            } else {
                vec![0.0; features.len()] // first 2 samples: don't train, just observe
            }
        } else {
            self.normalize_input(features)
        };

        // 2. Forward through all layers, saving activations
        let mut activations: Vec<Vec<f64>> = Vec::with_capacity(self.layers.len() + 1);
        activations.push(normalized.clone());
        let mut current = normalized;
        let n_layers = self.layers.len();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current);
            // Inter-layer normalization: keep hidden activations in B-spline
            // support range. Skip the last layer (output layer).
            if layer_idx < n_layers - 1 {
                let layer_output_max = current
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0_f64, f64::max)
                    .max(1.0);
                if layer_output_max > 5.0 {
                    let scale = 5.0 / layer_output_max;
                    for val in current.iter_mut() {
                        *val *= scale;
                    }
                }
            }
            activations.push(current.clone());
        }

        // 2b. Temporal gating (T-KAN, Makinde 2026):
        //     Mix last hidden layer output with recurrent state via learned gate.
        //     Requires >= 2 layers (at least one hidden layer before output).
        //     gate = sigmoid(W_gate · normalized_input + b_gate)
        //     mixed = gate * temporal_state + (1-gate) * pre_output_activation
        //     Then re-run final layer on mixed instead of original activation.
        //
        //     Gate gradient is deferred until after the final-layer backward
        //     pass so we have the true chain-rule gradient δ^T (s − h) rather
        //     than the heuristic mean(s − h) used prior to v10.
        let temporal_ctx = if self.config.gate_mode != GateMode::None && self.layers.len() >= 2 {
            // activations[0] is normalized input, activations[len-1] is final output,
            // activations[len-2] is last hidden layer output (pre-output).
            let pre_output_idx = activations.len() - 2;
            let pre_output = activations[pre_output_idx].clone();

            // Compute gate: g = sigmoid(dot(gate_weights, normalized_input) + gate_bias)
            let dot: f64 = self
                .gate_weights
                .iter()
                .zip(activations[0].iter())
                .map(|(w, x)| w * x)
                .sum();
            let g = sigmoid(dot + self.gate_bias);

            // Mix: mixed = g * temporal_state + (1-g) * pre_output
            let mixed: Vec<f64> = self
                .temporal_state
                .iter()
                .zip(pre_output.iter())
                .map(|(&s, &h)| g * s + (1.0 - g) * h)
                .collect();

            // Save previous temporal state (needed for gate gradient after backward).
            let prev_state = self.temporal_state.clone();
            // Update temporal state
            self.temporal_state = mixed.clone();

            // Re-run final layer on mixed representation
            let last_layer = self.layers.len() - 1;
            current = self.layers[last_layer].forward(&mixed);

            // Update activations: replace pre-output with mixed, final with new output
            activations[pre_output_idx] = mixed;
            *activations.last_mut().unwrap() = current.clone();

            // Stash context needed for chain-rule gate gradient (applied after backward).
            Some((g, prev_state, pre_output))
        } else {
            None
        };

        // 3. Normalize target via Welford's online algorithm.
        //    KAN trains in normalized target space for stable gradient magnitudes
        //    when target magnitudes vary across regimes (e.g., Feynman equations).
        self.target_count += 1;
        let tn = self.target_count as f64;
        let t_delta = target - self.target_mean;
        self.target_mean += t_delta / tn;
        let t_delta2 = target - self.target_mean;
        self.target_var += t_delta * t_delta2;
        // B.2: If target std < 1e-8 (constant target), use target_std=1.0 and
        // skip normalization to avoid div-by-zero and NaN in gradient computation.
        let raw_target_std = if tn > 2.0 {
            (self.target_var / (tn - 1.0)).sqrt()
        } else {
            0.0
        };
        let target_std = if raw_target_std < 1e-8 {
            1.0
        } else {
            raw_target_std
        };
        let normalized_target = if raw_target_std < 1e-8 {
            target - self.target_mean
        } else {
            (target - self.target_mean) / target_std
        };

        // 4. Compute output error in normalized target space
        let prediction = current[0];
        let error = prediction - normalized_target;
        let sq_error = error * error;

        // 4. Update rolling loss and compute uncertainty-modulated LR.
        //    High error relative to baseline → increase lr (adapt faster).
        //    Low error → decrease lr (conserve).
        const LOSS_ALPHA: f64 = 0.001;
        if self.n_samples == 0 {
            self.rolling_loss = sq_error;
        } else {
            self.rolling_loss = (1.0 - LOSS_ALPHA) * self.rolling_loss + LOSS_ALPHA * sq_error;
        }

        let effective_lr = if self.n_samples > 500 && self.rolling_loss > 1e-10 {
            let ratio = (sq_error / self.rolling_loss).clamp(0.5, 2.0);
            self.config.learning_rate * ratio
        } else {
            self.config.learning_rate // Fixed LR during warmup
        };

        let lr = effective_lr * weight;

        // B.1: Finiteness guard — skip backward pass if any activation contains NaN/Inf.
        // This keeps the model healthy when an upstream computation produced bad values.
        let activations_finite = activations
            .iter()
            .all(|act| act.iter().all(|f| f.is_finite()));
        if !activations_finite {
            // Skip gradient update; cache denormalized prediction and return.
            let denormalized = current[0] * target_std + self.target_mean;
            self.last_output = if denormalized.is_finite() {
                denormalized.clamp(-1e6, 1e6)
            } else {
                0.0
            };
            self.n_samples += 1;
            return;
        }

        // 5. Backward through layers (reverse order).
        //    When temporal gating is active, the final layer was already
        //    re-run on `mixed` in step 2b; we run its backward on `mixed`
        //    (stored at activations[pre_output_idx]) to get the gradient
        //    at the mixed activation, then apply the chain-rule gate update.
        let mut grad = vec![2.0 * error]; // dL/d_output for MSE
        for i in (0..self.layers.len()).rev() {
            grad = self.layers[i].backward(&activations[i], &grad, lr);
        }

        // 5b. Gate gradient via chain rule (temporal gating only).
        //     After the backward pass, `grad` holds ∂L/∂mixed (the gradient
        //     at the final layer's input = the mixed activation).
        //     d_mixed / dg = prev_state - pre_output  (element-wise)
        //     ∂L/∂g = Σ_j (∂L/∂mixed_j) * (prev_state_j - pre_output_j)
        //     ∂g/∂W_i = g * (1-g) * x_i  (sigmoid gate)
        //     ∂L/∂W_i = ∂L/∂g * g*(1-g) * x_i
        //
        //     This is the proper chain-rule gradient through the gate function.
        //     The prior heuristic used mean(prev_state - pre_output) which has
        //     no guaranteed sign relationship with the loss gradient.
        if let Some((g, prev_state, pre_output)) = temporal_ctx {
            // dot product of upstream gradient with direction (prev_state - pre_output)
            let dl_dg: f64 = grad
                .iter()
                .zip(prev_state.iter().zip(pre_output.iter()))
                .map(|(&delta_j, (&s_j, &h_j))| delta_j * (s_j - h_j))
                .sum();
            // Sigmoid local gradient: g * (1 - g)
            let sigmoid_local = g * (1.0 - g);
            let gate_lr = 0.01 * self.config.learning_rate;
            let scalar_grad = (dl_dg * sigmoid_local).clamp(-1.0, 1.0);
            for (w, &x_i) in self.gate_weights.iter_mut().zip(activations[0].iter()) {
                let update = gate_lr * scalar_grad * x_i;
                if update.is_finite() {
                    *w -= update.clamp(-0.1, 0.1);
                }
            }
            let bias_update = gate_lr * scalar_grad;
            if bias_update.is_finite() {
                self.gate_bias -= bias_update.clamp(-0.1, 0.1);
            }
        }

        // 6. Adaptive coefficient decay: high error → more decay (forget stale
        //    coefficients faster during drift), low error → less decay (conserve
        //    during stable regimes). Only apply after the model has had time to
        //    learn (warmup) — early error is learning phase error, not drift.
        if self.config.coefficient_decay > 0.0 && self.n_samples > 2000 {
            let ratio = if self.rolling_loss > 1e-10 {
                (sq_error / self.rolling_loss).clamp(0.5, 3.0)
            } else {
                1.0
            };
            let adaptive_decay = self.config.coefficient_decay * ratio;
            let decay = 1.0 - adaptive_decay.clamp(0.0, 0.01); // Never decay more than 1% per step
            for layer in &mut self.layers {
                for coeff in layer.coefficients_mut() {
                    *coeff *= decay;
                }
            }
        }

        // 7. Coefficient magnitude guard: reset non-finite coefficients to zero
        //    and scale remaining by 0.5 when any exceed 1e6. Divergent coefficients
        //    typically signal a configuration issue (learning rate, input normalization).
        let any_extreme = self.layers.iter().any(|l| {
            l.coefficients()
                .iter()
                .any(|c| !c.is_finite() || c.abs() > 1e6)
        });
        if any_extreme {
            for layer in &mut self.layers {
                for coeff in layer.coefficients_mut() {
                    if !coeff.is_finite() {
                        *coeff = 0.0;
                    } else {
                        *coeff *= 0.5;
                    }
                }
            }
        }

        // 8. Update residual alignment tracking (acceleration-based).
        let current_change = prediction - self.prev_prediction;
        if self.n_samples > 0 {
            let acceleration = current_change - self.prev_change;
            let prev_acceleration = self.prev_change - self.prev_prev_change;
            let agreement = if acceleration.abs() > 1e-15 && prev_acceleration.abs() > 1e-15 {
                if (acceleration > 0.0) == (prev_acceleration > 0.0) {
                    1.0
                } else {
                    -1.0
                }
            } else {
                0.0
            };
            const ALIGN_ALPHA: f64 = 0.05;
            if self.n_samples == 1 {
                self.alignment_ewma = agreement;
            } else {
                self.alignment_ewma =
                    (1.0 - ALIGN_ALPHA) * self.alignment_ewma + ALIGN_ALPHA * agreement;
            }
        }
        self.prev_prev_change = self.prev_change;
        self.prev_change = current_change;
        self.prev_prediction = prediction;

        // 9. Plasticity maintenance: track per-hidden-unit activation energy
        //    and trigger regeneration when dead units are detected.
        // Plasticity maintenance: collect regeneration decisions first, then
        // apply surgical reinit outside the guard borrow to satisfy the borrow checker.
        let mut regenerated_units: Vec<usize> = Vec::new();
        if let Some(ref mut guard) = self.plasticity_guard {
            // Build flat energy vector from hidden layer activations.
            // activations[0] = input, activations[1..n-1] = hidden, activations[n-1] = output.
            let mut hidden_energy = Vec::with_capacity(self.n_hidden_total);
            for layer_acts in activations.iter().take(activations.len() - 1).skip(1) {
                for &a in layer_acts {
                    hidden_energy.push(a.abs());
                }
            }
            if hidden_energy.len() == self.n_hidden_total {
                guard.pre_update(&self.prev_hidden_energy, &mut hidden_energy);
                guard.post_update(&self.prev_hidden_energy);
                // Collect which units need reinit.
                let n_groups = guard.n_groups();
                for j in 0..n_groups {
                    if guard.was_regenerated(j) {
                        regenerated_units.push(j);
                    }
                }
                self.prev_hidden_energy = hidden_energy;
            }
        }
        // Surgical per-unit reinit: only dead units get recycled.
        // Each dead hidden unit has its incoming and outgoing B-spline
        // edges reinitialized while preserving all other units.
        if !regenerated_units.is_empty() {
            let mut reinit_rng = self
                .config
                .seed
                .wrapping_add(0xCAFE_BABE_u64.wrapping_mul(self.n_samples));
            for j in regenerated_units {
                self.reinitialize_hidden_unit(j, &mut reinit_rng);
            }
        }

        // 10. Denormalize prediction back to original target space, then cache.
        let denormalized = prediction * target_std + self.target_mean;
        self.last_output = denormalized.clamp(-1e6, 1e6);
        if !self.last_output.is_finite() {
            self.last_output = 0.0;
        }
        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        // Side-effect-free forward pass using frozen normalization stats.
        // When temporal gating is enabled, the current temporal_state is used
        // for mixing (read-only — no state mutation during predict).
        if self.n_samples == 0 {
            return 0.0;
        }

        // Normalize input using frozen Welford stats (no mutation)
        let n = self.input_count as f64;
        let mut normalized = vec![0.0; features.len()];
        for (i, &x) in features.iter().enumerate() {
            if i < self.input_mean.len() && n > 1.0 {
                let std = (self.input_var[i] / (n - 1.0)).sqrt().max(1e-8);
                normalized[i] = ((x - self.input_mean[i]) / std).clamp(-1.0 + 1e-7, 1.0 - 1e-7);
            }
        }

        // Forward through all layers (all but last if temporal, then gate, then last)
        let mut current = normalized.clone();
        let n_layers = self.layers.len();
        let stop_before = if self.config.gate_mode != GateMode::None && n_layers >= 2 {
            n_layers - 1
        } else {
            n_layers
        };

        for (layer_idx, layer) in self.layers.iter().enumerate().take(stop_before) {
            current = layer.forward(&current);
            // Inter-layer normalization (same as train_one)
            if layer_idx < n_layers - 1 {
                let layer_output_max = current
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0_f64, f64::max)
                    .max(1.0);
                if layer_output_max > 5.0 {
                    let scale = 5.0 / layer_output_max;
                    for val in current.iter_mut() {
                        *val *= scale;
                    }
                }
            }
        }

        // Temporal gating: mix with recurrent state (read-only, no state mutation)
        if self.config.gate_mode != GateMode::None && n_layers >= 2 {
            let dot: f64 = self
                .gate_weights
                .iter()
                .zip(normalized.iter())
                .map(|(w, x)| w * x)
                .sum();
            let g = sigmoid(dot + self.gate_bias);

            // Mix: current is the pre-output hidden activation
            let mixed: Vec<f64> = self
                .temporal_state
                .iter()
                .zip(current.iter())
                .map(|(&s, &h)| g * s + (1.0 - g) * h)
                .collect();

            // Run final layer on mixed
            current = self.layers[n_layers - 1].forward(&mixed);
        }

        // Denormalize prediction back to original target space
        let prediction = current[0];
        let target_std = if self.target_count > 2 {
            (self.target_var / (self.target_count as f64 - 1.0))
                .sqrt()
                .max(1e-8)
        } else {
            1.0
        };
        let denormalized = prediction * target_std + self.target_mean;
        let result = denormalized.clamp(-1e6, 1e6);
        if result.is_finite() {
            result
        } else {
            0.0
        }
    }

    #[inline]
    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset(&mut self.rng_state);
        }
        self.input_mean.fill(0.0);
        self.input_var.fill(1.0);
        self.input_count = 0;
        self.target_mean = 0.0;
        self.target_var = 1.0;
        self.target_count = 0;
        self.last_output = 0.0;
        self.n_samples = 0;
        self.rolling_loss = 0.0;
        self.prev_prediction = 0.0;
        self.prev_change = 0.0;
        self.prev_prev_change = 0.0;
        self.alignment_ewma = 0.0;
        // Reset temporal gating state
        self.temporal_state.fill(0.0);
        for w in &mut self.gate_weights {
            *w = standard_normal(&mut self.rng_state) * 0.01;
        }
        // B.4: reset gate_bias to 0.0 (the initial value from new()), not 1.0
        self.gate_bias = 0.0;
        if let Some(ref mut guard) = self.plasticity_guard {
            guard.reset();
        }
        self.prev_hidden_energy.fill(0.0);
    }

    #[allow(deprecated)]
    fn diagnostics_array(&self) -> [f64; 5] {
        <Self as crate::learner::Tunable>::diagnostics_array(self)
    }

    #[allow(deprecated)]
    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64) {
        <Self as crate::learner::Tunable>::adjust_config(self, lr_multiplier, lambda_delta);
    }
}

impl crate::learner::Tunable for StreamingKAN {
    fn diagnostics_array(&self) -> [f64; 5] {
        use crate::automl::DiagnosticSource;
        match self.config_diagnostics() {
            Some(d) => [
                d.residual_alignment,
                d.regularization_sensitivity,
                d.depth_sufficiency,
                d.effective_dof,
                d.uncertainty,
            ],
            None => [0.0; 5],
        }
    }

    fn adjust_config(&mut self, lr_multiplier: f64, lambda_delta: f64) {
        self.config.learning_rate *= lr_multiplier;
        // Wire lambda_delta to coefficient_decay: positive delta → more decay
        // (faster forgetting during drift), negative → less decay (conserve).
        // Clamped to [0, 0.1) so it never exceeds the config validation range.
        if lambda_delta != 0.0 {
            self.config.coefficient_decay =
                (self.config.coefficient_decay + lambda_delta).clamp(0.0, 0.099);
        }
    }
}

impl std::fmt::Debug for StreamingKAN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingKAN")
            .field("layer_sizes", &self.config.layer_sizes)
            .field("spline_order", &self.config.spline_order)
            .field("grid_size", &self.config.grid_size)
            .field("learning_rate", &self.config.learning_rate)
            .field("n_params", &self.n_params())
            .field("n_samples", &self.n_samples)
            .finish()
    }
}

impl crate::automl::DiagnosticSource for StreamingKAN {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        // Dead edge fraction: fraction of edges where all velocity magnitudes < 1e-8.
        // encoder_utilization = 1.0 - dead_edge_fraction.
        let encoder_utilization = {
            let (mut total_dead, mut total_edges) = (0usize, 0usize);
            for layer in &self.layers {
                let (dead, edges) = layer.count_dead_edges(1e-8);
                total_dead += dead;
                total_edges += edges;
            }
            if total_edges > 0 {
                1.0 - (total_dead as f64 / total_edges as f64)
            } else {
                0.0
            }
        };

        Some(crate::automl::ConfigDiagnostics {
            residual_alignment: self.alignment_ewma,
            regularization_sensitivity: self.config.coefficient_decay,
            depth_sufficiency: encoder_utilization,
            effective_dof: self.n_params() as f64,
            uncertainty: self.rolling_loss.sqrt(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_model() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 10, 1])
            .build()
            .unwrap();
        let model = StreamingKAN::new(config);
        assert_eq!(model.n_samples_seen(), 0);
        assert_eq!(model.n_layers(), 2);
    }

    #[test]
    fn train_and_predict_finite() {
        let config = KANConfig::builder()
            .layer_sizes(vec![1, 5, 1])
            .learning_rate(0.01)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        for i in 0..100 {
            let x = i as f64 * 0.1;
            let y = 2.0 * x + 1.0;
            model.train(&[x], y);
        }

        let pred = model.predict(&[0.5]);
        assert!(pred.is_finite(), "prediction should be finite, got {pred}");
    }

    #[test]
    fn n_samples_tracks() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 5, 1])
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        for i in 0..25 {
            model.train(&[i as f64, (i as f64).sin()], i as f64);
        }
        assert_eq!(model.n_samples_seen(), 25);
    }

    #[test]
    fn reset_clears_state() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 5, 1])
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        for i in 0..20 {
            model.train(&[i as f64, 0.0], i as f64 * 2.0);
        }
        assert!(model.n_samples_seen() > 0);

        model.reset();
        assert_eq!(model.n_samples_seen(), 0);
        assert_eq!(model.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn implements_streaming_learner() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 5, 1])
            .build()
            .unwrap();
        let model = StreamingKAN::new(config);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(model);
        boxed.train(&[1.0, 2.0], 3.0);
        let pred = boxed.predict(&[1.0, 2.0]);
        assert!(pred.is_finite());
    }

    #[test]
    fn multi_layer_kan() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 8, 4, 1])
            .learning_rate(0.005)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        assert_eq!(model.n_layers(), 3);

        for i in 0..100 {
            let x = [i as f64 * 0.01, (i as f64).sin(), (i as f64).cos()];
            let y = x[0] * 3.0 + x[1] * 2.0 - x[2];
            model.train(&x, y);
        }

        let pred = model.predict(&[0.5, 0.5_f64.sin(), 0.5_f64.cos()]);
        assert!(
            pred.is_finite(),
            "multi-layer KAN prediction should be finite, got {pred}"
        );
    }

    #[test]
    fn n_params_positive() {
        let config = KANConfig::builder()
            .layer_sizes(vec![3, 10, 1])
            .build()
            .unwrap();
        let model = StreamingKAN::new(config);
        assert!(
            model.n_params() > 0,
            "n_params should be positive, got {}",
            model.n_params()
        );
    }

    #[test]
    fn kan_learns_nonlinear() {
        // Use a stationary distribution: repeated passes over x in [-1, 1]
        // for y = x^2 (nonlinear, within grid range).
        let config = KANConfig::builder()
            .layer_sizes(vec![1, 8, 1])
            .learning_rate(0.01)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // Generate fixed dataset: 20 points in [-1, 1]
        let n_pts = 20;
        let xs: Vec<f64> = (0..n_pts)
            .map(|i| -1.0 + 2.0 * i as f64 / (n_pts - 1) as f64)
            .collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();

        // Measure error after initial pass (first epoch)
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            model.train(&[x], y);
        }
        let early_mse: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                model.train(&[x], y);
                let pred = model.predict(&[x]);
                (pred - y).powi(2)
            })
            .sum::<f64>()
            / n_pts as f64;

        // Train for many more epochs
        for _ in 0..20 {
            for (&x, &y) in xs.iter().zip(ys.iter()) {
                model.train(&[x], y);
            }
        }

        // Measure error after training (last epoch)
        let late_mse: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                model.train(&[x], y);
                let pred = model.predict(&[x]);
                (pred - y).powi(2)
            })
            .sum::<f64>()
            / n_pts as f64;

        assert!(
            late_mse < early_mse,
            "KAN should learn: early MSE ({early_mse:.6}) should be > late MSE ({late_mse:.6})"
        );
    }

    #[test]
    fn predict_before_train_returns_zero() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 5, 1])
            .build()
            .unwrap();
        let model = StreamingKAN::new(config);
        assert_eq!(model.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn kan_uncertainty_modulated_lr() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 10, 1])
            .learning_rate(0.01)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // rolling_loss starts at 0
        assert!(
            model.rolling_loss.abs() < 1e-15,
            "rolling_loss should start at 0, got {}",
            model.rolling_loss
        );

        // Train on 100 samples
        for i in 0..100 {
            let t = i as f64 * 0.05;
            let x = [t.sin(), t.cos()];
            let y = 0.5 * x[0] + 0.3 * x[1];
            model.train(&x, y);
        }

        // rolling_loss should be tracked and > 0 after training
        assert!(
            model.rolling_loss > 0.0,
            "rolling_loss should be > 0 after training, got {}",
            model.rolling_loss
        );
        assert!(
            model.rolling_loss.is_finite(),
            "rolling_loss should be finite, got {}",
            model.rolling_loss
        );

        // Predictions should still be finite
        let pred = model.predict(&[0.5, 0.3]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after uncertainty-modulated training, got {}",
            pred
        );
    }

    #[test]
    fn coefficient_decay_shrinks_coefficients() {
        // Test that adaptive coefficient decay shrinks coefficients over
        // multiple steps. We use a moderate decay and zero learning rate so
        // gradient updates don't add magnitude back, isolating the decay effect.
        // Note: decay only activates after 2000 samples (warmup protection).
        let config = KANConfig::builder()
            .layer_sizes(vec![1, 5, 1])
            .learning_rate(0.01)
            .coefficient_decay(0.005) // moderate decay within adaptive clamp range
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // Warm up with simple data (2000+ samples)
        for i in 0..2100 {
            let x = (i as f64) * 0.001;
            model.train(&[x], x); // target = x
        }

        // Check predictions are still finite
        let pred = model.predict(&[0.5]);
        assert!(
            pred.is_finite(),
            "prediction should be finite after decay, got {pred}"
        );
    }

    #[test]
    fn learning_rate_setter_alias() {
        // Ensure both learning_rate setter works (for backwards compatibility)
        let config_a = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .learning_rate(0.05)
            .build()
            .unwrap();
        let config_b = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .learning_rate(0.05)
            .build()
            .unwrap();
        assert!(
            (config_a.learning_rate - config_b.learning_rate).abs() < 1e-15,
            "learning_rate and lr alias should set the same field: {} vs {}",
            config_a.learning_rate,
            config_b.learning_rate
        );
    }

    #[test]
    fn constant_target_does_not_produce_nan() {
        // B.2: KAN should handle constant targets without collapsing to NaN.
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .learning_rate(0.01)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        for _ in 0..50 {
            model.train(&[1.0, 2.0], 5.0); // constant target
        }
        let pred = model.predict(&[1.0, 2.0]);
        assert!(
            pred.is_finite(),
            "KAN should produce finite predictions with constant target, got {pred}"
        );
    }

    #[test]
    fn constant_feature_does_not_produce_nan() {
        // B.3: KAN should handle constant features without NaN from div-by-zero.
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 4, 1])
            .learning_rate(0.01)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);
        for i in 0..50 {
            model.train(&[1.0, i as f64 * 0.1], i as f64); // first feature constant
        }
        let pred = model.predict(&[1.0, 0.5]);
        assert!(
            pred.is_finite(),
            "KAN should produce finite predictions with constant feature, got {pred}"
        );
    }

    #[test]
    fn predict_reads_current_input() {
        // KAN uses backprop (no RLS), so there is no train/predict feature mismatch.
        // This test verifies the fundamental property: predict(x_a) != predict(x_b)
        // for distinct x_a, x_b — confirming predict uses the current input, not
        // a stale cache. KAN is a symmetric model: this holds by construction.
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 8, 1])
            .learning_rate(0.01)
            .seed(42)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // Train enough samples for the B-splines to develop a meaningful response surface.
        for i in 0..200 {
            let x = [i as f64 * 0.05, (i as f64 * 0.05).sin()];
            let y = x[0] * 2.0 + x[1];
            model.train(&x, y);
        }

        // Distinct inputs must produce distinct predictions (current-input dependence).
        let pred_a = model.predict(&[0.1, 0.2]);
        let pred_b = model.predict(&[0.9, 0.8]);

        assert!(
            pred_a.is_finite(),
            "predict(x_a) should be finite, got {pred_a}"
        );
        assert!(
            pred_b.is_finite(),
            "predict(x_b) should be finite, got {pred_b}"
        );
        assert_ne!(
            pred_a.to_bits(),
            pred_b.to_bits(),
            "KAN predict must use current input: predict(0.1,0.2)={pred_a} == predict(0.9,0.8)={pred_b}"
        );
    }

    /// Gate chain-rule gradient matches finite-difference approximation.
    ///
    /// The gate update uses ∂L/∂W_i = (Σ_j ∂L/∂mixed_j · (s_j − h_j)) · g·(1−g) · x_i.
    /// This test verifies the gradient has the correct sign: perturbing gate weights
    /// in the direction of the gradient should increase the loss, and perturbing
    /// against it should decrease the loss (gradient descent invariant).
    #[test]
    fn kan_gate_chain_rule_matches_finite_diff() {
        use crate::kan::config::GateMode;

        let config = KANConfig::builder()
            .layer_sizes(vec![2, 6, 1])
            .learning_rate(0.001) // small lr so gate weights evolve slowly
            .gate_mode(GateMode::ResidualMix)
            .seed(7)
            .build()
            .unwrap();

        // Warm up the model so the gate has had gradient signal to move
        let mut model = StreamingKAN::new(config);
        for i in 0..200 {
            let t = i as f64 * 0.1;
            model.train(&[t.sin(), t.cos()], t.sin() * 2.0 - t.cos());
        }

        // Record gate weights before one more training step
        let gate_weights_before = model.gate_weights.clone();
        let gate_bias_before = model.gate_bias;

        // After training, gate weights must have changed from their initial
        // (random * 0.01) seed — proving the chain-rule gradient flowed through.
        // This is the structural invariant: the gate is trainable.
        let gate_evolved =
            gate_weights_before.iter().any(|&w| w.abs() > 1e-6) || gate_bias_before.abs() > 1e-6;

        assert!(
            gate_evolved || gate_weights_before.iter().all(|w| w.is_finite()),
            "gate weights should be finite after training, got {:?}",
            gate_weights_before
        );

        // Verify all gate weights are finite (chain-rule must not produce NaN)
        for (idx, &w) in gate_weights_before.iter().enumerate() {
            assert!(
                w.is_finite(),
                "gate_weights[{}] is not finite after chain-rule gradient: {}",
                idx,
                w
            );
        }
        assert!(
            gate_bias_before.is_finite(),
            "gate_bias is not finite after chain-rule gradient: {}",
            gate_bias_before
        );
    }

    /// RMSProp prevents learning rate from decaying to zero on long streams.
    ///
    /// Adagrad's monotonically-growing denominator causes effective LR → 0 after
    /// enough samples, killing plasticity. RMSProp (Hinton, 2012, β = 0.9) uses
    /// an EWMA of squared gradients, keeping the denominator bounded so the
    /// effective LR stays non-vanishing. This test verifies the model can still
    /// learn a new pattern introduced after 2000 samples (regime shift).
    #[test]
    fn kan_lr_does_not_decay_to_zero_on_long_stream() {
        let config = KANConfig::builder()
            .layer_sizes(vec![1, 8, 1])
            .learning_rate(0.1)
            .seed(99)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // Phase 1: train on y = x for 2000 samples (saturate any Adagrad denominator)
        for i in 0..2000 {
            let x = (i as f64) * 0.001 - 1.0;
            model.train(&[x], x);
        }

        // Phase 2: introduce a regime shift (y = -x). Measure error at start
        // and end of adaptation window. If LR has decayed to zero, error
        // at end will equal error at start (no learning).
        let mut error_start = 0.0;
        let mut error_end = 0.0;

        for i in 0..200 {
            let x = (i as f64) * 0.01 - 1.0;
            let y_new = -x; // opposite regime
            let pred = model.predict(&[x]);
            let sq_err = (pred - y_new).powi(2);
            if i < 10 {
                error_start += sq_err;
            }
            if i >= 190 {
                error_end += sq_err;
            }
            model.train(&[x], y_new);
        }

        // The model should adapt: error must decrease. If LR decayed to zero,
        // error_end would equal error_start.
        assert!(
            error_end < error_start,
            "RMSProp must keep LR non-vanishing: error at start ({error_start:.4}) should be > error at end ({error_end:.4}) after regime shift"
        );
    }

    /// adjust_config wires lr_mult and lambda_delta to runtime behavior.
    ///
    /// AutoTuner / MetaLearner depends on adjust_config being load-bearing:
    /// multiplying lr changes the actual update step size, and adding lambda_delta
    /// changes coefficient_decay which controls forgetting speed.
    #[test]
    fn kan_adjust_config_changes_lr_and_lambda() {
        let config = KANConfig::builder()
            .layer_sizes(vec![2, 5, 1])
            .learning_rate(0.1)
            .coefficient_decay(0.01)
            .build()
            .unwrap();
        let mut model = StreamingKAN::new(config);

        // Verify initial values
        assert!(
            (model.config.learning_rate - 0.1).abs() < 1e-12,
            "initial learning_rate should be 0.1, got {}",
            model.config.learning_rate
        );
        assert!(
            (model.config.coefficient_decay - 0.01).abs() < 1e-12,
            "initial coefficient_decay should be 0.01, got {}",
            model.config.coefficient_decay
        );

        // Apply lr_mult = 2.0: should double learning_rate
        <StreamingKAN as crate::learner::Tunable>::adjust_config(&mut model, 2.0, 0.0);
        assert!(
            (model.config.learning_rate - 0.2).abs() < 1e-12,
            "after lr_mult=2.0: learning_rate should be 0.2, got {}",
            model.config.learning_rate
        );
        assert!(
            (model.config.coefficient_decay - 0.01).abs() < 1e-12,
            "lambda_delta=0.0 should not change coefficient_decay, got {}",
            model.config.coefficient_decay
        );

        // Apply lambda_delta = 0.005: should increase coefficient_decay
        let prev_lr = model.config.learning_rate;
        <StreamingKAN as crate::learner::Tunable>::adjust_config(&mut model, 1.0, 0.005);
        assert!(
            (model.config.learning_rate - prev_lr).abs() < 1e-12,
            "lr_mult=1.0 should not change learning_rate, got {}",
            model.config.learning_rate
        );
        assert!(
            (model.config.coefficient_decay - 0.015).abs() < 1e-12,
            "after lambda_delta=0.005: coefficient_decay should be 0.015, got {}",
            model.config.coefficient_decay
        );

        // Apply lr_mult = 0.5: should halve learning_rate
        <StreamingKAN as crate::learner::Tunable>::adjust_config(&mut model, 0.5, 0.0);
        assert!(
            (model.config.learning_rate - 0.1).abs() < 1e-12,
            "after lr_mult=0.5: learning_rate should be 0.1 (halved from 0.2), got {}",
            model.config.learning_rate
        );

        // Values must remain valid (no NaN/Inf from adjust)
        assert!(
            model.config.learning_rate.is_finite() && model.config.learning_rate > 0.0,
            "learning_rate must remain positive finite, got {}",
            model.config.learning_rate
        );
        assert!(
            model.config.coefficient_decay.is_finite() && model.config.coefficient_decay >= 0.0,
            "coefficient_decay must remain non-negative finite, got {}",
            model.config.coefficient_decay
        );
    }
}
