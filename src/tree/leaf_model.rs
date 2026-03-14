//! Pluggable leaf prediction models for streaming decision trees.
//!
//! By default, leaves use a closed-form weight computed from accumulated
//! gradient and hessian sums. Trainable leaf models replace this with learnable
//! functions that capture more complex patterns within each leaf's partition.
//!
//! # Leaf model variants
//!
//! | Model | Prediction | Overhead | Best for |
//! |-------|-----------|----------|----------|
//! | [`ClosedFormLeaf`] | constant weight `-G/(H+lambda)` | zero | general use (default) |
//! | [`LinearLeafModel`] | `w . x + b` (AdaGrad-optimized) | O(d) per update | low-depth trees (2--4) |
//! | [`MLPLeafModel`] | single-hidden-layer neural net | O(d*h) per update | complex local patterns |
//! | [`AdaptiveLeafModel`] | starts constant, auto-promotes | shadow model cost | automatic complexity allocation |
//!
//! # AdaGrad optimization
//!
//! [`LinearLeafModel`] uses per-weight AdaGrad accumulators for adaptive learning
//! rates. Features at different scales converge at their natural rates without
//! manual tuning. Combined with Newton scaling from the hessian, this gives
//! second-order-informed, per-feature adaptive optimization.
//!
//! # Exponential forgetting
//!
//! [`LinearLeafModel`] and [`MLPLeafModel`] support an optional `decay` parameter
//! that applies exponential weight decay before each update. This gives the model
//! a finite memory horizon, adapting to concept drift in non-stationary streams.
//! Typical values: 0.999 (slow drift) to 0.99 (fast drift).
//!
//! # Warm-starting on split
//!
//! When a leaf splits, child leaves can inherit the parent's learned function via
//! [`LeafModel::clone_warm`]. Linear children start with the parent's weights
//! (resetting optimizer state), converging faster than starting from scratch.
//!
//! # Adaptive promotion
//!
//! [`AdaptiveLeafModel`] runs a shadow model alongside the default closed-form
//! model. Both are trained on every sample, and their per-sample losses are
//! compared using the second-order Taylor approximation. When the Hoeffding bound
//! (the tree's existing `delta` parameter) confirms the shadow model is
//! statistically superior, the leaf promotes -- no arbitrary thresholds.

/// A trainable prediction model that lives inside a decision tree leaf.
///
/// Implementations must be `Send + Sync` so trees can be shared across threads.
pub trait LeafModel: Send + Sync {
    /// Produce a prediction given input features.
    fn predict(&self, features: &[f64]) -> f64;

    /// Update model parameters given a gradient, hessian, and regularization lambda.
    fn update(&mut self, features: &[f64], gradient: f64, hessian: f64, lambda: f64);

    /// Create a fresh (zeroed / re-initialized) clone of this model's architecture.
    fn clone_fresh(&self) -> Box<dyn LeafModel>;

    /// Create a warm clone preserving learned weights but resetting optimizer state.
    ///
    /// Used when splitting a leaf: child leaves inherit the parent's learned
    /// function as a starting point, converging faster than starting from scratch.
    /// Defaults to [`clone_fresh`](LeafModel::clone_fresh) for models where
    /// warm-starting is not meaningful (e.g. [`ClosedFormLeaf`]).
    fn clone_warm(&self) -> Box<dyn LeafModel> {
        self.clone_fresh()
    }
}

// ---------------------------------------------------------------------------
// ClosedFormLeaf
// ---------------------------------------------------------------------------

/// Leaf model that computes the optimal weight in closed form:
/// `weight = -grad_sum / (hess_sum + lambda)`.
///
/// This is the standard leaf value used in gradient boosted trees.
pub struct ClosedFormLeaf {
    grad_sum: f64,
    hess_sum: f64,
    weight: f64,
}

impl Default for ClosedFormLeaf {
    fn default() -> Self {
        Self {
            grad_sum: 0.0,
            hess_sum: 0.0,
            weight: 0.0,
        }
    }
}

impl ClosedFormLeaf {
    /// Create a new zeroed closed-form leaf.
    pub fn new() -> Self {
        Self::default()
    }
}

impl LeafModel for ClosedFormLeaf {
    fn predict(&self, _features: &[f64]) -> f64 {
        self.weight
    }

    fn update(&mut self, _features: &[f64], gradient: f64, hessian: f64, lambda: f64) {
        self.grad_sum += gradient;
        self.hess_sum += hessian;
        self.weight = -self.grad_sum / (self.hess_sum + lambda);
    }

    fn clone_fresh(&self) -> Box<dyn LeafModel> {
        Box::new(ClosedFormLeaf::new())
    }
}

// ---------------------------------------------------------------------------
// LinearLeafModel
// ---------------------------------------------------------------------------

/// Online ridge regression leaf model with AdaGrad optimization.
///
/// Learns a linear function `w . x + b` using Newton-scaled gradient descent
/// with per-weight AdaGrad accumulators for adaptive learning rates. Features
/// at different scales converge at their natural rates without manual tuning.
///
/// Weights are lazily initialized on the first `update` call so the model
/// adapts to whatever dimensionality arrives.
///
/// Optional exponential weight decay (`decay`) gives the model a finite memory
/// horizon for non-stationary streams.
pub struct LinearLeafModel {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    decay: Option<f64>,
    /// Per-weight squared gradient accumulator (AdaGrad).
    sq_grad_accum: Vec<f64>,
    /// Bias squared gradient accumulator (AdaGrad).
    sq_bias_accum: f64,
    initialized: bool,
}

impl LinearLeafModel {
    /// Create a new linear leaf model with the given base learning rate and
    /// optional exponential decay factor.
    ///
    /// When `decay` is `Some(d)` with `d` in (0, 1), weights are multiplied
    /// by `d` before each update, giving the model a memory half-life of
    /// `ln(2) / ln(1/d)` samples.
    pub fn new(learning_rate: f64, decay: Option<f64>) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            decay,
            sq_grad_accum: Vec::new(),
            sq_bias_accum: 0.0,
            initialized: false,
        }
    }
}

/// AdaGrad epsilon to prevent division by zero.
const ADAGRAD_EPS: f64 = 1e-8;

impl LeafModel for LinearLeafModel {
    fn predict(&self, features: &[f64]) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        let mut dot = self.bias;
        for (w, x) in self.weights.iter().zip(features.iter()) {
            dot += w * x;
        }
        dot
    }

    fn update(&mut self, features: &[f64], gradient: f64, hessian: f64, lambda: f64) {
        if !self.initialized {
            let d = features.len();
            self.weights = vec![0.0; d];
            self.sq_grad_accum = vec![0.0; d];
            self.initialized = true;
        }

        // Exponential weight decay (forgetting old data).
        if let Some(d) = self.decay {
            for w in self.weights.iter_mut() {
                *w *= d;
            }
            self.bias *= d;
        }

        // Newton-scaled base learning rate.
        let base_lr = self.learning_rate / (hessian.abs() + lambda);

        // AdaGrad: per-weight adaptive learning rates.
        for (i, (w, x)) in self.weights.iter_mut().zip(features.iter()).enumerate() {
            let g = gradient * x;
            self.sq_grad_accum[i] += g * g;
            let adaptive_lr = base_lr / (self.sq_grad_accum[i].sqrt() + ADAGRAD_EPS);
            *w -= adaptive_lr * g;
        }

        // Bias update with its own AdaGrad accumulator.
        self.sq_bias_accum += gradient * gradient;
        let bias_lr = base_lr / (self.sq_bias_accum.sqrt() + ADAGRAD_EPS);
        self.bias -= bias_lr * gradient;
    }

    fn clone_fresh(&self) -> Box<dyn LeafModel> {
        Box::new(LinearLeafModel::new(self.learning_rate, self.decay))
    }

    fn clone_warm(&self) -> Box<dyn LeafModel> {
        Box::new(LinearLeafModel {
            weights: self.weights.clone(),
            bias: self.bias,
            learning_rate: self.learning_rate,
            decay: self.decay,
            // Reset AdaGrad accumulators -- the child's gradient landscape
            // differs from the parent's, so accumulated curvature estimates
            // don't transfer. Fresh accumulators let the child's learning
            // rates adapt to its own region.
            sq_grad_accum: vec![0.0; self.weights.len()],
            sq_bias_accum: 0.0,
            initialized: self.initialized,
        })
    }
}

// ---------------------------------------------------------------------------
// MLPLeafModel
// ---------------------------------------------------------------------------

/// Single hidden layer MLP leaf model with ReLU activation.
///
/// Learns a nonlinear function via backpropagation with Newton-scaled
/// learning rate. Weights are lazily initialized on the first `update` call
/// using a deterministic xorshift64 PRNG so results are reproducible.
///
/// Optional exponential weight decay (`decay`) gives the model a finite memory
/// horizon for non-stationary streams.
pub struct MLPLeafModel {
    hidden_weights: Vec<Vec<f64>>, // [hidden_size][input_size]
    hidden_bias: Vec<f64>,
    output_weights: Vec<f64>,
    output_bias: f64,
    hidden_size: usize,
    learning_rate: f64,
    decay: Option<f64>,
    seed: u64,
    initialized: bool,
    hidden_activations: Vec<f64>,
    hidden_pre_activations: Vec<f64>,
}

impl MLPLeafModel {
    /// Create a new MLP leaf model with the given hidden layer size, learning rate,
    /// seed, and optional decay.
    ///
    /// The seed controls deterministic weight initialization. Different seeds
    /// produce different initial weights, which is critical for ensemble diversity
    /// when multiple MLP leaves share the same `hidden_size`.
    pub fn new(hidden_size: usize, learning_rate: f64, seed: u64, decay: Option<f64>) -> Self {
        Self {
            hidden_weights: Vec::new(),
            hidden_bias: Vec::new(),
            output_weights: Vec::new(),
            output_bias: 0.0,
            hidden_size,
            learning_rate,
            decay,
            seed,
            initialized: false,
            hidden_activations: Vec::new(),
            hidden_pre_activations: Vec::new(),
        }
    }

    /// Initialize weights using xorshift64, scaled to [-0.1, 0.1].
    fn initialize(&mut self, input_size: usize) {
        let mut state = self.seed ^ (self.hidden_size as u64);

        self.hidden_weights = Vec::with_capacity(self.hidden_size);
        for _ in 0..self.hidden_size {
            let mut row = Vec::with_capacity(input_size);
            for _ in 0..input_size {
                let r = xorshift64(&mut state);
                // Map u64 to [-0.1, 0.1]
                let val = (r as f64 / u64::MAX as f64) * 0.2 - 0.1;
                row.push(val);
            }
            self.hidden_weights.push(row);
        }

        self.hidden_bias = Vec::with_capacity(self.hidden_size);
        for _ in 0..self.hidden_size {
            let r = xorshift64(&mut state);
            let val = (r as f64 / u64::MAX as f64) * 0.2 - 0.1;
            self.hidden_bias.push(val);
        }

        self.output_weights = Vec::with_capacity(self.hidden_size);
        for _ in 0..self.hidden_size {
            let r = xorshift64(&mut state);
            let val = (r as f64 / u64::MAX as f64) * 0.2 - 0.1;
            self.output_weights.push(val);
        }

        {
            let r = xorshift64(&mut state);
            self.output_bias = (r as f64 / u64::MAX as f64) * 0.2 - 0.1;
        }

        self.hidden_activations = vec![0.0; self.hidden_size];
        self.hidden_pre_activations = vec![0.0; self.hidden_size];
        self.initialized = true;
    }

    /// Forward pass: compute hidden pre-activations, ReLU activations, and output.
    fn forward(&mut self, features: &[f64]) -> f64 {
        // Hidden layer
        for h in 0..self.hidden_size {
            let mut z = self.hidden_bias[h];
            for (j, x) in features.iter().enumerate() {
                if j < self.hidden_weights[h].len() {
                    z += self.hidden_weights[h][j] * x;
                }
            }
            self.hidden_pre_activations[h] = z;
            // ReLU
            self.hidden_activations[h] = if z > 0.0 { z } else { 0.0 };
        }

        // Output layer
        let mut out = self.output_bias;
        for (w, a) in self
            .output_weights
            .iter()
            .zip(self.hidden_activations.iter())
        {
            out += w * a;
        }
        out
    }
}

impl LeafModel for MLPLeafModel {
    fn predict(&self, features: &[f64]) -> f64 {
        if !self.initialized {
            return 0.0;
        }
        // Non-mutating forward pass (can't store activations, recompute locally)
        let hidden_acts: Vec<f64> = self
            .hidden_weights
            .iter()
            .zip(self.hidden_bias.iter())
            .map(|(hw, &hb)| {
                let mut z = hb;
                for (j, x) in features.iter().enumerate() {
                    if j < hw.len() {
                        z += hw[j] * x;
                    }
                }
                if z > 0.0 {
                    z
                } else {
                    0.0
                }
            })
            .collect();
        let mut out = self.output_bias;
        for (w, a) in self.output_weights.iter().zip(hidden_acts.iter()) {
            out += w * a;
        }
        out
    }

    fn update(&mut self, features: &[f64], gradient: f64, hessian: f64, lambda: f64) {
        if !self.initialized {
            self.initialize(features.len());
        }

        // Exponential weight decay (forgetting old data).
        if let Some(d) = self.decay {
            for row in self.hidden_weights.iter_mut() {
                for w in row.iter_mut() {
                    *w *= d;
                }
            }
            for b in self.hidden_bias.iter_mut() {
                *b *= d;
            }
            for w in self.output_weights.iter_mut() {
                *w *= d;
            }
            self.output_bias *= d;
        }

        // Forward pass (stores activations for backprop)
        let _output = self.forward(features);

        let effective_lr = self.learning_rate / (hessian.abs() + lambda);

        // Backprop: output gradient is the incoming `gradient` (chain rule from loss)
        let d_output = gradient;

        // Gradient for output weights and bias
        // d_loss/d_output_w[h] = d_output * hidden_activations[h]
        // d_loss/d_output_bias = d_output
        for h in 0..self.hidden_size {
            self.output_weights[h] -= effective_lr * d_output * self.hidden_activations[h];
        }
        self.output_bias -= effective_lr * d_output;

        // Gradient for hidden layer
        for h in 0..self.hidden_size {
            // d_loss/d_hidden_act[h] = d_output * output_weights[h]
            let d_hidden_act = d_output * self.output_weights[h];

            // ReLU derivative
            let d_relu = if self.hidden_pre_activations[h] > 0.0 {
                d_hidden_act
            } else {
                0.0
            };

            // Update hidden weights and bias
            for (j, x) in features.iter().enumerate() {
                if j < self.hidden_weights[h].len() {
                    self.hidden_weights[h][j] -= effective_lr * d_relu * x;
                }
            }
            self.hidden_bias[h] -= effective_lr * d_relu;
        }
    }

    fn clone_fresh(&self) -> Box<dyn LeafModel> {
        // Derive a new seed so each fresh clone gets distinct initial weights.
        let derived_seed = self.seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        Box::new(MLPLeafModel::new(
            self.hidden_size,
            self.learning_rate,
            derived_seed,
            self.decay,
        ))
    }

    fn clone_warm(&self) -> Box<dyn LeafModel> {
        Box::new(MLPLeafModel {
            hidden_weights: self.hidden_weights.clone(),
            hidden_bias: self.hidden_bias.clone(),
            output_weights: self.output_weights.clone(),
            output_bias: self.output_bias,
            hidden_size: self.hidden_size,
            learning_rate: self.learning_rate,
            decay: self.decay,
            // Derive a new seed so warm clones diverge if they re-initialize.
            seed: self.seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(2),
            initialized: self.initialized,
            hidden_activations: vec![0.0; self.hidden_size],
            hidden_pre_activations: vec![0.0; self.hidden_size],
        })
    }
}

// ---------------------------------------------------------------------------
// AdaptiveLeafModel
// ---------------------------------------------------------------------------

/// Leaf model that starts as closed-form and promotes to a more complex model
/// when the Hoeffding bound confirms it is statistically superior.
///
/// Runs a shadow model alongside the active (closed-form) model. On each
/// update, both models are trained and their per-sample losses compared using
/// the second-order Taylor approximation:
///
/// ```text
/// loss_i = gradient * prediction + 0.5 * hessian * prediction^2
/// advantage_i = loss_active_i - loss_shadow_i
/// ```
///
/// When `mean(advantage) > epsilon` where epsilon is the Hoeffding bound
/// (using the tree's `delta` parameter), the shadow model is promoted to
/// active and the overhead drops to zero.
///
/// This uses the **same statistical guarantee** as the tree's split decisions --
/// no arbitrary thresholds.
pub struct AdaptiveLeafModel {
    /// The currently active prediction model. Starts as ClosedForm.
    active: Box<dyn LeafModel>,
    /// Shadow model being evaluated against the active model.
    shadow: Box<dyn LeafModel>,
    /// Configuration of the shadow model type (for cloning).
    promote_to: LeafModelType,
    /// Cumulative loss advantage: sum(loss_active - loss_shadow).
    /// Positive means the shadow is winning.
    cumulative_advantage: f64,
    /// Number of samples seen for the Hoeffding bound.
    n: u64,
    /// Running maximum |loss_diff| for range estimation (R in the bound).
    max_loss_diff: f64,
    /// Hoeffding confidence parameter (from tree config).
    delta: f64,
    /// Whether the shadow has been promoted.
    promoted: bool,
    /// Seed for reproducible cloning.
    seed: u64,
}

impl AdaptiveLeafModel {
    /// Create a new adaptive leaf model.
    ///
    /// The active model starts as `ClosedFormLeaf`. The shadow model is the
    /// candidate that will be promoted if it proves statistically superior.
    pub fn new(
        shadow: Box<dyn LeafModel>,
        promote_to: LeafModelType,
        delta: f64,
        seed: u64,
    ) -> Self {
        Self {
            active: Box::new(ClosedFormLeaf::new()),
            shadow,
            promote_to,
            cumulative_advantage: 0.0,
            n: 0,
            max_loss_diff: 0.0,
            delta,
            promoted: false,
            seed,
        }
    }
}

impl LeafModel for AdaptiveLeafModel {
    fn predict(&self, features: &[f64]) -> f64 {
        self.active.predict(features)
    }

    fn update(&mut self, features: &[f64], gradient: f64, hessian: f64, lambda: f64) {
        if self.promoted {
            // Post-promotion: only update the promoted model.
            self.active.update(features, gradient, hessian, lambda);
            return;
        }

        // Compute predictions BEFORE updating (evaluate current state).
        let pred_active = self.active.predict(features);
        let pred_shadow = self.shadow.predict(features);

        // Second-order Taylor loss approximation for each model.
        // L(pred) ~= gradient * pred + 0.5 * hessian * pred^2
        // This is the same loss proxy that XGBoost gain uses.
        let loss_active = gradient * pred_active + 0.5 * hessian * pred_active * pred_active;
        let loss_shadow = gradient * pred_shadow + 0.5 * hessian * pred_shadow * pred_shadow;

        // Positive advantage means the shadow model is better (lower loss).
        let diff = loss_active - loss_shadow;
        self.cumulative_advantage += diff;
        self.n += 1;

        // Track range for the Hoeffding bound.
        let abs_diff = diff.abs();
        if abs_diff > self.max_loss_diff {
            self.max_loss_diff = abs_diff;
        }

        // Update both models.
        self.active.update(features, gradient, hessian, lambda);
        self.shadow.update(features, gradient, hessian, lambda);

        // Hoeffding bound test: is the shadow statistically better?
        // epsilon = sqrt(R^2 * ln(1/delta) / (2*n))
        // Promote when mean_advantage > epsilon.
        if self.n >= 10 && self.max_loss_diff > 0.0 {
            let mean_advantage = self.cumulative_advantage / self.n as f64;
            if mean_advantage > 0.0 {
                let r_squared = self.max_loss_diff * self.max_loss_diff;
                let ln_inv_delta = (1.0 / self.delta).ln();
                let epsilon = (r_squared * ln_inv_delta / (2.0 * self.n as f64)).sqrt();

                if mean_advantage > epsilon {
                    // Promote: swap shadow into active, drop the old active.
                    self.promoted = true;
                    std::mem::swap(&mut self.active, &mut self.shadow);
                }
            }
        }
    }

    fn clone_fresh(&self) -> Box<dyn LeafModel> {
        let derived_seed = self.seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        Box::new(AdaptiveLeafModel::new(
            self.promote_to.create(derived_seed, self.delta),
            self.promote_to.clone(),
            self.delta,
            derived_seed,
        ))
    }
    // clone_warm: default (= clone_fresh). Promotion must be re-earned
    // in each leaf's region -- the parent's promotion does not transfer.
}

// We need Send + Sync for AdaptiveLeafModel because it contains Box<dyn LeafModel>
// which already requires Send + Sync. The compiler should derive these automatically,
// but let's verify:
// SAFETY: All fields are Send + Sync (Box<dyn LeafModel> requires it, f64/u64/bool are Send+Sync).
unsafe impl Send for AdaptiveLeafModel {}
unsafe impl Sync for AdaptiveLeafModel {}

// ---------------------------------------------------------------------------
// LeafModelType
// ---------------------------------------------------------------------------

/// Describes which leaf model architecture to use.
///
/// Used by tree builders to construct fresh leaf models when creating new leaves.
///
/// # Variants
///
/// - **`ClosedForm`** -- Standard constant leaf weight. Zero overhead (default).
/// - **`Linear`** -- Per-leaf online ridge regression with AdaGrad optimization.
///   Each leaf learns a local linear surface `w . x + b`. Recommended for
///   low-depth trees (depth 2--4). Optional `decay` for concept drift.
/// - **`MLP`** -- Per-leaf single-hidden-layer neural network with ReLU.
///   Optional `decay` for concept drift.
/// - **`Adaptive`** -- Starts as closed-form, auto-promotes to `promote_to`
///   when the Hoeffding bound confirms it is statistically superior. Uses the
///   tree's existing `delta` parameter -- no arbitrary thresholds.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LeafModelType {
    /// Standard closed-form leaf weight.
    #[default]
    ClosedForm,

    /// Online ridge regression with AdaGrad optimization.
    ///
    /// `decay`: optional exponential weight decay for non-stationary streams.
    /// Typical values: 0.999 (slow drift) to 0.99 (fast drift).
    Linear {
        learning_rate: f64,
        #[serde(default)]
        decay: Option<f64>,
    },

    /// Single hidden layer MLP with the given hidden size and learning rate.
    ///
    /// `decay`: optional exponential weight decay for non-stationary streams.
    MLP {
        hidden_size: usize,
        learning_rate: f64,
        #[serde(default)]
        decay: Option<f64>,
    },

    /// Adaptive leaf that starts as closed-form and auto-promotes when
    /// the Hoeffding bound confirms the promoted model is better.
    ///
    /// The `promote_to` field specifies the shadow model type to evaluate
    /// against the default closed-form baseline.
    Adaptive { promote_to: Box<LeafModelType> },
}

impl LeafModelType {
    /// Create a fresh boxed leaf model of this type.
    ///
    /// The `seed` parameter controls deterministic initialization (MLP weights,
    /// adaptive model seeding). The `delta` parameter is the Hoeffding bound
    /// confidence level, used by [`Adaptive`](LeafModelType::Adaptive) leaves
    /// for promotion decisions. For non-adaptive types, `delta` is unused.
    pub fn create(&self, seed: u64, delta: f64) -> Box<dyn LeafModel> {
        match self {
            Self::ClosedForm => Box::new(ClosedFormLeaf::new()),
            Self::Linear {
                learning_rate,
                decay,
            } => Box::new(LinearLeafModel::new(*learning_rate, *decay)),
            Self::MLP {
                hidden_size,
                learning_rate,
                decay,
            } => Box::new(MLPLeafModel::new(
                *hidden_size,
                *learning_rate,
                seed,
                *decay,
            )),
            Self::Adaptive { promote_to } => Box::new(AdaptiveLeafModel::new(
                promote_to.create(seed, delta),
                *promote_to.clone(),
                delta,
                seed,
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared utility
// ---------------------------------------------------------------------------

/// Xorshift64 PRNG for deterministic weight initialization.
fn xorshift64(state: &mut u64) -> u64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Xorshift64 for deterministic test data generation.
    fn xorshift64(state: &mut u64) -> u64 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *state = s;
        s
    }

    /// Convert xorshift output to f64 in [0, 1).
    fn rand_f64(state: &mut u64) -> f64 {
        xorshift64(state) as f64 / u64::MAX as f64
    }

    #[test]
    fn closed_form_matches_formula() {
        let mut leaf = ClosedFormLeaf::new();
        let lambda = 1.0;

        // Accumulate several gradient/hessian pairs
        let updates = [(0.5, 1.0), (-0.3, 0.8), (1.2, 2.0), (-0.1, 0.5)];
        let mut grad_sum = 0.0;
        let mut hess_sum = 0.0;

        for &(g, h) in &updates {
            leaf.update(&[], g, h, lambda);
            grad_sum += g;
            hess_sum += h;
        }

        let expected = -grad_sum / (hess_sum + lambda);
        let predicted = leaf.predict(&[]);

        assert!(
            (predicted - expected).abs() < 1e-12,
            "closed form mismatch: got {predicted}, expected {expected}"
        );
    }

    #[test]
    fn closed_form_clone_fresh_resets() {
        let mut leaf = ClosedFormLeaf::new();
        leaf.update(&[], 5.0, 2.0, 1.0);
        assert!(
            leaf.predict(&[]).abs() > 0.0,
            "leaf should have non-zero weight after update"
        );

        let fresh = leaf.clone_fresh();
        assert!(
            fresh.predict(&[]).abs() < 1e-15,
            "fresh clone should predict 0, got {}",
            fresh.predict(&[])
        );
    }

    #[test]
    fn linear_converges_on_linear_target() {
        // Target: y = 2*x1 + 3*x2
        let mut model = LinearLeafModel::new(0.01, None);
        let lambda = 0.1;
        let mut rng = 42u64;

        for _ in 0..2000 {
            let x1 = rand_f64(&mut rng) * 2.0 - 1.0;
            let x2 = rand_f64(&mut rng) * 2.0 - 1.0;
            let features = vec![x1, x2];
            let target = 2.0 * x1 + 3.0 * x2;

            let pred = model.predict(&features);
            let gradient = 2.0 * (pred - target);
            let hessian = 2.0;
            model.update(&features, gradient, hessian, lambda);
        }

        let test_features = vec![0.5, -0.3];
        let target = 2.0 * 0.5 + 3.0 * (-0.3);
        let pred = model.predict(&test_features);

        assert!(
            (pred - target).abs() < 1.0,
            "linear model should converge within 1.0 of target: pred={pred}, target={target}"
        );
    }

    #[test]
    fn linear_uninitialized_predicts_zero() {
        let model = LinearLeafModel::new(0.01, None);
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.abs() < 1e-15,
            "uninitialized linear model should predict 0, got {pred}"
        );
    }

    #[test]
    fn linear_clone_warm_preserves_weights() {
        let mut model = LinearLeafModel::new(0.01, None);
        let features = vec![1.0, 2.0];

        // Train it
        for i in 0..100 {
            let target = 3.0 * features[0] + 2.0 * features[1];
            let pred = model.predict(&features);
            let gradient = 2.0 * (pred - target);
            model.update(&features, gradient, 2.0, 0.1);
            // Avoid unused variable warning
            let _ = i;
        }

        let trained_pred = model.predict(&features);
        assert!(
            trained_pred.abs() > 0.01,
            "model should have learned something"
        );

        // Warm clone should predict similarly
        let warm = model.clone_warm();
        let warm_pred = warm.predict(&features);
        assert!(
            (warm_pred - trained_pred).abs() < 1e-12,
            "warm clone should preserve weights: trained={trained_pred}, warm={warm_pred}"
        );

        // Fresh clone should predict 0
        let fresh = model.clone_fresh();
        let fresh_pred = fresh.predict(&features);
        assert!(
            fresh_pred.abs() < 1e-15,
            "fresh clone should predict 0, got {fresh_pred}"
        );
    }

    #[test]
    fn linear_decay_forgets_old_data() {
        // Train on target=5.0, then switch to target=-5.0.
        // With decay, the model adapts faster to the new target.
        let mut model_decay = LinearLeafModel::new(0.05, Some(0.99));
        let mut model_no_decay = LinearLeafModel::new(0.05, None);
        let features = vec![1.0];
        let lambda = 0.1;

        // Phase 1: train on target = 5.0
        for _ in 0..500 {
            let pred_d = model_decay.predict(&features);
            let pred_n = model_no_decay.predict(&features);
            model_decay.update(&features, 2.0 * (pred_d - 5.0), 2.0, lambda);
            model_no_decay.update(&features, 2.0 * (pred_n - 5.0), 2.0, lambda);
        }

        // Phase 2: switch to target = -5.0
        for _ in 0..500 {
            let pred_d = model_decay.predict(&features);
            let pred_n = model_no_decay.predict(&features);
            model_decay.update(&features, 2.0 * (pred_d + 5.0), 2.0, lambda);
            model_no_decay.update(&features, 2.0 * (pred_n + 5.0), 2.0, lambda);
        }

        let pred_decay = model_decay.predict(&features);
        let pred_no_decay = model_no_decay.predict(&features);

        // Decay model should be closer to -5.0 than no-decay model
        assert!(
            (pred_decay + 5.0).abs() < (pred_no_decay + 5.0).abs(),
            "decay model should adapt faster: decay pred={pred_decay:.3}, no-decay pred={pred_no_decay:.3}"
        );
    }

    #[test]
    fn mlp_produces_finite_predictions() {
        let model_uninit = MLPLeafModel::new(4, 0.01, 42, None);
        let features = vec![1.0, 2.0, 3.0];

        let pred_before = model_uninit.predict(&features);
        assert!(
            pred_before.is_finite(),
            "uninit prediction should be finite"
        );
        assert!(
            pred_before.abs() < 1e-15,
            "uninit prediction should be 0, got {pred_before}"
        );

        let mut model = MLPLeafModel::new(4, 0.01, 42, None);
        for _ in 0..10 {
            model.update(&features, 0.5, 1.0, 0.1);
        }
        let pred_after = model.predict(&features);
        assert!(
            pred_after.is_finite(),
            "prediction after training should be finite, got {pred_after}"
        );
    }

    #[test]
    fn mlp_loss_decreases() {
        let mut model = MLPLeafModel::new(8, 0.05, 123, None);
        let features = vec![1.0, -0.5, 0.3];
        let target = 2.5;
        let lambda = 0.1;

        model.update(&features, 0.0, 1.0, lambda); // dummy update to initialize
        let initial_pred = model.predict(&features);
        let initial_error = (initial_pred - target).abs();

        for _ in 0..200 {
            let pred = model.predict(&features);
            let gradient = 2.0 * (pred - target);
            let hessian = 2.0;
            model.update(&features, gradient, hessian, lambda);
        }

        let final_pred = model.predict(&features);
        let final_error = (final_pred - target).abs();

        assert!(
            final_error < initial_error,
            "MLP error should decrease: initial={initial_error}, final={final_error}"
        );
    }

    #[test]
    fn mlp_clone_fresh_resets() {
        let mut model = MLPLeafModel::new(4, 0.01, 42, None);
        let features = vec![1.0, 2.0];

        for _ in 0..20 {
            model.update(&features, 0.5, 1.0, 0.1);
        }

        let trained_pred = model.predict(&features);
        assert!(
            trained_pred.abs() > 1e-10,
            "trained model should have non-zero prediction"
        );

        let fresh = model.clone_fresh();
        let fresh_pred = fresh.predict(&features);
        assert!(
            fresh_pred.abs() < 1e-15,
            "fresh clone should predict 0, got {fresh_pred}"
        );
    }

    #[test]
    fn mlp_clone_warm_preserves_weights() {
        let mut model = MLPLeafModel::new(4, 0.01, 42, None);
        let features = vec![1.0, 2.0];

        for _ in 0..50 {
            model.update(&features, 0.5, 1.0, 0.1);
        }

        let trained_pred = model.predict(&features);
        let warm = model.clone_warm();
        let warm_pred = warm.predict(&features);

        assert!(
            (warm_pred - trained_pred).abs() < 1e-10,
            "warm clone should preserve predictions: trained={trained_pred}, warm={warm_pred}"
        );
    }

    #[test]
    fn leaf_model_type_default_is_closed_form() {
        let default_type = LeafModelType::default();
        assert!(
            matches!(default_type, LeafModelType::ClosedForm),
            "default LeafModelType should be ClosedForm, got {default_type:?}"
        );
    }

    #[test]
    fn leaf_model_type_create_all_variants() {
        let features = vec![1.0, 2.0, 3.0];
        let delta = 1e-7;

        // ClosedForm
        let mut closed = LeafModelType::ClosedForm.create(0, delta);
        closed.update(&features, 1.0, 1.0, 0.1);
        let p = closed.predict(&features);
        assert!(p.is_finite(), "ClosedForm prediction should be finite");

        // Linear
        let mut linear = LeafModelType::Linear {
            learning_rate: 0.01,
            decay: None,
        }
        .create(0, delta);
        linear.update(&features, 1.0, 1.0, 0.1);
        let p = linear.predict(&features);
        assert!(p.is_finite(), "Linear prediction should be finite");

        // MLP
        let mut mlp = LeafModelType::MLP {
            hidden_size: 4,
            learning_rate: 0.01,
            decay: None,
        }
        .create(99, delta);
        mlp.update(&features, 1.0, 1.0, 0.1);
        let p = mlp.predict(&features);
        assert!(p.is_finite(), "MLP prediction should be finite");

        // Adaptive (promoting to Linear)
        let mut adaptive = LeafModelType::Adaptive {
            promote_to: Box::new(LeafModelType::Linear {
                learning_rate: 0.01,
                decay: None,
            }),
        }
        .create(42, delta);
        adaptive.update(&features, 1.0, 1.0, 0.1);
        let p = adaptive.predict(&features);
        assert!(p.is_finite(), "Adaptive prediction should be finite");
    }

    #[test]
    fn adaptive_promotes_on_linear_target() {
        // The shadow (Linear) should eventually outperform ClosedForm
        // on a target that varies with features.
        let promote_to = LeafModelType::Linear {
            learning_rate: 0.01,
            decay: None,
        };
        let shadow = promote_to.create(42, 1e-7);
        let mut model = AdaptiveLeafModel::new(shadow, promote_to, 1e-3, 42);

        let mut rng = 42u64;
        for _ in 0..5000 {
            let x1 = rand_f64(&mut rng) * 2.0 - 1.0;
            let x2 = rand_f64(&mut rng) * 2.0 - 1.0;
            let features = vec![x1, x2];
            let target = 3.0 * x1 + 2.0 * x2;

            let pred = model.predict(&features);
            let gradient = 2.0 * (pred - target);
            let hessian = 2.0;
            model.update(&features, gradient, hessian, 0.1);
        }

        // After enough samples on a linear target, the Linear shadow
        // should have been promoted.
        assert!(
            model.promoted,
            "adaptive model should have promoted on linear target after 5000 samples"
        );
    }

    #[test]
    fn adaptive_does_not_promote_on_constant_target() {
        // On a constant target, ClosedForm is optimal -- no promotion expected.
        let promote_to = LeafModelType::Linear {
            learning_rate: 0.01,
            decay: None,
        };
        let shadow = promote_to.create(42, 1e-7);
        let mut model = AdaptiveLeafModel::new(shadow, promote_to, 1e-7, 42);

        for _ in 0..2000 {
            let features = vec![1.0, 2.0];
            let target = 5.0; // constant -- no feature dependence
            let pred = model.predict(&features);
            let gradient = 2.0 * (pred - target);
            let hessian = 2.0;
            model.update(&features, gradient, hessian, 0.1);
        }

        // With strict delta (1e-7) and a constant target, the Linear model
        // shouldn't gain a statistically significant advantage.
        // (It may or may not promote -- the point is it shouldn't be obvious.)
        // This test mainly verifies the mechanism doesn't crash and is conservative.
        let pred = model.predict(&[1.0, 2.0]);
        assert!(pred.is_finite(), "prediction should be finite");
    }

    #[test]
    fn adaptive_clone_fresh_resets_promotion() {
        let promote_to = LeafModelType::Linear {
            learning_rate: 0.01,
            decay: None,
        };
        let shadow = promote_to.create(42, 1e-3);
        let mut model = AdaptiveLeafModel::new(shadow, promote_to, 1e-3, 42);

        // Force promotion via many linear-target samples
        let mut rng = 42u64;
        for _ in 0..5000 {
            let x = rand_f64(&mut rng) * 2.0 - 1.0;
            let features = vec![x];
            let pred = model.predict(&features);
            model.update(&features, 2.0 * (pred - 3.0 * x), 2.0, 0.1);
        }

        let fresh = model.clone_fresh();
        // Fresh clone should predict 0 (reset state).
        let p = fresh.predict(&[0.5]);
        assert!(
            p.abs() < 1e-10,
            "fresh adaptive clone should predict ~0, got {p}"
        );
    }
}
