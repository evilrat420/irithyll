//! Pluggable leaf prediction models for streaming decision trees.
//!
//! By default, leaves use a closed-form weight computed from accumulated
//! gradient and hessian sums. Neural leaf models replace this with trainable
//! models that can capture more complex patterns within each leaf's partition.

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

/// Online ridge regression leaf model.
///
/// Learns a linear function `w . x + b` using Newton-scaled gradient descent.
/// Weights are lazily initialized on the first `update` call so the model
/// adapts to whatever dimensionality arrives.
pub struct LinearLeafModel {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    initialized: bool,
}

impl LinearLeafModel {
    /// Create a new linear leaf model with the given base learning rate.
    pub fn new(learning_rate: f64) -> Self {
        Self {
            weights: Vec::new(),
            bias: 0.0,
            learning_rate,
            initialized: false,
        }
    }
}

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
            self.weights = vec![0.0; features.len()];
            self.initialized = true;
        }
        let effective_lr = self.learning_rate / (hessian.abs() + lambda);
        for (w, x) in self.weights.iter_mut().zip(features.iter()) {
            *w -= effective_lr * gradient * x;
        }
        self.bias -= effective_lr * gradient;
    }

    fn clone_fresh(&self) -> Box<dyn LeafModel> {
        Box::new(LinearLeafModel::new(self.learning_rate))
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
pub struct MLPLeafModel {
    hidden_weights: Vec<Vec<f64>>, // [hidden_size][input_size]
    hidden_bias: Vec<f64>,
    output_weights: Vec<f64>,
    output_bias: f64,
    hidden_size: usize,
    learning_rate: f64,
    seed: u64,
    initialized: bool,
    hidden_activations: Vec<f64>,
    hidden_pre_activations: Vec<f64>,
}

impl MLPLeafModel {
    /// Create a new MLP leaf model with the given hidden layer size, learning rate, and seed.
    ///
    /// The seed controls deterministic weight initialization. Different seeds
    /// produce different initial weights, which is critical for ensemble diversity
    /// when multiple MLP leaves share the same `hidden_size`.
    pub fn new(hidden_size: usize, learning_rate: f64, seed: u64) -> Self {
        Self {
            hidden_weights: Vec::new(),
            hidden_bias: Vec::new(),
            output_weights: Vec::new(),
            output_bias: 0.0,
            hidden_size,
            learning_rate,
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
        ))
    }
}

// ---------------------------------------------------------------------------
// LeafModelType
// ---------------------------------------------------------------------------

/// Describes which leaf model architecture to use.
///
/// Used by tree builders to construct fresh leaf models when creating new leaves.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LeafModelType {
    /// Standard closed-form leaf weight.
    #[default]
    ClosedForm,
    /// Online ridge regression with the given learning rate.
    Linear { learning_rate: f64 },
    /// Single hidden layer MLP with the given hidden size and learning rate.
    MLP {
        hidden_size: usize,
        learning_rate: f64,
    },
}

impl LeafModelType {
    /// Create a fresh boxed leaf model of this type.
    ///
    /// The `seed` parameter is used for MLP weight initialization. Different
    /// seeds produce different initial weights, ensuring ensemble diversity.
    /// For ClosedForm and Linear models the seed is unused.
    pub fn create(&self, seed: u64) -> Box<dyn LeafModel> {
        match self {
            Self::ClosedForm => Box::new(ClosedFormLeaf::new()),
            Self::Linear { learning_rate } => Box::new(LinearLeafModel::new(*learning_rate)),
            Self::MLP {
                hidden_size,
                learning_rate,
            } => Box::new(MLPLeafModel::new(*hidden_size, *learning_rate, seed)),
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
        let mut model = LinearLeafModel::new(0.01);
        let lambda = 0.1;
        let mut rng = 42u64;

        for _ in 0..2000 {
            let x1 = rand_f64(&mut rng) * 2.0 - 1.0;
            let x2 = rand_f64(&mut rng) * 2.0 - 1.0;
            let features = vec![x1, x2];
            let target = 2.0 * x1 + 3.0 * x2;

            let pred = model.predict(&features);
            // gradient of squared loss: 2*(pred - target), hessian: 2
            let gradient = 2.0 * (pred - target);
            let hessian = 2.0;
            model.update(&features, gradient, hessian, lambda);
        }

        // Verify on a test point
        let test_features = vec![0.5, -0.3];
        let target = 2.0 * 0.5 + 3.0 * (-0.3); // 1.0 - 0.9 = 0.1
        let pred = model.predict(&test_features);

        assert!(
            (pred - target).abs() < 1.0,
            "linear model should converge within 1.0 of target: pred={pred}, target={target}"
        );
    }

    #[test]
    fn linear_uninitialized_predicts_zero() {
        let model = LinearLeafModel::new(0.01);
        let pred = model.predict(&[1.0, 2.0, 3.0]);
        assert!(
            pred.abs() < 1e-15,
            "uninitialized linear model should predict 0, got {pred}"
        );
    }

    #[test]
    fn mlp_produces_finite_predictions() {
        let model_uninit = MLPLeafModel::new(4, 0.01, 42);
        let features = vec![1.0, 2.0, 3.0];

        // Before training: should return 0
        let pred_before = model_uninit.predict(&features);
        assert!(
            pred_before.is_finite(),
            "uninit prediction should be finite"
        );
        assert!(
            pred_before.abs() < 1e-15,
            "uninit prediction should be 0, got {pred_before}"
        );

        // After training: should still be finite
        let mut model = MLPLeafModel::new(4, 0.01, 42);
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
        let mut model = MLPLeafModel::new(8, 0.05, 123);
        let features = vec![1.0, -0.5, 0.3];
        let target = 2.5;
        let lambda = 0.1;

        // Get initial loss (after first update initializes the model)
        model.update(&features, 0.0, 1.0, lambda); // dummy update to initialize
        let initial_pred = model.predict(&features);
        let initial_error = (initial_pred - target).abs();

        // Train for several iterations
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
        let mut model = MLPLeafModel::new(4, 0.01, 42);
        let features = vec![1.0, 2.0];

        // Train it
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

        // ClosedForm
        let mut closed = LeafModelType::ClosedForm.create(0);
        closed.update(&features, 1.0, 1.0, 0.1);
        let p = closed.predict(&features);
        assert!(p.is_finite(), "ClosedForm prediction should be finite");

        // Linear
        let mut linear = LeafModelType::Linear {
            learning_rate: 0.01,
        }
        .create(0);
        linear.update(&features, 1.0, 1.0, 0.1);
        let p = linear.predict(&features);
        assert!(p.is_finite(), "Linear prediction should be finite");

        // MLP
        let mut mlp = LeafModelType::MLP {
            hidden_size: 4,
            learning_rate: 0.01,
        }
        .create(99);
        mlp.update(&features, 1.0, 1.0, 0.1);
        let p = mlp.predict(&features);
        assert!(p.is_finite(), "MLP prediction should be finite");
    }
}
