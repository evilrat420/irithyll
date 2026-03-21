//! Streaming Mixture of Experts over SGBT ensembles.
//!
//! Implements a gated mixture of K independent [`SGBT`] experts with a learned
//! linear softmax gate. Each expert is a full streaming gradient boosted tree
//! ensemble; the gate routes incoming samples to the most relevant expert(s)
//! based on the feature vector, enabling capacity specialization across
//! different regions of the input space.
//!
//! # Algorithm
//!
//! The gating network computes K logits `z_k = W_k · x + b_k` and applies
//! softmax to obtain routing probabilities `p_k = softmax(z)_k`. Prediction
//! is the probability-weighted sum of expert predictions:
//!
//! ```text
//! ŷ = Σ_k  p_k(x) · f_k(x)
//! ```
//!
//! During training, the gate is updated via online SGD on the cross-entropy
//! loss between the softmax distribution and the one-hot indicator of the
//! best expert (lowest loss on the current sample). This encourages the gate
//! to learn which expert is most competent for each region.
//!
//! Two gating modes are supported:
//!
//! - **Soft** (default): All experts receive every sample, weighted by their
//!   gating probability. This maximizes information flow but has O(K) training
//!   cost per sample.
//! - **Hard (top-k)**: Only the top-k experts (by gating probability) receive
//!   the sample. This reduces computation when K is large, at the cost of
//!   slower expert specialization.
//!
//! # References
//!
//! - Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991).
//!   Adaptive Mixtures of Local Experts. *Neural Computation*, 3(1), 79–87.
//! - Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G.,
//!   & Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated
//!   Mixture-of-Experts Layer. *ICLR 2017*.
//!
//! # Example
//!
//! ```
//! use irithyll::ensemble::moe::{MoESGBT, GatingMode};
//! use irithyll::SGBTConfig;
//!
//! let config = SGBTConfig::builder()
//!     .n_steps(10)
//!     .learning_rate(0.1)
//!     .grace_period(10)
//!     .build()
//!     .unwrap();
//!
//! let mut moe = MoESGBT::new(config, 3);
//! moe.train_one(&irithyll::Sample::new(vec![1.0, 2.0], 3.0));
//! let pred = moe.predict(&[1.0, 2.0]);
//! ```

use alloc::vec;
use alloc::vec::Vec;

use core::fmt;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::SGBT;
use crate::loss::squared::SquaredLoss;
use crate::loss::Loss;
use crate::sample::{Observation, SampleRef};

// ---------------------------------------------------------------------------
// GatingMode
// ---------------------------------------------------------------------------

/// Controls how the gate routes samples to experts.
///
/// - [`Soft`](GatingMode::Soft): every expert sees every sample, weighted by
///   gating probability. Maximizes information flow.
/// - [`Hard`](GatingMode::Hard): only the `top_k` experts with highest gating
///   probability receive the sample. Reduces cost when K is large.
#[derive(Debug, Clone)]
pub enum GatingMode {
    /// All experts receive every sample, weighted by gating probability.
    Soft,
    /// Only the top-k experts receive the sample (sparse routing).
    Hard {
        /// Number of experts to route each sample to.
        top_k: usize,
    },
}

// ---------------------------------------------------------------------------
// Softmax (numerically stable)
// ---------------------------------------------------------------------------

/// Numerically stable softmax: subtract max logit before exponentiating to
/// prevent overflow, then normalize.
pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&z| crate::math::exp(z - max)).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// MoESGBT
// ---------------------------------------------------------------------------

/// Streaming Mixture of Experts over SGBT ensembles.
///
/// Combines K independent [`SGBT<L>`] experts with a learned linear softmax
/// gating network. The gate is trained online via SGD to route samples to the
/// expert with the lowest loss, while all experts (or the top-k in hard gating
/// mode) are trained on each incoming sample.
///
/// Generic over `L: Loss` so the expert loss function is monomorphized. The
/// default is [`SquaredLoss`] for regression tasks.
///
/// # Gate Architecture
///
/// The gate is a single linear layer: `z_k = W_k · x + b_k` followed by
/// softmax. Weights are lazily initialized to zeros on the first sample
/// (since the feature dimensionality is not known at construction time).
/// The gate learns via cross-entropy gradient descent against the one-hot
/// indicator of the best expert per sample.
pub struct MoESGBT<L: Loss = SquaredLoss> {
    /// The K expert SGBT ensembles.
    experts: Vec<SGBT<L>>,
    /// Gate weight matrix [K x d], lazily initialized on first sample.
    gate_weights: Vec<Vec<f64>>,
    /// Gate bias vector [K].
    gate_bias: Vec<f64>,
    /// Learning rate for the gating network SGD updates.
    gate_lr: f64,
    /// Number of features (set on first sample, `None` until then).
    n_features: Option<usize>,
    /// Gating mode (soft or hard top-k).
    gating_mode: GatingMode,
    /// Configuration used to construct each expert.
    config: SGBTConfig,
    /// Loss function (shared type with experts, used for best-expert selection).
    loss: L,
    /// Total training samples seen.
    samples_seen: u64,
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

impl<L: Loss + Clone> Clone for MoESGBT<L> {
    fn clone(&self) -> Self {
        Self {
            experts: self.experts.clone(),
            gate_weights: self.gate_weights.clone(),
            gate_bias: self.gate_bias.clone(),
            gate_lr: self.gate_lr,
            n_features: self.n_features,
            gating_mode: self.gating_mode.clone(),
            config: self.config.clone(),
            loss: self.loss.clone(),
            samples_seen: self.samples_seen,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl<L: Loss> fmt::Debug for MoESGBT<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MoESGBT")
            .field("n_experts", &self.experts.len())
            .field("gating_mode", &self.gating_mode)
            .field("samples_seen", &self.samples_seen)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Default loss constructor (SquaredLoss)
// ---------------------------------------------------------------------------

impl MoESGBT<SquaredLoss> {
    /// Create a new MoE ensemble with squared loss (regression) and soft gating.
    ///
    /// Each expert is seeded uniquely via `config.seed ^ (0x0000_0E00_0000_0000 | i)`.
    /// The gating learning rate defaults to 0.01.
    ///
    /// # Panics
    ///
    /// Panics if `n_experts < 1`.
    pub fn new(config: SGBTConfig, n_experts: usize) -> Self {
        Self::with_loss(config, SquaredLoss, n_experts)
    }
}

// ---------------------------------------------------------------------------
// General impl
// ---------------------------------------------------------------------------

impl<L: Loss + Clone> MoESGBT<L> {
    /// Create a new MoE ensemble with a custom loss and soft gating.
    ///
    /// # Panics
    ///
    /// Panics if `n_experts < 1`.
    pub fn with_loss(config: SGBTConfig, loss: L, n_experts: usize) -> Self {
        Self::with_gating(config, loss, n_experts, GatingMode::Soft, 0.01)
    }

    /// Create a new MoE ensemble with full control over gating mode and gate
    /// learning rate.
    ///
    /// # Panics
    ///
    /// Panics if `n_experts < 1`.
    pub fn with_gating(
        config: SGBTConfig,
        loss: L,
        n_experts: usize,
        gating_mode: GatingMode,
        gate_lr: f64,
    ) -> Self {
        assert!(n_experts >= 1, "MoESGBT requires at least 1 expert");

        let experts = (0..n_experts)
            .map(|i| {
                let mut cfg = config.clone();
                cfg.seed = config.seed ^ (0x0000_0E00_0000_0000 | i as u64);
                SGBT::with_loss(cfg, loss.clone())
            })
            .collect();

        let gate_bias = vec![0.0; n_experts];

        Self {
            experts,
            gate_weights: Vec::new(), // lazy init
            gate_bias,
            gate_lr,
            n_features: None,
            gating_mode,
            config,
            loss,
            samples_seen: 0,
        }
    }
}

impl<L: Loss> MoESGBT<L> {
    // -------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------

    /// Ensure the gate weight matrix is initialized to the correct dimensions.
    /// Called lazily on the first sample when `n_features` is discovered.
    fn ensure_gate_init(&mut self, d: usize) {
        if self.n_features.is_none() {
            let k = self.experts.len();
            self.gate_weights = vec![vec![0.0; d]; k];
            self.n_features = Some(d);
        }
    }

    /// Compute raw gate logits: z_k = W_k · x + b_k.
    fn gate_logits(&self, features: &[f64]) -> Vec<f64> {
        let k = self.experts.len();
        let mut logits = Vec::with_capacity(k);
        for i in 0..k {
            let dot: f64 = self.gate_weights[i]
                .iter()
                .zip(features.iter())
                .map(|(&w, &x)| w * x)
                .sum();
            logits.push(dot + self.gate_bias[i]);
        }
        logits
    }

    // -------------------------------------------------------------------
    // Public API -- gating
    // -------------------------------------------------------------------

    /// Compute gating probabilities for a feature vector.
    ///
    /// Returns a vector of K probabilities that sum to 1.0, one per expert.
    /// The gate must be initialized (at least one training sample seen),
    /// otherwise returns uniform probabilities.
    pub fn gating_probabilities(&self, features: &[f64]) -> Vec<f64> {
        let k = self.experts.len();
        if self.n_features.is_none() {
            // Gate not initialized yet -- return uniform
            return vec![1.0 / k as f64; k];
        }
        let logits = self.gate_logits(features);
        softmax(&logits)
    }

    // -------------------------------------------------------------------
    // Public API -- training
    // -------------------------------------------------------------------

    /// Train on a single observation.
    ///
    /// 1. Lazily initializes the gate weights if this is the first sample.
    /// 2. Computes gating probabilities via softmax over the linear gate.
    /// 3. Routes the sample to experts according to the gating mode:
    ///    - **Soft**: all experts receive the sample, each weighted by its
    ///      gating probability (via `SampleRef::weighted`).
    ///    - **Hard(top_k)**: only the top-k experts by probability receive
    ///      the sample (with unit weight).
    /// 4. Updates gate weights via SGD on the cross-entropy gradient:
    ///    find the best expert (lowest loss), compute `dz_k = p_k - 1{k==best}`,
    ///    and apply `W_k -= gate_lr * dz_k * x`, `b_k -= gate_lr * dz_k`.
    pub fn train_one(&mut self, sample: &impl Observation) {
        let features = sample.features();
        let target = sample.target();
        let d = features.len();

        // Step 1: lazy gate initialization
        self.ensure_gate_init(d);

        // Step 2: compute gating probabilities
        let logits = self.gate_logits(features);
        let probs = softmax(&logits);
        let k = self.experts.len();

        // Step 3: train experts based on gating mode
        match &self.gating_mode {
            GatingMode::Soft => {
                // Every expert gets the sample, weighted by gating probability
                for (expert, &prob) in self.experts.iter_mut().zip(probs.iter()) {
                    let weighted = SampleRef::weighted(features, target, prob);
                    expert.train_one(&weighted);
                }
            }
            GatingMode::Hard { top_k } => {
                // Only the top-k experts get the sample
                let top_k = (*top_k).min(k);
                let mut indices: Vec<usize> = (0..k).collect();
                indices.sort_unstable_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                for &i in indices.iter().take(top_k) {
                    let obs = SampleRef::new(features, target);
                    self.experts[i].train_one(&obs);
                }
            }
        }

        // Step 4: update gate weights via SGD on cross-entropy gradient
        // Find best expert (lowest loss on this sample)
        let mut best_idx = 0;
        let mut best_loss = f64::INFINITY;
        for (i, expert) in self.experts.iter().enumerate() {
            let pred = expert.predict(features);
            let l = self.loss.loss(target, pred);
            if l < best_loss {
                best_loss = l;
                best_idx = i;
            }
        }

        // Cross-entropy gradient: dz_k = p_k - 1{k == best}
        // SGD update: W_k -= lr * dz_k * x,  b_k -= lr * dz_k
        for (i, (weights_row, bias)) in self
            .gate_weights
            .iter_mut()
            .zip(self.gate_bias.iter_mut())
            .enumerate()
        {
            let indicator = if i == best_idx { 1.0 } else { 0.0 };
            let grad = probs[i] - indicator;
            let lr = self.gate_lr;

            for (j, &xj) in features.iter().enumerate() {
                weights_row[j] -= lr * grad * xj;
            }
            *bias -= lr * grad;
        }

        self.samples_seen += 1;
    }

    /// Train on a batch of observations.
    pub fn train_batch<O: Observation>(&mut self, samples: &[O]) {
        for sample in samples {
            self.train_one(sample);
        }
    }

    // -------------------------------------------------------------------
    // Public API -- prediction
    // -------------------------------------------------------------------

    /// Predict the output for a feature vector.
    ///
    /// Computes the probability-weighted sum of expert predictions:
    /// `ŷ = Σ_k p_k(x) · f_k(x)`.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let probs = self.gating_probabilities(features);
        let mut pred = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            pred += p * self.experts[i].predict(features);
        }
        pred
    }

    /// Predict with gating probabilities returned alongside the prediction.
    ///
    /// Returns `(prediction, probabilities)` where probabilities is a K-length
    /// vector summing to 1.0.
    pub fn predict_with_gating(&self, features: &[f64]) -> (f64, Vec<f64>) {
        let probs = self.gating_probabilities(features);
        let mut pred = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            pred += p * self.experts[i].predict(features);
        }
        (pred, probs)
    }

    /// Get each expert's individual prediction for a feature vector.
    ///
    /// Returns a K-length vector of raw predictions, one per expert.
    pub fn expert_predictions(&self, features: &[f64]) -> Vec<f64> {
        self.experts.iter().map(|e| e.predict(features)).collect()
    }

    // -------------------------------------------------------------------
    // Public API -- inspection
    // -------------------------------------------------------------------

    /// Number of experts in the mixture.
    #[inline]
    pub fn n_experts(&self) -> usize {
        self.experts.len()
    }

    /// Total training samples seen.
    #[inline]
    pub fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    /// Immutable access to all experts.
    pub fn experts(&self) -> &[SGBT<L>] {
        &self.experts
    }

    /// Immutable access to a specific expert.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_experts`.
    pub fn expert(&self, idx: usize) -> &SGBT<L> {
        &self.experts[idx]
    }

    /// Reset the entire MoE to its initial state.
    ///
    /// Resets all experts, clears gate weights and biases back to zeros,
    /// and resets the sample counter.
    pub fn reset(&mut self) {
        for expert in &mut self.experts {
            expert.reset();
        }
        let k = self.experts.len();
        self.gate_weights.clear();
        self.gate_bias = vec![0.0; k];
        self.n_features = None;
        self.samples_seen = 0;
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

use crate::learner::StreamingLearner;

impl<L: Loss> StreamingLearner for MoESGBT<L> {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let sample = SampleRef::weighted(features, target, weight);
        // UFCS: call the inherent train_one(&impl Observation), not this trait method.
        MoESGBT::train_one(self, &sample);
    }

    fn predict(&self, features: &[f64]) -> f64 {
        MoESGBT::predict(self, features)
    }

    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        MoESGBT::reset(self);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::huber::HuberLoss;
    use crate::sample::Sample;
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Helper: build a minimal config for tests.
    fn test_config() -> SGBTConfig {
        SGBTConfig::builder()
            .n_steps(5)
            .learning_rate(0.1)
            .grace_period(5)
            .build()
            .unwrap()
    }

    #[test]
    fn test_creation() {
        let moe = MoESGBT::new(test_config(), 3);
        assert_eq!(moe.n_experts(), 3);
        assert_eq!(moe.n_samples_seen(), 0);
    }

    #[test]
    fn test_with_loss() {
        let moe = MoESGBT::with_loss(test_config(), HuberLoss { delta: 1.0 }, 4);
        assert_eq!(moe.n_experts(), 4);
        assert_eq!(moe.n_samples_seen(), 0);
    }

    #[test]
    fn test_soft_gating_trains_all() {
        let mut moe = MoESGBT::new(test_config(), 3);
        let sample = Sample::new(vec![1.0, 2.0], 5.0);

        moe.train_one(&sample);

        // In soft mode, every expert should have seen the sample
        for i in 0..3 {
            assert_eq!(moe.expert(i).n_samples_seen(), 1);
        }
    }

    #[test]
    fn test_hard_gating_top_k() {
        let mut moe = MoESGBT::with_gating(
            test_config(),
            SquaredLoss,
            4,
            GatingMode::Hard { top_k: 2 },
            0.01,
        );
        let sample = Sample::new(vec![1.0, 2.0], 5.0);

        moe.train_one(&sample);

        // Exactly top_k=2 experts should have received the sample
        let trained_count = (0..4)
            .filter(|&i| moe.expert(i).n_samples_seen() > 0)
            .count();
        assert_eq!(trained_count, 2);
    }

    #[test]
    fn test_gating_probabilities_sum_to_one() {
        let mut moe = MoESGBT::new(test_config(), 5);

        // Before training: uniform probabilities
        let probs = moe.gating_probabilities(&[1.0, 2.0]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "pre-training sum = {}", sum);

        // After training: probabilities should still sum to 1
        for i in 0..20 {
            let sample = Sample::new(vec![i as f64, (i * 2) as f64], i as f64);
            moe.train_one(&sample);
        }
        let probs = moe.gating_probabilities(&[5.0, 10.0]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "post-training sum = {}", sum);
    }

    #[test]
    fn test_prediction_changes_after_training() {
        let mut moe = MoESGBT::new(test_config(), 3);
        let features = vec![1.0, 2.0, 3.0];

        let pred_before = moe.predict(&features);

        for i in 0..50 {
            let sample = Sample::new(features.clone(), 10.0 + i as f64 * 0.1);
            moe.train_one(&sample);
        }

        let pred_after = moe.predict(&features);
        assert!(
            (pred_after - pred_before).abs() > 1e-6,
            "prediction should change after training: before={}, after={}",
            pred_before,
            pred_after
        );
    }

    #[test]
    fn test_expert_specialization() {
        // Two regions: x < 0 targets ~-10, x >= 0 targets ~+10
        let mut moe = MoESGBT::with_gating(test_config(), SquaredLoss, 2, GatingMode::Soft, 0.05);

        // Train with separable data
        for i in 0..200 {
            let x = if i % 2 == 0 {
                -(i as f64 + 1.0)
            } else {
                i as f64 + 1.0
            };
            let target = if x < 0.0 { -10.0 } else { 10.0 };
            let sample = Sample::new(vec![x], target);
            moe.train_one(&sample);
        }

        // After training, the gating probabilities should differ for
        // negative vs positive inputs
        let probs_neg = moe.gating_probabilities(&[-5.0]);
        let probs_pos = moe.gating_probabilities(&[5.0]);

        // The dominant expert should be different (or at least the distributions
        // should be noticeably different)
        let diff: f64 = probs_neg
            .iter()
            .zip(probs_pos.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "gate should route differently: neg={:?}, pos={:?}",
            probs_neg,
            probs_pos
        );
    }

    #[test]
    fn test_predict_with_gating() {
        let mut moe = MoESGBT::new(test_config(), 3);
        let sample = Sample::new(vec![1.0, 2.0], 5.0);
        moe.train_one(&sample);

        let (pred, probs) = moe.predict_with_gating(&[1.0, 2.0]);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Prediction should equal weighted sum of expert predictions
        let expert_preds = moe.expert_predictions(&[1.0, 2.0]);
        let expected: f64 = probs
            .iter()
            .zip(expert_preds.iter())
            .map(|(p, e)| p * e)
            .sum();
        assert!(
            (pred - expected).abs() < 1e-10,
            "pred={} expected={}",
            pred,
            expected
        );
    }

    #[test]
    fn test_expert_predictions() {
        let mut moe = MoESGBT::new(test_config(), 3);
        for i in 0..10 {
            let sample = Sample::new(vec![i as f64], i as f64);
            moe.train_one(&sample);
        }

        let preds = moe.expert_predictions(&[5.0]);
        assert_eq!(preds.len(), 3);
        // Each expert should produce a finite prediction
        for &p in &preds {
            assert!(p.is_finite(), "expert prediction should be finite: {}", p);
        }
    }

    #[test]
    fn test_n_experts() {
        let moe = MoESGBT::new(test_config(), 7);
        assert_eq!(moe.n_experts(), 7);
        assert_eq!(moe.experts().len(), 7);
    }

    #[test]
    fn test_n_samples_seen() {
        let mut moe = MoESGBT::new(test_config(), 2);
        assert_eq!(moe.n_samples_seen(), 0);

        for i in 0..25 {
            moe.train_one(&Sample::new(vec![i as f64], i as f64));
        }
        assert_eq!(moe.n_samples_seen(), 25);
    }

    #[test]
    fn test_reset() {
        let mut moe = MoESGBT::new(test_config(), 3);

        for i in 0..50 {
            moe.train_one(&Sample::new(vec![i as f64, (i * 2) as f64], i as f64));
        }
        assert_eq!(moe.n_samples_seen(), 50);

        moe.reset();

        assert_eq!(moe.n_samples_seen(), 0);
        assert_eq!(moe.n_experts(), 3);
        // Gate should be re-lazily-initialized
        let probs = moe.gating_probabilities(&[1.0, 2.0]);
        assert_eq!(probs.len(), 3);
        // After reset, probabilities are uniform again
        for &p in &probs {
            assert!(
                (p - 1.0 / 3.0).abs() < 1e-10,
                "expected uniform after reset, got {}",
                p
            );
        }
    }

    #[test]
    fn test_single_expert() {
        // With a single expert, MoE should behave like a plain SGBT
        let config = test_config();
        let mut moe = MoESGBT::new(config.clone(), 1);

        let mut plain = SGBT::new({
            let mut cfg = config.clone();
            cfg.seed = config.seed ^ 0x0000_0E00_0000_0000;
            cfg
        });

        // The single expert gets weight=1.0 always, so predictions should
        // be very close (both see same data, same seed)
        for i in 0..30 {
            let sample = Sample::new(vec![i as f64], i as f64 * 2.0);
            moe.train_one(&sample);
            // For the plain SGBT, we need to replicate the soft-gating weight.
            // With one expert, p=1.0, so SampleRef::weighted(features, target, 1.0)
            // is equivalent to a normal sample (weight=1.0).
            let weighted = SampleRef::weighted(&sample.features, sample.target, 1.0);
            plain.train_one(&weighted);
        }

        let moe_pred = moe.predict(&[15.0]);
        let plain_pred = plain.predict(&[15.0]);
        assert!(
            (moe_pred - plain_pred).abs() < 1e-6,
            "single expert MoE should match plain SGBT: moe={}, plain={}",
            moe_pred,
            plain_pred
        );
    }

    #[test]
    fn test_gate_lr_effect() {
        // A higher gate learning rate should cause the gate to diverge from
        // uniform faster than a lower one.
        let config = test_config();

        let mut moe_low =
            MoESGBT::with_gating(config.clone(), SquaredLoss, 3, GatingMode::Soft, 0.001);
        let mut moe_high = MoESGBT::with_gating(config, SquaredLoss, 3, GatingMode::Soft, 0.1);

        // Train both on the same data
        for i in 0..50 {
            let sample = Sample::new(vec![i as f64], i as f64);
            moe_low.train_one(&sample);
            moe_high.train_one(&sample);
        }

        // Measure deviation from uniform for both
        let uniform = 1.0 / 3.0;
        let probs_low = moe_low.gating_probabilities(&[25.0]);
        let probs_high = moe_high.gating_probabilities(&[25.0]);

        let dev_low: f64 = probs_low.iter().map(|p| (p - uniform).abs()).sum();
        let dev_high: f64 = probs_high.iter().map(|p| (p - uniform).abs()).sum();

        assert!(
            dev_high > dev_low,
            "higher gate_lr should cause more deviation from uniform: low={}, high={}",
            dev_low,
            dev_high
        );
    }

    #[test]
    fn test_batch_training() {
        let mut moe = MoESGBT::new(test_config(), 3);

        let samples: Vec<Sample> = (0..20)
            .map(|i| Sample::new(vec![i as f64, (i * 3) as f64], i as f64))
            .collect();

        moe.train_batch(&samples);

        assert_eq!(moe.n_samples_seen(), 20);

        // Should produce non-zero predictions after batch training
        let pred = moe.predict(&[10.0, 30.0]);
        assert!(pred.is_finite());
    }

    #[test]
    fn streaming_learner_trait_object() {
        let config = test_config();
        let model = MoESGBT::new(config, 3);
        let mut boxed: Box<dyn StreamingLearner> = Box::new(model);
        for i in 0..100 {
            let x = i as f64 * 0.1;
            boxed.train(&[x], x * 2.0);
        }
        assert_eq!(boxed.n_samples_seen(), 100);
        let pred = boxed.predict(&[5.0]);
        assert!(pred.is_finite());
        boxed.reset();
        assert_eq!(boxed.n_samples_seen(), 0);
    }
}
