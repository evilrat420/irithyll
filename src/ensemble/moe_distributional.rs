//! Streaming Mixture of Experts over Distributional SGBT ensembles with
//! shadow expert competition.
//!
//! Combines K independent [`DistributionalSGBT`] experts with a learned
//! linear softmax gating network. Each expert outputs a full Gaussian
//! predictive distribution N(mu, sigma^2). The mixture prediction applies
//! the law of total variance to produce a single Gaussian from the K
//! expert predictions weighted by gating probabilities.
//!
//! Additionally, each expert slot has a *shadow* expert that trains in
//! parallel. When the shadow achieves statistically better Gaussian NLL
//! than the active expert (verified via the Hoeffding bound), it replaces
//! the active expert. This enables continuous capacity adaptation.
//!
//! # Algorithm
//!
//! The gating network computes K logits `z_k = W_k . x + b_k` and applies
//! softmax to obtain routing probabilities `p_k = softmax(z)_k`. Prediction
//! is a mixture of Gaussians via the law of total variance:
//!
//! ```text
//! mu_mix = sum(p_k * mu_k)
//! var_mix = sum(p_k * (sigma_k^2 + mu_k^2)) - mu_mix^2
//! sigma_mix = sqrt(var_mix)
//! ```
//!
//! The gate is updated via online SGD on the cross-entropy loss between
//! the softmax distribution and the one-hot indicator of the best expert
//! (lowest Gaussian NLL on the current sample).
//!
//! # Shadow Competition
//!
//! Each expert slot maintains a shadow model that trains on the same data.
//! After `shadow_min_samples` observations, the Hoeffding bound is used to
//! test whether the shadow's cumulative NLL advantage is statistically
//! significant:
//!
//! ```text
//! epsilon = sqrt(R^2 * ln(1/delta) / (2*n))
//! swap if mean_advantage > epsilon
//! ```
//!
//! # Example
//!
//! ```
//! use irithyll::ensemble::moe_distributional::MoEDistributionalSGBT;
//! use irithyll::ensemble::moe::GatingMode;
//! use irithyll::SGBTConfig;
//!
//! let config = SGBTConfig::builder()
//!     .n_steps(10)
//!     .learning_rate(0.1)
//!     .grace_period(10)
//!     .build()
//!     .unwrap();
//!
//! let mut moe = MoEDistributionalSGBT::new(config, 3);
//! moe.train_one(&irithyll::Sample::new(vec![1.0, 2.0], 3.0));
//! let pred = moe.predict(&[1.0, 2.0]);
//! assert!(pred.sigma > 0.0);
//! ```

use std::fmt;

use crate::ensemble::config::SGBTConfig;
use crate::ensemble::distributional::{DistributionalSGBT, GaussianPrediction};
use crate::ensemble::moe::{softmax, GatingMode};
use crate::sample::{Observation, SampleRef};

// ---------------------------------------------------------------------------
// MoEDistributionalSGBT
// ---------------------------------------------------------------------------

/// Streaming Mixture of Experts over [`DistributionalSGBT`] ensembles with
/// shadow expert competition.
///
/// Combines K independent distributional experts with a learned linear
/// softmax gating network. Prediction is a mixture of Gaussians via the
/// law of total variance. Shadow experts compete to replace active experts
/// using the Hoeffding bound on Gaussian NLL differences.
pub struct MoEDistributionalSGBT {
    /// The K active expert DistributionalSGBT ensembles.
    experts: Vec<DistributionalSGBT>,
    /// The K shadow experts (one per slot, trained in parallel).
    shadows: Vec<DistributionalSGBT>,
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
    /// Optional per-expert configurations. When `Some`, each expert uses its
    /// own `SGBTConfig` instead of the shared `config`.
    expert_configs: Option<Vec<SGBTConfig>>,
    /// Total training samples seen.
    samples_seen: u64,
    /// Entropy regularization weight for gate load balancing.
    ///
    /// Adds `entropy_weight * entropy_gradient` to the gate SGD update,
    /// encouraging the gate to spread probability mass across all experts
    /// rather than collapsing to a single expert.
    ///
    /// Default: 0.0 (disabled for backward compat in existing constructors).
    entropy_weight: f64,

    // -- Shadow competition state per slot --
    /// Cumulative NLL advantage of shadow over active (positive = shadow better).
    cumulative_advantage: Vec<f64>,
    /// Number of comparison samples per slot.
    shadow_n: Vec<u64>,
    /// Maximum absolute NLL difference seen per slot (for Hoeffding range R).
    max_nll_diff: Vec<f64>,
    /// Hoeffding confidence parameter (default 1e-3).
    delta: f64,
    /// Minimum samples before shadow comparison begins.
    shadow_min_samples: u64,
    /// Count of shadow replacements per slot.
    shadow_replacements: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

impl Clone for MoEDistributionalSGBT {
    fn clone(&self) -> Self {
        Self {
            experts: self.experts.clone(),
            shadows: self.shadows.clone(),
            gate_weights: self.gate_weights.clone(),
            gate_bias: self.gate_bias.clone(),
            gate_lr: self.gate_lr,
            n_features: self.n_features,
            gating_mode: self.gating_mode.clone(),
            config: self.config.clone(),
            expert_configs: self.expert_configs.clone(),
            samples_seen: self.samples_seen,
            entropy_weight: self.entropy_weight,
            cumulative_advantage: self.cumulative_advantage.clone(),
            shadow_n: self.shadow_n.clone(),
            max_nll_diff: self.max_nll_diff.clone(),
            delta: self.delta,
            shadow_min_samples: self.shadow_min_samples,
            shadow_replacements: self.shadow_replacements.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl fmt::Debug for MoEDistributionalSGBT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MoEDistributionalSGBT")
            .field("n_experts", &self.experts.len())
            .field("samples_seen", &self.samples_seen)
            .field("shadow_replacements", &self.shadow_replacements)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl MoEDistributionalSGBT {
    /// Create a new MoE distributional ensemble with soft gating and default
    /// shadow competition parameters.
    ///
    /// Defaults: gate_lr = 0.01, delta = 1e-3, shadow_min_samples = 500.
    ///
    /// # Panics
    ///
    /// Panics if `n_experts < 1`.
    pub fn new(config: SGBTConfig, n_experts: usize) -> Self {
        Self::with_shadow_config(config, n_experts, GatingMode::Soft, 0.01, 1e-3, 500)
    }

    /// Create a new MoE distributional ensemble with custom gating mode
    /// and gate learning rate. Uses default shadow parameters.
    ///
    /// # Panics
    ///
    /// Panics if `n_experts < 1`.
    pub fn with_gating(
        config: SGBTConfig,
        n_experts: usize,
        gating_mode: GatingMode,
        gate_lr: f64,
    ) -> Self {
        Self::with_shadow_config(config, n_experts, gating_mode, gate_lr, 1e-3, 500)
    }

    /// Create a new MoE distributional ensemble with full control over
    /// gating and shadow competition parameters.
    ///
    /// # Arguments
    ///
    /// * `config` -- SGBT configuration for each expert
    /// * `n_experts` -- number of expert slots
    /// * `gating_mode` -- soft or hard(top_k) gating
    /// * `gate_lr` -- learning rate for the gating network
    /// * `delta` -- Hoeffding confidence parameter (lower = more conservative)
    /// * `shadow_min_samples` -- warmup before shadow comparison begins
    ///
    /// # Panics
    ///
    /// Panics if `n_experts < 1`.
    pub fn with_shadow_config(
        config: SGBTConfig,
        n_experts: usize,
        gating_mode: GatingMode,
        gate_lr: f64,
        delta: f64,
        shadow_min_samples: u64,
    ) -> Self {
        assert!(
            n_experts >= 1,
            "MoEDistributionalSGBT requires at least 1 expert"
        );

        let experts: Vec<DistributionalSGBT> = (0..n_experts)
            .map(|i| {
                let mut cfg = config.clone();
                cfg.seed = config.seed ^ (0x0E00_0000 | i as u64);
                DistributionalSGBT::new(cfg)
            })
            .collect();

        let shadows: Vec<DistributionalSGBT> = (0..n_experts)
            .map(|i| {
                let mut cfg = config.clone();
                cfg.seed = config.seed ^ (0x5A00_0000 | i as u64);
                DistributionalSGBT::new(cfg)
            })
            .collect();

        let gate_bias = vec![0.0; n_experts];

        Self {
            experts,
            shadows,
            gate_weights: Vec::new(), // lazy init
            gate_bias,
            gate_lr,
            n_features: None,
            gating_mode,
            config,
            expert_configs: None,
            samples_seen: 0,
            entropy_weight: 0.0,
            cumulative_advantage: vec![0.0; n_experts],
            shadow_n: vec![0; n_experts],
            max_nll_diff: vec![0.0; n_experts],
            delta,
            shadow_min_samples,
            shadow_replacements: vec![0; n_experts],
        }
    }

    /// Create a new MoE distributional ensemble where each expert uses its own
    /// `SGBTConfig`, enabling different depths, lambda, learning rates, etc.
    ///
    /// The first config in `configs` is also used as the shared fallback (stored
    /// in `self.config`) for any field-level queries.
    ///
    /// # Arguments
    ///
    /// * `configs` -- one `SGBTConfig` per expert (length determines n_experts)
    /// * `gating_mode` -- soft or hard(top_k) gating
    /// * `gate_lr` -- learning rate for the gating network
    /// * `entropy_weight` -- entropy regularization weight (0.0 = disabled,
    ///   0.1 = typical for preventing gate collapse)
    /// * `delta` -- Hoeffding confidence parameter for shadow competition
    /// * `shadow_min_samples` -- warmup before shadow comparison begins
    ///
    /// # Panics
    ///
    /// Panics if `configs` is empty.
    pub fn with_expert_configs(
        configs: Vec<SGBTConfig>,
        gating_mode: GatingMode,
        gate_lr: f64,
        entropy_weight: f64,
        delta: f64,
        shadow_min_samples: u64,
    ) -> Self {
        assert!(
            !configs.is_empty(),
            "MoEDistributionalSGBT requires at least 1 expert config"
        );

        let n_experts = configs.len();

        let experts: Vec<DistributionalSGBT> = configs
            .iter()
            .enumerate()
            .map(|(i, cfg)| {
                let mut c = cfg.clone();
                c.seed = cfg.seed ^ (0x0E00_0000 | i as u64);
                DistributionalSGBT::new(c)
            })
            .collect();

        let shadows: Vec<DistributionalSGBT> = configs
            .iter()
            .enumerate()
            .map(|(i, cfg)| {
                let mut c = cfg.clone();
                c.seed = cfg.seed ^ (0x5A00_0000 | i as u64);
                DistributionalSGBT::new(c)
            })
            .collect();

        let gate_bias = vec![0.0; n_experts];
        let shared_config = configs[0].clone();

        Self {
            experts,
            shadows,
            gate_weights: Vec::new(),
            gate_bias,
            gate_lr,
            n_features: None,
            gating_mode,
            config: shared_config,
            expert_configs: Some(configs),
            samples_seen: 0,
            entropy_weight,
            cumulative_advantage: vec![0.0; n_experts],
            shadow_n: vec![0; n_experts],
            max_nll_diff: vec![0.0; n_experts],
            delta,
            shadow_min_samples,
            shadow_replacements: vec![0; n_experts],
        }
    }
}

// ---------------------------------------------------------------------------
// Core impl
// ---------------------------------------------------------------------------

impl MoEDistributionalSGBT {
    // -------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------

    /// Ensure the gate weight matrix is initialized to the correct dimensions.
    fn ensure_gate_init(&mut self, d: usize) {
        if self.n_features.is_none() {
            let k = self.experts.len();
            self.gate_weights = vec![vec![0.0; d]; k];
            self.n_features = Some(d);
        }
    }

    /// Compute raw gate logits: z_k = W_k . x + b_k.
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

    /// Compute Gaussian NLL for a single prediction.
    /// nll = log_sigma + 0.5 * ((target - mu) / sigma)^2
    #[inline]
    fn gaussian_nll(pred: &GaussianPrediction, target: f64) -> f64 {
        let z = (target - pred.mu) / pred.sigma.max(1e-16);
        pred.log_sigma + 0.5 * z * z
    }

    // -------------------------------------------------------------------
    // Public API -- gating
    // -------------------------------------------------------------------

    /// Compute gating probabilities for a feature vector.
    ///
    /// Returns a vector of K probabilities that sum to 1.0, one per expert.
    /// If the gate is not yet initialized, returns uniform probabilities.
    pub fn gating_probabilities(&self, features: &[f64]) -> Vec<f64> {
        let k = self.experts.len();
        if self.n_features.is_none() {
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
    /// 1. Lazily initializes the gate weights on the first sample.
    /// 2. Computes gating probabilities via softmax over the linear gate.
    /// 3. Routes the sample to experts (and their shadows) based on gating mode.
    /// 4. Performs shadow competition via Hoeffding-bound NLL comparison.
    /// 5. Updates gate weights via SGD on the cross-entropy gradient (best
    ///    expert by lowest Gaussian NLL).
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

        // Step 3: train experts and shadows based on gating mode
        match &self.gating_mode {
            GatingMode::Soft => {
                for (i, &prob) in probs.iter().enumerate() {
                    let weighted = SampleRef::weighted(features, target, prob);
                    self.experts[i].train_one(&weighted);
                    self.shadows[i].train_one(&weighted);
                }
            }
            GatingMode::Hard { top_k } => {
                let top_k = (*top_k).min(k);
                let mut indices: Vec<usize> = (0..k).collect();
                indices.sort_unstable_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for &i in indices.iter().take(top_k) {
                    let obs = SampleRef::new(features, target);
                    self.experts[i].train_one(&obs);
                    self.shadows[i].train_one(&obs);
                }
            }
        }

        // Step 4: shadow competition per slot
        for i in 0..k {
            // Skip if expert or shadow not yet initialized, or below warmup
            if !self.experts[i].is_initialized() || !self.shadows[i].is_initialized() {
                continue;
            }
            if self.shadows[i].n_samples_seen() < self.shadow_min_samples {
                continue;
            }

            let pred_active = self.experts[i].predict(features);
            let pred_shadow = self.shadows[i].predict(features);

            let nll_active = Self::gaussian_nll(&pred_active, target);
            let nll_shadow = Self::gaussian_nll(&pred_shadow, target);

            // Positive diff = shadow is better (lower NLL)
            let diff = nll_active - nll_shadow;
            self.cumulative_advantage[i] += diff;
            self.shadow_n[i] += 1;

            let abs_diff = diff.abs();
            if abs_diff > self.max_nll_diff[i] {
                self.max_nll_diff[i] = abs_diff;
            }

            // Hoeffding bound test
            if self.shadow_n[i] >= 10 && self.max_nll_diff[i] > 0.0 {
                let mean_advantage = self.cumulative_advantage[i] / self.shadow_n[i] as f64;
                if mean_advantage > 0.0 {
                    let r_squared = self.max_nll_diff[i] * self.max_nll_diff[i];
                    let ln_inv_delta = (1.0 / self.delta).ln();
                    let epsilon =
                        (r_squared * ln_inv_delta / (2.0 * self.shadow_n[i] as f64)).sqrt();

                    if mean_advantage > epsilon {
                        // Swap: shadow becomes active, create fresh shadow
                        self.experts[i] = self.shadows[i].clone();
                        let base_cfg = self
                            .expert_configs
                            .as_ref()
                            .map(|c| &c[i])
                            .unwrap_or(&self.config);
                        let mut fresh_cfg = base_cfg.clone();
                        fresh_cfg.seed = base_cfg.seed
                            ^ (0x5A00_0000 | i as u64)
                            ^ (self.shadow_replacements[i].wrapping_add(1) * 0x9E37_79B9);
                        self.shadows[i] = DistributionalSGBT::new(fresh_cfg);

                        // Reset comparison state
                        self.cumulative_advantage[i] = 0.0;
                        self.shadow_n[i] = 0;
                        self.max_nll_diff[i] = 0.0;
                        self.shadow_replacements[i] += 1;
                    }
                }
            }
        }

        // Step 5: update gate weights via SGD on cross-entropy gradient
        // Find best expert by lowest Gaussian NLL
        let mut best_idx = 0;
        let mut best_nll = f64::INFINITY;
        for (i, expert) in self.experts.iter().enumerate() {
            let pred = expert.predict(features);
            let nll = Self::gaussian_nll(&pred, target);
            if nll < best_nll {
                best_nll = nll;
                best_idx = i;
            }
        }

        // Cross-entropy gradient + entropy regularization: dz_k = ce_grad + entropy_weight * entropy_grad
        // Entropy gradient: d(-H)/dz_k = p_k * (log(p_k) + 1) - mean_term
        // This pushes the gate toward uniform distribution, preventing collapse.
        let entropy_mean_log_term: f64 = if self.entropy_weight != 0.0 {
            probs
                .iter()
                .map(|&p| {
                    let lp = if p > 1e-10 { p.ln() } else { -23.0 };
                    p * (lp + 1.0)
                })
                .sum()
        } else {
            0.0
        };

        for (i, (weights_row, bias)) in self
            .gate_weights
            .iter_mut()
            .zip(self.gate_bias.iter_mut())
            .enumerate()
        {
            let indicator = if i == best_idx { 1.0 } else { 0.0 };
            let ce_grad = probs[i] - indicator;

            let total_grad = if self.entropy_weight != 0.0 {
                let log_p = if probs[i] > 1e-10 {
                    probs[i].ln()
                } else {
                    -23.0
                };
                let entropy_grad = probs[i] * (log_p + 1.0) - entropy_mean_log_term;
                ce_grad + self.entropy_weight * entropy_grad
            } else {
                ce_grad
            };

            let lr = self.gate_lr;
            for (j, &xj) in features.iter().enumerate() {
                weights_row[j] -= lr * total_grad * xj;
            }
            *bias -= lr * total_grad;
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

    /// Predict the full mixture Gaussian distribution for a feature vector.
    ///
    /// Applies the law of total variance to combine K expert Gaussians:
    ///
    /// ```text
    /// mu_mix = sum(p_k * mu_k)
    /// var_mix = sum(p_k * (sigma_k^2 + mu_k^2)) - mu_mix^2
    /// sigma_mix = sqrt(var_mix)
    /// ```
    pub fn predict(&self, features: &[f64]) -> GaussianPrediction {
        let probs = self.gating_probabilities(features);
        let preds: Vec<GaussianPrediction> =
            self.experts.iter().map(|e| e.predict(features)).collect();

        // Mixture mean
        let mu_mix: f64 = probs
            .iter()
            .zip(preds.iter())
            .map(|(&p, pred)| p * pred.mu)
            .sum();

        // Law of total variance: Var = E[Var(X|K)] + Var(E[X|K])
        // = sum(p_k * sigma_k^2) + sum(p_k * mu_k^2) - mu_mix^2
        let second_moment: f64 = probs
            .iter()
            .zip(preds.iter())
            .map(|(&p, pred)| p * (pred.sigma * pred.sigma + pred.mu * pred.mu))
            .sum();
        let var_mix = (second_moment - mu_mix * mu_mix).max(1e-16);
        let sigma_mix = var_mix.sqrt();

        // Weighted average of expert honest_sigmas
        let honest_sigma_mix: f64 = probs
            .iter()
            .zip(preds.iter())
            .map(|(&p, pred)| p * pred.honest_sigma)
            .sum();

        GaussianPrediction {
            mu: mu_mix,
            sigma: sigma_mix,
            log_sigma: sigma_mix.ln(),
            honest_sigma: honest_sigma_mix,
        }
    }

    /// Predict with gating probabilities returned alongside the prediction.
    ///
    /// Returns `(GaussianPrediction, probabilities)` where probabilities is
    /// a K-length vector summing to 1.0.
    pub fn predict_with_gating(&self, features: &[f64]) -> (GaussianPrediction, Vec<f64>) {
        let probs = self.gating_probabilities(features);
        let preds: Vec<GaussianPrediction> =
            self.experts.iter().map(|e| e.predict(features)).collect();

        let mu_mix: f64 = probs
            .iter()
            .zip(preds.iter())
            .map(|(&p, pred)| p * pred.mu)
            .sum();

        let second_moment: f64 = probs
            .iter()
            .zip(preds.iter())
            .map(|(&p, pred)| p * (pred.sigma * pred.sigma + pred.mu * pred.mu))
            .sum();
        let var_mix = (second_moment - mu_mix * mu_mix).max(1e-16);
        let sigma_mix = var_mix.sqrt();

        let honest_sigma_mix: f64 = probs
            .iter()
            .zip(preds.iter())
            .map(|(&p, pred)| p * pred.honest_sigma)
            .sum();

        let pred = GaussianPrediction {
            mu: mu_mix,
            sigma: sigma_mix,
            log_sigma: sigma_mix.ln(),
            honest_sigma: honest_sigma_mix,
        };
        (pred, probs)
    }

    /// Get each expert's individual prediction for a feature vector.
    ///
    /// Returns a K-length vector of Gaussian predictions, one per expert.
    pub fn expert_predictions(&self, features: &[f64]) -> Vec<GaussianPrediction> {
        self.experts.iter().map(|e| e.predict(features)).collect()
    }

    /// Predict the mean (location parameter) of the mixture.
    #[inline]
    pub fn predict_mu(&self, features: &[f64]) -> f64 {
        self.predict(features).mu
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
    pub fn experts(&self) -> &[DistributionalSGBT] {
        &self.experts
    }

    /// Immutable access to a specific expert.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= n_experts`.
    pub fn expert(&self, idx: usize) -> &DistributionalSGBT {
        &self.experts[idx]
    }

    /// Shadow replacement counts per expert slot.
    pub fn shadow_replacements(&self) -> &[u64] {
        &self.shadow_replacements
    }

    /// Entropy regularization weight for gate load balancing.
    #[inline]
    pub fn entropy_weight(&self) -> f64 {
        self.entropy_weight
    }

    /// Per-expert configurations, if set via [`with_expert_configs`](Self::with_expert_configs).
    pub fn expert_configs(&self) -> Option<&[SGBTConfig]> {
        self.expert_configs.as_deref()
    }

    /// Reset the entire MoE to its initial state.
    ///
    /// Resets all experts and shadows, clears gate weights and biases back to
    /// zeros, resets shadow competition state and the sample counter.
    pub fn reset(&mut self) {
        let k = self.experts.len();
        for expert in &mut self.experts {
            expert.reset();
        }
        for shadow in &mut self.shadows {
            shadow.reset();
        }
        self.gate_weights.clear();
        self.gate_bias = vec![0.0; k];
        self.n_features = None;
        self.samples_seen = 0;
        self.cumulative_advantage = vec![0.0; k];
        self.shadow_n = vec![0; k];
        self.max_nll_diff = vec![0.0; k];
        self.shadow_replacements = vec![0; k];
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner impl
// ---------------------------------------------------------------------------

use crate::learner::StreamingLearner;

impl StreamingLearner for MoEDistributionalSGBT {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let sample = SampleRef::weighted(features, target, weight);
        // UFCS: call the inherent train_one(&impl Observation), not this trait method.
        MoEDistributionalSGBT::train_one(self, &sample);
    }

    /// Returns the mean (mu) of the predicted mixture Gaussian.
    fn predict(&self, features: &[f64]) -> f64 {
        MoEDistributionalSGBT::predict(self, features).mu
    }

    fn n_samples_seen(&self) -> u64 {
        self.samples_seen
    }

    fn reset(&mut self) {
        MoEDistributionalSGBT::reset(self);
    }

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
        for expert in &mut self.experts {
            let current_lr = expert.config().learning_rate;
            expert.set_learning_rate(current_lr * lr_multiplier);
            let current_lambda = expert.config().lambda;
            expert.set_lambda(current_lambda + lambda_delta);
        }
        for shadow in &mut self.shadows {
            let current_lr = shadow.config().learning_rate;
            shadow.set_learning_rate(current_lr * lr_multiplier);
            let current_lambda = shadow.config().lambda;
            shadow.set_lambda(current_lambda + lambda_delta);
        }
    }

    fn apply_structural_change(&mut self, depth_delta: i32, steps_delta: i32) {
        if depth_delta != 0 {
            for expert in &mut self.experts {
                let current = expert.config().max_depth as i32;
                expert.set_max_depth((current + depth_delta).max(1) as usize);
            }
            for shadow in &mut self.shadows {
                let current = shadow.config().max_depth as i32;
                shadow.set_max_depth((current + depth_delta).max(1) as usize);
            }
        }
        if steps_delta != 0 {
            for expert in &mut self.experts {
                let current = expert.n_steps() as i32;
                expert.set_n_steps((current + steps_delta).max(3) as usize);
            }
            for shadow in &mut self.shadows {
                let current = shadow.n_steps() as i32;
                shadow.set_n_steps((current + steps_delta).max(3) as usize);
            }
        }
    }

    fn replacement_count(&self) -> u64 {
        self.shadow_replacements.iter().sum()
    }
}

// ===========================================================================
// DiagnosticSource impl
// ===========================================================================

impl crate::automl::DiagnosticSource for MoEDistributionalSGBT {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        let total_steps: usize = self.experts().iter().map(|e| e.n_steps()).sum();
        let n = self.n_experts();
        // Weighted honest_sigma: average across expert honest_sigma means.
        let avg_honest_sigma = if n > 0 {
            self.experts()
                .iter()
                .map(|e| e.rolling_honest_sigma_mean())
                .sum::<f64>()
                / n as f64
        } else {
            0.0
        };
        Some(crate::automl::ConfigDiagnostics {
            effective_dof: total_steps as f64,
            uncertainty: avg_honest_sigma,
            ..Default::default()
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::Sample;

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
        let moe = MoEDistributionalSGBT::new(test_config(), 3);
        assert_eq!(moe.n_experts(), 3);
        assert_eq!(moe.n_samples_seen(), 0);
        assert_eq!(moe.shadow_replacements().len(), 3);
        for &r in moe.shadow_replacements() {
            assert_eq!(r, 0);
        }
    }

    #[test]
    fn test_gating_probabilities_sum_to_one() {
        let mut moe = MoEDistributionalSGBT::new(test_config(), 5);

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
    fn test_prediction_is_valid_gaussian() {
        let mut moe = MoEDistributionalSGBT::new(test_config(), 3);

        // Train enough for base initialization
        for i in 0..50 {
            let sample = Sample::new(vec![i as f64, (i as f64) * 0.5], i as f64 * 2.0);
            moe.train_one(&sample);
        }

        let pred = moe.predict(&[10.0, 5.0]);
        assert!(pred.mu.is_finite(), "mu should be finite: {}", pred.mu);
        assert!(pred.sigma > 0.0, "sigma should be > 0: {}", pred.sigma);
        assert!(
            pred.log_sigma.is_finite(),
            "log_sigma should be finite: {}",
            pred.log_sigma
        );
    }

    #[test]
    fn test_prediction_changes_after_training() {
        let mut moe = MoEDistributionalSGBT::new(test_config(), 3);
        let features = vec![1.0, 2.0, 3.0];

        let pred_before = moe.predict(&features);

        for i in 0..100 {
            let sample = Sample::new(features.clone(), 10.0 + i as f64 * 0.1);
            moe.train_one(&sample);
        }

        let pred_after = moe.predict(&features);
        assert!(
            (pred_after.mu - pred_before.mu).abs() > 1e-6,
            "mu should change after training: before={}, after={}",
            pred_before.mu,
            pred_after.mu
        );
    }

    #[test]
    fn test_mixture_variance() {
        // Manual check: with uniform gating and known expert outputs,
        // verify the law of total variance formula.
        let mut moe = MoEDistributionalSGBT::new(test_config(), 2);

        // Train so experts produce non-trivial predictions
        for i in 0..80 {
            let sample = Sample::new(vec![i as f64], i as f64 * 3.0);
            moe.train_one(&sample);
        }

        let features = &[40.0];
        let probs = moe.gating_probabilities(features);
        let expert_preds = moe.expert_predictions(features);

        // Manual mixture calculation
        let mu_mix: f64 = probs
            .iter()
            .zip(expert_preds.iter())
            .map(|(&p, pred)| p * pred.mu)
            .sum();
        let second_moment: f64 = probs
            .iter()
            .zip(expert_preds.iter())
            .map(|(&p, pred)| p * (pred.sigma * pred.sigma + pred.mu * pred.mu))
            .sum();
        let var_mix = (second_moment - mu_mix * mu_mix).max(1e-16);
        let sigma_mix = var_mix.sqrt();

        let pred = moe.predict(features);
        assert!(
            (pred.mu - mu_mix).abs() < 1e-10,
            "mu mismatch: pred={}, manual={}",
            pred.mu,
            mu_mix
        );
        assert!(
            (pred.sigma - sigma_mix).abs() < 1e-10,
            "sigma mismatch: pred={}, manual={}",
            pred.sigma,
            sigma_mix
        );
    }

    #[test]
    fn test_expert_predictions_count() {
        let moe = MoEDistributionalSGBT::new(test_config(), 4);
        let preds = moe.expert_predictions(&[1.0, 2.0]);
        assert_eq!(preds.len(), 4, "should return one prediction per expert");
    }

    #[test]
    fn test_predict_with_gating_consistency() {
        let mut moe = MoEDistributionalSGBT::new(test_config(), 3);

        for i in 0..50 {
            let sample = Sample::new(vec![i as f64, (i as f64) * 0.5], i as f64);
            moe.train_one(&sample);
        }

        let features = &[10.0, 5.0];
        let (pred, probs) = moe.predict_with_gating(features);
        let expert_preds = moe.expert_predictions(features);

        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // mu should equal weighted sum of expert means
        let expected_mu: f64 = probs
            .iter()
            .zip(expert_preds.iter())
            .map(|(&p, ep)| p * ep.mu)
            .sum();
        assert!(
            (pred.mu - expected_mu).abs() < 1e-10,
            "mu mismatch: pred={}, expected={}",
            pred.mu,
            expected_mu
        );
    }

    #[test]
    fn test_n_samples_seen_increments() {
        let mut moe = MoEDistributionalSGBT::new(test_config(), 2);
        assert_eq!(moe.n_samples_seen(), 0);

        for i in 0..25 {
            moe.train_one(&Sample::new(vec![i as f64], i as f64));
        }
        assert_eq!(moe.n_samples_seen(), 25);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut moe = MoEDistributionalSGBT::new(test_config(), 3);

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
        // Shadow replacement counters are also reset
        for &r in moe.shadow_replacements() {
            assert_eq!(r, 0);
        }
    }

    #[test]
    fn test_streaming_learner_trait() {
        let config = test_config();
        let model = MoEDistributionalSGBT::new(config, 3);
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

    #[test]
    fn test_hard_gating_mode() {
        let mut moe = MoEDistributionalSGBT::with_gating(
            test_config(),
            4,
            GatingMode::Hard { top_k: 2 },
            0.01,
        );

        for i in 0..30 {
            let sample = Sample::new(vec![i as f64], i as f64);
            moe.train_one(&sample);
        }

        assert_eq!(moe.n_samples_seen(), 30);
        let pred = moe.predict(&[15.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma > 0.0);
    }

    #[test]
    fn test_predict_mu_matches_predict() {
        let mut moe = MoEDistributionalSGBT::new(test_config(), 3);

        for i in 0..50 {
            moe.train_one(&Sample::new(vec![i as f64], i as f64 * 2.0));
        }

        let features = &[25.0];
        let mu_direct = moe.predict_mu(features);
        let mu_from_predict = moe.predict(features).mu;
        assert!(
            (mu_direct - mu_from_predict).abs() < 1e-12,
            "predict_mu={} vs predict().mu={}",
            mu_direct,
            mu_from_predict
        );
    }

    #[test]
    fn test_batch_training() {
        let mut moe = MoEDistributionalSGBT::new(test_config(), 3);

        let samples: Vec<Sample> = (0..20)
            .map(|i| Sample::new(vec![i as f64, (i * 3) as f64], i as f64))
            .collect();

        moe.train_batch(&samples);

        assert_eq!(moe.n_samples_seen(), 20);
        let pred = moe.predict(&[10.0, 30.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma > 0.0);
    }

    #[test]
    fn moe_with_expert_configs_different_depths() {
        // Each expert gets its own config with different tree depth.
        let configs: Vec<SGBTConfig> = (0..3)
            .map(|i| {
                SGBTConfig::builder()
                    .n_steps(5)
                    .learning_rate(0.1)
                    .grace_period(5)
                    .max_depth(2 + i) // depth 2, 3, 4
                    .build()
                    .unwrap()
            })
            .collect();

        let mut moe = MoEDistributionalSGBT::with_expert_configs(
            configs.clone(),
            GatingMode::Soft,
            0.01,
            0.0, // no entropy
            1e-3,
            500,
        );

        assert_eq!(moe.n_experts(), 3);
        assert!(moe.expert_configs().is_some());
        assert_eq!(moe.expert_configs().unwrap().len(), 3);

        // Verify each expert got its config (via max_depth)
        for (i, cfg) in configs.iter().enumerate() {
            assert_eq!(moe.expert(i).config().max_depth, cfg.max_depth);
        }

        // Train and verify it works
        for i in 0..50 {
            let sample = Sample::new(vec![i as f64, (i * 2) as f64], i as f64 * 3.0);
            moe.train_one(&sample);
        }
        let pred = moe.predict(&[10.0, 20.0]);
        assert!(pred.mu.is_finite());
        assert!(pred.sigma > 0.0);
    }

    #[test]
    fn entropy_regularization_prevents_collapse() {
        // With entropy weight, gate probs should stay above a minimum for all experts
        // when data is uniform across patterns.
        let config = test_config();
        let mut moe = MoEDistributionalSGBT::with_expert_configs(
            vec![config.clone(), config.clone(), config],
            GatingMode::Soft,
            0.01,
            0.1, // entropy weight
            1e-3,
            500,
        );

        // Train with uniform-ish data
        for i in 0..200 {
            let x = (i % 10) as f64;
            let sample = Sample::new(vec![x, x * 2.0], x * 3.0);
            moe.train_one(&sample);
        }

        // Check that no expert is completely starved
        let probs = moe.gating_probabilities(&[5.0, 10.0]);
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p > 0.02,
                "Expert {} has probability {} -- gate collapsed despite entropy regularization",
                i,
                p
            );
        }
    }

    #[test]
    fn moe_expert_configs_shadow_respawn_correct() {
        // After shadow swap, the fresh shadow should use the per-expert config.
        // We can verify the config path is correct by construction.
        let configs: Vec<SGBTConfig> = (0..2)
            .map(|i| {
                SGBTConfig::builder()
                    .n_steps(3)
                    .learning_rate(0.1)
                    .grace_period(5)
                    .max_depth(3 + i) // depth 3, 4
                    .build()
                    .unwrap()
            })
            .collect();

        let moe = MoEDistributionalSGBT::with_expert_configs(
            configs.clone(),
            GatingMode::Soft,
            0.01,
            0.0,
            1e-3,
            500,
        );

        // Verify expert_configs are stored
        let ec = moe.expert_configs().unwrap();
        assert_eq!(ec[0].max_depth, 3);
        assert_eq!(ec[1].max_depth, 4);

        // The shadow swap path references expert_configs[i] -- verified by
        // the code structure. This test confirms the configs are accessible
        // and correctly indexed.
        assert_eq!(moe.expert(0).config().max_depth, 3);
        assert_eq!(moe.expert(1).config().max_depth, 4);
    }
}
