//! Linear softmax router for streaming neural MoE.
//!
//! Computes gating probabilities via a linear layer + softmax, with an
//! additive per-expert bias for load balancing (DeepSeek-v3 style, Wang+2024).
//! Trained online via SGD on cross-entropy against the best-performing expert.

/// Numerically stable softmax.
pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-16 {
        // Uniform fallback
        let n = logits.len() as f64;
        return vec![1.0 / n; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Linear softmax router with load-balancing bias.
///
/// Produces K gating probabilities from input features via:
/// `z_k = W_k . x + b_k + load_bias_k`
/// `p = softmax(z)`
///
/// The load bias is updated to encourage uniform expert utilization,
/// preventing routing collapse where a few experts dominate.
pub(crate) struct LinearRouter {
    weights: Vec<Vec<f64>>, // [K x d] weight matrix
    gate_bias: Vec<f64>,    // [K] learned gate bias
    load_bias: Vec<f64>,    // [K] load-balancing bias (not learned by SGD)
    load_ewma: Vec<f64>,    // [K] EWMA of actual routing load per expert
    lr: f64,                // Learning rate for gate SGD
    load_balance_rate: f64, // Rate of load bias adjustment
    load_alpha: f64,        // EWMA decay factor for load tracking
    k_experts: usize,       // Total number of routed experts
    initialized: bool,      // Whether weights have been lazily initialized
}

impl LinearRouter {
    /// Create a new router for `k_experts` experts.
    ///
    /// Weights are lazily initialized on first use (when input dimension is known).
    pub fn new(k_experts: usize, lr: f64, load_balance_rate: f64, utilization_span: usize) -> Self {
        let load_alpha = 2.0 / (utilization_span as f64 + 1.0);
        Self {
            weights: Vec::new(),
            gate_bias: vec![0.0; k_experts],
            load_bias: vec![0.0; k_experts],
            load_ewma: vec![0.0; k_experts],
            lr,
            load_balance_rate,
            load_alpha,
            k_experts,
            initialized: false,
        }
    }

    /// Lazily initialize weights to zeros when input dimension is first known.
    fn ensure_init(&mut self, d: usize) {
        if !self.initialized {
            self.weights = vec![vec![0.0; d]; self.k_experts];
            self.initialized = true;
        }
    }

    /// Compute raw logits: z_k = W_k . x + b_k + load_bias_k
    pub fn logits(&self, features: &[f64]) -> Vec<f64> {
        if !self.initialized {
            return vec![0.0; self.k_experts]; // uniform before init
        }
        (0..self.k_experts)
            .map(|k| {
                let dot: f64 = self.weights[k]
                    .iter()
                    .zip(features.iter())
                    .map(|(&w, &x)| w * x)
                    .sum();
                dot + self.gate_bias[k] + self.load_bias[k]
            })
            .collect()
    }

    /// Compute softmax probabilities from logits.
    pub fn probabilities(&self, features: &[f64]) -> Vec<f64> {
        softmax(&self.logits(features))
    }

    /// Select top-k expert indices by highest probability.
    ///
    /// Returns indices sorted by descending probability.
    /// If k >= n_experts, returns all indices.
    pub fn select_top_k(&self, features: &[f64], k: usize) -> Vec<usize> {
        let probs = self.probabilities(features);
        let mut indexed: Vec<(usize, f64)> = probs.into_iter().enumerate().collect();
        // Sort by probability descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .into_iter()
            .take(k.min(self.k_experts))
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute renormalized probabilities for the given active indices.
    ///
    /// Returns (index, renormalized_weight) pairs where weights sum to 1.0
    /// among the active experts only.
    pub fn renormalized_weights(
        &self,
        features: &[f64],
        active_indices: &[usize],
    ) -> Vec<(usize, f64)> {
        let probs = self.probabilities(features);
        let sum: f64 = active_indices.iter().map(|&i| probs[i]).sum();
        if sum < 1e-16 {
            // Uniform fallback
            let w = 1.0 / active_indices.len() as f64;
            return active_indices.iter().map(|&i| (i, w)).collect();
        }
        active_indices
            .iter()
            .map(|&i| (i, probs[i] / sum))
            .collect()
    }

    /// Update router weights via SGD on cross-entropy loss.
    ///
    /// `best_idx` is the expert with lowest prediction error.
    /// Gradient: d_k = p_k - 1{k == best_idx}
    /// Update: W_k -= lr * d_k * x,  b_k -= lr * d_k
    pub fn update(&mut self, features: &[f64], best_idx: usize) {
        self.ensure_init(features.len());

        let probs = self.probabilities(features);

        for (k, prob_k) in probs.iter().enumerate() {
            let indicator = if k == best_idx { 1.0 } else { 0.0 };
            let grad = prob_k - indicator;

            // Update weights
            for (j, &xj) in features.iter().enumerate() {
                self.weights[k][j] -= self.lr * grad * xj;
            }
            // Update bias
            self.gate_bias[k] -= self.lr * grad;
        }
    }

    /// Update load balancing: bias experts toward uniform utilization.
    ///
    /// For each expert, updates the load EWMA based on whether it was
    /// activated this step. The load bias is adjusted to push underloaded
    /// experts' routing probability up and overloaded experts' down.
    pub fn update_load_balance(&mut self, active_indices: &[usize]) {
        let target_load = 1.0 / self.k_experts as f64;

        for k in 0..self.k_experts {
            // 1.0 if active this step, 0.0 otherwise
            let active = if active_indices.contains(&k) {
                1.0
            } else {
                0.0
            };

            // Update EWMA of load
            self.load_ewma[k] =
                self.load_alpha * active + (1.0 - self.load_alpha) * self.load_ewma[k];

            // Adjust bias: positive if underloaded (need more), negative if overloaded
            self.load_bias[k] += self.load_balance_rate * (target_load - self.load_ewma[k]);
        }
    }

    /// Current load distribution (EWMA of activation frequency per expert).
    pub fn load_distribution(&self) -> &[f64] {
        &self.load_ewma
    }

    /// Number of experts this router serves.
    #[allow(dead_code)]
    pub fn n_experts(&self) -> usize {
        self.k_experts
    }

    /// Reset all weights, biases, and load tracking to initial state.
    pub fn reset(&mut self) {
        self.weights.clear();
        self.gate_bias.fill(0.0);
        self.load_bias.fill(0.0);
        self.load_ewma.fill(0.0);
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_uniform_input() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let probs = softmax(&logits);
        let expected = 0.25;
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-10,
                "expert {} should have probability {}, got {}",
                i,
                expected,
                p
            );
        }
    }

    #[test]
    fn softmax_numerical_stability() {
        let logits = vec![1000.0, 1000.0, 1000.0];
        let probs = softmax(&logits);
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p.is_finite(),
                "expert {} probability should be finite with large logits, got {}",
                i,
                p
            );
        }
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax with large logits should still sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![0.5, -1.2, 3.7, 0.0, 2.1];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax probabilities should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn router_uniform_before_init() {
        let router = LinearRouter::new(4, 0.01, 0.001, 100);
        let features = vec![1.0, 2.0, 3.0];
        let probs = router.probabilities(&features);
        let expected = 0.25;
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-10,
                "expert {} should have uniform probability {} before init, got {}",
                i,
                expected,
                p
            );
        }
    }

    #[test]
    fn router_select_top_k() {
        let mut router = LinearRouter::new(4, 0.1, 0.0, 100);
        let features = vec![1.0, 0.5];
        // Train repeatedly to favor expert 2
        for _ in 0..100 {
            router.update(&features, 2);
        }
        let top = router.select_top_k(&features, 2);
        assert_eq!(
            top.len(),
            2,
            "select_top_k(2) should return exactly 2 indices"
        );
        assert_eq!(
            top[0], 2,
            "expert 2 should be the highest-probability expert after training, got {}",
            top[0]
        );
    }

    #[test]
    fn router_select_top_k_exceeds_n() {
        let router = LinearRouter::new(3, 0.01, 0.0, 100);
        let features = vec![1.0];
        let top = router.select_top_k(&features, 10);
        assert_eq!(
            top.len(),
            3,
            "select_top_k(10) with 3 experts should return all 3, got {}",
            top.len()
        );
    }

    #[test]
    fn router_update_shifts_distribution() {
        let mut router = LinearRouter::new(3, 0.1, 0.0, 100);
        let features = vec![1.0, -0.5, 2.0];
        // Train heavily toward expert 0
        for _ in 0..200 {
            router.update(&features, 0);
        }
        let probs = router.probabilities(&features);
        assert!(
            probs[0] > probs[1],
            "expert 0 probability ({}) should exceed expert 1 ({}) after training on expert 0",
            probs[0],
            probs[1]
        );
        assert!(
            probs[0] > probs[2],
            "expert 0 probability ({}) should exceed expert 2 ({}) after training on expert 0",
            probs[0],
            probs[2]
        );
    }

    #[test]
    fn router_renormalized_weights_sum_to_one() {
        let mut router = LinearRouter::new(5, 0.05, 0.0, 100);
        let features = vec![0.3, 1.0, -0.7];
        // Do some training to create non-uniform distribution
        for _ in 0..50 {
            router.update(&features, 1);
        }
        for _ in 0..30 {
            router.update(&features, 3);
        }
        let active = vec![0, 2, 4];
        let renorm = router.renormalized_weights(&features, &active);
        assert_eq!(
            renorm.len(),
            3,
            "renormalized_weights should return one entry per active index"
        );
        let sum: f64 = renorm.iter().map(|(_, w)| w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "renormalized weights should sum to 1.0, got {}",
            sum
        );
        // Verify correct indices
        for (i, &(idx, _)) in renorm.iter().enumerate() {
            assert_eq!(
                idx, active[i],
                "renormalized weight index mismatch at position {}",
                i
            );
        }
    }

    #[test]
    fn router_load_balance_corrects_imbalance() {
        let mut router = LinearRouter::new(4, 0.01, 0.1, 20);
        let features = vec![1.0];
        // Initialize weights so logits reflect biases
        router.update(&features, 0);
        // Reset gate_bias so only load_bias drives asymmetry
        router.gate_bias.fill(0.0);
        for row in router.weights.iter_mut() {
            row.fill(0.0);
        }
        // Always activate only expert 0
        for _ in 0..200 {
            router.update_load_balance(&[0]);
        }
        let loads = router.load_distribution();
        // Expert 0 should have high load, others near zero
        assert!(
            loads[0] > 0.5,
            "expert 0 load should be high after always being active, got {}",
            loads[0]
        );
        for k in 1..4 {
            assert!(
                loads[k] < 0.05,
                "expert {} load should be near zero when never active, got {}",
                k,
                loads[k]
            );
        }
        // Load bias should favor underloaded experts
        // Expert 0 is overloaded: its load_bias should be negative
        // Others are underloaded: their load_bias should be positive
        let logits_out = router.logits(&features);
        // Underloaded experts should have higher load_bias contribution
        // Since expert 0 is overloaded, its bias penalty should push its logit down
        // relative to others (weights and gate_bias are zeroed)
        assert!(
            logits_out[1] > logits_out[0],
            "underloaded expert 1 logit ({}) should exceed overloaded expert 0 logit ({}) due to load balancing",
            logits_out[1],
            logits_out[0]
        );
    }

    #[test]
    fn router_reset_clears_state() {
        let mut router = LinearRouter::new(3, 0.1, 0.01, 50);
        let features = vec![1.0, 2.0];
        // Train to create non-trivial state
        for _ in 0..100 {
            router.update(&features, 1);
            router.update_load_balance(&[1]);
        }
        // Verify non-trivial state
        let probs_before = router.probabilities(&features);
        assert!(
            (probs_before[0] - probs_before[1]).abs() > 0.01,
            "router should have non-uniform distribution before reset"
        );

        router.reset();

        // After reset: should be uniform again (uninitialized)
        let probs_after = router.probabilities(&features);
        let expected = 1.0 / 3.0;
        for (i, &p) in probs_after.iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-10,
                "expert {} should have uniform probability {} after reset, got {}",
                i,
                expected,
                p
            );
        }
        // Load tracking should be zeroed
        for (k, &load) in router.load_distribution().iter().enumerate() {
            assert!(
                load.abs() < 1e-16,
                "expert {} load should be zero after reset, got {}",
                k,
                load
            );
        }
        assert_eq!(
            router.n_experts(),
            3,
            "n_experts should be unchanged after reset"
        );
    }
}
