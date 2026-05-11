//! Multi-Query Associative Recall (MQAR) generator.
//!
//! Generates a two-phase stream: a **bind** phase that presents N (key, value)
//! pairs, followed by a **recall** phase that presents each key as a query and
//! expects the associated value as the target. Useful for benchmarking models
//! that need in-context associative memory.
//!
//! # Protocol
//!
//! Each epoch consists of:
//! 1. **Bind phase** (`n_pairs` samples): `(key_i, value_i[0])` — the model
//!    observes each key with its first value component as the regression target.
//! 2. **Recall phase** (`n_pairs` samples): `(key_i, value_i[0])` — the model
//!    must reproduce the same values from the same keys presented in the same order.
//!
//! The key observation: a model with no temporal memory cannot succeed at recall
//! because the (key, value) mappings were only available during the bind phase.
//! A model with sufficient associative memory can bind during bind and reproduce
//! during recall.
//!
//! # References
//!
//! Based on the MQAR benchmark used in Arora et al. (2024), "Simple Linear
//! Attention Language Models Balance the Recall-Throughput Tradeoff."

use super::{Rng, StreamGenerator, TaskType};

/// Multi-Query Associative Recall stream generator.
///
/// # Parameters
/// - `seed`: PRNG seed for reproducibility
/// - `n_pairs`: number of (key, value) pairs per epoch (default: 128)
/// - `d_key`: dimension of each key vector (default: 8)
/// - `d_value`: dimension of each value vector (default: 4)
///
/// # Output
/// - Features: key vector (`d_key` dims)
/// - Target: first component of the associated value vector
///
/// # Phase tracking
///
/// `drift_occurred()` returns `true` at the boundary between bind and recall
/// phases, and again at each epoch wrap (recall → bind). This lets benchmarks
/// identify phase transitions.
#[derive(Debug, Clone)]
pub struct MqarStream {
    /// Keys for all pairs in the current epoch (row-major, n_pairs × d_key).
    keys: Vec<f64>,
    /// Values for all pairs in the current epoch (row-major, n_pairs × d_value).
    values: Vec<f64>,
    /// PRNG for generating new epochs.
    rng: Rng,
    /// Number of (key, value) pairs per epoch.
    n_pairs: usize,
    /// Key dimensionality.
    d_key: usize,
    /// Value dimensionality.
    d_value: usize,
    /// Current index within the active phase (0..n_pairs).
    pair_idx: usize,
    /// Whether we are in the recall phase (`true`) or bind phase (`false`).
    in_recall: bool,
    /// Whether a phase boundary occurred on the most recent `next_sample` call.
    drift_flag: bool,
}

impl MqarStream {
    /// Default number of (key, value) pairs per epoch.
    pub const DEFAULT_N_PAIRS: usize = 128;
    /// Default key dimension.
    pub const DEFAULT_D_KEY: usize = 8;
    /// Default value dimension.
    pub const DEFAULT_D_VALUE: usize = 4;

    /// Create an MQAR generator.
    ///
    /// - `seed`: PRNG seed.
    /// - `d_key`: key and feature dimension.
    /// - `n_pairs`: number of (key, value) pairs per epoch.
    ///
    /// Value dimension defaults to `d_key / 2` (minimum 1).
    pub fn new(seed: u64, d_key: usize, n_pairs: usize) -> Self {
        let d_value = (d_key / 2).max(1);
        Self::with_config(seed, n_pairs, d_key, d_value)
    }

    /// Create an MQAR generator with custom parameters.
    ///
    /// # Panics
    ///
    /// Panics if `n_pairs == 0`, `d_key == 0`, or `d_value == 0`.
    pub fn with_config(seed: u64, n_pairs: usize, d_key: usize, d_value: usize) -> Self {
        assert!(n_pairs > 0, "n_pairs must be > 0");
        assert!(d_key > 0, "d_key must be > 0");
        assert!(d_value > 0, "d_value must be > 0");

        let mut rng = Rng::new(seed);
        let (keys, values) = Self::generate_epoch(&mut rng, n_pairs, d_key, d_value);

        Self {
            keys,
            values,
            rng,
            n_pairs,
            d_key,
            d_value,
            pair_idx: 0,
            in_recall: false,
            drift_flag: false,
        }
    }

    /// Generate a fresh set of random keys and values for one epoch.
    ///
    /// Keys are unit-normalized random vectors (L2-norm = 1) in `d_key` dims.
    /// Values are uniform in `[-1, 1]` in `d_value` dims.
    fn generate_epoch(
        rng: &mut Rng,
        n_pairs: usize,
        d_key: usize,
        d_value: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut keys = Vec::with_capacity(n_pairs * d_key);
        let mut values = Vec::with_capacity(n_pairs * d_value);

        for _ in 0..n_pairs {
            // Random unit-norm key (avoids degenerate zero or near-zero keys).
            let mut norm_sq = 0.0;
            let mut raw: Vec<f64> = (0..d_key)
                .map(|_| {
                    let v = rng.normal(0.0, 1.0);
                    norm_sq += v * v;
                    v
                })
                .collect();
            let norm = norm_sq.sqrt().max(1e-12);
            for v in raw.iter_mut() {
                *v /= norm;
            }
            keys.extend_from_slice(&raw);

            // Random value in [-1, 1].
            for _ in 0..d_value {
                values.push(rng.uniform_range(-1.0, 1.0));
            }
        }

        (keys, values)
    }

    /// Return the key vector for the current pair index.
    fn current_key(&self) -> Vec<f64> {
        let start = self.pair_idx * self.d_key;
        self.keys[start..start + self.d_key].to_vec()
    }

    /// Return the first component of the value vector for the current pair index.
    fn current_target(&self) -> f64 {
        self.values[self.pair_idx * self.d_value]
    }

    /// Number of (key, value) pairs per epoch.
    pub fn n_pairs(&self) -> usize {
        self.n_pairs
    }

    /// Key dimension.
    pub fn d_key(&self) -> usize {
        self.d_key
    }

    /// Value dimension.
    pub fn d_value(&self) -> usize {
        self.d_value
    }

    /// Whether the generator is currently in the recall phase.
    pub fn in_recall_phase(&self) -> bool {
        self.in_recall
    }
}

impl StreamGenerator for MqarStream {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        self.drift_flag = false;

        let features = self.current_key();
        let target = self.current_target();

        self.pair_idx += 1;

        if self.pair_idx >= self.n_pairs {
            self.pair_idx = 0;
            if self.in_recall {
                // End of recall phase: start a fresh epoch (new keys/values).
                let (new_keys, new_values) =
                    Self::generate_epoch(&mut self.rng, self.n_pairs, self.d_key, self.d_value);
                self.keys = new_keys;
                self.values = new_values;
                self.in_recall = false;
            } else {
                // End of bind phase: enter recall with the same pairs.
                self.in_recall = true;
            }
            self.drift_flag = true;
        }

        (features, target)
    }

    fn n_features(&self) -> usize {
        self.d_key
    }

    fn task_type(&self) -> TaskType {
        TaskType::Regression
    }

    fn drift_occurred(&self) -> bool {
        self.drift_flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mqar_produces_correct_n_features() {
        let d_key = MqarStream::DEFAULT_D_KEY;
        let mut gen = MqarStream::new(42, d_key, MqarStream::DEFAULT_N_PAIRS);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            d_key,
            "features should have d_key={} dims, got {}",
            d_key,
            features.len()
        );
    }

    #[test]
    fn mqar_task_type_is_regression() {
        let gen = MqarStream::new(42, MqarStream::DEFAULT_D_KEY, MqarStream::DEFAULT_N_PAIRS);
        assert_eq!(gen.task_type(), TaskType::Regression);
    }

    #[test]
    fn mqar_produces_finite_values() {
        let mut gen = MqarStream::new(7, 8, 32);
        for i in 0..512 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(f.is_finite(), "feature {} at sample {} is not finite", j, i);
            }
            assert!(target.is_finite(), "target at sample {} is not finite", i);
        }
    }

    #[test]
    fn mqar_deterministic_with_same_seed() {
        let mut gen1 = MqarStream::new(42, 8, 32);
        let mut gen2 = MqarStream::new(42, 8, 32);
        for _ in 0..512 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }

    #[test]
    fn mqar_recall_keys_match_bind_keys() {
        // During the bind phase, record (key, target) pairs.
        // During the recall phase, the same keys must appear with the same targets.
        let n = 16;
        let mut gen = MqarStream::with_config(1234, n, 4, 2);

        // Consume bind phase
        let mut bind_pairs: Vec<(Vec<f64>, f64)> = Vec::new();
        for _ in 0..n {
            bind_pairs.push(gen.next_sample());
        }
        // The n-th call advanced into recall phase (drift_flag set on last sample return).

        // Consume recall phase
        let mut recall_pairs: Vec<(Vec<f64>, f64)> = Vec::new();
        for _ in 0..n {
            recall_pairs.push(gen.next_sample());
        }

        assert_eq!(bind_pairs.len(), recall_pairs.len());
        for (i, ((bf, bt), (rf, rt))) in bind_pairs.iter().zip(recall_pairs.iter()).enumerate() {
            assert_eq!(bf, rf, "bind and recall keys must match at pair {}", i);
            assert!(
                (bt - rt).abs() < 1e-12,
                "bind and recall targets must match at pair {}: bind={}, recall={}",
                i,
                bt,
                rt
            );
        }
    }

    #[test]
    fn mqar_phase_boundary_drift_flag() {
        let n = 8;
        let mut gen = MqarStream::with_config(99, n, 4, 2);

        // Collect n-1 samples (no drift yet within bind phase).
        for i in 0..n - 1 {
            gen.next_sample();
            assert!(
                !gen.drift_occurred(),
                "no drift expected at bind sample {}",
                i
            );
        }
        // n-th sample completes bind phase → drift.
        gen.next_sample();
        assert!(
            gen.drift_occurred(),
            "drift expected at bind→recall boundary"
        );

        // n-1 samples in recall phase (no drift).
        for i in 0..n - 1 {
            gen.next_sample();
            assert!(
                !gen.drift_occurred(),
                "no drift expected at recall sample {}",
                i
            );
        }
        // n-th recall sample → new epoch drift.
        gen.next_sample();
        assert!(
            gen.drift_occurred(),
            "drift expected at recall→bind boundary"
        );
    }

    #[test]
    fn mqar_custom_config_dimensions() {
        let mut gen = MqarStream::with_config(1, 32, 6, 3);
        let (features, _) = gen.next_sample();
        assert_eq!(features.len(), 6);
        assert_eq!(gen.n_pairs(), 32);
        assert_eq!(gen.d_key(), 6);
        assert_eq!(gen.d_value(), 3);
        assert_eq!(gen.n_features(), 6);

        // Also test the 3-arg constructor.
        let gen2 = MqarStream::new(1, 8, 16);
        assert_eq!(gen2.n_features(), 8);
        assert_eq!(gen2.n_pairs(), 16);
        assert_eq!(gen2.d_value(), 4); // d_key / 2
    }

    #[test]
    fn mqar_keys_are_unit_norm() {
        let n = 16;
        let mut gen = MqarStream::with_config(55, n, 8, 4);
        for i in 0..n {
            let (features, _) = gen.next_sample();
            let norm_sq: f64 = features.iter().map(|v| v * v).sum();
            assert!(
                (norm_sq.sqrt() - 1.0).abs() < 1e-9,
                "bind key {} should be unit-norm, got norm_sq={}",
                i,
                norm_sq
            );
        }
    }
}
