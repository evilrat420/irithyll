//! Configuration space for hyperparameter search.
//!
//! Defines the search space over hyperparameters and provides samplers
//! for generating candidate configurations. Supports continuous (linear
//! and log-scale), integer, and categorical parameters.

// ---------------------------------------------------------------------------
// RNG helpers (xorshift64, same as the rest of irithyll)
// ---------------------------------------------------------------------------

/// Advance xorshift64 state and return raw u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Uniform f64 in [0, 1) from xorshift64.
#[inline]
fn xorshift64_f64(state: &mut u64) -> f64 {
    // Use 53-bit mantissa for full f64 precision.
    (xorshift64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

// ---------------------------------------------------------------------------
// HyperParam
// ---------------------------------------------------------------------------

/// A single hyperparameter definition.
#[derive(Debug, Clone)]
pub enum HyperParam {
    /// Continuous float parameter.
    Float {
        name: &'static str,
        low: f64,
        high: f64,
        log_scale: bool,
    },
    /// Integer parameter.
    Int {
        name: &'static str,
        low: i64,
        high: i64,
    },
    /// Categorical parameter (encoded as index `0..n_choices-1`).
    Categorical {
        name: &'static str,
        n_choices: usize,
    },
}

impl HyperParam {
    /// Returns the name of the hyperparameter regardless of variant.
    pub fn name(&self) -> &str {
        match self {
            HyperParam::Float { name, .. } => name,
            HyperParam::Int { name, .. } => name,
            HyperParam::Categorical { name, .. } => name,
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigSpace
// ---------------------------------------------------------------------------

/// A configuration space defining the hyperparameter search bounds.
#[derive(Debug, Clone)]
pub struct ConfigSpace {
    params: Vec<HyperParam>,
}

impl ConfigSpace {
    /// Create an empty configuration space.
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Add a parameter to the space (builder-style).
    pub fn push(mut self, param: HyperParam) -> Self {
        self.params.push(param);
        self
    }

    /// Access the parameter definitions.
    pub fn params(&self) -> &[HyperParam] {
        &self.params
    }

    /// Number of parameters in the space.
    pub fn n_params(&self) -> usize {
        self.params.len()
    }

    /// Dimensionality of the space (alias for [`n_params`](Self::n_params)).
    pub fn dim(&self) -> usize {
        self.params.len()
    }
}

impl Default for ConfigSpace {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HyperConfig
// ---------------------------------------------------------------------------

/// A concrete hyperparameter configuration (point in the config space).
///
/// All values are encoded as `f64`: floats are raw values, ints are cast,
/// categoricals are indices.
#[derive(Debug, Clone)]
pub struct HyperConfig {
    values: Vec<f64>,
}

impl HyperConfig {
    /// Create a new configuration from raw values.
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Access the raw values slice.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Get the value at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    pub fn get(&self, index: usize) -> f64 {
        self.values[index]
    }

    /// Number of values in this configuration.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether this configuration has no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ConfigSampler
// ---------------------------------------------------------------------------

/// Sampler for generating hyperparameter configurations from a [`ConfigSpace`].
pub struct ConfigSampler {
    space: ConfigSpace,
    rng_state: u64,
}

impl ConfigSampler {
    /// Create a new sampler for `space` with the given PRNG seed.
    ///
    /// # Panics
    ///
    /// Panics if `seed == 0` (xorshift64 requires a non-zero seed).
    pub fn new(space: ConfigSpace, seed: u64) -> Self {
        assert!(seed != 0, "ConfigSampler seed must be non-zero");
        Self {
            space,
            rng_state: seed,
        }
    }

    /// Access the underlying configuration space.
    pub fn space(&self) -> &ConfigSpace {
        &self.space
    }

    /// Sample one random configuration from the space.
    ///
    /// - **Float** (linear): uniform in `[low, high]`.
    /// - **Float** (log-scale): uniform in `[ln(low), ln(high)]`, then `exp()`.
    /// - **Int**: uniform in `[low, high]` (inclusive), cast to `f64`.
    /// - **Categorical**: uniform in `[0, n_choices-1]`, as `f64`.
    pub fn random(&mut self) -> HyperConfig {
        let rng = &mut self.rng_state;
        let values = self
            .space
            .params
            .iter()
            .map(|p| {
                let u = xorshift64_f64(rng);
                Self::map_unit_to_param(u, p)
            })
            .collect();
        HyperConfig { values }
    }

    /// Generate `n` configurations using Latin Hypercube Sampling.
    ///
    /// For each dimension the `[0, 1]` interval is divided into `n` equal
    /// strata; one point is sampled per stratum, then the strata are shuffled
    /// independently per dimension. The stratified samples are mapped to the
    /// actual parameter ranges (respecting log-scale, integer rounding, and
    /// categorical encoding).
    pub fn latin_hypercube(&mut self, n: usize) -> Vec<HyperConfig> {
        if n == 0 {
            return Vec::new();
        }

        let d = self.space.n_params();
        // For each dimension, generate stratified [0,1) samples and shuffle.
        // stratified[dim][sample_index]
        let mut stratified: Vec<Vec<f64>> = Vec::with_capacity(d);
        for _ in 0..d {
            let mut column: Vec<f64> = (0..n)
                .map(|i| {
                    let lo = i as f64 / n as f64;
                    let hi = (i + 1) as f64 / n as f64;
                    let u = xorshift64_f64(&mut self.rng_state);
                    lo + u * (hi - lo)
                })
                .collect();

            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = (xorshift64(&mut self.rng_state) as usize) % (i + 1);
                column.swap(i, j);
            }

            stratified.push(column);
        }

        // Build configs by mapping [0,1) to actual parameter ranges.
        (0..n)
            .map(|i| {
                let values = self
                    .space
                    .params
                    .iter()
                    .enumerate()
                    .map(|(dim, param)| Self::map_unit_to_param(stratified[dim][i], param))
                    .collect();
                HyperConfig { values }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Map a uniform `[0, 1)` value to the parameter's range.
    fn map_unit_to_param(u: f64, param: &HyperParam) -> f64 {
        match param {
            HyperParam::Float {
                low,
                high,
                log_scale,
                ..
            } => {
                if *log_scale {
                    let ln_low = low.ln();
                    let ln_high = high.ln();
                    (ln_low + u * (ln_high - ln_low)).exp()
                } else {
                    low + u * (high - low)
                }
            }
            HyperParam::Int { low, high, .. } => {
                let range = (*high - *low + 1) as f64;
                let v = *low as f64 + (u * range).floor();
                // Clamp to [low, high] in case u rounds up.
                v.min(*high as f64).max(*low as f64)
            }
            HyperParam::Categorical { n_choices, .. } => {
                let v = (u * *n_choices as f64).floor();
                v.min((*n_choices - 1) as f64)
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a space with 3 params (float, int, categorical) and verify n_params.
    #[test]
    fn config_space_builder() {
        let space = ConfigSpace::new()
            .push(HyperParam::Float {
                name: "learning_rate",
                low: 0.001,
                high: 1.0,
                log_scale: true,
            })
            .push(HyperParam::Int {
                name: "n_trees",
                low: 10,
                high: 200,
            })
            .push(HyperParam::Categorical {
                name: "loss",
                n_choices: 3,
            });

        assert_eq!(
            space.n_params(),
            3,
            "expected 3 params, got {}",
            space.n_params()
        );
        assert_eq!(space.dim(), space.n_params(), "dim should alias n_params");
        assert_eq!(space.params()[0].name(), "learning_rate");
        assert_eq!(space.params()[1].name(), "n_trees");
        assert_eq!(space.params()[2].name(), "loss");
    }

    /// Sample 100 random configs and verify all values fall within bounds.
    #[test]
    fn random_sample_in_bounds() {
        let space = ConfigSpace::new()
            .push(HyperParam::Float {
                name: "lr",
                low: 0.001,
                high: 1.0,
                log_scale: false,
            })
            .push(HyperParam::Int {
                name: "depth",
                low: 1,
                high: 20,
            })
            .push(HyperParam::Categorical {
                name: "act",
                n_choices: 5,
            });

        let mut sampler = ConfigSampler::new(space, 42);
        for i in 0..100 {
            let cfg = sampler.random();
            assert_eq!(cfg.len(), 3, "config should have 3 values");

            let lr = cfg.get(0);
            assert!(
                (0.001..=1.0).contains(&lr),
                "sample {i}: lr={lr} out of [0.001, 1.0]"
            );

            let depth = cfg.get(1);
            assert!(
                (1.0..=20.0).contains(&depth),
                "sample {i}: depth={depth} out of [1, 20]"
            );
            assert_eq!(
                depth,
                depth.floor(),
                "sample {i}: depth={depth} should be integer"
            );

            let act = cfg.get(2);
            assert!(
                (0.0..=4.0).contains(&act),
                "sample {i}: act={act} out of [0, 4]"
            );
            assert_eq!(
                act,
                act.floor(),
                "sample {i}: act={act} should be integer index"
            );
        }
    }

    /// Verify log-scale params produce values in [low, high].
    #[test]
    fn log_scale_sampling() {
        let space = ConfigSpace::new().push(HyperParam::Float {
            name: "lr",
            low: 1e-5,
            high: 1.0,
            log_scale: true,
        });

        let mut sampler = ConfigSampler::new(space, 77);
        for i in 0..200 {
            let cfg = sampler.random();
            let v = cfg.get(0);
            assert!(
                (1e-5..=1.0).contains(&v),
                "sample {i}: log-scale value {v} out of [1e-5, 1.0]"
            );
        }
    }

    /// Verify LHS produces n configs with good coverage (no two in same stratum).
    #[test]
    fn latin_hypercube_coverage() {
        let space = ConfigSpace::new()
            .push(HyperParam::Float {
                name: "x",
                low: 0.0,
                high: 1.0,
                log_scale: false,
            })
            .push(HyperParam::Float {
                name: "y",
                low: 0.0,
                high: 1.0,
                log_scale: false,
            });

        let n = 10;
        let mut sampler = ConfigSampler::new(space, 99);
        let configs = sampler.latin_hypercube(n);

        assert_eq!(configs.len(), n, "expected {n} configs from LHS");

        // For each dimension, assign each value to a stratum and verify
        // every stratum is covered exactly once.
        for dim in 0..2 {
            let mut strata = vec![false; n];
            for cfg in &configs {
                let v = cfg.get(dim);
                let stratum = (v * n as f64).floor() as usize;
                let stratum = stratum.min(n - 1);
                assert!(
                    !strata[stratum],
                    "dim {dim}: stratum {stratum} hit twice (value={v})"
                );
                strata[stratum] = true;
            }
            assert!(
                strata.iter().all(|&hit| hit),
                "dim {dim}: not all strata covered"
            );
        }
    }

    /// Verify all LHS configs are within parameter bounds.
    #[test]
    fn latin_hypercube_in_bounds() {
        let space = ConfigSpace::new()
            .push(HyperParam::Float {
                name: "lr",
                low: 0.01,
                high: 0.5,
                log_scale: true,
            })
            .push(HyperParam::Int {
                name: "k",
                low: 5,
                high: 50,
            })
            .push(HyperParam::Categorical {
                name: "opt",
                n_choices: 4,
            });

        let n = 20;
        let mut sampler = ConfigSampler::new(space, 1337);
        let configs = sampler.latin_hypercube(n);

        assert_eq!(configs.len(), n, "expected {n} LHS configs");
        for (i, cfg) in configs.iter().enumerate() {
            let lr = cfg.get(0);
            assert!(
                (0.01..=0.5).contains(&lr),
                "LHS {i}: lr={lr} out of [0.01, 0.5]"
            );

            let k = cfg.get(1);
            assert!((5.0..=50.0).contains(&k), "LHS {i}: k={k} out of [5, 50]");
            assert_eq!(k, k.floor(), "LHS {i}: k={k} should be integer");

            let opt = cfg.get(2);
            assert!(
                (0.0..=3.0).contains(&opt),
                "LHS {i}: opt={opt} out of [0, 3]"
            );
            assert_eq!(
                opt,
                opt.floor(),
                "LHS {i}: opt={opt} should be integer index"
            );
        }
    }

    /// Verify get(), len(), is_empty() on HyperConfig.
    #[test]
    fn hyper_config_access() {
        let cfg = HyperConfig::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(cfg.len(), 3, "expected 3 values");
        assert!(!cfg.is_empty(), "config with 3 values should not be empty");
        assert_eq!(cfg.get(0), 1.0, "expected first value to be 1.0");
        assert_eq!(cfg.get(1), 2.0, "expected second value to be 2.0");
        assert_eq!(cfg.get(2), 3.0, "expected third value to be 3.0");
        assert_eq!(cfg.values(), &[1.0, 2.0, 3.0], "values slice mismatch");

        let empty = HyperConfig::new(vec![]);
        assert!(empty.is_empty(), "empty config should be empty");
        assert_eq!(empty.len(), 0, "empty config len should be 0");
    }

    /// Verify categorical values are valid integer indices.
    #[test]
    fn categorical_sampling() {
        let space = ConfigSpace::new().push(HyperParam::Categorical {
            name: "color",
            n_choices: 7,
        });

        let mut sampler = ConfigSampler::new(space, 55);
        for i in 0..200 {
            let cfg = sampler.random();
            let v = cfg.get(0);
            assert!(
                (0.0..=6.0).contains(&v),
                "sample {i}: categorical={v} out of [0, 6]"
            );
            assert_eq!(
                v,
                v.floor(),
                "sample {i}: categorical={v} should be integer index"
            );
        }
    }

    /// Verify ConfigSpace::new() has 0 params.
    #[test]
    fn empty_config_space() {
        let space = ConfigSpace::new();
        assert_eq!(space.n_params(), 0, "new space should have 0 params");
        assert_eq!(space.dim(), 0, "new space dim should be 0");
        assert!(
            space.params().is_empty(),
            "new space params should be empty"
        );
    }

    /// Same seed produces identical configurations.
    #[test]
    fn deterministic_with_seed() {
        let make_space = || {
            ConfigSpace::new()
                .push(HyperParam::Float {
                    name: "lr",
                    low: 0.001,
                    high: 1.0,
                    log_scale: true,
                })
                .push(HyperParam::Int {
                    name: "n",
                    low: 1,
                    high: 100,
                })
                .push(HyperParam::Categorical {
                    name: "opt",
                    n_choices: 4,
                })
        };

        let seed = 123456;

        let mut s1 = ConfigSampler::new(make_space(), seed);
        let mut s2 = ConfigSampler::new(make_space(), seed);

        // Random samples should be identical.
        for i in 0..50 {
            let c1 = s1.random();
            let c2 = s2.random();
            assert_eq!(
                c1.values(),
                c2.values(),
                "random sample {i} differs between same-seed samplers"
            );
        }

        // LHS batches should also be identical.
        let mut s3 = ConfigSampler::new(make_space(), seed);
        let mut s4 = ConfigSampler::new(make_space(), seed);
        let lhs3 = s3.latin_hypercube(15);
        let lhs4 = s4.latin_hypercube(15);
        for (i, (c3, c4)) in lhs3.iter().zip(lhs4.iter()).enumerate() {
            assert_eq!(
                c3.values(),
                c4.values(),
                "LHS config {i} differs between same-seed samplers"
            );
        }
    }
}
