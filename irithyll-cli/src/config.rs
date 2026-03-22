use serde::{Deserialize, Serialize};

/// Top-level TOML configuration file structure.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CliConfig {
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub data: DataConfig,
    #[serde(default)]
    pub training: TrainingConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_n_steps")]
    pub n_steps: usize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_max_depth")]
    pub max_depth: usize,
    #[serde(default = "default_n_bins")]
    pub n_bins: usize,
    #[serde(default = "default_grace_period")]
    pub grace_period: usize,
    #[serde(default = "default_lambda")]
    pub lambda: f64,
    #[serde(default)]
    pub gamma: f64,
    #[serde(default = "default_delta")]
    pub delta: f64,
    #[serde(default = "default_loss")]
    pub loss: String,
    #[serde(default)]
    pub seed: u64,

    // --- Advanced fields (v8.2+) ---
    /// Fraction of features to subsample per tree. Default 0.75.
    #[serde(default = "default_feature_subsample_rate")]
    pub feature_subsample_rate: f64,
    /// Per-leaf gradient clipping threshold (sigma multiplier). None = disabled.
    #[serde(default)]
    pub gradient_clip_sigma: Option<f64>,
    /// Maximum absolute leaf output value. None = no clamping.
    #[serde(default)]
    pub max_leaf_output: Option<f64>,
    /// Per-leaf adaptive output bound (sigma multiplier). None = disabled.
    #[serde(default)]
    pub adaptive_leaf_bound: Option<f64>,
    /// Minimum hessian sum before a leaf produces non-zero output. None = disabled.
    #[serde(default)]
    pub min_hessian_sum: Option<f64>,
    /// Interval (samples per leaf) at which max-depth leaves re-evaluate splits. None = disabled.
    #[serde(default)]
    pub split_reeval_interval: Option<usize>,
    /// Maximum samples a single tree processes before proactive replacement. None = disabled.
    #[serde(default)]
    pub max_tree_samples: Option<u64>,
    /// Half-life for exponential leaf decay (in samples per leaf). None = disabled.
    #[serde(default)]
    pub leaf_half_life: Option<usize>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DataConfig {
    pub path: Option<String>,
    pub target: Option<String>,
    #[serde(default)]
    pub delimiter: Option<char>,
    #[serde(default)]
    pub header: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub output: Option<String>,
    #[serde(default = "default_format")]
    pub format: String,
}

// Defaults
fn default_n_steps() -> usize {
    100
}
fn default_learning_rate() -> f64 {
    0.0125
}
fn default_max_depth() -> usize {
    6
}
fn default_n_bins() -> usize {
    64
}
fn default_grace_period() -> usize {
    200
}
fn default_lambda() -> f64 {
    1.0
}
fn default_delta() -> f64 {
    1e-7
}
fn default_loss() -> String {
    "squared".to_string()
}
fn default_feature_subsample_rate() -> f64 {
    0.75
}
fn default_format() -> String {
    "json".to_string()
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_steps: default_n_steps(),
            learning_rate: default_learning_rate(),
            max_depth: default_max_depth(),
            n_bins: default_n_bins(),
            grace_period: default_grace_period(),
            lambda: default_lambda(),
            gamma: 0.0,
            delta: default_delta(),
            loss: default_loss(),
            seed: 0,
            feature_subsample_rate: default_feature_subsample_rate(),
            gradient_clip_sigma: None,
            max_leaf_output: None,
            adaptive_leaf_bound: None,
            min_hessian_sum: None,
            split_reeval_interval: None,
            max_tree_samples: None,
            leaf_half_life: None,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            output: None,
            format: default_format(),
        }
    }
}

impl CliConfig {
    pub fn from_file(path: &str) -> color_eyre::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }

    pub fn to_sgbt_config(&self) -> color_eyre::Result<irithyll::SGBTConfig> {
        Ok(self.to_sgbt_config_builder()?.build()?)
    }

    pub fn to_sgbt_config_builder(
        &self,
    ) -> color_eyre::Result<irithyll::ensemble::config::SGBTConfigBuilder> {
        let mut builder = irithyll::SGBTConfig::builder()
            .n_steps(self.model.n_steps)
            .learning_rate(self.model.learning_rate)
            .max_depth(self.model.max_depth)
            .n_bins(self.model.n_bins)
            .grace_period(self.model.grace_period)
            .lambda(self.model.lambda)
            .gamma(self.model.gamma)
            .delta(self.model.delta)
            .seed(self.model.seed)
            .feature_subsample_rate(self.model.feature_subsample_rate);

        // Pass through advanced optional fields
        if let Some(v) = self.model.gradient_clip_sigma {
            builder = builder.gradient_clip_sigma(v);
        }
        if let Some(v) = self.model.max_leaf_output {
            builder = builder.max_leaf_output(v);
        }
        if let Some(v) = self.model.adaptive_leaf_bound {
            builder = builder.adaptive_leaf_bound(v);
        }
        if let Some(v) = self.model.min_hessian_sum {
            builder = builder.min_hessian_sum(v);
        }
        if let Some(v) = self.model.split_reeval_interval {
            builder = builder.split_reeval_interval(v);
        }
        if let Some(v) = self.model.max_tree_samples {
            builder = builder.max_tree_samples(v);
        }
        if let Some(v) = self.model.leaf_half_life {
            builder = builder.leaf_half_life(v);
        }

        Ok(builder)
    }
}
