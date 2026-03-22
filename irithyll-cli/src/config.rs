use serde::{Deserialize, Serialize};

/// Top-level TOML configuration file structure.
#[derive(Debug, Serialize, Deserialize)]
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

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            data: DataConfig::default(),
            training: TrainingConfig::default(),
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
        Ok(irithyll::SGBTConfig::builder()
            .n_steps(self.model.n_steps)
            .learning_rate(self.model.learning_rate)
            .max_depth(self.model.max_depth)
            .n_bins(self.model.n_bins)
            .grace_period(self.model.grace_period)
            .lambda(self.model.lambda)
            .gamma(self.model.gamma)
            .delta(self.model.delta)
            .seed(self.model.seed))
    }
}
