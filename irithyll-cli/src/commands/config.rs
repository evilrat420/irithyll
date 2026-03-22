use clap::Args;
use color_eyre::eyre::eyre;
use color_eyre::Result;

use crate::config::CliConfig;

#[derive(Args)]
pub struct ConfigArgs {
    /// Generate a default config file
    #[arg(long)]
    pub generate: bool,

    /// Validate an existing config file
    #[arg(long)]
    pub validate: Option<String>,

    /// Output path for generated config
    #[arg(short, long, default_value = "irithyll.toml")]
    pub output: String,
}

pub fn run(args: ConfigArgs) -> Result<()> {
    if args.generate {
        // Generate default config with commented advanced options
        let config = CliConfig::default();
        let mut toml_str = toml::to_string_pretty(&config)?;

        // Append commented-out advanced options as documentation
        toml_str.push_str("\n# --- Advanced options (uncomment to enable) ---\n");
        toml_str.push_str("# gradient_clip_sigma = 3.0\n");
        toml_str.push_str("# max_leaf_output = 3.0\n");
        toml_str.push_str("# adaptive_leaf_bound = 3.0\n");
        toml_str.push_str("# min_hessian_sum = 10.0\n");
        toml_str.push_str("# split_reeval_interval = 500\n");
        toml_str.push_str("# max_tree_samples = 50000\n");
        toml_str.push_str("# leaf_half_life = 1000\n");
        toml_str.push_str("#\n");
        toml_str.push_str("# --- Loss types ---\n");
        toml_str.push_str("# loss = \"squared\"       # regression (default)\n");
        toml_str.push_str("# loss = \"logistic\"      # binary classification\n");
        toml_str.push_str("# loss = \"huber:1.0\"     # robust regression (delta=1.0)\n");
        toml_str.push_str("# loss = \"softmax:3\"     # multiclass (3 classes)\n");
        toml_str.push_str("# loss = \"quantile:0.5\"  # median regression (tau=0.5)\n");
        toml_str.push_str("# loss = \"expectile:0.9\" # expectile regression (tau=0.9)\n");

        std::fs::write(&args.output, &toml_str)?;
        println!("Generated default config at {}", args.output);
        return Ok(());
    }

    if let Some(ref path) = args.validate {
        // Validate config
        match CliConfig::from_file(path) {
            Ok(config) => {
                // Also validate that it converts to a valid SGBTConfig
                match config.to_sgbt_config() {
                    Ok(_) => {
                        println!("[OK] Config at {} is valid", path);
                        println!();
                        println!("  Model:");
                        println!("    n_steps:       {}", config.model.n_steps);
                        println!("    learning_rate: {}", config.model.learning_rate);
                        println!("    max_depth:     {}", config.model.max_depth);
                        println!("    n_bins:        {}", config.model.n_bins);
                        println!("    grace_period:  {}", config.model.grace_period);
                        println!("    lambda:        {}", config.model.lambda);
                        println!("    gamma:         {}", config.model.gamma);
                        println!("    delta:         {}", config.model.delta);
                        println!("    loss:          {}", config.model.loss);
                        println!("    seed:          {}", config.model.seed);
                        println!(
                            "    feature_subsample_rate: {}",
                            config.model.feature_subsample_rate
                        );
                        if let Some(v) = config.model.gradient_clip_sigma {
                            println!("    gradient_clip_sigma: {}", v);
                        }
                        if let Some(v) = config.model.max_leaf_output {
                            println!("    max_leaf_output: {}", v);
                        }
                        if let Some(v) = config.model.adaptive_leaf_bound {
                            println!("    adaptive_leaf_bound: {}", v);
                        }
                        if let Some(v) = config.model.min_hessian_sum {
                            println!("    min_hessian_sum: {}", v);
                        }
                        if let Some(v) = config.model.split_reeval_interval {
                            println!("    split_reeval_interval: {}", v);
                        }
                        if let Some(v) = config.model.max_tree_samples {
                            println!("    max_tree_samples: {}", v);
                        }
                        if let Some(v) = config.model.leaf_half_life {
                            println!("    leaf_half_life: {}", v);
                        }
                    }
                    Err(e) => {
                        println!("[ERROR] Config at {} parses but has invalid values:", path);
                        println!("  {}", e);
                        return Err(eyre!("config validation failed: {}", e));
                    }
                }
            }
            Err(e) => {
                println!("[ERROR] Failed to parse config at {}:", path);
                println!("  {}", e);
                return Err(eyre!("config parse failed: {}", e));
            }
        }
        return Ok(());
    }

    // Neither --generate nor --validate provided
    Err(eyre!(
        "specify --generate to create a config file or --validate <path> to check one"
    ))
}
