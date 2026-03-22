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
        // Generate default config
        let config = CliConfig::default();
        let toml_str = toml::to_string_pretty(&config)?;
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
