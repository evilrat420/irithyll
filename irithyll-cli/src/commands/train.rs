use clap::Args;
use color_eyre::eyre::eyre;
use color_eyre::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::time::Instant;

use irithyll::loss::LossType;
use irithyll::serde_support::to_json_pretty;
use irithyll::{DynSGBT, Loss, Sample};

use crate::config::CliConfig;
use crate::data::Dataset;

#[cfg(feature = "tui")]
use std::sync::{Arc, Mutex};

#[derive(Args)]
pub struct TrainArgs {
    /// Path to training data (CSV or Parquet)
    pub data: String,

    /// Path to config file (TOML)
    #[arg(short, long)]
    pub config: Option<String>,

    /// Target column name (default: last column)
    #[arg(short, long)]
    pub target: Option<String>,

    /// Output model path
    #[arg(short, long, default_value = "model.json")]
    pub output: String,

    /// Number of boosting steps
    #[arg(long)]
    pub n_steps: Option<usize>,

    /// Learning rate
    #[arg(long)]
    pub learning_rate: Option<f64>,

    /// Max tree depth
    #[arg(long)]
    pub max_depth: Option<usize>,

    /// Launch TUI dashboard
    #[arg(long)]
    #[cfg(feature = "tui")]
    pub tui: bool,
}

pub fn run(args: TrainArgs) -> Result<()> {
    // 1. Load config from TOML file if provided, otherwise use defaults
    let mut cli_config = if let Some(ref path) = args.config {
        CliConfig::from_file(path)?
    } else {
        CliConfig::default()
    };

    // 2. Apply CLI overrides
    if let Some(n) = args.n_steps {
        cli_config.model.n_steps = n;
    }
    if let Some(lr) = args.learning_rate {
        cli_config.model.learning_rate = lr;
    }
    if let Some(d) = args.max_depth {
        cli_config.model.max_depth = d;
    }

    // 3. Parse loss type from config string
    let loss_type = parse_loss_type(&cli_config.model.loss)?;

    // 4. Load dataset first so we can pass feature names to the config
    let dataset = Dataset::from_csv(Path::new(&args.data), args.target.as_deref())?;

    // 5. Build SGBTConfig with feature names from dataset
    let sgbt_config = cli_config
        .to_sgbt_config_builder()?
        .feature_names(dataset.feature_names.clone())
        .build()?;

    // 6. Create model
    let mut model = DynSGBT::with_loss(sgbt_config, loss_type.clone().into_loss());

    // 7. Branch: TUI or headless
    #[cfg(feature = "tui")]
    if args.tui {
        return run_with_tui(model, loss_type, dataset, &args.output);
    }

    run_headless(&mut model, &loss_type, &dataset, &args.output)
}

fn run_headless(
    model: &mut DynSGBT,
    loss_type: &LossType,
    dataset: &Dataset,
    output_path: &str,
) -> Result<()> {
    println!(
        "Loaded {} samples, {} features",
        dataset.n_samples, dataset.n_features
    );

    let pb = ProgressBar::new(dataset.n_samples as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({per_sec})")
            .unwrap()
            .progress_chars("=> "),
    );

    let start = Instant::now();
    for i in 0..dataset.n_samples {
        let sample = Sample::new(dataset.features[i].clone(), dataset.targets[i]);
        model.train_one(&sample);
        pb.inc(1);
    }
    pb.finish_with_message("done");
    let elapsed = start.elapsed();

    // Save model
    let state = model.to_model_state_with(loss_type.clone());
    let json = to_json_pretty(&state)?;
    std::fs::write(output_path, &json)?;

    println!();
    println!("Training complete");
    println!("  Samples:  {}", dataset.n_samples);
    println!("  Steps:    {}", model.n_steps());
    println!("  Leaves:   {}", model.total_leaves());
    println!("  Time:     {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Speed:    {:.0} samples/sec",
        dataset.n_samples as f64 / elapsed.as_secs_f64()
    );
    println!("  Saved to: {}", output_path);

    Ok(())
}

#[cfg(feature = "tui")]
fn run_with_tui(
    mut model: DynSGBT,
    loss_type: LossType,
    dataset: Dataset,
    output_path: &str,
) -> Result<()> {
    use crate::tui::{AppState, SharedState};

    let state: SharedState = Arc::new(Mutex::new(AppState::new(dataset.n_samples as u64)));
    let tui_state = state.clone();
    let output = output_path.to_string();

    // Build a tokio runtime for the TUI async event loop
    let rt = tokio::runtime::Runtime::new()?;

    rt.block_on(async {
        // Spawn training on a background thread
        let train_state = state.clone();
        let train_handle = tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            let update_interval = (dataset.n_samples / 200).max(1); // ~200 UI updates

            for i in 0..dataset.n_samples {
                let sample = Sample::new(dataset.features[i].clone(), dataset.targets[i]);

                // Compute loss before training for the loss chart
                let pred = model.predict(&dataset.features[i]);
                let loss_val = model.loss().loss(dataset.targets[i], pred);

                model.train_one(&sample);

                // Update shared state periodically
                if i % update_interval == 0 || i == dataset.n_samples - 1 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let throughput = if elapsed > 0.0 {
                        (i + 1) as f64 / elapsed
                    } else {
                        0.0
                    };

                    let mut s = train_state.lock().unwrap();
                    s.n_samples = (i + 1) as u64;
                    s.elapsed_secs = elapsed;
                    s.throughput = throughput;
                    s.loss_history.push(loss_val);
                    s.status_message = format!("Training... {:.0} samples/sec", throughput);
                }
            }

            let elapsed = start.elapsed();

            // Mark done and update final metrics
            {
                let mut s = train_state.lock().unwrap();
                s.is_training = false;
                s.is_done = true;
                s.n_samples = dataset.n_samples as u64;
                s.elapsed_secs = elapsed.as_secs_f64();
                s.throughput = dataset.n_samples as f64 / elapsed.as_secs_f64();
                s.metrics = vec![
                    ("Samples".to_string(), dataset.n_samples as f64),
                    ("Steps".to_string(), model.n_steps() as f64),
                    ("Leaves".to_string(), model.total_leaves() as f64),
                    (
                        "Throughput".to_string(),
                        dataset.n_samples as f64 / elapsed.as_secs_f64(),
                    ),
                    ("Time (s)".to_string(), elapsed.as_secs_f64()),
                ];
            }

            // Save model
            let model_state = model.to_model_state_with(loss_type);
            let json = to_json_pretty(&model_state).unwrap();
            std::fs::write(&output, &json).unwrap();

            Ok::<(), color_eyre::Report>(())
        });

        // Run TUI on the main async task
        let tui_result = crate::tui::run_tui(tui_state).await;

        // Wait for training to finish (it may already be done)
        let _ = train_handle.await?;

        tui_result
    })
}

pub fn parse_loss_type(s: &str) -> Result<LossType> {
    match s.to_lowercase().as_str() {
        "squared" => Ok(LossType::Squared),
        "logistic" => Ok(LossType::Logistic),
        "huber" => Ok(LossType::Huber { delta: 1.0 }),
        _ => Err(eyre!(
            "unknown loss type '{}'. supported: squared, logistic, huber",
            s
        )),
    }
}
