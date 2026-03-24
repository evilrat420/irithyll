use clap::Args;
use color_eyre::eyre::eyre;
use color_eyre::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::time::Instant;

use irithyll::loss::LossType;
use irithyll::serde_support::to_json_pretty;
#[cfg(feature = "tui")]
use irithyll::Loss;
use irithyll::{DynSGBT, Sample, StreamingLearner};

use crate::config::CliConfig;
use crate::data::Dataset;

#[cfg(feature = "tui")]
use std::sync::{Arc, Mutex};

/// Model type selection for the CLI.
#[derive(Debug, Clone, Default)]
pub enum ModelType {
    /// Standard DynSGBT (default).
    #[default]
    Sgbt,
    /// DistributionalSGBT -- outputs Gaussian N(mu, sigma^2).
    Distributional,
    /// MulticlassSGBT -- one-vs-rest committee.
    Multiclass,
    /// BaggedSGBT -- Oza online bagging for variance reduction.
    Bagged,
    /// Next Generation Reservoir Computer (NG-RC).
    Ngrc,
    /// Echo State Network (ESN).
    Esn,
    /// Streaming Mamba (selective SSM).
    Mamba,
    /// Spiking Neural Network with e-prop learning.
    SpikeNet,
    /// Gated Linear Attention (SOTA streaming attention).
    Gla,
    /// Gated DeltaNet (strongest retrieval, NVIDIA 2024).
    DeltaNet,
    /// Hawk (lightest streaming attention, vector state).
    Hawk,
    /// Retentive Network (simplest, fixed decay).
    RetNet,
}

impl ModelType {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "sgbt" => Ok(ModelType::Sgbt),
            "distributional" => Ok(ModelType::Distributional),
            "multiclass" => Ok(ModelType::Multiclass),
            "bagged" => Ok(ModelType::Bagged),
            "ngrc" => Ok(ModelType::Ngrc),
            "esn" => Ok(ModelType::Esn),
            "mamba" => Ok(ModelType::Mamba),
            "spikenet" => Ok(ModelType::SpikeNet),
            "gla" => Ok(ModelType::Gla),
            "deltanet" => Ok(ModelType::DeltaNet),
            "hawk" => Ok(ModelType::Hawk),
            "retnet" => Ok(ModelType::RetNet),
            _ => Err(eyre!(
                "unknown model type '{}'. supported: sgbt, distributional, multiclass, bagged, ngrc, esn, mamba, spikenet, gla, deltanet, hawk, retnet",
                s
            )),
        }
    }
}

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

    /// Model type: sgbt, distributional, multiclass, bagged, ngrc, esn, mamba, spikenet, gla, deltanet, hawk, retnet
    #[arg(long, default_value = "sgbt")]
    pub model_type: String,

    /// Number of classes (required for softmax loss and multiclass model type)
    #[arg(long)]
    pub n_classes: Option<usize>,

    /// Number of bags for bagged model type (default: 10)
    #[arg(long, default_value = "10")]
    pub n_bags: usize,

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

    // 3. Parse loss type and model type
    let loss_type = parse_loss_type(&cli_config.model.loss, args.n_classes)?;
    let model_type = ModelType::from_str(&args.model_type)?;

    // 4. Load dataset
    let dataset = Dataset::from_csv(Path::new(&args.data), args.target.as_deref())?;

    // 5. Branch on model type
    match model_type {
        ModelType::Sgbt => run_sgbt(args, cli_config, loss_type, dataset),
        ModelType::Distributional => run_distributional(args, cli_config, dataset),
        ModelType::Multiclass => run_multiclass(args, cli_config, dataset),
        ModelType::Bagged => run_bagged(args, cli_config, loss_type, dataset),
        ModelType::Ngrc => run_ngrc(cli_config, dataset),
        ModelType::Esn => run_esn(cli_config, dataset),
        ModelType::Mamba => run_mamba(cli_config, dataset),
        ModelType::SpikeNet => run_spikenet(cli_config, dataset),
        ModelType::Gla => run_gla(&dataset, &cli_config),
        ModelType::DeltaNet => run_deltanet(&dataset, &cli_config),
        ModelType::Hawk => run_hawk(&dataset, &cli_config),
        ModelType::RetNet => run_retnet(&dataset, &cli_config),
    }
}

// ---------------------------------------------------------------------------
// SGBT (default path -- unchanged except for TUI wiring)
// ---------------------------------------------------------------------------

fn run_sgbt(
    args: TrainArgs,
    cli_config: CliConfig,
    loss_type: LossType,
    dataset: Dataset,
) -> Result<()> {
    let sgbt_config = cli_config
        .to_sgbt_config_builder()?
        .feature_names(dataset.feature_names.clone())
        .build()?;

    let mut model = DynSGBT::with_loss(sgbt_config, loss_type.clone().into_loss());

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

            // Mark done and update final metrics + feature importances
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
                s.feature_importances = model
                    .feature_importances()
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (format!("f{}", i), v))
                    .filter(|(_, v)| *v > 0.0)
                    .collect();
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

// ---------------------------------------------------------------------------
// Distributional SGBT
// ---------------------------------------------------------------------------

fn run_distributional(_args: TrainArgs, cli_config: CliConfig, dataset: Dataset) -> Result<()> {
    use irithyll::ensemble::distributional::DistributionalSGBT;

    let sgbt_config = cli_config
        .to_sgbt_config_builder()?
        .feature_names(dataset.feature_names.clone())
        .build()?;

    let mut model = DistributionalSGBT::new(sgbt_config);

    println!(
        "Loaded {} samples, {} features (distributional mode)",
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

    // Print last prediction as summary
    if dataset.n_samples > 0 {
        let last = &dataset.features[dataset.n_samples - 1];
        let pred = model.predict(last);
        println!();
        println!("Training complete (distributional)");
        println!("  Samples:  {}", dataset.n_samples);
        println!("  Steps:    {}", model.n_steps());
        println!("  Leaves:   {}", model.total_leaves());
        println!("  Time:     {:.2}s", elapsed.as_secs_f64());
        println!(
            "  Speed:    {:.0} samples/sec",
            dataset.n_samples as f64 / elapsed.as_secs_f64()
        );
        println!("  Last pred: mu={:.4}, sigma={:.4}", pred.mu, pred.sigma);
    }

    // NOTE: JSON serialization not supported for DistributionalSGBT.
    // Model is trained but not saved to disk.
    println!("  [NOTE] Distributional model save not yet supported -- train-only mode");

    Ok(())
}

// ---------------------------------------------------------------------------
// Multiclass SGBT
// ---------------------------------------------------------------------------

fn run_multiclass(args: TrainArgs, cli_config: CliConfig, dataset: Dataset) -> Result<()> {
    use irithyll::ensemble::multiclass::MulticlassSGBT;

    let n_classes = args
        .n_classes
        .ok_or_else(|| eyre!("--n-classes is required for multiclass model type"))?;

    let sgbt_config = cli_config
        .to_sgbt_config_builder()?
        .feature_names(dataset.feature_names.clone())
        .build()?;

    let mut model = MulticlassSGBT::new(sgbt_config, n_classes)
        .map_err(|e| eyre!("failed to create multiclass model: {}", e))?;

    println!(
        "Loaded {} samples, {} features (multiclass, {} classes)",
        dataset.n_samples, dataset.n_features, n_classes
    );

    let pb = ProgressBar::new(dataset.n_samples as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({per_sec})")
            .unwrap()
            .progress_chars("=> "),
    );

    let mut n_correct: u64 = 0;

    let start = Instant::now();
    for i in 0..dataset.n_samples {
        // Test-then-train for accuracy tracking
        let pred_class = model.predict(&dataset.features[i]);
        let target_class = dataset.targets[i] as usize;
        if pred_class == target_class {
            n_correct += 1;
        }

        let sample = Sample::new(dataset.features[i].clone(), dataset.targets[i]);
        model.train_one(&sample);
        pb.inc(1);
    }
    pb.finish_with_message("done");
    let elapsed = start.elapsed();

    let accuracy = if dataset.n_samples > 0 {
        n_correct as f64 / dataset.n_samples as f64
    } else {
        0.0
    };

    println!();
    println!("Training complete (multiclass)");
    println!("  Samples:  {}", dataset.n_samples);
    println!("  Classes:  {}", n_classes);
    println!(
        "  Accuracy: {:.4} ({}/{})",
        accuracy, n_correct, dataset.n_samples
    );
    println!("  Time:     {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Speed:    {:.0} samples/sec",
        dataset.n_samples as f64 / elapsed.as_secs_f64()
    );

    // NOTE: JSON serialization not supported for MulticlassSGBT.
    println!("  [NOTE] Multiclass model save not yet supported -- train-only mode");

    Ok(())
}

// ---------------------------------------------------------------------------
// Bagged SGBT
// ---------------------------------------------------------------------------

fn run_bagged(
    args: TrainArgs,
    cli_config: CliConfig,
    loss_type: LossType,
    dataset: Dataset,
) -> Result<()> {
    use irithyll::ensemble::bagged::BaggedSGBT;
    use irithyll::loss::squared::SquaredLoss;

    let n_bags = args.n_bags;

    let sgbt_config = cli_config
        .to_sgbt_config_builder()?
        .feature_names(dataset.feature_names.clone())
        .build()?;

    // BaggedSGBT::new only supports SquaredLoss.
    // For other losses, BaggedSGBT::with_loss requires Clone on the loss,
    // which Box<dyn Loss> does not satisfy. Only squared loss for now.
    match loss_type {
        LossType::Squared => {}
        _ => {
            return Err(eyre!(
                "bagged model currently only supports squared loss (got '{:?}')",
                loss_type
            ));
        }
    }

    let mut model = BaggedSGBT::<SquaredLoss>::new(sgbt_config, n_bags)
        .map_err(|e| eyre!("failed to create bagged model: {}", e))?;

    println!(
        "Loaded {} samples, {} features (bagged, {} bags)",
        dataset.n_samples, dataset.n_features, n_bags
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

    println!();
    println!("Training complete (bagged)");
    println!("  Samples:  {}", dataset.n_samples);
    println!("  Bags:     {}", n_bags);
    println!("  Time:     {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Speed:    {:.0} samples/sec",
        dataset.n_samples as f64 / elapsed.as_secs_f64()
    );

    // NOTE: JSON serialization not supported for BaggedSGBT.
    println!("  [NOTE] Bagged model save not yet supported -- train-only mode");

    Ok(())
}

// ---------------------------------------------------------------------------
// NG-RC (Next Generation Reservoir Computer)
// ---------------------------------------------------------------------------

fn run_ngrc(cli_config: CliConfig, dataset: Dataset) -> Result<()> {
    use irithyll::reservoir::{NGRCConfig, NextGenRC};

    let nc = &cli_config.neural.ngrc;
    let config = NGRCConfig::builder()
        .k(nc.k)
        .s(nc.s)
        .degree(nc.degree)
        .forgetting_factor(nc.forgetting_factor)
        .build()
        .map_err(|e| eyre!("invalid NGRC config: {}", e))?;

    let mut model = NextGenRC::new(config);

    println!(
        "Loaded {} samples, {} features (ngrc, k={}, s={}, degree={})",
        dataset.n_samples, dataset.n_features, nc.k, nc.s, nc.degree,
    );

    run_neural_headless(&mut model, &dataset, "ngrc")
}

// ---------------------------------------------------------------------------
// ESN (Echo State Network)
// ---------------------------------------------------------------------------

fn run_esn(cli_config: CliConfig, dataset: Dataset) -> Result<()> {
    use irithyll::reservoir::{ESNConfig, EchoStateNetwork};

    let ec = &cli_config.neural.esn;
    let config = ESNConfig::builder()
        .n_reservoir(ec.n_reservoir)
        .spectral_radius(ec.spectral_radius)
        .leak_rate(ec.leak_rate)
        .input_scaling(ec.input_scaling)
        .seed(ec.seed)
        .warmup(ec.warmup)
        .build()
        .map_err(|e| eyre!("invalid ESN config: {}", e))?;

    let mut model = EchoStateNetwork::new(config);

    println!(
        "Loaded {} samples, {} features (esn, n_reservoir={}, spectral_radius={}, leak_rate={})",
        dataset.n_samples, dataset.n_features, ec.n_reservoir, ec.spectral_radius, ec.leak_rate,
    );

    run_neural_headless(&mut model, &dataset, "esn")
}

// ---------------------------------------------------------------------------
// Streaming Mamba (selective SSM)
// ---------------------------------------------------------------------------

fn run_mamba(cli_config: CliConfig, dataset: Dataset) -> Result<()> {
    use irithyll::ssm::{MambaConfig, StreamingMamba};

    let mc = &cli_config.neural.mamba;
    let config = MambaConfig::builder()
        .d_in(dataset.n_features)
        .n_state(mc.n_state)
        .seed(mc.seed)
        .warmup(mc.warmup)
        .build()
        .map_err(|e| eyre!("invalid Mamba config: {}", e))?;

    let mut model = StreamingMamba::new(config);

    println!(
        "Loaded {} samples, {} features (mamba, d_in={}, n_state={})",
        dataset.n_samples, dataset.n_features, dataset.n_features, mc.n_state,
    );

    run_neural_headless(&mut model, &dataset, "mamba")
}

// ---------------------------------------------------------------------------
// SpikeNet (Spiking Neural Network)
// ---------------------------------------------------------------------------

fn run_spikenet(cli_config: CliConfig, dataset: Dataset) -> Result<()> {
    use irithyll::snn::{SpikeNet, SpikeNetConfig};

    let sc = &cli_config.neural.spikenet;
    let config = SpikeNetConfig::builder()
        .n_hidden(sc.n_hidden)
        .learning_rate(sc.learning_rate)
        .seed(sc.seed)
        .build()
        .map_err(|e| eyre!("invalid SpikeNet config: {}", e))?;

    let mut model = SpikeNet::new(config);

    println!(
        "Loaded {} samples, {} features (spikenet, n_hidden={}, lr={})",
        dataset.n_samples, dataset.n_features, sc.n_hidden, sc.learning_rate,
    );

    run_neural_headless(&mut model, &dataset, "spikenet")
}

// ---------------------------------------------------------------------------
// GLA (Gated Linear Attention)
// ---------------------------------------------------------------------------

fn run_gla(dataset: &Dataset, config: &CliConfig) -> Result<()> {
    use irithyll::attention::{AttentionMode, StreamingAttentionConfig, StreamingAttentionModel};

    let att = &config.neural.attention;
    let att_config = StreamingAttentionConfig::builder()
        .d_model(dataset.n_features)
        .n_heads(att.n_heads)
        .mode(AttentionMode::GLA)
        .seed(att.seed)
        .warmup(att.warmup)
        .build()
        .map_err(|e| eyre!("invalid GLA config: {}", e))?;

    let mut model = StreamingAttentionModel::new(att_config);

    println!(
        "Loaded {} samples, {} features (gla, n_heads={})",
        dataset.n_samples, dataset.n_features, att.n_heads,
    );

    run_neural_headless(&mut model, dataset, "gla")
}

// ---------------------------------------------------------------------------
// DeltaNet (Gated DeltaNet)
// ---------------------------------------------------------------------------

fn run_deltanet(dataset: &Dataset, config: &CliConfig) -> Result<()> {
    use irithyll::attention::{AttentionMode, StreamingAttentionConfig, StreamingAttentionModel};

    let att = &config.neural.attention;
    let att_config = StreamingAttentionConfig::builder()
        .d_model(dataset.n_features)
        .n_heads(att.n_heads)
        .mode(AttentionMode::GatedDeltaNet)
        .seed(att.seed)
        .warmup(att.warmup)
        .build()
        .map_err(|e| eyre!("invalid DeltaNet config: {}", e))?;

    let mut model = StreamingAttentionModel::new(att_config);

    println!(
        "Loaded {} samples, {} features (deltanet, n_heads={})",
        dataset.n_samples, dataset.n_features, att.n_heads,
    );

    run_neural_headless(&mut model, dataset, "deltanet")
}

// ---------------------------------------------------------------------------
// Hawk (lightest attention, vector state)
// ---------------------------------------------------------------------------

fn run_hawk(dataset: &Dataset, config: &CliConfig) -> Result<()> {
    use irithyll::attention::{AttentionMode, StreamingAttentionConfig, StreamingAttentionModel};

    let att = &config.neural.attention;
    let att_config = StreamingAttentionConfig::builder()
        .d_model(dataset.n_features)
        .n_heads(1) // Hawk always uses 1 head (vector state)
        .mode(AttentionMode::Hawk)
        .seed(att.seed)
        .warmup(att.warmup)
        .build()
        .map_err(|e| eyre!("invalid Hawk config: {}", e))?;

    let mut model = StreamingAttentionModel::new(att_config);

    println!(
        "Loaded {} samples, {} features (hawk, single-head vector state)",
        dataset.n_samples, dataset.n_features,
    );

    run_neural_headless(&mut model, dataset, "hawk")
}

// ---------------------------------------------------------------------------
// RetNet (Retentive Network, fixed decay)
// ---------------------------------------------------------------------------

fn run_retnet(dataset: &Dataset, config: &CliConfig) -> Result<()> {
    use irithyll::attention::{AttentionMode, StreamingAttentionConfig, StreamingAttentionModel};

    let att = &config.neural.attention;
    let att_config = StreamingAttentionConfig::builder()
        .d_model(dataset.n_features)
        .n_heads(1) // RetNet uses 1 head with fixed gamma decay
        .mode(AttentionMode::RetNet { gamma: att.gamma })
        .seed(att.seed)
        .warmup(att.warmup)
        .build()
        .map_err(|e| eyre!("invalid RetNet config: {}", e))?;

    let mut model = StreamingAttentionModel::new(att_config);

    println!(
        "Loaded {} samples, {} features (retnet, gamma={})",
        dataset.n_samples, dataset.n_features, att.gamma,
    );

    run_neural_headless(&mut model, dataset, "retnet")
}

// ---------------------------------------------------------------------------
// Shared headless training loop for all neural models (prequential)
// ---------------------------------------------------------------------------

fn run_neural_headless(
    model: &mut dyn StreamingLearner,
    dataset: &Dataset,
    model_name: &str,
) -> Result<()> {
    use irithyll::metrics::RegressionMetrics;

    let pb = ProgressBar::new(dataset.n_samples as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({per_sec})")
            .unwrap()
            .progress_chars("=> "),
    );

    let mut metrics = RegressionMetrics::new();
    let print_interval = (dataset.n_samples / 10).max(1);

    let start = Instant::now();
    for i in 0..dataset.n_samples {
        let features = &dataset.features[i];
        let target = dataset.targets[i];

        // Test-then-train (prequential evaluation)
        let pred = model.predict(features);
        metrics.update(target, pred);

        model.train(features, target);

        if (i + 1) % print_interval == 0 {
            pb.println(format!(
                "  [{}/{}] RMSE={:.6}  MAE={:.6}  R2={:.6}",
                i + 1,
                dataset.n_samples,
                metrics.rmse(),
                metrics.mae(),
                metrics.r_squared(),
            ));
        }

        pb.inc(1);
    }
    pb.finish_with_message("done");
    let elapsed = start.elapsed();

    println!();
    println!("Training complete ({})", model_name);
    println!("  Samples:  {}", dataset.n_samples);
    println!("  RMSE:     {:.6}", metrics.rmse());
    println!("  MAE:      {:.6}", metrics.mae());
    println!("  R2:       {:.6}", metrics.r_squared());
    println!("  Time:     {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Speed:    {:.0} samples/sec",
        dataset.n_samples as f64 / elapsed.as_secs_f64()
    );
    println!("  [NOTE] Neural model save not yet supported -- train-only mode");

    Ok(())
}

// ---------------------------------------------------------------------------
// Loss type parsing
// ---------------------------------------------------------------------------

/// Parse a loss type string from config.
///
/// Supports:
/// - "squared"
/// - "logistic"
/// - "huber" or "huber:1.5" (custom delta)
/// - "softmax:3" (n_classes required)
/// - "quantile:0.5" (tau required)
/// - "expectile:0.9" (tau required)
pub fn parse_loss_type(s: &str, n_classes_override: Option<usize>) -> Result<LossType> {
    let lower = s.to_lowercase();
    let parts: Vec<&str> = lower.splitn(2, ':').collect();
    let name = parts[0].trim();
    let param = parts.get(1).map(|p| p.trim());

    match name {
        "squared" => Ok(LossType::Squared),
        "logistic" => Ok(LossType::Logistic),
        "huber" => {
            let delta = if let Some(p) = param {
                p.parse::<f64>().map_err(|_| {
                    eyre!(
                        "invalid huber delta '{}' -- expected a float (e.g. huber:1.5)",
                        p
                    )
                })?
            } else {
                1.0
            };
            Ok(LossType::Huber { delta })
        }
        "softmax" => {
            // n_classes from param string or from --n-classes flag
            let n_classes = if let Some(p) = param {
                p.parse::<usize>().map_err(|_| {
                    eyre!(
                        "invalid softmax n_classes '{}' -- expected an integer (e.g. softmax:3)",
                        p
                    )
                })?
            } else if let Some(n) = n_classes_override {
                n
            } else {
                return Err(eyre!(
                    "softmax loss requires n_classes -- use 'softmax:3' or --n-classes 3"
                ));
            };
            if n_classes < 2 {
                return Err(eyre!("softmax n_classes must be >= 2, got {}", n_classes));
            }
            Ok(LossType::Softmax { n_classes })
        }
        "quantile" => {
            let tau = param
                .ok_or_else(|| eyre!("quantile loss requires tau -- use 'quantile:0.5'"))?
                .parse::<f64>()
                .map_err(|_| {
                    eyre!("invalid quantile tau -- expected a float (e.g. quantile:0.5)")
                })?;
            if tau <= 0.0 || tau >= 1.0 {
                return Err(eyre!("quantile tau must be in (0, 1), got {}", tau));
            }
            Ok(LossType::Quantile { tau })
        }
        "expectile" => {
            let tau = param
                .ok_or_else(|| eyre!("expectile loss requires tau -- use 'expectile:0.9'"))?
                .parse::<f64>()
                .map_err(|_| {
                    eyre!("invalid expectile tau -- expected a float (e.g. expectile:0.9)")
                })?;
            if tau <= 0.0 || tau >= 1.0 {
                return Err(eyre!("expectile tau must be in (0, 1), got {}", tau));
            }
            Ok(LossType::Expectile { tau })
        }
        _ => Err(eyre!(
            "unknown loss type '{}'. supported: squared, logistic, huber[:delta], softmax:N, quantile:tau, expectile:tau",
            s
        )),
    }
}
