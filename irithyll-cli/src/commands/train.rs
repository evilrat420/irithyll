use clap::Args;
use color_eyre::eyre::eyre;
use color_eyre::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::time::Instant;

use irithyll::loss::LossType;
use irithyll::serde_support::to_json_pretty;
use irithyll::{DynSGBT, Sample, StreamingLearner};

use crate::config::CliConfig;
use crate::data::Dataset;

/// Model type selection for the CLI.
///
/// Names accept either kebab-case or compact form (e.g. `mamba-3` or `mamba3`).
/// The `auto-tune` value runs `Factory` racing with the factories listed in
/// `--factories`.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
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
    /// Streaming Mamba-3 (MIMO groups, complex states, trapezoidal discretisation).
    Mamba3,
    /// Streaming Mamba BD-LRU (block-diagonal linear recurrence).
    MambaBd,
    /// Streaming sLSTM (exponential gating with log-domain stabilisation).
    Slstm,
    /// Streaming mGRADE (minimal recurrent gating with delay convolutions).
    Mgrade,
    /// Spiking Neural Network with e-prop learning.
    SpikeNet,
    /// Gated Linear Attention.
    Gla,
    /// Gated DeltaNet.
    DeltaNet,
    /// DeltaProduct attention (Householder delta rule composition).
    DeltaProduct,
    /// RWKV-7 attention (vector-gated delta rule, DPLR transitions).
    Rwkv7,
    /// HGRN2 (lower-bounded gated linear RNN with state expansion).
    Hgrn2,
    /// Hawk (vector state, single-head).
    Hawk,
    /// Retentive Network (fixed exponential decay).
    RetNet,
    /// Log-Linear Attention (hierarchical Fenwick state).
    LogLinear,
    /// Streaming TTT (test-time training with fast weights).
    Ttt,
    /// Streaming KAN (B-spline edge activations).
    Kan,
    /// Automated model selection via `Factory` racing (`AutoTuner`).
    Factory,
}

/// Available factory keys for `--factories` and the help text on `--model-type`.
///
/// Keep in sync with the match arms in `factory_from_name`.
pub(crate) const FACTORY_KEYS: &[&str] = &[
    "sgbt",
    "distributional",
    "multiclass-sgbt",
    "esn",
    "mamba",
    "mamba-3",
    "mamba-bd",
    "s-lstm",
    "mgrade",
    "attention",
    "delta-product",
    "rwkv-7",
    "spike-net",
    "kan",
    "ttt",
];

impl ModelType {
    /// Parse a model-type string. Accepts both kebab-case and compact aliases.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "sgbt" => Ok(ModelType::Sgbt),
            "distributional" => Ok(ModelType::Distributional),
            "multiclass" | "multiclass-sgbt" => Ok(ModelType::Multiclass),
            "bagged" => Ok(ModelType::Bagged),
            "ngrc" => Ok(ModelType::Ngrc),
            "esn" => Ok(ModelType::Esn),
            "mamba" => Ok(ModelType::Mamba),
            "mamba-3" | "mamba3" => Ok(ModelType::Mamba3),
            "mamba-bd" | "mambabd" => Ok(ModelType::MambaBd),
            "s-lstm" | "slstm" => Ok(ModelType::Slstm),
            "mgrade" => Ok(ModelType::Mgrade),
            "spike-net" | "spikenet" => Ok(ModelType::SpikeNet),
            "gla" => Ok(ModelType::Gla),
            "delta-net" | "deltanet" => Ok(ModelType::DeltaNet),
            "delta-product" | "deltaproduct" => Ok(ModelType::DeltaProduct),
            "rwkv-7" | "rwkv7" => Ok(ModelType::Rwkv7),
            "hgrn2" => Ok(ModelType::Hgrn2),
            "hawk" => Ok(ModelType::Hawk),
            "ret-net" | "retnet" => Ok(ModelType::RetNet),
            "log-linear" | "loglinear" => Ok(ModelType::LogLinear),
            "ttt" => Ok(ModelType::Ttt),
            "kan" => Ok(ModelType::Kan),
            "factory" | "auto-tune" | "autotune" | "autotuner" => Ok(ModelType::Factory),
            _ => Err(eyre!(
                "unknown model type '{}'.\n  supported: {}",
                s,
                MODEL_TYPE_KEYS.join(", "),
            )),
        }
    }
}

/// Keys accepted by `--model-type`. Kept here as a single source of truth for
/// help text and error messages.
pub(crate) const MODEL_TYPE_KEYS: &[&str] = &[
    "sgbt",
    "distributional",
    "multiclass",
    "bagged",
    "ngrc",
    "esn",
    "mamba",
    "mamba-3",
    "mamba-bd",
    "s-lstm",
    "mgrade",
    "spike-net",
    "gla",
    "delta-net",
    "delta-product",
    "rwkv-7",
    "hgrn2",
    "hawk",
    "ret-net",
    "log-linear",
    "ttt",
    "kan",
    "factory",
];

#[derive(Args)]
pub struct TrainArgs {
    /// Path to training data (CSV)
    pub data: String,

    /// Path to a TOML config file
    #[arg(short, long)]
    pub config: Option<String>,

    /// Target column name (default: last column)
    #[arg(short, long)]
    pub target: Option<String>,

    /// Output model path
    #[arg(short, long, default_value = "model.json")]
    pub output: String,

    /// Number of boosting steps (SGBT family)
    #[arg(long)]
    pub n_steps: Option<usize>,

    /// Learning rate (overrides config)
    #[arg(long)]
    pub learning_rate: Option<f64>,

    /// Max tree depth (SGBT family)
    #[arg(long)]
    pub max_depth: Option<usize>,

    /// Model type. See --help for the full list.
    #[arg(long, default_value = "sgbt", value_name = "TYPE")]
    pub model_type: String,

    /// Number of classes (required for softmax loss and multiclass model type)
    #[arg(long)]
    pub n_classes: Option<usize>,

    /// Number of bags for bagged model type
    #[arg(long, default_value = "10")]
    pub n_bags: usize,

    /// Wrap the chosen model in an AutoTuner; equivalent to --model-type factory --factories <type>
    #[arg(long)]
    pub auto_tune: bool,

    /// Comma-separated list of factories to race when --model-type=factory
    #[arg(long, default_value = "sgbt,esn,mamba", value_name = "LIST")]
    pub factories: String,

    /// Initial candidates per AutoTuner tournament
    #[arg(long, value_name = "N")]
    pub n_initial: Option<usize>,

    /// Maximum bracket size for adaptive AutoTuner tournaments
    #[arg(long, value_name = "N")]
    pub max_n_initial: Option<usize>,

    /// Enable drift-triggered re-racing in AutoTuner
    #[arg(long)]
    pub use_drift_rerace: bool,

    /// Launch the TUI dashboard
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

    // 5. If --auto-tune was passed, route to factory racing using the chosen
    //    model as the seed factory. Explicit `--model-type factory` still uses
    //    the `--factories` list.
    if args.auto_tune && !matches!(model_type, ModelType::Factory) {
        let seed_factory = factory_key_for_model(&model_type)?;
        return run_factory_with_keys(&args, dataset, &[seed_factory]);
    }

    // 5b. If --tui was passed, route to the multi-family TUI dashboard for
    //     any supported family. Each family builds a `tui::DemoModel` sized
    //     for the loaded CSV. Unsupported model types fall through with a
    //     clear error rather than silently ignoring `--tui`.
    #[cfg(feature = "tui")]
    if args.tui {
        let family = match model_type {
            ModelType::Sgbt => Some(crate::tui::ModelFamily::Sgbt),
            ModelType::Mamba => Some(crate::tui::ModelFamily::Mamba),
            ModelType::Ttt => Some(crate::tui::ModelFamily::Ttt),
            ModelType::Kan => Some(crate::tui::ModelFamily::Kan),
            ModelType::Esn => Some(crate::tui::ModelFamily::Esn),
            ModelType::Ngrc => Some(crate::tui::ModelFamily::Ngrc),
            ModelType::SpikeNet => Some(crate::tui::ModelFamily::SpikeNet),
            _ => None,
        };
        if let Some(family) = family {
            let model = crate::tui::DemoModel::build_for_dataset(family, dataset.n_features);
            let label = crate::tui::label_from_csv_path(&args.data);
            return crate::tui::run_with_dataset(model, dataset, &args.output, label);
        } else {
            return Err(eyre!(
                "--tui not yet supported for --model-type {}. Supported: sgbt, mamba, ttt, kan, esn, ngrc, spike-net",
                args.model_type
            ));
        }
    }

    // 6. Branch on model type
    match model_type {
        ModelType::Sgbt => run_sgbt(args, cli_config, loss_type, dataset),
        ModelType::Distributional => run_distributional(args, cli_config, dataset),
        ModelType::Multiclass => run_multiclass(args, cli_config, dataset),
        ModelType::Bagged => run_bagged(args, cli_config, loss_type, dataset),
        ModelType::Ngrc => run_ngrc(cli_config, dataset),
        ModelType::Esn => run_esn(cli_config, dataset),
        ModelType::Mamba => run_mamba(cli_config, dataset),
        ModelType::Mamba3 => run_mamba3(cli_config, dataset),
        ModelType::MambaBd => run_mamba_bd(cli_config, dataset),
        ModelType::Slstm => run_slstm(dataset),
        ModelType::Mgrade => run_mgrade(dataset),
        ModelType::SpikeNet => run_spikenet(cli_config, dataset),
        ModelType::Gla => run_gla(&dataset, &cli_config),
        ModelType::DeltaNet => run_deltanet(&dataset, &cli_config),
        ModelType::DeltaProduct => run_delta_product(&dataset, &cli_config),
        ModelType::Rwkv7 => run_rwkv7(&dataset, &cli_config),
        ModelType::Hgrn2 => run_hgrn2(&dataset, &cli_config),
        ModelType::Hawk => run_hawk(&dataset, &cli_config),
        ModelType::RetNet => run_retnet(&dataset, &cli_config),
        ModelType::LogLinear => run_log_linear(&dataset, &cli_config),
        ModelType::Ttt => run_ttt(&cli_config, dataset),
        ModelType::Kan => run_kan(&cli_config, dataset),
        ModelType::Factory => run_factory(&args, dataset),
    }
}

/// Public re-export so `eval.rs` can share the same mapping.
pub(crate) fn factory_key_for_model_pub(model: &ModelType) -> Result<&'static str> {
    factory_key_for_model(model)
}

/// Map a ModelType to its `Factory::*` key so `--auto-tune` can wrap it.
fn factory_key_for_model(model: &ModelType) -> Result<&'static str> {
    Ok(match model {
        ModelType::Sgbt => "sgbt",
        ModelType::Distributional => "distributional",
        ModelType::Multiclass => "multiclass-sgbt",
        ModelType::Esn => "esn",
        ModelType::Mamba => "mamba",
        ModelType::Mamba3 => "mamba-3",
        ModelType::MambaBd => "mamba-bd",
        ModelType::Slstm => "s-lstm",
        ModelType::Mgrade => "mgrade",
        ModelType::SpikeNet => "spike-net",
        ModelType::Gla
        | ModelType::DeltaNet
        | ModelType::Hawk
        | ModelType::RetNet
        | ModelType::Hgrn2
        | ModelType::LogLinear => "attention",
        ModelType::DeltaProduct => "delta-product",
        ModelType::Rwkv7 => "rwkv-7",
        ModelType::Ttt => "ttt",
        ModelType::Kan => "kan",
        ModelType::Bagged | ModelType::Ngrc | ModelType::Factory => {
            return Err(eyre!(
                "--auto-tune is not supported for model type '{:?}'. Use --model-type factory --factories <list> instead.",
                model
            ));
        }
    })
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
    // --tui dispatch happens in `run()` before this branch, via the
    // multi-family `tui::run_with_dataset` path.
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

// ---------------------------------------------------------------------------
// Distributional SGBT
// ---------------------------------------------------------------------------

fn run_distributional(args: TrainArgs, cli_config: CliConfig, dataset: Dataset) -> Result<()> {
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

    // Save distributional model
    let state = model.to_distributional_state();
    let json = irithyll::serde_support::save_distributional_model(&state)
        .map_err(|e| eyre!("failed to serialize distributional model: {}", e))?;
    std::fs::write(&args.output, &json)?;
    println!("  Saved to: {}", args.output);

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

    // Save multiclass model
    let state = model.to_multiclass_state();
    let json = irithyll::serde_support::save_multiclass_model(&state)
        .map_err(|e| eyre!("failed to serialize multiclass model: {}", e))?;
    std::fs::write(&args.output, &json)?;
    println!("  Saved to: {}", args.output);

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

    // Save bagged model
    let state = model
        .to_bagged_state()
        .map_err(|e| eyre!("failed to snapshot bagged model: {}", e))?;
    let json = irithyll::serde_support::save_bagged_model(&state)
        .map_err(|e| eyre!("failed to serialize bagged model: {}", e))?;
    std::fs::write(&args.output, &json)?;
    println!("  Saved to: {}", args.output);

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
    use irithyll::attention::{
        AttentionMode, GatedDeltaMode, StreamingAttentionConfig, StreamingAttentionModel,
    };

    let att = &config.neural.attention;
    let att_config = StreamingAttentionConfig::builder()
        .d_model(dataset.n_features)
        .n_heads(att.n_heads)
        .mode(AttentionMode::GatedDeltaNet {
            beta_scale: 1.0,
            gate_mode_delta: GatedDeltaMode::Static,
        })
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
// TTT (Test-Time Training)
// ---------------------------------------------------------------------------

fn run_ttt(cli_config: &CliConfig, dataset: Dataset) -> Result<()> {
    let tc = &cli_config.neural.ttt;
    let d_model = tc.d_model.unwrap_or(32);
    let eta = tc.eta.unwrap_or(0.01);

    let mut model = irithyll::streaming_ttt(d_model, eta);

    println!(
        "Loaded {} samples, {} features (ttt, d_model={}, eta={})",
        dataset.n_samples, dataset.n_features, d_model, eta,
    );

    run_neural_headless(&mut model, &dataset, "ttt")
}

// ---------------------------------------------------------------------------
// KAN (Kolmogorov-Arnold Network)
// ---------------------------------------------------------------------------

fn run_kan(cli_config: &CliConfig, dataset: Dataset) -> Result<()> {
    let kc = &cli_config.neural.kan;
    let hidden = kc.hidden_size.unwrap_or(10);
    let lr = kc.lr.unwrap_or(0.01);

    let mut model = irithyll::streaming_kan(&[dataset.n_features, hidden, 1], lr);

    println!(
        "Loaded {} samples, {} features (kan, hidden={}, lr={})",
        dataset.n_samples, dataset.n_features, hidden, lr,
    );

    run_neural_headless(&mut model, &dataset, "kan")
}

// ---------------------------------------------------------------------------
// New v10 attention modes
// ---------------------------------------------------------------------------

fn run_delta_product(dataset: &Dataset, _config: &CliConfig) -> Result<()> {
    let mut model = irithyll::attention::delta_product(dataset.n_features.max(2), 1, 3);
    println!(
        "Loaded {} samples, {} features (delta-product, n_compositions=3)",
        dataset.n_samples, dataset.n_features,
    );
    run_neural_headless(&mut model, dataset, "delta-product")
}

fn run_rwkv7(dataset: &Dataset, _config: &CliConfig) -> Result<()> {
    let mut model = irithyll::attention::rwkv7(dataset.n_features.max(2), 1);
    println!(
        "Loaded {} samples, {} features (rwkv-7)",
        dataset.n_samples, dataset.n_features,
    );
    run_neural_headless(&mut model, dataset, "rwkv-7")
}

fn run_hgrn2(dataset: &Dataset, _config: &CliConfig) -> Result<()> {
    let mut model = irithyll::attention::hgrn2(dataset.n_features.max(2), 1, 0.9);
    println!(
        "Loaded {} samples, {} features (hgrn2, lower_bound=0.9)",
        dataset.n_samples, dataset.n_features,
    );
    run_neural_headless(&mut model, dataset, "hgrn2")
}

fn run_log_linear(dataset: &Dataset, _config: &CliConfig) -> Result<()> {
    use irithyll::attention::AttentionMode;

    let mut model = irithyll::log_linear(
        dataset.n_features.max(2),
        1,
        AttentionMode::GLA,
        irithyll::DEFAULT_MAX_LEVELS,
    );
    println!(
        "Loaded {} samples, {} features (log-linear, inner=gla)",
        dataset.n_samples, dataset.n_features,
    );
    run_neural_headless(&mut model, dataset, "log-linear")
}

// ---------------------------------------------------------------------------
// New v10 SSM / recurrent variants
// ---------------------------------------------------------------------------

fn run_mamba3(cli_config: CliConfig, dataset: Dataset) -> Result<()> {
    use irithyll::ssm::{MambaConfig, MambaVersion, StreamingMamba};

    let mc = &cli_config.neural.mamba;
    let config = MambaConfig::builder()
        .d_in(dataset.n_features)
        .n_state(mc.n_state)
        .version(MambaVersion::V3Exp { use_bcnorm: true })
        .seed(mc.seed)
        .warmup(mc.warmup)
        .build()
        .map_err(|e| eyre!("invalid Mamba-3 config: {}", e))?;

    let mut model = StreamingMamba::new(config);
    println!(
        "Loaded {} samples, {} features (mamba-3, n_state={})",
        dataset.n_samples, dataset.n_features, mc.n_state,
    );
    run_neural_headless(&mut model, &dataset, "mamba-3")
}

fn run_mamba_bd(cli_config: CliConfig, dataset: Dataset) -> Result<()> {
    let mc = &cli_config.neural.mamba;
    let block_size = (dataset.n_features / 2).max(1);
    let mut model = irithyll::mamba_bd(dataset.n_features, mc.n_state, block_size);
    println!(
        "Loaded {} samples, {} features (mamba-bd, block_size={})",
        dataset.n_samples, dataset.n_features, block_size,
    );
    run_neural_headless(&mut model, &dataset, "mamba-bd")
}

fn run_slstm(dataset: Dataset) -> Result<()> {
    let mut model = irithyll::streaming_slstm(dataset.n_features.max(2));
    println!(
        "Loaded {} samples, {} features (s-lstm)",
        dataset.n_samples, dataset.n_features,
    );
    run_neural_headless(&mut model, &dataset, "s-lstm")
}

fn run_mgrade(dataset: Dataset) -> Result<()> {
    let d_hidden = (dataset.n_features * 2).max(8);
    let mut model = irithyll::mgrade(dataset.n_features, d_hidden);
    println!(
        "Loaded {} samples, {} features (mgrade, d_hidden={})",
        dataset.n_samples, dataset.n_features, d_hidden,
    );
    run_neural_headless(&mut model, &dataset, "mgrade")
}

// ---------------------------------------------------------------------------
// Factory / AutoTuner (automated model selection)
// ---------------------------------------------------------------------------

fn run_factory(args: &TrainArgs, dataset: Dataset) -> Result<()> {
    let factory_names: Vec<&str> = args.factories.split(',').map(|s| s.trim()).collect();
    run_factory_with_keys(args, dataset, &factory_names)
}

/// Build an `AutoTuner` from a list of factory keys and stream the dataset
/// through it. Used by both `--model-type factory` and the `--auto-tune` shortcut.
fn run_factory_with_keys(args: &TrainArgs, dataset: Dataset, factory_keys: &[&str]) -> Result<()> {
    use irithyll::{AutoTuner, AutoTunerBuilder};

    let n_features = dataset.n_features;

    let mut builder: Option<AutoTunerBuilder> = None;
    for name in factory_keys {
        let factory = factory_from_name(name, n_features)?;
        builder = Some(match builder {
            None => AutoTuner::builder().factory(factory),
            Some(b) => b.add_factory(factory),
        });
    }

    let mut builder =
        builder.ok_or_else(|| eyre!("--factories must specify at least one factory"))?;
    if let Some(n) = args.n_initial {
        builder = builder.n_initial(n);
    }
    if let Some(n) = args.max_n_initial {
        builder = builder.max_n_initial(n);
    }
    if args.use_drift_rerace {
        builder = builder.use_drift_rerace(true);
    }

    let mut model = builder
        .build()
        .map_err(|e| eyre!("AutoTuner config error: {}", e))?;

    println!(
        "Loaded {} samples, {} features (auto-tune, racing: {})",
        dataset.n_samples,
        dataset.n_features,
        factory_keys.join(" + "),
    );

    run_neural_headless(&mut model, &dataset, "auto-tune")
}

/// Resolve a factory key string to a concrete `Factory`. Single source of
/// truth for the factories the CLI exposes; updates here propagate to both
/// `--factories` and `--auto-tune`.
pub(crate) fn factory_from_name(name: &str, n_features: usize) -> Result<irithyll::Factory> {
    use irithyll::Factory;
    Ok(match name {
        "sgbt" => Factory::sgbt(n_features),
        "distributional" => Factory::distributional(n_features),
        "multiclass-sgbt" => Factory::multiclass_sgbt(n_features, 4),
        "esn" => Factory::esn(),
        "mamba" => Factory::mamba(n_features),
        "mamba-3" | "mamba3" => Factory::mamba3(n_features),
        "mamba-bd" | "mambabd" => Factory::mamba_bd(n_features),
        "s-lstm" | "slstm" => Factory::slstm(n_features),
        "mgrade" => Factory::mgrade(n_features),
        "spike-net" | "spikenet" => Factory::spike_net(),
        "attention" => Factory::attention(n_features),
        "delta-product" | "deltaproduct" => Factory::delta_product(n_features),
        "rwkv-7" | "rwkv7" => Factory::rwkv7(n_features),
        "kan" => Factory::kan(n_features),
        "ttt" => Factory::ttt(n_features),
        _ => {
            return Err(eyre!(
                "unknown factory '{}'.\n  available: {}",
                name,
                FACTORY_KEYS.join(", "),
            ));
        }
    })
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
    println!("  [NOTE] Neural model serialization not yet implemented -- train-only mode");

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
