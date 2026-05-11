use clap::Args;
use color_eyre::eyre::eyre;
use color_eyre::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::time::Instant;

use irithyll::metrics::RegressionMetrics;
use irithyll::{CohenKappa, DynSGBT, Loss, Sample, StreamingLearner};

use super::train::ModelType;
use crate::config::CliConfig;
use crate::data::Dataset;
use crate::output::print_metrics_table;

#[derive(Args)]
pub struct EvalArgs {
    /// Path to evaluation data (CSV or Parquet)
    pub data: String,

    /// Path to config file (TOML)
    #[arg(short, long)]
    pub config: Option<String>,

    /// Target column name (default: last column)
    #[arg(short, long)]
    pub target: Option<String>,

    /// Number of boosting steps
    #[arg(long)]
    pub n_steps: Option<usize>,

    /// Learning rate
    #[arg(long)]
    pub learning_rate: Option<f64>,

    /// Max tree depth
    #[arg(long)]
    pub max_depth: Option<usize>,

    /// Model type. See `train --help` for the full list.
    #[arg(long, default_value = "sgbt", value_name = "TYPE")]
    pub model_type: String,

    /// Number of classes (required for softmax loss and multiclass model type)
    #[arg(long)]
    pub n_classes: Option<usize>,

    /// Rolling window size for metrics
    #[arg(long, default_value = "1000")]
    pub window: usize,

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

pub fn run(args: EvalArgs) -> Result<()> {
    // 1. Load or create config
    let mut cli_config = if let Some(ref path) = args.config {
        CliConfig::from_file(path)?
    } else {
        CliConfig::default()
    };

    if let Some(n) = args.n_steps {
        cli_config.model.n_steps = n;
    }
    if let Some(lr) = args.learning_rate {
        cli_config.model.learning_rate = lr;
    }
    if let Some(d) = args.max_depth {
        cli_config.model.max_depth = d;
    }

    let model_type = ModelType::from_str(&args.model_type)?;
    let dataset = Dataset::from_csv(Path::new(&args.data), args.target.as_deref())?;

    // --auto-tune shortcut: wrap the chosen model in AutoTuner racing.
    if args.auto_tune && !matches!(model_type, ModelType::Factory) {
        let seed_factory = super::train::factory_key_for_model_pub(&model_type)?;
        return run_neural_eval_factory_with_keys(&args, &dataset, &[seed_factory]);
    }

    // --tui dispatch for any supported family. Multi-family routing matches
    // `train --tui` so users can flip between train/eval with the same
    // dashboard and same model selection rules.
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
            return crate::tui::run_eval_with_dataset(model, dataset, label);
        } else {
            return Err(eyre!(
                "--tui not yet supported for --model-type {}. Supported: sgbt, mamba, ttt, kan, esn, ngrc, spike-net",
                args.model_type
            ));
        }
    }

    // Route neural models through the neural eval path
    match model_type {
        ModelType::Sgbt | ModelType::Distributional | ModelType::Multiclass | ModelType::Bagged => {
            let loss_type = super::train::parse_loss_type(&cli_config.model.loss, args.n_classes)?;
            let sgbt_config = cli_config
                .to_sgbt_config_builder()?
                .feature_names(dataset.feature_names.clone())
                .build()?;
            let mut model = DynSGBT::with_loss(sgbt_config, loss_type.into_loss());

            run_eval_headless(&mut model, &dataset)
        }
        ModelType::Ngrc => run_neural_eval_ngrc(&cli_config, &dataset),
        ModelType::Esn => run_neural_eval_esn(&cli_config, &dataset),
        ModelType::Mamba => run_neural_eval_mamba(&cli_config, &dataset),
        ModelType::Mamba3 => run_neural_eval_mamba3(&cli_config, &dataset),
        ModelType::MambaBd => run_neural_eval_mamba_bd(&cli_config, &dataset),
        ModelType::Slstm => run_neural_eval_slstm(&dataset),
        ModelType::Mgrade => run_neural_eval_mgrade(&dataset),
        ModelType::SpikeNet => run_neural_eval_spikenet(&cli_config, &dataset),
        ModelType::Gla => run_neural_eval_gla(&cli_config, &dataset),
        ModelType::DeltaNet => run_neural_eval_deltanet(&cli_config, &dataset),
        ModelType::DeltaProduct => run_neural_eval_delta_product(&dataset),
        ModelType::Rwkv7 => run_neural_eval_rwkv7(&dataset),
        ModelType::Hgrn2 => run_neural_eval_hgrn2(&dataset),
        ModelType::Hawk => run_neural_eval_hawk(&cli_config, &dataset),
        ModelType::RetNet => run_neural_eval_retnet(&cli_config, &dataset),
        ModelType::LogLinear => run_neural_eval_log_linear(&dataset),
        ModelType::Ttt => run_neural_eval_ttt(&cli_config, &dataset),
        ModelType::Kan => run_neural_eval_kan(&cli_config, &dataset),
        ModelType::Factory => run_neural_eval_factory(&args, &dataset),
    }
}

fn run_eval_headless(model: &mut DynSGBT, dataset: &Dataset) -> Result<()> {
    println!(
        "Loaded {} samples, {} features",
        dataset.n_samples, dataset.n_features
    );

    let mut reg_metrics = RegressionMetrics::new();
    let mut kappa = CohenKappa::new();
    let mut n_correct: u64 = 0;
    let mut n_total: u64 = 0;

    let pb = ProgressBar::new(dataset.n_samples as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({per_sec})")
            .unwrap()
            .progress_chars("=> "),
    );

    let start = Instant::now();
    for i in 0..dataset.n_samples {
        let features = &dataset.features[i];
        let target = dataset.targets[i];
        let raw_pred = model.predict(features);
        let pred = model.loss().predict_transform(raw_pred);

        reg_metrics.update(target, pred);

        let pred_class = pred.round() as usize;
        let target_class = target.round() as usize;
        if pred_class == target_class {
            n_correct += 1;
        }
        n_total += 1;
        kappa.update(target_class, pred_class);

        let sample = Sample::new(features.clone(), target);
        model.train_one(&sample);
        pb.inc(1);
    }
    pb.finish_with_message("done");
    let elapsed = start.elapsed();

    println!();
    let accuracy = if n_total > 0 {
        n_correct as f64 / n_total as f64
    } else {
        0.0
    };

    print_metrics_table(&[
        ("Accuracy", accuracy),
        ("RMSE", reg_metrics.rmse()),
        ("MAE", reg_metrics.mae()),
        ("R-squared", reg_metrics.r_squared()),
        ("Kappa", kappa.kappa()),
    ]);

    println!();
    println!(
        "Evaluated {} samples in {:.2}s ({:.0} samples/sec)",
        dataset.n_samples,
        elapsed.as_secs_f64(),
        dataset.n_samples as f64 / elapsed.as_secs_f64()
    );
    println!("  Steps:  {}", model.n_steps());
    println!("  Leaves: {}", model.total_leaves());

    Ok(())
}

// ---------------------------------------------------------------------------
// Neural model eval constructors (prequential test-then-train)
// ---------------------------------------------------------------------------

fn run_neural_eval_ngrc(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
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
    run_neural_eval_headless(&mut model, dataset, "ngrc")
}

fn run_neural_eval_esn(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
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
    run_neural_eval_headless(&mut model, dataset, "esn")
}

fn run_neural_eval_mamba(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
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
    run_neural_eval_headless(&mut model, dataset, "mamba")
}

fn run_neural_eval_spikenet(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
    use irithyll::snn::{SpikeNet, SpikeNetConfig};

    let sc = &cli_config.neural.spikenet;
    let config = SpikeNetConfig::builder()
        .n_hidden(sc.n_hidden)
        .learning_rate(sc.learning_rate)
        .seed(sc.seed)
        .build()
        .map_err(|e| eyre!("invalid SpikeNet config: {}", e))?;

    let mut model = SpikeNet::new(config);
    run_neural_eval_headless(&mut model, dataset, "spikenet")
}

fn run_neural_eval_gla(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
    use irithyll::attention::{AttentionMode, StreamingAttentionConfig, StreamingAttentionModel};

    let att = &cli_config.neural.attention;
    let config = StreamingAttentionConfig::builder()
        .d_model(dataset.n_features)
        .n_heads(att.n_heads)
        .mode(AttentionMode::GLA)
        .seed(att.seed)
        .warmup(att.warmup)
        .build()
        .map_err(|e| eyre!("invalid GLA config: {}", e))?;

    let mut model = StreamingAttentionModel::new(config);
    run_neural_eval_headless(&mut model, dataset, "gla")
}

fn run_neural_eval_deltanet(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
    use irithyll::attention::{
        AttentionMode, GatedDeltaMode, StreamingAttentionConfig, StreamingAttentionModel,
    };

    let att = &cli_config.neural.attention;
    let config = StreamingAttentionConfig::builder()
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

    let mut model = StreamingAttentionModel::new(config);
    run_neural_eval_headless(&mut model, dataset, "deltanet")
}

fn run_neural_eval_hawk(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
    use irithyll::attention::{AttentionMode, StreamingAttentionConfig, StreamingAttentionModel};

    let att = &cli_config.neural.attention;
    let config = StreamingAttentionConfig::builder()
        .d_model(dataset.n_features)
        .n_heads(1)
        .mode(AttentionMode::Hawk)
        .seed(att.seed)
        .warmup(att.warmup)
        .build()
        .map_err(|e| eyre!("invalid Hawk config: {}", e))?;

    let mut model = StreamingAttentionModel::new(config);
    run_neural_eval_headless(&mut model, dataset, "hawk")
}

fn run_neural_eval_retnet(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
    use irithyll::attention::{AttentionMode, StreamingAttentionConfig, StreamingAttentionModel};

    let att = &cli_config.neural.attention;
    let config = StreamingAttentionConfig::builder()
        .d_model(dataset.n_features)
        .n_heads(1)
        .mode(AttentionMode::RetNet { gamma: att.gamma })
        .seed(att.seed)
        .warmup(att.warmup)
        .build()
        .map_err(|e| eyre!("invalid RetNet config: {}", e))?;

    let mut model = StreamingAttentionModel::new(config);
    run_neural_eval_headless(&mut model, dataset, "retnet")
}

fn run_neural_eval_ttt(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
    let tc = &cli_config.neural.ttt;
    let d_model = tc.d_model.unwrap_or(32);
    let eta = tc.eta.unwrap_or(0.01);

    let mut model = irithyll::streaming_ttt(d_model, eta);
    run_neural_eval_headless(&mut model, dataset, "ttt")
}

fn run_neural_eval_kan(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
    let kc = &cli_config.neural.kan;
    let hidden = kc.hidden_size.unwrap_or(10);
    let lr = kc.lr.unwrap_or(0.01);

    let mut model = irithyll::streaming_kan(&[dataset.n_features, hidden, 1], lr);
    run_neural_eval_headless(&mut model, dataset, "kan")
}

// ---------------------------------------------------------------------------
// New v10 model eval constructors
// ---------------------------------------------------------------------------

fn run_neural_eval_mamba3(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
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
    run_neural_eval_headless(&mut model, dataset, "mamba-3")
}

fn run_neural_eval_mamba_bd(cli_config: &CliConfig, dataset: &Dataset) -> Result<()> {
    let mc = &cli_config.neural.mamba;
    let block_size = (dataset.n_features / 2).max(1);
    let mut model = irithyll::mamba_bd(dataset.n_features, mc.n_state, block_size);
    run_neural_eval_headless(&mut model, dataset, "mamba-bd")
}

fn run_neural_eval_slstm(dataset: &Dataset) -> Result<()> {
    let mut model = irithyll::streaming_slstm(dataset.n_features.max(2));
    run_neural_eval_headless(&mut model, dataset, "s-lstm")
}

fn run_neural_eval_mgrade(dataset: &Dataset) -> Result<()> {
    let d_hidden = (dataset.n_features * 2).max(8);
    let mut model = irithyll::mgrade(dataset.n_features, d_hidden);
    run_neural_eval_headless(&mut model, dataset, "mgrade")
}

fn run_neural_eval_delta_product(dataset: &Dataset) -> Result<()> {
    let mut model = irithyll::attention::delta_product(dataset.n_features.max(2), 1, 3);
    run_neural_eval_headless(&mut model, dataset, "delta-product")
}

fn run_neural_eval_rwkv7(dataset: &Dataset) -> Result<()> {
    let mut model = irithyll::attention::rwkv7(dataset.n_features.max(2), 1);
    run_neural_eval_headless(&mut model, dataset, "rwkv-7")
}

fn run_neural_eval_hgrn2(dataset: &Dataset) -> Result<()> {
    let mut model = irithyll::attention::hgrn2(dataset.n_features.max(2), 1, 0.9);
    run_neural_eval_headless(&mut model, dataset, "hgrn2")
}

fn run_neural_eval_log_linear(dataset: &Dataset) -> Result<()> {
    use irithyll::attention::AttentionMode;
    let mut model = irithyll::log_linear(
        dataset.n_features.max(2),
        1,
        AttentionMode::GLA,
        irithyll::DEFAULT_MAX_LEVELS,
    );
    run_neural_eval_headless(&mut model, dataset, "log-linear")
}

fn run_neural_eval_factory(args: &EvalArgs, dataset: &Dataset) -> Result<()> {
    let factory_names: Vec<&str> = args.factories.split(',').map(|s| s.trim()).collect();
    run_neural_eval_factory_with_keys(args, dataset, &factory_names)
}

/// Build an `AutoTuner` from a list of factory keys and stream the dataset
/// through prequential eval.
fn run_neural_eval_factory_with_keys(
    args: &EvalArgs,
    dataset: &Dataset,
    factory_keys: &[&str],
) -> Result<()> {
    use irithyll::{AutoTuner, AutoTunerBuilder};

    let n_features = dataset.n_features;

    let mut builder: Option<AutoTunerBuilder> = None;
    for name in factory_keys {
        let factory = super::train::factory_from_name(name, n_features)?;
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

    println!("Racing factories: {}", factory_keys.join(" + "));
    run_neural_eval_headless(&mut model, dataset, "auto-tune")
}

// ---------------------------------------------------------------------------
// Shared prequential eval loop for neural models
// ---------------------------------------------------------------------------

fn run_neural_eval_headless(
    model: &mut dyn StreamingLearner,
    dataset: &Dataset,
    model_name: &str,
) -> Result<()> {
    println!(
        "Loaded {} samples, {} features ({})",
        dataset.n_samples, dataset.n_features, model_name,
    );

    let mut metrics = RegressionMetrics::new();
    let mut n_correct: u64 = 0;
    let mut n_total: u64 = 0;

    let pb = ProgressBar::new(dataset.n_samples as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({per_sec})")
            .unwrap()
            .progress_chars("=> "),
    );

    let print_interval = (dataset.n_samples / 10).max(1);
    let start = Instant::now();

    for i in 0..dataset.n_samples {
        let features = &dataset.features[i];
        let target = dataset.targets[i];

        // Test-then-train (prequential evaluation)
        let pred = model.predict(features);
        metrics.update(target, pred);

        let pred_class = pred.round() as usize;
        let target_class = target.round() as usize;
        if pred_class == target_class {
            n_correct += 1;
        }
        n_total += 1;

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

    let accuracy = if n_total > 0 {
        n_correct as f64 / n_total as f64
    } else {
        0.0
    };

    println!();

    print_metrics_table(&[
        ("Accuracy", accuracy),
        ("RMSE", metrics.rmse()),
        ("MAE", metrics.mae()),
        ("R-squared", metrics.r_squared()),
    ]);

    println!();
    println!(
        "Evaluated {} samples in {:.2}s ({:.0} samples/sec)",
        dataset.n_samples,
        elapsed.as_secs_f64(),
        dataset.n_samples as f64 / elapsed.as_secs_f64()
    );
    println!("  [NOTE] Neural model serialization not yet implemented -- eval is inline prequential only");

    Ok(())
}
