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

    /// Model type: sgbt (default), distributional, multiclass, bagged, ngrc, esn, mamba, spikenet, gla, deltanet, hawk, retnet, ttt, kan
    #[arg(long, default_value = "sgbt")]
    pub model_type: String,

    /// Number of classes (required for softmax loss and multiclass model type)
    #[arg(long)]
    pub n_classes: Option<usize>,

    /// Rolling window size for metrics
    #[arg(long, default_value = "1000")]
    pub window: usize,

    /// Launch TUI dashboard
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

    // Route neural models through the neural eval path
    match model_type {
        ModelType::Sgbt | ModelType::Distributional | ModelType::Multiclass | ModelType::Bagged => {
            let loss_type = super::train::parse_loss_type(&cli_config.model.loss, args.n_classes)?;
            let sgbt_config = cli_config
                .to_sgbt_config_builder()?
                .feature_names(dataset.feature_names.clone())
                .build()?;
            let mut model = DynSGBT::with_loss(sgbt_config, loss_type.into_loss());

            #[cfg(feature = "tui")]
            if args.tui {
                return run_eval_tui(model, dataset);
            }

            run_eval_headless(&mut model, &dataset)
        }
        ModelType::Ngrc => run_neural_eval_ngrc(&cli_config, &dataset),
        ModelType::Esn => run_neural_eval_esn(&cli_config, &dataset),
        ModelType::Mamba => run_neural_eval_mamba(&cli_config, &dataset),
        ModelType::SpikeNet => run_neural_eval_spikenet(&cli_config, &dataset),
        ModelType::Gla => run_neural_eval_gla(&cli_config, &dataset),
        ModelType::DeltaNet => run_neural_eval_deltanet(&cli_config, &dataset),
        ModelType::Hawk => run_neural_eval_hawk(&cli_config, &dataset),
        ModelType::RetNet => run_neural_eval_retnet(&cli_config, &dataset),
        ModelType::Ttt => run_neural_eval_ttt(&cli_config, &dataset),
        ModelType::Kan => run_neural_eval_kan(&cli_config, &dataset),
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

#[cfg(feature = "tui")]
fn run_eval_tui(mut model: DynSGBT, dataset: Dataset) -> Result<()> {
    use crate::tui::{AppState, SharedState};
    use std::sync::{Arc, Mutex};

    let state: SharedState = Arc::new(Mutex::new(AppState::new(dataset.n_samples as u64)));
    let tui_state = state.clone();

    let rt = tokio::runtime::Runtime::new()?;

    rt.block_on(async {
        let train_state = state.clone();
        let train_handle = tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            let update_interval = (dataset.n_samples / 200).max(1);

            let mut reg_metrics = RegressionMetrics::new();
            let mut n_correct: u64 = 0;
            let mut n_total: u64 = 0;

            for i in 0..dataset.n_samples {
                let features = &dataset.features[i];
                let target = dataset.targets[i];

                let raw_pred = model.predict(features);
                let pred = model.loss().predict_transform(raw_pred);
                let loss_val = model.loss().loss(target, pred);

                reg_metrics.update(target, pred);
                let pred_class = pred.round() as usize;
                let target_class = target.round() as usize;
                if pred_class == target_class {
                    n_correct += 1;
                }
                n_total += 1;

                let sample = Sample::new(features.clone(), target);
                model.train_one(&sample);

                if i % update_interval == 0 || i == dataset.n_samples - 1 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let throughput = if elapsed > 0.0 {
                        (i + 1) as f64 / elapsed
                    } else {
                        0.0
                    };
                    let accuracy = if n_total > 0 {
                        n_correct as f64 / n_total as f64
                    } else {
                        0.0
                    };

                    let mut s = train_state.lock().unwrap();
                    s.n_samples = (i + 1) as u64;
                    s.elapsed_secs = elapsed;
                    s.throughput = throughput;
                    s.loss_history.push(loss_val);
                    s.accuracy_history.push(accuracy);
                    s.metrics = vec![
                        ("Accuracy".to_string(), accuracy),
                        ("RMSE".to_string(), reg_metrics.rmse()),
                        ("MAE".to_string(), reg_metrics.mae()),
                        ("Throughput".to_string(), throughput),
                    ];
                    s.status_message = format!("Evaluating... {:.0} s/s", throughput);
                }
            }

            let elapsed = start.elapsed();
            let accuracy = if n_total > 0 {
                n_correct as f64 / n_total as f64
            } else {
                0.0
            };

            {
                let mut s = train_state.lock().unwrap();
                s.is_training = false;
                s.is_done = true;
                s.metrics = vec![
                    ("Accuracy".to_string(), accuracy),
                    ("RMSE".to_string(), reg_metrics.rmse()),
                    ("MAE".to_string(), reg_metrics.mae()),
                    ("R-squared".to_string(), reg_metrics.r_squared()),
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

            Ok::<(), color_eyre::Report>(())
        });

        let tui_result = crate::tui::run_tui(tui_state).await;
        let _ = train_handle.await?;
        tui_result
    })
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
    use irithyll::attention::{AttentionMode, StreamingAttentionConfig, StreamingAttentionModel};

    let att = &cli_config.neural.attention;
    let config = StreamingAttentionConfig::builder()
        .d_model(dataset.n_features)
        .n_heads(att.n_heads)
        .mode(AttentionMode::GatedDeltaNet)
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
    println!("  [NOTE] Neural model save not yet supported -- eval is inline prequential only");

    Ok(())
}
