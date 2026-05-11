use clap::{Parser, Subcommand};
use color_eyre::Result;

mod commands;
mod config;
mod data;
mod output;
#[cfg(feature = "tui")]
pub mod tui;

#[derive(Parser)]
#[command(
    name = "irithyll",
    version,
    about = "irithyll streaming ML — train, predict, evaluate from the command line",
    long_about = "irithyll streaming ML\n\
                  Train, predict, evaluate, inspect, and export streaming ML models from the command line.\n\
                  Models compose like LEGO: any --model-type can be wrapped in --auto-tune racing."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Launch the TUI dashboard on a synthetic regression stream (Friedman) for demos and screenshots.
    #[cfg(feature = "tui")]
    #[arg(long, global = false)]
    tui: bool,

    /// Model family to drive the TUI demo. One of:
    /// `sgbt`, `mamba`, `ttt`, `kan`, `esn`, `ngrc`, `spikenet`, `linear`.
    /// Defaults to SGBT for backwards-compatible screenshots.
    #[cfg(feature = "tui")]
    #[arg(long, global = false, default_value = "sgbt")]
    family: String,

    /// Per-sample throttle in microseconds for the TUI demo. 0 = max throughput
    /// (default). Useful for reproducing the README GIF pace (~800 samp/s with 500).
    /// Only affects the no-subcommand demo.
    #[cfg(feature = "tui")]
    #[arg(long, global = false, default_value = "0")]
    throttle_us: u64,

    /// Benchmark generator for the TUI demo. One of:
    /// `friedman` (default), `lorenz`, `mackey-glass`, `periodic`, `mqar`, `needle`.
    /// All are regression streams from `irithyll::generators`.
    #[cfg(feature = "tui")]
    #[arg(long, global = false, default_value = "friedman")]
    bench: String,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model from CSV data
    Train(commands::train::TrainArgs),
    /// Run predictions with a trained model
    Predict(commands::predict::PredictArgs),
    /// Evaluate a model with prequential test-then-train
    Eval(commands::eval::EvalArgs),
    /// Inspect a saved model
    Inspect(commands::inspect::InspectArgs),
    /// Export a model to embedded packed format
    Export(commands::export::ExportArgs),
    /// Generate or validate config files
    Config(commands::config::ConfigArgs),
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let cli = Cli::parse();

    // Setup logging based on verbosity
    let filter = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // No subcommand → drop into the TUI demo on a self-contained Friedman
    // regression stream. The `--tui` flag is preserved for backward compat
    // but is no longer required: `irithyll` with no arguments lands on the
    // dashboard. Subcommands (train / predict / eval / inspect / export /
    // config) skip the TUI and run directly.
    #[cfg(feature = "tui")]
    if cli.command.is_none() {
        let family = parse_family_flag(&cli.family)?;
        return run_tui_demo(family, &cli.bench, cli.throttle_us);
    }

    let command = cli.command.ok_or_else(|| {
        color_eyre::eyre::eyre!("no subcommand given. Run `irithyll --help` for usage.")
    })?;

    match command {
        Commands::Train(args) => commands::train::run(args),
        Commands::Predict(args) => commands::predict::run(args),
        Commands::Eval(args) => commands::eval::run(args),
        Commands::Inspect(args) => commands::inspect::run(args),
        Commands::Export(args) => commands::export::run(args),
        Commands::Config(args) => commands::config::run(args),
    }
}

/// Parse the `--family` CLI argument into a [`tui::ModelFamily`].
///
/// Accepts case-insensitive matches plus a couple of common aliases. The list
/// must stay in sync with `ModelFamily::ALL` and `ModelFamily::label`.
#[cfg(feature = "tui")]
fn parse_family_flag(flag: &str) -> Result<tui::ModelFamily> {
    use tui::ModelFamily;
    match flag.to_ascii_lowercase().as_str() {
        "sgbt" => Ok(ModelFamily::Sgbt),
        "mamba" => Ok(ModelFamily::Mamba),
        "ttt" => Ok(ModelFamily::Ttt),
        "kan" => Ok(ModelFamily::Kan),
        "esn" => Ok(ModelFamily::Esn),
        "ngrc" | "ng-rc" | "ng_rc" => Ok(ModelFamily::Ngrc),
        "spikenet" | "snn" => Ok(ModelFamily::SpikeNet),
        "linear" => Ok(ModelFamily::Linear),
        other => Err(color_eyre::eyre::eyre!(
            "unknown --family `{other}`. Expected one of: sgbt, mamba, ttt, kan, esn, ngrc, spikenet, linear"
        )),
    }
}

// DemoModel + refresh_family_diagnostics moved to `tui::demo`. See
// `tui/demo.rs` for the multi-family dispatch + generic training loop.

/// Run the TUI on an in-tree streaming generator — used for screenshots,
/// quick smoke tests, and "try irithyll on a benchmark" exploration.
///
/// All supported generators are real `irithyll::generators` components. Each
/// arm sizes the `DemoModel` to the generator's actual feature dimension so
/// fixed-d_in models (Mamba, TTT, KAN) don't index out of bounds.
#[cfg(feature = "tui")]
fn run_tui_demo(family: tui::ModelFamily, bench: &str, throttle_us: u64) -> Result<()> {
    use irithyll::generators::{
        Friedman, Lorenz, MackeyGlass, MqarStream, NeedleStream, PeriodicStream, StreamGenerator,
    };
    const N_SAMPLES: usize = 20_000;

    macro_rules! launch {
        ($gen:expr, $label:expr) => {{
            let gen = $gen;
            let n_features = gen.n_features();
            let model = tui::DemoModel::build_for_dataset(family, n_features);
            tui::run_with_generator(model, gen, N_SAMPLES, $label.to_string(), throttle_us)
        }};
    }

    match bench.to_ascii_lowercase().as_str() {
        "friedman" => launch!(Friedman::with_config(42, 1.0, 0.001), "friedman"),
        "lorenz" => launch!(
            Lorenz::with_config(10.0, 28.0, 8.0 / 3.0, 0.01, 10, 1000, N_SAMPLES + 100),
            "lorenz"
        ),
        "mackey-glass" | "mackey_glass" | "mackeyglass" => launch!(
            MackeyGlass::with_config(42, 17, 6, 500, N_SAMPLES + 100),
            "mackey-glass"
        ),
        "periodic" => launch!(
            PeriodicStream::with_config(42, 20, 1.0, 3, 0.05, 10),
            "periodic"
        ),
        "mqar" => launch!(MqarStream::with_config(42, 128, 8, 4), "mqar"),
        "needle" => launch!(
            NeedleStream::with_config(42, 8, 256, 0, 3.0, 1.0),
            "needle"
        ),
        other => Err(color_eyre::eyre::eyre!(
            "unknown --bench `{other}`. Expected one of: friedman, lorenz, mackey-glass, periodic, mqar, needle"
        )),
    }
}
