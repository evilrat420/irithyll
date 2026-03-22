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
    about = "irithyll streaming ML — train, predict, evaluate from the command line"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model from CSV/Parquet data
    Train(commands::train::TrainArgs),
    /// Run predictions with a trained model
    Predict(commands::predict::PredictArgs),
    /// Evaluate a model with prequential test-then-train
    Eval(commands::eval::EvalArgs),
    /// Inspect a saved model
    Inspect(commands::inspect::InspectArgs),
    /// Export model to embedded/ONNX format
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

    match cli.command {
        Commands::Train(args) => commands::train::run(args),
        Commands::Predict(args) => commands::predict::run(args),
        Commands::Eval(args) => commands::eval::run(args),
        Commands::Inspect(args) => commands::inspect::run(args),
        Commands::Export(args) => commands::export::run(args),
        Commands::Config(args) => commands::config::run(args),
    }
}
