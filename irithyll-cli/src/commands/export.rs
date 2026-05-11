use clap::Args;
use color_eyre::eyre::eyre;
use color_eyre::Result;

use irithyll::export_embedded::{export_packed, export_packed_i16};
use irithyll::serde_support::load_model;

/// Supported export format keys. Single source of truth for help text and
/// error messages.
const FORMAT_KEYS: &[&str] = &["packed", "packed-i16", "json"];

#[derive(Args)]
pub struct ExportArgs {
    /// Path to saved model (JSON)
    pub model: String,

    /// Export format. See --help for supported values.
    #[arg(short, long, default_value = "packed", value_name = "FORMAT")]
    pub format: String,

    /// Output path. Defaults vary per format.
    #[arg(short, long)]
    pub output: Option<String>,

    /// Number of input features (required for packed and packed-i16 export)
    #[arg(long)]
    pub n_features: Option<usize>,
}

pub fn run(args: ExportArgs) -> Result<()> {
    // 1. Load model
    let json = std::fs::read_to_string(&args.model)?;
    let model = load_model(&json)?;
    println!(
        "Loaded model from {} ({} steps, {} leaves)",
        args.model,
        model.n_steps(),
        model.total_leaves()
    );

    // 2. Match on format
    match args.format.as_str() {
        "packed" => {
            let n_features = args
                .n_features
                .ok_or_else(|| eyre!("--n-features is required for packed export"))?;

            let bytes = export_packed(&model, n_features);
            let out_path = args.output.unwrap_or_else(|| "model.bin".to_string());

            std::fs::write(&out_path, &bytes)?;

            println!();
            println!("Export complete");
            println!("  Format:   packed (f32, 12-byte nodes)");
            println!("  Size:     {} bytes", bytes.len());
            println!("  Features: {}", n_features);
            println!("  Saved to: {}", out_path);
        }
        "packed-i16" | "quantized" => {
            let n_features = args
                .n_features
                .ok_or_else(|| eyre!("--n-features is required for packed-i16 export"))?;

            let bytes = export_packed_i16(&model, n_features);
            let out_path = args.output.unwrap_or_else(|| "model_i16.bin".to_string());

            std::fs::write(&out_path, &bytes)?;

            println!();
            println!("Export complete");
            println!("  Format:   packed-i16 (per-feature quantized thresholds)");
            println!("  Size:     {} bytes", bytes.len());
            println!("  Features: {}", n_features);
            println!("  Saved to: {}", out_path);
        }
        "json" => {
            // Re-export as JSON (useful for format conversion / pretty-printing)
            let out_path = args
                .output
                .unwrap_or_else(|| "model_export.json".to_string());

            std::fs::write(&out_path, &json)?;

            println!();
            println!("Export complete");
            println!("  Format:   JSON");
            println!("  Size:     {} bytes", json.len());
            println!("  Saved to: {}", out_path);
        }
        other => {
            return Err(eyre!(
                "unknown export format '{}'.\n  supported: {}",
                other,
                FORMAT_KEYS.join(", "),
            ));
        }
    }

    Ok(())
}
