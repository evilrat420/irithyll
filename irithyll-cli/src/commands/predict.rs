use clap::Args;
use color_eyre::Result;
use std::path::Path;

use irithyll::serde_support::load_model;
use irithyll::Loss;

use crate::data::Dataset;

#[derive(Args)]
pub struct PredictArgs {
    /// Path to input data (CSV or Parquet)
    pub data: String,

    /// Path to trained model (JSON)
    #[arg(short, long, default_value = "model.json")]
    pub model: String,

    /// Target column name (default: last column)
    #[arg(short, long)]
    pub target: Option<String>,

    /// Output predictions path (CSV)
    #[arg(short, long)]
    pub output: Option<String>,
}

pub fn run(args: PredictArgs) -> Result<()> {
    // 1. Load model
    let json = std::fs::read_to_string(&args.model)?;
    let model = load_model(&json)?;
    println!(
        "Loaded model from {} ({} steps)",
        args.model,
        model.n_steps()
    );

    // 2. Load dataset
    let dataset = Dataset::from_csv(Path::new(&args.data), args.target.as_deref())?;
    println!(
        "Loaded {} samples, {} features",
        dataset.n_samples, dataset.n_features
    );

    // 3. Generate predictions
    let mut predictions = Vec::with_capacity(dataset.n_samples);
    for i in 0..dataset.n_samples {
        let raw = model.predict(&dataset.features[i]);
        let pred = model.loss().predict_transform(raw);
        predictions.push(pred);
    }

    // 4. Output predictions
    if let Some(ref out_path) = args.output {
        let mut wtr = csv::Writer::from_path(out_path)?;
        wtr.write_record(["prediction"])?;
        for p in &predictions {
            wtr.write_record([format!("{:.6}", p)])?;
        }
        wtr.flush()?;
        println!("Predictions written to {}", out_path);
    } else {
        for p in &predictions {
            println!("{:.6}", p);
        }
    }

    // 5. If targets exist, compute RMSE
    let has_targets = dataset.targets.iter().any(|t| *t != 0.0) || !dataset.targets.is_empty();
    if has_targets && !dataset.targets.is_empty() {
        let mut sum_sq = 0.0;
        for (i, pred) in predictions.iter().enumerate() {
            let err = dataset.targets[i] - pred;
            sum_sq += err * err;
        }
        let rmse = (sum_sq / dataset.n_samples as f64).sqrt();
        println!();
        println!("RMSE: {:.6}", rmse);
    }

    Ok(())
}
