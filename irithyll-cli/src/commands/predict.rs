use clap::Args;
use color_eyre::Result;
use std::path::Path;

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

/// Model type detected from JSON structure.
enum LoadedModel {
    /// Standard SGBT (scalar regression/classification).
    Sgbt(irithyll::DynSGBT),
    /// Distributional SGBT (predicts mu + sigma).
    Distributional(irithyll::DistributionalSGBT),
    /// Multi-class SGBT (predicts class probabilities).
    Multiclass(irithyll::MulticlassSGBT),
    /// Bagged SGBT (averaged scalar prediction).
    Bagged(irithyll::BaggedSGBT),
}

/// Detect model type from JSON and load the appropriate model.
///
/// Distinguishes model types by checking for unique top-level keys:
/// - `location_steps` => Distributional
/// - `committees` => Multiclass
/// - `bags` => Bagged
/// - otherwise => standard SGBT
fn load_any_model(json: &str) -> Result<LoadedModel> {
    let value: serde_json::Value = serde_json::from_str(json)?;

    if value.get("location_steps").is_some() {
        let state: irithyll::serde_support::DistributionalModelState =
            irithyll::serde_support::load_distributional_model(json)?;
        Ok(LoadedModel::Distributional(
            irithyll::DistributionalSGBT::from_distributional_state(state),
        ))
    } else if value.get("committees").is_some() {
        let state: irithyll::serde_support::MulticlassModelState =
            irithyll::serde_support::load_multiclass_model(json)?;
        Ok(LoadedModel::Multiclass(
            irithyll::MulticlassSGBT::from_multiclass_state(state),
        ))
    } else if value.get("bags").is_some() {
        let state: irithyll::serde_support::BaggedModelState =
            irithyll::serde_support::load_bagged_model(json)?;
        Ok(LoadedModel::Bagged(
            irithyll::BaggedSGBT::from_bagged_state(state, irithyll::loss::squared::SquaredLoss),
        ))
    } else {
        let model = irithyll::serde_support::load_model(json)?;
        Ok(LoadedModel::Sgbt(model))
    }
}

pub fn run(args: PredictArgs) -> Result<()> {
    // 1. Load model (auto-detect type from JSON)
    let json = std::fs::read_to_string(&args.model)?;
    let loaded = load_any_model(&json)?;

    match &loaded {
        LoadedModel::Sgbt(m) => println!(
            "Loaded SGBT model from {} ({} steps)",
            args.model,
            m.n_steps()
        ),
        LoadedModel::Distributional(m) => println!(
            "Loaded distributional model from {} ({} samples seen)",
            args.model,
            m.n_samples_seen()
        ),
        LoadedModel::Multiclass(m) => println!(
            "Loaded multiclass model from {} ({} classes)",
            args.model,
            m.n_classes()
        ),
        LoadedModel::Bagged(m) => println!(
            "Loaded bagged model from {} ({} bags)",
            args.model,
            m.n_bags()
        ),
    }

    // 2. Load dataset
    let dataset = Dataset::from_csv(Path::new(&args.data), args.target.as_deref())?;
    println!(
        "Loaded {} samples, {} features",
        dataset.n_samples, dataset.n_features
    );

    // 3. Generate predictions and output based on model type
    match loaded {
        LoadedModel::Sgbt(model) => {
            let predictions: Vec<f64> = (0..dataset.n_samples)
                .map(|i| {
                    let raw = model.predict(&dataset.features[i]);
                    model.loss().predict_transform(raw)
                })
                .collect();

            output_scalar_predictions(&args, &dataset, &predictions, &["prediction"])?;
        }

        LoadedModel::Distributional(model) => {
            let predictions: Vec<(f64, f64)> = (0..dataset.n_samples)
                .map(|i| {
                    let pred = model.predict(&dataset.features[i]);
                    (pred.mu, pred.sigma)
                })
                .collect();

            if let Some(ref out_path) = args.output {
                let mut wtr = csv::Writer::from_path(out_path)?;
                wtr.write_record(["mu", "sigma"])?;
                for (mu, sigma) in &predictions {
                    wtr.write_record([format!("{:.6}", mu), format!("{:.6}", sigma)])?;
                }
                wtr.flush()?;
                println!("Predictions written to {}", out_path);
            } else {
                println!("{:<16} {:<16}", "mu", "sigma");
                for (mu, sigma) in &predictions {
                    println!("{:<16.6} {:<16.6}", mu, sigma);
                }
            }

            // RMSE on mu if targets exist
            if !dataset.targets.is_empty() {
                let sum_sq: f64 = predictions
                    .iter()
                    .enumerate()
                    .map(|(i, (mu, _))| {
                        let err = dataset.targets[i] - mu;
                        err * err
                    })
                    .sum();
                let rmse = (sum_sq / dataset.n_samples as f64).sqrt();
                println!();
                println!("RMSE (mu vs target): {:.6}", rmse);
            }
        }

        LoadedModel::Multiclass(model) => {
            let predictions: Vec<(usize, Vec<f64>)> = (0..dataset.n_samples)
                .map(|i| {
                    let proba = model.predict_proba(&dataset.features[i]);
                    let class = proba
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    (class, proba)
                })
                .collect();

            let n_classes = model.n_classes();

            if let Some(ref out_path) = args.output {
                let mut wtr = csv::Writer::from_path(out_path)?;
                let mut headers: Vec<String> = vec!["predicted_class".to_string()];
                for c in 0..n_classes {
                    headers.push(format!("prob_class_{}", c));
                }
                wtr.write_record(&headers)?;
                for (class, proba) in &predictions {
                    let mut row = vec![class.to_string()];
                    for p in proba {
                        row.push(format!("{:.6}", p));
                    }
                    wtr.write_record(&row)?;
                }
                wtr.flush()?;
                println!("Predictions written to {}", out_path);
            } else {
                // Print header
                print!("{:<16}", "class");
                for c in 0..n_classes {
                    print!("{:<12}", format!("p({})", c));
                }
                println!();
                for (class, proba) in &predictions {
                    print!("{:<16}", class);
                    for p in proba {
                        print!("{:<12.6}", p);
                    }
                    println!();
                }
            }

            // Accuracy if targets exist
            if !dataset.targets.is_empty() {
                let correct: usize = predictions
                    .iter()
                    .enumerate()
                    .filter(|(i, (class, _))| *class == dataset.targets[*i] as usize)
                    .count();
                let accuracy = correct as f64 / dataset.n_samples as f64;
                println!();
                println!(
                    "Accuracy: {:.4} ({}/{})",
                    accuracy, correct, dataset.n_samples
                );
            }
        }

        LoadedModel::Bagged(model) => {
            let predictions: Vec<f64> = (0..dataset.n_samples)
                .map(|i| model.predict(&dataset.features[i]))
                .collect();

            output_scalar_predictions(&args, &dataset, &predictions, &["prediction"])?;
        }
    }

    Ok(())
}

/// Output scalar predictions (shared by SGBT and Bagged).
fn output_scalar_predictions(
    args: &PredictArgs,
    dataset: &Dataset,
    predictions: &[f64],
    headers: &[&str],
) -> Result<()> {
    if let Some(ref out_path) = args.output {
        let mut wtr = csv::Writer::from_path(out_path)?;
        wtr.write_record(headers)?;
        for p in predictions {
            wtr.write_record([format!("{:.6}", p)])?;
        }
        wtr.flush()?;
        println!("Predictions written to {}", out_path);
    } else {
        for p in predictions {
            println!("{:.6}", p);
        }
    }

    // RMSE if targets exist
    if !dataset.targets.is_empty() {
        let sum_sq: f64 = predictions
            .iter()
            .enumerate()
            .map(|(i, pred)| {
                let err = dataset.targets[i] - pred;
                err * err
            })
            .sum();
        let rmse = (sum_sq / dataset.n_samples as f64).sqrt();
        println!();
        println!("RMSE: {:.6}", rmse);
    }

    Ok(())
}
