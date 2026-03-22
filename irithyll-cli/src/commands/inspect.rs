use clap::Args;
use color_eyre::Result;

use irithyll::serde_support::{from_json, load_model, ModelState};

use crate::output::{print_metrics_table, print_model_summary};

#[derive(Args)]
pub struct InspectArgs {
    /// Path to saved model (JSON)
    pub model: String,

    /// Show full config details
    #[arg(long)]
    pub full: bool,

    /// Show feature importances
    #[arg(long)]
    pub importances: bool,
}

pub fn run(args: InspectArgs) -> Result<()> {
    // 1. Load model
    let json = std::fs::read_to_string(&args.model)?;
    let model = load_model(&json)?;
    let state: ModelState = from_json(&json)?;

    // 2. Print model summary
    let config_json = irithyll::serde_support::to_json_pretty(&state.config)
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "(could not serialize config)".to_string());

    print_model_summary(
        model.n_steps(),
        model.n_samples_seen(),
        model.total_leaves(),
        &config_json,
    );

    println!();
    println!("Loss type: {:?}", state.loss_type);
    println!("Base prediction: {:.6}", state.base_prediction);

    // 3. Feature importances
    if args.importances {
        println!();
        let importances = model.feature_importances();
        if importances.is_empty() {
            println!("No feature importances available (no splits occurred).");
        } else {
            // Sort by importance descending
            let mut indexed: Vec<(usize, f64)> = importances
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("Feature Importances:");
            let feature_names = state.config.feature_names.as_ref();
            let rows: Vec<(&str, f64)> = indexed
                .iter()
                .take(20)
                .map(|(i, v)| {
                    let name = if let Some(names) = feature_names {
                        if *i < names.len() {
                            names[*i].as_str()
                        } else {
                            Box::leak(format!("feature_{}", i).into_boxed_str())
                        }
                    } else {
                        Box::leak(format!("feature_{}", i).into_boxed_str())
                    };
                    (name, *v)
                })
                .collect();
            print_metrics_table(&rows);
        }
    }

    // 4. Full config JSON
    if args.full {
        println!();
        println!("Full Config:");
        println!("{}", config_json);
    }

    Ok(())
}
