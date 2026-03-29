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

    /// Show ensemble diagnostics (tree structure, replacements, per-tree stats)
    #[arg(long)]
    pub diagnostics: bool,
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

    // 4. Ensemble diagnostics
    if args.diagnostics {
        println!();
        let diag = model.diagnostics_overview();

        println!("=== Ensemble Diagnostics ===");
        println!();
        print_metrics_table(&[
            ("Trees", diag.n_trees as f64),
            ("Total Samples", diag.n_samples as f64),
            ("Total Replacements", diag.total_replacements as f64),
            ("Base Prediction", diag.base_prediction),
            ("Learning Rate", diag.learning_rate),
        ]);

        // Per-tree stats
        if !diag.trees.is_empty() {
            println!();
            println!(
                "{:>5}  {:>7}  {:>7}  {:>5}  {:>9}  {:>13}",
                "Tree", "Nodes", "Leaves", "Depth", "Samples", "Replacements"
            );
            println!(
                "{:>5}  {:>7}  {:>7}  {:>5}  {:>9}  {:>13}",
                "-----", "-------", "-------", "-----", "---------", "-------------"
            );
            for (i, tree) in diag.trees.iter().enumerate() {
                println!(
                    "{:>5}  {:>7}  {:>7}  {:>5}  {:>9}  {:>13}",
                    i,
                    tree.n_nodes,
                    tree.n_leaves,
                    tree.max_depth,
                    tree.n_samples,
                    tree.n_replacements,
                );
            }

            // Aggregate stats
            let avg_depth = diag.trees.iter().map(|t| t.max_depth).sum::<usize>() as f64
                / diag.trees.len() as f64;
            let avg_nodes = diag.trees.iter().map(|t| t.n_nodes).sum::<usize>() as f64
                / diag.trees.len() as f64;
            let avg_leaves = diag.trees.iter().map(|t| t.n_leaves).sum::<usize>() as f64
                / diag.trees.len() as f64;
            let total_nodes: usize = diag.trees.iter().map(|t| t.n_nodes).sum();
            let max_depth = diag.trees.iter().map(|t| t.max_depth).max().unwrap_or(0);

            println!();
            println!("Aggregate:");
            println!("  Avg depth:  {:.1}", avg_depth);
            println!("  Max depth:  {}", max_depth);
            println!("  Avg nodes:  {:.1}", avg_nodes);
            println!("  Avg leaves: {:.1}", avg_leaves);
            println!("  Total nodes: {}", total_nodes);
        }
    }

    // 5. Full config JSON
    if args.full {
        println!();
        println!("Full Config:");
        println!("{}", config_json);
    }

    Ok(())
}
