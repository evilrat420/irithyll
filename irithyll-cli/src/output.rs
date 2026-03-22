use comfy_table::{ContentArrangement, Table};

/// Print a metrics table to stdout.
pub fn print_metrics_table(metrics: &[(&str, f64)]) {
    let mut table = Table::new();
    table.set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Metric", "Value"]);

    for (name, value) in metrics {
        table.add_row(vec![name.to_string(), format!("{:.6}", value)]);
    }

    println!("{table}");
}

/// Print a model summary.
pub fn print_model_summary(n_steps: usize, n_samples: u64, n_leaves: usize, config_json: &str) {
    println!("Model Summary");
    println!("  Steps:   {}", n_steps);
    println!("  Samples: {}", n_samples);
    println!("  Leaves:  {}", n_leaves);
    println!();
    println!("Config: {}", config_json);
}
