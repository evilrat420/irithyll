use color_eyre::Result;
use std::collections::HashMap;
use std::path::Path;

/// Loaded dataset with features and targets.
pub struct Dataset {
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
    pub feature_names: Vec<String>,
    pub n_samples: usize,
    pub n_features: usize,
}

impl Dataset {
    /// Load from CSV file. Last column is target by default,
    /// or specify target column name.
    ///
    /// Non-numeric values are automatically label-encoded (mapped to
    /// sequential integers). This handles string targets like "UP"/"DOWN"
    /// and categorical features.
    pub fn from_csv(path: &Path, target_col: Option<&str>) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;

        let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();

        let target_idx = if let Some(name) = target_col {
            headers
                .iter()
                .position(|h| h == name)
                .ok_or_else(|| color_eyre::eyre::eyre!("target column '{}' not found", name))?
        } else {
            headers.len() - 1
        };

        let feature_names: Vec<String> = headers
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != target_idx)
            .map(|(_, h)| h.clone())
            .collect();

        // Label encoders: one per column, maps string -> f64
        let mut label_maps: Vec<HashMap<String, f64>> = vec![HashMap::new(); headers.len()];

        let mut features = Vec::new();
        let mut targets = Vec::new();

        for result in reader.records() {
            let record = result?;
            let mut row = Vec::with_capacity(headers.len() - 1);
            let mut target = 0.0;

            for (i, field) in record.iter().enumerate() {
                let val: f64 = if let Ok(v) = field.parse::<f64>() {
                    v
                } else {
                    // Label-encode: assign sequential integer per unique string
                    let map = &mut label_maps[i];
                    let next_id = map.len() as f64;
                    *map.entry(field.to_string()).or_insert(next_id)
                };

                if i == target_idx {
                    target = val;
                } else {
                    row.push(val);
                }
            }
            features.push(row);
            targets.push(target);
        }

        let n_samples = features.len();
        let n_features = feature_names.len();

        // Report any label-encoded columns
        for (i, map) in label_maps.iter().enumerate() {
            if !map.is_empty() {
                let col_name = &headers[i];
                let labels: Vec<_> = map
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, *v as usize))
                    .collect();
                eprintln!("  Label-encoded '{}': {}", col_name, labels.join(", "));
            }
        }

        Ok(Self {
            features,
            targets,
            feature_names,
            n_samples,
            n_features,
        })
    }
}
