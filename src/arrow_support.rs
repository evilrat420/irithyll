//! Arrow and Parquet integration for zero-copy data ingestion.
//!
//! Provides functions to train SGBT models directly from Arrow `RecordBatch`
//! and Parquet files, avoiding intermediate data copies.

#[cfg(feature = "arrow")]
use arrow::array::{AsArray, Float64Array, RecordBatch};
#[cfg(feature = "arrow")]
use arrow::datatypes::DataType;

/// Train an SGBT model from an Arrow RecordBatch.
///
/// Iterates rows of the batch, treating all Float64 columns (except the target)
/// as features. Columns that are not Float64 are silently skipped.
///
/// # Arguments
/// * `model` - The SGBT model to train
/// * `batch` - Arrow RecordBatch containing feature and target columns
/// * `target_col` - Name of the target column
/// * `loss` - Loss function for gradient/hessian computation
///
/// # Errors
/// Returns error if `target_col` is not found or is not Float64.
#[cfg(feature = "arrow")]
pub fn train_from_record_batch(
    model: &mut crate::ensemble::SGBT,
    batch: &RecordBatch,
    target_col: &str,
    loss: &dyn crate::loss::Loss,
) -> crate::error::Result<()> {
    let schema = batch.schema();

    // Find target column index by name.
    let target_idx = schema
        .index_of(target_col)
        .map_err(|_| crate::error::IrithyllError::Serialization(
            format!("target column '{}' not found in RecordBatch schema", target_col),
        ))?;

    // Verify target column is Float64.
    if schema.field(target_idx).data_type() != &DataType::Float64 {
        return Err(crate::error::IrithyllError::Serialization(format!(
            "target column '{}' must be Float64, got {:?}",
            target_col,
            schema.field(target_idx).data_type()
        )));
    }

    let target_array: &Float64Array = batch.column(target_idx).as_primitive();

    // Collect feature column indices: all Float64 columns except the target.
    let feature_indices: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(i, f)| *i != target_idx && f.data_type() == &DataType::Float64)
        .map(|(i, _)| i)
        .collect();

    // Downcast feature columns to Float64Array slices.
    let feature_columns: Vec<&Float64Array> = feature_indices
        .iter()
        .map(|&i| batch.column(i).as_primitive::<arrow::datatypes::Float64Type>())
        .collect();

    let n_features = feature_columns.len();
    let n_rows = batch.num_rows();

    // Pre-allocate feature buffer to avoid per-row allocation.
    let mut features = vec![0.0_f64; n_features];

    for row in 0..n_rows {
        let target = target_array.values()[row];
        if !target.is_finite() {
            continue;
        }
        let mut has_non_finite = false;
        for (j, col) in feature_columns.iter().enumerate() {
            let v = col.values()[row];
            if !v.is_finite() {
                has_non_finite = true;
                break;
            }
            features[j] = v;
        }
        if has_non_finite {
            continue;
        }
        model.train_one_slice(&features, target, loss);
    }

    Ok(())
}

/// Predict from an Arrow RecordBatch, returning a Float64Array of predictions.
///
/// All Float64 columns are treated as features (in schema order).
/// Non-Float64 columns are silently skipped.
#[cfg(feature = "arrow")]
pub fn predict_from_record_batch(
    model: &crate::ensemble::SGBT,
    batch: &RecordBatch,
) -> Float64Array {
    let schema = batch.schema();

    // Find all Float64 columns.
    let feature_columns: Vec<&Float64Array> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| f.data_type() == &DataType::Float64)
        .map(|(i, _)| batch.column(i).as_primitive::<arrow::datatypes::Float64Type>())
        .collect();

    let n_features = feature_columns.len();
    let n_rows = batch.num_rows();

    // Pre-allocate feature buffer.
    let mut features = vec![0.0_f64; n_features];
    let mut predictions = Vec::with_capacity(n_rows);

    for row in 0..n_rows {
        for (j, col) in feature_columns.iter().enumerate() {
            features[j] = col.values()[row];
        }
        predictions.push(model.predict(&features));
    }

    Float64Array::from(predictions)
}

/// Extract feature/target data from a RecordBatch as vectors of (features, target) pairs.
///
/// Useful for inspection or when you need the raw data.
///
/// # Errors
/// Returns error if `target_col` is not found or is not Float64.
#[cfg(feature = "arrow")]
pub fn record_batch_to_samples(
    batch: &RecordBatch,
    target_col: &str,
) -> crate::error::Result<Vec<(Vec<f64>, f64)>> {
    let schema = batch.schema();

    let target_idx = schema
        .index_of(target_col)
        .map_err(|_| crate::error::IrithyllError::Serialization(
            format!("target column '{}' not found in RecordBatch schema", target_col),
        ))?;

    if schema.field(target_idx).data_type() != &DataType::Float64 {
        return Err(crate::error::IrithyllError::Serialization(format!(
            "target column '{}' must be Float64, got {:?}",
            target_col,
            schema.field(target_idx).data_type()
        )));
    }

    let target_array: &Float64Array = batch.column(target_idx).as_primitive();

    let feature_columns: Vec<&Float64Array> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(i, f)| *i != target_idx && f.data_type() == &DataType::Float64)
        .map(|(i, _)| batch.column(i).as_primitive::<arrow::datatypes::Float64Type>())
        .collect();

    let n_features = feature_columns.len();
    let n_rows = batch.num_rows();
    let mut samples = Vec::with_capacity(n_rows);

    for row in 0..n_rows {
        let target = target_array.values()[row];
        if !target.is_finite() {
            continue;
        }
        let mut features = Vec::with_capacity(n_features);
        let mut has_non_finite = false;
        for col in &feature_columns {
            let v = col.values()[row];
            if !v.is_finite() {
                has_non_finite = true;
                break;
            }
            features.push(v);
        }
        if has_non_finite {
            continue;
        }
        samples.push((features, target));
    }

    Ok(samples)
}

/// Read all RecordBatches from a Parquet file.
///
/// # Errors
/// Returns error if the file cannot be opened or read.
#[cfg(feature = "parquet")]
pub fn read_parquet_batches(
    path: &std::path::Path,
) -> crate::error::Result<Vec<RecordBatch>> {
    let file = std::fs::File::open(path).map_err(|e| {
        crate::error::IrithyllError::Serialization(format!(
            "failed to open parquet file '{}': {}",
            path.display(),
            e
        ))
    })?;

    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| {
            crate::error::IrithyllError::Serialization(format!(
                "failed to create parquet reader for '{}': {}",
                path.display(),
                e
            ))
        })?
        .build()
        .map_err(|e| {
            crate::error::IrithyllError::Serialization(format!(
                "failed to build parquet reader for '{}': {}",
                path.display(),
                e
            ))
        })?;

    let mut batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            crate::error::IrithyllError::Serialization(format!(
                "failed to read parquet batch from '{}': {}",
                path.display(),
                e
            ))
        })?;
        batches.push(batch);
    }

    Ok(batches)
}

/// Stream-train an SGBT model from a Parquet file.
///
/// Reads the file in batches and trains incrementally, avoiding loading the
/// entire dataset into memory at once.
///
/// # Errors
/// Returns error if the file cannot be read or if the target column is invalid.
#[cfg(feature = "parquet")]
pub fn train_from_parquet(
    model: &mut crate::ensemble::SGBT,
    path: &std::path::Path,
    target_col: &str,
    loss: &dyn crate::loss::Loss,
) -> crate::error::Result<()> {
    let file = std::fs::File::open(path).map_err(|e| {
        crate::error::IrithyllError::Serialization(format!(
            "failed to open parquet file '{}': {}",
            path.display(),
            e
        ))
    })?;

    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| {
            crate::error::IrithyllError::Serialization(format!(
                "failed to create parquet reader for '{}': {}",
                path.display(),
                e
            ))
        })?
        .build()
        .map_err(|e| {
            crate::error::IrithyllError::Serialization(format!(
                "failed to build parquet reader for '{}': {}",
                path.display(),
                e
            ))
        })?;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            crate::error::IrithyllError::Serialization(format!(
                "failed to read parquet batch from '{}': {}",
                path.display(),
                e
            ))
        })?;
        train_from_record_batch(model, &batch, target_col, loss)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // Only run tests when the arrow feature is enabled.
    #[cfg(feature = "arrow")]
    mod arrow_tests {
        use super::super::*;
        use arrow::array::{Float64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        fn make_test_batch() -> RecordBatch {
            let schema = Arc::new(Schema::new(vec![
                Field::new("x1", DataType::Float64, false),
                Field::new("x2", DataType::Float64, false),
                Field::new("target", DataType::Float64, false),
            ]));
            let x1 = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
            let x2 = Arc::new(Float64Array::from(vec![0.5, 1.0, 1.5, 2.0, 2.5]));
            let target = Arc::new(Float64Array::from(vec![2.5, 5.0, 7.5, 10.0, 12.5]));
            RecordBatch::try_new(schema, vec![x1, x2, target]).unwrap()
        }

        #[test]
        fn train_from_batch_does_not_panic() {
            use crate::ensemble::config::SGBTConfig;
            use crate::loss::squared::SquaredLoss;

            let config = SGBTConfig::builder()
                .n_steps(5)
                .learning_rate(0.1)
                .grace_period(2)
                .build()
                .unwrap();
            let mut model = crate::ensemble::SGBT::new(config);
            let batch = make_test_batch();
            let loss = SquaredLoss;

            let result = train_from_record_batch(&mut model, &batch, "target", &loss);
            assert!(result.is_ok());
            assert_eq!(model.n_samples_seen(), 5);
        }

        #[test]
        fn predict_from_batch_returns_correct_count() {
            use crate::ensemble::config::SGBTConfig;

            let config = SGBTConfig::builder()
                .n_steps(3)
                .learning_rate(0.1)
                .grace_period(2)
                .build()
                .unwrap();
            let model = crate::ensemble::SGBT::new(config);

            // Create a batch without target column for prediction.
            let schema = Arc::new(Schema::new(vec![
                Field::new("x1", DataType::Float64, false),
                Field::new("x2", DataType::Float64, false),
            ]));
            let x1 = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
            let x2 = Arc::new(Float64Array::from(vec![0.5, 1.0, 1.5]));
            let batch = RecordBatch::try_new(schema, vec![x1, x2]).unwrap();

            let preds = predict_from_record_batch(&model, &batch);
            assert_eq!(preds.len(), 3);
            for i in 0..3 {
                assert!(preds.value(i).is_finite());
            }
        }

        #[test]
        fn missing_target_column_returns_error() {
            use crate::ensemble::config::SGBTConfig;
            use crate::loss::squared::SquaredLoss;

            let config = SGBTConfig::builder().n_steps(3).build().unwrap();
            let mut model = crate::ensemble::SGBT::new(config);
            let batch = make_test_batch();
            let loss = SquaredLoss;

            let result = train_from_record_batch(&mut model, &batch, "nonexistent", &loss);
            assert!(result.is_err());
        }

        #[test]
        fn record_batch_to_samples_works() {
            let batch = make_test_batch();
            let samples = record_batch_to_samples(&batch, "target").unwrap();
            assert_eq!(samples.len(), 5);
            assert_eq!(samples[0].0.len(), 2); // 2 feature columns
            assert!((samples[0].1 - 2.5).abs() < 1e-10); // first target
        }

        #[test]
        fn nan_inf_rows_are_skipped_in_training() {
            use crate::ensemble::config::SGBTConfig;
            use crate::loss::squared::SquaredLoss;

            let schema = Arc::new(Schema::new(vec![
                Field::new("x1", DataType::Float64, false),
                Field::new("target", DataType::Float64, false),
            ]));
            let x1 = Arc::new(Float64Array::from(vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0]));
            let target = Arc::new(Float64Array::from(vec![2.0, 4.0, f64::NAN, 8.0, 10.0]));
            let batch = RecordBatch::try_new(schema, vec![x1, target]).unwrap();

            let config = SGBTConfig::builder()
                .n_steps(3)
                .learning_rate(0.1)
                .grace_period(2)
                .build()
                .unwrap();
            let mut model = crate::ensemble::SGBT::new(config);
            let loss = SquaredLoss;

            let result = train_from_record_batch(&mut model, &batch, "target", &loss);
            assert!(result.is_ok());
            // Rows 0 (1.0, 2.0) and 4 (5.0, 10.0) are clean. Rows 1,2,3 have NaN/Inf.
            assert_eq!(model.n_samples_seen(), 2);
        }

        #[test]
        fn nan_inf_rows_are_skipped_in_samples() {
            let schema = Arc::new(Schema::new(vec![
                Field::new("x1", DataType::Float64, false),
                Field::new("target", DataType::Float64, false),
            ]));
            let x1 = Arc::new(Float64Array::from(vec![1.0, f64::NAN, 3.0]));
            let target = Arc::new(Float64Array::from(vec![2.0, 4.0, f64::NEG_INFINITY]));
            let batch = RecordBatch::try_new(schema, vec![x1, target]).unwrap();

            let samples = record_batch_to_samples(&batch, "target").unwrap();
            // Only row 0 is fully finite.
            assert_eq!(samples.len(), 1);
            assert!((samples[0].0[0] - 1.0).abs() < 1e-10);
            assert!((samples[0].1 - 2.0).abs() < 1e-10);
        }
    }
}
