# irithyll Benchmarks

Streaming ML benchmark comparisons for irithyll against established libraries.

## Methodology

All benchmarks use **prequential evaluation** (test-then-train): each sample is first
used for prediction (to measure accuracy), then used for training. This is the standard
evaluation protocol for online/streaming learners and reflects real-world deployment
where models must predict on unseen data before learning from it.

For batch learners (XGBoost, LightGBM), we simulate a realistic streaming deployment:
the model is periodically retrained on a sliding window of recent samples. This is how
batch models are typically deployed in production streaming scenarios.

Throughput is measured as **samples per second** (s/s) including both the predict and
train steps, reflecting end-to-end online learning speed.

## Evaluation Protocol

- **Metric: Accuracy** -- proportion of correct predictions
- **Metric: Cohen's Kappa** -- agreement corrected for chance (more informative under class imbalance)
- **Metric: Macro-F1** -- macro-averaged F1 score (used for multi-class datasets like Covertype)
- **Throughput** -- samples/second for the full train+predict loop
- **Protocol:** Prequential (test-then-train), single pass over the dataset

## Electricity (45K samples, binary, concept drift)

The [Electricity dataset](https://www.openml.org/d/151) is a standard benchmark for
concept drift in streaming ML. It contains 45,312 samples with 8 features, representing
electricity demand in New South Wales, Australia. The task is binary classification
(price up/down), and the data exhibits real-world concept drift.

- **Samples:** 45,312
- **Features:** 8
- **Task:** Binary classification
- **Drift:** Yes (real-world, non-stationary)

### Streaming Models

| Model | Library | Accuracy | Kappa | Throughput (s/s) |
|-------|---------|----------|-------|------------------|
| irithyll SGBT 25t d4 (lr=0.05) | irithyll | 0.7159 | 0.3709 | 67,063 |
| irithyll SGBT 50t d6 (lr=0.05) | irithyll | 0.8188 | 0.6155 | 16,347 |
| irithyll SGBT 50t d6 (lr=0.1) | irithyll | 0.8583 | 0.7041 | 19,011 |
| irithyll SGBT 100t d6 (lr=0.1) | irithyll | 0.8852 | 0.7619 | 8,184 |
| hoeffding_tree | River | 0.7956 | 0.5779 | 12,029 |
| hoeffding_adaptive_tree | River | 0.8293 | 0.6476 | 3,357 |
| arf_n10 | River | 0.8858 | 0.7652 | 534 |
| arf_n25 | River | 0.8913 | 0.7767 | 200 |

### Batch Models

| Model | Library | Window | Accuracy | Kappa | Throughput (s/s) |
|-------|---------|--------|----------|-------|------------------|
| xgb_w500 | XGBoost | 500 | 0.7637 | 0.5169 | 1,997 |
| xgb_w1000 | XGBoost | 1000 | 0.7542 | 0.4960 | 2,058 |
| xgb_w5000 | XGBoost | 5000 | 0.7053 | 0.4111 | 2,134 |
| lgbm_w500 | LightGBM | 500 | 0.7632 | 0.5155 | 1,434 |
| lgbm_w1000 | LightGBM | 1000 | 0.7572 | 0.5026 | 1,448 |
| lgbm_w5000 | LightGBM | 5000 | 0.7107 | 0.4234 | 1,483 |

## Airlines (539K samples, binary, large scale)

The [Airlines dataset](https://www.openml.org/d/1169) is a large-scale benchmark for
streaming classification. It contains 539,383 samples with 7 features, representing
US flight delay records. The task is binary classification (delayed/on-time), with
temporal concept drift from seasonal and operational changes.

- **Samples:** 539,383
- **Features:** 7
- **Task:** Binary classification
- **Drift:** Yes (temporal, seasonal patterns)

### Streaming Models

| Model | Library | Accuracy | Kappa | Throughput (s/s) |
|-------|---------|----------|-------|------------------|
| irithyll SGBT 50t d6 (lr=0.05) | irithyll | 0.6253 | 0.1802 | 9,222 |
| irithyll SGBT 50t d6 (lr=0.1) | irithyll | 0.6488 | 0.2449 | 9,054 |
| irithyll SGBT 100t d6 (lr=0.1) | irithyll | 0.6558 | 0.2684 | 4,094 |
| hoeffding_tree | River | 0.6383 | 0.2429 | 9,100 |
| hoeffding_adaptive_tree | River | 0.6348 | 0.2413 | 3,067 |
| arf_n10 | River | 0.6565 | 0.2895 | 448 |
| arf_n25 | River | 0.6675 | 0.3102 | 171 |

### Batch Models

| Model | Library | Window | Accuracy | Kappa | Throughput (s/s) |
|-------|---------|--------|----------|-------|------------------|
| xgb_w500 | XGBoost | 500 | 0.6216 | 0.2287 | 1,980 |
| xgb_w1000 | XGBoost | 1000 | 0.6299 | 0.2457 | 2,057 |
| xgb_w5000 | XGBoost | 5000 | 0.6317 | 0.2501 | 2,131 |
| lgbm_w500 | LightGBM | 500 | 0.6352 | 0.2532 | 1,425 |
| lgbm_w1000 | LightGBM | 1000 | 0.6460 | 0.2751 | 1,429 |
| lgbm_w5000 | LightGBM | 5000 | 0.6439 | 0.2738 | 1,419 |

## Covertype (581K samples, 7-class, high dimensionality)

The [Covertype dataset](https://www.openml.org/d/150) is a large-scale multi-class
benchmark. It contains 581,012 samples with 54 features (10 quantitative, 44 binary),
representing forest cover types in the Roosevelt National Forest. The task is 7-class
classification, testing a model's ability to handle high dimensionality and class imbalance.

- **Samples:** 581,012
- **Features:** 54
- **Task:** 7-class classification
- **Drift:** No (static distribution, but challenging multi-class)

### Streaming Models

| Model | Library | Accuracy | Kappa | Macro-F1 | Throughput (s/s) |
|-------|---------|----------|-------|----------|------------------|
| irithyll SGBT 50t d6 (lr=0.05) | irithyll | 0.8938 | 0.8265 | 0.8173 | 591 |
| irithyll SGBT 50t d6 (lr=0.1) | irithyll | 0.9247 | 0.8780 | 0.8710 | 584 |
| irithyll SGBT 100t d6 (lr=0.1) | irithyll | 0.9456 | 0.9122 | 0.9098 | 200 |
| hoeffding_tree | River | 0.7655 | 0.6186 | --- | 2,134 |
| hoeffding_adaptive_tree | River | 0.7731 | 0.6309 | --- | 687 |
| arf_n10 | River | 0.8727 | 0.7921 | --- | 461 |
| arf_n25 | River | 0.8858 | 0.8133 | --- | 207 |

### Batch Models

| Model | Library | Window | Accuracy | Kappa | Macro-F1 | Throughput (s/s) |
|-------|---------|--------|----------|-------|----------|------------------|
| xgb_w500 | XGBoost | 500 | 0.4988 | 0.2312 | --- | 2,176 |
| xgb_w1000 | XGBoost | 1000 | 0.4753 | 0.1735 | --- | 2,143 |
| xgb_w5000 | XGBoost | 5000 | 0.5931 | 0.3232 | --- | 2,079 |
| lgbm_w500 | LightGBM | 500 | 0.4596 | 0.1868 | --- | 1,434 |
| lgbm_w1000 | LightGBM | 1000 | 0.4856 | 0.1905 | --- | 1,443 |
| lgbm_w5000 | LightGBM | 5000 | 0.5979 | 0.3577 | --- | 1,428 |

## Summary

Best result per library across all datasets:

| Dataset | irithyll Best | River Best | XGBoost Best | LightGBM Best |
|---------|--------------|------------|--------------|---------------|
| Electricity | 88.5% (8K s/s) | 89.1% (200 s/s) | 76.4% (1K s/s) | 76.3% (1K s/s) |
| Airlines | 65.6% (4K s/s) | 66.8% (171 s/s) | 63.2% (2K s/s) | 64.6% (1K s/s) |
| Covertype | 94.6% (200 s/s) | 88.6% (207 s/s) | 59.3% (2K s/s) | 59.8% (1K s/s) |

## Learning Curves

Learning curve CSVs (accuracy over time) are available in `datasets/results/` for plotting:
- `electricity_learning_curve.csv`
- `airlines_learning_curve.csv`
- `covertype_learning_curve.csv`

## Hardware

> Results collected on:
>
> - **CPU:** AMD Ryzen 5 5500 (6C/12T, 3.6GHz base)
> - **RAM:** 16GB DDR4 3200MHz (2x8GB)
> - **OS:** Windows 11 Home 10.0.26200
> - **Rust:** stable (single-threaded, release profile)

## Reproducing

```bash
# 0. Download datasets
python datasets/download.py

# 1. Run irithyll benchmarks (Rust)
cargo bench --bench real_dataset_bench -- detailed

# 2. Run River benchmarks (Python)
python comparison/river/bench_river.py

# 3. Run XGBoost/LightGBM benchmarks (Python)
python comparison/xgboost/bench_xgb.py

# 4. Collect and generate BENCHMARKS.md
python comparison/collect_results.py
```

## Limitations

This comparison is provided for transparency and to help users make informed decisions.
We acknowledge the following limitations:

1. **Hyperparameter sensitivity.** Each model was tuned with reasonable but not
   exhaustive hyperparameter search. Better configurations may exist for all models.

2. **Apples-to-oranges on batch vs streaming.** Batch models (XGBoost, LightGBM) with
   sliding-window retraining have a fundamentally different compute profile than true
   streaming learners. Throughput comparisons across paradigms should be interpreted
   with this in mind.

3. **No ensemble/pipeline comparisons.** River supports rich pipelines
   (preprocessing + drift detection + model). We compare base models only.

4. **Single machine, single thread.** All benchmarks are single-threaded. Libraries
   with parallel prediction (e.g., LightGBM) may perform differently in multi-threaded
   settings.

5. **irithyll results are from Rust `cargo bench`.** River and XGBoost results are from
   Python. The language runtime difference is a real factor in throughput but not in
   accuracy/kappa.

6. **Dataset selection.** These three datasets are well-known streaming ML benchmarks,
   but they do not capture the full range of streaming ML challenges. Results may differ
   on other data distributions, feature types, or class structures.
