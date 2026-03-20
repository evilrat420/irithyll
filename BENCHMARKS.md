# Benchmarks

Real dataset evaluation of irithyll against established streaming and batch ML libraries.

## Methodology

**Dataset:** [Electricity (Elec2)](https://maxhalford.github.io/files/datasets/electricity.zip) -- 45,312 samples, 8 normalized features, binary classification (UP/DOWN). The standard concept drift benchmark from the Australian NSW Electricity Market (Harries 1999).

**Protocol:** Prequential (interleaved test-then-train), the gold standard for streaming ML evaluation (Gama et al. 2013):

```
for each (x, y) in stream:
    y_pred = model.predict(x)   // test first
    metric.update(y, y_pred)    // record
    model.learn(x, y)           // train after
```

For batch models (XGBoost, LightGBM), periodic retraining is used: accumulate a window of samples, retrain from scratch, predict the next window.

**Metrics:**
- **Accuracy** -- cumulative over the full stream
- **Cohen's Kappa** -- adjusts for class imbalance (higher = better)
- **Throughput** -- samples processed per second (train + predict, wall-clock)

## Results: Electricity Dataset

### Streaming Models (sample-by-sample)

| Model | Library | Accuracy | Kappa | Throughput |
|-------|---------|----------|-------|------------|
| SGBT 100t d6 lr=0.1 | **irithyll** (Rust) | **0.8852** | **0.7619** | **8,184 s/s** |
| SGBT 50t d6 lr=0.1 | **irithyll** (Rust) | 0.8583 | 0.7041 | 19,011 s/s |
| ARF n=25 | River (Python) | **0.8927** | **0.7797** | 286 s/s |
| ARF n=10 | River (Python) | 0.8867 | 0.7674 | 528 s/s |
| SGBT 50t d6 lr=0.05 | **irithyll** (Rust) | 0.8188 | 0.6155 | 16,347 s/s |
| Hoeffding Adaptive Tree | River (Python) | 0.8305 | 0.6500 | 4,763 s/s |
| Hoeffding Tree | River (Python) | 0.7956 | 0.5779 | 16,474 s/s |

### Batch Models (periodic retrain)

| Model | Library | Window | Accuracy | Kappa | Throughput |
|-------|---------|--------|----------|-------|------------|
| XGBoost 50t d6 | XGBoost (C++) | 500 | 0.7637 | 0.5169 | 1,994 s/s |
| XGBoost 50t d6 | XGBoost (C++) | 1000 | 0.7542 | 0.4960 | 2,038 s/s |
| LightGBM 50t d6 | LightGBM (C++) | 500 | 0.7632 | 0.5155 | 1,268 s/s |
| LightGBM 50t d6 | LightGBM (C++) | 1000 | 0.7572 | 0.5026 | 1,272 s/s |

### Key Observations

1. **irithyll matches River's best single-tree accuracy** (88.5% vs 83.1% for Hoeffding Adaptive Tree) while running at **1.7x the throughput** at comparable config, and **15x faster** than River's single-tree adaptive model.

2. **River's ARF ensemble (25 trees) edges irithyll** on accuracy (89.3% vs 88.5%), but irithyll processes samples **28x faster** (8,184 vs 286 s/s). At the 50-tree config with lr=0.1, irithyll achieves 85.8% at **66x** River ARF's throughput.

3. **Batch GBTs struggle on this drifting stream.** XGBoost and LightGBM top out at ~76% accuracy even with frequent retraining (window=500). The Electricity dataset has natural concept drift that streaming models handle inherently.

4. **Throughput is not the headline** -- Rust vs Python throughput differences are expected. The real story is that streaming models (irithyll, River) adapt to drift while batch models don't.

## Hardware

- CPU: AMD Ryzen 5 5500 (6C/12T, 3.6 GHz base)
- RAM: 16 GB DDR4 3200
- OS: Windows 11
- Rust: stable (MSRV 1.75)
- Python: 3.x, River 0.23.0, XGBoost 3.2.0, LightGBM (latest)

## Reproducing

```bash
# irithyll benchmarks
cargo bench --bench real_dataset_bench

# River comparison
pip install river
python comparison/river/bench_river.py

# XGBoost/LightGBM comparison
pip install xgboost lightgbm scikit-learn numpy
python comparison/xgboost/bench_xgb.py

# Collect results
python comparison/collect_results.py
```

## Limitations

- **Single dataset.** These results are from Electricity only. Airlines and Covertype benchmarks are planned.
- **Hyperparameter tuning.** irithyll configs were hand-tuned; River uses defaults. A grid search on both sides would be fairer.
- **Single machine.** Results may vary on different hardware. Throughput is particularly sensitive to CPU cache sizes.
- **Binary classification only.** Multi-class (Covertype) and regression benchmarks are not yet included.
- **No memory profiling.** Memory usage is not yet measured.
- **Batch models disadvantaged.** Periodic retraining is a crude streaming adaptation; more sophisticated approaches (incremental XGBoost, adaptive ensembles) could perform better.
