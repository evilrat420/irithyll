# dev.to Article Outline

## Title
"Building a Streaming ML Platform in Rust: From One Paper to 12 Algorithms"

## Subtitle/Tagline
How irithyll grew from an SGBT implementation into a composable streaming machine learning toolkit

## Tags
#rust #machinelearning #opensource #algorithms

## Section Structure

### 1. The Problem (200 words)
- Batch ML assumes you have all the data upfront
- Real-world data streams continuously -- sensors, financial markets, user behavior
- When distributions shift (concept drift), batch models go stale
- You shouldn't need to retrain from scratch every time the world changes

### 2. The Streaming Paradigm (300 words)
- One sample at a time: `model.train(&features, target)`
- O(1) memory per model -- no data storage
- Hoeffding bound: statistically sound splits without storing the data
- Concept drift detection: automatically detect and adapt to distribution changes
- This isn't "mini-batch" or "incremental batch" -- it's true online learning

### 3. From Paper to Library (300 words)
- Started with Gunasekara et al., 2024 -- SGBT for evolving data streams
- The paper gives you the algorithm; production needs more:
  - EWMA leaf decay (gradual forgetting)
  - Lazy histogram decay (O(1) not O(n_bins))
  - Proactive tree replacement
  - Split re-evaluation at leaf boundaries
- These additions close the gap between research and deployable systems

### 4. The StreamingLearner Unification (300 words)
- One trait, twelve+ algorithms
- Show the trait: `train()`, `predict()`, `n_samples_seen()`
- Algorithm table (same as README)
- Key insight: streaming algorithms are more similar than different -- they all maintain sufficient statistics updated one sample at a time

### 5. Composable Pipelines (200 words)
- Code: `pipe(normalizer()).learner(sgbt(50, 0.01))`
- StreamingPreprocessor trait
- Chain multiple preprocessors
- The pipeline handles the plumbing -- you think about the algorithm

### 6. What Makes This Different (200 words)
- vs Python's River: Rust performance, type safety, async-native
- vs batch Rust ML (linfa, smartcore): true streaming, concept drift
- vs deep learning (burn, candle): classical ML algorithms, interpretable, lightweight
- Unique combination: streaming + composable + drift-aware + Rust

### 7. Performance & Production (200 words)
- SIMD histogram acceleration
- Rayon parallel training
- Arrow/Parquet I/O for data pipeline integration
- ONNX export for cross-platform deployment
- Async training with tokio
- 780+ tests, zero clippy warnings

### 8. Try It (100 words)
- `cargo add irithyll`
- Factory functions for quick start
- 11 examples covering every major feature
- Full docs on docs.rs with all features enabled

## Cover Image Concept
- ASCII-art style visualization of a streaming data pipeline
- Or: simple diagram showing data flowing through normalizer -> SGBT with predictions coming out

## Estimated Length
~2000 words

## Cross-posting
- Can be cross-posted to Hashnode and Medium
- Canonical URL should point to dev.to version
