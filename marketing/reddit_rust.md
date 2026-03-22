# r/rust Show Post Outline

## Post Type
Text post, "Show" flair if available

## Title Options (pick one)
- "irithyll: 12+ streaming ML algorithms in pure Rust -- gradient boosted trees, kernel methods, pipelines, all learning one sample at a time"
- "Show r/rust: irithyll -- streaming machine learning with composable pipelines, concept drift adaptation, and zero unsafe"

## Opening Hook
Personal story angle: Built this as a streaming alternative to batch ML for real-time systems.
The problem: existing Rust ML crates (linfa, smartcore) are batch-oriented. If your data arrives continuously and the distribution shifts, you need models that learn incrementally without storing past data.

## Key Selling Points (in order of r/rust appeal)
1. **Pure Rust, zero unsafe** -- the entire library is safe Rust
2. **One unified trait** -- `StreamingLearner` with `.train()` and `.predict()` for all 12+ algorithms
3. **Composable pipelines** -- `pipe(normalizer()).learner(sgbt(50, 0.01))` -- this is the visual hook
4. **Concept drift adaptation** -- models automatically replace degraded components when data shifts
5. **780+ tests, zero clippy warnings** -- production quality signal
6. **Factory functions** -- `sgbt()`, `krls()`, `rls()`, `gaussian_nb()`, `mondrian()` -- Rust ergonomics
7. **Feature-gated acceleration** -- SIMD, Rayon parallel, Arrow/Parquet, ONNX export
8. **Async-native** -- tokio channels with concurrent prediction

## Code Snippet (the hook)
```rust
use irithyll::{pipe, normalizer, sgbt, StreamingLearner};

let mut model = pipe(normalizer()).learner(sgbt(50, 0.01));
model.train(&[100.0, 0.5], 42.0);
let prediction = model.predict(&[100.0, 0.5]);
```

## Differentiators vs Existing Crates
- **vs linfa**: linfa is batch-only. irithyll is true streaming -- O(1) memory per model, continuous learning.
- **vs smartcore**: smartcore is batch. Also less actively maintained.
- **vs burn/candle**: These are deep learning frameworks. irithyll is traditional ML algorithms made streaming.
- **Unique**: Concept drift detection, streaming PCA, kernel methods, prediction confidence intervals -- none of these exist in the Rust ecosystem as streaming implementations.

## Algorithm Count Breakdown
SGBT (+ 5 variants), KRLS, RLS, StreamingLinear, StreamingPolynomial, GaussianNB, MondrianForest, LocallyWeightedRegression, plus CCIPCA/IncrementalNormalizer/OnlineFeatureSelector preprocessors, HalfSpaceTree anomaly detection

## Stats to Include
- 780+ tests
- 85 doc tests
- 11 examples
- Zero clippy warnings
- Pure Rust, no unsafe
- MIT/Apache-2.0 dual license

## Links
- GitHub: https://github.com/evilrat420/irithyll
- crates.io: https://crates.io/crates/irithyll
- docs.rs: https://docs.rs/irithyll

## Posting Strategy
- **Best days**: Tuesday, Wednesday, Thursday
- **Best time**: 9-11am EST (most active)
- **Reply strategy**: Reply to every comment in the first 2 hours. Be genuine, technical, and appreciative.
- **Don't**: Over-sell, compare negatively to other crates, claim it's better than X. Do: explain the streaming paradigm, share what you learned building it, engage with technical questions.
