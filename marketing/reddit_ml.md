# r/MachineLearning [P] Post Outline

## Post Type
[P] Project flair

## Title Options
- "[P] irithyll: Open-source streaming ML library implementing SGBT (Gunasekara et al., 2024), KRLS, Mondrian forests, and 9 more algorithms -- all true online learners in Rust"
- "[P] Streaming gradient boosted trees + kernel methods + composable pipelines in Rust -- irithyll v6.3"

## Opening Angle
Academic/research angle: Implementation of the SGBT paper (Gunasekara et al., 2024, Machine Learning journal) extended with practical streaming ML infrastructure.

## Key Points for ML Audience
1. **Paper implementation**: SGBT for evolving data streams with Hoeffding-bound splits, concept drift detection (Page-Hinkley, ADWIN, DDM)
2. **Beyond the paper**: EWMA leaf decay, lazy O(1) histogram decay, proactive tree replacement, EFDT-style split re-evaluation
3. **Algorithm breadth**: Not just trees -- KRLS with ALD sparsification (RBF/polynomial/linear kernels), RLS with confidence intervals, CCIPCA for streaming PCA, Gaussian NB, Mondrian forests
4. **Unified interface**: One `StreamingLearner` trait across all algorithms -- swap algorithms without changing pipeline code
5. **Streaming paradigm**: No batches, no windows. Each model processes one sample with O(1) memory. Handles concept drift natively.
6. **Prediction uncertainty**: RLS confidence intervals, DistributionalSGBT (mean + variance), adaptive conformal intervals

## Algorithm Table
| Algorithm | Paper/Method | Streaming? | Key Feature |
|-----------|-------------|------------|-------------|
| SGBT | Gunasekara et al., 2024 | Yes | Hoeffding trees + drift detection |
| KRLS | Engel et al., 2004 | Yes | ALD sparsification, kernel trick |
| RLS | Haykin, 2014 | Yes | Confidence intervals via P matrix |
| CCIPCA | Weng et al., 2003 | Yes | Covariance-free incremental PCA |
| Mondrian Forest | Lakshminarayanan et al., 2014 | Yes | Online random forests |
| Gaussian NB | -- | Yes | Streaming sufficient statistics |

## Concept Drift Section
- Three detectors: Page-Hinkley, ADWIN, DDM
- Automatic tree replacement when drift detected
- Importance drift monitoring via streaming SHAP
- This is the key differentiator: most ML libraries assume i.i.d. data

## Use Cases to Mention
- Real-time sensor/IoT data
- Financial time series with regime changes
- Network traffic anomaly detection
- Any domain where data distributions shift over time

## Links
- GitHub: https://github.com/evilrat420/irithyll
- Paper: https://doi.org/10.1007/s10994-024-06517-y (Gunasekara et al., 2024)
- docs.rs: https://docs.rs/irithyll

## Posting Strategy
- r/MachineLearning values technical depth -- lead with the algorithm, not the language
- Expect questions about benchmarks vs River (Python streaming ML) -- be honest, focus on Rust performance + type safety angle
- Mention the paper upfront for credibility
