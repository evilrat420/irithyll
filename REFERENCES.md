# References

Complete bibliography for irithyll. Papers are organized by category.

---

## Section 1: Implemented Models

Papers with a direct code counterpart in the library.

### Core Gradient Boosting

- **SGBT** — Gunasekara, N., Pfahringer, B., Gomes, H. M., & Bifet, A. (2024). *Gradient boosted trees for evolving data streams.* Machine Learning, 113, 3325-3352. [DOI](https://doi.org/10.1007/s10994-024-06517-y) → `src/ensemble/sgbt.rs`

- **TreeSHAP** — Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Banber, N., & Lee, S.-I. (2020). *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2, 56-67. → `src/explain/`

- **CCIPCA** — Weng, J., Zhang, Y., & Hwang, W.-S. (2003). *Candid covariance-free incremental principal component analysis.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 25(8), 1034-1040. → `src/preprocessing/ccipca.rs`

### Plasticity and Continual Learning

- **Proactive pruning / ContinualLearner** — Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., & Sutton, R. S. (2024). *Loss of plasticity in deep continual learning.* Nature, 632, 768-774. → `src/ensemble/sgbt.rs` (proactive_prune), `src/continual/`

- **Elastic Weight Consolidation** — Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS, 114(13), 3521-3526. → `src/continual/` (EWC strategy)

### Reservoir Computing

- **NextGenRC** — Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021). *Next generation reservoir computing.* Nature Communications, 12, 5564. → `src/reservoir/ngrc.rs`

- **EchoStateNetwork** — Rodan, A., & Tino, P. (2010). *Minimum complexity echo state network.* IEEE Transactions on Neural Networks, 23(1), 131-144. → `src/reservoir/esn.rs`

- **EchoStateNetwork** — Martinuzzi, F. (2025). *Minimal deterministic echo state networks outperform random reservoirs in learning chaotic dynamics.* Chaos, 35. → `src/reservoir/esn.rs`

### State Space Models

- **StreamingMamba (V1)** — Gu, A., & Dao, T. (2023). *Mamba: Linear-time sequence modeling with selective state spaces.* arXiv:2312.00752. → `src/ssm/mamba.rs`

- **S4D-Inv initialization** — Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). *On the parameterization and initialization of diagonal state space models.* NeurIPS 2022. → `src/ssm/mamba.rs` (A-matrix init)

- **StreamingMambaV3 / SSM duality** — Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality.* arXiv:2405.21060. → `src/ssm/mamba_v3.rs`

- **Mamba-3 SelectiveSSMv3** — Lahoti, A., Li, K. Y., Chen, B., Wang, C., Bick, A., Kolter, J. Z., Dao, T., & Gu, A. (2026). *Mamba-3: Improved sequence modeling using state space principles.* ICLR 2026. → `src/ssm/v3/` (SelectiveSSMv3Exp, SelectiveSSMv3Mimo)

### Spiking Neural Networks

- **SpikeNet (e-prop)** — Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). *A solution to the learning dilemma for recurrent networks of spiking neurons.* Nature Communications, 11, 3625. → `src/snn/spikenet.rs`

- **SpikeNet (surrogate gradients)** — Neftci, E. O., Mostafa, H., & Zenke, F. (2019). *Surrogate gradient learning in spiking neural networks.* IEEE Signal Processing Magazine, 36(6), 51-63. → `src/snn/spikenet.rs`

### Test-Time Training

- **StreamingTTT** — Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Chang, Y., Shen, S., Dong, L., Lu, K., Zhai, X., Keutzer, K., & Darrell, T. (2024). *Learning to (Learn at Test Time): RNNs with expressive hidden states.* ICML 2025. → `src/ttt/`

- **StreamingTTT (momentum + weight decay)** — Behrouz, A., Zhong, P., & Mirrokni, V. (2025). *Titans: Learning to memorize at test time.* arXiv:2501.00663. → `src/ttt/` (Titans extensions)

### Kolmogorov-Arnold Networks

- **StreamingKAN** — Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljacic, M., Hou, T. Y., & Tegmark, M. (2024). *KAN: Kolmogorov-Arnold Networks.* ICLR 2025. → `src/kan/`

- **StreamingKAN (online convergence)** — Hoang, T. T., et al. (2026). *Ultrafast on-chip online learning via Kolmogorov-Arnold Networks.* arXiv:2602.02056. → `src/kan/`

### Streaming Linear Attention

- **GLA mode** — Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2023). *Gated linear attention transformers with hardware-efficient training.* arXiv:2312.06635. → `src/attention/` (GLA)

- **DeltaNet mode** — Yang, S., et al. (2024). *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464. → `src/attention/` (DeltaNet)

- **Log-Linear Attention mode** — Han, S., Lin, Y., Sun, Y., Tsvetkov, Y., Pham, H., Sankaranarayanan, A., Awadalla, H., Khabsa, M., & Yang, S. (2026). *Log-Linear Attention.* ICLR 2026. → `src/attention/` (LogLinear)

- **RWKV mode** — Peng, B., et al. (2024). *Eagle and Finch: RWKV with matrix-valued states and dynamic recurrence.* arXiv:2404.05892. → `src/attention/` (RWKV)

- **mLSTM mode / sLSTM** — Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). *xLSTM: Extended long short-term memory.* NeurIPS 2024. → `src/attention/` (mLSTM), `src/snn/` (sLSTM)

- **RetNet mode** — Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., & Wei, F. (2023). *Retentive network: A successor to transformer for large language models.* arXiv:2307.08621. → `src/attention/` (RetNet)

- **Hawk/Griffin mode** — De, S., Smith, S. L., Fernando, A., Botev, A., Cristian-Muraru, G., Gu, A., Harber, R., Kadous, L., Karaletsos, T., Lewis, A., Zhu, D., De Freitas, N., Doucet, A., & Buchatskaya, E. (2024). *Griffin: Mixing gated linear recurrences with local attention for efficient language models.* arXiv:2402.19427. → `src/attention/` (Hawk)

### Neural MoE

- **NeuralMoE (foundations)** — Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). *Adaptive mixtures of local experts.* Neural Computation, 3(1), 79-87. → `src/moe/`

- **NeuralMoE (top-k routing)** — Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). *Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.* ICLR 2017. → `src/moe/`

- **NeuralMoE (load balancing)** — Wang, B., et al. (2024). *Auxiliary-loss-free load balancing strategy for mixture-of-experts.* arXiv:2408.15664. → `src/moe/router.rs`

- **NeuralMoE (concept drift)** — Aspis, M., et al. (2025). *DriftMoE: Mixture of experts for streaming classification with concept drift.* ECMLPKDD 2025. → `src/moe/`

### Streaming AutoML

- **AutoTuner (tournament racing)** — Wu, Q., Iyer, C., & Wang, C. (2021). *ChaCha for online AutoML.* ICML 2021. → `src/automl/`

- **DiscountedThompsonSampling** — Qi, Y., et al. (2023). *Discounted Thompson Sampling for non-stationary bandits.* arXiv:2305.10718. → `src/automl/bandit.rs`

- **Complexity-adjusted elimination** — Yamanishi, K. (2018). *Stochastic complexity for online learning with finite-state models.* IEEE Transactions on Information Theory. → `src/automl/racing.rs`

### Conformal Prediction

- **ConformalPID** — Angelopoulos, A. N., Candes, E. J., & Tibshirani, R. J. (2023). *Conformal PID control for time series prediction.* NeurIPS 2023. → `src/metrics/conformal.rs`

- **Strongly adaptive conformal** — Bhatnagar, A., Wang, H., Xiong, C., & Bai, Y. (2023). *Improved online conformal prediction via strongly adaptive online learning.* ICML 2023. → `src/metrics/conformal.rs`

- **OnlinePlattScaling** — Gupta, C., & Ramdas, A. (2023). *Online Platt scaling with calibeating.* ICML 2023. → `src/metrics/calibration.rs`

### Projection Learning

- **SubspaceTracker (PAST)** — Yang, B. (1995). *Projection approximation subspace tracking.* IEEE Transactions on Signal Processing, 43(1), 95-107. → `src/projection/subspace.rs`

### Mathematical Foundations

- **Banach Fixed-Point Theorem** — Banach, S. (1922). *Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales.* Fundamenta Mathematicae, 3, 133-181. → `src/automl/adaptation_bus.rs` (contraction inequality enforcement)

---

## Section 2: Algorithmic Foundations

Papers that inform core algorithms without mapping to a single named struct.

- Gunasekara, N., et al. (2024) → SGBT architecture and drift-replacement cycle (primary implementation paper, see above)

- Bifet, A., & Gavalda, R. (2007). *Learning from time-changing data with adaptive windowing.* SIAM SDM 2007. → `drift::adwin::Adwin` (ADWIN algorithm)

- Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004). *Learning with drift detection.* SBIA 2004. → `drift::ddm::DDM`

- Page, E. S. (1954). *Continuous inspection schemes.* Biometrika, 41(1-2), 100-115. → `drift::page_hinkley::PageHinkley`

- Domingos, P., & Hulten, G. (2000). *Mining high-speed data streams.* KDD 2000. → Hoeffding tree splitting criterion in `tree/`

- Lunde, R., Kleppe, T. S., & Skaug, H. J. (2020). *An information criterion for automatic gradient tree boosting.* arXiv:2008.05926. → per-split information criterion in `tree/`

- Rahimi, A., & Recht, B. (2007). *Random features for large-scale kernel machines.* NIPS 2007. → `src/ssm/v3/readout_lift.rs` (random feature network lift)

---

## Section 3: Related Work

Papers that influenced design but are not directly implemented as named models in irithyll. Moved from README.

- Jaeger, H. (2001). *The "echo state" approach to analysing and training recurrent neural networks.* GMD Report 148. — Original ESN paper. irithyll uses the cycle/ring topology variant, not the random matrix original.

- Lukoševičius, M., & Jaeger, H. (2009). *Reservoir computing approaches to recurrent neural network training.* Computer Science Review, 3(3), 127-149. — Survey. Foundational reading for understanding reservoir computing variants.

- Sussillo, D., & Abbott, L. F. (2009). *Generating coherent patterns of activity from chaotic neural networks.* Neuron, 63(4), 544-557. — FORCE learning. Related to online RNN training; not directly implemented.

- Yan, M., Huang, C., Bienstman, P., Tino, P., Lin, W., & Sun, J. (2024). *Emerging opportunities and challenges for the future of reservoir computing.* Nature Communications, 15, 2056. — Survey of RC landscape.

- Gu, A., Goel, K., & Ré, C. (2021). *Efficiently modeling long sequences with structured state spaces.* arXiv:2111.00396. — Original S4 paper. irithyll uses Mamba (selective SSM), not S4.

- Kleyko, D., Frady, E. P., Kheffache, M., & Osipov, E. (2020). *Integer echo state networks: Efficient reservoir computing for digital hardware.* IEEE TNNLS. — Integer ESN. Informed SpikeNetFixed fixed-point design; not a direct implementation.

- Zenke, F., & Ganguli, S. (2018). *SuperSpike: Supervised learning in multilayer spiking neural networks.* Neural Computation, 30(6), 1514-1541. — SNN learning. Related to e-prop; irithyll uses Bellec 2020 e-prop directly.

- Eshraghian, J. K., Ward, M., Neftci, E. O., et al. (2023). *Training spiking neural networks using lessons from deep learning.* Proceedings of the IEEE, 111(9), 1016-1054. — SNN training survey. Background reading.

- Frenkel, C., & Indiveri, G. (2022). *ReckOn: A 28nm sub-mm² task-agnostic spiking recurrent neural network processor.* ISSCC 2022. — Neuromorphic hardware. Motivated the 22KB SpikeNetFixed target; not a software implementation.

- Meyer, S. M., et al. (2024). *Diagonal state space model on Loihi 2 for efficient streaming.* arXiv:2409.15022. — SSM on neuromorphic hardware. Related to embedded deployment goals.

- Jaeger, H., Lukoševičius, M., Popovici, D., & Siewert, U. (2007). *Optimization and applications of echo state networks with leaky-integrator neurons.* Neural Networks, 20(3), 335-352. — Leaky ESN. Informs the leaky integration coefficient in EchoStateNetwork.

- Javed, K., Shah, H., Sutton, R., & White, M. (2023). *Scalable real-time recurrent learning.* arXiv:2302.05326. — RTRL/GROUSE. Related to online projection; irithyll uses PAST (Yang 1995) for SubspaceTracker.

- Liu, B., Wang, R., Wu, L., Feng, Y., Stone, P., & Liu, Q. (2024). *Longhorn: State space models are amortized online learners.* NeurIPS 2024. — SSM as online learners. Thematically related to StreamingMamba but distinct architecture.
