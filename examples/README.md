# irithyll Examples

Examples are organized in four tiers by complexity. Start at tier 1 and go
as deep as your use case demands.

```
examples/
  01_quickstart/   -- first 5 minutes: zero-to-prediction
  02_essentials/   -- production patterns you ship
  03_neural/       -- advanced neural streaming architectures
  04_advanced/     -- power-user: AutoML, kernels, projections, custom loss
```

## Quick navigation

| If you want to...                              | Start here                                    |
|------------------------------------------------|-----------------------------------------------|
| See the smallest possible program              | `01_quickstart/hello_streaming`               |
| Understand the full SGBT API                   | `01_quickstart/basic_regression`              |
| Chain preprocessing + model                    | `02_essentials/pipeline_composition`          |
| Evaluate models with prequential protocol      | `02_essentials/streaming_metrics`             |
| Handle concept drift                           | `02_essentials/drift_detection`               |
| Do classification                              | `02_essentials/classification`                |
| Get confidence intervals from RLS              | `02_essentials/rls_confidence`                |
| Use the async / tokio training API             | `02_essentials/async_ingestion`               |
| Try a neural architecture (start here)         | `03_neural/kan_regression`                    |
| Run Mamba-3 temporal sequence learning         | `03_neural/mamba3_temporal`                   |
| Tune hyperparameters automatically             | `04_advanced/automl` or `04_advanced/factory_racing` |
| Learn a non-linear function with KRLS          | `04_advanced/krls_nonlinear`                  |
| Reduce dimensionality online                   | `04_advanced/ccipca_reduction`                |

## Running examples

```sh
cargo run --example hello_streaming
cargo run --example basic_regression
cargo run --example pipeline_composition
# ... and so on by name
```

All examples are self-contained. No external data files required.
