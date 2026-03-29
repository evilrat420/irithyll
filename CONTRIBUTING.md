# Contributing to irithyll

Thanks for considering contributing. irithyll is a streaming ML library in Rust, and we welcome contributions from researchers, engineers, and anyone interested in online learning.

## Getting started

```sh
git clone https://github.com/evilrat420/irithyll.git
cd irithyll
cargo test --lib          # run library tests (~1,200+)
cargo clippy --all-features -- -D warnings  # lint
cargo fmt --check         # formatting
```

The workspace has four crates:

| Crate | Purpose |
|-------|---------|
| `irithyll` | Main library -- algorithms, pipelines, I/O |
| `irithyll-core` | `no_std` training engine + zero-alloc inference |
| `irithyll-python` | PyO3 Python bindings |
| `irithyll-cli` | CLI + TUI for training and evaluation |

## Code standards

- **All CI must pass** before merge: tests, clippy (`-D warnings`), formatting, docs, MSRV (1.75)
- **Tests go in `#[cfg(test)] mod tests`** at the bottom of each file
- **Descriptive test names** in `snake_case` with assert messages explaining expected vs actual
- **No unsafe** in `irithyll` (the std crate). `irithyll-core` allows it only where verified necessary
- **Every algorithm cites its source paper** in doc comments and README references
- **No version references** in docs or source (e.g., "v9.1 introduces..."). Use changelog + releases

## Architecture principles

- **`StreamingLearner` is the universal interface.** Every algorithm implements `train_one` + `predict`. No exceptions
- **One sample at a time.** O(1) memory per model. No batches, no windows, no retraining
- **Composable.** Preprocessors + learners chain via `pipe()`. Models compose in `NeuralMoE` and `AutoTuner`
- **No external ML dependencies.** Pure Rust. Math is hand-derived, not autograd
- **Core is `no_std`.** If it can run on a Cortex-M0+ with 32KB SRAM, it goes in `irithyll-core`

## What we are looking for

- Bug fixes with regression tests
- New streaming algorithms (with paper references)
- Performance improvements (with benchmarks)
- Documentation improvements
- Real-world use case reports

## What we are not looking for

- Batch-only algorithms that don't implement `StreamingLearner`
- External ML framework dependencies (PyTorch, TensorFlow, ONNX Runtime)
- Features that break `no_std` compatibility in `irithyll-core`

## Commit style

```
feat: short description of the feature
fix: what was broken and how it's fixed
chore: maintenance (version bumps, CI, formatting)
docs: documentation changes
```

## Ethical use

Please read [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md). We build irithyll for research, education, healthcare, and environmental science -- not for weapons or surveillance.

## License

By contributing, you agree that your contributions will be licensed under the same MIT OR Apache-2.0 dual license as the project.
