# Contributing to irithyll

## Building and Testing

```sh
# Run library tests (fast, ~2 min)
cargo test -p irithyll --lib
cargo test -p irithyll-core --lib

# Run specific module tests
cargo test -p irithyll --lib -- ensemble
cargo test -p irithyll --lib -- kan

# Full test suite (slow, ~20 min — includes neural benchmarks)
cargo test --workspace

# CI checks
cargo fmt --check
cargo clippy --workspace --lib --tests -- -D warnings
RUSTDOCFLAGS="--cfg docsrs -Dwarnings" cargo doc --no-deps --all-features
```

## CI Contract

All changes must pass: formatting, clippy, documentation build, and tests on stable Rust.

## Adding a New StreamingLearner

1. Implement the `StreamingLearner` trait with all relevant method overrides
2. Add a factory constructor in `src/automl/factories.rs`
3. Add a convenience function in `src/lib.rs`
4. Add to `tests/compliance_harness.rs`
5. Document with examples and paper reference
6. Update `CHANGELOG.md`

## Commit Style

- `feat:` new features
- `fix:` bug fixes
- `chore:` maintenance, dependencies, documentation
