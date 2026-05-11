//! Instruction-count regression bench for the packed-inference hot path.
//!
//! This bench uses [`iai-callgrind`](https://github.com/iai-callgrind/iai-callgrind)
//! to measure CPU instructions, branch counts, and cache behavior of
//! [`EnsembleView::predict`], [`EnsembleView::predict_batch`], and
//! [`EnsembleView::from_bytes`] on synthetic packed binaries. Unlike wall-clock
//! benches, instruction counts are deterministic across runs and machines, which
//! makes them suitable as a CI regression gate for embedded-class hot paths.
//!
//! # Platform constraints
//!
//! `iai-callgrind` runs each benchmark under [Valgrind](https://valgrind.org/)'s
//! callgrind tool. Valgrind is **Linux-only** (no Windows or macOS support).
//! On other platforms this file still **compiles** under `--features iai-bench`
//! (so CI on Windows/macOS catches API drift) but cannot be executed.
//!
//! # Building and running
//!
//! ```bash
//! # Compile-check (any platform):
//! cargo check -p irithyll-core --features iai-bench --bench packed_inference_iai
//!
//! # Run on Linux (requires valgrind installed):
//! cargo bench -p irithyll-core --features iai-bench --bench packed_inference_iai
//! ```
//!
//! # Interpreting results
//!
//! Each bench reports instructions retired, L1/LL data cache hits/misses, and
//! branch counts. A **regression** is any non-trivial increase in instructions
//! at fixed inputs. Treat absolute numbers as machine-relative; the regression
//! gate is *delta against the prior committed baseline*, which `iai-callgrind`
//! computes automatically and prints alongside each measurement.
//!
//! # Bench inventory
//!
//! - `single_predict` — 50 trees, depth 4, single-sample [`EnsembleView::predict`].
//! - `batch_predict` — 50 trees, depth 4, [`EnsembleView::predict_batch`] over 1000 samples.
//! - `deserialize_view` — 100 trees, depth 6, [`EnsembleView::from_bytes`] construction cost.

#![cfg(feature = "iai-bench")]

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use irithyll_core::packed::{EnsembleHeader, PackedNode, TreeEntry};
use irithyll_core::EnsembleView;
use std::hint::black_box;
use std::mem::size_of;

// ---------------------------------------------------------------------------
// Deterministic synthetic packed-binary builder
// ---------------------------------------------------------------------------
//
// The bench is self-contained inside `irithyll-core`: it does NOT depend on
// the main `irithyll` training pipeline. Instead it builds packed binaries
// directly from `PackedNode` / `TreeEntry` / `EnsembleHeader`, with a tiny
// xorshift64 PRNG to keep node thresholds and leaf values deterministic
// across runs.

/// Deterministic xorshift64 PRNG.
fn xorshift64(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32) / (u64::MAX as f32)
}

/// Cast a `repr(C)` value to its byte slice.
fn as_bytes<T: Sized>(val: &T) -> &[u8] {
    // SAFETY: `T` is `repr(C)` and we only read its bytes for the length of `T`.
    unsafe { core::slice::from_raw_parts(val as *const T as *const u8, size_of::<T>()) }
}

/// Build a complete binary tree of depth `depth` over `n_features`.
///
/// Layout: BFS index order. Internal node at index `i` has children
/// `2*i + 1` (left) and `2*i + 2` (right). Leaves occupy the last
/// `2^depth` slots.
fn build_complete_tree(depth: u8, n_features: u16, rng: &mut u64) -> Vec<PackedNode> {
    let n_internal = (1usize << depth) - 1;
    let n_leaves = 1usize << depth;
    let total = n_internal + n_leaves;
    let mut nodes = Vec::with_capacity(total);

    for i in 0..n_internal {
        let left = (2 * i + 1) as u16;
        let right = (2 * i + 2) as u16;
        let feat = (i as u16) % n_features;
        let threshold = xorshift64(rng) * 4.0 - 2.0;
        nodes.push(PackedNode::split(threshold, feat, left, right));
    }
    for _ in 0..n_leaves {
        let leaf_value = (xorshift64(rng) - 0.5) * 0.2;
        nodes.push(PackedNode::leaf(leaf_value));
    }
    nodes
}

/// Build a synthetic packed-ensemble binary with `n_trees` complete trees
/// of `depth`, each over `n_features`. Returns the buffer ready to feed
/// to [`EnsembleView::from_bytes`].
fn build_packed_buffer(n_trees: u16, depth: u8, n_features: u16) -> Vec<u8> {
    let mut rng: u64 = 0xC0FF_EE12_3456_7890;
    let nodes_per_tree = (1usize << (depth + 1)) - 1;
    let bytes_per_tree = nodes_per_tree * size_of::<PackedNode>();

    let header = EnsembleHeader {
        magic: EnsembleHeader::MAGIC,
        version: EnsembleHeader::VERSION,
        n_trees,
        n_features,
        _reserved: 0,
        base_prediction: 0.0,
    };

    let mut entries = Vec::with_capacity(n_trees as usize);
    for t in 0..n_trees as usize {
        entries.push(TreeEntry {
            n_nodes: nodes_per_tree as u32,
            offset: (t * bytes_per_tree) as u32,
        });
    }

    let mut all_nodes: Vec<PackedNode> = Vec::with_capacity(n_trees as usize * nodes_per_tree);
    for _ in 0..n_trees {
        all_nodes.extend(build_complete_tree(depth, n_features, &mut rng));
    }

    let mut buf = Vec::with_capacity(
        size_of::<EnsembleHeader>()
            + (n_trees as usize) * size_of::<TreeEntry>()
            + (n_trees as usize) * bytes_per_tree,
    );
    buf.extend_from_slice(as_bytes(&header));
    for e in &entries {
        buf.extend_from_slice(as_bytes(e));
    }
    for n in &all_nodes {
        buf.extend_from_slice(as_bytes(n));
    }
    buf
}

/// Generate `n` deterministic feature vectors of length `n_features`.
fn generate_samples(n: usize, n_features: usize) -> Vec<Vec<f32>> {
    let mut rng: u64 = 0xDEAD_BEEF_BAAD_F00D;
    (0..n)
        .map(|_| {
            (0..n_features)
                .map(|_| xorshift64(&mut rng) * 4.0 - 2.0)
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benches
// ---------------------------------------------------------------------------

// Single-sample predict, 50 trees, depth 4, 8 features.
// The model is built outside the measured region; only `predict` is timed.
#[library_benchmark]
fn single_predict() -> f32 {
    let buf = build_packed_buffer(50, 4, 8);
    let view = EnsembleView::from_bytes(&buf).expect("valid synthetic buffer");
    let sample: Vec<f32> = generate_samples(1, 8).pop().unwrap();
    black_box(view.predict(black_box(&sample)))
}

// Batch predict over 1000 samples, 50 trees, depth 4, 8 features.
// Exercises the x4-interleaved batch path. The buffer construction and
// sample generation are setup-only; the timed region is `predict_batch`.
#[library_benchmark]
fn batch_predict() -> f32 {
    let buf = build_packed_buffer(50, 4, 8);
    let view = EnsembleView::from_bytes(&buf).expect("valid synthetic buffer");
    let samples = generate_samples(1000, 8);
    let sample_refs: Vec<&[f32]> = samples.iter().map(|s| s.as_slice()).collect();
    let mut out = vec![0.0f32; 1000];
    view.predict_batch(black_box(&sample_refs), black_box(&mut out));
    // Return a value derived from `out` to keep the compiler from eliding it.
    black_box(out.iter().sum())
}

// View construction (parsing + validation) for a 100-tree, depth-6, 16-feature
// ensemble. This is the cold-start cost paid once per model load.
#[library_benchmark]
fn deserialize_view() -> usize {
    let buf = build_packed_buffer(100, 6, 16);
    let view = EnsembleView::from_bytes(black_box(&buf)).expect("valid synthetic buffer");
    black_box(view.total_nodes())
}

library_benchmark_group!(
    name = packed_inference;
    benchmarks = single_predict, batch_predict, deserialize_view
);

main!(library_benchmark_groups = packed_inference);
