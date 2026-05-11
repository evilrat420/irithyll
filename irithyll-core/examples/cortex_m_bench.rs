//! Cortex-M0+ embedded inference benchmark.
//!
//! Proves irithyll-core runs on bare-metal ARM (no_std, no alloc, no FPU).
//! Both f32 packed and i16 quantized inference on Cortex-M0+.
//!
//! # Build (from irithyll-core/)
//!
//! ```bash
//! cargo build --features embedded-bench --target thumbv6m-none-eabi --release --example cortex_m_bench
//! ```
//!
//! # Run under QEMU
//!
//! ```bash
//! qemu-system-arm -cpu cortex-m0 -machine lm3s6965evb -nographic \
//!     -semihosting-config enable=on,target=native \
//!     -kernel target/thumbv6m-none-eabi/release/examples/cortex_m_bench
//! ```
//!
//! # TODO: Extended embedded benchmarks
//!
//! The following benchmarks are planned but require cross-compilation setup
//! (thumbv6m-none-eabi / thumbv7em-none-eabihf toolchain + QEMU ARM).
//!
//! ## Model size sweep (tree ensembles)
//!
//! Benchmark packed inference latency at three ensemble sizes:
//! - Small: 10 trees, depth 3 (bench_model_10t.bin)
//! - Medium: 50 trees, depth 4 (bench_model_50t.bin, current)
//! - Large: 100 trees, depth 5 (bench_model_100t.bin)
//!
//! Expected: latency scales ~linearly with n_trees * avg_depth.
//! Pack new test_data binaries with `export_packed` + `export_packed_i16`.
//!
//! ## Step-count throughput (10, 50, 100 prediction steps)
//!
//! Loop the prediction call N times and report total cycles / N:
//! - 10 steps: single-sample latency baseline
//! - 50 steps: representative online inference burst
//! - 100 steps: sustained throughput with instruction cache warmed up
//!
//! Use `cortex_m::asm::nop()` fences between loops to prevent DCE.
//!
//! ## Cortex-M4 FPU comparison (thumbv7em-none-eabihf)
//!
//! Re-run f32 packed inference on M4 with hardware FPU enabled.
//! Expected: ~2-4x speedup on tree walks (FPU fmul, fmadd pipelining).
//! Use qemu-system-arm with `-cpu cortex-m4` and the lm3s6965evb machine.
//!
//! ## TurboQuant weight inference on M0+
//!
//! Benchmark `TurboQuantizedView::predict_with_scratch` on ARM:
//! - 64-weight vector (d_model=64 RLS readout), 3.5-bit mode
//! - Measure cycles for FWHT rotation + base-11 unpack + dot product
//! - Compare against raw f32 dot product of same vector
//!   Requires: static `PACKED_WEIGHTS: &[u8]` embedded via `include_bytes!`.

#![cfg_attr(target_arch = "arm", no_std)]
#![cfg_attr(target_arch = "arm", no_main)]

#[cfg(target_arch = "arm")]
use cortex_m_rt::entry;
#[cfg(target_arch = "arm")]
use cortex_m_semihosting::hprintln;
#[cfg(target_arch = "arm")]
use panic_halt as _;

#[cfg(target_arch = "arm")]
#[entry]
fn main() -> ! {
    hprintln!("irithyll-core Cortex-M0+ benchmark");

    // f32 packed inference
    static MODEL_F32: &[u8] = include_bytes!("../test_data/bench_model_50t.bin");
    let view = irithyll_core::EnsembleView::from_bytes(MODEL_F32).expect("model");
    hprintln!(
        "f32: {} trees, {} nodes, {} bytes",
        view.n_trees(),
        view.total_nodes(),
        MODEL_F32.len()
    );

    let features = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let pred_f32 = view.predict(&features);
    hprintln!("f32 predict: {}", pred_f32);

    // i16 quantized inference
    static MODEL_I16: &[u8] = include_bytes!("../test_data/bench_model_50t_i16.bin");
    let view_i16 = irithyll_core::QuantizedEnsembleView::from_bytes(MODEL_I16).expect("i16");
    hprintln!(
        "i16: {} trees, {} nodes, {} bytes",
        view_i16.n_trees(),
        view_i16.total_nodes(),
        MODEL_I16.len()
    );

    let pred_i16 = view_i16.predict(&features);
    hprintln!("i16 predict: {}", pred_i16);

    hprintln!("BENCH_COMPLETE");

    cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_SUCCESS);
    loop {}
}

// Stub for non-ARM hosts (x86 CI, cargo check --all-targets).
#[cfg(not(target_arch = "arm"))]
fn main() {}
