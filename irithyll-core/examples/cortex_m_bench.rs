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

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::hprintln;
use panic_halt as _;

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
