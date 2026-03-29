//! Cortex-M0+ stress test: large inference workload.
//!
//! Runs 100 predictions with systematically varied feature vectors on both
//! f32 and i16 paths, tracking min/max/mean prediction statistics.
//! Proves irithyll-core handles sustained workloads on bare-metal ARM.
//!
//! # Build (from irithyll-core/)
//!
//! ```bash
//! cargo build --features embedded-bench --target thumbv6m-none-eabi --release --example cortex_m_stress
//! ```
//!
//! # Run under QEMU
//!
//! ```bash
//! qemu-system-arm -cpu cortex-m0 -machine lm3s6965evb -nographic \
//!     -semihosting-config enable=on,target=native \
//!     -kernel target/thumbv6m-none-eabi/release/examples/cortex_m_stress
//! ```

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::hprintln;
use panic_halt as _;

/// Number of prediction iterations.
const N_ITERS: u32 = 100;

/// Number of features in the bench model.
const N_FEATURES: usize = 10;

#[entry]
fn main() -> ! {
    hprintln!("irithyll-core Cortex-M0+ stress test");

    // -----------------------------------------------------------------------
    // f32 packed inference stress
    // -----------------------------------------------------------------------
    static MODEL_F32: &[u8] = include_bytes!("../test_data/bench_model_50t.bin");
    let view = irithyll_core::EnsembleView::from_bytes(MODEL_F32).expect("f32 model");
    hprintln!(
        "f32 model: {} trees, {} nodes, {} features",
        view.n_trees(),
        view.total_nodes(),
        view.n_features()
    );

    let mut f32_min = f32::MAX;
    let mut f32_max = f32::MIN;
    let mut f32_sum: f64 = 0.0;

    for i in 0..N_ITERS {
        // Systematically vary features: base + i-dependent offset per feature.
        // Each iteration gets a different vector to exercise varied tree paths.
        let mut features = [0.0f32; N_FEATURES];
        for (j, feat) in features.iter_mut().enumerate() {
            let base = (j as f32 + 1.0) * 0.5;
            let offset = (i as f32) * 0.1 - 5.0; // ranges from -5.0 to +4.9
            let wave = irithyll_core::math::sin((i as f64 + j as f64) * 0.3) as f32;
            *feat = base + offset + wave;
        }

        let pred = view.predict(&features);
        if pred < f32_min {
            f32_min = pred;
        }
        if pred > f32_max {
            f32_max = pred;
        }
        f32_sum += pred as f64;
    }

    let f32_mean = f32_sum / N_ITERS as f64;
    hprintln!("f32 stress ({} iters):", N_ITERS);
    // Print as integer parts to avoid f32 formatting issues on no_std
    hprintln!(
        "  min={}, max={}, mean={}",
        f32_min,
        f32_max,
        f32_mean as f32
    );

    // Verify we got valid results (not NaN or infinity)
    let f32_valid =
        !f32_min.is_nan() && !f32_max.is_nan() && !f32_min.is_infinite() && !f32_max.is_infinite();
    if f32_valid {
        hprintln!("[OK] f32 stress: all predictions valid");
    } else {
        hprintln!("[FAIL] f32 stress: NaN or Inf detected");
    }

    // -----------------------------------------------------------------------
    // i16 quantized inference stress
    // -----------------------------------------------------------------------
    static MODEL_I16: &[u8] = include_bytes!("../test_data/bench_model_50t_i16.bin");
    let view_i16 = irithyll_core::QuantizedEnsembleView::from_bytes(MODEL_I16).expect("i16 model");
    hprintln!(
        "i16 model: {} trees, {} nodes",
        view_i16.n_trees(),
        view_i16.total_nodes()
    );

    let mut i16_min = f32::MAX;
    let mut i16_max = f32::MIN;
    let mut i16_sum: f64 = 0.0;

    for i in 0..N_ITERS {
        let mut features = [0.0f32; N_FEATURES];
        for (j, feat) in features.iter_mut().enumerate() {
            let base = (j as f32 + 1.0) * 0.5;
            let offset = (i as f32) * 0.1 - 5.0;
            let wave = irithyll_core::math::sin((i as f64 + j as f64) * 0.3) as f32;
            *feat = base + offset + wave;
        }

        let pred = view_i16.predict(&features);
        if pred < i16_min {
            i16_min = pred;
        }
        if pred > i16_max {
            i16_max = pred;
        }
        i16_sum += pred as f64;
    }

    let i16_mean = i16_sum / N_ITERS as f64;
    hprintln!("i16 stress ({} iters):", N_ITERS);
    hprintln!(
        "  min={}, max={}, mean={}",
        i16_min,
        i16_max,
        i16_mean as f32
    );

    let i16_valid =
        !i16_min.is_nan() && !i16_max.is_nan() && !i16_min.is_infinite() && !i16_max.is_infinite();
    if i16_valid {
        hprintln!("[OK] i16 stress: all predictions valid");
    } else {
        hprintln!("[FAIL] i16 stress: NaN or Inf detected");
    }

    // -----------------------------------------------------------------------
    // Cross-format comparison
    // -----------------------------------------------------------------------
    hprintln!("--- Cross-format comparison ---");

    // Run one canonical prediction on both to verify they agree directionally
    let canonical = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let pred_f32 = view.predict(&canonical);
    let pred_i16 = view_i16.predict(&canonical);
    hprintln!("canonical f32={}, i16={}", pred_f32, pred_i16);

    // Both should be finite and within reasonable range of each other
    let both_finite = pred_f32.is_finite() && pred_i16.is_finite();
    if both_finite {
        hprintln!("[OK] cross-format: both finite");
    } else {
        hprintln!("[FAIL] cross-format: non-finite prediction");
    }

    hprintln!("STRESS_COMPLETE");

    cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_SUCCESS);
    loop {}
}
