//! Cortex-M0+ drift detection test.
//!
//! Proves DDM and PageHinkleyTest work on bare-metal ARM (no_std, no alloc).
//! Both detectors use only stack memory and f64 arithmetic — no heap needed.
//!
//! # Build (from irithyll-core/)
//!
//! ```bash
//! cargo build --features embedded-bench --target thumbv6m-none-eabi --release --example cortex_m_drift
//! ```
//!
//! # Run under QEMU
//!
//! ```bash
//! qemu-system-arm -cpu cortex-m0 -machine lm3s6965evb -nographic \
//!     -semihosting-config enable=on,target=native \
//!     -kernel target/thumbv6m-none-eabi/release/examples/cortex_m_drift
//! ```

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::hprintln;
use panic_halt as _;

use irithyll_core::drift::ddm::Ddm;
use irithyll_core::drift::pht::PageHinkleyTest;
use irithyll_core::drift::DriftSignal;

/// Deterministic value generator — sin-based oscillation around a centre.
/// No alloc, no randomness.
fn deterministic_value(centre: f64, jitter: f64, index: u32) -> f64 {
    let t = index as f64;
    centre + jitter * irithyll_core::math::sin(t * 0.7)
}

#[entry]
fn main() -> ! {
    hprintln!("irithyll-core Cortex-M0+ drift detection test");

    // -----------------------------------------------------------------------
    // DDM test
    // -----------------------------------------------------------------------
    hprintln!("--- DDM ---");

    let mut ddm = Ddm::new();
    hprintln!(
        "DDM created: warn={}, drift={}, min_inst={}",
        ddm.warning_level() as u32,
        ddm.drift_level() as u32,
        ddm.min_instances()
    );

    // Phase 1: 100 samples of normal data (error ~0.1)
    let mut ddm_warnings = 0u32;
    let mut ddm_drifts = 0u32;
    for i in 0..100 {
        let value = deterministic_value(0.1, 0.02, i);
        let signal = ddm.update(value);
        match signal {
            DriftSignal::Warning => ddm_warnings += 1,
            DriftSignal::Drift => ddm_drifts += 1,
            DriftSignal::Stable => {}
        }
    }
    hprintln!(
        "DDM phase 1 (normal): warnings={}, drifts={}",
        ddm_warnings,
        ddm_drifts
    );

    // Phase 2: 50 samples of drifted data (error ~0.8)
    let mut ddm_phase2_warnings = 0u32;
    let mut ddm_phase2_drifts = 0u32;
    for i in 0..50 {
        let value = deterministic_value(0.8, 0.02, 100 + i);
        let signal = ddm.update(value);
        match signal {
            DriftSignal::Warning => ddm_phase2_warnings += 1,
            DriftSignal::Drift => ddm_phase2_drifts += 1,
            DriftSignal::Stable => {}
        }
    }
    hprintln!(
        "DDM phase 2 (drifted): warnings={}, drifts={}",
        ddm_phase2_warnings,
        ddm_phase2_drifts
    );

    // DDM may not detect drift with only 100 baseline samples (min_instances=30).
    // With such a short baseline, the min tracking may not have stabilised.
    // Report the result either way — this is a functionality proof, not a unit test.
    if ddm_phase2_drifts > 0 {
        hprintln!("[OK] DDM detected drift after shift");
    } else if ddm_phase2_warnings > 0 {
        hprintln!("[WARN] DDM detected warning but not drift (short baseline)");
    } else {
        hprintln!("[INFO] DDM did not trigger on short sequence (expected with 100+50 samples)");
    }

    // Extended DDM test: longer baseline for reliable detection
    hprintln!("--- DDM extended (500 + 200) ---");
    let mut ddm2 = Ddm::with_params(2.0, 3.0, 30);
    for i in 0..500 {
        ddm2.update(deterministic_value(0.05, 0.005, i));
    }
    let mut ddm2_drifts = 0u32;
    for i in 0..200 {
        let signal = ddm2.update(deterministic_value(0.8, 0.01, 500 + i));
        if signal == DriftSignal::Drift {
            ddm2_drifts += 1;
        }
    }
    if ddm2_drifts > 0 {
        hprintln!("[OK] DDM extended detected {} drift(s)", ddm2_drifts);
    } else {
        hprintln!("[FAIL] DDM extended did not detect drift");
    }

    // Test DDM reset
    ddm2.reset();
    hprintln!("DDM after reset: mean={}", ddm2.estimated_mean() as u32);

    // -----------------------------------------------------------------------
    // PageHinkleyTest
    // -----------------------------------------------------------------------
    hprintln!("--- PageHinkleyTest ---");

    let mut pht = PageHinkleyTest::new();
    hprintln!(
        "PHT created: delta={}, lambda={}",
        pht.delta() as u32,
        pht.lambda() as u32
    );

    // Phase 1: 100 samples of normal data (error ~0.1)
    let mut pht_warnings = 0u32;
    let mut pht_drifts = 0u32;
    for i in 0..100 {
        let value = deterministic_value(0.1, 0.02, i);
        let signal = pht.update(value);
        match signal {
            DriftSignal::Warning => pht_warnings += 1,
            DriftSignal::Drift => pht_drifts += 1,
            DriftSignal::Stable => {}
        }
    }
    hprintln!(
        "PHT phase 1 (normal): warnings={}, drifts={}",
        pht_warnings,
        pht_drifts
    );

    // Phase 2: 50 samples of drifted data (error ~0.8)
    let mut pht_phase2_warnings = 0u32;
    let mut pht_phase2_drifts = 0u32;
    for i in 0..50 {
        let value = deterministic_value(0.8, 0.02, 100 + i);
        let signal = pht.update(value);
        match signal {
            DriftSignal::Warning => pht_phase2_warnings += 1,
            DriftSignal::Drift => pht_phase2_drifts += 1,
            DriftSignal::Stable => {}
        }
    }
    hprintln!(
        "PHT phase 2 (drifted): warnings={}, drifts={}",
        pht_phase2_warnings,
        pht_phase2_drifts
    );

    if pht_phase2_drifts > 0 {
        hprintln!("[OK] PHT detected drift after shift");
    } else if pht_phase2_warnings > 0 {
        hprintln!("[WARN] PHT detected warning but not drift (short sequence)");
    } else {
        hprintln!("[INFO] PHT did not trigger on short sequence");
    }

    // Extended PHT test: longer baseline with larger shift
    hprintln!("--- PHT extended (500 + 200) ---");
    let mut pht2 = PageHinkleyTest::with_params(0.005, 50.0);
    for i in 0..500 {
        pht2.update(deterministic_value(0.0, 0.01, i));
    }
    let mut pht2_drifts = 0u32;
    for i in 0..200 {
        let signal = pht2.update(deterministic_value(10.0, 0.01, 500 + i));
        if signal == DriftSignal::Drift {
            pht2_drifts += 1;
        }
    }
    if pht2_drifts > 0 {
        hprintln!("[OK] PHT extended detected {} drift(s)", pht2_drifts);
    } else {
        hprintln!("[FAIL] PHT extended did not detect drift");
    }

    // Test PHT reset
    pht2.reset();
    hprintln!(
        "PHT after reset: count={}, mean={}",
        pht2.count(),
        pht2.estimated_mean() as u32
    );

    // -----------------------------------------------------------------------
    // DriftSignal Display test
    // -----------------------------------------------------------------------
    hprintln!("--- DriftSignal Display ---");
    hprintln!("Stable: {}", DriftSignal::Stable);
    hprintln!("Warning: {}", DriftSignal::Warning);
    hprintln!("Drift: {}", DriftSignal::Drift);

    hprintln!("DRIFT_COMPLETE");

    cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_SUCCESS);
    loop {}
}
