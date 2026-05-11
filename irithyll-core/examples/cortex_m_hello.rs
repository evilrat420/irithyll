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
    hprintln!("HELLO from Cortex-M0+");
    hprintln!("irithyll-core alive on bare metal");

    // Quick single predict test (no loop)
    static MODEL: &[u8] = include_bytes!("../test_data/bench_model_50t.bin");
    let view = irithyll_core::EnsembleView::from_bytes(MODEL).expect("model");
    hprintln!(
        "model loaded: {} trees, {} nodes",
        view.n_trees(),
        view.total_nodes()
    );

    let features = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let pred = view.predict(&features);
    hprintln!("predict result: {}", pred);
    hprintln!("DONE");

    cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_SUCCESS);
    loop {}
}

// Stub for non-ARM hosts (x86 CI, cargo check --all-targets).
// The real entry point is the #[entry] fn above, which only compiles for ARM.
#[cfg(not(target_arch = "arm"))]
fn main() {}
