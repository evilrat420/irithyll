//! Cortex-M0+ error handling validation test.
//!
//! Verifies that malformed inputs produce proper FormatError variants (not panics)
//! on bare-metal ARM. Tests truncated, empty, and garbage binary inputs for both
//! f32 and i16 model paths.
//!
//! # Build (from irithyll-core/)
//!
//! ```bash
//! cargo build --features embedded-bench --target thumbv6m-none-eabi --release --example cortex_m_validation
//! ```
//!
//! # Run under QEMU
//!
//! ```bash
//! qemu-system-arm -cpu cortex-m0 -machine lm3s6965evb -nographic \
//!     -semihosting-config enable=on,target=native \
//!     -kernel target/thumbv6m-none-eabi/release/examples/cortex_m_validation
//! ```

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::hprintln;
use panic_halt as _;

use irithyll_core::{EnsembleView, FormatError, QuantizedEnsembleView};

#[entry]
fn main() -> ! {
    hprintln!("irithyll-core Cortex-M0+ validation test");

    let mut passed = 0u32;
    let mut failed = 0u32;

    // We include the full model to extract truncated slices from it.
    static MODEL_F32: &[u8] = include_bytes!("../test_data/bench_model_50t.bin");
    static MODEL_I16: &[u8] = include_bytes!("../test_data/bench_model_50t_i16.bin");

    // -----------------------------------------------------------------------
    // Test 1: Truncated f32 binary (first 10 bytes)
    // -----------------------------------------------------------------------
    hprintln!("--- Test 1: truncated f32 (10 bytes) ---");
    {
        // Use a stack-local aligned buffer so the pointer is 4-byte aligned.
        // This ensures we test Truncated rather than Unaligned.
        #[repr(align(4))]
        struct Aligned10([u8; 12]); // 12 to be 4-aligned size
        let mut buf = Aligned10([0u8; 12]);
        // Copy first 10 bytes from model (pad remaining with 0).
        let copy_len = if MODEL_F32.len() >= 10 {
            10
        } else {
            MODEL_F32.len()
        };
        let mut i = 0;
        while i < copy_len {
            buf.0[i] = MODEL_F32[i];
            i += 1;
        }
        let slice = &buf.0[..10];

        match EnsembleView::from_bytes(slice) {
            Err(FormatError::Truncated) => {
                hprintln!("[OK] truncated f32 -> FormatError::Truncated");
                passed += 1;
            }
            Err(other) => {
                hprintln!("[OK] truncated f32 -> FormatError::{} (acceptable)", other);
                passed += 1;
            }
            Ok(_) => {
                hprintln!("[FAIL] truncated f32 was accepted as valid");
                failed += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: Empty bytes (f32)
    // -----------------------------------------------------------------------
    hprintln!("--- Test 2: empty bytes (f32) ---");
    {
        #[repr(align(4))]
        struct AlignedEmpty([u8; 4]);
        let buf = AlignedEmpty([0u8; 4]);
        let empty: &[u8] = &buf.0[..0];

        match EnsembleView::from_bytes(empty) {
            Err(FormatError::Truncated) => {
                hprintln!("[OK] empty f32 -> FormatError::Truncated");
                passed += 1;
            }
            Err(other) => {
                hprintln!("[OK] empty f32 -> FormatError::{} (acceptable)", other);
                passed += 1;
            }
            Ok(_) => {
                hprintln!("[FAIL] empty f32 was accepted as valid");
                failed += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: Garbage bytes (f32) -- all 0xFF
    // -----------------------------------------------------------------------
    hprintln!("--- Test 3: garbage bytes (f32) ---");
    {
        #[repr(align(4))]
        struct AlignedGarbage([u8; 64]);
        let buf = AlignedGarbage([0xFF; 64]);

        match EnsembleView::from_bytes(&buf.0) {
            Err(e) => {
                hprintln!("[OK] garbage f32 -> FormatError::{}", e);
                passed += 1;
            }
            Ok(_) => {
                hprintln!("[FAIL] garbage f32 was accepted as valid");
                failed += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: Truncated i16 binary (first 10 bytes)
    // -----------------------------------------------------------------------
    hprintln!("--- Test 4: truncated i16 (10 bytes) ---");
    {
        #[repr(align(4))]
        struct Aligned10([u8; 12]);
        let mut buf = Aligned10([0u8; 12]);
        let copy_len = if MODEL_I16.len() >= 10 {
            10
        } else {
            MODEL_I16.len()
        };
        let mut i = 0;
        while i < copy_len {
            buf.0[i] = MODEL_I16[i];
            i += 1;
        }
        let slice = &buf.0[..10];

        match QuantizedEnsembleView::from_bytes(slice) {
            Err(FormatError::Truncated) => {
                hprintln!("[OK] truncated i16 -> FormatError::Truncated");
                passed += 1;
            }
            Err(other) => {
                hprintln!("[OK] truncated i16 -> FormatError::{} (acceptable)", other);
                passed += 1;
            }
            Ok(_) => {
                hprintln!("[FAIL] truncated i16 was accepted as valid");
                failed += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: Empty bytes (i16)
    // -----------------------------------------------------------------------
    hprintln!("--- Test 5: empty bytes (i16) ---");
    {
        #[repr(align(4))]
        struct AlignedEmpty([u8; 4]);
        let buf = AlignedEmpty([0u8; 4]);
        let empty: &[u8] = &buf.0[..0];

        match QuantizedEnsembleView::from_bytes(empty) {
            Err(FormatError::Truncated) => {
                hprintln!("[OK] empty i16 -> FormatError::Truncated");
                passed += 1;
            }
            Err(other) => {
                hprintln!("[OK] empty i16 -> FormatError::{} (acceptable)", other);
                passed += 1;
            }
            Ok(_) => {
                hprintln!("[FAIL] empty i16 was accepted as valid");
                failed += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: Garbage bytes (i16) -- all 0xFF
    // -----------------------------------------------------------------------
    hprintln!("--- Test 6: garbage bytes (i16) ---");
    {
        #[repr(align(4))]
        struct AlignedGarbage([u8; 64]);
        let buf = AlignedGarbage([0xFF; 64]);

        match QuantizedEnsembleView::from_bytes(&buf.0) {
            Err(e) => {
                hprintln!("[OK] garbage i16 -> FormatError::{}", e);
                passed += 1;
            }
            Ok(_) => {
                hprintln!("[FAIL] garbage i16 was accepted as valid");
                failed += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: Valid model still works after all the error tests
    // -----------------------------------------------------------------------
    hprintln!("--- Test 7: valid model sanity check ---");
    {
        match EnsembleView::from_bytes(MODEL_F32) {
            Ok(view) => {
                let features = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
                let pred = view.predict(&features);
                if pred.is_finite() {
                    hprintln!("[OK] valid f32 model: pred={}", pred);
                    passed += 1;
                } else {
                    hprintln!("[FAIL] valid f32 model produced non-finite: {}", pred);
                    failed += 1;
                }
            }
            Err(e) => {
                hprintln!("[FAIL] valid f32 model rejected: {}", e);
                failed += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: Valid i16 model sanity check
    // -----------------------------------------------------------------------
    hprintln!("--- Test 8: valid i16 model sanity check ---");
    {
        match QuantizedEnsembleView::from_bytes(MODEL_I16) {
            Ok(view) => {
                let features = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
                let pred = view.predict(&features);
                if pred.is_finite() {
                    hprintln!("[OK] valid i16 model: pred={}", pred);
                    passed += 1;
                } else {
                    hprintln!("[FAIL] valid i16 model produced non-finite: {}", pred);
                    failed += 1;
                }
            }
            Err(e) => {
                hprintln!("[FAIL] valid i16 model rejected: {}", e);
                failed += 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    hprintln!("--- Summary ---");
    hprintln!(
        "passed={}, failed={}, total={}",
        passed,
        failed,
        passed + failed
    );

    if failed == 0 {
        hprintln!("[OK] All validation tests passed");
    } else {
        hprintln!("[FAIL] {} validation test(s) failed", failed);
    }

    hprintln!("VALIDATION_COMPLETE");

    cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_SUCCESS);
    loop {}
}
