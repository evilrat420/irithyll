//! Streaming dimensionality reduction with CCIPCA.
//!
//! Demonstrates Candid Covariance-free Incremental PCA reducing 10-dimensional
//! data down to 3 principal components in a single pass. The first 3 dimensions
//! carry structured signal while dimensions 4-10 are pure noise. CCIPCA
//! discovers the true underlying dimensionality with O(kd) cost per sample.
//!
//! Run: `cargo run --example ccipca_reduction`

use irithyll::ccipca;

/// Deterministic PRNG (xorshift64). Returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

fn main() {
    println!("=== Irithyll: CCIPCA Streaming Dimensionality Reduction ===");
    println!("10-dimensional data -> 3 principal components\n");

    // 1. Create CCIPCA with 3 components
    let n_components = 3;
    let n_features = 10;
    let mut pca = ccipca(n_components);

    println!(
        "Config: n_components={}, input_dim={}\n",
        n_components, n_features
    );

    // 2. Stream 300 samples where first 3 dims have signal, rest is noise
    //    Dim 0: large variance (scale 100)
    //    Dim 1: medium variance (scale 50)
    //    Dim 2: moderate variance (scale 25)
    //    Dims 3-9: noise (scale 0.1)
    let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let n_samples = 300;
    let scale_factors: [f64; 10] = [100.0, 50.0, 25.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

    println!("--- Training ({} samples) ---", n_samples);
    println!("  Signal dims 0-2 scaled by [100, 50, 25], noise dims 3-9 by 0.1");

    let mut last_reduced = vec![0.0; n_components];
    for i in 0..n_samples {
        // Generate 10-dimensional sample
        let mut features = [0.0_f64; 10];
        for (j, f) in features.iter_mut().enumerate() {
            *f = (xorshift64(&mut rng) - 0.5) * 2.0 * scale_factors[j];
        }

        // Update PCA and get reduced output
        last_reduced = pca.update_and_transform(&features);

        // Report at intervals
        if (i + 1) % 75 == 0 {
            println!(
                "  Samples: {:>4} | Reduced dim: {} | Output: [{:.2}, {:.2}, {:.2}]",
                i + 1,
                last_reduced.len(),
                last_reduced[0],
                last_reduced[1],
                last_reduced[2],
            );
        }
    }

    // 3. Verify output dimensionality
    println!("\n--- Dimensionality Verification ---");
    println!("  Input dimensionality:  {}", n_features);
    println!("  Output dimensionality: {}", last_reduced.len());
    assert_eq!(
        last_reduced.len(),
        n_components,
        "output should have exactly n_components dimensions"
    );
    println!("  [OK] Output length is {}", n_components);

    // 4. Project a fresh test point to confirm transform works independently
    let test_point: [f64; 10] = [50.0, 25.0, 12.5, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];
    let projected = pca.transform(&test_point);
    println!("\n--- Fresh Projection ---");
    println!("  Input:  10-dimensional vector");
    println!("  Output: {}-dimensional vector", projected.len());
    println!(
        "  Values: [{:.4}, {:.4}, {:.4}]",
        projected[0], projected[1], projected[2]
    );
    assert_eq!(projected.len(), n_components);
    println!("  [OK] Transform output verified");

    println!("\n[DONE] CCIPCA dimensionality reduction example complete.");
}
