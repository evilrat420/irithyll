//! Streaming KAN: learnable B-spline activations for nonlinear regression.
use irithyll::{streaming_kan, StreamingLearner};

fn main() {
    // KAN with 2 inputs, 8 hidden nodes, 1 output
    let mut model = streaming_kan(&[2, 8, 1], 0.01);

    println!("Training StreamingKAN on y = sin(x1) + x2^2...");

    for i in 0..2000 {
        let x1 = (i as f64 * 0.01 - 10.0) * 0.1;
        let x2 = (i as f64 * 0.007 - 7.0) * 0.1;
        let y = x1.sin() + x2 * x2;
        model.train(&[x1, x2], y);
    }

    let pred = model.predict(&[0.5, 0.3]);
    println!("Prediction at [0.5, 0.3]: {:.4}", pred);
    println!("Expected: {:.4}", 0.5_f64.sin() + 0.3 * 0.3);
    println!("Parameters: {}", model.n_params());
    println!("Samples seen: {}", model.n_samples_seen());
}
