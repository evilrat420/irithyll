//! Hello, streaming ML.
//!
//! The smallest possible irithyll program: create a model, feed it samples
//! one at a time, and make a prediction. Nothing else.
//!
//! Run: `cargo run --example hello_streaming`

use irithyll::{sgbt, StreamingLearner};

fn main() {
    // Create a streaming gradient-boosted-trees model: 50 boosting steps,
    // learning rate 0.05.
    let mut model = sgbt(50, 0.05);

    // Feed it 200 samples of y = 2*x + 1.
    // train() takes a feature slice and a target value.
    for i in 0..200 {
        let x = i as f64 * 0.1;
        let y = 2.0 * x + 1.0;
        model.train(&[x], y);
    }

    // Predict on a new point.
    let prediction = model.predict(&[10.0]);
    println!("Prediction for x=10: {:.3}  (true: 21.000)", prediction);
    println!("[OK] Hello streaming complete.");
}
