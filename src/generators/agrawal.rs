//! Agrawal generator (Agrawal et al., 1993).
//!
//! Generates samples with 9 features (salary, commission, age, education level,
//! car, zipcode, and 3 derived features) and 10 classification functions for
//! loan approval. Concept drift is modeled by switching the classification function.

use super::{Rng, StreamGenerator, TaskType};

/// Agrawal stream generator with 10 classification functions.
///
/// # Features
/// 0. salary: uniform [20_000, 150_000]
/// 1. commission: if salary < 75_000 then uniform [10_000, 75_000] else 0
/// 2. age: uniform [20, 80]
/// 3. elevel (education level): uniform {0, 1, 2, 3, 4}
/// 4. car: uniform {1, 2, ..., 20}
/// 5. zipcode: uniform {0, 1, ..., 8}
/// 6. hvalue (house value): derived from zipcode
/// 7. hyears (years at house): uniform [1, 30]
/// 8. loan: derived from other features
#[derive(Debug, Clone)]
pub struct Agrawal {
    rng: Rng,
    function_idx: usize,
    change_points: Vec<(usize, usize)>, // (sample_idx, new_function_idx)
    noise_rate: f64,
    sample_idx: usize,
    next_change: usize,
    drift_flag: bool,
}

impl Agrawal {
    /// Create a new Agrawal generator using classification function 0.
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, 0, Vec::new(), 0.05)
    }

    /// Create with a specific function index, change points, and noise rate.
    ///
    /// `change_points` is a list of `(sample_index, new_function_index)` pairs.
    pub fn with_config(
        seed: u64,
        function_idx: usize,
        change_points: Vec<(usize, usize)>,
        noise_rate: f64,
    ) -> Self {
        assert!(function_idx < 10, "function_idx must be in 0..9");
        Self {
            rng: Rng::new(seed),
            function_idx,
            change_points,
            noise_rate,
            sample_idx: 0,
            next_change: 0,
            drift_flag: false,
        }
    }

    /// Generate the 9 raw features.
    fn generate_features(&mut self) -> Vec<f64> {
        let salary = self.rng.uniform_range(20_000.0, 150_000.0);
        let commission = if salary < 75_000.0 {
            self.rng.uniform_range(10_000.0, 75_000.0)
        } else {
            0.0
        };
        let age = self.rng.uniform_range(20.0, 80.0);
        let elevel = self.rng.uniform_int(5) as f64;
        let car = (self.rng.uniform_int(20) + 1) as f64;
        let zipcode = self.rng.uniform_int(9) as f64;

        // Derived: house value depends on zipcode
        let hvalue =
            (zipcode as u32 + 1) as f64 * 100_000.0 * (1.0 + self.rng.uniform_range(-0.3, 0.3));
        let hyears = self.rng.uniform_range(1.0, 30.0);
        let loan = self.rng.uniform_range(0.0, 500_000.0);

        vec![
            salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan,
        ]
    }

    /// Evaluate classification function on features.
    fn classify(&self, f: &[f64]) -> f64 {
        let salary = f[0];
        let commission = f[1];
        let age = f[2];
        let elevel = f[3];
        let car = f[4];
        let zipcode = f[5];
        let hvalue = f[6];
        let hyears = f[7];
        let loan = f[8];

        let result = match self.function_idx {
            0 => !(40.0..60.0).contains(&age),
            1 => salary < 60_000.0,
            2 => (40.0..60.0).contains(&age) || (salary >= 100_000.0),
            3 => (elevel >= 3.0) || (salary < 40_000.0),
            4 => {
                let disposable = salary + commission - loan;
                disposable > 50_000.0
            }
            5 => (car > 15.0) || (hvalue > 300_000.0),
            6 => (hyears > 10.0) && (loan < 200_000.0),
            7 => {
                let _ = zipcode; // zipcode involvement
                (age > 50.0) && (zipcode >= 4.0)
            }
            8 => (commission > 40_000.0) || (hvalue > 400_000.0),
            9 => (salary > 80_000.0) && (age < 50.0) && (elevel >= 2.0),
            _ => unreachable!(),
        };
        if result {
            1.0
        } else {
            0.0
        }
    }
}

impl StreamGenerator for Agrawal {
    fn next_sample(&mut self) -> (Vec<f64>, f64) {
        // Check for concept drift
        self.drift_flag = false;
        if self.next_change < self.change_points.len() {
            let (cp_idx, new_fn) = self.change_points[self.next_change];
            if self.sample_idx == cp_idx {
                self.function_idx = new_fn;
                self.next_change += 1;
                self.drift_flag = true;
            }
        }

        let features = self.generate_features();
        let mut label = self.classify(&features);

        // Noise: flip label
        if self.rng.bernoulli(self.noise_rate) {
            label = 1.0 - label;
        }

        self.sample_idx += 1;
        (features, label)
    }

    fn n_features(&self) -> usize {
        9
    }

    fn task_type(&self) -> TaskType {
        TaskType::BinaryClassification
    }

    fn drift_occurred(&self) -> bool {
        self.drift_flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agrawal_produces_correct_n_features() {
        let mut gen = Agrawal::new(42);
        let (features, _) = gen.next_sample();
        assert_eq!(
            features.len(),
            9,
            "Agrawal should produce 9 features, got {}",
            features.len()
        );
    }

    #[test]
    fn agrawal_task_type_is_binary_classification() {
        let gen = Agrawal::new(42);
        assert_eq!(
            gen.task_type(),
            TaskType::BinaryClassification,
            "Agrawal task type should be BinaryClassification"
        );
    }

    #[test]
    fn agrawal_produces_finite_values() {
        let mut gen = Agrawal::new(55);
        for i in 0..500 {
            let (features, target) = gen.next_sample();
            for (j, f) in features.iter().enumerate() {
                assert!(f.is_finite(), "feature {} at sample {} is not finite", j, i);
            }
            assert!(target.is_finite(), "target at sample {} is not finite", i);
        }
    }

    #[test]
    fn agrawal_labels_are_binary() {
        let mut gen = Agrawal::new(99);
        for _ in 0..500 {
            let (_, target) = gen.next_sample();
            assert!(
                target == 0.0 || target == 1.0,
                "Agrawal label should be 0.0 or 1.0, got {}",
                target
            );
        }
    }

    #[test]
    fn agrawal_all_functions_produce_valid_output() {
        for func_idx in 0..10 {
            let mut gen = Agrawal::with_config(42, func_idx, Vec::new(), 0.0);
            for _ in 0..100 {
                let (features, target) = gen.next_sample();
                assert_eq!(features.len(), 9, "function {} wrong n_features", func_idx);
                assert!(
                    target == 0.0 || target == 1.0,
                    "function {} produced invalid label {}",
                    func_idx,
                    target
                );
            }
        }
    }

    #[test]
    fn agrawal_drift_at_change_points() {
        let mut gen = Agrawal::with_config(42, 0, vec![(100, 3), (200, 7)], 0.0);

        for _ in 0..100 {
            gen.next_sample();
        }
        // Sample 100 triggers drift to function 3
        gen.next_sample();
        assert!(
            gen.drift_occurred(),
            "drift should occur at change point 100"
        );

        for _ in 102..200 {
            gen.next_sample();
            assert!(
                !gen.drift_occurred(),
                "no drift expected between change points"
            );
        }
    }

    #[test]
    fn agrawal_deterministic_with_same_seed() {
        let mut gen1 = Agrawal::new(42);
        let mut gen2 = Agrawal::new(42);
        for _ in 0..200 {
            let (f1, t1) = gen1.next_sample();
            let (f2, t2) = gen2.next_sample();
            assert_eq!(f1, f2, "same seed should produce identical features");
            assert_eq!(t1, t2, "same seed should produce identical targets");
        }
    }

    #[test]
    fn agrawal_salary_in_range() {
        let mut gen = Agrawal::with_config(77, 0, Vec::new(), 0.0);
        for _ in 0..500 {
            let (features, _) = gen.next_sample();
            let salary = features[0];
            assert!(
                (20_000.0..=150_000.0).contains(&salary),
                "salary should be in [20K, 150K], got {}",
                salary
            );
        }
    }
}
