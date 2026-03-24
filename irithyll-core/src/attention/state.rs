//! State containers for streaming attention heads.
//!
//! Attention heads maintain either a vector state (Hawk) or a matrix state
//! (all other variants). The matrix state stores an outer-product accumulation
//! `S` of shape `d_key x d_value` in row-major order.

use alloc::vec;
use alloc::vec::Vec;

/// Attention head state -- either a vector or a matrix.
///
/// - **Vector**: Used by Hawk (gated scalar recurrence). Shape: `[d]`.
/// - **Matrix**: Used by RetNet, GLA, DeltaNet, GatedDeltaNet, RWKV, mLSTM.
///   Shape: `d_key x d_value`, stored row-major in a flat `Vec<f64>`.
#[derive(Clone, Debug)]
pub enum AttentionState {
    /// Vector state of dimension `d`.
    Vector(Vec<f64>),
    /// Matrix state of shape `rows x cols`, stored row-major.
    Matrix {
        /// Flat row-major data of length `rows * cols`.
        data: Vec<f64>,
        /// Number of rows (typically `d_key`).
        rows: usize,
        /// Number of columns (typically `d_value`).
        cols: usize,
    },
}

impl AttentionState {
    /// Create a zero-initialized vector state of dimension `d`.
    pub fn new_vector(d: usize) -> Self {
        Self::Vector(vec![0.0; d])
    }

    /// Create a zero-initialized matrix state of shape `d_k x d_v`.
    pub fn new_matrix(d_k: usize, d_v: usize) -> Self {
        Self::Matrix {
            data: vec![0.0; d_k * d_v],
            rows: d_k,
            cols: d_v,
        }
    }

    /// Reset all state values to zero.
    pub fn reset(&mut self) {
        match self {
            Self::Vector(v) => {
                for x in v.iter_mut() {
                    *x = 0.0;
                }
            }
            Self::Matrix { data, .. } => {
                for x in data.iter_mut() {
                    *x = 0.0;
                }
            }
        }
    }

    /// Flat view of the state for readout (borrows the underlying data).
    pub fn as_slice(&self) -> &[f64] {
        match self {
            Self::Vector(v) => v.as_slice(),
            Self::Matrix { data, .. } => data.as_slice(),
        }
    }

    /// Get a matrix element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if the state is a Vector, or if indices are out of bounds.
    pub fn get_matrix(&self, row: usize, col: usize) -> f64 {
        match self {
            Self::Matrix { data, cols, rows } => {
                debug_assert!(row < *rows, "row {} out of bounds (rows={})", row, rows);
                debug_assert!(col < *cols, "col {} out of bounds (cols={})", col, cols);
                data[row * cols + col]
            }
            Self::Vector(_) => panic!("get_matrix called on Vector state"),
        }
    }

    /// Set a matrix element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if the state is a Vector, or if indices are out of bounds.
    pub fn set_matrix(&mut self, row: usize, col: usize, val: f64) {
        match self {
            Self::Matrix { data, cols, rows } => {
                debug_assert!(row < *rows, "row {} out of bounds (rows={})", row, rows);
                debug_assert!(col < *cols, "col {} out of bounds (cols={})", col, cols);
                data[row * *cols + col] = val;
            }
            Self::Vector(_) => panic!("set_matrix called on Vector state"),
        }
    }

    /// Multiply all state values by `factor`.
    pub fn scale(&mut self, factor: f64) {
        match self {
            Self::Vector(v) => {
                for x in v.iter_mut() {
                    *x *= factor;
                }
            }
            Self::Matrix { data, .. } => {
                for x in data.iter_mut() {
                    *x *= factor;
                }
            }
        }
    }

    /// Add outer product: `S += u * v^T`.
    ///
    /// For matrix state, `u` must have length `rows` and `v` must have length `cols`.
    ///
    /// # Panics
    ///
    /// Panics if the state is a Vector, or if dimensions don't match.
    #[allow(clippy::needless_range_loop)]
    pub fn add_outer_product(&mut self, u: &[f64], v: &[f64]) {
        match self {
            Self::Matrix { data, rows, cols } => {
                debug_assert_eq!(u.len(), *rows, "u.len() must equal rows");
                debug_assert_eq!(v.len(), *cols, "v.len() must equal cols");
                for i in 0..*rows {
                    let row_start = i * *cols;
                    for j in 0..*cols {
                        data[row_start + j] += u[i] * v[j];
                    }
                }
            }
            Self::Vector(_) => panic!("add_outer_product called on Vector state"),
        }
    }

    /// Compute `S^T * q` for matrix state, returning a vector of length `cols`.
    ///
    /// This is the standard attention readout: `o = S^T * q` where `q` has
    /// length `rows` (d_key) and the output has length `cols` (d_value).
    ///
    /// # Panics
    ///
    /// Panics if the state is a Vector or if `q.len() != rows`.
    #[allow(clippy::needless_range_loop)]
    pub fn query(&self, q: &[f64]) -> Vec<f64> {
        match self {
            Self::Matrix { data, rows, cols } => {
                debug_assert_eq!(q.len(), *rows, "q.len() must equal rows (d_key)");
                let mut out = vec![0.0; *cols];
                // S^T * q: out[j] = sum_i S[i][j] * q[i]
                for i in 0..*rows {
                    let row_start = i * *cols;
                    let qi = q[i];
                    for j in 0..*cols {
                        out[j] += data[row_start + j] * qi;
                    }
                }
                out
            }
            Self::Vector(_) => panic!("query called on Vector state"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_state_init_and_reset() {
        let mut state = AttentionState::new_vector(8);
        match &state {
            AttentionState::Vector(v) => {
                assert_eq!(v.len(), 8, "vector should have length 8");
                assert!(
                    v.iter().all(|&x| x == 0.0),
                    "initial vector should be all zeros"
                );
            }
            _ => panic!("expected Vector state"),
        }
        // Modify then reset
        if let AttentionState::Vector(v) = &mut state {
            v[0] = 5.0;
            v[3] = -2.0;
        }
        state.reset();
        assert!(
            state.as_slice().iter().all(|&x| x == 0.0),
            "after reset, all values should be zero"
        );
    }

    #[test]
    fn matrix_state_init_dimensions() {
        let state = AttentionState::new_matrix(3, 5);
        match &state {
            AttentionState::Matrix { data, rows, cols } => {
                assert_eq!(*rows, 3, "should have 3 rows");
                assert_eq!(*cols, 5, "should have 5 cols");
                assert_eq!(data.len(), 15, "data should have 3*5=15 elements");
                assert!(
                    data.iter().all(|&x| x == 0.0),
                    "initial matrix should be all zeros"
                );
            }
            _ => panic!("expected Matrix state"),
        }
    }

    #[test]
    fn matrix_get_set() {
        let mut state = AttentionState::new_matrix(2, 3);
        state.set_matrix(0, 1, 7.0);
        state.set_matrix(1, 2, -3.0);
        assert!(
            (state.get_matrix(0, 1) - 7.0).abs() < 1e-12,
            "get(0,1) should be 7.0"
        );
        assert!(
            (state.get_matrix(1, 2) - (-3.0)).abs() < 1e-12,
            "get(1,2) should be -3.0"
        );
        assert!(
            state.get_matrix(0, 0).abs() < 1e-12,
            "unset elements should be 0.0"
        );
    }

    #[test]
    fn scale_multiplies_all() {
        let mut state = AttentionState::new_matrix(2, 2);
        state.set_matrix(0, 0, 1.0);
        state.set_matrix(0, 1, 2.0);
        state.set_matrix(1, 0, 3.0);
        state.set_matrix(1, 1, 4.0);
        state.scale(0.5);
        assert!(
            (state.get_matrix(0, 0) - 0.5).abs() < 1e-12,
            "scaled (0,0) should be 0.5"
        );
        assert!(
            (state.get_matrix(1, 1) - 2.0).abs() < 1e-12,
            "scaled (1,1) should be 2.0"
        );
    }

    #[test]
    fn add_outer_product_correct() {
        let mut state = AttentionState::new_matrix(2, 3);
        let u = [1.0, 2.0];
        let v = [3.0, 4.0, 5.0];
        state.add_outer_product(&u, &v);
        // Expected: [[3, 4, 5], [6, 8, 10]]
        assert!(
            (state.get_matrix(0, 0) - 3.0).abs() < 1e-12,
            "S[0][0] should be 1*3=3"
        );
        assert!(
            (state.get_matrix(0, 2) - 5.0).abs() < 1e-12,
            "S[0][2] should be 1*5=5"
        );
        assert!(
            (state.get_matrix(1, 1) - 8.0).abs() < 1e-12,
            "S[1][1] should be 2*4=8"
        );
        assert!(
            (state.get_matrix(1, 2) - 10.0).abs() < 1e-12,
            "S[1][2] should be 2*5=10"
        );
    }

    #[test]
    fn query_computes_st_times_q() {
        let mut state = AttentionState::new_matrix(2, 3);
        // S = [[1, 2, 3], [4, 5, 6]]
        state.set_matrix(0, 0, 1.0);
        state.set_matrix(0, 1, 2.0);
        state.set_matrix(0, 2, 3.0);
        state.set_matrix(1, 0, 4.0);
        state.set_matrix(1, 1, 5.0);
        state.set_matrix(1, 2, 6.0);
        let q = [1.0, 1.0];
        // S^T * q = [[1,4],[2,5],[3,6]] * [1,1] = [5, 7, 9]
        let out = state.query(&q);
        assert_eq!(out.len(), 3, "query output should have d_value=3 elements");
        assert!(
            (out[0] - 5.0).abs() < 1e-12,
            "S^T*q[0] should be 1+4=5, got {}",
            out[0]
        );
        assert!(
            (out[1] - 7.0).abs() < 1e-12,
            "S^T*q[1] should be 2+5=7, got {}",
            out[1]
        );
        assert!(
            (out[2] - 9.0).abs() < 1e-12,
            "S^T*q[2] should be 3+6=9, got {}",
            out[2]
        );
    }

    #[test]
    fn as_slice_returns_flat_data() {
        let mut state = AttentionState::new_matrix(2, 2);
        state.set_matrix(0, 0, 1.0);
        state.set_matrix(1, 1, 4.0);
        let s = state.as_slice();
        assert_eq!(s.len(), 4, "flat slice should have 4 elements");
        assert!((s[0] - 1.0).abs() < 1e-12, "s[0] should be 1.0");
        assert!((s[3] - 4.0).abs() < 1e-12, "s[3] should be 4.0");
    }

    #[test]
    fn scale_vector_state() {
        let mut state = AttentionState::new_vector(3);
        if let AttentionState::Vector(v) = &mut state {
            v[0] = 2.0;
            v[1] = -4.0;
            v[2] = 6.0;
        }
        state.scale(0.5);
        let s = state.as_slice();
        assert!((s[0] - 1.0).abs() < 1e-12, "scaled v[0] should be 1.0");
        assert!((s[1] - (-2.0)).abs() < 1e-12, "scaled v[1] should be -2.0");
        assert!((s[2] - 3.0).abs() < 1e-12, "scaled v[2] should be 3.0");
    }
}
