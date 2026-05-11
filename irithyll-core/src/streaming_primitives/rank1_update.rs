//! Rank-1 outer-product in-place matrix update: `M += α · x · y^T`.
//!
//! Core operation for every model that updates a weight matrix per sample.
//! Used by delta-family attention variants (DeltaNet, GatedDeltaNet,
//! DeltaProduct, RWKV-7), Titans/TTT, and sLSTM's recurrent block.
//!
//! The matrix `M` is stored **row-major**, so element `(i, j)` lives at
//! `matrix[i * cols + j]`.

/// Rank-1 outer-product update: `M += α · x · y^T`, in-place.
///
/// `matrix` is a row-major flattened `rows × cols` matrix (length `rows * cols`).
/// `x` has length `rows`, `y` has length `cols`.
///
/// # Release vs debug mode
///
/// Bounds are checked in debug mode via `debug_assert!`. In release mode the
/// inner loop uses unchecked indexing to avoid repeated bounds checks — see
/// the `// SAFETY:` comment inside.
///
/// # Panics
///
/// Panics in debug mode if any dimension is inconsistent.
#[inline]
pub fn rank1_outer_update(
    matrix: &mut [f64],
    rows: usize,
    cols: usize,
    alpha: f64,
    x: &[f64],
    y: &[f64],
) {
    debug_assert_eq!(matrix.len(), rows * cols, "matrix len != rows*cols");
    debug_assert_eq!(x.len(), rows, "x len != rows");
    debug_assert_eq!(y.len(), cols, "y len != cols");

    for (i, &xi) in x.iter().enumerate() {
        let alpha_xi = alpha * xi;
        let row_start = i * cols;
        // SAFETY: row_start + j < matrix.len() because:
        //   row_start = i * cols, i < rows (enforced by x.iter().enumerate()),
        //   j < cols (enforced by y.iter().enumerate()).
        //   ⇒ row_start + j < rows * cols == matrix.len().
        // debug_assert above guarantees matrix.len() == rows * cols.
        for (j, &yj) in y.iter().enumerate() {
            unsafe {
                *matrix.get_unchecked_mut(row_start + j) += alpha_xi * yj;
            }
        }
    }
}

/// Rank-1 outer-product update with `α = 1.0`: `M += x · y^T`, in-place.
///
/// Equivalent to `rank1_outer_update(matrix, rows, cols, 1.0, x, y)` but
/// saves one multiply per element when `α` is constant 1.
#[inline]
pub fn rank1_outer_update_inplace(
    matrix: &mut [f64],
    rows: usize,
    cols: usize,
    x: &[f64],
    y: &[f64],
) {
    debug_assert_eq!(matrix.len(), rows * cols);
    debug_assert_eq!(x.len(), rows);
    debug_assert_eq!(y.len(), cols);

    for (i, &xi) in x.iter().enumerate() {
        let row_start = i * cols;
        // SAFETY: same invariant as rank1_outer_update.
        // i < rows (from x.iter()), j < cols (from y.iter()),
        // row_start + j < rows * cols == matrix.len().
        for (j, &yj) in y.iter().enumerate() {
            unsafe {
                *matrix.get_unchecked_mut(row_start + j) += xi * yj;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive O(rows*cols) reference implementation for comparison.
    fn naive_rank1(matrix: &mut [f64], rows: usize, cols: usize, alpha: f64, x: &[f64], y: &[f64]) {
        for i in 0..rows {
            for j in 0..cols {
                matrix[i * cols + j] += alpha * x[i] * y[j];
            }
        }
    }

    /// fast and naive implementations agree on a 4×3 matrix.
    #[test]
    fn rank1_matches_naive_reference() {
        let rows = 4;
        let cols = 3;
        let alpha = 2.5;
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [0.5, -1.0, 0.25];
        // 4*3 = 12 elements, use fixed-size arrays.
        let mut m_fast = [0.0f64; 12];
        let mut m_naive = [0.0f64; 12];

        rank1_outer_update(&mut m_fast, rows, cols, alpha, &x, &y);
        naive_rank1(&mut m_naive, rows, cols, alpha, &x, &y);

        for (a, b) in m_fast.iter().zip(m_naive.iter()) {
            assert!((a - b).abs() < 1e-12, "mismatch: fast={a}, naive={b}");
        }
    }

    /// alpha=1 path and inplace path give identical results.
    #[test]
    fn rank1_inplace_matches_alpha_one() {
        let rows = 3;
        let cols = 5;
        let x = [1.0, -2.0, 0.5];
        let y = [0.1, 0.2, 0.3, 0.4, 0.5];
        // 3*5 = 15 elements.
        let mut m_alpha = [0.0f64; 15];
        let mut m_inplace = [0.0f64; 15];

        rank1_outer_update(&mut m_alpha, rows, cols, 1.0, &x, &y);
        rank1_outer_update_inplace(&mut m_inplace, rows, cols, &x, &y);

        for (a, b) in m_alpha.iter().zip(m_inplace.iter()) {
            assert!((a - b).abs() < 1e-15, "alpha=1 and inplace must match");
        }
    }

    /// Two sequential updates accumulate correctly.
    #[test]
    fn rank1_accumulates_correctly() {
        // Two updates: M += 1·[1,2]·[1,2]^T + 1·[2,4]·[2,4]^T on 2×2.
        let rows = 2;
        let cols = 2;
        let mut m = [0.0f64; 4];

        rank1_outer_update_inplace(&mut m, rows, cols, &[1.0, 2.0], &[1.0, 2.0]);
        rank1_outer_update_inplace(&mut m, rows, cols, &[2.0, 4.0], &[2.0, 4.0]);

        // m[0][0] = 1*1 + 2*2 = 5, m[0][1] = 1*2 + 2*4 = 10,
        // m[1][0] = 2*1 + 4*2 = 10, m[1][1] = 2*2 + 4*4 = 20.
        assert!(
            (m[0] - 5.0).abs() < 1e-12,
            "m[0][0] expected 5, got {}",
            m[0]
        );
        assert!(
            (m[1] - 10.0).abs() < 1e-12,
            "m[0][1] expected 10, got {}",
            m[1]
        );
        assert!(
            (m[2] - 10.0).abs() < 1e-12,
            "m[1][0] expected 10, got {}",
            m[2]
        );
        assert!(
            (m[3] - 20.0).abs() < 1e-12,
            "m[1][1] expected 20, got {}",
            m[3]
        );
    }

    /// zero alpha leaves matrix unchanged.
    #[test]
    fn rank1_with_zero_alpha_leaves_matrix_unchanged() {
        let rows = 2;
        let cols = 2;
        let initial = [1.0, 2.0, 3.0, 4.0];
        let mut m = initial;
        rank1_outer_update(&mut m, rows, cols, 0.0, &[100.0, 200.0], &[300.0, 400.0]);
        for (a, &b) in m.iter().zip(initial.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "zero-alpha update should not change matrix"
            );
        }
    }

    /// 1×1 matrix: scalar case.
    #[test]
    fn rank1_1x1_matrix() {
        let mut m = [0.0f64; 1];
        rank1_outer_update(&mut m, 1, 1, 3.0, &[4.0], &[5.0]);
        assert!((m[0] - 60.0).abs() < 1e-12, "1x1: 3*4*5 = 60");
    }
}
