//! B-spline basis function evaluation for KAN layers.
//!
//! Implements De Boor's algorithm for evaluating B-spline basis functions
//! and their derivatives. All knot vectors use uniform spacing with k-fold
//! boundary repetition (clamped B-splines).
//!
//! Reference: De Boor, C. (1978) "A Practical Guide to Splines"

/// Create a uniform knot vector with k-fold boundary knots.
///
/// For `g` grid intervals and order `k`, produces `g + 2*k + 1` knots.
/// The interior knots are uniformly spaced from `low` to `high`.
/// Boundary knots are repeated `k` times to clamp the spline.
pub(crate) fn make_grid(low: f64, high: f64, g: usize, k: usize) -> Vec<f64> {
    let step = (high - low) / g as f64;
    let mut grid = Vec::with_capacity(g + 2 * k + 1);
    // k repeated low boundary knots
    for _ in 0..k {
        grid.push(low);
    }
    // G+1 interior knots (uniformly spaced)
    for i in 0..=g {
        grid.push(low + i as f64 * step);
    }
    // k repeated high boundary knots
    for _ in 0..k {
        grid.push(high);
    }
    grid
}

/// Find active knot span: index `i` where `grid[i] <= x < grid[i+1]`.
///
/// Clamps to `[k, g+k-1]` for boundary safety, ensuring that
/// `span - k` is always non-negative.
pub(crate) fn find_span(x: f64, grid: &[f64], g: usize, k: usize) -> usize {
    // Handle boundaries
    if x <= grid[k] {
        return k;
    }
    if x >= grid[g + k] {
        return g + k - 1;
    }
    // Linear scan (sufficient for typical grid sizes G <= 20)
    for i in k..g + k {
        if x < grid[i + 1] {
            return i;
        }
    }
    g + k - 1
}

/// Evaluate `k+1` non-zero B-spline basis functions at `x`.
///
/// Returns `(span, bases)` where `bases` has `k+1` values corresponding
/// to basis functions `B_{span-k}` through `B_{span}`.
///
/// Uses De Boor's triangular recurrence table.
pub(crate) fn evaluate_basis(x: f64, grid: &[f64], g: usize, k: usize) -> (usize, Vec<f64>) {
    let span = find_span(x, grid, g, k);

    // De Boor's triangular table — build up from order 0 to order k
    let mut bases = vec![0.0; k + 1];
    bases[0] = 1.0;

    for d in 1..=k {
        let mut new_bases = vec![0.0; d + 1];
        for j in 0..=d {
            let left_idx = span as i64 - d as i64 + j as i64;
            let right_idx = left_idx + 1;

            // Left contribution: B_{left, d-1} * (x - t_{left}) / (t_{left+d} - t_{left})
            if j > 0 && left_idx >= 0 {
                let li = left_idx as usize;
                let denom = grid[li + d] - grid[li];
                if denom.abs() > 1e-15 {
                    new_bases[j] += bases[j - 1] * (x - grid[li]) / denom;
                }
            }
            // Right contribution: B_{right, d-1} * (t_{right+d} - x) / (t_{right+d} - t_{right})
            if j < d && right_idx >= 0 {
                let ri = right_idx as usize;
                if ri + d < grid.len() {
                    let denom = grid[ri + d] - grid[ri];
                    if denom.abs() > 1e-15 {
                        new_bases[j] += bases[j] * (grid[ri + d] - x) / denom;
                    }
                }
            }
        }
        bases = new_bases;
    }

    (span, bases)
}

/// Evaluate derivatives of `k+1` non-zero basis functions at `x`.
///
/// Uses the standard derivative formula:
/// `B'_{i,k}(x) = k * (B_{i,k-1}(x) / (t_{i+k} - t_i) - B_{i+1,k-1}(x) / (t_{i+k+1} - t_{i+1}))`
///
/// Returns `(span, derivs)` where `derivs` has `k+1` values.
pub(crate) fn evaluate_basis_derivatives(
    x: f64,
    grid: &[f64],
    g: usize,
    k: usize,
) -> (usize, Vec<f64>) {
    if k == 0 {
        let span = find_span(x, grid, g, 0);
        return (span, vec![0.0]);
    }

    let span = find_span(x, grid, g, k);

    // We need B_{span-k, k-1} through B_{span, k-1} — that's k+1 values.
    // Order k-1 has only k non-zero values at a point, so one of the k+1
    // will be zero (the one outside the support).
    let mut bases_km1 = vec![0.0; k + 1];

    // Compute order-(k-1) bases using De Boor at the same span
    {
        let mut b = vec![0.0; k];
        b[0] = 1.0;
        for d in 1..k {
            let mut nb = vec![0.0; d + 1];
            for j in 0..=d {
                let left_idx = span as i64 - d as i64 + j as i64;
                let right_idx = left_idx + 1;
                if j > 0 && left_idx >= 0 {
                    let li = left_idx as usize;
                    let upper = grid.get(li + d).copied().unwrap_or(0.0);
                    let lower = grid.get(li).copied().unwrap_or(0.0);
                    let denom = upper - lower;
                    if denom.abs() > 1e-15 {
                        nb[j] += b[j - 1] * (x - lower) / denom;
                    }
                }
                if j < d && right_idx >= 0 {
                    let ri = right_idx as usize;
                    let upper = grid.get(ri + d).copied().unwrap_or(0.0);
                    let lower = grid.get(ri).copied().unwrap_or(0.0);
                    let denom = upper - lower;
                    if denom.abs() > 1e-15 {
                        nb[j] += b[j] * (upper - x) / denom;
                    }
                }
            }
            b = nb;
        }
        // b has k values: B_{span-k+1, k-1} through B_{span, k-1}
        // Map into bases_km1 with offset: index 0 = B_{span-k, k-1} = 0
        for (j, &val) in b.iter().enumerate() {
            bases_km1[j + 1] = val;
        }
    }

    // Apply the derivative formula
    let mut derivs = vec![0.0; k + 1];
    let kf = k as f64;

    for j in 0..=k {
        let idx = (span as i64 - k as i64 + j as i64) as usize;

        // Positive term: k * B_{idx, k-1}(x) / (t_{idx+k} - t_{idx})
        {
            let upper = grid.get(idx + k).copied().unwrap_or(0.0);
            let lower = grid.get(idx).copied().unwrap_or(0.0);
            let denom = upper - lower;
            if denom.abs() > 1e-15 {
                derivs[j] += kf * bases_km1[j] / denom;
            }
        }

        // Negative term: -k * B_{idx+1, k-1}(x) / (t_{idx+k+1} - t_{idx+1})
        if j + 1 < bases_km1.len() {
            let upper = grid.get(idx + k + 1).copied().unwrap_or(0.0);
            let lower = grid.get(idx + 1).copied().unwrap_or(0.0);
            let denom = upper - lower;
            if denom.abs() > 1e-15 {
                derivs[j] -= kf * bases_km1[j + 1] / denom;
            }
        }
    }

    (span, derivs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_construction() {
        let grid = make_grid(-1.0, 1.0, 5, 3);
        // Expected length: G + 2*k + 1 = 5 + 6 + 1 = 12
        assert_eq!(
            grid.len(),
            12,
            "grid length should be G + 2*k + 1 = 12, got {}",
            grid.len()
        );
        // First k knots should be low
        assert_eq!(grid[0], -1.0, "first knot should be -1.0");
        assert_eq!(grid[1], -1.0, "second knot should be -1.0");
        assert_eq!(grid[2], -1.0, "third knot should be -1.0");
        // Last k knots should be high
        assert_eq!(grid[9], 1.0, "knot [9] should be 1.0");
        assert_eq!(grid[10], 1.0, "knot [10] should be 1.0");
        assert_eq!(grid[11], 1.0, "knot [11] should be 1.0");
    }

    #[test]
    fn basis_partition_of_unity() {
        let g = 5;
        let k = 3;
        let grid = make_grid(-1.0, 1.0, g, k);
        // B-spline bases should sum to 1.0 at any interior point
        let test_points = [-0.8, -0.3, 0.0, 0.25, 0.7, 0.99];
        for &x in &test_points {
            let (_, bases) = evaluate_basis(x, &grid, g, k);
            let sum: f64 = bases.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "partition of unity violated at x={}: sum={}, expected 1.0",
                x,
                sum
            );
        }
    }

    #[test]
    fn basis_non_negative() {
        let g = 8;
        let k = 3;
        let grid = make_grid(0.0, 1.0, g, k);
        let test_points = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0];
        for &x in &test_points {
            let (_, bases) = evaluate_basis(x, &grid, g, k);
            for (b_idx, &val) in bases.iter().enumerate() {
                assert!(
                    val >= -1e-15,
                    "negative basis value at x={}, index {}: {}",
                    x,
                    b_idx,
                    val
                );
            }
        }
    }

    #[test]
    fn basis_boundary_safety() {
        let g = 5;
        let k = 3;
        let grid = make_grid(-1.0, 1.0, g, k);
        // Evaluate at boundaries and beyond — should not panic
        let boundary_points = [-1.0, 1.0, -2.0, 2.0, -1.0001, 1.0001];
        for &x in &boundary_points {
            let (span, bases) = evaluate_basis(x, &grid, g, k);
            assert!(
                span >= k,
                "span {} below k={} at x={}",
                span,
                k,
                x
            );
            assert!(
                span <= g + k - 1,
                "span {} above g+k-1={} at x={}",
                span,
                g + k - 1,
                x
            );
            assert_eq!(
                bases.len(),
                k + 1,
                "wrong number of basis values at x={}",
                x
            );
            // All values should be finite
            for &val in &bases {
                assert!(val.is_finite(), "non-finite basis value at x={}", x);
            }
        }
    }

    #[test]
    fn derivatives_finite() {
        let g = 5;
        let k = 3;
        let grid = make_grid(-1.0, 1.0, g, k);
        let test_points = [-1.0, -0.5, 0.0, 0.5, 1.0, -0.99, 0.99];
        for &x in &test_points {
            let (span, derivs) = evaluate_basis_derivatives(x, &grid, g, k);
            assert!(
                span >= k,
                "derivative span {} below k={} at x={}",
                span,
                k,
                x
            );
            assert_eq!(
                derivs.len(),
                k + 1,
                "wrong number of derivative values at x={}",
                x
            );
            for (d_idx, &val) in derivs.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "non-finite derivative at x={}, index {}: {}",
                    x,
                    d_idx,
                    val
                );
            }
        }
    }
}
