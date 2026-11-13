//! Tridiagonal and block-tridiagonal solves along an axis, against a dense `f64` solve of every line.

mod common;

use common::{assert_close, values};
use luma::{
    solve_block_tridiagonal, solve_block_tridiagonal_from, solve_tridiagonal, solve_tridiagonal_from, Array, Context,
    LumaError,
};

/// Solves `m x = r` (`n x n`, row-major, `k` right-hand sides in `r`) by Gaussian elimination with
/// partial pivoting in `f64`.
fn dense_solve(mut m: Vec<f64>, mut r: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| m[i * n + col].abs().total_cmp(&m[j * n + col].abs())).unwrap();
        for j in 0..n {
            m.swap(col * n + j, pivot * n + j);
        }
        r.swap(col, pivot);
        for i in col + 1..n {
            let f = m[i * n + col] / m[col * n + col];
            for j in col..n {
                m[i * n + j] -= f * m[col * n + j];
            }
            r[i] -= f * r[col];
        }
    }
    for i in (0..n).rev() {
        let mut x = r[i];
        for j in i + 1..n {
            x -= m[i * n + j] * r[j];
        }
        r[i] = x / m[i * n + i];
    }
    r
}

/// The host reference. `grid` are the dims of the grid, `axis` the axis of the grid the lines run
/// along, `b` the block size; coefficient arrays hold `b * b` values per cell, `rhs` `b`.
/// `start[l]` is the first cell of line `l` (0 if `None`). Returns the solution (`b` per cell) and
/// the largest residual of the equations.
#[allow(clippy::too_many_arguments)]
fn reference(
    b: usize,
    grid: [usize; 4],
    axis: usize,
    lo: &[f32],
    diag: &[f32],
    up: &[f32],
    rhs: &[f32],
    start: Option<&[u32]>,
) -> (Vec<f32>, f64) {
    let cells: usize = grid.iter().product();
    let n = grid[axis];
    let stride: usize = grid[..axis].iter().product();
    let lines = cells / n;
    let mut x = vec![0.0f32; cells * b];
    let mut residual = 0.0f64;
    for l in 0..lines {
        let base = (l % stride) + (l / stride) * stride * n;
        let s0 = start.map_or(0, |s| (s[l] as usize).min(n));
        let m = n - s0;
        if m == 0 {
            continue;
        }
        let size = m * b;
        let mut matrix = vec![0.0f64; size * size];
        let mut r = vec![0.0f64; size];
        for s in 0..m {
            let q = base + (s + s0) * stride;
            for i in 0..b {
                r[s * b + i] = f64::from(rhs[q * b + i]);
                for j in 0..b {
                    matrix[(s * b + i) * size + s * b + j] = f64::from(diag[q * b * b + i * b + j]);
                    if s > 0 {
                        matrix[(s * b + i) * size + (s - 1) * b + j] = f64::from(lo[q * b * b + i * b + j]);
                    }
                    if s + 1 < m {
                        matrix[(s * b + i) * size + (s + 1) * b + j] = f64::from(up[q * b * b + i * b + j]);
                    }
                }
            }
        }
        let solution = dense_solve(matrix.clone(), r.clone(), size);
        for i in 0..size {
            let mut sum = -r[i];
            for j in 0..size {
                sum += matrix[i * size + j] * solution[j];
            }
            residual = residual.max(sum.abs());
        }
        for s in 0..m {
            let q = base + (s + s0) * stride;
            for i in 0..b {
                x[q * b + i] = solution[s * b + i] as f32;
            }
        }
    }
    (x, residual)
}

struct Scalar {
    lo: Vec<f32>,
    diag: Vec<f32>,
    up: Vec<f32>,
    rhs: Vec<f32>,
}

fn scalar_system(n: usize, seed: u64) -> Scalar {
    Scalar {
        lo: values(n, seed, -1.0, 1.0),
        diag: values(n, seed + 1, -1.0, 1.0).iter().map(|v| 4.0 + v).collect(),
        up: values(n, seed + 2, -1.0, 1.0),
        rhs: values(n, seed + 3, -1.0, 1.0),
    }
}

#[test]
fn scalar_systems_along_each_axis() {
    let ctx = Context::new().unwrap();
    let dims = [40usize, 7];
    let sys = scalar_system(40 * 7, 10);
    let arr = |d: &[f32]| Array::from_slice_on(&ctx, d, dims).unwrap();
    let (lo, diag, up, rhs) = (arr(&sys.lo), arr(&sys.diag), arr(&sys.up), arr(&sys.rhs));
    for axis in 0..2 {
        let got = solve_tridiagonal(axis, &lo, &diag, &up, &rhs);
        assert_eq!(got.shape().dims(), [40, 7, 1, 1]);
        let got = got.to_vec().unwrap();
        let (want, residual) = reference(1, [40, 7, 1, 1], axis, &sys.lo, &sys.diag, &sys.up, &sys.rhs, None);
        assert!(residual < 1e-9, "reference residual {residual}");
        assert_close(&got, &want, 1e-4, 1e-5);

        // The equations hold on the host in f64 over the GPU solution.
        let n = dims[axis];
        let stride: usize = dims[..axis].iter().product();
        let mut worst = 0.0f64;
        for l in 0..(40 * 7 / n) {
            let base = (l % stride) + (l / stride) * stride * n;
            for s in 0..n {
                let q = base + s * stride;
                let mut e = f64::from(sys.diag[q]) * f64::from(got[q]) - f64::from(sys.rhs[q]);
                if s > 0 {
                    e += f64::from(sys.lo[q]) * f64::from(got[q - stride]);
                }
                if s + 1 < n {
                    e += f64::from(sys.up[q]) * f64::from(got[q + stride]);
                }
                worst = worst.max(e.abs());
            }
        }
        assert!(worst < 1e-4, "axis {axis}: residual {worst}");
    }
}

#[test]
fn block_systems() {
    let ctx = Context::new().unwrap();
    for (b, grid, axes) in [
        (4usize, [30usize, 5, 1, 1], vec![1usize, 2]),
        (2, [30, 5, 1, 1], vec![1, 2]),
        (3, [17, 6, 1, 1], vec![1, 2]),
        (5, [12, 4, 1, 1], vec![1, 2]),
        (6, [9, 3, 1, 1], vec![1, 2]),
    ] {
        let cells = grid[0] * grid[1];
        let bb = b * b;
        let mut diag = values(cells * bb, 20 + b as u64, -1.0, 1.0);
        for q in 0..cells {
            for i in 0..b {
                diag[q * bb + i * b + i] += 8.0;
            }
        }
        let lo = values(cells * bb, 30, -1.0, 1.0);
        let up = values(cells * bb, 31, -1.0, 1.0);
        let rhs = values(cells * b, 32, -1.0, 1.0);
        let big = [bb, grid[0], grid[1]];
        let small = [b, grid[0], grid[1]];
        let (lo_a, diag_a, up_a) = (
            Array::from_slice_on(&ctx, &lo, big).unwrap(),
            Array::from_slice_on(&ctx, &diag, big).unwrap(),
            Array::from_slice_on(&ctx, &up, big).unwrap(),
        );
        let rhs_a = Array::from_slice_on(&ctx, &rhs, small).unwrap();
        for &axis in &axes {
            let got = solve_block_tridiagonal(b, axis, &lo_a, &diag_a, &up_a, &rhs_a);
            assert_eq!(got.shape().dims(), [b, grid[0], grid[1], 1]);
            let got = got.to_vec().unwrap();
            let (want, residual) = reference(b, grid, axis - 1, &lo, &diag, &up, &rhs, None);
            assert!(residual < 1e-9);
            assert_close(&got, &want, 1e-4, 1e-5);
        }
    }
}

/// Diagonal blocks that are (scaled) anti-identities plus noise with an exactly zero diagonal, so
/// that the pivot of every column is in another row than the diagonal (without pivoting the
/// first division is by zero) and every elimination step exchanges rows; with `starts`
/// the systems begin at different cells.
#[test]
fn block_systems_that_exchange_rows_in_every_column() {
    let ctx = Context::new().unwrap();
    for b in [2usize, 3, 4, 5, 6] {
        let grid = [14usize, 5, 1, 1];
        let cells = grid[0] * grid[1];
        let bb = b * b;
        let mut diag = values(cells * bb, 70 + b as u64, -0.5, 0.5);
        for q in 0..cells {
            for i in 0..b {
                diag[q * bb + i * b + (b - 1 - i)] += 5.0;
                diag[q * bb + i * b + i] = if i == b - 1 - i { 5.0 } else { 0.0 };
            }
        }
        let lo = values(cells * bb, 80, -0.3, 0.3);
        let up = values(cells * bb, 81, -0.3, 0.3);
        let rhs = values(cells * b, 82, -1.0, 1.0);
        let big = [bb, grid[0], grid[1]];
        let arrays = (
            Array::from_slice_on(&ctx, &lo, big).unwrap(),
            Array::from_slice_on(&ctx, &diag, big).unwrap(),
            Array::from_slice_on(&ctx, &up, big).unwrap(),
            Array::from_slice_on(&ctx, &rhs, [b, grid[0], grid[1]]).unwrap(),
        );
        let got = solve_block_tridiagonal(b, 1, &arrays.0, &arrays.1, &arrays.2, &arrays.3).to_vec().unwrap();
        let (want, residual) = reference(b, grid, 0, &lo, &diag, &up, &rhs, None);
        assert!(residual < 1e-9, "block {b}");
        assert_close(&got, &want, 1e-3, 1e-4);

        let start: Vec<u32> = (0..grid[1]).map(|l| (l * 3 % 16) as u32).collect();
        let start_a = Array::from_slice_on(&ctx, &start, grid[1]).unwrap();
        let got = solve_block_tridiagonal_from(b, 1, &arrays.0, &arrays.1, &arrays.2, &arrays.3, &start_a).to_vec().unwrap();
        let (want, _) = reference(b, grid, 0, &lo, &diag, &up, &rhs, Some(&start));
        assert_close(&got, &want, 1e-3, 1e-4);
    }
}

#[test]
fn systems_starting_at_a_cell() {
    let ctx = Context::new().unwrap();
    let dims = [40usize, 7];
    let sys = scalar_system(40 * 7, 40);
    let arr = |d: &[f32]| Array::from_slice_on(&ctx, d, dims).unwrap();
    let (lo, diag, up, rhs) = (arr(&sys.lo), arr(&sys.diag), arr(&sys.up), arr(&sys.rhs));
    let start = [0u32, 3, 40, 5, 39, 1, 100];
    let start_a = Array::from_slice_on(&ctx, &start, 7).unwrap();
    let got = solve_tridiagonal_from(0, &lo, &diag, &up, &rhs, &start_a).to_vec().unwrap();
    let (want, _) = reference(1, [40, 7, 1, 1], 0, &sys.lo, &sys.diag, &sys.up, &sys.rhs, Some(&start));
    assert_close(&got, &want, 1e-4, 1e-5);
    // Cells before the start are exactly zero; the line starting at 3 equals the truncated system.
    assert!(got[40..43].iter().all(|&v| v == 0.0));
    assert!(got[80..120].iter().all(|&v| v == 0.0));

    // Along the other axis: 40 lines of 7 cells.
    let start: Vec<u32> = (0..40).map(|l| (l % 9) as u32).collect();
    let start_a = Array::from_slice_on(&ctx, &start, 40).unwrap();
    let got = solve_tridiagonal_from(1, &lo, &diag, &up, &rhs, &start_a).to_vec().unwrap();
    let (want, _) = reference(1, [40, 7, 1, 1], 1, &sys.lo, &sys.diag, &sys.up, &sys.rhs, Some(&start));
    assert_close(&got, &want, 1e-4, 1e-5);

    // Blocks.
    let (b, grid) = (2usize, [30usize, 5, 1, 1]);
    let cells = 150;
    let mut diag = values(cells * 4, 50, -1.0, 1.0);
    for q in 0..cells {
        diag[q * 4] += 8.0;
        diag[q * 4 + 3] += 8.0;
    }
    let lo = values(cells * 4, 51, -1.0, 1.0);
    let up = values(cells * 4, 52, -1.0, 1.0);
    let rhs = values(cells * b, 53, -1.0, 1.0);
    let start: Vec<u32> = (0..30).map(|l| (l % 7) as u32 * 5).collect();
    let big = [4, 30, 5];
    let got = solve_block_tridiagonal_from(
        b,
        2,
        &Array::from_slice_on(&ctx, &lo, big).unwrap(),
        &Array::from_slice_on(&ctx, &diag, big).unwrap(),
        &Array::from_slice_on(&ctx, &up, big).unwrap(),
        &Array::from_slice_on(&ctx, &rhs, [2, 30, 5]).unwrap(),
        &Array::from_slice_on(&ctx, &start, 30).unwrap(),
    )
    .to_vec()
    .unwrap();
    let (want, _) = reference(b, grid, 1, &lo, &diag, &up, &rhs, Some(&start));
    assert_close(&got, &want, 1e-4, 1e-5);
}

#[test]
fn lazy_operands_and_errors() {
    let ctx = Context::new().unwrap();
    let sys = scalar_system(40 * 7, 60);
    let arr = |d: &[f32]| Array::from_slice_on(&ctx, d, [40, 7]).unwrap();
    let (lo, diag, up, rhs) = (arr(&sys.lo), arr(&sys.diag), arr(&sys.up), arr(&sys.rhs));
    let got = solve_tridiagonal(0, &(&lo * 1.0), &(&diag + 0.0), &up, &(&rhs * 1.0)).to_vec().unwrap();
    let want = solve_tridiagonal(0, &lo, &diag, &up, &rhs).to_vec().unwrap();
    assert_eq!(got, want);

    let other = Array::from_slice_on(&ctx, &values(40 * 6, 1, 0.0, 1.0), [40, 6]).unwrap();
    let err = solve_tridiagonal(0, &lo, &other, &up, &rhs).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "{err}");
    let err = solve_tridiagonal(4, &lo, &diag, &up, &rhs).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
    let err = solve_block_tridiagonal(9, 1, &lo, &diag, &up, &rhs).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
    let short = Array::from_slice_on(&ctx, &[0u32, 1], 2).unwrap();
    let err = solve_tridiagonal_from(0, &lo, &diag, &up, &rhs, &short).to_vec().unwrap_err();
    assert!(matches!(err.root(), LumaError::InvalidShape(_)), "{err}");
}
