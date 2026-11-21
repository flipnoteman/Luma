//! Tridiagonal and block-tridiagonal solves along an axis, against a dense `f64` solve of every line.

mod common;

use common::{assert_close, values};
use luma::{
    solve_block_tridiagonal, solve_block_tridiagonal_from, solve_block_tridiagonal_parts, solve_block_tridiagonal_parts_from,
    compile_until_all, solve_tridiagonal, solve_tridiagonal_from, solve_tridiagonal_parts, solve_tridiagonal_parts_from, Array, Context, LumaError, Until,
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

#[test]
fn long_lines_of_every_parity_satisfy_their_equations() {
    // Lengths around powers of two exercise every level of the reduction, forward and backward.
    let ctx = Context::new().unwrap();
    let lines = 3;
    for n in [1usize, 2, 3, 7, 1023, 1024, 1025, 4099] {
        let sys = scalar_system(n * lines, n as u64);
        let arr = |d: &[f32]| Array::from_slice_on(&ctx, d, [n, lines]).unwrap();
        let (lo, diag, up, rhs) = (arr(&sys.lo), arr(&sys.diag), arr(&sys.up), arr(&sys.rhs));
        let got = solve_tridiagonal(0, &lo, &diag, &up, &rhs).to_vec().unwrap();
        let mut worst = 0.0f64;
        for l in 0..lines {
            for s in 0..n {
                let q = l * n + s;
                let mut e = f64::from(sys.diag[q]) * f64::from(got[q]) - f64::from(sys.rhs[q]);
                if s > 0 {
                    e += f64::from(sys.lo[q]) * f64::from(got[q - 1]);
                }
                if s + 1 < n {
                    e += f64::from(sys.up[q]) * f64::from(got[q + 1]);
                }
                worst = worst.max(e.abs());
            }
        }
        assert!(worst < 1e-4, "n {n}: residual {worst}");
    }
}

/// The entries of the blocks of `lo`, `diag`, `up` and the components of `rhs` (the host
/// layout of the stacked API) as the parts of the parts API, as arrays of `grid`; every other part
/// is an expression (`2 * half`).
fn parts_of(ctx: &Context, grid: [usize; 2], b: usize, host: &[Vec<f32>; 4], lazy_every_other: bool) -> Vec<Array<f32>> {
    let cells = grid[0] * grid[1];
    let bb = b * b;
    let mut parts = Vec::new();
    for (which, width) in [(0, bb), (1, bb), (2, bb), (3, b)] {
        for e in 0..width {
            let lazy = lazy_every_other && (which * 7 + e) % 2 == 1;
            let entry: Vec<f32> = (0..cells).map(|q| host[which][q * width + e] * if lazy { 0.5 } else { 1.0 }).collect();
            let array = Array::from_slice_on(ctx, &entry, grid).unwrap();
            parts.push(if lazy { &array * 2.0 } else { array });
        }
    }
    parts
}

fn block_system(b: usize, cells: usize, seed: u64) -> [Vec<f32>; 4] {
    let bb = b * b;
    let mut diag = values(cells * bb, seed, -1.0, 1.0);
    for q in 0..cells {
        for i in 0..b {
            diag[q * bb + i * b + i] += 8.0;
        }
    }
    [values(cells * bb, seed + 1, -1.0, 1.0), diag, values(cells * bb, seed + 2, -1.0, 1.0), values(cells * b, seed + 3, -1.0, 1.0)]
}

#[test]
fn block_parts_equal_the_stacked_solve() {
    let ctx = Context::new().unwrap();
    let grid = [30usize, 5];
    for b in 1..=4usize {
        let host = block_system(b, 150, 100 + b as u64);
        for lazy in [false, true] {
            let parts = parts_of(&ctx, grid, b, &host, lazy);
            let refs: Vec<&Array<f32>> = parts.iter().collect();
            let bb = b * b;
            let stacked = |h: &Vec<f32>, width: usize| Array::from_slice_on(&ctx, h, [width, grid[0], grid[1]]).unwrap();
            for axis in 0..2 {
                let want = solve_block_tridiagonal(b, axis + 1, &stacked(&host[0], bb), &stacked(&host[1], bb), &stacked(&host[2], bb), &stacked(&host[3], b)).to_vec().unwrap();
                let got = match b {
                    1 => solve_block_tridiagonal_parts(axis, [refs[0]], [refs[1]], [refs[2]], [refs[3]]).map(|x| x.to_vec().unwrap()).to_vec(),
                    2 => solve_block_tridiagonal_parts(axis, std::array::from_fn::<_, 4, _>(|k| refs[k]), std::array::from_fn::<_, 4, _>(|k| refs[4 + k]), std::array::from_fn::<_, 4, _>(|k| refs[8 + k]), [refs[12], refs[13]])
                        .map(|x| x.to_vec().unwrap())
                        .to_vec(),
                    3 => solve_block_tridiagonal_parts(axis, std::array::from_fn::<_, 9, _>(|k| refs[k]), std::array::from_fn::<_, 9, _>(|k| refs[9 + k]), std::array::from_fn::<_, 9, _>(|k| refs[18 + k]), [refs[27], refs[28], refs[29]])
                        .map(|x| x.to_vec().unwrap())
                        .to_vec(),
                    _ => solve_block_tridiagonal_parts(axis, std::array::from_fn::<_, 16, _>(|k| refs[k]), std::array::from_fn::<_, 16, _>(|k| refs[16 + k]), std::array::from_fn::<_, 16, _>(|k| refs[32 + k]), std::array::from_fn::<_, 4, _>(|k| refs[48 + k]))
                        .map(|x| x.to_vec().unwrap())
                        .to_vec(),
                };
                assert_eq!(got.len(), b);
                for (c, component) in got.iter().enumerate() {
                    let expected: Vec<f32> = (0..150).map(|q| want[q * b + c]).collect();
                    assert_close(component, &expected, 1e-4, 1e-5);
                }
                let (host_x, residual) = reference(b, [30, 5, 1, 1], axis, &host[0], &host[1], &host[2], &host[3], None);
                assert!(residual < 1e-9);
                for (c, component) in got.iter().enumerate() {
                    let expected: Vec<f32> = (0..150).map(|q| host_x[q * b + c]).collect();
                    assert_close(component, &expected, 1e-4, 1e-5);
                }
            }
        }
    }
}

#[test]
fn block_parts_from_a_start_cell() {
    let ctx = Context::new().unwrap();
    let (b, grid) = (2usize, [30usize, 5]);
    let host = block_system(b, 150, 7);
    let parts = parts_of(&ctx, grid, b, &host, true);
    let r: Vec<&Array<f32>> = parts.iter().collect();
    let start: Vec<u32> = (0..5).map(|l| (l * 7 % 31) as u32).collect();
    let start_a = Array::from_slice_on(&ctx, &start, 5).unwrap();
    let got = solve_block_tridiagonal_parts_from(
        0,
        std::array::from_fn::<_, 4, _>(|k| r[k]),
        std::array::from_fn::<_, 4, _>(|k| r[4 + k]),
        std::array::from_fn::<_, 4, _>(|k| r[8 + k]),
        [r[12], r[13]],
        &start_a,
    );
    let (want, _) = reference(b, [30, 5, 1, 1], 0, &host[0], &host[1], &host[2], &host[3], Some(&start));
    for (c, component) in got.iter().enumerate() {
        let expected: Vec<f32> = (0..150).map(|q| want[q * b + c]).collect();
        assert_close(&component.to_vec().unwrap(), &expected, 1e-4, 1e-5);
    }
}

#[test]
fn tridiagonal_parts_equal_separate_solves() {
    let ctx = Context::new().unwrap();
    let dims = [40usize, 7];
    let (k, w) = (scalar_system(280, 1), scalar_system(280, 50));
    let arr = |d: &[f32]| Array::from_slice_on(&ctx, d, dims).unwrap();
    let (klo, kdiag, kup, krhs) = (arr(&k.lo), arr(&k.diag), arr(&k.up), arr(&k.rhs));
    let (wlo, wdiag, wup, wrhs) = (arr(&w.lo), arr(&w.diag), arr(&w.up), arr(&w.rhs));
    for axis in 0..2 {
        // Lazy parts: nothing is evaluated before the solve.
        let [x, y] = solve_tridiagonal_parts(axis, [(&(&klo * 1.0), &(&kdiag + 0.0), &kup, &(&krhs * 2.0)), (&wlo, &wdiag, &(&wup * 1.0), &wrhs)]);
        let want_k = solve_tridiagonal(axis, &klo, &kdiag, &kup, &(&krhs * 2.0)).to_vec().unwrap();
        let want_w = solve_tridiagonal(axis, &wlo, &wdiag, &wup, &wrhs).to_vec().unwrap();
        assert_close(&x.to_vec().unwrap(), &want_k, 1e-4, 1e-5);
        assert_close(&y.to_vec().unwrap(), &want_w, 1e-4, 1e-5);
    }
    let start = Array::from_slice_on(&ctx, &[0u32, 3, 40, 5, 39, 1, 100], 7).unwrap();
    let [x, y] = solve_tridiagonal_parts_from(0, [(&klo, &kdiag, &kup, &krhs), (&wlo, &wdiag, &wup, &wrhs)], &start);
    let want_k = solve_tridiagonal_from(0, &klo, &kdiag, &kup, &krhs, &start).to_vec().unwrap();
    let want_w = solve_tridiagonal_from(0, &wlo, &wdiag, &wup, &wrhs, &start).to_vec().unwrap();
    assert_close(&x.to_vec().unwrap(), &want_k, 1e-4, 1e-5);
    assert_close(&y.to_vec().unwrap(), &want_w, 1e-4, 1e-5);
}

/// A scalar solve of an array with a batch axis (`[2, ni, nj]` along axis 1 or 2) solves every
/// batch element's lines on its own.
#[test]
fn scalar_solve_batches_over_a_leading_axis() {
    let ctx = Context::new().unwrap();
    let (ni, nj) = (30usize, 9usize);
    let cells = 2 * ni * nj;
    let sys = scalar_system(cells, 3);
    let arr = |d: &[f32]| Array::from_slice_on(&ctx, d, [2, ni, nj]).unwrap();
    let (lo, diag, up, rhs) = (arr(&sys.lo), arr(&sys.diag), arr(&sys.up), arr(&sys.rhs));
    for axis in 1..=2usize {
        let got = solve_tridiagonal(axis, &lo, &diag, &up, &rhs).to_vec().unwrap();
        let (want, residual) = reference(1, [2, ni, nj, 1], axis, &sys.lo, &sys.diag, &sys.up, &sys.rhs, None);
        assert!(residual < 1e-9);
        assert_close(&got, &want, 1e-4, 1e-5);
    }
}

#[test]
fn parts_errors_poison_every_result() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &values(40 * 7, 1, 1.0, 2.0), [40, 7]).unwrap();
    let other = Array::from_slice_on(&ctx, &values(40 * 6, 1, 1.0, 2.0), [40, 6]).unwrap();
    let [x, y] = solve_tridiagonal_parts(0, [(&a, &a, &a, &a), (&a, &other, &a, &a)]);
    for r in [x, y] {
        let err = r.to_vec().unwrap_err();
        assert!(matches!(err.root(), LumaError::ShapeMismatch { .. }), "{err}");
    }
    let [x] = solve_tridiagonal_parts(4, [(&a, &a, &a, &a)]);
    assert!(matches!(x.to_vec().unwrap_err().root(), LumaError::InvalidShape(_)));
    // A poisoned part reports its own error.
    let bad = solve_tridiagonal(0, &a, &other, &a, &a);
    let [x] = solve_tridiagonal_parts(0, [(&a, &a, &a, &bad)]);
    assert!(matches!(x.to_vec().unwrap_err().root(), LumaError::ShapeMismatch { .. }));
    // Blocks: four dimensions.
    let four = Array::from_slice_on(&ctx, &values(16, 1, 1.0, 2.0), [2, 2, 2, 2]).unwrap();
    let [x, _] = solve_block_tridiagonal_parts(0, [&four; 4], [&four; 4], [&four; 4], [&four; 2]);
    assert!(matches!(x.to_vec().unwrap_err().root(), LumaError::InvalidShape(_)));
}

/// Block systems of every size on grids whose lines do not fill the vectors and groups of a native
/// solve (one, three, four, five, nine lines; rows of one, two, three cells), along every axis of
/// a three-dimensional grid, stacked and from a start cell.
#[test]
fn block_systems_of_every_size_on_awkward_grids() {
    let ctx = Context::new().unwrap();
    for b in 1..=8usize {
        for grid in [[1usize, 1, 1, 1], [3, 1, 2, 1], [5, 3, 4, 1], [2, 9, 1, 1], [7, 5, 3, 1], [4, 4, 3, 1], [13, 2, 5, 1]] {
            let cells: usize = grid.iter().product();
            let [lo, diag, up, rhs] = block_system(b, cells, 500 + b as u64);
            let big = [b * b, grid[0], grid[1], grid[2]];
            let small = [b, grid[0], grid[1], grid[2]];
            let from = |host: &[f32], dims: [usize; 4]| Array::from_slice_on(&ctx, host, dims).unwrap();
            let (lo_a, diag_a, up_a, rhs_a) = (from(&lo, big), from(&diag, big), from(&up, big), from(&rhs, small));
            for axis in 1..=3usize {
                let n = grid[axis - 1];
                let lines = cells / n;
                let tol = if b >= 7 { 5e-4 } else { 1e-4 };
                let got = solve_block_tridiagonal(b, axis, &lo_a, &diag_a, &up_a, &rhs_a).to_vec().unwrap();
                let (want, residual) = reference(b, grid, axis - 1, &lo, &diag, &up, &rhs, None);
                assert!(residual < 1e-9);
                assert_close(&got, &want, tol, 1e-5);
                // Every line starts at its own cell, some past the end.
                let start: Vec<u32> = (0..lines).map(|l| (l * 2 % (n + 2)) as u32).collect();
                let start_a = Array::from_slice_on(&ctx, &start, lines).unwrap();
                let got = solve_block_tridiagonal_from(b, axis, &lo_a, &diag_a, &up_a, &rhs_a, &start_a).to_vec().unwrap();
                let (want, _) = reference(b, grid, axis - 1, &lo, &diag, &up, &rhs, Some(&start));
                assert_close(&got, &want, tol, 1e-5);
            }
        }
    }
}

macro_rules! parts_solve {
    ($b:literal, $bb:literal, $axis:expr, $refs:expr) => {{
        let r: &[&Array<f32>] = $refs;
        solve_block_tridiagonal_parts::<$bb, $b>(
            $axis,
            std::array::from_fn::<_, $bb, _>(|k| r[k]),
            std::array::from_fn::<_, $bb, _>(|k| r[$bb + k]),
            std::array::from_fn::<_, $bb, _>(|k| r[2 * $bb + k]),
            std::array::from_fn::<_, $b, _>(|k| r[3 * $bb + k]),
        )
        .map(|x| x.to_vec().unwrap())
        .to_vec()
    }};
}

/// The solve on parts for every block size (the native solve takes all of 1 to 8) on awkward
/// grids, evaluated and lazy, along both axes.
#[test]
fn block_parts_of_every_size_on_awkward_grids() {
    let ctx = Context::new().unwrap();
    for b in 1..=8usize {
        for grid in [[1usize, 1], [3, 2], [4, 5], [5, 4], [9, 7], [13, 3], [8, 8]] {
            let cells = grid[0] * grid[1];
            let host = block_system(b, cells, 700 + b as u64);
            for lazy in [false, true] {
                let parts = parts_of(&ctx, grid, b, &host, lazy);
                let refs: Vec<&Array<f32>> = parts.iter().collect();
                for axis in 0..2usize {
                    let got = match b {
                        1 => parts_solve!(1, 1, axis, &refs),
                        2 => parts_solve!(2, 4, axis, &refs),
                        3 => parts_solve!(3, 9, axis, &refs),
                        4 => parts_solve!(4, 16, axis, &refs),
                        5 => parts_solve!(5, 25, axis, &refs),
                        6 => parts_solve!(6, 36, axis, &refs),
                        7 => parts_solve!(7, 49, axis, &refs),
                        _ => parts_solve!(8, 64, axis, &refs),
                    };
                    let (want, residual) = reference(b, [grid[0], grid[1], 1, 1], axis, &host[0], &host[1], &host[2], &host[3], None);
                    assert!(residual < 1e-9);
                    let tol = if b >= 7 { 5e-4 } else { 1e-4 };
                    for (c, component) in got.iter().enumerate() {
                        let expected: Vec<f32> = (0..cells).map(|q| want[q * b + c]).collect();
                        assert_close(component, &expected, tol, 1e-5);
                    }
                }
            }
        }
    }
}

/// A traced step with a block solve, replayed many times by `compile_until_all`, matches the same
/// steps run eagerly one by one (stacked and on parts), also from another initial state.
#[test]
fn a_traced_block_solve_replays_like_the_eager_loop() {
    let ctx = Context::new().unwrap();
    let (b, grid) = (5usize, [24usize, 9]);
    let cells = grid[0] * grid[1];
    let [lo, diag, up, forcing] = block_system(b, cells, 900);
    let big = [b * b, grid[0], grid[1]];
    let small = [b, grid[0], grid[1]];
    let (lo_a, diag_a, up_a) = (
        Array::from_slice_on(&ctx, &lo, big).unwrap(),
        Array::from_slice_on(&ctx, &diag, big).unwrap(),
        Array::from_slice_on(&ctx, &up, big).unwrap(),
    );
    let force_a = Array::from_slice_on(&ctx, &forcing, small).unwrap();
    // The parts of the same blocks: entry `e` of every cell.
    let part = |host: &[f32], width: usize, e: usize| -> Array<f32> {
        let entry: Vec<f32> = (0..cells).map(|q| host[q * width + e]).collect();
        Array::from_slice_on(&ctx, &entry, grid).unwrap()
    };
    let (lo_p, diag_p, up_p): (Vec<_>, Vec<_>, Vec<_>) = ((0..25).map(|e| part(&lo, 25, e)).collect(), (0..25).map(|e| part(&diag, 25, e)).collect(), (0..25).map(|e| part(&up, 25, e)).collect());
    let stacked_step = |x: &Array<f32>| solve_block_tridiagonal(b, 1, &lo_a, &diag_a, &up_a, &(x * 0.5 + &force_a));
    let parts_step = |x: &Array<f32>| {
        let comp: Vec<Array<f32>> = (0..b).map(|c| x.slice(0, c..c + 1).reshape(grid)).collect();
        // Lazy right-hand sides: they depend on the state, so they are computed inside the step.
        let rhs: Vec<Array<f32>> = comp.iter().enumerate().map(|(c, v)| v * 0.5 + &part(&forcing, b, c)).collect();
        let (l, d, u) = (lo_p.iter().collect::<Vec<_>>(), diag_p.iter().collect::<Vec<_>>(), up_p.iter().collect::<Vec<_>>());
        let solved = solve_block_tridiagonal_parts::<25, 5>(
            0,
            std::array::from_fn(|k| l[k]),
            std::array::from_fn(|k| d[k]),
            std::array::from_fn(|k| u[k]),
            std::array::from_fn(|k| &rhs[k]),
        );
        luma::stack(0, solved.each_ref()).reshape(small)
    };
    let steps = 12;
    for (name, step) in [("stacked", &stacked_step as &dyn Fn(&Array<f32>) -> Array<f32>), ("parts", &parts_step as &dyn Fn(&Array<f32>) -> Array<f32>)] {
        for seed in [11u64, 12] {
            let init = || Array::from_slice_on(&ctx, &values(cells * b, seed, -1.0, 1.0), small).unwrap();
            let mut eager = init();
            for _ in 0..steps {
                eager = step(&eager).cache();
                eager.eval().unwrap();
            }
            let mut compiled = compile_until_all(&(init(),), Until::new(steps), |s: &(Array<f32>,), _: &Array<f32>| (step(&s.0),)).unwrap();
            let run = compiled.run((init(),)).unwrap();
            assert_eq!(run.steps, steps, "{name}");
            assert_close(&run.state.0.to_vec().unwrap(), &eager.to_vec().unwrap(), 1e-4, 1e-5);
        }
    }
}
