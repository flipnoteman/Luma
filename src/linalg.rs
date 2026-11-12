//! Line solvers: one tridiagonal or block-tridiagonal system per grid line along an axis.
//!
//! Two algorithms: the block Thomas algorithm runs one invocation per line (see
//! [`Builtin::Tridiagonal`](crate::builtin::Builtin)), `O(n)` work per line with no intermediate
//! passes over memory, which suits many lines; for few, long lines one workgroup solves a line by
//! cyclic reduction in workgroup memory (see [`crate::cyclic`]). The choice is automatic
//! (`LUMA_TRIDIAGONAL=thomas|cyclic` forces one). Operands are evaluated first (arrays split over
//! several buffers fail with [`LumaError::Chunked`]); the solve is recorded, not waited for.

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::builtin::{tridiagonal_workgroup, Builtin};
use crate::cyclic;
use crate::context::{RawArg, Scratch};
use crate::error::LumaError;
use crate::ops::{fold_workgroups, workgroups_1d, Outcome};
use crate::shape::Shape;
use crate::tuning::Tuning;

/// Solves `lo[s] x[s-1] + diag[s] x[s] + up[s] x[s+1] = rhs[s]` along `axis` for every line of
/// the four arrays, which share one shape. `lo` at the first and `up` at the last cell of a line
/// are ignored. The result has the shape of `rhs`.
///
/// A singular pivot gives `Inf`/`NaN` in that line; it is not checked.
///
/// # Errors
///
/// The result is poisoned if the arrays differ in context or shape or `axis >= 4`.
#[track_caller]
pub fn solve_tridiagonal(
    axis: usize,
    lo: &Array<f32>,
    diag: &Array<f32>,
    up: &Array<f32>,
    rhs: &Array<f32>,
) -> Array<f32> {
    let location = Location::caller();
    let result = run(None, None, axis, [lo, diag, up, rhs], None, location);
    Array::resolve(rhs.context(), rhs.shape(), result, "solve_tridiagonal", location)
}

/// [`solve_tridiagonal`] where the system of line `l` begins at cell `start[l]` (clamped to the
/// extent): the cells before it get `x = 0`. `start` has one element per line; lines are numbered
/// with the dimensions below `axis` varying fastest.
#[track_caller]
pub fn solve_tridiagonal_from(
    axis: usize,
    lo: &Array<f32>,
    diag: &Array<f32>,
    up: &Array<f32>,
    rhs: &Array<f32>,
    start: &Array<u32>,
) -> Array<f32> {
    let location = Location::caller();
    let result = run(None, None, axis, [lo, diag, up, rhs], Some(start), location);
    Array::resolve(rhs.context(), rhs.shape(), result, "solve_tridiagonal", location)
}

/// Solves a block-tridiagonal system (`block` x `block` blocks, `1 <= block <= 8`) along `axis`
/// for every line of the grid. `rhs` has dimensions `[block, g0, g1, g2]`, `lo`, `diag` and `up`
/// have `[block * block, g0, g1, g2]` (entry `(row, col)` of a cell's block at `row * block +
/// col`); `axis` is an array axis `1..=3`. The result has the shape of `rhs`.
///
/// The blocks are eliminated with partial pivoting; a singular block gives `Inf`/`NaN` in that
/// line.
#[track_caller]
pub fn solve_block_tridiagonal(
    block: usize,
    axis: usize,
    lo: &Array<f32>,
    diag: &Array<f32>,
    up: &Array<f32>,
    rhs: &Array<f32>,
) -> Array<f32> {
    let location = Location::caller();
    let result = run(None, Some(block), axis, [lo, diag, up, rhs], None, location);
    Array::resolve(rhs.context(), rhs.shape(), result, "solve_block_tridiagonal", location)
}

/// [`solve_block_tridiagonal`] where the system of line `l` begins at cell `start[l]`, like
/// [`solve_tridiagonal_from`].
#[track_caller]
pub fn solve_block_tridiagonal_from(
    block: usize,
    axis: usize,
    lo: &Array<f32>,
    diag: &Array<f32>,
    up: &Array<f32>,
    rhs: &Array<f32>,
    start: &Array<u32>,
) -> Array<f32> {
    let location = Location::caller();
    let result = run(None, Some(block), axis, [lo, diag, up, rhs], Some(start), location);
    Array::resolve(rhs.context(), rhs.shape(), result, "solve_block_tridiagonal", location)
}

fn invalid(message: String) -> Failure {
    LumaError::InvalidShape(message).into()
}

/// `block == None`: the scalar form. `arrays` = `[lo, diag, up, rhs]`. `tuning` replaces the
/// context's (tests force an algorithm and the workgroup memory budget).
fn run(
    tuning: Option<Tuning>,
    block: Option<usize>,
    axis: usize,
    arrays: [&Array<f32>; 4],
    start: Option<&Array<u32>>,
    _location: &'static Location<'static>,
) -> Outcome<Array<f32>> {
    let [lo, diag, up, rhs] = arrays;
    let ctx = rhs.context();
    // Operands first, so a poisoned one reports its own error.
    let (rhs_pin, rhs_bytes) = rhs.ready()?;
    let (lo_pin, lo_bytes) = lo.ready()?;
    let (diag_pin, diag_bytes) = diag.ready()?;
    let (up_pin, up_bytes) = up.ready()?;
    let start_ready = start.map(|s| s.ready()).transpose()?;
    if !ctx.same_as(lo.context()) || !ctx.same_as(diag.context()) || !ctx.same_as(up.context()) {
        return Err(LumaError::ContextMismatch.into());
    }
    if let Some(start) = start
        && !ctx.same_as(start.context())
    {
        return Err(LumaError::ContextMismatch.into());
    }
    let shape = rhs.shape();
    let dims = shape.dims();
    if axis >= 4 {
        return Err(invalid(format!("tridiagonal solve along axis {axis} of shape {shape}")));
    }
    let b = block.unwrap_or(1);
    let coefficient = match block {
        None => shape,
        Some(b) => {
            if !(1..=8).contains(&b) {
                return Err(invalid(format!("block size {b} is outside 1..=8")));
            }
            if axis == 0 || dims[0] != b {
                return Err(invalid(format!(
                    "block tridiagonal solve of size {b} along axis {axis} of a right-hand side of shape {shape}"
                )));
            }
            Shape::from([b * b, dims[1], dims[2], dims[3]])
        }
    };
    for (array, expected) in [(lo, coefficient), (diag, coefficient), (up, coefficient)] {
        if array.shape() != expected {
            return Err(LumaError::ShapeMismatch { lhs: expected.dims(), rhs: array.shape().dims() }.into());
        }
    }
    let n = dims[axis];
    let stride: usize = dims[block.map_or(0, |_| 1)..axis].iter().product();
    let cells = rhs.len() / b;
    let lines = cells / n;
    if let Some(start) = start
        && start.len() != lines
    {
        return Err(invalid(format!("{} start cells for {lines} lines", start.len())));
    }
    let to_u32 = |v: usize| u32::try_from(v).map_err(|_| Failure::from(LumaError::DimensionTooLarge { shape: dims }));
    let (n, stride, lines32) = (to_u32(n)?, to_u32(stride)?, to_u32(lines)?);

    let out = Array::<f32>::alloc(ctx, shape)?;
    let out_storage = Arc::clone(out.storage()?);
    let out_pin = out_storage.pin_single()?;
    let mut scratch = Scratch::new(ctx);
    let params = [n, stride, lines32];
    let tuning = tuning.unwrap_or(*ctx.tuning());
    let (core, groups, scratch_bytes) = if let Some(plan) = cyclic::plan(&tuning, b as u32, n, lines32, stride, ctx.max_buffer_bytes()) {
        let core = ctx.builtin(
            Builtin::CyclicTridiagonal {
                block: b as u8,
                starts: start.is_some(),
                workgroup: plan.workgroup,
                shift: plan.shift,
                shared_rows: plan.shared_rows,
            },
            "f32",
        )?;
        (core, fold_workgroups(ctx, lines32)?, plan.scratch_bytes(b as u32, n, lines32))
    } else {
        let workgroup = tridiagonal_workgroup(ctx.tuning(), lines32);
        let core = ctx.builtin(Builtin::Tridiagonal { block: b as u8, starts: start.is_some(), workgroup }, "f32")?;
        let groups = workgroups_1d(ctx, &core, lines32)?;
        (core, groups, cells as u64 * (b * b) as u64 * 4)
    };
    let scratch_buffer = if scratch_bytes > 0 { Some(scratch.take(scratch_bytes)?) } else { None };
    let mut args = vec![RawArg::write(out_pin.buffer(), rhs_bytes)];
    if let Some(buffer) = &scratch_buffer {
        args.push(RawArg::write(buffer, scratch_bytes));
    }
    args.extend([
        RawArg::read(lo_pin.buffer(), lo_bytes),
        RawArg::read(diag_pin.buffer(), diag_bytes),
        RawArg::read(up_pin.buffer(), up_bytes),
        RawArg::read(rhs_pin.buffer(), rhs_bytes),
    ]);
    if let Some((pin, bytes)) = &start_ready {
        args.push(RawArg::read(pin.buffer(), *bytes));
    }
    args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
    ctx.launch(&core, groups, [lines32, 1, 1], &args)?;
    drop(args);
    crate::jit::check_finite(ctx, &out_storage, crate::jit::ElemTy::F32)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::panic::Location;

    use super::run;
    use crate::array::Array;
    use crate::context::Context;
    use crate::tuning::{LineSolver, Tuning};

    /// Deterministic values in `[-0.5, 0.5)`.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }
    }

    /// Solves `m x = r` (`n x n`, row-major) in `f64` with partial pivoting.
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

    /// Block Thomas algorithm in `f64` for the cells `start..n` of one line; `cell(s)` gives the
    /// blocks `(lo, diag, up, rhs)` of cell `s`. Returns `x` of the cells `start..n`.
    fn host_line(b: usize, start: usize, n: usize, cell: &dyn Fn(usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)) -> Vec<Vec<f64>> {
        let mut c: Vec<Vec<f64>> = Vec::new();
        let mut d: Vec<Vec<f64>> = Vec::new();
        for s in start..n {
            let (lo, mut m, up, mut r) = cell(s);
            if s > start {
                let (cp, dp) = (&c[s - start - 1], &d[s - start - 1]);
                for i in 0..b {
                    for k in 0..b {
                        for j in 0..b {
                            m[i * b + j] -= lo[i * b + k] * cp[k * b + j];
                        }
                        r[i] -= lo[i * b + k] * dp[k];
                    }
                }
            }
            let mut cs = vec![0.0; b * b];
            for j in 0..b {
                let column = dense_solve(m.clone(), (0..b).map(|i| up[i * b + j]).collect(), b);
                for i in 0..b {
                    cs[i * b + j] = column[i];
                }
            }
            d.push(dense_solve(m, r, b));
            c.push(cs);
        }
        let mut x = vec![vec![0.0; b]; n - start];
        for s in (0..n - start).rev() {
            for i in 0..b {
                let next: f64 = if s + 1 < n - start { (0..b).map(|k| c[s][i * b + k] * x[s + 1][k]).sum() } else { 0.0 };
                x[s][i] = d[s][i] - next;
            }
        }
        x
    }

    /// Runs one solve of `lines` lines of `n` cells of `block`-blocks (`b == 0`: the scalar form)
    /// along grid axis `axis` (0 or 1) of an `n x lines` or `lines x n` grid with `tuning`, with
    /// per-line starts if `starts`, and compares it with the host solve.
    #[track_caller]
    fn check(ctx: &Context, tuning: Tuning, b: usize, n: usize, lines: usize, axis: usize, starts: bool, seed: u64) {
        let bs = b.max(1);
        let grid = if axis == 0 { [n, lines] } else { [lines, n] };
        let cells = n * lines;
        let mut rng = Lcg(seed);
        let lo: Vec<f32> = (0..cells * bs * bs).map(|_| rng.next()).collect();
        let up: Vec<f32> = (0..cells * bs * bs).map(|_| rng.next()).collect();
        let rhs: Vec<f32> = (0..cells * bs).map(|_| rng.next() * 4.0).collect();
        // Diagonally dominant blocks: the off-diagonal blocks and the rest of the diagonal block are
        // below 0.5 per entry, the block diagonal gets `2 * block`.
        let diag: Vec<f32> = (0..cells * bs * bs)
            .map(|k| rng.next() + if (k % (bs * bs)) / bs == k % bs { 2.0 * bs as f32 } else { 0.0 })
            .collect();
        let start: Vec<u32> = (0..lines).map(|l| [0, 1, (n / 2) as u32, n as u32, (n + 3) as u32, 0][(l + seed as usize) % 6]).collect();

        let make = |data: &[f32], per_cell: usize| -> Array<f32> {
            match b {
                0 => Array::from_slice_on(ctx, data, grid).unwrap(),
                _ => Array::from_slice_on(ctx, data, [per_cell, grid[0], grid[1]]).unwrap(),
            }
        };
        let (lo_a, diag_a, up_a, rhs_a) = (make(&lo, bs * bs), make(&diag, bs * bs), make(&up, bs * bs), make(&rhs, bs));
        let start_a = Array::from_slice_on(ctx, &start, lines).unwrap();
        let location = Location::caller();
        let result = run(
            Some(tuning),
            (b > 0).then_some(b),
            if b == 0 { axis } else { axis + 1 },
            [&lo_a, &diag_a, &up_a, &rhs_a],
            starts.then_some(&start_a),
            location,
        );
        let got = Array::resolve(ctx, rhs_a.shape(), result, "solve", location).to_vec().unwrap();

        let stride = if axis == 0 { 1 } else { lines };
        let label = format!("b={b} n={n} lines={lines} axis={axis} starts={starts} line_bytes={}", tuning.line_bytes);
        for l in 0..lines {
            let base = (l % stride) + (l / stride) * stride * n;
            let s0 = if starts { (start[l] as usize).min(n) } else { 0 };
            let slice = |a: &[f32], per: usize, s: usize| -> Vec<f64> { (0..per).map(|k| f64::from(a[(base + s * stride) * per + k])).collect() };
            let want = host_line(bs, s0, n, &|s| (slice(&lo, bs * bs, s), slice(&diag, bs * bs, s), slice(&up, bs * bs, s), slice(&rhs, bs, s)));
            for s in 0..n {
                for i in 0..bs {
                    let g = f64::from(got[(base + s * stride) * bs + i]);
                    let w = if s < s0 { 0.0 } else { want[s - s0][i] };
                    let err = (g - w).abs() / (1.0 + w.abs());
                    assert!(err < 2e-4 && g.is_finite(), "{label}: line {l} cell {s} component {i}: got {g}, want {w}");
                }
            }
        }
    }

    fn cyclic(line_bytes: u32) -> Tuning {
        Tuning { line_solver: LineSolver::Cyclic, line_bytes, ..Tuning::for_tests() }
    }

    fn thomas() -> Tuning {
        Tuning { line_solver: LineSolver::Thomas, ..Tuning::for_tests() }
    }

    #[test]
    fn scalar_systems_by_both_solvers() {
        let ctx = Context::new().unwrap();
        // Sizes that are powers of two, one less and one more, tiny and long; the small byte budget
        // moves rows of the line to the global scratch.
        for &(n, lines) in &[(1, 3), (2, 5), (3, 4), (7, 6), (32, 3), (33, 5), (64, 2), (100, 7), (255, 3), (256, 2), (1000, 3), (4097, 2)] {
            for axis in 0..2 {
                for starts in [false, true] {
                    let seed = (n * 31 + lines) as u64;
                    check(&ctx, thomas(), 0, n, lines, axis, starts, seed);
                    for bytes in [16 * 1024, 2048, 256] {
                        check(&ctx, cyclic(bytes), 0, n, lines, axis, starts, seed);
                    }
                }
            }
        }
    }

    #[test]
    fn block_systems_by_both_solvers() {
        let ctx = Context::new().unwrap();
        for b in 1..=8 {
            // Big blocks compile slowly (on a software rasterizer most of all): fewer sizes.
            let sizes: &[(usize, usize)] = if b <= 5 { &[(1, 2), (2, 3), (33, 3), (130, 2)] } else { &[(5, 2), (66, 2)] };
            for &(n, lines) in sizes {
                // One line count and axis per size keeps the number of kernels (one per block
                // size, size, budget and `starts`) small.
                let axis = (n + b) % 2;
                for starts in [false, true] {
                    let seed = (b * 1000 + n) as u64;
                    check(&ctx, thomas(), b, n, lines, axis, starts, seed);
                    for bytes in [16 * 1024, 1024] {
                        check(&ctx, cyclic(bytes), b, n, lines, axis, starts, seed);
                    }
                }
            }
        }
    }

    #[test]
    fn plan_follows_the_workgroup_memory_budget() {
        let tuning = Tuning { line_solver: LineSolver::Cyclic, line_bytes: 16 * 1024, ..Tuning::for_tests() };
        // 3 floats per row of a scalar line: 1024 rows fit, 4096 keep every fourth row (1024) in workgroup memory.
        let fits = crate::cyclic::plan(&tuning, 1, 1024, 10, 1, u64::MAX).unwrap();
        assert_eq!((fits.shift, fits.shared_rows), (0, 1024));
        let spills = crate::cyclic::plan(&tuning, 1, 4096, 10, 1, u64::MAX).unwrap();
        assert_eq!((spills.shift, spills.shared_rows), (2, 1024));
        assert_eq!(spills.scratch_bytes(1, 4096, 10), 10 * 3072 * 3 * 4);
        // Blocks of 4: 37 floats per row.
        let block = crate::cyclic::plan(&tuning, 4, 1024, 10, 1, u64::MAX).unwrap();
        assert!(u64::from(block.shared_rows) * 37 * 4 <= 16 * 1024 && block.shift > 0);
        // The scratch must fit a buffer, or Thomas runs.
        assert!(crate::cyclic::plan(&tuning, 4, 1024, 10, 1, 1000).is_none());
        let thomas = Tuning { line_solver: LineSolver::Thomas, ..tuning };
        assert!(crate::cyclic::plan(&thomas, 1, 1024, 10, 1, u64::MAX).is_none());
        // The automatic choice: few lines yes, many lines no, strided lines sooner than contiguous
        // ones, bigger blocks sooner than small ones.
        let auto = Tuning { line_solver: LineSolver::Auto, line_cyclic_work: 8192, ..tuning };
        assert!(crate::cyclic::plan(&auto, 1, 4096, 200, 1, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 1, 4096, 8192, 1, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 1, 4096, 8193, 1, u64::MAX).is_none());
        assert!(crate::cyclic::plan(&auto, 1, 4096, 4096, 4096, u64::MAX).is_none());
        assert!(crate::cyclic::plan(&auto, 1, 4096, 1024, 1024, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 4, 1024, 100, 1, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 4, 1024, 1024, 1, u64::MAX).is_none());
        let never = Tuning { line_cyclic_work: 0, ..auto };
        assert!(crate::cyclic::plan(&never, 1, 4096, 1, 1, u64::MAX).is_none());
    }
}
