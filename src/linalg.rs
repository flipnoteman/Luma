//! Line solvers: one tridiagonal or block-tridiagonal system per grid line along an axis.
//!
//! The solve runs one invocation per line (block Thomas algorithm, see
//! [`Builtin::Tridiagonal`](crate::builtin::Builtin)), so for many systems it is `O(n)` work per
//! line with no intermediate passes over memory. Operands are evaluated first (arrays split over
//! several buffers fail with [`LumaError::Chunked`]); the solve is recorded, not waited for.

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::builtin::{tridiagonal_workgroup, Builtin};
use crate::context::{RawArg, Scratch};
use crate::error::LumaError;
use crate::ops::{launch_1d, Outcome};
use crate::shape::Shape;

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
    let result = run(None, axis, [lo, diag, up, rhs], None, location);
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
    let result = run(None, axis, [lo, diag, up, rhs], Some(start), location);
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
    let result = run(Some(block), axis, [lo, diag, up, rhs], None, location);
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
    let result = run(Some(block), axis, [lo, diag, up, rhs], Some(start), location);
    Array::resolve(rhs.context(), rhs.shape(), result, "solve_block_tridiagonal", location)
}

fn invalid(message: String) -> Failure {
    LumaError::InvalidShape(message).into()
}

/// `block == None`: the scalar form. `arrays` = `[lo, diag, up, rhs]`.
fn run(
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
    let cp_bytes = cells as u64 * (b * b) as u64 * 4;
    let cp = scratch.take(cp_bytes)?;
    let workgroup = tridiagonal_workgroup(ctx.tuning(), lines32);
    let core = ctx.builtin(Builtin::Tridiagonal { block: b as u8, starts: start.is_some(), workgroup }, "f32")?;
    let params = [n, stride, lines32];
    let mut args = vec![
        RawArg::write(out_pin.buffer(), rhs_bytes),
        RawArg::write(&cp, cp_bytes),
        RawArg::read(lo_pin.buffer(), lo_bytes),
        RawArg::read(diag_pin.buffer(), diag_bytes),
        RawArg::read(up_pin.buffer(), up_bytes),
        RawArg::read(rhs_pin.buffer(), rhs_bytes),
    ];
    if let Some((pin, bytes)) = &start_ready {
        args.push(RawArg::read(pin.buffer(), *bytes));
    }
    args.push(RawArg::Uniform(bytemuck::bytes_of(&params)));
    launch_1d(ctx, &core, lines32, &args)?;
    drop(args);
    crate::jit::check_finite(ctx, &out_storage, crate::jit::ElemTy::F32)?;
    Ok(out)
}
