//! Line solvers: one tridiagonal or block-tridiagonal system per grid line along an axis.
//!
//! Two algorithms: the block Thomas algorithm runs one invocation per line (see
//! [`Builtin::Tridiagonal`](crate::builtin::Builtin)), `O(n)` work per line with no intermediate
//! passes over memory, which suits many lines; for few, long lines one workgroup solves a line by
//! cyclic reduction in workgroup memory (see [`crate::cyclic`]). The choice is automatic
//! (`LUMA_TRIDIAGONAL=thomas|cyclic` forces one). [`solve_tridiagonal`] and
//! [`solve_tridiagonal_from`] compute their pending operands (`lo`, `diag`, `up`, `rhs` that are
//! lazy expressions) inside the solver kernel, like the `_parts` functions; an operand that is
//! already a buffer is read as is. [`solve_block_tridiagonal`] evaluates its stacked operands
//! first (arrays split over several buffers fail with [`LumaError::Chunked`]); use
//! [`solve_block_tridiagonal_parts`] to fuse the entries of the blocks. The solve is recorded,
//! not waited for. The `_parts` functions take every entry of the blocks as an array of its own
//! and compute the ones that are lazy expressions inside the solver kernel (see
//! [`crate::solve`]); `solve_tridiagonal_parts` solves several systems on one grid in one launch.
//!
//! The arrays may have any shape: the lines are the lines of `axis` of every other index, so an
//! array `[2, ni, nj]` solved along axis 1 or 2 holds a batch of two systems per line.

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Array, Failure};
use crate::builtin::{thomas_scratch_floats, thomas_workgroups, Builtin};
use crate::jit::structured::boundary::LineSolveAttrs;
use crate::jit::structured::strategy::{first_candidate, note, Boundary, BoundaryAttrs, FamilyTag, Params, StrategyId};
use crate::context::{RawArg, Scratch};
use crate::error::LumaError;
use crate::jit::PartSide;
use crate::ops::{fold_workgroups, Outcome};
use crate::shape::Shape;
use crate::solve::{Algo, Io, SolveSpec};
use crate::tuning::Tuning;

/// The tuning a line solve selects with: the context's (a test forces one for the thread, to pick an algorithm or a budget).
#[inline]
pub(crate) fn tuning_of(ctx: &crate::context::Context) -> std::borrow::Cow<'_, Tuning> {
    #[cfg(test)]
    if let Some(forced) = tests::FORCED.with(std::cell::Cell::get) {
        return std::borrow::Cow::Owned(forced);
    }
    std::borrow::Cow::Borrowed(ctx.tuning())
}

/// Solves `lo[s] x[s-1] + diag[s] x[s] + up[s] x[s+1] = rhs[s]` along `axis` for every line of
/// the four arrays, which share one shape. `lo` at the first and `up` at the last cell of a line
/// are ignored. The result has the shape of `rhs`.
///
/// A singular pivot gives `Inf`/`NaN` in that line; it is not checked.
///
/// The arrays may have up to four dimensions and the lines run along `axis` for every index of
/// the others, so a batch of systems is an extra axis: `[2, ni, nj]` arrays solved along axis 1
/// or 2 are two systems per line (`ni` and `nj` lines each), whatever algorithm runs. The
/// systems of one batch must share their grid; to solve independent systems without stacking
/// them first, see [`solve_tridiagonal_parts`].
///
/// Operands that are lazy expressions are computed inside the solver kernel (no launch of their
/// own); evaluated ones are read as they are.
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
    let arrays = [lo, diag, up, rhs];
    let result = if arrays.iter().any(|a| a.pending().is_some()) {
        run_parts(1, 1, axis, &arrays, None, location).map(|mut solved| solved.remove(0))
    } else {
        run(None, axis, arrays, None, location)
    };
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
    let arrays = [lo, diag, up, rhs];
    let result = if arrays.iter().any(|a| a.pending().is_some()) {
        run_parts(1, 1, axis, &arrays, Some(start), location).map(|mut solved| solved.remove(0))
    } else {
        run(None, axis, arrays, Some(start), location)
    };
    Array::resolve(rhs.context(), rhs.shape(), result, "solve_tridiagonal", location)
}

/// Solves a block-tridiagonal system (`block` x `block` blocks, `1 <= block <= 8`) along `axis`
/// for every line of the grid. `rhs` has dimensions `[block, g0, g1, g2]`, `lo`, `diag` and `up`
/// have `[block * block, g0, g1, g2]` (entry `(row, col)` of a cell's block at `row * block +
/// col`); `axis` is an array axis `1..=3`. The result has the shape of `rhs`.
///
/// The blocks are eliminated with partial pivoting by the solve that runs an invocation per line;
/// the solve that runs a workgroup per line inverts blocks of up to 4 from their cofactors (no
/// pivot is divided by, so a block with a zero diagonal is fine) and eliminates bigger ones with
/// partial pivoting. Both are insensitive to the magnitude of the entries: the first pivots by
/// magnitude, the second scales every block by its largest entry before it inverts or
/// eliminates, so blocks of uniformly huge or tiny entries (`1e10`, `1e-10`) solve as well as unit
/// ones; the error of either grows with the condition number of the block. A singular block (a
/// zero row, or a determinant that is zero) gives `Inf`/`NaN` in that line. To solve on the entries of
/// the blocks held as arrays of their own, without stacking them, see
/// [`solve_block_tridiagonal_parts`].
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

/// Solves a block-tridiagonal system (`B` x `B` blocks, `1 <= B <= 8`) along `axis` on *parts*: every
/// entry of every block, and every component of the right-hand side, is an array of its own, all of
/// the grid's shape. `lo`, `diag` and `up` hold the entries of the lower, diagonal and upper blocks,
/// entry `(row, col)` of a cell's block at index `row * B + col` (so `B2 = B * B`; the same order
/// as the first axis of the `[B * B, ..]` stacks of [`solve_block_tridiagonal`]), `rhs` the
/// components of the right-hand side. The result holds the components of the solution, again as
/// arrays of the grid's shape. `axis` is an axis of the grid (`0..4`; the grid has at most three
/// dimensions), unlike the array axis of [`solve_block_tridiagonal`], which counts the block axis.
/// The solution is the same as of [`solve_block_tridiagonal`] on the stacks of the parts, up to
/// rounding.
///
/// A part that is a lazy expression is not evaluated: it is computed inside the solver kernel,
/// where the coefficients of a cell are read, so the whole step from the expressions to the
/// solution is one launch, with no stack and no unstack (which a [`stack`](crate::stack) of the
/// parts and `take`s of the components would be). Parts that are evaluated arrays are bound as
/// buffers: one binding per part, `3 B^2 + B` and `B` for the result. If the expressions need more
/// buffers or scalars than a kernel can bind, or the solve needs the elimination of blocks above
/// 5 on one invocation per line, the parts are stacked and the stacked solve runs instead.
///
/// Each part is a buffer of its own, so a workgroup that solves a line which is not contiguous in
/// memory (`axis > 0`) reads every part at a stride, a sector per element, where the stacked
/// blocks of [`solve_block_tridiagonal`] share their sectors. On an RTX 3090 the solve of 4x4
/// blocks of 156 x 122 cells takes 18 us on stacked blocks along either axis, plus 15 us to stack
/// the parts and unstack the solution; on parts it takes 19 us along axis 0 and 33 us along
/// axis 1.
///
/// `B2` must be `B * B` and `B` at most 8: a compile-time error otherwise.
///
/// ```
/// # use luma::*;
/// # fn main() -> Result<()> {
/// // A 2x2 block system on a [5, 3] grid, as 4 + 4 + 4 + 2 arrays: x_s = rhs_s / diag_s, the
/// // blocks being diagonal, with the diagonal an expression that is never evaluated alone.
/// let zero = Array::<f32>::zeros([5, 3])?;
/// let two = Array::<f32>::full([5, 3], 2.0)?;
/// let (r0, r1) = (Array::<f32>::full([5, 3], 3.0)?, Array::<f32>::full([5, 3], 8.0)?);
/// let d0 = &two + 0.0; // lazy
/// let [x0, x1] = solve_block_tridiagonal_parts(
///     0,
///     [&zero; 4],
///     [&d0, &zero, &zero, &(&two * 2.0)],
///     [&zero; 4],
///     [&r0, &r1],
/// );
/// assert_eq!(x0.to_vec()?, vec![1.5; 15]);
/// assert_eq!(x1.to_vec()?, vec![2.0; 15]);
/// # Ok(()) }
/// ```
///
/// # Errors
///
/// Every result is poisoned if the arrays differ in context or shape, or `axis >= 4`, or
/// `B > 1` and the fourth dimension of the grid is not 1 (a scalar system, `B == 1`, may use all
/// four).
#[track_caller]
pub fn solve_block_tridiagonal_parts<const B2: usize, const B: usize>(
    axis: usize,
    lo: [&Array<f32>; B2],
    diag: [&Array<f32>; B2],
    up: [&Array<f32>; B2],
    rhs: [&Array<f32>; B],
) -> [Array<f32>; B] {
    const { assert!(B >= 1 && B <= 8 && B2 == B * B, "block parts: 1 <= B <= 8 and B2 == B * B") };
    let location = Location::caller();
    let parts: Vec<&Array<f32>> = lo.into_iter().chain(diag).chain(up).chain(rhs).collect();
    let result = run_parts(B, 1, axis, &parts, None, location);
    resolve_parts(rhs[0], result, "solve_block_tridiagonal_parts", location)
}

/// [`solve_block_tridiagonal_parts`] where the system of line `l` begins at cell `start[l]`, like
/// [`solve_block_tridiagonal_from`].
#[track_caller]
pub fn solve_block_tridiagonal_parts_from<const B2: usize, const B: usize>(
    axis: usize,
    lo: [&Array<f32>; B2],
    diag: [&Array<f32>; B2],
    up: [&Array<f32>; B2],
    rhs: [&Array<f32>; B],
    start: &Array<u32>,
) -> [Array<f32>; B] {
    const { assert!(B >= 1 && B <= 8 && B2 == B * B, "block parts: 1 <= B <= 8 and B2 == B * B") };
    let location = Location::caller();
    let parts: Vec<&Array<f32>> = lo.into_iter().chain(diag).chain(up).chain(rhs).collect();
    let result = run_parts(B, 1, axis, &parts, Some(start), location);
    resolve_parts(rhs[0], result, "solve_block_tridiagonal_parts_from", location)
}

/// Solves `N` independent tridiagonal systems on the same grid in one launch: `systems[k]` is
/// `(lo, diag, up, rhs)` of system `k`, as for [`solve_tridiagonal`], and the result holds the
/// solution of every one, in order. Equal to `N` calls of [`solve_tridiagonal`] but for the launch
/// count, and so the time of the launch overheads, of reading the grid's lines once, of filling
/// the device with `N` times the lines; lazy parts are computed inside the kernel, as for
/// [`solve_block_tridiagonal_parts`]. `N` is at most 64.
///
/// ```
/// # use luma::*;
/// # fn main() -> Result<()> {
/// // Two systems on a [6, 2] grid, along axis 0: x - 0.25 (x[s-1] + x[s+1]) = rhs.
/// let (lo, up) = (Array::<f32>::full([6, 2], -0.25)?, Array::<f32>::full([6, 2], -0.25)?);
/// let diag = Array::<f32>::full([6, 2], 1.0)?;
/// let k_rhs = Array::<f32>::full([6, 2], 1.0)?;
/// let w_rhs = &k_rhs * 2.0;
/// let [k, w] = solve_tridiagonal_parts(0, [(&lo, &diag, &up, &k_rhs), (&lo, &diag, &up, &w_rhs)]);
/// let (k, w) = (k.to_vec()?, w.to_vec()?);
/// assert!((w[0] - 2.0 * k[0]).abs() < 1e-5);
/// # Ok(()) }
/// ```
///
/// # Errors
///
/// Every result is poisoned if the arrays differ in context or shape, or `axis >= 4`.
#[track_caller]
#[allow(clippy::type_complexity)]
pub fn solve_tridiagonal_parts<const N: usize>(
    axis: usize,
    systems: [(&Array<f32>, &Array<f32>, &Array<f32>, &Array<f32>); N],
) -> [Array<f32>; N] {
    const { assert!(N >= 1 && N <= 64, "tridiagonal systems: 1 <= N <= 64") };
    let location = Location::caller();
    let parts: Vec<&Array<f32>> = systems.iter().flat_map(|&(lo, diag, up, rhs)| [lo, diag, up, rhs]).collect();
    let result = run_parts(1, N, axis, &parts, None, location);
    resolve_parts(systems[0].3, result, "solve_tridiagonal_parts", location)
}

/// [`solve_tridiagonal_parts`] where the system of line `l` begins at cell `start[l]` in every
/// system, like [`solve_tridiagonal_from`].
#[track_caller]
#[allow(clippy::type_complexity)]
pub fn solve_tridiagonal_parts_from<const N: usize>(
    axis: usize,
    systems: [(&Array<f32>, &Array<f32>, &Array<f32>, &Array<f32>); N],
    start: &Array<u32>,
) -> [Array<f32>; N] {
    const { assert!(N >= 1 && N <= 64, "tridiagonal systems: 1 <= N <= 64") };
    let location = Location::caller();
    let parts: Vec<&Array<f32>> = systems.iter().flat_map(|&(lo, diag, up, rhs)| [lo, diag, up, rhs]).collect();
    let result = run_parts(1, N, axis, &parts, Some(start), location);
    resolve_parts(systems[0].3, result, "solve_tridiagonal_parts_from", location)
}

/// The `N` arrays of `result`, or `N` arrays poisoned by its failure, of the shape of `like`.
fn resolve_parts<const N: usize>(
    like: &Array<f32>,
    result: Outcome<Vec<Array<f32>>>,
    op: &'static str,
    location: &'static Location<'static>,
) -> [Array<f32>; N] {
    let outcome = result.and_then(|arrays| {
        <[Array<f32>; N]>::try_from(arrays).map_err(|arrays| Failure::New(LumaError::Gpu(format!("{} solutions for {N} systems", arrays.len()))))
    });
    match outcome {
        Ok(arrays) => arrays,
        Err(failure) => {
            let error = match failure {
                Failure::Inherited(error) => error,
                Failure::New(error) => Arc::new(LumaError::Op { op, location, source: Arc::new(error) }),
            };
            std::array::from_fn(|_| Array::poisoned(like.context(), like.shape(), Arc::clone(&error)))
        }
    }
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
    let params = [n, stride, lines32];
    let boundary = Boundary::new(
        FamilyTag::LineSolve,
        "tridiagonal",
        Location::caller(),
        [n, stride, lines32, 1],
        BoundaryAttrs::LineSolve(LineSolveAttrs {
            block: b as u8,
            parts: false,
            fused: false,
            starts: start.is_some(),
            len: cells as u64,
            max_scratch: ctx.max_buffer_bytes(),
        }),
    );
    let (chosen, params_of) = first_candidate(ctx, &boundary).expect("a stacked line solve always fits");
    note(ctx, &boundary, chosen, &params_of, &[]);
    let Params::LineSolve(plan) = params_of else { unreachable!("a line-solve strategy runs with line-solve parameters") };
    let (core, groups, scratch_bytes) = if chosen == StrategyId::LINE_SOLVE_CYCLIC {
        let core = ctx.builtin(
            Builtin::CyclicTridiagonal { block: b as u8, starts: start.is_some(), workgroup: plan.workgroup, shift: plan.shift, shared_rows: plan.shared_rows },
            "f32",
        )?;
        (core, fold_workgroups(ctx, lines32)?, plan.scratch_bytes)
    } else {
        let core = ctx.builtin(Builtin::Tridiagonal { block: b as u8, starts: start.is_some(), workgroup: plan.workgroup as u8, twisted: plan.twisted }, "f32")?;
        let groups = fold_workgroups(ctx, thomas_workgroups(lines32, plan.workgroup as u8, plan.twisted))?;
        (core, groups, plan.scratch_bytes)
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

/// The side of a solve `array` stands for: its expression if it is pending (so it may be fused),
/// else its buffer.
fn part_side(ctx: &crate::context::Context, array: &Array<f32>) -> Outcome<PartSide> {
    match array.pending() {
        Some(node) => {
            if node.len() as u64 * 4 > ctx.segment_bytes() {
                return Err(LumaError::Chunked("a line solve of an array split over several buffers").into());
            }
            Ok(PartSide::Pending(Arc::clone(node)))
        }
        None => {
            let (pin, bytes) = array.ready()?;
            Ok(PartSide::Ready(pin, bytes))
        }
    }
}

/// The solve on parts: `systems` systems of `block`-blocks whose parts are `parts`, `systems` runs
/// of `3 block^2 + block` (`lo`, `diag`, `up`, `rhs`); the result is the `block` components of the
/// solution of every system.
fn run_parts(
    block: usize,
    systems: usize,
    axis: usize,
    parts: &[&Array<f32>],
    start: Option<&Array<u32>>,
    location: &'static Location<'static>,
) -> Outcome<Vec<Array<f32>>> {
    let per = Io::parts_per_system(block);
    let ctx = parts[per - 1].context();
    // Operands first, so that a poisoned one reports its own error.
    let mut sides = parts.iter().map(|array| part_side(array.context(), array)).collect::<Outcome<Vec<_>>>()?;
    let start_ready = start.map(|s| s.ready()).transpose()?;
    if parts.iter().any(|array| !ctx.same_as(array.context())) || start.is_some_and(|start| !ctx.same_as(start.context())) {
        return Err(LumaError::ContextMismatch.into());
    }
    let shape = parts[per - 1].shape();
    let dims = shape.dims();
    if let Some(array) = parts.iter().find(|array| array.shape() != shape) {
        return Err(LumaError::ShapeMismatch { lhs: dims, rhs: array.shape().dims() }.into());
    }
    if axis >= 4 {
        return Err(invalid(format!("tridiagonal solve along axis {axis} of shape {shape}")));
    }
    if block > 1 && dims[3] != 1 {
        return Err(invalid(format!("block tridiagonal solve on parts of shape {shape}: at most three dimensions")));
    }
    let (n, stride) = (dims[axis], dims[..axis].iter().product::<usize>());
    let lines = parts[0].len() / n;
    if let Some(start) = start
        && start.len() != lines
    {
        return Err(invalid(format!("{} start cells for {lines} lines", start.len())));
    }
    let to_u32 = |v: usize| u32::try_from(v).map_err(|_| Failure::from(LumaError::DimensionTooLarge { shape: dims }));
    let (n32, stride32, lines32) = (to_u32(n)?, to_u32(stride)?, to_u32(lines)?);
    let total = to_u32(lines * systems)?;
    let fused = sides.iter().any(|s| matches!(s, PartSide::Pending(_)));
    let boundary = Boundary::new(
        FamilyTag::LineSolve,
        "tridiagonal",
        Location::caller(),
        [n32, stride32, total, systems as u32],
        BoundaryAttrs::LineSolve(LineSolveAttrs {
            block: block as u8,
            parts: true,
            fused,
            starts: start.is_some(),
            len: parts[0].len() as u64,
            max_scratch: ctx.max_buffer_bytes(),
        }),
    );
    let (id, params) = first_candidate(ctx, &boundary).expect("a parts solve always fits (stacked_from_parts at the latest)");
    note(ctx, &boundary, id, &params, &[]);
    let Params::LineSolve(plan) = params else { unreachable!("a line-solve strategy runs with line-solve parameters") };
    let algo = match id {
        StrategyId::LINE_SOLVE_CYCLIC => Some(Algo::Cyclic { workgroup: plan.workgroup, shift: plan.shift, shared_rows: plan.shared_rows }),
        StrategyId::LINE_SOLVE_NATIVE | StrategyId::LINE_SOLVE_THOMAS => Some(Algo::Thomas { workgroup: plan.workgroup as u8, tiled: plan.tiled, twisted: plan.twisted }),
        _ => None,
    };
    if id == StrategyId::LINE_SOLVE_NATIVE && fused {
        // The native solve reads buffers: the parts that are expressions are computed first, all
        // together (one kernel with an output each, sharing what they read).
        for (array, side) in parts.iter().zip(&sides) {
            if matches!(side, PartSide::Pending(_)) {
                array.request()?;
            }
        }
        ctx.jit().evaluate_requested();
        for (array, side) in parts.iter().zip(&mut sides) {
            if matches!(side, PartSide::Pending(_)) {
                let (pin, bytes) = array.ready()?;
                *side = PartSide::Ready(pin, bytes);
            }
        }
    }
    if let Some(algo) = algo {
        let spec = SolveSpec { algo, block: block as u8, systems: systems as u8, starts: start.is_some(), fused: [0; 4] };
        // A launch that declines (the kernel cannot read the parts, see `jit::solve`) leaves the composition: `stacked_from_parts`.
        if let Some(solved) = launch_parts(ctx, sides, spec, shape, [n32, stride32, lines32], plan.scratch_bytes, start_ready)? {
            return Ok(solved);
        }
    }
    solve_stacked(block, systems, axis, parts, start, location)
}

/// The scratch bytes of the Thomas solve of `systems` systems of `block x block` blocks over
/// `len` cells (the modified upper blocks of every system, `tiled` as in
/// [`thomas_plan`](crate::builtin::thomas_plan)), or `None` if they exceed `max_buffer` (the
/// solve then falls back to the stacked one, whose scratch is one system's).
pub(crate) fn thomas_parts_scratch(len: usize, block: usize, systems: usize, tiled: bool, max_buffer: u64) -> Option<u64> {
    let bytes = thomas_scratch_floats(len as u64, block as u32, tiled)?.checked_mul(4 * systems as u64)?;
    (bytes <= max_buffer).then_some(bytes)
}

/// Launches the solve of `spec` over `sides` into new arrays of `shape` and returns them (the
/// components of every system, in order), or `None` if the kernel cannot read the parts (see
/// `jit::solve`).
fn launch_parts(
    ctx: &crate::context::Context,
    sides: Vec<PartSide>,
    spec: SolveSpec,
    shape: Shape,
    words: [u32; 3],
    scratch_bytes: u64,
    start: Option<(crate::gpu::pager::Pin, u64)>,
) -> Outcome<Option<Vec<Array<f32>>>> {
    let outs = (0..usize::from(spec.systems) * usize::from(spec.block)).map(|_| Array::<f32>::alloc(ctx, shape)).collect::<Result<Vec<_>, _>>()?;
    let storages = outs.iter().map(|out| out.storage().cloned()).collect::<Result<Vec<_>, _>>()?;
    let pins = storages.iter().map(|storage| storage.pin_single()).collect::<Result<Vec<_>, _>>()?;
    let mut scratch = Scratch::new(ctx);
    let scratch_buffer = if scratch_bytes > 0 { Some(scratch.take(scratch_bytes)?) } else { None };
    let mut writes: Vec<RawArg<'_>> = pins.iter().zip(&storages).map(|(pin, storage)| RawArg::write(pin.buffer(), storage.byte_len())).collect();
    if let Some(buffer) = &scratch_buffer {
        writes.push(RawArg::write(buffer, scratch_bytes));
    }
    let starts = start.as_ref().map(|(pin, bytes)| RawArg::read(pin.buffer(), *bytes));
    let total = words[2] * u32::from(spec.systems);
    let groups = match spec.algo {
        Algo::Cyclic { .. } => fold_workgroups(ctx, total)?,
        Algo::Thomas { workgroup, twisted, .. } => fold_workgroups(ctx, thomas_workgroups(total, workgroup, twisted))?,
    };
    if !crate::jit::solve(ctx, sides, spec, words, groups, [total, 1, 1], &writes, starts)? {
        return Ok(None);
    }
    drop(writes);
    drop(scratch);
    for storage in &storages {
        crate::jit::check_finite(ctx, storage, crate::jit::ElemTy::F32)?;
    }
    Ok(Some(outs))
}

/// The solve on parts by the stacked solver: the parts of every system are stacked into the
/// arrays of [`solve_block_tridiagonal`] (or are the four arrays of [`solve_tridiagonal`]), and
/// the components of the solution are slices of its result.
fn solve_stacked(
    block: usize,
    systems: usize,
    axis: usize,
    parts: &[&Array<f32>],
    start: Option<&Array<u32>>,
    location: &'static Location<'static>,
) -> Outcome<Vec<Array<f32>>> {
    let per = Io::parts_per_system(block);
    let bb = block * block;
    let shape = parts[per - 1].shape();
    let mut solutions = Vec::with_capacity(systems * block);
    for system in parts.chunks(per) {
        if block == 1 {
            solutions.push(run(None, axis, [system[0], system[1], system[2], system[3]], start, location)?);
            continue;
        }
        let stack = |parts: &[&Array<f32>]| crate::view::stack_of(0, parts, location);
        let (lo, diag, up, rhs) = (stack(&system[..bb]), stack(&system[bb..2 * bb]), stack(&system[2 * bb..3 * bb]), stack(&system[3 * bb..]));
        let x = run(Some(block), axis + 1, [&lo, &diag, &up, &rhs], start, location)?;
        solutions.extend((0..block).map(|c| x.slice(0, c..c + 1).reshape(shape)));
    }
    Ok(solutions)
}

#[cfg(test)]
mod tests {
    use std::panic::Location;

    use std::cell::Cell;

    use super::{run, run_parts, thomas_parts_scratch};
    use crate::array::Array;
    use crate::builtin::{thomas_plan, ThomasPlan};
    use crate::context::Context;
    use crate::tuning::{LineSolver, Tuning, TuningProfile};

    thread_local! {
        /// The tuning `tuning_of` gives on this thread while a test forces one.
        pub(super) static FORCED: Cell<Option<Tuning>> = const { Cell::new(None) };
    }

    /// Runs `f` with line solves selecting under `tuning` instead of the context's.
    pub(super) fn forced<R>(tuning: Tuning, f: impl FnOnce() -> R) -> R {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                FORCED.with(|forced| forced.set(None));
            }
        }
        FORCED.with(|forced| forced.set(Some(tuning)));
        let _reset = Reset;
        f()
    }

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
    #[allow(clippy::type_complexity)]
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
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    #[track_caller]
    fn check(ctx: &Context, tuning: Tuning, b: usize, n: usize, lines: usize, axis: usize, starts: bool, seed: u64) {
        check_scaled(ctx, tuning, b, n, lines, axis, starts, seed, 1.0);
    }

    /// [`check`] of the system with every coefficient and right-hand side entry multiplied by
    /// `scale` (the solution is the same).
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    #[track_caller]
    fn check_scaled(ctx: &Context, tuning: Tuning, b: usize, n: usize, lines: usize, axis: usize, starts: bool, seed: u64, scale: f32) {
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
        let scaled = |data: Vec<f32>| -> Vec<f32> { data.into_iter().map(|v| v * scale).collect() };
        let (lo, up, rhs, diag) = (scaled(lo), scaled(up), scaled(rhs), scaled(diag));
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
        let result = forced(tuning, || {
            run((b > 0).then_some(b), if b == 0 { axis } else { axis + 1 }, [&lo_a, &diag_a, &up_a, &rhs_a], starts.then_some(&start_a), location)
        });
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

    /// How the parts of [`parts_check`] are given: evaluated arrays, expressions (`2 * half`, which
    /// are exact), or alternately.
    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Evaluated,
        Pending,
        Mixed,
    }

    /// Solves `systems` systems of `b`-blocks of the grid `n x lines` / `lines x n` along grid axis
    /// `axis` on parts, with `tuning` (and the buffer budget of the jit `max_inputs` if given), and
    /// compares every component with the host solve. Returns the launches of the solve.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    #[track_caller]
    fn parts_check(ctx: &Context, tuning: Tuning, b: usize, systems: usize, n: usize, lines: usize, axis: usize, starts: bool, kind: Kind, seed: u64) -> u32 {
        let (bb, per) = (b * b, 3 * b * b + b);
        let grid = if axis == 0 { [n, lines] } else { [lines, n] };
        let cells = n * lines;
        let mut rng = Lcg(seed);
        let mut host: Vec<[Vec<f32>; 4]> = Vec::new();
        for _ in 0..systems {
            let lo: Vec<f32> = (0..cells * bb).map(|_| rng.next()).collect();
            let up: Vec<f32> = (0..cells * bb).map(|_| rng.next()).collect();
            let rhs: Vec<f32> = (0..cells * b).map(|_| rng.next() * 4.0).collect();
            let diag: Vec<f32> = (0..cells * bb).map(|k| rng.next() + if (k % bb) / b == k % b { 2.0 * b as f32 } else { 0.0 }).collect();
            host.push([lo, diag, up, rhs]);
        }
        let start: Vec<u32> = (0..lines).map(|l| [0, 1, (n / 2) as u32, n as u32, (n + 3) as u32, 0][(l + seed as usize) % 6]).collect();
        let start_a = Array::from_slice_on(ctx, &start, lines).unwrap();
        // Part `k` of system `s`: entry `e` of every cell of the array `a` of kind `kind_of`.
        let mut arrays: Vec<Array<f32>> = Vec::new();
        for (s, sys) in host.iter().enumerate() {
            for k in 0..per {
                let (source, width, e) = if k < bb { (&sys[0], bb, k) } else if k < 2 * bb { (&sys[1], bb, k - bb) } else if k < 3 * bb { (&sys[2], bb, k - 2 * bb) } else { (&sys[3], b, k - 3 * bb) };
                let pending = kind == Kind::Pending || (kind == Kind::Mixed && (s * per + k) % 2 == 1);
                let entry: Vec<f32> = (0..cells).map(|q| source[q * width + e] * if pending { 0.5 } else { 1.0 }).collect();
                let array = Array::from_slice_on(ctx, &entry, grid).unwrap();
                arrays.push(if pending { &array * 2.0 } else { array });
            }
        }
        let refs: Vec<&Array<f32>> = arrays.iter().collect();
        let location = Location::caller();
        let (solved, profile) = ctx
            .profile(|| {
                let solved = forced(tuning, || run_parts(b, systems, axis, &refs, starts.then_some(&start_a), location));
                ctx.sync().unwrap();
                solved
            })
            .unwrap();
        let solved = solved.unwrap_or_else(|_| panic!("run_parts failed"));
        assert_eq!(solved.len(), systems * b);
        let stride = if axis == 0 { 1 } else { lines };
        let label = format!("parts b={b} systems={systems} n={n} lines={lines} axis={axis} starts={starts} kind={}", kind as u8);
        let got: Vec<Vec<f32>> = solved.iter().map(|x| x.to_vec().unwrap()).collect();
        for (s, sys) in host.iter().enumerate() {
            for l in 0..lines {
                let base = (l % stride) + (l / stride) * stride * n;
                let s0 = if starts { (start[l] as usize).min(n) } else { 0 };
                let slice = |a: &[f32], per: usize, c: usize| -> Vec<f64> { (0..per).map(|k| f64::from(a[(base + c * stride) * per + k])).collect() };
                let want = host_line(b, s0, n, &|c| (slice(&sys[0], bb, c), slice(&sys[1], bb, c), slice(&sys[2], bb, c), slice(&sys[3], b, c)));
                for c in 0..n {
                    for i in 0..b {
                        let g = f64::from(got[s * b + i][base + c * stride]);
                        let w = if c < s0 { 0.0 } else { want[c - s0][i] };
                        let err = (g - w).abs() / (1.0 + w.abs());
                        assert!(err < 2e-4 && g.is_finite(), "{label}: line {l} cell {c} component {i}: got {g}, want {w}");
                    }
                }
            }
        }
        profile.launches()
    }

    /// Whether one kernel binds the buffers of a solve of `systems` systems of `b`-blocks on parts: the
    /// parts, the outputs, the scratch and `starts` (a device with few bindings stacks them instead).
    fn binds(ctx: &Context, b: usize, systems: usize) -> bool {
        ctx.jit().max_inputs() >= systems * (3 * b * b + 2 * b) + 2
    }

    fn cyclic(line_bytes: u32) -> Tuning {
        Tuning { line_solver: LineSolver::Cyclic, line_bytes, ..Tuning::for_tests() }
    }

    fn thomas() -> Tuning {
        Tuning { line_solver: LineSolver::Thomas, ..Tuning::for_tests() }
    }

    /// The Thomas solve as it runs along a strided axis of an NVIDIA GPU: the wider workgroups and
    /// the tiled scratch (see `builtin::thomas_plan`), whatever device the test runs on.
    fn thomas_strided() -> Tuning {
        Tuning { line_solver: LineSolver::Thomas, line_strided_from: 16, ..Tuning::for_tests() }
    }

    /// Lines at a stride of 16 and more, of every size the tiled scratch has a kernel for, on
    /// parts agree with the host: cells in one tile, in several, a last tile that is not full,
    /// lines that start part-way.
    #[test]
    fn strided_thomas_solves_with_a_tiled_scratch() {
        let ctx = Context::new().unwrap();
        for b in 1..=5 {
            for &(n, lines) in &[(1usize, 16usize), (7, 33), (40, 70)] {
                for starts in [false, true] {
                    let seed = (b * 77 + n) as u64;
                    for kind in [Kind::Evaluated, Kind::Pending] {
                        parts_check(&ctx, thomas_strided(), b, 1, n, lines, 1, starts, kind, seed);
                    }
                }
            }
        }
        // Scalar systems batch (and never tile, see `thomas_plan`).
        parts_check(&ctx, thomas_strided(), 1, 2, 40, 33, 1, false, Kind::Pending, 3);
    }

    /// The twisted solve (two invocations per line meeting in the middle) of lines of 1 to 4
    /// cells, which have a top half only, halves of one or two cells, with starts up to and past
    /// the end, stacked and on parts, in workgroups with lines left over.
    #[test]
    fn thomas_solves_short_lines() {
        let ctx = Context::new().unwrap();
        for n in 1..=4usize {
            for starts in [false, true] {
                for b in [0usize, 1, 2, 3, 5] {
                    for axis in 0..2 {
                        check(&ctx, thomas(), b, n, 13, axis, starts, (n * 5 + b) as u64);
                    }
                }
                for b in 1..=5usize {
                    for kind in [Kind::Evaluated, Kind::Pending] {
                        parts_check(&ctx, thomas(), b, 1, n, 13, 0, starts, kind, (n * 9 + b) as u64);
                        parts_check(&ctx, thomas_strided(), b, 1, n, 37, 1, starts, kind, (n * 11 + b) as u64);
                    }
                }
                parts_check(&ctx, thomas(), 1, 2, n, 13, 0, starts, Kind::Pending, 5);
            }
        }
    }

    /// The one invocation per line solve of lines of small blocks (the plan picks it from 8192
    /// lines on), the same lines as [`thomas_solves_short_lines`].
    #[test]
    fn plain_thomas_solves_short_lines() {
        let ctx = Context::new().unwrap();
        let lines = 8200;
        for n in [1usize, 2, 3, 4] {
            for starts in [false, true] {
                for b in [0usize, 1, 2, 4] {
                    check(&ctx, thomas(), b, n, lines, 0, starts, (n * 5 + b) as u64);
                }
                for b in [1usize, 2, 4] {
                    parts_check(&ctx, thomas_strided(), b, 1, n, lines, 1, starts, Kind::Pending, (n * 11 + b) as u64);
                }
            }
        }
        assert!(!thomas_plan(&thomas(), 2, lines as u32, 1, false).twisted);
        assert!(thomas_plan(&thomas(), 2, 8191, 1, false).twisted);
    }

    #[test]
    fn thomas_plan_scales_with_the_multiprocessors() {
        let plan = |workgroup, tiled, twisted| ThomasPlan { workgroup, tiled, twisted };
        let nvidia = |compute_units| Tuning { profile: TuningProfile::Nvidia, line_strided_from: 16, compute_units, ..Tuning::for_tests() };
        // The RTX 3090 (82): 512 workgroups' worth of lines per step of the workgroup size, twisted below 8192 lines.
        assert_eq!(thomas_plan(&nvidia(82), 4, 8191, 8, true), plan(16, false, true));
        assert_eq!(thomas_plan(&nvidia(82), 4, 8192, 8, true), plan(16, false, false));
        assert_eq!(thomas_plan(&nvidia(82), 5, 2047, 1, true), plan(8, false, false));
        assert_eq!(thomas_plan(&nvidia(82), 5, 2048, 1, true), plan(8, false, false));
        assert_eq!(thomas_plan(&nvidia(82), 5, 4096, 1, true), plan(8, false, false));
        assert_eq!(thomas_plan(&nvidia(82), 5, 8192, 1, true), plan(16, false, false));
        // The 3080 Laptop (48): 300 workgroups, twisted below 4795 lines.
        assert_eq!(thomas_plan(&nvidia(48), 4, 4794, 8, true), plan(16, false, true));
        assert_eq!(thomas_plan(&nvidia(48), 4, 4795, 8, true), plan(16, false, false));
        assert_eq!(thomas_plan(&nvidia(48), 5, 4800, 1, true), plan(16, false, false));
        assert_eq!(thomas_plan(&nvidia(48), 5, 4096, 1, true), plan(16, false, false));
        assert_eq!(thomas_plan(&nvidia(48), 5, 2400, 1, true), plan(8, false, false));
    }

    #[test]
    fn thomas_plan_widens_strided_lines() {
        let nvidia = Tuning { profile: TuningProfile::Nvidia, line_strided_from: 16, ..Tuning::for_tests() };
        let plan = |workgroup, tiled, twisted| ThomasPlan { workgroup, tiled, twisted };
        // Strided lines of blocks of 2 to 5: a warp or more per workgroup and a tiled scratch.
        assert_eq!(thomas_plan(&nvidia, 5, 4488, 187, true), plan(64, true, false));
        assert_eq!(thomas_plan(&nvidia, 2, 4096, 16, true), plan(128, true, true));
        // Scalars: no tile (it is the same layout), a warp.
        assert_eq!(thomas_plan(&nvidia, 1, 4096, 64, true), plan(32, false, true));
        // Contiguous lines, strides below 16 and blocks above 5: the narrow workgroups of one
        // invocation per line and the scratch one cell after the other.
        assert_eq!(thomas_plan(&nvidia, 5, 1680, 1, true), plan(8, false, false));
        assert_eq!(thomas_plan(&nvidia, 5, 13090, 8, true), plan(32, false, false));
        assert_eq!(thomas_plan(&nvidia, 6, 4488, 187, true), plan(8, false, false));
        // Stacked blocks too.
        assert_eq!(thomas_plan(&nvidia, 5, 4488, 187, false), plan(8, false, false));
        assert_eq!(thomas_plan(&nvidia, 4, 1024, 187, false), plan(8, false, true));
        // Stacked scalars are the parts layout: the same plan.
        assert_eq!(thomas_plan(&nvidia, 1, 4096, 4096, false), plan(32, false, true));
        assert_eq!(thomas_plan(&nvidia, 1, 4096, 1, false), plan(8, false, true));
        // Many lines: the plain solve.
        assert_eq!(thomas_plan(&nvidia, 4, 13090, 8, true), plan(32, false, false));
        // Other devices keep the old rule.
        let other = Tuning { line_strided_from: 0, ..nvidia };
        assert_eq!(thomas_plan(&other, 5, 4488, 187, true), plan(8, false, false));
        let cpu = Tuning { profile: TuningProfile::Cpu, elementwise_workgroup: 64, ..nvidia };
        assert_eq!(thomas_plan(&cpu, 4, 1024, 187, true), plan(64, false, false));
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

    /// Blocks of uniformly huge or tiny entries (the cofactors of a 4x4 block of `1e10` are `1e30`
    /// and its determinant `1e40`; of `1e-10`, `1e-30` and `1e-40`) solve on both algorithms.
    #[test]
    fn block_systems_of_any_magnitude_by_both_solvers() {
        let ctx = Context::new().unwrap();
        for b in 1..=4 {
            for scale in [1e10f32, 1e-10, 1e-18, 1e18] {
                for tuning in [thomas(), cyclic(16 * 1024)] {
                    for axis in 0..2 {
                        check_scaled(&ctx, tuning, b, 40, 3, axis, false, (b * 10 + axis) as u64, scale);
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

    /// The shapes of a line relaxation of a 2-D finite-volume grid: a few hundred lines of 100 to
    /// 400 cells along either axis, with scalar, 2x2 and 4x4 blocks (the long lines spill).
    #[test]
    fn line_relaxation_shapes() {
        let ctx = Context::new().unwrap();
        for (b, n, lines) in [(0, 400, 20), (2, 400, 20), (4, 300, 8), (4, 100, 20), (2, 100, 20)] {
            for axis in 0..2 {
                check(&ctx, thomas(), b, n, lines, axis, false, 7);
                for bytes in [32 * 1024, 4096] {
                    check(&ctx, cyclic(bytes), b, n, lines, axis, false, 7);
                }
            }
        }
    }

    #[test]
    fn block_parts_by_both_solvers() {
        let ctx = Context::new().unwrap();
        for b in 1..=6 {
            for &(n, lines) in &[(1usize, 2usize), (33, 3), (130, 2)] {
                for (algo, tuning) in [("thomas", thomas()), ("cyclic", cyclic(16 * 1024)), ("spill", cyclic(1024))] {
                    // Blocks above 5 have no Thomas kernel on parts: the stacked solve runs.
                    let axis = (n + b) % 2;
                    for kind in [Kind::Evaluated, Kind::Pending, Kind::Mixed] {
                        let starts = kind == Kind::Mixed;
                        let launches = parts_check(&ctx, tuning, b, 1, n, lines, axis, starts, kind, (b * 100 + n) as u64);
                        if kind != Kind::Mixed && !(algo == "thomas" && b > 5) && binds(&ctx, b, 1) {
                            // One launch: no stack, no unstack, and the pending parts are no kernels of their own.
                            assert_eq!(launches, 1, "{algo} b={b} n={n} kind={}", kind as u8);
                        }
                    }
                }
            }
        }
    }

    /// `solve_tridiagonal` and `solve_tridiagonal_from` compute their pending operands in the solver
    /// kernel: the same result as on evaluated operands, in one launch.
    #[test]
    fn solve_tridiagonal_fuses_pending_operands() {
        let ctx = Context::new().unwrap();
        let (n, lines) = (40usize, 5usize);
        let mut rng = Lcg(7);
        let cells = n * lines;
        let lo: Vec<f32> = (0..cells).map(|_| rng.next()).collect();
        let up: Vec<f32> = (0..cells).map(|_| rng.next()).collect();
        let diag: Vec<f32> = (0..cells).map(|_| rng.next() + 2.0).collect();
        let rhs: Vec<f32> = (0..cells).map(|_| rng.next() * 4.0).collect();
        let start: Vec<u32> = (0..lines as u32).map(|l| l * 3).collect();
        let grid = [n, lines];
        let (lo_a, diag_a, up_a, rhs_a) = (
            Array::from_slice_on(&ctx, &lo, grid).unwrap(),
            Array::from_slice_on(&ctx, &diag, grid).unwrap(),
            Array::from_slice_on(&ctx, &up, grid).unwrap(),
            Array::from_slice_on(&ctx, &rhs, grid).unwrap(),
        );
        let start_a = Array::from_slice_on(&ctx, &start, lines).unwrap();
        let want = super::solve_tridiagonal(0, &lo_a, &diag_a, &up_a, &rhs_a).to_vec().unwrap();
        let want_from = super::solve_tridiagonal_from(0, &lo_a, &diag_a, &up_a, &rhs_a, &start_a).to_vec().unwrap();
        // Exact in f32: the right-hand side is halved, then doubled back.
        let rhs_half = Array::from_slice_on(&ctx, &rhs.iter().map(|v| v * 0.5).collect::<Vec<_>>(), grid).unwrap();
        for from in [false, true] {
            let (got, profile) = ctx
                .profile(|| {
                    let (lo_p, diag_p, rhs_p) = (&lo_a * 1.0, &diag_a + 0.0, &rhs_half * 2.0);
                    let x = if from {
                        super::solve_tridiagonal_from(0, &lo_p, &diag_p, &up_a, &rhs_p, &start_a)
                    } else {
                        super::solve_tridiagonal(0, &lo_p, &diag_p, &up_a, &rhs_p)
                    };
                    ctx.sync().unwrap();
                    x
                })
                .unwrap();
            assert_eq!(&got.to_vec().unwrap(), if from { &want_from } else { &want }, "from = {from}");
            if binds(&ctx, 1, 1) {
                // A native solve reads buffers: the pending operands are computed together first.
                assert_eq!(profile.launches(), if ctx.tuning().native_solve { 2 } else { 1 }, "from = {from}");
            }
        }
    }

    #[test]
    fn batched_scalar_systems_by_both_solvers() {
        let ctx = Context::new().unwrap();
        for &(n, lines) in &[(7usize, 3usize), (100, 4), (1025, 2)] {
            for axis in 0..2 {
                for tuning in [thomas(), cyclic(16 * 1024), cyclic(512)] {
                    for systems in [1usize, 2, 3] {
                        for kind in [Kind::Evaluated, Kind::Pending] {
                            let launches = parts_check(&ctx, tuning, 1, systems, n, lines, axis, systems == 2, kind, (n * 7 + systems) as u64);
                            if binds(&ctx, 1, systems) {
                                assert_eq!(launches, 1, "n={n} axis={axis} systems={systems}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// An axis of a bigger array batches the systems: `[2, n, lines]` solved along axis 1 is two
    /// systems of lines, on both algorithms.
    #[test]
    fn scalar_solve_batches_over_an_extra_axis() {
        let ctx = Context::new().unwrap();
        for &(n, lines) in &[(5usize, 3usize), (70, 4), (300, 2)] {
            for tuning in [thomas(), cyclic(16 * 1024), cyclic(256)] {
                let mut rng = Lcg(n as u64);
                for axis in [1usize, 2] {
                    let grid = if axis == 1 { [2, n, lines] } else { [2, lines, n] };
                    let cells = 2 * n * lines;
                    let lo: Vec<f32> = (0..cells).map(|_| rng.next()).collect();
                    let up: Vec<f32> = (0..cells).map(|_| rng.next()).collect();
                    let rhs: Vec<f32> = (0..cells).map(|_| rng.next() * 4.0).collect();
                    let diag: Vec<f32> = (0..cells).map(|_| rng.next() + 2.0).collect();
                    let make = |d: &[f32]| Array::from_slice_on(&ctx, d, grid).unwrap();
                    let (lo_a, diag_a, up_a, rhs_a) = (make(&lo), make(&diag), make(&up), make(&rhs));
                    let location = Location::caller();
                    let got = Array::resolve(&ctx, rhs_a.shape(), forced(tuning, || run(None, axis, [&lo_a, &diag_a, &up_a, &rhs_a], None, location)), "solve", location).to_vec().unwrap();
                    // Element `[j, ..]` of the batch is the solve of its own coefficients alone.
                    for j in 0..2 {
                        let pick = |d: &[f32]| -> Vec<f32> { d.iter().skip(j).step_by(2).copied().collect() };
                        let sub = [n, lines];
                        let sub = if axis == 1 { sub } else { [lines, n] };
                        let alone = |d: &[f32]| Array::from_slice_on(&ctx, &pick(d), sub).unwrap();
                        let (a, b, c, d) = (alone(&lo), alone(&diag), alone(&up), alone(&rhs));
                        let one = Array::resolve(&ctx, d.shape(), forced(tuning, || run(None, axis - 1, [&a, &b, &c, &d], None, location)), "solve", location).to_vec().unwrap();
                        for (k, want) in one.iter().enumerate() {
                            let have = got[2 * k + j];
                            assert!((have - want).abs() <= 2e-4 * (1.0 + want.abs()), "n={n} axis={axis} batch {j} cell {k}: {have} vs {want}");
                        }
                    }
                }
            }
        }
    }

    /// More parts than the kernel can bind: the stacked solve runs and gives the same solution.
    #[test]
    fn parts_beyond_the_bindings_are_stacked() {
        let ctx = Context::new().unwrap();
        let original = ctx.jit().max_inputs();
        ctx.jit().set_max_inputs(8);
        for (b, systems) in [(2usize, 1usize), (4, 1), (1, 3)] {
            for kind in [Kind::Evaluated, Kind::Pending] {
                let launches = parts_check(&ctx, cyclic(16 * 1024), b, systems, 40, 3, 0, false, kind, 5);
                assert!(launches > 1, "b={b} systems={systems}: {launches} launches");
            }
        }
        ctx.jit().set_max_inputs(original);
        let ok = parts_check(&ctx, cyclic(16 * 1024), 2, 1, 40, 3, 0, false, Kind::Pending, 5);
        if binds(&ctx, 2, 1) {
            assert_eq!(ok, 1);
        }
    }

    #[test]
    fn thomas_parts_scratch_must_fit_a_buffer() {
        assert_eq!(thomas_parts_scratch(1000, 2, 3, false, 1 << 20), Some(1000 * 4 * 4 * 3));
        assert_eq!(thomas_parts_scratch(1000, 2, 3, false, 1000 * 4 * 4 * 3), Some(1000 * 4 * 4 * 3));
        assert_eq!(thomas_parts_scratch(1000, 2, 3, false, 1000 * 4 * 4 * 3 - 1), None);
        assert_eq!(thomas_parts_scratch(usize::MAX, 8, 8, false, u64::MAX), None);
        // Tiled: whole tiles of 32 cells per system.
        assert_eq!(thomas_parts_scratch(1000, 2, 3, true, 1 << 20), Some(1024 * 4 * 4 * 3));
        assert_eq!(thomas_parts_scratch(1024, 2, 3, true, 1 << 20), Some(1024 * 4 * 4 * 3));
    }

    #[test]
    fn plan_follows_the_workgroup_memory_budget() {
        let tuning = Tuning { line_solver: LineSolver::Cyclic, line_bytes: 16 * 1024, ..Tuning::for_tests() };
        // 3 floats per row of a scalar line: 1024 rows fit, 4096 keep every fourth row (1024) in workgroup memory.
        let fits = crate::cyclic::plan(&tuning, 1, 1024, 10, 1, false, false, u64::MAX).unwrap();
        assert_eq!((fits.shift, fits.shared_rows), (0, 1024));
        let spills = crate::cyclic::plan(&tuning, 1, 4096, 10, 1, false, false, u64::MAX).unwrap();
        assert_eq!((spills.shift, spills.shared_rows), (2, 1024));
        assert_eq!(spills.scratch_bytes(1, 4096, 10), 10 * 3072 * 3 * 4);
        // Blocks of 4: nine vec4s per row.
        let block = crate::cyclic::plan(&tuning, 4, 1024, 10, 1, false, false, u64::MAX).unwrap();
        assert!(u64::from(block.shared_rows) * 9 * 16 <= 16 * 1024 && block.shift > 0);
        // Spilled lines in one 32-row bucket share the layout and the kernel; the scratch follows each exact length.
        let a = crate::cyclic::plan(&tuning, 4, 300, 10, 1, false, false, u64::MAX).unwrap();
        let b = crate::cyclic::plan(&tuning, 4, 312, 10, 1, false, false, u64::MAX).unwrap();
        assert_eq!((a.shift, a.shared_rows), (2, 96));
        assert_eq!((b.shift, b.shared_rows), (a.shift, a.shared_rows));
        assert_eq!(
            crate::cyclic::source(
                4,
                false,
                u32::from(a.workgroup),
                u32::from(a.shift),
                a.shared_rows
            ),
            crate::cyclic::source(
                4,
                false,
                u32::from(b.workgroup),
                u32::from(b.shift),
                b.shared_rows
            )
        );
        assert_eq!(a.scratch_bytes(4, 300, 10), 10 * (300 - 75) * 9 * 16);
        assert_eq!(b.scratch_bytes(4, 312, 10), 10 * (312 - 78) * 9 * 16);
        // The scratch must fit a buffer, or Thomas runs.
        assert!(crate::cyclic::plan(&tuning, 4, 1024, 10, 1, false, false, 1000).is_none());
        let thomas = Tuning { line_solver: LineSolver::Thomas, ..tuning };
        assert!(crate::cyclic::plan(&thomas, 1, 1024, 10, 1, false, false, u64::MAX).is_none());
        // The automatic choice: few lines yes, many lines no, strided lines sooner than contiguous
        // ones, bigger blocks sooner than small ones.
        let auto = Tuning { line_solver: LineSolver::Auto, line_cyclic_work: 8192, ..tuning };
        assert!(crate::cyclic::plan(&auto, 1, 4096, 200, 1, false, false, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 1, 4096, 8192, 1, false, false, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 1, 4096, 8193, 1, false, false, u64::MAX).is_none());
        assert!(crate::cyclic::plan(&auto, 1, 4096, 4096, 4096, false, false, u64::MAX).is_none());
        assert!(crate::cyclic::plan(&auto, 1, 4096, 1024, 1024, false, false, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 4, 1024, 100, 1, false, false, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 4, 1024, 1024, 1, false, false, u64::MAX).is_none());
        // Parts: contiguous lines always (5x5 blocks, 1680 lines of 187 were 9x faster than
        // Thomas); strided ones up to `8192 / (2 max(block, 2))` lines, half that for a line that
        // spills.
        assert!(crate::cyclic::plan(&auto, 5, 187, 1680, 1, true, false, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 4, 70, 1024, 70, true, false, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 4, 70, 1025, 70, true, false, u64::MAX).is_none());
        assert!(crate::cyclic::plan(&auto, 4, 1024, 512, 1024, true, false, u64::MAX).is_some());
        assert!(crate::cyclic::plan(&auto, 4, 1024, 513, 1024, true, false, u64::MAX).is_none());
        let never = Tuning { line_cyclic_work: 0, ..auto };
        assert!(crate::cyclic::plan(&never, 1, 4096, 1, 1, false, false, u64::MAX).is_none());
        assert!(crate::cyclic::plan(&never, 5, 187, 1680, 1, true, false, u64::MAX).is_none());
    }
}
