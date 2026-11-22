//! Dense linear algebra on matrices: Cholesky, triangular and SPD solves (this module), LU, QR,
//! SVD and the symmetric eigenproblem (its submodules).
//!
//! Matrices follow [`matmul`](crate::matmul)'s convention: axis 0 varies fastest, so an array of
//! shape `[cols, rows]` is a `rows x cols` matrix whose row `i` is contiguous, element `(i, j)` at
//! `j + cols * i`. A batch of matrices is a third axis, `[cols, rows, batch]`. A right-hand side
//! of `k` columns for an `n x n` system is `[k, n]` (`[1, n]` for one vector; a plain `[n]` array
//! is taken as that vector, and the solution has the shape of the right-hand side).
//!
//! Every operation runs on the device and is recorded like any other (nothing waits). A single
//! big matrix is factored recursively: the leaves are small kernels (one workgroup per matrix, or
//! one invocation per right-hand-side column), every other flop is a [`matmul_f32`] of the
//! halves, in full `f32` whatever the context's TF32 setting says. A batch of small matrices
//! runs one workgroup per matrix in one launch. Numerical failures (a matrix that is not
//! positive definite, a singular one) do not poison the result: they show up as `NaN` or
//! `Inf` entries, as in the tridiagonal solvers.

use std::panic::Location;

use crate::array::{Array, Failure};
use crate::context::Context;
use crate::error::LumaError;
use crate::kernel::KernelArg;
use crate::shape::Shape;

pub(crate) mod chol;
pub(crate) mod eigh;
pub(crate) mod lu;
pub(crate) mod svd;
pub(crate) mod qr;
pub(crate) mod tri;
pub(crate) mod tsqr;

/// Which triangle of a matrix an operation reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Triangle {
    /// The lower triangle, `j <= i` for element `(i, j)`.
    Lower,
    /// The upper triangle, `j >= i`.
    Upper,
}

/// Rows, columns and batch of a matrix array.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dims {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) batch: usize,
}

impl Dims {
    /// The `[cols, rows, batch]` of `a`; the fourth axis must be 1.
    pub(crate) fn of(a: &Array<f32>) -> Result<Dims, LumaError> {
        let [cols, rows, batch, last] = a.shape().dims();
        if last != 1 {
            return Err(LumaError::InvalidShape(format!("a matrix is [cols, rows, batch], got shape {}", a.shape())));
        }
        Ok(Dims { rows, cols, batch })
    }

    /// Like [`Dims::of`], and the matrices must be square.
    pub(crate) fn square(a: &Array<f32>) -> Result<Dims, LumaError> {
        let d = Dims::of(a)?;
        if d.rows != d.cols {
            return Err(LumaError::InvalidShape(format!("a square matrix is required, got shape {}", a.shape())));
        }
        Ok(d)
    }
}

/// An array poisoned with a new failure of operation `op`, as `shape`.
pub(crate) fn fail(ctx: &Context, shape: Shape, error: LumaError, op: &'static str, location: &'static Location<'static>) -> Array<f32> {
    Array::resolve(ctx, shape, Err(Failure::New(error)), op, location)
}

/// The workgroups of a grid over `x` columns of tiles and `count` independent matrices: matrix
/// `wid.y + wid.z * nwg.y` (the batch folds over the dimension limit).
pub(crate) fn grid(ctx: &Context, x: usize, count: usize) -> [u32; 3] {
    let max = ctx.max_workgroups_per_dimension() as usize;
    let y = count.min(max);
    [x as u32, y as u32, count.div_ceil(y) as u32]
}

/// The workgroups of one workgroup per matrix: matrix `wid.x + wid.y * nwg.x`.
pub(crate) fn per_matrix(ctx: &Context, count: usize) -> [u32; 3] {
    let max = ctx.max_workgroups_per_dimension() as usize;
    let x = count.min(max);
    [x as u32, count.div_ceil(x) as u32, 1]
}

/// Like [`launch`], for a kernel whose every access to workgroup and storage memory is in bounds by
/// construction (its sizes come from the launching code, which allocates and indexes to them; the
/// tests run it on every backend): built without the per-access bounds checks, which in the
/// column loops of the small factorisation kernels are most of their instructions.
pub(crate) fn launch_unchecked(ctx: &Context, key: &str, label: &str, make: impl FnOnce() -> String, groups: [u32; 3], args: &[KernelArg<'_>]) -> Result<(), LumaError> {
    // SAFETY: the callers' kernels index only within the sizes they allocate (see above).
    unsafe { ctx.keyed_kernel_unchecked(&format!("unchecked/{key}"), label, make) }?.launch_raw(groups, args)
}

/// Records the launch over `groups` workgroups of the kernel `make()` names by `key` (which must
/// determine the source) on `ctx`; the kernel is built once per context: the sources of the
/// register kernels are tens of kilobytes, and generating, copying and hashing one per launch cost
/// more host time than the launch. The kernels are cached in the context itself
/// ([`Context::keyed_kernel`]), never in a process-wide table: such a table (a kernel holds its
/// context) would keep contexts, and so their GPU devices, alive after their users finished.
pub(crate) fn launch(ctx: &Context, key: &str, label: &str, make: impl FnOnce() -> String, groups: [u32; 3], args: &[KernelArg<'_>]) -> Result<(), LumaError> {
    ctx.keyed_kernel(key, label, make)?.launch_raw(groups, args)
}

/// The NaN a failed pivot leaves, as WGSL.
pub(crate) const NAN: &str = "bitcast<f32>(0x7fc00000u)";

/// Matrix `index` of a batch as an array of its own (a view).
pub(crate) fn nth(a: &Array<f32>, index: usize) -> Array<f32> {
    a.slice(2, index..index + 1)
}

/// The block of rows `r` and columns `c` of a matrix (a view).
pub(crate) fn block(a: &Array<f32>, r: std::ops::Range<usize>, c: std::ops::Range<usize>) -> Array<f32> {
    a.slice(1, r).slice(0, c)
}

/// Joins `parts` along the batch axis.
pub(crate) fn stack_batch(parts: &[Array<f32>]) -> Array<f32> {
    crate::view::concat_slice(2, parts)
}

/// The barriers of a kernel that hands data between invocations through storage buffers: a storage
/// fence goes before the workgroup barrier on every backend but the CPU one, whose phases run in
/// order (a workgroup barrier alone orders only workgroup memory in WGSL, and on CUDA the compiler
/// may treat the `noalias` buffers as untouched by the barrier). `LUMA_LINALG_FENCE=0` leaves it
/// out (measured in the changelog), `1` forces it on.
pub(crate) fn fence(ctx: &Context) -> &'static str {
    use crate::BackendKind;
    let needed = match std::env::var("LUMA_LINALG_FENCE").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => !matches!(ctx.backend(), BackendKind::Cpu),
    };
    if needed { "storageBarrier();" } else { "" }
}

/// Rows `perm[i]` of `a` as row `i`: `a` is `[w, m, batch]`, `perm` `[m, 1, batch]`. Lazy.
pub(crate) fn gather_rows(a: &Array<f32>, perm: &Array<u32>, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = a.context();
    let [w, m, batch, _] = a.shape().dims();
    let shape = Shape::from([w, m, batch]);
    let (col, mat) = match (Array::<u32>::range_on(ctx, shape, 0), Array::<u32>::range_on(ctx, shape, 2)) {
        (Ok(c), Ok(b)) => (c, b),
        (Err(e), _) | (_, Err(e)) => return fail(ctx, a.shape(), e, "gather_rows", loc),
    };
    let rows = perm.reshape([1, m, batch]).broadcast_to(shape);
    let idx = &(&rows * w as u32 + &col) + &(&mat * (m * w) as u32);
    a.take(&idx)
}

/// Columns of the panels a big single matrix is factored in by the LU and QR panel kernels (8, 16 or
/// 32; `LUMA_LINALG_PANEL`): the kernels stream the whole panel through one multiprocessor's cache
/// for every column, so a narrow panel moves fewer bytes per column but leaves more levels of
/// recursion.
pub(crate) fn panel_width() -> usize {
    static WIDTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WIDTH.get_or_init(|| std::env::var("LUMA_LINALG_PANEL").ok().and_then(|v| v.parse().ok()).filter(|w| [8, 16, 32].contains(w)).unwrap_or(32))
}

/// Where the recursion over columns splits `w` of them: about half, at a multiple of 8 for panels
/// that are still narrow.
pub(crate) fn split_cols(w: usize) -> usize {
    if w <= 64 { (w / 2).next_multiple_of(8).min(w - 1).max(1) } else { tri::split(w) }
}

/// Whether the trailing-update products of the factorisations run on tensor cores in TF32
/// (`LUMA_LINALG_TF32=1`; default off: every product is the exact `f32` one). The operands are
/// rounded to a 10-bit mantissa, so the factors carry a backward error of about 1e-3 of the
/// entries of the updates instead of 1e-7; only the CUDA backend on a device with tensor cores of
/// compute capability 8.0 or higher takes it (see [`Array::matmul_tf32`]), every other
/// backend computes the `f32` product anyway.
pub(crate) fn tf32_updates() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LUMA_LINALG_TF32").is_ok_and(|v| v.trim() == "1"))
}

/// The product of the update of a factorisation: [`matmul_f32`](crate::matmul::matmul_f32), or the TF32
/// one with [`tf32_updates`].
pub(crate) fn product(lhs: &Array<f32>, rhs: &Array<f32>) -> Array<f32> {
    if tf32_updates() { crate::matmul::matmul_tf32(lhs, rhs) } else { crate::matmul::matmul_f32(lhs, rhs) }
}
