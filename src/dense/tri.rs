//! Triangular solves.
//!
//! Every variant (upper or lower, transposed or not) is turned into one canonical solve,
//! `L X = B` with `L` lower triangular, by flipping and transposing views; those fuse into the
//! copy the kernels make of their operands. The canonical solve is recursive: split `L` in
//! halves, solve the top block, update the bottom right-hand side with a [`matmul_f32`] and
//! solve the bottom block. Below [`BASE`] rows (or for a batch of small matrices) a kernel
//! solves by substitution, one invocation per right-hand-side column and the whole batch in one
//! launch; the entries of `L` it reads are the same for every invocation of a workgroup.

use std::panic::Location;

use super::{fail, grid, launch, nth, stack_batch, Dims, Triangle};
use crate::array::Array;
use crate::error::LumaError;
use crate::kernel::{uniform, KernelArg};
use crate::matmul::matmul_f32;
use crate::shape::Shape;

/// Largest triangle a substitution kernel solves; bigger ones recurse.
pub(crate) const BASE: usize = 64;

#[derive(encase::ShaderType)]
struct Params {
    n: u32,
    k: u32,
    batch: u32,
    unit: u32,
}

/// Rows solved together by the substitution kernel: the entries of the earlier rows are read once
/// for all of them, and their sums are independent chains of fused multiply-adds.
const ROWS: usize = 8;
// The loop over the solved rows takes four at a step (ROWS is a multiple of 4), with their loads
// issued together: the loads of `x` are what the kernel waits for.

/// The substitution kernel: `x = l^-1 b` for every matrix of a batch, an invocation per column
/// of `b`, the triangle in workgroup memory (every invocation reads the same entry). Blocks of
/// [`ROWS`] rows: first the contributions of every solved row to the block's rows, then the
/// block's own small triangle.
fn source() -> String {
    let mut accs = String::new();
    let mut update = String::new();
    let mut diag = String::new();
    for r in 0..ROWS {
        accs += &format!("        var s{r} = b[bb + min(i0 + {r}u, n - 1u) * k + c];\n");
        let row = format!("min(i0 + {r}u, n - 1u) * n + j");
        update += &format!(
            "            s{r} = fma(-ls[{row}], xa, s{r});\n            s{r} = fma(-ls[{row} + 1u], xb, s{r});\n            s{r} = fma(-ls[{row} + 2u], xc, s{r});\n            s{r} = fma(-ls[{row} + 3u], xd, s{r});\n"
        );
        let mut inner = String::new();
        for q in 0..r {
            inner += &format!("        s{r} = fma(-ls[min(i0 + {r}u, n - 1u) * n + i0 + {q}u], x{q}, s{r});\n");
        }
        diag += &format!(
            "{inner}        if (p.unit == 0u) {{ s{r} = s{r} / ls[min(i0 + {r}u, n - 1u) * n + min(i0 + {r}u, n - 1u)]; }}\n        let x{r} = s{r};\n        if (i0 + {r}u < n) {{ x[bb + (i0 + {r}u) * k + c] = x{r}; }}\n"
        );
    }
    format!(
        "struct Params {{ n: u32, k: u32, batch: u32, unit: u32 }}
@group(1) @binding(0) var<storage, read> l: array<f32>;
@group(1) @binding(1) var<storage, read> b: array<f32>;
@group(1) @binding(2) var<storage, read_write> x: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;
var<workgroup> ls: array<f32, {size}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let m = wid.y + wid.z * nwg.y;
    let n = select(0u, p.n, m < p.batch);
    let k = p.k;
    let lb = m * p.n * p.n;
    let bb = m * p.n * k;
    for (var idx = t; idx < n * n; idx = idx + 64u) {{
        ls[idx] = l[lb + idx];
    }}
    workgroupBarrier();
    let c = wid.x * 64u + t;
    if (c >= k) {{ return; }}
    for (var i0 = 0u; i0 < n; i0 = i0 + {ROWS}u) {{
{accs}        for (var j = 0u; j < i0; j = j + 4u) {{
            let xa = x[bb + j * k + c];
            let xb = x[bb + (j + 1u) * k + c];
            let xc = x[bb + (j + 2u) * k + c];
            let xd = x[bb + (j + 3u) * k + c];
{update}        }}
{diag}    }}
}}
",
        size = BASE * BASE
    )
}

/// `L X = B` for the lower triangles of the `[n, n, batch]` matrices `l` and `[k, n, batch]`
/// right-hand sides `b`; the strict upper triangle of `l` (and the diagonal with `unit`) is not
/// read.
pub(crate) fn lower_left(l: &Array<f32>, b: &Array<f32>, unit: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = l.context();
    let shape = b.shape();
    let dims = match (Dims::square(l), Dims::of(b)) {
        (Ok(ld), Ok(bd)) if ld.rows == bd.rows && ld.batch == bd.batch => (ld, bd),
        (Ok(_), Ok(_)) => {
            let error = LumaError::ShapeMismatch { lhs: l.shape().dims(), rhs: b.shape().dims() };
            return fail(ctx, shape, error, op, loc);
        }
        (Err(e), _) | (_, Err(e)) => return fail(ctx, shape, e, op, loc),
    };
    let (ld, bd) = dims;
    if ld.rows <= BASE {
        return substitute(l, b, ld, bd, unit, op, loc);
    }
    if ld.batch > 1 {
        let parts: Vec<Array<f32>> = (0..ld.batch).map(|i| recurse(&nth(l, i), &nth(b, i), unit, op, loc)).collect();
        return stack_batch(&parts);
    }
    recurse(l, b, unit, op, loc)
}

/// One kernel launch over the whole batch.
fn substitute(l: &Array<f32>, b: &Array<f32>, ld: Dims, bd: Dims, unit: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = l.context();
    let out = Array::<f32>::alloc(ctx, b.shape());
    let mut out = match out {
        Ok(out) => out,
        Err(e) => return fail(ctx, b.shape(), e, op, loc),
    };
    let params = Params { n: ld.rows as u32, k: bd.cols as u32, batch: ld.batch as u32, unit: u32::from(unit) };
    let groups = grid(ctx, bd.cols.div_ceil(64), ld.batch);
    let args: [KernelArg<'_>; 4] = [l.into(), b.into(), (&mut out).into(), uniform(&params)];
    match launch(ctx, "trsm", "trsm", source, groups, &args) {
        Ok(()) => out,
        Err(e) => fail(ctx, b.shape(), e, op, loc),
    }
}

/// The recursion over one `[n, n]` matrix.
fn recurse(l: &Array<f32>, b: &Array<f32>, unit: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let [_, n, ..] = l.shape().dims();
    let [k, ..] = b.shape().dims();
    if n <= BASE {
        let (ld, bd) = (Dims { rows: n, cols: n, batch: 1 }, Dims { rows: n, cols: k, batch: 1 });
        return substitute(l, b, ld, bd, unit, op, loc);
    }
    let h = split(n);
    let blk = |a: &Array<f32>, r: std::ops::Range<usize>, c: std::ops::Range<usize>| a.slice(1, r).slice(0, c);
    let x1 = recurse(&blk(l, 0..h, 0..h), &blk(b, 0..h, 0..k), unit, op, loc).cache();
    let update = &blk(b, h..n, 0..k) - &matmul_f32(&blk(l, h..n, 0..h).cache(), &x1);
    let x2 = recurse(&blk(l, h..n, h..n), &update, unit, op, loc);
    crate::view::concat(1, [&x1, &x2])
}

/// Where a triangle of `n` rows splits: about half, at a multiple of 32 so that the blocks of the
/// products are whole tiles.
pub(crate) fn split(n: usize) -> usize {
    (n / 2).next_multiple_of(32).min(n - 1).max(1)
}

impl Array<f32> {
    /// Solves `op(A) X = B` for triangular `A` (`self`, `[n, n]` or a batch `[n, n, batch]`) and
    /// right-hand sides `b` of `[k, n]` (`[k, n, batch]`; a plain `[n]` or `[n, 1, batch]` array is
    /// one vector). `triangle` says which triangle of `A` is read (the other one is ignored),
    /// `transpose` solves with `Aᵀ`, `unit_diag` takes the diagonal as ones without reading it.
    ///
    /// A zero on the diagonal (not `unit_diag`) gives `Inf` / `NaN` entries; it is not checked.
    /// Big triangles are solved recursively with [`matmul_f32`] products, small ones and
    /// batches in one kernel with one invocation per column of `b`. Matrices are `[cols, rows]`, as in
    /// [`matmul`](crate::matmul).
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// // [[2, 0], [1, 4]] x = [2, 9]  =>  x = [1, 2]
    /// let l = Array::from_slice(&[2.0f32, 0.0, 1.0, 4.0], [2, 2])?;
    /// let b = Array::from_slice(&[2.0f32, 9.0], 2)?;
    /// let x = l.solve_triangular(&b, Triangle::Lower, false, false);
    /// assert_eq!(x.to_vec()?, vec![1.0, 2.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The result is poisoned if `A` is not square, `b` does not have `n` rows, or the batches
    /// differ.
    #[track_caller]
    pub fn solve_triangular(&self, b: &Array<f32>, triangle: Triangle, transpose: bool, unit_diag: bool) -> Array<f32> {
        let loc = Location::caller();
        let (rhs, vector) = rhs_matrix(self, b);
        let x = match (triangle, transpose) {
            (Triangle::Lower, false) => lower_left(self, &rhs, unit_diag, "solve_triangular", loc),
            (Triangle::Upper, true) => lower_left(&self.transpose(), &rhs, unit_diag, "solve_triangular", loc),
            (Triangle::Upper, false) => lower_left(&self.flip(0).flip(1), &rhs.flip(1), unit_diag, "solve_triangular", loc).flip(1),
            (Triangle::Lower, true) => lower_left(&self.transpose().flip(0).flip(1), &rhs.flip(1), unit_diag, "solve_triangular", loc).flip(1),
        };
        if vector { x.reshape(b.shape()) } else { x }
    }
}

/// The right-hand side `b` of a system with matrix `a` as `[k, n, batch]`: a vector
/// (`[n, 1, batch]`, or `[n]`) becomes one column. Whether it was a vector.
pub(crate) fn rhs_matrix(a: &Array<f32>, b: &Array<f32>) -> (Array<f32>, bool) {
    let [n, rows, _, last] = a.shape().dims();
    let [bc, br, bb, bl] = b.shape().dims();
    if rows == n && br == 1 && bc == n && n > 1 && bl == 1 && last == 1 {
        return (b.reshape(Shape::from([1, n, bb])), true);
    }
    (b.clone(), false)
}
