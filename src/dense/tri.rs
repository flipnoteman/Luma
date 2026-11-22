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

use super::{fail, grid, launch, launch_unchecked, nth, stack_batch, Dims, Triangle};
use crate::array::Array;
use crate::error::LumaError;
use crate::kernel::{uniform, view, KernelArg};
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
        accs += &format!("        var s{r} = ldb(m, min(i0 + {r}u, n - 1u), c);\n");
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
@group(1) @binding(1) var<uniform> lv: LumaView;
@group(1) @binding(2) var<storage, read> b: array<f32>;
@group(1) @binding(3) var<uniform> bv: LumaView;
@group(1) @binding(4) var<storage, read_write> x: array<f32>;
@group(1) @binding(5) var<uniform> p: Params;
fn ldl(m: u32, i: u32, j: u32) -> f32 {{ return l[luma_view_idx3(lv, j, i, m)]; }}
fn ldb(m: u32, i: u32, c: u32) -> f32 {{ return b[luma_view_idx3(bv, c, i, m)]; }}

var<workgroup> ls: array<f32, {size}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let m = wid.y + wid.z * nwg.y;
    let n = select(0u, p.n, m < p.batch);
    let k = p.k;
    let lb = m * p.n * p.n;
    let bb = m * p.n * k;
    for (var idx = t; idx < n * n; idx = idx + 64u) {{
        ls[idx] = ldl(m, idx / n, idx % n);
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
        let parts: Vec<Array<f32>> = (0..ld.batch).map(|i| blocked(&nth(l, i), &nth(b, i), unit, op, loc)).collect();
        return stack_batch(&parts);
    }
    blocked(l, b, unit, op, loc)
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
    let args: [KernelArg<'_>; 4] = [view(l), view(b), (&mut out).into(), uniform(&params)];
    match launch(ctx, "trsm", "trsm", source, groups, &args) {
        Ok(()) => out,
        Err(e) => fail(ctx, b.shape(), e, op, loc),
    }
}

/// Rows of a diagonal block the blocked solve takes: 128 where a workgroup has the 33 KiB its packed
/// lower triangle takes, else 64 (`LUMA_CHOL_NB` overrides, as it does the factorisation's).
pub(crate) fn block_size(ctx: &crate::context::Context) -> usize {
    if let Some(nb) = std::env::var("LUMA_CHOL_NB").ok().and_then(|v| v.parse().ok()).filter(|nb| [32, 64, 128].contains(nb)) {
        return nb;
    }
    if ctx.info().limits.max_workgroup_storage_bytes >= (128 * 129 / 2) * 4 + 256 { 128 } else { 64 }
}

/// Columns of the right-hand sides a workgroup of the block solve takes.
const COLS: usize = 8;

#[derive(encase::ShaderType)]
struct BlockParams {
    n: u32,
    k: u32,
    j0: u32,
    nbw: u32,
    has_t: u32,
    unit: u32,
}

/// The block solve kernel: `X[j0..j0+nbw] = L_jj⁻¹ (B[j0..j0+nbw] - T)` where `T` (`[k, nbw]`) is
/// the contribution of the solved rows above. An invocation per row of the block holds [`COLS`]
/// right-hand-side columns of it; the triangle is in workgroup memory, packed (row `i` at
/// `i (i + 1) / 2`, a warp's rows read one column without bank conflicts). Column oriented:
/// after row `q` is solved its `COLS` values go to workgroup memory and every row below takes its
/// share, one barrier per row of the block.
fn block_source(nb: usize) -> String {
    let size = nb * (nb + 1) / 2;
    let mut init = String::new();
    let mut solve = String::new();
    let mut upd = String::new();
    for c in 0..COLS {
        init += &format!(
            "    var a{c} = 0.0;\n    if (col0 + {c}u < k && ii < nbw) {{
        a{c} = b[luma_view_idx3(bv, col0 + {c}u, j0 + ii, 0u)];
        if (p.has_t != 0u) {{ a{c} = a{c} - tm[ii * k + col0 + {c}u]; }}
    }}\n"
        );
        solve += &format!(
            "            let x{c} = a{c} / dv;\n            xs[buf * {COLS}u + {c}u] = x{c};\n            if (col0 + {c}u < k) {{ o[(j0 + q) * k + col0 + {c}u] = x{c}; }}\n"
        );
        upd += &format!("            a{c} = fma(-lq, xs[buf * {COLS}u + {c}u], a{c});\n");
    }
    // The triangle is read eight rows at a step, the loads issued together (see `chol::diag_source`).
    let mut lload = String::new();
    let mut lstore = String::new();
    for u in 0..8 {
        lload += &format!(
            "        var v{u} = 0.0;\n        let live{u} = ii < nbw && ii <= i0 + {u}u && i0 + {u}u < nbw;\n        if (live{u}) {{ v{u} = l[luma_view_idx3(lv, j0 + ii, j0 + i0 + {u}u, 0u)]; }}\n"
        );
        lstore += &format!("        if (live{u}) {{ ls[off(i0 + {u}u) + ii] = v{u}; }}\n");
    }
    format!(
        "struct Params {{ n: u32, k: u32, j0: u32, nbw: u32, has_t: u32, unit: u32 }}
@group(1) @binding(0) var<storage, read> l: array<f32>;
@group(1) @binding(1) var<uniform> lv: LumaView;
@group(1) @binding(2) var<storage, read> b: array<f32>;
@group(1) @binding(3) var<uniform> bv: LumaView;
@group(1) @binding(4) var<storage, read> tm: array<f32>;
@group(1) @binding(5) var<storage, read_write> o: array<f32>;
@group(1) @binding(6) var<uniform> p: Params;

var<workgroup> ls: array<f32, {size}>;
var<workgroup> xs: array<f32, {xsize}>;

fn off(i: u32) -> u32 {{ return i * (i + 1u) / 2u; }}

@compute @workgroup_size({nb})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) ii: u32) {{
    let k = p.k;
    let j0 = p.j0;
    let nbw = p.nbw;
    let col0 = wid.x * {COLS}u;
    for (var i0 = 0u; i0 < nbw; i0 = i0 + 8u) {{
{lload}{lstore}    }}
{init}    workgroupBarrier();
    for (var q = 0u; q < nbw; q = q + 1u) {{
        let buf = q & 1u;
        if (ii == q) {{
            let dv = select(ls[off(q) + q], 1.0, p.unit != 0u);
{solve}        }}
        workgroupBarrier();
        if (ii > q && ii < nbw) {{
            let lq = ls[off(ii) + q];
{upd}        }}
    }}
}}
",
        xsize = 2 * COLS
    )
}

/// `L X = B` for one matrix, in blocks of rows (left-looking): the rows above a block contribute
/// `T = L[block, ..j0] X[..j0]`, one [`matmul_f32`], and the block's own triangle is solved by one
/// launch of [`block_source`]; two launches a block, all of them writing into the one solution.
fn blocked(l: &Array<f32>, b: &Array<f32>, unit: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = l.context();
    let n = l.shape().dims()[1];
    let [k, ..] = b.shape().dims();
    let nb = block_size(ctx);
    let setup = (Array::<f32>::alloc(ctx, Shape::from([k, n])), Array::<f32>::alloc(ctx, Shape::from(1)));
    let (mut out, dummy) = match setup {
        (Ok(o), Ok(d)) => (o, d),
        (Err(e), _) | (_, Err(e)) => return fail(ctx, b.shape(), e, op, loc),
    };
    let mut j0 = 0;
    while j0 < n {
        let nbw = nb.min(n - j0);
        let t = (j0 > 0).then(|| matmul_f32(&l.slice(1, j0..j0 + nbw).slice(0, 0..j0), &out.slice(1, 0..j0)));
        let params = BlockParams { n: n as u32, k: k as u32, j0: j0 as u32, nbw: nbw as u32, has_t: u32::from(j0 > 0), unit: u32::from(unit) };
        let args: [KernelArg<'_>; 5] = [view(l), view(b), t.as_ref().unwrap_or(&dummy).into(), (&mut out).into(), uniform(&params)];
        let key = format!("trsm_block/{nb}");
        if let Err(e) = launch_unchecked(ctx, &key, "trsm_block", || block_source(nb), [k.div_ceil(COLS) as u32, 1, 1], &args) {
            return fail(ctx, b.shape(), e, op, loc);
        }
        j0 += nbw;
    }
    out
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
