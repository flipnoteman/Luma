//! Cholesky factorisation and SPD solves.
//!
//! A big matrix is factored recursively, `A = [A11 A21ᵀ; A21 A22]`: `L11 = chol(A11)`,
//! `L21ᵀ = L11⁻¹ A21ᵀ` (a triangular solve, see [`tri`](super::tri)), `L22 = chol(A22 - L21 L21ᵀ)`
//! with the update one [`matmul_f32`]; the big update happens once at the top, so the elementwise
//! passes over the trailing matrix stay `O(n²)` in all. Up to 64 rows, and for every matrix of a
//! batch, one workgroup factors a matrix in workgroup memory.

use std::panic::Location;

use super::tri::{block_size, lower_left, rhs_matrix};
use super::{block, fail, launch, nth, per_matrix, stack_batch, Dims, NAN};
use crate::array::Array;
use crate::context::Context;
use crate::kernel::{uniform, view, KernelArg};
use crate::matmul::matmul_nt;
use crate::shape::Shape;

/// Rows of the matrix a workgroup factors in workgroup memory, at most: 64 where the device has the
/// 16.5 KiB that takes, else 48.
fn leaf(ctx: &Context) -> usize {
    if ctx.info().limits.max_workgroup_storage_bytes >= 64 * 65 * 4 + 64 * 4 { 64 } else { 48 }
}

#[derive(encase::ShaderType)]
struct Params {
    n: u32,
    batch: u32,
}

/// The leaf kernel: a workgroup of `LEAF` invocations factors one matrix of up to `leaf` rows in
/// workgroup memory, an invocation per row. Column `j` of every row `i >= j` is
/// `(a_ij - sum_k l_ik l_jk) / l_jj`, where the pivot `l_jj` is computed by every invocation from
/// row `j` (a few more flops, one barrier per column instead of two); the diagonal goes to its own
/// array so that the row `j` the others read is not overwritten under them.
fn source(leaf: usize) -> String {
    let pitch = leaf | 1;
    format!(
        "struct Params {{ n: u32, batch: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read_write> l: array<f32>;
@group(1) @binding(3) var<uniform> p: Params;
fn ld(mi: u32, i: u32, c: u32) -> f32 {{ return a[luma_view_idx3(av, c, i, mi)]; }}

var<workgroup> s: array<f32, {size}>;
var<workgroup> dv: array<f32, {leaf}>;

@compute @workgroup_size({leaf})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let m = wid.x + wid.y * nwg.x;
    let n = select(0u, p.n, m < p.batch);
    let base = m * p.n * p.n;
    for (var idx = t; idx < n * n; idx = idx + {leaf}u) {{
        let i = idx / n;
        let j = idx - i * n;
        s[i * {pitch}u + j] = select(0.0, ld(m, i, j), j <= i);
    }}
    workgroupBarrier();
    let ri = t * {pitch}u;
    for (var j = 0u; j < n; j = j + 1u) {{
        let rj = j * {pitch}u;
        var v = s[ri + j];
        var d = s[rj + j];
        var k = 0u;
        for (; k + 4u <= j; k = k + 4u) {{
            let b0 = s[rj + k];
            let b1 = s[rj + k + 1u];
            let b2 = s[rj + k + 2u];
            let b3 = s[rj + k + 3u];
            let a0 = s[ri + k];
            let a1 = s[ri + k + 1u];
            let a2 = s[ri + k + 2u];
            let a3 = s[ri + k + 3u];
            v = fma(-a0, b0, v);
            v = fma(-a1, b1, v);
            v = fma(-a2, b2, v);
            v = fma(-a3, b3, v);
            d = fma(-b0, b0, d);
            d = fma(-b1, b1, d);
            d = fma(-b2, b2, d);
            d = fma(-b3, b3, d);
        }}
        for (; k < j; k = k + 1u) {{
            let lj = s[rj + k];
            v = fma(-s[ri + k], lj, v);
            d = fma(-lj, lj, d);
        }}
        let r = select({NAN}, sqrt(d), d > 0.0);
        if (t > j && t < n) {{
            s[ri + j] = v / r;
        }}
        if (t == j) {{
            dv[j] = r;
        }}
        workgroupBarrier();
    }}
    for (var idx = t; idx < n * n; idx = idx + {leaf}u) {{
        let i = idx / n;
        let j = idx - i * n;
        l[base + idx] = select(select(0.0, dv[i], i == j), s[i * {pitch}u + j], j < i);
    }}
}}
",
        size = leaf * pitch,
    )
}

/// One launch over the batch.
fn small(a: &Array<f32>, d: Dims, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = a.context();
    let mut out = match Array::<f32>::alloc(ctx, a.shape()) {
        Ok(out) => out,
        Err(e) => return fail(ctx, a.shape(), e, "cholesky", loc),
    };
    let params = Params { n: d.rows as u32, batch: d.batch as u32 };
    let args: [KernelArg<'_>; 3] = [view(a), (&mut out).into(), uniform(&params)];
    match launch(ctx, &format!("chol/{}", leaf(ctx)), "cholesky", || source(leaf(ctx)), per_matrix(ctx, d.batch), &args) {
        Ok(()) => out,
        Err(e) => fail(ctx, a.shape(), e, "cholesky", loc),
    }
}

#[derive(encase::ShaderType)]
struct BlockParams {
    n: u32,
    k: u32,
    nbw: u32,
    has_t: u32,
}

/// The kernel that factors the diagonal block of columns `k..k + nbw`: `A_kk - T[..nbw]` (`T` the
/// product of the finished columns, see [`blocked`]) in workgroup memory as a packed lower
/// triangle (row `i` at `i (i + 1) / 2`, so that the 32 rows of a warp read one column without bank
/// conflicts), two invocations per row. Panels of 8 columns: each column of a panel takes one
/// barrier, the row's multipliers coming from the (at most 8) columns of the panel so far; then a
/// rank-8 update of the rest of the triangle, the two invocations of a row sharing its columns.
/// The factor goes to `dk` (the panel kernel reads it) and into its place in the output.
fn diag_source(nb: usize) -> String {
    let t = 2 * nb;
    let size = nb * (nb + 1) / 2;
    let mut cols = String::new();
    for q in 0..8 {
        let mut dsum = String::new();
        let mut vsum = String::new();
        for r in 0..q {
            dsum += &format!("            {{ let lj = s[off(j) + c0 + {r}u]; d = fma(-lj, lj, d); }}\n");
            vsum += &format!("                v = fma(-s[off(i) + c0 + {r}u], s[off(j) + c0 + {r}u], v);\n");
        }
        cols += &format!(
            "        {{
            let j = c0 + {q}u;
            if (j < nbw) {{
                if (h == 0u && i >= j && i < nbw) {{
                    var d = s[off(j) + j];
{dsum}                    let r = select({NAN}, sqrt(d), d > 0.0);
                    if (i == j) {{
                        dg[j] = r;
                    }} else {{
                        var v = s[off(i) + j];
{vsum}                        s[off(i) + j] = v / r;
                    }}
                }}
                workgroupBarrier();
            }}
        }}
"
        );
    }
    let loads = (0..8).map(|r| format!("            let l{r} = s[off(i) + c0 + {r}u];\n")).collect::<String>();
    let fmas = (0..8).map(|r| format!("                acc = fma(-l{r}, s[off(q) + c0 + {r}u], acc);\n")).collect::<String>();
    format!(
        "struct Params {{ n: u32, k: u32, nbw: u32, has_t: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read> tm: array<f32>;
@group(1) @binding(3) var<storage, read_write> dk: array<f32>;
@group(1) @binding(4) var<storage, read_write> o: array<f32>;
@group(1) @binding(5) var<uniform> p: Params;

var<workgroup> s: array<f32, {size}>;
var<workgroup> dg: array<f32, {nb}>;

fn off(i: u32) -> u32 {{ return i * (i + 1u) / 2u; }}

@compute @workgroup_size({t})
fn main(@builtin(local_invocation_index) t: u32) {{
    let nbw = p.nbw;
    let k = p.k;
    let n = p.n;
    for (var idx = t; idx < nbw * nbw; idx = idx + {t}u) {{
        let i = idx / nbw;
        let j = idx - i * nbw;
        if (j <= i) {{
            var v = a[luma_view_idx3(av, k + j, k + i, 0u)];
            if (p.has_t != 0u) {{ v = v - tm[idx]; }}
            s[off(i) + j] = v;
        }}
    }}
    workgroupBarrier();
    let i = t % {nb}u;
    let h = t / {nb}u;
    for (var c0 = 0u; c0 < nbw; c0 = c0 + 8u) {{
{cols}        if (i < nbw && i >= c0 + 8u) {{
{loads}            for (var q = c0 + 8u + h; q <= i; q = q + 2u) {{
                var acc = s[off(i) + q];
{fmas}                s[off(i) + q] = acc;
            }}
        }}
        workgroupBarrier();
    }}
    for (var idx = t; idx < nbw * nbw; idx = idx + {t}u) {{
        let i = idx / nbw;
        let j = idx - i * nbw;
        let v = select(select(0.0, s[off(i) + j], j < i), dg[i], i == j);
        dk[i * {nb}u + j] = v;
        o[(k + i) * n + k + j] = v;
    }}
}}
"
    )
}

/// The kernel that solves the rows below a diagonal block: `X = (A_panel - T) L_kkᵀ⁻¹` by forward
/// substitution along each row, a warp (workgroup) taking 32 rows, a lane a row. The rows live in
/// workgroup memory column by column (pitch 33, no bank conflicts); `L_kk` is read from `dk`, the
/// same entry by every lane. Four columns at a step share the loads of the solved ones.
fn panel_source(nb: usize) -> String {
    let mut acc = String::new();
    let mut upd = String::new();
    let mut diag = String::new();
    for u in 0..4 {
        acc += &format!("        var s{u} = tile[(j + {u}u) * 33u + lane];\n");
        for v in 0..4 {
            upd += &format!("            s{u} = fma(-dk[(j + {u}u) * {nb}u + q + {v}u], x{v}, s{u});\n");
        }
        for v in 0..u {
            diag += &format!("        s{u} = fma(-dk[(j + {u}u) * {nb}u + j + {v}u], s{v}, s{u});\n");
        }
        diag += &format!("        s{u} = s{u} / dk[(j + {u}u) * {nb}u + j + {u}u];\n        tile[(j + {u}u) * 33u + lane] = s{u};\n");
    }
    let xs = (0..4).map(|v| format!("            let x{v} = tile[(q + {v}u) * 33u + lane];\n")).collect::<String>();
    // The last block of columns may be short: the columns past `nbw` are never read or written.
    format!(
        "struct Params {{ n: u32, k: u32, nbw: u32, has_t: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read> tm: array<f32>;
@group(1) @binding(3) var<storage, read> dk: array<f32>;
@group(1) @binding(4) var<storage, read_write> o: array<f32>;
@group(1) @binding(5) var<uniform> p: Params;

var<workgroup> tile: array<f32, {size}>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {{
    let nbw = p.nbw;
    let k = p.k;
    let n = p.n;
    let r0 = k + nbw + wid.x * 32u;
    let rows = min(32u, n - r0);
    for (var rr = 0u; rr < rows; rr = rr + 1u) {{
        for (var c = lane; c < nbw; c = c + 32u) {{
            var v = a[luma_view_idx3(av, k + c, r0 + rr, 0u)];
            if (p.has_t != 0u) {{ v = v - tm[(r0 + rr - k) * nbw + c]; }}
            tile[c * 33u + rr] = v;
        }}
    }}
    workgroupBarrier();
    if (lane < rows) {{
        for (var j = 0u; j + 4u <= nbw; j = j + 4u) {{
{acc}            for (var q = 0u; q < j; q = q + 4u) {{
{xs}{upd}            }}
{diag}        }}
        // The columns of a block short of a multiple of four, one at a time.
        for (var j = nbw - nbw % 4u; j < nbw; j = j + 1u) {{
            var s0 = tile[j * 33u + lane];
            for (var q = 0u; q < j; q = q + 1u) {{
                s0 = fma(-dk[j * {nb}u + q], tile[q * 33u + lane], s0);
            }}
            tile[j * 33u + lane] = s0 / dk[j * {nb}u + j];
        }}
    }}
    workgroupBarrier();
    for (var rr = 0u; rr < rows; rr = rr + 1u) {{
        for (var c = lane; c < nbw; c = c + 32u) {{
            o[(r0 + rr) * n + k + c] = tile[c * 33u + rr];
        }}
    }}
}}
",
        size = nb * 33
    )
}

/// The lower Cholesky factor of one `[n, n]` matrix, left-looking in blocks of `nb` columns. Per
/// block: `T = L[k.., ..k] L[k..k+nb, ..k]ᵀ` (one [`matmul_nt`] over everything finished so far, the
/// only big flops: `n³/3` in all, in `n / nb` products), the diagonal block `A_kk - T` factored by
/// one workgroup ([`diag_source`]) and the rows below it by a launch of one warp per 32 rows
/// ([`panel_source`]): three launches a block, every one of them writing into the one output.
fn blocked(a: &Array<f32>, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = a.context();
    let [n, ..] = a.shape().dims();
    let nb = block_size(ctx);
    let shape = Shape::from([n, n]);
    let setup = (Array::<f32>::zeros_on(ctx, shape), Array::<f32>::alloc(ctx, Shape::from([nb, nb])), Array::<f32>::alloc(ctx, Shape::from(1)));
    let (mut out, mut dk, dummy) = match setup {
        (Ok(o), Ok(d), Ok(m)) => (o, d, m),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return fail(ctx, a.shape(), e, "cholesky", loc),
    };
    let mut k = 0;
    while k < n {
        let nbw = nb.min(n - k);
        let t = (k > 0).then(|| matmul_nt(&block(&out, k..n, 0..k), &block(&out, k..k + nbw, 0..k)));
        let params = BlockParams { n: n as u32, k: k as u32, nbw: nbw as u32, has_t: u32::from(k > 0) };
        let tref = t.as_ref().unwrap_or(&dummy);
        let diag: [KernelArg<'_>; 5] = [view(a), tref.into(), (&mut dk).into(), (&mut out).into(), uniform(&params)];
        let key = format!("chol_diag/{nb}");
        if let Err(e) = launch(ctx, &key, "chol_diag", || diag_source(nb), [1, 1, 1], &diag) {
            return fail(ctx, a.shape(), e, "cholesky", loc);
        }
        let rows = n - k - nbw;
        if rows > 0 {
            let panel: [KernelArg<'_>; 5] = [view(a), tref.into(), (&dk).into(), (&mut out).into(), uniform(&params)];
            let key = format!("chol_panel/{nb}");
            if let Err(e) = launch(ctx, &key, "chol_panel", || panel_source(nb), [rows.div_ceil(32) as u32, 1, 1], &panel) {
                return fail(ctx, a.shape(), e, "cholesky", loc);
            }
        }
        k += nbw;
    }
    out
}

/// The lower Cholesky factor of the lower triangle of every matrix of `a`.
pub(crate) fn lower(a: &Array<f32>, loc: &'static Location<'static>) -> Array<f32> {
    let d = match Dims::square(a) {
        Ok(d) => d,
        Err(e) => return fail(a.context(), a.shape(), e, "cholesky", loc),
    };
    if d.rows <= leaf(a.context()) {
        return small(a, d, loc);
    }
    if d.batch > 1 {
        let parts: Vec<Array<f32>> = (0..d.batch).map(|i| blocked(&nth(a, i), loc)).collect();
        return stack_batch(&parts);
    }
    blocked(a, loc)
}

impl Array<f32> {
    /// The Cholesky factor `L` of a symmetric positive definite matrix `A` (`self`, `[n, n]` or a
    /// batch `[n, n, batch]`), `A = L Lᵀ` with `L` lower triangular (the strict upper triangle is
    /// zero). Only the lower triangle of `A` is read.
    ///
    /// A matrix that is not positive definite (a pivot that is not positive, or `NaN`) leaves a
    /// `NaN` on the diagonal of its factor from the failing column on; nothing is checked or
    /// raised. Matrices of up to 64 rows (and every matrix of a batch) are factored by one
    /// workgroup each in one launch; bigger ones recursively with [`matmul_f32`] products.
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// // [[4, 2], [2, 5]] = L Lᵀ with L = [[2, 0], [1, 2]]
    /// let a = Array::from_slice(&[4.0f32, 2.0, 2.0, 5.0], [2, 2])?;
    /// assert_eq!(a.cholesky().to_vec()?, vec![2.0, 0.0, 1.0, 2.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The result is poisoned if the matrices are not square.
    #[track_caller]
    pub fn cholesky(&self) -> Array<f32> {
        lower(self, Location::caller())
    }

    /// The upper Cholesky factor `U`, `A = Uᵀ U`, reading the upper triangle of `A`: the
    /// transpose of the lower factor of `Aᵀ`.
    ///
    /// # Errors
    ///
    /// As [`Array::cholesky`].
    #[track_caller]
    pub fn cholesky_upper(&self) -> Array<f32> {
        lower(&self.transpose(), Location::caller()).transpose()
    }

    /// Solves `A X = B` for a symmetric positive definite `A` (`self`, only its lower triangle is
    /// read) by Cholesky: `L Lᵀ X = B`. `b` is `[k, n]` (`[k, n, batch]`), or a vector (see
    /// [`Array::solve_triangular`]); the result has its shape. A matrix that is not positive
    /// definite gives `NaN`.
    ///
    /// # Errors
    ///
    /// The result is poisoned if `A` is not square or `b` does not match it.
    #[track_caller]
    pub fn solve_spd(&self, b: &Array<f32>) -> Array<f32> {
        let loc = Location::caller();
        let l = lower(self, loc);
        cholesky_solve(&l, b, loc)
    }

    /// Solves `L Lᵀ X = B` given the lower Cholesky factor `L` (`self`, as [`Array::cholesky`]
    /// returns it): the two triangular solves of [`Array::solve_spd`].
    ///
    /// # Errors
    ///
    /// The result is poisoned if `L` is not square or `b` does not match it.
    #[track_caller]
    pub fn cholesky_solve(&self, b: &Array<f32>) -> Array<f32> {
        cholesky_solve(self, b, Location::caller())
    }
}

fn cholesky_solve(l: &Array<f32>, b: &Array<f32>, loc: &'static Location<'static>) -> Array<f32> {
    let (rhs, vector) = rhs_matrix(l, b);
    let y = lower_left(l, &rhs, false, "cholesky_solve", loc);
    // Lᵀ X = Y: the transpose is upper, solved on its flip.
    let lt = l.transpose().flip(0).flip(1);
    let x = lower_left(&lt, &y.flip(1), false, "cholesky_solve", loc).flip(1);
    if vector { x.reshape(b.shape()) } else { x }
}


