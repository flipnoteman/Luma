//! LU factorisation with partial pivoting, general solves, inverse and determinant.
//!
//! `P A = L U` with `L` unit lower and `U` upper triangular, packed in one matrix; `P` is the
//! permutation vector `perm` (row `i` of `P A` is row `perm[i]` of `A`).
//!
//! Matrices up to 64 rows (a batch of them in one launch) are factored by a workgroup each in
//! workgroup memory; a tall panel of up to 32 columns by one workgroup in place; anything
//! bigger by recursion on the columns (Toledo): factor the left half, apply its row swaps to the
//! right half, `U12 = L11⁻¹ A12` (a triangular solve), update `A22 - L21 U12` with one
//! [`matmul_f32`] and factor that.

use std::panic::Location;

use super::tri::{lower_left, rhs_matrix, split};
use super::{block, fail, fence, gather_rows, launch, nth, per_matrix, stack_batch, Dims};
use crate::array::Array;
use crate::error::LumaError;
use crate::kernel::{uniform, KernelArg};
use crate::matmul::matmul_f32;
use crate::ops::select;
use crate::shape::Shape;
use crate::view::{concat, concat_slice};


/// Rows and columns a workgroup factors in workgroup memory.
const SMALL: usize = 64;
/// Columns of a panel one workgroup factors in global memory.
const PANEL: usize = 32;

/// A packed LU factorisation, see [`Array::lu`].
#[derive(Clone, Debug)]
pub struct Lu {
    /// `L` (strictly below the diagonal, its unit diagonal is implied) and `U` (on and above it),
    /// `[n, n]` or `[n, n, batch]`.
    pub lu: Array<f32>,
    /// The row permutation: row `i` of `P A` is row `perm[i]` of `A`; `[n, 1]` or `[n, 1, batch]`.
    pub perm: Array<u32>,
}

#[derive(encase::ShaderType)]
struct Params {
    m: u32,
    w: u32,
    batch: u32,
}

/// Workgroup-memory kernel: a matrix of up to 64 x 64 per workgroup, an invocation per row.
fn small_source() -> String {
    let pitch = SMALL | 1;
    format!(
        "struct Params {{ m: u32, w: u32, batch: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<storage, read_write> f: array<f32>;
@group(1) @binding(2) var<storage, read_write> perm: array<u32>;
@group(1) @binding(3) var<uniform> p: Params;
var<workgroup> s: array<f32, {size}>;
var<workgroup> pm: array<u32, {SMALL}>;

@compute @workgroup_size({SMALL})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let mi = wid.x + wid.y * nwg.x;
    let valid = mi < p.batch;
    let m = select(0u, p.m, valid);
    let w = p.w;
    let base = mi * p.m * w;
    for (var idx = t; idx < m * w; idx = idx + {SMALL}u) {{
        let i = idx / w;
        s[i * {pitch}u + idx - i * w] = a[base + idx];
    }}
    pm[t] = t;
    workgroupBarrier();
    let ri = t * {pitch}u;
    for (var j = 0u; j < min(m, w); j = j + 1u) {{
        var best = -1.0;
        var bi = j;
        for (var i = j; i < m; i = i + 1u) {{
            let v = abs(s[i * {pitch}u + j]);
            if (v > best) {{ best = v; bi = i; }}
        }}
        workgroupBarrier();
        if (bi != j) {{
            if (t < w) {{
                let x = s[j * {pitch}u + t];
                s[j * {pitch}u + t] = s[bi * {pitch}u + t];
                s[bi * {pitch}u + t] = x;
            }}
            if (t == 0u) {{
                let x = pm[j];
                pm[j] = pm[bi];
                pm[bi] = x;
            }}
        }}
        workgroupBarrier();
        if (t > j && t < m) {{
            let rj = j * {pitch}u;
            let piv = s[rj + j];
            let l = select(0.0, s[ri + j] / piv, piv != 0.0);
            s[ri + j] = l;
            for (var c = j + 1u; c < w; c = c + 1u) {{
                s[ri + c] = fma(-l, s[rj + c], s[ri + c]);
            }}
        }}
        workgroupBarrier();
    }}
    for (var idx = t; idx < m * w; idx = idx + {SMALL}u) {{
        let i = idx / w;
        f[base + idx] = s[i * {pitch}u + idx - i * w];
    }}
    if (t < m) {{ perm[mi * p.m + t] = pm[t]; }}
}}
",
        size = SMALL * pitch
    )
}

/// Invocations of a panel kernel for a matrix of `m` rows: more rows mean more loads in flight (the
/// kernel is bound by the latency of its global loads, not by arithmetic), capped by the device.
pub(crate) fn panel_threads(ctx: &crate::Context, m: usize) -> usize {
    let limits = &ctx.info().limits;
    let cap = limits.max_workgroup_invocations.min(limits.max_workgroup_size[0]) as usize;
    let want = if m <= 64 { 64 } else if m <= 512 { 256 } else { 1024 };
    want.min(cap / 32 * 32).max(32)
}

/// Global-memory kernel: one workgroup factors a tall panel of up to [`PANEL`] columns in place.
/// Invocation `(g, c) = (t / 32, t % 32)` owns column `c` of rows `g, g + G, ...`, so a warp reads
/// and writes a row segment. Per column: swap the pivot row in, then one pass over the rows below
/// that updates the other columns with the multiplier (the column itself is scaled after the last
/// one), four rows per step with all their loads issued first, in which the lanes of the next column
/// also keep the largest entry they have updated: the next pivot comes from a reduction over the
/// row groups, no separate scan of the column. (An RTX 3090 spent 60 us per column on this loop
/// with one row per step and 256 invocations: 8 loads in flight per row group.)
fn panel_source(fence: &str, threads: usize) -> String {
    let t = threads;
    let g = threads / 32;
    format!(
        "struct Params {{ m: u32, w: u32, batch: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<storage, read_write> f: array<f32>;
@group(1) @binding(2) var<storage, read_write> perm: array<u32>;
@group(1) @binding(3) var<uniform> p: Params;
var<workgroup> rv: array<f32, {g}>;
var<workgroup> ri: array<u32, {g}>;

fn next_pivot() -> u32 {{
    var bv = -2.0;
    var bidx = 0xffffffffu;
    for (var q = 0u; q < {g}u; q = q + 1u) {{
        if (rv[q] > bv || (rv[q] == bv && ri[q] < bidx)) {{ bv = rv[q]; bidx = ri[q]; }}
    }}
    return bidx;
}}

@compute @workgroup_size({t})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let mi = wid.x + wid.y * nwg.x;
    let m = select(0u, p.m, mi < p.batch);
    let w = p.w;
    let base = mi * p.m * w;
    let pb = mi * p.m;
    let g = t / 32u;
    let c = t % 32u;
    var best = -1.0;
    var bi = 0u;
    if (c < w) {{
        var i = g;
        for (; i + {g3}u < m; i = i + {g4}u) {{
            let r0 = base + i * w + c;
            let v0 = a[r0];
            let v1 = a[r0 + {g1}u * w];
            let v2 = a[r0 + {g2}u * w];
            let v3 = a[r0 + {g3}u * w];
            f[r0] = v0;
            f[r0 + {g1}u * w] = v1;
            f[r0 + {g2}u * w] = v2;
            f[r0 + {g3}u * w] = v3;
            if (c == 0u) {{
                if (abs(v0) > best) {{ best = abs(v0); bi = i; }}
                if (abs(v1) > best) {{ best = abs(v1); bi = i + {g1}u; }}
                if (abs(v2) > best) {{ best = abs(v2); bi = i + {g2}u; }}
                if (abs(v3) > best) {{ best = abs(v3); bi = i + {g3}u; }}
            }}
        }}
        for (; i < m; i = i + {g1}u) {{
            let v = a[base + i * w + c];
            f[base + i * w + c] = v;
            if (c == 0u && abs(v) > best) {{ best = abs(v); bi = i; }}
        }}
    }}
    for (var i = t; i < m; i = i + {t}u) {{
        perm[pb + i] = i;
    }}
    if (c == 0u) {{ rv[g] = best; ri[g] = bi; }}
    {fence}
    workgroupBarrier();
    var pv = next_pivot();
    for (var j = 0u; j < min(m, w); j = j + 1u) {{
        if (pv != j) {{
            if (t < w) {{
                let x = f[base + j * w + t];
                f[base + j * w + t] = f[base + pv * w + t];
                f[base + pv * w + t] = x;
            }}
            if (t == 0u) {{
                let x = perm[pb + j];
                perm[pb + j] = perm[pb + pv];
                perm[pb + pv] = x;
            }}
        }}
        {fence}
        workgroupBarrier();
        let piv = f[base + j * w + j];
        let rpiv = select(0.0, 1.0 / piv, piv != 0.0);
        let prow = select(0.0, f[base + j * w + c], c < w);
        let act = c > j && c < w;
        best = -1.0;
        bi = j + 1u;
        var i = j + 1u + g;
        for (; i + {g3}u < m; i = i + {g4}u) {{
            let r0 = base + i * w;
            let l0 = f[r0 + j] * rpiv;
            let l1 = f[r0 + {g1}u * w + j] * rpiv;
            let l2 = f[r0 + {g2}u * w + j] * rpiv;
            let l3 = f[r0 + {g3}u * w + j] * rpiv;
            if (act) {{
                let x0 = fma(-l0, prow, f[r0 + c]);
                let x1 = fma(-l1, prow, f[r0 + {g1}u * w + c]);
                let x2 = fma(-l2, prow, f[r0 + {g2}u * w + c]);
                let x3 = fma(-l3, prow, f[r0 + {g3}u * w + c]);
                f[r0 + c] = x0;
                f[r0 + {g1}u * w + c] = x1;
                f[r0 + {g2}u * w + c] = x2;
                f[r0 + {g3}u * w + c] = x3;
                if (c == j + 1u) {{
                    if (abs(x0) > best) {{ best = abs(x0); bi = i; }}
                    if (abs(x1) > best) {{ best = abs(x1); bi = i + {g1}u; }}
                    if (abs(x2) > best) {{ best = abs(x2); bi = i + {g2}u; }}
                    if (abs(x3) > best) {{ best = abs(x3); bi = i + {g3}u; }}
                }}
            }}
        }}
        for (; i < m; i = i + {g1}u) {{
            let ro = base + i * w;
            let l = f[ro + j] * rpiv;
            if (act) {{
                let nv = fma(-l, prow, f[ro + c]);
                f[ro + c] = nv;
                if (c == j + 1u && abs(nv) > best) {{ best = abs(nv); bi = i; }}
            }}
        }}
        if (c == j + 1u) {{ rv[g] = best; ri[g] = bi; }}
        {fence}
        workgroupBarrier();
        pv = next_pivot();
    }}
    // The multipliers are scaled last: the invocations of a warp read the column they scale
    // while the others update, so nothing may write it before the loop is done.
    for (var i = g; i < m; i = i + {g1}u) {{
        if (c < min(m, w) && i > c) {{
            let piv = f[base + c * w + c];
            if (piv != 0.0) {{ f[base + i * w + c] = f[base + i * w + c] * (1.0 / piv); }}
        }}
    }}
}}
",
        g1 = g,
        g2 = 2 * g,
        g3 = 3 * g,
        g4 = 4 * g
    )
}

/// Factors every `[w, m, batch]` matrix of `a` (`m >= w`) with one of the two kernels.
fn leaf(a: &Array<f32>, d: Dims, op: &'static str, loc: &'static Location<'static>) -> Lu {
    let ctx = a.context();
    let perm_shape = Shape::from([d.rows, 1, d.batch]);
    let poisoned = |e: LumaError| Lu { lu: fail(ctx, a.shape(), e, op, loc), perm: Array::resolve(ctx, perm_shape, Err(crate::array::Failure::New(LumaError::InvalidShape("lu failed".into()))), op, loc) };
    let (mut f, mut perm) = match (Array::<f32>::alloc(ctx, a.shape()), Array::<u32>::alloc(ctx, perm_shape)) {
        (Ok(f), Ok(p)) => (f, p),
        (Err(e), _) | (_, Err(e)) => return poisoned(e),
    };
    let params = Params { m: d.rows as u32, w: d.cols as u32, batch: d.batch as u32 };
    let args: [KernelArg<'_>; 4] = [a.into(), (&mut f).into(), (&mut perm).into(), uniform(&params)];
    let result = if d.rows <= SMALL {
        launch(ctx, &small_source(), "lu_small", per_matrix(ctx, d.batch), &args)
    } else {
        launch(ctx, &panel_source(fence(ctx), panel_threads(ctx, d.rows)), "lu_panel", per_matrix(ctx, d.batch), &args)
    };
    match result {
        Ok(()) => Lu { lu: f, perm },
        Err(e) => poisoned(e),
    }
}

/// The recursion over one `[w, m]` matrix, `m >= w`.
fn recurse(a: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Lu {
    let [w, m, ..] = a.shape().dims();
    if m <= SMALL || w <= PANEL {
        return leaf(a, Dims { rows: m, cols: w, batch: 1 }, op, loc);
    }
    let n1 = split(w);
    let n2 = w - n1;
    let Lu { lu: f1, perm: p1 } = recurse(&block(a, 0..m, 0..n1), op, loc);
    let f1 = f1.cache();
    // The right columns in the order of the left panel's pivots.
    let right = gather_rows(&block(a, 0..m, n1..w), &p1, loc);
    let a12 = block(&right, 0..n1, 0..n2);
    let a22 = block(&right, n1..m, 0..n2);
    let u12 = lower_left(&block(&f1, 0..n1, 0..n1), &a12, true, op, loc).cache();
    let l21 = block(&f1, n1..m, 0..n1);
    let s = &a22 - &matmul_f32(&l21.cache(), &u12);
    let Lu { lu: f2, perm: p2 } = recurse(&s, op, loc);
    let l21 = gather_rows(&l21, &p2, loc);
    let perm = concat(0, [&p1.slice(0, 0..n1), &p1.slice(0, n1..m).take(&p2)]);
    let top = concat(0, [&block(&f1, 0..n1, 0..n1), &u12]);
    let bottom = concat(0, [&l21, &f2]);
    Lu { lu: concat(1, [&top, &bottom]), perm }
}

fn factor(a: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Lu {
    let ctx = a.context();
    let d = match Dims::of(a) {
        Ok(d) if d.rows >= d.cols => d,
        Ok(_) => {
            let e = LumaError::InvalidShape(format!("lu needs at least as many rows as columns, got shape {}", a.shape()));
            return Lu { lu: fail(ctx, a.shape(), e, op, loc), perm: Array::resolve(ctx, Shape::from(1), Err(crate::array::Failure::New(LumaError::InvalidShape("lu failed".into()))), op, loc) };
        }
        Err(e) => return Lu { lu: fail(ctx, a.shape(), e, op, loc), perm: Array::resolve(ctx, Shape::from(1), Err(crate::array::Failure::New(LumaError::InvalidShape("lu failed".into()))), op, loc) },
    };
    if d.rows <= SMALL || (d.cols <= PANEL && d.batch == 1) {
        return leaf(a, d, op, loc);
    }
    if d.batch > 1 {
        let parts: Vec<Lu> = (0..d.batch).map(|i| recurse(&nth(a, i), op, loc)).collect();
        let lus: Vec<Array<f32>> = parts.iter().map(|p| p.lu.clone()).collect();
        let perms: Vec<Array<u32>> = parts.iter().map(|p| p.perm.clone()).collect();
        return Lu { lu: stack_batch(&lus), perm: concat_slice(2, &perms) };
    }
    recurse(a, op, loc)
}

impl Array<f32> {
    /// LU factorisation with partial pivoting of a matrix (`[n, m]`, `m >= n`, or a batch
    /// `[n, m, batch]`): `P A = L U`, see [`Lu`]. Rows are swapped to bring the entry of largest
    /// magnitude of the column to the diagonal; a column with no nonzero entry (a singular
    /// matrix) leaves a zero on the diagonal of `U`, which [`Array::solve`] turns into `Inf` /
    /// `NaN` and [`Array::det`] into 0; nothing is checked.
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// // [[0, 1], [2, 3]]: the first pivot is the 2 of row 1.
    /// let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], [2, 2])?;
    /// let f = a.lu();
    /// assert_eq!(f.perm.to_vec()?, vec![1, 0]);
    /// assert_eq!(f.lu.to_vec()?, vec![2.0, 3.0, 0.0, 1.0]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The factors are poisoned if there are more columns than rows.
    #[track_caller]
    pub fn lu(&self) -> Lu {
        factor(self, "lu", Location::caller())
    }

    /// Solves `A X = B` (`self` is `A`) by LU with partial pivoting for a square `A`; `b` is
    /// `[k, n]` (`[k, n, batch]`) or, for a square `A`, a vector (see [`Array::solve_triangular`]).
    /// A singular matrix gives `Inf` / `NaN`. A tall `A` (more rows than columns) gives the
    /// least-squares solution ([`Array::lstsq`]), a wide one the minimum norm solution (by QR of
    /// the transpose), as ArrayFire's `solve` does.
    ///
    /// # Errors
    ///
    /// The result is poisoned if `b` does not match `A`.
    #[track_caller]
    pub fn solve(&self, b: &Array<f32>) -> Array<f32> {
        let loc = Location::caller();
        match Dims::of(self) {
            Ok(d) if d.rows == d.cols => solve_with(&factor(self, "solve", loc), b, "solve", loc),
            Ok(d) if d.rows > d.cols => super::qr::Qr::lstsq(&self.qr(), b),
            Ok(_) => super::qr::min_norm(self, b, "solve", loc),
            Err(e) => fail(self.context(), b.shape(), e, "solve", loc),
        }
    }

    /// The inverse of a square matrix (or of every matrix of a batch), `A X = I` by LU. A
    /// singular matrix gives `Inf` / `NaN`.
    ///
    /// # Errors
    ///
    /// The result is poisoned if the matrices are not square.
    #[track_caller]
    pub fn inverse(&self) -> Array<f32> {
        let loc = Location::caller();
        match Dims::square(self) {
            Ok(_) => inverse_with(&factor(self, "inverse", loc), "inverse", loc),
            Err(e) => fail(self.context(), self.shape(), e, "inverse", loc),
        }
    }

    /// The determinant of a square matrix (`[1, 1]`, or `[1, 1, batch]`), the product of the
    /// diagonal of `U` times the sign of the permutation, as `sign * exp(sum(ln |u_ii|))`: no
    /// overflow of the running product, but the result is `Inf` / 0 where the determinant is
    /// beyond `f32`; see [`Array::slogdet`] for those.
    ///
    /// # Errors
    ///
    /// The result is poisoned if the matrices are not square.
    #[track_caller]
    pub fn det(&self) -> Array<f32> {
        let loc = Location::caller();
        match Dims::square(self) {
            Ok(_) => {
                let (sign, logabs) = slogdet_of(&factor(self, "det", loc), "det", loc);
                &sign * logabs.exp()
            }
            Err(e) => fail(self.context(), Shape::from([1, 1, self.shape().dims()[2]]), e, "det", loc),
        }
    }

    /// The sign and the natural logarithm of the absolute value of the determinant, each `[1, 1]`
    /// (`[1, 1, batch]`): sign 0 and log `-Inf` for a singular matrix.
    ///
    /// # Errors
    ///
    /// As [`Array::det`].
    #[track_caller]
    pub fn slogdet(&self) -> (Array<f32>, Array<f32>) {
        let loc = Location::caller();
        match Dims::square(self) {
            Ok(_) => slogdet_of(&factor(self, "slogdet", loc), "slogdet", loc),
            Err(e) => {
                let shape = Shape::from([1, 1, self.shape().dims()[2]]);
                (fail(self.context(), shape, e.clone(), "slogdet", loc), fail(self.context(), shape, e, "slogdet", loc))
            }
        }
    }
}

impl Lu {
    /// The unit lower triangular factor `L`.
    pub fn lower(&self) -> Array<f32> {
        let [n, m, ..] = self.lu.shape().dims();
        let rows = self.lu.index(1);
        let cols = self.lu.index(0);
        let l = select(&cols.lt(&rows), self.lu.clone(), 0.0);
        select(&cols.eq(&rows), 1.0f32, l).slice(0, 0..n.min(m))
    }

    /// The upper triangular factor `U` (`[n, n]`: the first `n` rows).
    pub fn upper(&self) -> Array<f32> {
        let [n, ..] = self.lu.shape().dims();
        let rows = self.lu.index(1);
        let cols = self.lu.index(0);
        select(&cols.ge(&rows), self.lu.clone(), 0.0).slice(1, 0..n)
    }

    /// Solves `A X = B` with this factorisation of the square `A`, as [`Array::solve`].
    #[track_caller]
    pub fn solve(&self, b: &Array<f32>) -> Array<f32> {
        solve_with(self, b, "solve", Location::caller())
    }

    /// The inverse of the factored square matrix, as [`Array::inverse`].
    #[track_caller]
    pub fn inverse(&self) -> Array<f32> {
        inverse_with(self, "inverse", Location::caller())
    }

    /// The determinant of the factored square matrix, as [`Array::det`].
    #[track_caller]
    pub fn det(&self) -> Array<f32> {
        let (sign, logabs) = slogdet_of(self, "det", Location::caller());
        &sign * logabs.exp()
    }
}

/// `U⁻¹ L⁻¹ P b`.
fn solve_with(f: &Lu, b: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let (rhs, vector) = rhs_matrix(&f.lu, b);
    let x = solve_matrix(f, &rhs, op, loc);
    if vector { x.reshape(b.shape()) } else { x }
}

fn solve_matrix(f: &Lu, rhs: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let pb = gather_rows(rhs, &f.perm, loc);
    let y = lower_left(&f.lu, &pb, true, op, loc);
    lower_left(&f.lu.flip(0).flip(1), &y.flip(1), false, op, loc).flip(1)
}

fn inverse_with(f: &Lu, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let [n, _, batch, _] = f.lu.shape().dims();
    let shape = Shape::from([n, n, batch]);
    let (cols, rows) = match (Array::<u32>::range_on(f.lu.context(), shape, 0), Array::<u32>::range_on(f.lu.context(), shape, 1)) {
        (Ok(c), Ok(r)) => (c, r),
        (Err(e), _) | (_, Err(e)) => return fail(f.lu.context(), f.lu.shape(), e, op, loc),
    };
    let identity = select(&cols.eq(&rows), 1.0f32, 0.0f32);
    solve_matrix(f, &identity, op, loc)
}

/// Sign and `ln |det|` of every matrix of a factorisation, `[1, 1, batch]` each.
fn slogdet_of(f: &Lu, op: &'static str, loc: &'static Location<'static>) -> (Array<f32>, Array<f32>) {
    let ctx = f.lu.context();
    let [n, _, batch, _] = f.lu.shape().dims();
    let out = Shape::from([1, 1, batch]);
    let (idx, mat) = match (Array::<u32>::range_on(ctx, [n, 1, batch], 0), Array::<u32>::range_on(ctx, [n, 1, batch], 2)) {
        (Ok(i), Ok(m)) => (i, m),
        (Err(e), _) | (_, Err(e)) => return (fail(ctx, out, e.clone(), op, loc), fail(ctx, out, e, op, loc)),
    };
    // The diagonal of every matrix.
    let at = &(&idx * (n as u32 + 1)) + &(&mat * (n * n) as u32);
    let diag = f.lu.reshape(n * n * batch).take(&at).reshape([n, 1, batch]);
    let logabs = diag.abs().ln().sum_axis(0);
    let negatives = select(&diag.lt(0.0f32), 1u32, 0u32).sum_axis(0);
    // Inversions of the permutation: pairs `i < j` with `perm[i] > perm[j]`.
    let shape = Shape::from([n, n, batch]);
    let (cols, rows) = match (Array::<u32>::range_on(ctx, shape, 0), Array::<u32>::range_on(ctx, shape, 1)) {
        (Ok(c), Ok(r)) => (c, r),
        (Err(e), _) | (_, Err(e)) => return (fail(ctx, out, e.clone(), op, loc), fail(ctx, out, e, op, loc)),
    };
    let p_col = f.perm.broadcast_to(shape);
    let p_row = f.perm.reshape([1, n, batch]).broadcast_to(shape);
    let inverted = select(&cols.gt(&rows), select(&p_row.gt(&p_col), 1u32, 0u32), 0u32);
    let flips = &inverted.sum_axis(0).sum_axis(1) + &negatives;
    let odd = (&flips % 2u32).eq(1u32);
    let sign = select(&logabs.eq(f32::NEG_INFINITY), 0.0f32, select(&odd, -1.0f32, 1.0f32));
    (sign, logabs)
}
