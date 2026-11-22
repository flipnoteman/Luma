//! Householder QR in compact WY form, least squares.
//!
//! `A = Q R` for `A` of `m` rows and `n <= m` columns: `R` is upper triangular (`n x n`) and `Q`
//! is the product of `n` Householder reflections `H_j = I - tau_j v_j v_jᵀ` (`v_j` has a unit
//! entry at row `j`, zeros above it and its other entries stored below the diagonal of the
//! packed factor). The reflections are also kept as the blocked product `Q = I - V T Vᵀ` with
//! an upper triangular `T`, which turns applying `Q` into three matrix products.
//!
//! Panels of up to 32 columns (a batch of such matrices in one launch) are factored by one
//! workgroup per matrix, which also builds their `T`; wider matrices recursively on the columns
//! (Elmroth-Gustavson): factor the left half, apply its reflections to the right half with
//! [`matmul_f32`]s, factor the trailing part, merge the two `T`s.

use std::panic::Location;

use super::tri::lower_left;
use super::lu::UNROLL;
use super::{block, fail, fence, launch, nth, per_matrix, stack_batch, Dims};
use crate::array::Array;
use crate::bmm::matmul_batched;
use crate::error::LumaError;
use crate::kernel::{uniform, view, KernelArg};
use crate::matmul::matmul_f32;
use crate::ops::select;
use crate::shape::Shape;
use crate::view::{concat, concat_slice};

/// Columns of a panel one workgroup factors.
const PANEL: usize = 32;

/// A packed QR factorisation, see [`Array::qr`].
#[derive(Clone, Debug)]
pub struct Qr {
    /// `R` on and above the diagonal, the Householder vectors (without their unit entry) below it;
    /// `[n, m]` or `[n, m, batch]`.
    pub qr: Array<f32>,
    /// The scalars `tau_j` of the reflections, `[n, 1]` or `[n, 1, batch]`.
    pub tau: Array<f32>,
    /// The triangular factor of `Q = I - V T Vᵀ`, `[n, n]` or `[n, n, batch]` (upper triangular).
    pub t: Array<f32>,
}

#[derive(encase::ShaderType)]
struct Params {
    m: u32,
    w: u32,
    batch: u32,
}

/// One workgroup per matrix of up to [`PANEL`] columns, in place in global memory. Invocation
/// `(g, c) = (t / 32, t % 32)` owns column `c` of rows `g, g + G, ...`, so a warp reads a row
/// segment. Per column `j`: the norm below the diagonal by a tree reduction; the reflector
/// (LAPACK's `dlarfg`); the products `v_jᵀ F[:, c]` for every other column `c` (those with `c > j`
/// are what the reflection needs, those with `c < j` the entries of the new column of `T`); the
/// update; the scaling of `v_j`. The row loops take [`UNROLL`] rows a step with their loads issued
/// first (the kernel is bound by the latency of its global loads), see [`super::lu`].
fn source(threads: usize, fence: &str) -> String {
    let groups = threads / 32;
    format!(
        "struct Params {{ m: u32, w: u32, batch: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read_write> f: array<f32>;
@group(1) @binding(3) var<storage, read_write> tau: array<f32>;
@group(1) @binding(4) var<storage, read_write> tf: array<f32>;
@group(1) @binding(5) var<uniform> p: Params;
fn ld(mi: u32, i: u32, c: u32) -> f32 {{ return a[luma_view_idx3(av, c, i, mi)]; }}

var<workgroup> red: array<f32, {threads}>;
var<workgroup> part: array<f32, {parts}>;
var<workgroup> dd: array<f32, 32>;
var<workgroup> tm: array<f32, 1024>;

@compute @workgroup_size({threads})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let mi = wid.x + wid.y * nwg.x;
    let m = select(0u, p.m, mi < p.batch);
    let w = p.w;
    let base = mi * p.m * w;
    let g = t / 32u;
    let c = t % 32u;
    for (var idx = t; idx < m * w; idx = idx + {threads}u) {{
        let i = idx / w;
        f[base + idx] = ld(mi, i, idx - i * w);
    }}
    for (var idx = t; idx < 1024u; idx = idx + {threads}u) {{
        tm[idx] = 0.0;
    }}
    {fence}
    workgroupBarrier();
    for (var j = 0u; j < min(m, w); j = j + 1u) {{
        var sq = 0.0;
        var k = j + 1u + t;
        for (; k + {tlast}u < m; k = k + {tstep}u) {{
{norm_loads}{norm_fmas}        }}
        for (; k < m; k = k + {t1}u) {{
            let v = f[base + k * w + j];
            sq = fma(v, v, sq);
        }}
        red[t] = sq;
        workgroupBarrier();
        for (var h = {half}u; h > 0u; h = h >> 1u) {{
            if (t < h) {{ red[t] = red[t] + red[t + h]; }}
            workgroupBarrier();
        }}
        let xn2 = red[0];
        let alpha = f[base + j * w + j];
        var beta = alpha;
        var tj = 0.0;
        var inv = 0.0;
        if (xn2 != 0.0) {{
            let nrm = sqrt(fma(alpha, alpha, xn2));
            beta = select(nrm, -nrm, alpha >= 0.0);
            tj = (beta - alpha) / beta;
            inv = 1.0 / (alpha - beta);
        }}
        if (c < w && c != j) {{
            var acc = 0.0;
            if (g == 0u) {{ acc = f[base + j * w + c]; }}
            var i = j + 1u + g;
            for (; i + {glast}u < m; i = i + {gstep}u) {{
                let r0 = base + i * w;
{dot_loads}{dot_fmas}            }}
            for (; i < m; i = i + {g1}u) {{
                acc = fma(f[base + i * w + j] * inv, f[base + i * w + c], acc);
            }}
            part[g * 32u + c] = acc;
        }}
        workgroupBarrier();
        if (t < 32u) {{
            var s = 0.0;
            for (var q = 0u; q < {groups}u; q = q + 1u) {{ s = s + part[q * 32u + t]; }}
            dd[t] = s;
        }}
        workgroupBarrier();
        if (t < j) {{
            var s = 0.0;
            for (var q = t; q < j; q = q + 1u) {{ s = fma(tm[t * 32u + q], dd[q], s); }}
            tm[t * 32u + j] = -tj * s;
        }}
        if (t == 0u) {{
            tm[j * 32u + j] = tj;
            tau[mi * w + j] = tj;
        }}
        if (c < w && c > j) {{
            let wc = tj * dd[c];
            if (g == 0u) {{ f[base + j * w + c] = f[base + j * w + c] - wc; }}
            var i = j + 1u + g;
            for (; i + {glast}u < m; i = i + {gstep}u) {{
                let r0 = base + i * w;
{upd_loads}{upd_stores}            }}
            for (; i < m; i = i + {g1}u) {{
                f[base + i * w + c] = fma(-(f[base + i * w + j] * inv), wc, f[base + i * w + c]);
            }}
        }}
        {fence}
        workgroupBarrier();
        for (var i = j + 1u + t; i < m; i = i + {t1}u) {{
            f[base + i * w + j] = f[base + i * w + j] * inv;
        }}
        if (t == 0u) {{ f[base + j * w + j] = beta; }}
        {fence}
        workgroupBarrier();
    }}
    for (var idx = t; idx < w * w; idx = idx + {t1}u) {{
        let i = idx / w;
        tf[mi * w * w + idx] = tm[i * 32u + idx - i * w];
    }}
}}
",
        parts = groups * 32,
        half = threads / 2,
        t1 = threads,
        tlast = (UNROLL - 1) * threads,
        tstep = UNROLL * threads,
        g1 = groups,
        glast = (UNROLL - 1) * groups,
        gstep = UNROLL * groups,
        norm_loads = (0..UNROLL).map(|k| format!("            let v{k} = f[base + (k + {}u) * w + j];\n", k * threads)).collect::<String>(),
        norm_fmas = (0..UNROLL).map(|k| format!("            sq = fma(v{k}, v{k}, sq);\n")).collect::<String>(),
        dot_loads = (0..UNROLL).map(|k| format!("                let u{k} = f[r0 + {}u * w + j];\n                let y{k} = f[r0 + {}u * w + c];\n", k * groups, k * groups)).collect::<String>(),
        dot_fmas = (0..UNROLL).map(|k| format!("                acc = fma(u{k} * inv, y{k}, acc);\n")).collect::<String>(),
        upd_loads = (0..UNROLL).map(|k| format!("                let u{k} = f[r0 + {}u * w + j] * inv;\n                let y{k} = f[r0 + {}u * w + c];\n", k * groups, k * groups)).collect::<String>(),
        upd_stores = (0..UNROLL).map(|k| format!("                f[r0 + {}u * w + c] = fma(-u{k}, wc, y{k});\n", k * groups)).collect::<String>(),
    )
}

fn failed_qr(ctx: &crate::Context, a: &Array<f32>, e: LumaError, op: &'static str, loc: &'static Location<'static>) -> Qr {
    let [n, _, batch, _] = a.shape().dims();
    let tau = fail(ctx, Shape::from([n, 1, batch]), e.clone(), op, loc);
    let t = fail(ctx, Shape::from([n, n, batch]), e.clone(), op, loc);
    Qr { qr: fail(ctx, a.shape(), e, op, loc), tau, t }
}

/// One launch over a batch of matrices of up to [`PANEL`] columns.
fn leaf(a: &Array<f32>, d: Dims, op: &'static str, loc: &'static Location<'static>) -> Qr {
    let ctx = a.context();
    let n = d.cols;
    let (mut f, mut tau, mut t) = match (
        Array::<f32>::alloc(ctx, a.shape()),
        Array::<f32>::alloc(ctx, Shape::from([n, 1, d.batch])),
        Array::<f32>::alloc(ctx, Shape::from([n, n, d.batch])),
    ) {
        (Ok(f), Ok(tau), Ok(t)) => (f, tau, t),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return failed_qr(ctx, a, e, op, loc),
    };
    let params = Params { m: d.rows as u32, w: n as u32, batch: d.batch as u32 };
    let args: [KernelArg<'_>; 5] = [view(a), (&mut f).into(), (&mut tau).into(), (&mut t).into(), uniform(&params)];
    let threads = super::lu::panel_threads(ctx, d.rows);
    match launch(ctx, &format!("qr_panel/{threads}/{}", fence(ctx).len()), "qr_panel", || source(threads, fence(ctx)), per_matrix(ctx, d.batch), &args) {
        Ok(()) => Qr { qr: f, tau, t },
        Err(e) => failed_qr(ctx, a, e, op, loc),
    }
}

/// The unit lower trapezoid `V` of a packed `[w, m]` factor (ones on the diagonal, zeros above).
pub(crate) fn unit_lower(f: &Array<f32>) -> Array<f32> {
    let rows = f.index(1);
    let cols = f.index(0);
    let below = select(&cols.lt(&rows), f.clone(), 0.0f32);
    select(&cols.eq(&rows), 1.0f32, below)
}

/// A recursion over one `[w, m]` matrix, `m >= w`.
fn recurse(a: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Qr {
    let [w, m, ..] = a.shape().dims();
    if w <= if m <= 64 { PANEL } else { super::panel_width() } {
        return leaf(a, Dims { rows: m, cols: w, batch: 1 }, op, loc);
    }
    let n1 = super::split_cols(w);
    let n2 = w - n1;
    let Qr { qr: f1, tau: tau1, t: t1 } = recurse(&block(a, 0..m, 0..n1), op, loc);
    let f1 = f1.cache();
    let v1 = unit_lower(&f1).cache();
    let c = block(a, 0..m, n1..w).cache();
    // C <- Q1ᵀ C = C - V1 (T1ᵀ (V1ᵀ C)).
    let wmat = matmul_f32(&v1.transpose(), &c);
    let wmat = matmul_f32(&t1.transpose(), &wmat);
    let c = (&c - &matmul_f32(&v1, &wmat)).cache();
    let Qr { qr: f2, tau: tau2, t: t2 } = recurse(&block(&c, n1..m, 0..n2), op, loc);
    let v2 = unit_lower(&f2).cache();
    let v1b = block(&v1, n1..m, 0..n1).cache();
    // T12 = -T1 (V1bᵀ V2) T2.
    let z = matmul_f32(&v1b.transpose(), &v2);
    let t12 = matmul_f32(&t1, &matmul_f32(&z, &t2)) * -1.0f32;
    let zeros = match Array::<f32>::zeros_on(a.context(), [n1, n2]) {
        Ok(z) => z,
        Err(e) => return failed_qr(a.context(), a, e, op, loc),
    };
    let t = concat(1, [&concat(0, [&t1, &t12]), &concat(0, [&zeros, &t2])]);
    let right = concat(1, [&block(&c, 0..n1, 0..n2), &f2]);
    Qr { qr: concat(0, [&f1, &right]), tau: concat(0, [&tau1, &tau2]), t }
}

fn factor(a: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Qr {
    let ctx = a.context();
    let d = match Dims::of(a) {
        Ok(d) if d.rows >= d.cols => d,
        Ok(_) => {
            let e = LumaError::InvalidShape(format!("qr needs at least as many rows as columns, got shape {}", a.shape()));
            return failed_qr(ctx, a, e, op, loc);
        }
        Err(e) => return failed_qr(ctx, a, e, op, loc),
    };
    if d.cols <= if d.batch > 1 || d.rows <= 64 { PANEL } else { super::panel_width() } {
        return leaf(a, d, op, loc);
    }
    if d.batch > 1 {
        let parts: Vec<Qr> = (0..d.batch).map(|i| recurse(&nth(a, i), op, loc)).collect();
        let pick = |f: fn(&Qr) -> &Array<f32>| -> Vec<Array<f32>> { parts.iter().map(|p| f(p).clone()).collect() };
        return Qr { qr: stack_batch(&pick(|p| &p.qr)), tau: concat_slice(2, &pick(|p| &p.tau)), t: concat_slice(2, &pick(|p| &p.t)) };
    }
    recurse(a, op, loc)
}

/// `a b` for matrices or batches of them.
fn mm(a: &Array<f32>, b: &Array<f32>, batched: bool) -> Array<f32> {
    if batched { matmul_batched(a, b) } else { matmul_f32(a, b) }
}

impl Array<f32> {
    /// Householder QR of a matrix (`[n, m]` with `m >= n`, or a batch `[n, m, batch]`), see
    /// [`Qr`]. A rank-deficient matrix factors normally (the zero shows on the diagonal of `R`).
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// let a = Array::from_slice(&[3.0f32, 0.0, 4.0, 0.0], [2, 2])?; // [[3, 0], [4, 0]]
    /// let f = a.qr();
    /// let r = f.r().to_vec()?;
    /// assert!((r[0].abs() - 5.0).abs() < 1e-6);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The factors are poisoned if there are more columns than rows.
    #[track_caller]
    pub fn qr(&self) -> Qr {
        factor(self, "qr", Location::caller())
    }

    /// The least-squares solution of `A X = B` for `A` with `m >= n` rows by QR:
    /// `X = R⁻¹ (Qᵀ B)[..n]`, `b` is `[k, m]` (`[k, m, batch]`), the result `[k, n]`. Full column
    /// rank is assumed (a zero on the diagonal of `R` gives `Inf` / `NaN`; use the SVD based
    /// [`Array::pinv`] otherwise).
    ///
    /// # Errors
    ///
    /// The result is poisoned if `A` has more columns than rows or `b` does not have `m` rows.
    #[track_caller]
    pub fn lstsq(&self, b: &Array<f32>) -> Array<f32> {
        let loc = Location::caller();
        let f = factor(self, "lstsq", loc);
        f.lstsq_with(b, "lstsq", loc)
    }
}

/// `A X = B` for a wide `A` (`m < n` rows): the minimum norm solution `X = Q R⁻ᵀ B` from the QR
/// factorisation `Aᵀ = Q R`.
pub(crate) fn min_norm(a: &Array<f32>, b: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let [n, m, batch, _] = a.shape().dims();
    let ctx = a.context();
    let [k, brows, bbatch, _] = b.shape().dims();
    if brows != m || bbatch != batch {
        let e = LumaError::ShapeMismatch { lhs: a.shape().dims(), rhs: b.shape().dims() };
        return fail(ctx, Shape::from([k, n, batch]), e, op, loc);
    }
    let f = factor(&a.transpose(), op, loc);
    let y = f.r().solve_triangular(b, crate::Triangle::Upper, true, false);
    mm(&f.q_thin(), &y, batch > 1)
}

impl Qr {
    fn dims(&self) -> Option<Dims> {
        Dims::of(&self.qr).ok()
    }

    /// `Qᵀ B` for `B` of `[k, m]` (`[k, m, batch]`): `B - V (Tᵀ (Vᵀ B))`.
    pub fn apply_qt(&self, b: &Array<f32>) -> Array<f32> {
        self.apply(b, true)
    }

    /// `Q B` for `B` of `[k, m]` (`[k, m, batch]`): `B - V (T (Vᵀ B))`, with the full `m x m` `Q`.
    pub fn apply_q(&self, b: &Array<f32>) -> Array<f32> {
        self.apply(b, false)
    }

    fn apply(&self, b: &Array<f32>, transposed: bool) -> Array<f32> {
        let batched = self.dims().is_some_and(|d| d.batch > 1);
        let v = unit_lower(&self.qr).cache();
        let t = if transposed { self.t.transpose() } else { self.t.clone() };
        let b = b.cache();
        let w = mm(&v.transpose(), &b, batched);
        let w = mm(&t, &w, batched);
        &b - &mm(&v, &w, batched)
    }

    /// The upper triangular factor `R`, `[n, n]` (`[n, n, batch]`).
    pub fn r(&self) -> Array<f32> {
        let [n, ..] = self.qr.shape().dims();
        let rows = self.qr.index(1);
        let cols = self.qr.index(0);
        select(&cols.ge(&rows), self.qr.clone(), 0.0f32).slice(1, 0..n)
    }

    /// The first `n` columns of `Q` (the thin factor, `[n, m]` or `[n, m, batch]`: `m x n` with
    /// orthonormal columns), `A = Q_thin R`.
    pub fn q_thin(&self) -> Array<f32> {
        let [n, m, batch, _] = self.qr.shape().dims();
        self.apply_q(&identity(self.qr.context(), n, m, batch))
    }

    /// The full `m x m` orthogonal `Q` (`[m, m]` or `[m, m, batch]`).
    pub fn q(&self) -> Array<f32> {
        let [_, m, batch, _] = self.qr.shape().dims();
        self.apply_q(&identity(self.qr.context(), m, m, batch))
    }

    /// The least-squares solution of `A X = B` with this factorisation of `A`, see [`Array::lstsq`].
    #[track_caller]
    pub fn lstsq(&self, b: &Array<f32>) -> Array<f32> {
        self.lstsq_with(b, "lstsq", Location::caller())
    }

    fn lstsq_with(&self, b: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
        let [n, m, batch, _] = self.qr.shape().dims();
        let [k, brows, bbatch, _] = b.shape().dims();
        let ctx = self.qr.context();
        if brows != m || bbatch != batch {
            let e = LumaError::ShapeMismatch { lhs: self.qr.shape().dims(), rhs: b.shape().dims() };
            return fail(ctx, Shape::from([k, n, batch]), e, op, loc);
        }
        let qtb = self.apply_qt(b).slice(1, 0..n);
        let r = self.qr.slice(1, 0..n);
        lower_left(&r.flip(0).flip(1), &qtb.flip(1), false, op, loc).flip(1)
    }
}

/// The `[cols, rows, batch]` matrix with ones on the diagonal.
fn identity(ctx: &crate::Context, cols: usize, rows: usize, batch: usize) -> Array<f32> {
    let shape = Shape::from([cols, rows, batch]);
    match (Array::<u32>::range_on(ctx, shape, 0), Array::<u32>::range_on(ctx, shape, 1)) {
        (Ok(c), Ok(r)) => select(&c.eq(&r), 1.0f32, 0.0f32),
        (Err(e), _) | (_, Err(e)) => fail(ctx, shape, e, "qr", Location::caller()),
    }
}
