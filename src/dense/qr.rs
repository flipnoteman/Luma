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
use crate::jit::comprehension::Props;
use crate::jit::structured::boundary::FactorKind;
use crate::jit::structured::families::factor::{boundary_of, qr_stage};
use crate::jit::structured::strategy::{first_candidate_in, note_in, StrategyId};
use crate::kernel::{uniform, view, KernelArg};
use crate::matmul::matmul_f32;
use crate::ops::select;
use crate::shape::Shape;
use crate::view::concat_slice;

/// Columns of a panel one workgroup factors.
pub(crate) const PANEL: usize = 32;

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

/// A factorisation under construction by the recursion: the packed factor with the unit lower trapezoid
/// `V` of its reflectors next to it (the updates of the columns to its right are products with it).
struct Node {
    qr: Array<f32>,
    v: Array<f32>,
    tau: Array<f32>,
    t: Array<f32>,
}

impl Node {
    fn failed(ctx: &crate::Context, a: &Array<f32>, e: LumaError, op: &'static str, loc: &'static Location<'static>) -> Node {
        let Qr { qr, tau, t } = failed_qr(ctx, a, e, op, loc);
        Node { v: qr.clone(), qr, tau, t }
    }

    fn of(q: Qr) -> Node {
        Node { v: unit_lower(&q.qr).cache(), qr: q.qr, tau: q.tau, t: q.t }
    }

    fn into_qr(self) -> Qr {
        Qr { qr: self.qr, tau: self.tau, t: self.t }
    }
}

/// One panel of up to [`PANEL`] columns of a single matrix by [`super::tsqr`].
fn tsqr_panel(a: &Array<f32>, plan: super::tsqr::Plan, op: &'static str, loc: &'static Location<'static>) -> Node {
    match super::tsqr::panel(a, plan) {
        Ok([qr, v, tau, t]) => Node { qr, v, tau, t },
        Err(e) => Node::failed(a.context(), a, e, op, loc),
    }
}

/// The unit lower trapezoid `V` of a packed `[w, m]` factor (ones on the diagonal, zeros above).
pub(crate) fn unit_lower(f: &Array<f32>) -> Array<f32> {
    let rows = f.index(1);
    let cols = f.index(0);
    let below = select(&cols.lt(&rows), f.clone(), 0.0f32);
    select(&cols.eq(&rows), 1.0f32, below)
}

/// The packed factor and `V` of a node from those of its halves: columns `0..n1` are the left
/// half's, the right half's rows `0..n1` are the updated `c` (the rows of `R` above its own
/// factor), the rest its own factor (`f2`, `v2`, from row `n1` on); `V` is zero above its own.
const JOIN: &str = "struct Params { m: u32, w: u32, n1: u32 }
@group(1) @binding(0) var<storage, read> f1: array<f32>;
@group(1) @binding(1) var<uniform> f1v: LumaView;
@group(1) @binding(2) var<storage, read> v1: array<f32>;
@group(1) @binding(3) var<uniform> v1v: LumaView;
@group(1) @binding(4) var<storage, read> c: array<f32>;
@group(1) @binding(5) var<uniform> cv: LumaView;
@group(1) @binding(6) var<storage, read> f2: array<f32>;
@group(1) @binding(7) var<uniform> f2v: LumaView;
@group(1) @binding(8) var<storage, read> v2: array<f32>;
@group(1) @binding(9) var<uniform> v2v: LumaView;
@group(1) @binding(10) var<storage, read_write> fo: array<f32>;
@group(1) @binding(11) var<storage, read_write> vo: array<f32>;
@group(1) @binding(12) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = p.w;
    let n1 = p.n1;
    let total = p.m * w;
    for (var idx = gid.x; idx < total; idx = idx + nwg.x * 256u) {
        let i = idx / w;
        let col = idx - i * w;
        if (col < n1) {
            fo[idx] = f1[luma_view_idx3(f1v, col, i, 0u)];
            vo[idx] = v1[luma_view_idx3(v1v, col, i, 0u)];
        } else if (i < n1) {
            fo[idx] = c[luma_view_idx3(cv, col - n1, i, 0u)];
            vo[idx] = 0.0;
        } else {
            fo[idx] = f2[luma_view_idx3(f2v, col - n1, i - n1, 0u)];
            vo[idx] = v2[luma_view_idx3(v2v, col - n1, i - n1, 0u)];
        }
    }
}
";

fn join(ctx: &crate::Context, parts: [&Array<f32>; 5], shape: Shape, n1: usize) -> Result<(Array<f32>, Array<f32>), LumaError> {
    let [w, m, ..] = shape.dims();
    let (mut fo, mut vo) = (Array::<f32>::alloc(ctx, shape)?, Array::<f32>::alloc(ctx, shape)?);
    // (`batch` carries `n1`: the kernel's struct has the same three words.)
    let params = Params { m: m as u32, w: w as u32, batch: n1 as u32 };
    let args: [KernelArg<'_>; 8] = [
        view(parts[0]),
        view(parts[1]),
        view(parts[2]),
        view(parts[3]),
        view(parts[4]),
        (&mut fo).into(),
        (&mut vo).into(),
        uniform(&params),
    ];
    let groups = (m * w).div_ceil(256).min(4096) as u32;
    launch(ctx, "qr_join", "qr_join", || JOIN.to_owned(), [groups, 1, 1], &args)?;
    Ok((fo, vo))
}

/// `T = [[T1, -T12], [0, T2]]` and `tau = [tau1, tau2]` of a node, one element per invocation.
const TJOIN: &str = "struct Params { w: u32, n1: u32 }
@group(1) @binding(0) var<storage, read> t1: array<f32>;
@group(1) @binding(1) var<uniform> t1v: LumaView;
@group(1) @binding(2) var<storage, read> t12: array<f32>;
@group(1) @binding(3) var<uniform> t12v: LumaView;
@group(1) @binding(4) var<storage, read> t2: array<f32>;
@group(1) @binding(5) var<uniform> t2v: LumaView;
@group(1) @binding(6) var<storage, read> u1: array<f32>;
@group(1) @binding(7) var<uniform> u1v: LumaView;
@group(1) @binding(8) var<storage, read> u2: array<f32>;
@group(1) @binding(9) var<uniform> u2v: LumaView;
@group(1) @binding(10) var<storage, read_write> to: array<f32>;
@group(1) @binding(11) var<storage, read_write> uo: array<f32>;
@group(1) @binding(12) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = p.w;
    let n1 = p.n1;
    for (var idx = gid.x; idx < w * w; idx = idx + nwg.x * 256u) {
        let i = idx / w;
        let col = idx - i * w;
        var v = 0.0;
        if (i < n1) {
            if (col < n1) { v = t1[luma_view_idx3(t1v, col, i, 0u)]; } else { v = -t12[luma_view_idx3(t12v, col - n1, i, 0u)]; }
        } else if (col >= n1) {
            v = t2[luma_view_idx3(t2v, col - n1, i - n1, 0u)];
        }
        to[idx] = v;
        if (idx < w) {
            if (idx < n1) { uo[idx] = u1[luma_view_idx3(u1v, idx, 0u, 0u)]; } else { uo[idx] = u2[luma_view_idx3(u2v, idx - n1, 0u, 0u)]; }
        }
    }
}
";

/// `T` and `tau` of a node from those of its halves and `T12 = T1 Z T2` (with a minus sign).
fn tjoin(ctx: &crate::Context, [t1, t12, t2, tau1, tau2]: [&Array<f32>; 5], n1: usize, w: usize) -> Result<(Array<f32>, Array<f32>), LumaError> {
    let (mut to, mut uo) = (Array::<f32>::alloc(ctx, Shape::from([w, w, 1]))?, Array::<f32>::alloc(ctx, Shape::from([w, 1, 1]))?);
    let params = TParams { w: w as u32, n1: n1 as u32 };
    let args: [KernelArg<'_>; 8] = [view(t1), view(t12), view(t2), view(tau1), view(tau2), (&mut to).into(), (&mut uo).into(), uniform(&params)];
    let groups = (w * w).div_ceil(256).min(4096) as u32;
    launch(ctx, "qr_tjoin", "qr_tjoin", || TJOIN.to_owned(), [groups, 1, 1], &args)?;
    Ok((to, uo))
}

#[derive(encase::ShaderType)]
struct TParams {
    w: u32,
    n1: u32,
}

/// The strategy of [`qr_stage`] that factors the `d` matrices of `a`, and the TSQR plan if that is the panel. The `QrPanel` decision
/// stays where it was made: a single matrix of at most [`PANEL`] columns says which panel it took.
fn pick(a: &Array<f32>, d: Dims, op: &'static str, loc: &'static Location<'static>) -> (StrategyId, Option<super::tsqr::Plan>) {
    let ctx = a.context();
    let boundary = boundary_of(FactorKind::Qr, op, loc, d, 0);
    let (id, params) = first_candidate_in(qr_stage(), ctx, &boundary).expect("a QR always fits (the recursion at the latest)");
    note_in(ctx, &boundary, qr_stage(), id, &params, &[]);
    let plan = (id == StrategyId::QR_TSQR_PANEL).then(|| super::tsqr::choose(ctx, d.rows, d.cols).expect("the TSQR panel fit this panel"));
    if d.batch == 1 && d.cols <= PANEL {
        super::tsqr::note(ctx, plan, op, loc);
    }
    (id, plan)
}

/// A recursion over one `[w, m]` matrix, `m >= w`.
fn recurse(a: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Node {
    let [w, m, ..] = a.shape().dims();
    let d = Dims { rows: m, cols: w, batch: 1 };
    let (id, plan) = pick(a, d, op, loc);
    if let Some(plan) = plan {
        return tsqr_panel(a, plan, op, loc);
    }
    if id == StrategyId::QR_LEAF {
        return Node::of(leaf(a, d, op, loc));
    }
    let n1 = super::split_cols(w);
    let n2 = w - n1;
    let Node { qr: f1, v: v1, tau: tau1, t: t1 } = recurse(&block(a, 0..m, 0..n1), op, loc);
    let c = block(a, 0..m, n1..w).cache();
    // C <- Q1ᵀ C = C - V1 (T1ᵀ (V1ᵀ C)).
    let wmat = matmul_f32(&v1.transpose(), &c);
    let wmat = matmul_f32(&t1.transpose(), &wmat);
    let c = (&c - &matmul_f32(&v1, &wmat)).cache();
    let Node { qr: f2, v: v2, tau: tau2, t: t2 } = recurse(&block(&c, n1..m, 0..n2), op, loc);
    let v1b = block(&v1, n1..m, 0..n1);
    // T12 = -T1 (V1bᵀ V2) T2.
    let z = matmul_f32(&v1b.transpose(), &v2);
    let t12 = matmul_f32(&t1, &matmul_f32(&z, &t2));
    let joined = tjoin(a.context(), [&t1, &t12, &t2, &tau1, &tau2], n1, w).and_then(|(t, tau)| join(a.context(), [&f1, &v1, &c, &f2, &v2], a.shape(), n1).map(|(qr, v)| Node { qr, v, tau, t }));
    joined.unwrap_or_else(|e| Node::failed(a.context(), a, e, op, loc))
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
    let (id, plan) = pick(a, d, op, loc);
    if let Some(plan) = plan {
        return tsqr_panel(a, plan, op, loc).into_qr();
    }
    if id == StrategyId::QR_LEAF {
        return leaf(a, d, op, loc);
    }
    if d.batch > 1 {
        let parts: Vec<Qr> = (0..d.batch).map(|i| recurse(&nth(a, i), op, loc).into_qr()).collect();
        let pick = |f: fn(&Qr) -> &Array<f32>| -> Vec<Array<f32>> { parts.iter().map(|p| f(p).clone()).collect() };
        return Qr { qr: stack_batch(&pick(|p| &p.qr)), tau: concat_slice(2, &pick(|p| &p.tau)), t: concat_slice(2, &pick(|p| &p.t)) };
    }
    recurse(a, op, loc).into_qr()
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
        select(&cols.ge(&rows), self.qr.clone(), 0.0f32).slice(1, 0..n).with_props(Props::triangle(false, false, n, n))
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
