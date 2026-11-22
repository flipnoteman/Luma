//! Singular value decomposition, pseudo-inverse and rank, and the engine behind the symmetric
//! eigenproblem ([`eigh`](super::eigh)): the one-sided (Hestenes) Jacobi method.
//!
//! `A = U Σ Vᵀ` is found by rotating pairs of columns of `A` until all of them are orthogonal:
//! `A V = U Σ` with `V` the product of the rotations, the singular values the norms of the
//! columns and `U` the normalised columns. Jacobi is slower than bidiagonalisation in flops but
//! it is built from independent steps (the pairs of a round-robin round are disjoint) and needs
//! no reflectors or divide and conquer. It is accurate in the normwise sense (`||A - U S Vᵀ||` and
//! the singular values within a few `eps ||A||`), and for matrices whose columns are badly scaled
//! it computes the small singular values to high *relative* accuracy.
//!
//! A matrix with fewer rows than columns is decomposed through its transpose. Then:
//!
//! * a matrix (and every matrix of a batch) that fits in workgroup memory with its `V` (up to
//!   64 x 64 where the device has 32 KiB, 32 x 32 with 16 KiB) is decomposed by *one workgroup*:
//!   columns and `V` live in workgroup memory, the pairs of a round-robin round are rotated in
//!   parallel (a few invocations per pair split the dot products), and the sweeps stop inside
//!   the kernel when no rotation is left (at most [`MAX_SWEEPS`]). A whole batch is one launch.
//! * a bigger matrix lives in global memory, columns contiguous. From [`BLOCK_MIN`] columns on it
//!   is decomposed by block Jacobi (see [`block`]: a Gram matrix and a small eigenproblem per
//!   block pair, then two thin products); below that every round-robin round is one launch of a
//!   workgroup per column pair. Either way the sweeps run until one of them rotates nothing:
//!   a flag in global memory records it, the rounds after convergence return at once, and the
//!   host looks at the flags every few sweeps (so a big decomposition waits for the device).
//!
//! The rank of a big matrix does not need the singular values when two QR factorisations bound it
//! from both sides (the `rank` module); it takes them when they do not.
//!
//! The matrix is scaled by a power of two (small kernel) or by its largest element (big matrix) so
//! that the squares of the column norms neither overflow nor underflow, whatever the scale of
//! the input.

mod block;
mod rank;

use std::panic::Location;

use super::{fail, launch, nth, per_matrix, stack_batch, Dims};
use crate::array::{Array, Failure};
use crate::context::Context;
use crate::error::LumaError;
use crate::kernel::{uniform, KernelArg};
use crate::matmul::matmul_f32;
use crate::ops::select;
use crate::shape::Shape;
use crate::view::concat;

/// Most sweeps the workgroup kernel runs; it stops earlier when no pair is rotated.
const MAX_SWEEPS: u32 = 30;

/// Which decomposition a kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The SVD of the matrix as given: singular values descending.
    Svd,
    /// The eigendecomposition of the symmetric matrix whose lower triangle is given, as the
    /// SVD of that matrix shifted by `||A||_F I` (positive semi-definite, so `U = V`): the
    /// values are the singular values minus the shift, ascending.
    Eigh,
}

/// The arrays a decomposition produces; `u` and `vt` are `None` for values only.
pub(crate) struct Outputs {
    /// Singular values (descending) or eigenvalues (ascending), `[k, 1, batch]`.
    pub(crate) s: Array<f32>,
    /// Left vectors `[k, m, batch]` (SVD only).
    pub(crate) u: Option<Array<f32>>,
    /// Right vectors transposed `[n, k, batch]` (SVD), or eigenvectors as columns `[n, n, batch]`.
    pub(crate) vt: Option<Array<f32>>,
}

/// The convergence threshold of a pair: the columns are orthogonal enough when
/// `|a_p . a_q| <= tol |a_p| |a_q|`. The dot products of `rows` terms carry rounding errors of
/// about `eps sqrt(rows)`, which is where the tolerance has to sit to be reachable.
fn tolerance(rows: usize) -> f32 {
    0.25 * 1.2e-7 * (rows as f32).sqrt().max(1.0)
}

#[derive(encase::ShaderType)]
struct SmallParams {
    r: u32,
    c: u32,
    batch: u32,
    wide: u32,
    tol: f32,
}

/// The shape of a workgroup-memory kernel: the largest matrix it takes and its thread layout.
#[derive(Clone, Copy, Debug)]
struct Plan {
    kind: Kind,
    vectors: bool,
    /// Rows and columns of the largest matrix the compiled kernel takes (`rmax >= cmax`).
    rmax: usize,
    cmax: usize,
    /// Invocations per workgroup and per pair.
    t: usize,
    g: usize,
}

impl Plan {
    /// Words of workgroup memory the kernel declares.
    fn words(&self) -> usize {
        let pa = self.rmax | 1;
        let pv = self.cmax | 1;
        let pm = self.cmax / 2 + 1;
        self.cmax * pa + if self.vectors { self.cmax * pv } else { 0 } + 3 * self.t + 4 * self.cmax + 2 * pm + self.rmax + 8
    }
}

fn bucket(n: usize) -> usize {
    const STEPS: [usize; 16] = [2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96];
    STEPS.iter().copied().find(|&s| s >= n).unwrap_or(n)
}

/// The workgroup-memory kernel for `r x c` matrices (`r >= c`), if the device has room.
fn plan(ctx: &Context, kind: Kind, vectors: bool, r: usize, c: usize) -> Option<Plan> {
    if c > 96 {
        return None;
    }
    let (rmax, cmax) = (bucket(r), bucket(c));
    let pairs = cmax.div_ceil(2);
    let mut g = 1;
    while g < 8 && g * 2 <= rmax / 4 && pairs * g * 2 <= 128 {
        g *= 2;
    }
    let t = (pairs * g).next_multiple_of(32);
    let plan = Plan { kind, vectors, rmax, cmax, t, g };
    (plan.words() * 4 <= ctx.info().limits.max_workgroup_storage_bytes as usize).then_some(plan)
}

/// The WGSL of a column rotation, shared by the small and the big kernels.
const ROTATION: &str = r#"
fn rotation(app: f32, aqq: f32, apq: f32) -> vec2<f32> {
    let zeta = (aqq - app) / (2.0 * apq);
    let t = select(-1.0, 1.0, zeta >= 0.0) / (abs(zeta) + sqrt(fma(zeta, zeta, 1.0)));
    let c0 = inverseSqrt(fma(t, t, 1.0));
    let s0 = c0 * t;
    let r = fma(s0, s0, fma(c0, c0, -1.0));
    return vec2<f32>(fma(-0.5 * r, c0, c0), fma(-0.5 * r, s0, s0));
}
fn isnan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}
fn key(x: f32) -> f32 {
    return select(x, bitcast<f32>(0x7f800000u), isnan(x));
}
"#;

/// The kernel that decomposes a matrix per workgroup in workgroup memory.
///
/// `X` is the tall matrix (`R >= C`) the algorithm works on: `A` itself, or `Aᵀ` if `A` is wide.
/// Its columns are rotated in `ac`, `V` in `vc`; the singular values are sorted by rank
/// counting (`NaN` counts as the largest), the left vectors normalised, the left vectors of null
/// columns completed to an orthonormal set by Gram-Schmidt against the others.
fn small_source(plan: &Plan) -> String {
    let Plan { kind, vectors, rmax, cmax, t, g } = *plan;
    let eigh = kind == Kind::Eigh;
    let pa = rmax | 1;
    let pv = cmax | 1;
    let pm = cmax / 2 + 1;
    let outputs = match (kind, vectors) {
        (_, false) => String::new(),
        (Kind::Svd, true) => "@group(1) @binding(2) var<storage, read_write> lout: array<f32>;\n@group(1) @binding(3) var<storage, read_write> rout: array<f32>;".to_string(),
        (Kind::Eigh, true) => "@group(1) @binding(2) var<storage, read_write> rout: array<f32>;".to_string(),
    };
    let pb = match (kind, vectors) {
        (_, false) => 2,
        (Kind::Svd, true) => 4,
        (Kind::Eigh, true) => 3,
    };
    let read = if eigh {
        "let v = a[base + max(i, j) * C + min(i, j)];"
    } else {
        "var v = 0.0;
        if (p.wide != 0u) { v = a[base + j * R + i]; } else { v = a[base + i * C + j]; }"
    };
    let vdecl = if vectors { format!("var<workgroup> vc: array<f32, {}>;", cmax * pv) } else { String::new() };
    let vinit = if vectors {
        "for (var idx = t; idx < C * C; idx = idx + @T@u) {
        let j = idx / C;
        let c = idx - j * C;
        vc[j * @PV@u + c] = select(0.0, 1.0, c == j);
    }"
    } else {
        ""
    };
    let vrot = if vectors {
        "for (var i = lane; i < C; i = i + @G@u) {
                    let x = vc[bp2 + i];
                    let y = vc[bq2 + i];
                    vc[bp2 + i] = c * x - s * y;
                    vc[bq2 + i] = s * x + c * y;
                }"
    } else {
        ""
    };
    let shift_pass = if eigh {
        "var shift = 0.0;
    for (var l = 0u; l < @T@u; l = l + 1u) { shift = shift + red[@T@u + l]; }
    shift = sqrt(shift);
    for (var j = t; j < C; j = j + @T@u) { ac[j * @PA@u + j] = ac[j * @PA@u + j] + shift; }
    workgroupBarrier();"
    } else {
        "let shift = 0.0;"
    };
    let svd_out = if vectors && !eigh {
        "
    for (var j = t; j < C; j = j + @T@u) {
        let sj = sig[j];
        let nul = sj <= 1.0e-18;
        let inv = select(1.0 / sj, 0.0, nul);
        for (var i = 0u; i < R; i = i + 1u) { ac[j * @PA@u + i] = ac[j * @PA@u + i] * inv; }
        nulc[j] = select(0u, 1u, nul);
    }
    workgroupBarrier();
    for (var j = 0u; j < C; j = j + 1u) {
        if (workgroupUniformLoad(&nulc[j]) != 0u) {
            for (var i = t; i < R; i = i + @T@u) {
                var q = 0.0;
                for (var k = 0u; k < C; k = k + 1u) { let x = ac[k * @PA@u + i]; q = fma(x, x, q); }
                rsq[i] = q;
            }
            workgroupBarrier();
            var best = 3.0e38;
            var c0 = 0u;
            for (var i = 0u; i < R; i = i + 1u) {
                let q = rsq[i];
                if (q < best) { best = q; c0 = i; }
            }
            for (var i = t; i < R; i = i + @T@u) {
                var v = select(0.0, 1.0, i == c0);
                for (var k = 0u; k < C; k = k + 1u) {
                    if (k != j) { v = fma(-ac[k * @PA@u + c0], ac[k * @PA@u + i], v); }
                }
                ac[j * @PA@u + i] = v;
            }
            workgroupBarrier();
            for (var k = t; k < C; k = k + @T@u) {
                var d = 0.0;
                if (k != j) {
                    for (var i = 0u; i < R; i = i + 1u) { d = fma(ac[j * @PA@u + i], ac[k * @PA@u + i], d); }
                }
                cf[k] = d;
            }
            workgroupBarrier();
            for (var i = t; i < R; i = i + @T@u) {
                var v = ac[j * @PA@u + i];
                for (var k = 0u; k < C; k = k + 1u) {
                    if (k != j) { v = fma(-cf[k], ac[k * @PA@u + i], v); }
                }
                ac[j * @PA@u + i] = v;
            }
            workgroupBarrier();
            var nv = 0.0;
            for (var i = 0u; i < R; i = i + 1u) { let x = ac[j * @PA@u + i]; nv = fma(x, x, nv); }
            workgroupBarrier();
            let inv = inverseSqrt(nv);
            for (var i = t; i < R; i = i + @T@u) { ac[j * @PA@u + i] = ac[j * @PA@u + i] * inv; }
            workgroupBarrier();
        }
    }
    let wide = p.wide != 0u;
    for (var d = t; d < RC; d = d + @T@u) {
        let dv = select(C, R, wide);
        let q1 = d / dv;
        let q2 = d - q1 * dv;
        let i = select(q1, q2, wide);
        let r = select(q2, q1, wide);
        lout[mi * RC + d] = ac[ord[r] * @PA@u + i];
    }"
    } else {
        ""
    };
    let vec_out = if vectors {
        "
    let wide2 = p.wide != 0u || @EIGH@;
    for (var d = t; d < C * C; d = d + @T@u) {
        let q1 = d / C;
        let q2 = d - q1 * C;
        let rr = select(q1, q2, wide2);
        let cc = select(q2, q1, wide2);
        rout[mi * C * C + d] = vc[ord[rr] * @PV@u + cc];
    }"
    } else {
        ""
    };
    let gsum = if g > 1 {
        "red[t * 3u] = app;
            red[t * 3u + 1u] = aqq;
            red[t * 3u + 2u] = apq;
            workgroupBarrier();
            app = 0.0;
            aqq = 0.0;
            apq = 0.0;
            let gb = slot * @G@u * 3u;
            for (var l = 0u; l < @G@u; l = l + 1u) {
                app = app + red[gb + l * 3u];
                aqq = aqq + red[gb + l * 3u + 1u];
                apq = apq + red[gb + l * 3u + 2u];
            }"
    } else {
        ""
    };
    let template = r#"
struct Params { r: u32, c: u32, batch: u32, wide: u32, tol: f32 }
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<storage, read_write> sv: array<f32>;
@OUTPUTS@
@group(1) @binding(@PB@) var<uniform> p: Params;
var<workgroup> ac: array<f32, @ACSIZE@>;
@VDECL@
var<workgroup> red: array<f32, @REDSIZE@>;
var<workgroup> sig: array<f32, @CMAX@>;
var<workgroup> ord: array<u32, @CMAX@>;
var<workgroup> nulc: array<u32, @CMAX@>;
var<workgroup> cf: array<f32, @CMAX@>;
var<workgroup> rsq: array<f32, @RMAX@>;
var<workgroup> flg: array<u32, @FLGSIZE@>;
var<workgroup> ctl: array<u32, 2>;
@ROTATION@

@compute @workgroup_size(@T@)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let mi = wid.x + wid.y * nwg.x;
    let valid = mi < p.batch;
    let R = select(0u, p.r, valid);
    let C = select(0u, p.c, valid);
    let RC = R * C;
    let base = mi * RC;
    let slot = t / @G@u;
    let lane = t - slot * @G@u;

    var mx = 0.0;
    for (var idx = t; idx < RC; idx = idx + @T@u) {
        let i = idx / C;
        let j = idx - i * C;
        @READ@
        ac[j * @PA@u + i] = v;
        mx = max(mx, abs(v));
    }
    red[t] = mx;
    workgroupBarrier();
    var gm = 0.0;
    for (var l = 0u; l < @T@u; l = l + 1u) { gm = max(gm, red[l]); }
    let e = (bitcast<u32>(gm) >> 23u) & 0xffu;
    let scaled = e >= 2u && e <= 252u;
    let sc = select(1.0, bitcast<f32>((254u - e) << 23u), scaled);
    let isc = select(1.0, bitcast<f32>(e << 23u), scaled);
    var ss = 0.0;
    for (var idx = t; idx < RC; idx = idx + @T@u) {
        let i = idx / C;
        let j = idx - i * C;
        let v = ac[j * @PA@u + i] * sc;
        ac[j * @PA@u + i] = v;
        ss = fma(v, v, ss);
    }
    red[@T@u + t] = ss;
    @VINIT@
    for (var f = t; f < @FLGSIZE@u; f = f + @T@u) { flg[f] = 0u; }
    workgroupBarrier();
    @SHIFT@

    let cp = C + (C & 1u);
    let np = cp / 2u;
    let m1 = cp - 1u;
    for (var sweep = 0u; sweep < @MAXS@u && C > 1u; sweep = sweep + 1u) {
        let fb = (sweep & 1u) * @PM@u;
        for (var r = 0u; r < m1; r = r + 1u) {
            var p0 = 0u;
            var q0 = 0u;
            var ok = false;
            if (slot < np) {
                let jb = cp - 1u - slot;
                p0 = (slot + r) % m1;
                q0 = select((jb + r) % m1, m1, jb == m1);
                ok = p0 < C && q0 < C;
            }
            let bp = p0 * @PA@u;
            let bq = q0 * @PA@u;
            let bp2 = p0 * @PV@u;
            let bq2 = q0 * @PV@u;
            var app = 0.0;
            var aqq = 0.0;
            var apq = 0.0;
            if (ok) {
                for (var i = lane; i < R; i = i + @G@u) {
                    let x = ac[bp + i];
                    let y = ac[bq + i];
                    app = fma(x, x, app);
                    aqq = fma(y, y, aqq);
                    apq = fma(x, y, apq);
                }
            }
            @GSUM@
            if (ok && abs(apq) > p.tol * (sqrt(app) * sqrt(aqq))) {
                let cs = rotation(app, aqq, apq);
                let c = cs.x;
                let s = cs.y;
                for (var i = lane; i < R; i = i + @G@u) {
                    let x = ac[bp + i];
                    let y = ac[bq + i];
                    ac[bp + i] = c * x - s * y;
                    ac[bq + i] = s * x + c * y;
                }
                @VROT@
                if (lane == 0u) { flg[fb + slot] = 1u; }
            }
            workgroupBarrier();
        }
        if (t == 0u) {
            var any = 0u;
            for (var k = 0u; k < np; k = k + 1u) { any = any | flg[fb + k]; }
            ctl[0] = any;
        }
        for (var f = t; f < @PM@u; f = f + @T@u) { flg[(1u - (sweep & 1u)) * @PM@u + f] = 0u; }
        if (workgroupUniformLoad(&ctl[0]) == 0u) { break; }
    }

    for (var j = t; j < C; j = j + @T@u) {
        var s2 = 0.0;
        for (var i = 0u; i < R; i = i + 1u) { let x = ac[j * @PA@u + i]; s2 = fma(x, x, s2); }
        sig[j] = sqrt(s2);
    }
    workgroupBarrier();
    for (var j = t; j < C; j = j + @T@u) {
        let kj = key(sig[j]);
        var rk = 0u;
        for (var k = 0u; k < C; k = k + 1u) {
            let kk = key(sig[k]);
            rk = rk + select(0u, 1u, kk @BEFORE@ kj || (kk == kj && k < j));
        }
        ord[rk] = j;
    }
    workgroupBarrier();
    for (var r = t; r < C; r = r + @T@u) { sv[mi * C + r] = (sig[ord[r]] - shift) * isc; }
    @SVDOUT@
    @VECOUT@
}
"#;
    template
        .replace("@OUTPUTS@", &outputs)
        .replace("@VDECL@", &vdecl)
        .replace("@ROTATION@", ROTATION)
        .replace("@READ@", read)
        .replace("@VINIT@", vinit)
        .replace("@VROT@", vrot)
        .replace("@SHIFT@", shift_pass)
        .replace("@SVDOUT@", svd_out)
        .replace("@VECOUT@", vec_out)
        .replace("@GSUM@", gsum)
        .replace("@BEFORE@", if eigh { "<" } else { ">" })
        .replace("@EIGH@", if eigh { "true" } else { "false" })
        .replace("@PB@", &pb.to_string())
        .replace("@ACSIZE@", &(cmax * pa).to_string())
        .replace("@REDSIZE@", &(3 * t).to_string())
        .replace("@CMAX@", &cmax.to_string())
        .replace("@RMAX@", &rmax.to_string())
        .replace("@FLGSIZE@", &(2 * pm).to_string())
        .replace("@PM@", &pm.to_string())
        .replace("@PA@", &pa.to_string())
        .replace("@PV@", &pv.to_string())
        .replace("@MAXS@", &MAX_SWEEPS.to_string())
        .replace("@T@", &t.to_string())
        .replace("@G@", &g.to_string())
}

/// Shapes of the arrays of a decomposition of `rows x cols` matrices: `(s, left, right)` where
/// `left` and `right` are the SVD's `u` and `vt` (eigenvectors for `Kind::Eigh`).
fn output_shapes(kind: Kind, d: Dims) -> (Shape, Shape, Shape) {
    let k = d.rows.min(d.cols);
    let s = Shape::from([k, 1, d.batch]);
    match kind {
        Kind::Svd => (s, Shape::from([k, d.rows, d.batch]), Shape::from([d.cols, k, d.batch])),
        Kind::Eigh => (s, Shape::from([d.cols, d.cols, d.batch]), Shape::from([d.cols, d.cols, d.batch])),
    }
}

/// Decomposes every matrix of `a` with the workgroup kernel of `plan`, one launch.
fn small(a: &Array<f32>, d: Dims, plan: Plan) -> Result<Outputs, LumaError> {
    let ctx = a.context();
    let (rr, cc) = if d.rows >= d.cols { (d.rows, d.cols) } else { (d.cols, d.rows) };
    let wide = d.rows < d.cols;
    let (s_shape, u_shape, vt_shape) = output_shapes(plan.kind, d);
    let mut s = Array::<f32>::alloc(ctx, s_shape)?;
    let params = SmallParams { r: rr as u32, c: cc as u32, batch: d.batch as u32, wide: u32::from(wide), tol: tolerance(rr) };
    let key = format!("svd_small:{plan:?}");
    let groups = per_matrix(ctx, d.batch);
    match (plan.kind, plan.vectors) {
        (_, false) => {
            let args: [KernelArg<'_>; 3] = [a.into(), (&mut s).into(), uniform(&params)];
            launch(ctx, &key, "svd_small", || small_source(&plan), groups, &args)?;
            Ok(Outputs { s, u: None, vt: None })
        }
        (Kind::Svd, true) => {
            let mut u = Array::<f32>::alloc(ctx, u_shape)?;
            let mut vt = Array::<f32>::alloc(ctx, vt_shape)?;
            {
                // The kernel's "left" output holds `X`'s left vectors: `u` for a tall matrix, `vt` for a wide one.
                let (left, right) = if wide { (&mut vt, &mut u) } else { (&mut u, &mut vt) };
                let args: [KernelArg<'_>; 5] = [a.into(), (&mut s).into(), left.into(), right.into(), uniform(&params)];
                launch(ctx, &key, "svd_small", || small_source(&plan), groups, &args)?;
            }
            Ok(Outputs { s, u: Some(u), vt: Some(vt) })
        }
        (Kind::Eigh, true) => {
            let mut v = Array::<f32>::alloc(ctx, vt_shape)?;
            let args: [KernelArg<'_>; 4] = [a.into(), (&mut s).into(), (&mut v).into(), uniform(&params)];
            launch(ctx, &key, "eigh_small", || small_source(&plan), groups, &args)?;
            Ok(Outputs { s, u: None, vt: Some(v) })
        }
    }
}

// ---- Matrices that do not fit a workgroup ---------------------------------------------------

#[derive(encase::ShaderType)]
struct BigParams {
    r: u32,
    c: u32,
    round: u32,
    sweep: u32,
    tol: f32,
}

#[derive(encase::ShaderType)]
struct DimParams {
    r: u32,
    c: u32,
}

#[derive(encase::ShaderType)]
struct EmitParams {
    r: u32,
    c: u32,
    wide: u32,
}

/// Invocations per workgroup of the big-matrix kernels for `r`-row columns.
fn big_threads(r: usize) -> usize {
    (r / 8).next_power_of_two().clamp(32, 256)
}

/// One round-robin round, one workgroup per column pair: the three dot products of the two
/// columns (a workgroup reduction), the rotation, and the update of the two columns and the two
/// columns of `V`.
fn round_source(t: usize, vectors: bool) -> String {
    let (vbind, vrot, pb) = if vectors {
        (
            "@group(1) @binding(1) var<storage, read_write> v: array<f32>;",
            "for (var i = t; i < rows2; i = i + @T@u) {
                let x = v[bp2 + i];
                let y = v[bq2 + i];
                v[bp2 + i] = c * x - s * y;
                v[bq2 + i] = s * x + c * y;
            }",
            3,
        )
    } else {
        ("", "", 2)
    };
    let chg = pb - 1;
    let template = r#"
struct Params { r: u32, c: u32, round: u32, sweep: u32, tol: f32 }
@group(1) @binding(0) var<storage, read_write> w: array<f32>;
@VBIND@
@group(1) @binding(@CHG@) var<storage, read_write> chg: array<u32>;
@group(1) @binding(@PB@) var<uniform> p: Params;
var<workgroup> red: array<f32, @RED@>;
@ROTATION@

@compute @workgroup_size(@T@)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let k = wid.x + wid.y * nwg.x;
    let cp = p.c + (p.c & 1u);
    let m1 = cp - 1u;
    let jb = cp - 1u - min(k, cp - 1u);
    let p0 = (k + p.round) % m1;
    let q0 = select((jb + p.round) % m1, m1, jb == m1);
    let done = p.sweep > 0u && chg[max(p.sweep, 1u) - 1u] == 0u;
    let ok = k < cp / 2u && p0 < p.c && q0 < p.c && !done;
    let rows = select(0u, p.r, ok);
    let rows2 = select(0u, p.c, ok);
    let bp = p0 * p.r;
    let bq = q0 * p.r;
    let bp2 = p0 * p.c;
    let bq2 = q0 * p.c;
    var app = 0.0;
    var aqq = 0.0;
    var apq = 0.0;
    for (var i = t; i < rows; i = i + @T@u) {
        let x = w[bp + i];
        let y = w[bq + i];
        app = fma(x, x, app);
        aqq = fma(y, y, aqq);
        apq = fma(x, y, apq);
    }
    red[t] = app;
    red[@T@u + t] = aqq;
    red[2u * @T@u + t] = apq;
    workgroupBarrier();
    for (var h = @HALF@u; h > 0u; h = h >> 1u) {
        if (t < h) {
            red[t] = red[t] + red[t + h];
            red[@T@u + t] = red[@T@u + t] + red[@T@u + t + h];
            red[2u * @T@u + t] = red[2u * @T@u + t] + red[2u * @T@u + t + h];
        }
        workgroupBarrier();
    }
    app = red[0];
    aqq = red[@T@u];
    apq = red[2u * @T@u];
    if (ok && abs(apq) > p.tol * (sqrt(app) * sqrt(aqq))) {
        let cs = rotation(app, aqq, apq);
        let c = cs.x;
        let s = cs.y;
        for (var i = t; i < rows; i = i + @T@u) {
            let x = w[bp + i];
            let y = w[bq + i];
            w[bp + i] = c * x - s * y;
            w[bq + i] = s * x + c * y;
        }
        @VROT@
        if (t == 0u) { chg[p.sweep] = 1u; }
    }
}
"#;
    template
        .replace("@VBIND@", vbind)
        .replace("@VROT@", vrot)
        .replace("@ROTATION@", ROTATION)
        .replace("@CHG@", &chg.to_string())
        .replace("@PB@", &pb.to_string())
        .replace("@RED@", &(3 * t).to_string())
        .replace("@HALF@", &(t / 2).to_string())
        .replace("@T@", &t.to_string())
}

/// Column norms: one workgroup per column of `w`.
fn norms_source() -> String {
    r#"
struct Params { r: u32, c: u32 }
@group(1) @binding(0) var<storage, read> w: array<f32>;
@group(1) @binding(1) var<storage, read_write> sig: array<f32>;
@group(1) @binding(2) var<uniform> p: Params;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let j = wid.x + wid.y * nwg.x;
    let rows = select(0u, p.r, j < p.c);
    var s2 = 0.0;
    for (var i = t; i < rows; i = i + 128u) {
        let x = w[j * p.r + i];
        s2 = fma(x, x, s2);
    }
    red[t] = s2;
    workgroupBarrier();
    for (var h = 64u; h > 0u; h = h >> 1u) {
        if (t < h) { red[t] = red[t] + red[t + h]; }
        workgroupBarrier();
    }
    if (t == 0u && j < p.c) { sig[j] = sqrt(red[0]); }
}
"#
    .to_string()
}

/// Sorts the singular values by rank counting: `sv[rank j] = sig[j] - shift`, `ord[rank j] = j`.
fn rank_source(eigh: bool) -> String {
    r#"
struct Params { r: u32, c: u32 }
@group(1) @binding(0) var<storage, read> sig: array<f32>;
@group(1) @binding(1) var<storage, read_write> sv: array<f32>;
@group(1) @binding(2) var<storage, read_write> ord: array<u32>;
@group(1) @binding(3) var<storage, read> shift: array<f32>;
@group(1) @binding(4) var<uniform> p: Params;
@ROTATION@

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let j = (wid.x + wid.y * nwg.x) * 128u + t;
    if (j < p.c) {
        let kj = key(sig[j]);
        var rk = 0u;
        for (var k = 0u; k < p.c; k = k + 1u) {
            let kk = key(sig[k]);
            rk = rk + select(0u, 1u, kk @BEFORE@ kj || (kk == kj && k < j));
        }
        ord[rk] = j;
        sv[rk] = sig[j] - shift[0];
    }
}
"#
    .replace("@ROTATION@", ROTATION)
    .replace("@BEFORE@", if eigh { "<" } else { ">" })
}

/// The vectors in the order of the values: `lout[i, r] = w[:, ord r] / sigma`, `rout = v[:, ord r]`
/// in the layout of [`small_source`] (`wide` selects the transposed one; eigenvectors are `wide`).
fn emit_source(kind: Kind) -> String {
    let eigh = kind == Kind::Eigh;
    let (bind, left) = if eigh {
        ("@group(1) @binding(2) var<storage, read_write> rout: array<f32>;\n@group(1) @binding(3) var<uniform> p: Params;", "")
    } else {
        (
            "@group(1) @binding(2) var<storage, read_write> rout: array<f32>;
@group(1) @binding(3) var<storage, read_write> lout: array<f32>;
@group(1) @binding(4) var<storage, read> w: array<f32>;
@group(1) @binding(5) var<storage, read> sv: array<f32>;
@group(1) @binding(6) var<uniform> p: Params;",
            "if (d < p.r * p.c) {
        let dv = select(p.c, p.r, wide);
        let q1 = d / dv;
        let q2 = d - q1 * dv;
        let i = select(q1, q2, wide);
        let r = select(q2, q1, wide);
        let sj = sv[r];
        lout[d] = w[ord[r] * p.r + i] * select(1.0 / sj, 0.0, sj <= 1.0e-18);
    }",
        )
    };
    r#"
struct Params { r: u32, c: u32, wide: u32 }
@group(1) @binding(0) var<storage, read> v: array<f32>;
@group(1) @binding(1) var<storage, read> ord: array<u32>;
@BIND@

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let d = (wid.x + wid.y * nwg.x) * 256u + t;
    let wide = p.wide != 0u || @EIGH@;
    @LEFT@
    if (d < p.c * p.c) {
        let q1 = d / p.c;
        let q2 = d - q1 * p.c;
        let rr = select(q1, q2, wide);
        let cc = select(q2, q1, wide);
        rout[d] = v[ord[rr] * p.c + cc];
    }
}
"#
    .replace("@BIND@", bind)
    .replace("@LEFT@", left)
    .replace("@EIGH@", if eigh { "true" } else { "false" })
}

/// Most sweeps the global-memory algorithm runs.
const MAX_BIG_SWEEPS: u32 = 60;
/// Sweeps recorded before the first look at whether the iteration has converged, and between the
/// following looks. Nothing converges in fewer than a handful of sweeps, and a look costs one wait
/// for the device (and the rounds recorded after convergence return at once).
const FIRST_CHUNK: u32 = 8;
const NEXT_CHUNK: u32 = 2;

/// Columns from which a matrix is decomposed by block Jacobi ([`block`]) instead of rotating one pair of
/// columns per workgroup.
const BLOCK_MIN: usize = 128;

/// Whether a big matrix of `rr >= cc` is decomposed through the QR factorisation of its tall form
/// (see [`preconditioned`]): from 2048 columns on, and from [`BLOCK_MIN`] columns on when it is
/// three times taller than wide (the Jacobi sweeps then work on the square `R` instead of on
/// columns of all the rows). `LUMA_SVD_PRECOND=off` turns it off, a number changes the 2048.
fn precondition(rr: usize, cc: usize) -> bool {
    let setting = std::env::var("LUMA_SVD_PRECOND").ok();
    if setting.as_deref() == Some("off") {
        return false;
    }
    let min = setting.and_then(|v| v.parse().ok()).unwrap_or(2048);
    cc >= min || (cc >= BLOCK_MIN && rr >= 3 * cc)
}

/// Decomposes one matrix (`[cols, rows]`) in global memory.
fn big(a: &Array<f32>, kind: Kind, vectors: bool, tol: f32) -> Result<Outputs, LumaError> {
    let [cols, rows, ..] = a.shape().dims();
    if kind == Kind::Svd && tol == 0.0 && precondition(rows.max(cols), rows.min(cols)) {
        preconditioned(a, vectors)
    } else {
        plain(a, kind, vectors, tol)
    }
}

/// The decomposition of a big matrix through `A = Q R` (`A` tall, a wide matrix as its transpose):
/// the Jacobi iteration works on `Rᵀ`, `Rᵀ = U' Σ V'ᵀ`, so `A = (Q V') Σ U'ᵀ`. The columns of the
/// triangular `Rᵀ` are far from orthogonal in a different way than those of a dense `A` are: the
/// iteration converges in about three quarters of the sweeps, for the price of the factorisation
/// and one product with `Q`.
fn preconditioned(a: &Array<f32>, vectors: bool) -> Result<Outputs, LumaError> {
    let ctx = a.context();
    let [cols, rows, ..] = a.shape().dims();
    let wide = rows < cols;
    let (rr, cc) = if wide { (cols, rows) } else { (rows, cols) };
    let tall = if wide { a.transpose() } else { a.clone() };
    let mx = tall.abs().max();
    let mxs = select(&mx.gt(0.0f32), &mx, 1.0f32).cache();
    let f = (&tall / &mxs).cache().qr();
    let r = f.r().cache();
    let inner = plain(&r.transpose().cache(), Kind::Svd, vectors, 0.0)?;
    let s = (&inner.s * &mxs.reshape([1, 1, 1])).cache();
    if !vectors {
        return Ok(Outputs { s, u: None, vt: None });
    }
    let (u_r, vt_r) = (inner.u.expect("svd computes u"), inner.vt.expect("svd computes vt"));
    // Q V', the left vectors of the tall matrix; U' are its right ones.
    let v = vt_r.transpose();
    let padded = if rr > cc { concat(1, [&v, &Array::<f32>::zeros_on(ctx, [cc, rr - cc])?]) } else { v };
    let left = f.apply_q(&padded).cache();
    let (u, vt) = if wide { (u_r, left.transpose().cache()) } else { (left, u_r.transpose().cache()) };
    Ok(Outputs { s, u: Some(u), vt: Some(vt) })
}

/// Decomposes one matrix (`[cols, rows]`) in global memory by Jacobi iteration on its columns.
fn plain(a: &Array<f32>, kind: Kind, vectors: bool, rank_tol: f32) -> Result<Outputs, LumaError> {
    let ctx = a.context();
    let [cols, rows, ..] = a.shape().dims();
    let wide = rows < cols;
    let (rr, cc) = if wide { (cols, rows) } else { (rows, cols) };
    let eigh = kind == Kind::Eigh;
    // The symmetric matrix of the lower triangle, scaled so that its largest element is 1.
    let x = if eigh {
        let (c, r) = (Array::<u32>::range_on(ctx, a.shape(), 0)?, Array::<u32>::range_on(ctx, a.shape(), 1)?);
        select(&c.le(&r), a.clone(), a.transpose())
    } else {
        a.clone()
    };
    let mx = x.abs().max();
    let mxs = select(&mx.gt(0.0f32), &mx, 1.0f32).cache();
    let xs = &x / &mxs;
    let shift = if eigh { (&xs * &xs).sum().sqrt().cache() } else { Array::<f32>::zeros_on(ctx, [1])? };
    let xs = if eigh {
        let (c, r) = (Array::<u32>::range_on(ctx, a.shape(), 0)?, Array::<u32>::range_on(ctx, a.shape(), 1)?);
        select(&c.eq(&r), &xs + &shift, &xs)
    } else {
        xs
    };
    // Columns contiguous: the transpose of a tall matrix, a wide one as it is.
    let mut w = if wide { xs.cache() } else { xs.transpose().cache() };
    let mut v = if vectors {
        let shape = Shape::from([cc, cc]);
        let (c, r) = (Array::<u32>::range_on(ctx, shape, 0)?, Array::<u32>::range_on(ctx, shape, 1)?);
        select(&c.eq(&r), 1.0f32, 0.0f32).cache()
    } else {
        Array::<f32>::alloc(ctx, Shape::from(1))?
    };
    let cap = MAX_BIG_SWEEPS;
    let mut chg = Array::<u32>::zeros_on(ctx, [cap as usize + 1])?;
    let tol = tolerance(rr);
    if cc >= BLOCK_MIN {
        // The columns below the floor (squared) are not rotated: for counting singular values only.
        let floor2 = if rank_tol > 0.0 && !vectors {
            let longest = (&w * &w).sum_axis(0).max().sqrt().item()?;
            (rank_tol * longest / (2.0 * (cc as f32).sqrt())).powi(2)
        } else {
            0.0
        };
        block::sweeps(ctx, &mut w, vectors.then_some(&mut v), &mut chg, rr, cc, floor2, cap, FIRST_CHUNK, NEXT_CHUNK)?;
    } else if cc > 1 {
        let t = big_threads(rr);
        let kernel = ctx.kernel(&round_source(t, vectors)).label("svd_round").build()?;
        let pairs = (cc + (cc & 1)) / 2;
        let groups = per_matrix(ctx, pairs);
        let mut done = 0;
        while done < cap {
            let upto = (done + if done == 0 { FIRST_CHUNK } else { NEXT_CHUNK }).min(cap);
            for sweep in done..upto {
                for round in 0..(cc + (cc & 1) - 1) {
                    let params = BigParams { r: rr as u32, c: cc as u32, round: round as u32, sweep, tol };
                    if vectors {
                        let args: [KernelArg<'_>; 4] = [(&mut w).into(), (&mut v).into(), (&mut chg).into(), uniform(&params)];
                        kernel.launch_raw(groups, &args)?;
                    } else {
                        let args: [KernelArg<'_>; 3] = [(&mut w).into(), (&mut chg).into(), uniform(&params)];
                        kernel.launch_raw(groups, &args)?;
                    }
                }
            }
            done = upto;
            // Converged when the last sweep rotated nothing (the sweeps after it did not run).
            let flags = chg.to_vec()?;
            if flags[done as usize - 1] == 0 {
                break;
            }
        }
    }
    let dims = DimParams { r: rr as u32, c: cc as u32 };
    let dims3 = EmitParams { r: rr as u32, c: cc as u32, wide: u32::from(wide) };
    let mut sig = Array::<f32>::alloc(ctx, Shape::from([cc]))?;
    {
        let args: [KernelArg<'_>; 3] = [(&w).into(), (&mut sig).into(), uniform(&dims)];
        launch(ctx, "svd_norms", "svd_norms", norms_source, per_matrix(ctx, cc), &args)?;
    }
    let mut sv = Array::<f32>::alloc(ctx, Shape::from([cc, 1]))?;
    let mut ord = Array::<u32>::alloc(ctx, Shape::from([cc]))?;
    {
        let args: [KernelArg<'_>; 5] = [(&sig).into(), (&mut sv).into(), (&mut ord).into(), (&shift).into(), uniform(&dims)];
        launch(ctx, &format!("svd_rank:{eigh}"), "svd_rank", || rank_source(eigh), per_matrix(ctx, cc.div_ceil(128)), &args)?;
    }
    let s = (&sv * &mxs).reshape([cc, 1, 1]);
    if !vectors {
        return Ok(Outputs { s, u: None, vt: None });
    }
    let dg = Dims { rows, cols, batch: 1 };
    let (_, u_shape, vt_shape) = output_shapes(kind, dg);
    let work = if eigh { cc * cc } else { rr * cc };
    let groups = per_matrix(ctx, work.div_ceil(256));
    if eigh {
        let mut vecs = Array::<f32>::alloc(ctx, vt_shape)?;
        let args: [KernelArg<'_>; 4] = [(&v).into(), (&ord).into(), (&mut vecs).into(), uniform(&dims3)];
        launch(ctx, "eigh_emit", "eigh_emit", || emit_source(kind), groups, &args)?;
        return Ok(Outputs { s, u: None, vt: Some(vecs) });
    }
    let mut u = Array::<f32>::alloc(ctx, u_shape)?;
    let mut vt = Array::<f32>::alloc(ctx, vt_shape)?;
    {
        let (left, right) = if wide { (&mut vt, &mut u) } else { (&mut u, &mut vt) };
        let args: [KernelArg<'_>; 7] = [(&v).into(), (&ord).into(), right.into(), left.into(), (&w).into(), (&sv).into(), uniform(&dims3)];
        launch(ctx, "svd_emit", "svd_emit", || emit_source(kind), groups, &args)?;
    }
    // Exactly zero singular values (sorted last) have no left vector: complete them to an orthonormal
    // set, the last columns of the full Q of the others. Reading the values back costs one wait.
    let nonzero = sv.to_vec()?.iter().take_while(|&&x| x.is_nan() || x > NULL_SIGMA).count();
    if nonzero < cc {
        let left = if wide { vt.transpose() } else { u.clone() };
        let complete = if nonzero == 0 {
            // [cc, rr]: the first columns of the identity as rows.
            let shape = Shape::from([cc, rr]);
            select(&Array::<u32>::range_on(ctx, shape, 0)?.eq(&Array::<u32>::range_on(ctx, shape, 1)?), 1.0f32, 0.0f32)
        } else {
            let kept = left.slice(0, 0..nonzero).cache();
            let q = kept.qr().q();
            concat(0, [&kept, &q.slice(0, nonzero..cc)])
        };
        if wide {
            vt = complete.transpose().cache();
        } else {
            u = complete.cache();
        }
    }
    Ok(Outputs { s, u: Some(u), vt: Some(vt) })
}

/// Singular values at or below this (after scaling the matrix to a largest element of 1) are zero.
const NULL_SIGMA: f32 = 1.0e-18;

/// The decomposition of every matrix of `a`: the entry point of the public operations.
pub(crate) fn decompose(a: &Array<f32>, kind: Kind, vectors: bool, op: &'static str, loc: &'static Location<'static>) -> Outputs {
    decompose_ranking(a, kind, vectors, 0.0, op, loc)
}

/// [`decompose`] for singular values that are only *counted* above `tol * s_max` (`tol > 0`, no
/// vectors): the columns of a big matrix whose norm is below `tol * s_max / (2 sqrt(n))` are not
/// rotated any more (the singular values they make are below the cut: `n` of them that are all
/// parallel make `tol * s_max / 2` at most). A low-rank matrix then converges in the sweeps of
/// its rank, the rounding noise of the rest is left alone.
pub(crate) fn decompose_ranking(a: &Array<f32>, kind: Kind, vectors: bool, tol: f32, op: &'static str, loc: &'static Location<'static>) -> Outputs {
    let ctx = a.context();
    let poisoned = |shape: Shape, e: LumaError| {
        let bad = || fail(ctx, shape, e.clone(), op, loc);
        Outputs { s: bad(), u: (vectors && kind == Kind::Svd).then(bad), vt: vectors.then(bad) }
    };
    let d = match if kind == Kind::Eigh { Dims::square(a) } else { Dims::of(a) } {
        Ok(d) => d,
        Err(e) => return poisoned(a.shape(), e),
    };
    let (rr, cc) = if d.rows >= d.cols { (d.rows, d.cols) } else { (d.cols, d.rows) };
    let result = if let Some(plan) = plan(ctx, kind, vectors, rr, cc) {
        small(a, d, plan)
    } else if d.batch == 1 {
        big(a, kind, vectors, tol)
    } else {
        (0..d.batch).map(|i| big(&nth(a, i), kind, vectors, tol)).collect::<Result<Vec<_>, _>>().map(|parts| {
            let join = |f: &dyn Fn(&Outputs) -> Option<Array<f32>>| -> Option<Array<f32>> {
                let arrays: Vec<Array<f32>> = parts.iter().filter_map(f).collect();
                (!arrays.is_empty()).then(|| stack_batch(&arrays))
            };
            Outputs { s: join(&|o| Some(o.s.clone())).expect("one matrix at least"), u: join(&|o| o.u.clone()), vt: join(&|o| o.vt.clone()) }
        })
    };
    result.unwrap_or_else(|e| poisoned(a.shape(), e))
}

/// The singular value decomposition `A = U diag(s) Vᵀ`, see [`Array::svd`].
#[derive(Clone, Debug)]
pub struct Svd {
    /// Left singular vectors as columns, `[k, m]` (`[k, m, batch]`), `k = min(m, n)`: `UᵀU = I`
    /// (the column of a singular value that is exactly zero is completed to an orthonormal set:
    /// by Gram-Schmidt in the workgroup kernel, with the full `Q` of a QR factorisation of the
    /// other columns for a big matrix).
    pub u: Array<f32>,
    /// The singular values in descending order, `[k, 1]` (`[k, 1, batch]`).
    pub s: Array<f32>,
    /// The right singular vectors as rows, `[n, k]` (`[n, k, batch]`): `Vᵀ V = I`.
    pub vt: Array<f32>,
}

/// The reciprocal of the singular values above `tol * s_max`, 0 for the others, `[k, 1, batch]`.
fn reciprocal(s: &Array<f32>, tol: f32) -> Array<f32> {
    let cut = &s.slice(0, 0..1) * tol;
    // `!(s <= cut)` rather than `s > cut`: a NaN value keeps its NaN.
    let ones = select(&s.le(&cut), 1.0f32, 1.0f32);
    select(&s.le(&cut), 0.0f32, &ones / s)
}

/// The tolerance of [`Array::pinv`] and [`Array::rank`]: `max(m, n)` machine epsilons.
fn default_tolerance(a: &Array<f32>) -> f32 {
    let [n, m, ..] = a.shape().dims();
    n.max(m) as f32 * f32::EPSILON
}

impl Array<f32> {
    /// The thin singular value decomposition of a matrix (`[n, m]`, any `m` and `n`) or of every
    /// matrix of a batch (`[n, m, batch]`): `A = U diag(s) Vᵀ` with `k = min(m, n)`, `u` of shape
    /// `[k, m]`, `s` `[k, 1]` and `vt` `[n, k]`, the singular values descending.
    ///
    /// The method is one-sided Jacobi (see the module notes): it computes small singular values to high
    /// *relative* accuracy, a matrix and its factors are orthogonal to a few `f32` epsilons, and a
    /// batch of small matrices (up to 32 x 32 on every device, 64 x 64 where workgroup memory
    /// allows) is one launch with one workgroup per matrix. A matrix with a `NaN` has `NaN` singular
    /// values; nothing hangs (the sweeps are bounded). Matrices with fewer rows than columns are
    /// decomposed through the transpose.
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// let a = Array::from_slice(&[3.0f32, 0.0, 0.0, -2.0], [2, 2])?;
    /// let svd = a.svd();
    /// let s = svd.s.to_vec()?;
    /// assert!((s[0] - 3.0).abs() < 1e-5 && (s[1] - 2.0).abs() < 1e-5);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// The factors are poisoned if the array is not a matrix or a batch of matrices (a fourth axis).
    #[track_caller]
    pub fn svd(&self) -> Svd {
        let o = decompose(self, Kind::Svd, true, "svd", Location::caller());
        Svd { u: o.u.expect("svd computes u"), s: o.s, vt: o.vt.expect("svd computes vt") }
    }

    /// The singular values alone, descending, `[k, 1]` (`[k, 1, batch]`): cheaper than
    /// [`Array::svd`] (no `V`, no vectors written).
    #[track_caller]
    pub fn singular_values(&self) -> Array<f32> {
        decompose(self, Kind::Svd, false, "singular_values", Location::caller()).s
    }

    /// The Moore-Penrose pseudo-inverse of a matrix `[n, m]` (`[n, m, batch]`) as `[m, n]`
    /// (`[m, n, batch]`): `V diag(1/s) Uᵀ` where the singular values at or below
    /// `max(m, n) * eps * s_max` count as zero (see [`Array::pinv_tol`]).
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// // The pseudo-inverse of a column vector (2 rows, 1 column) is its transpose over its squared norm.
    /// let a = Array::from_slice(&[3.0f32, 4.0], [1, 2])?;
    /// let p = a.pinv().to_vec()?;
    /// assert!((p[0] - 0.12).abs() < 1e-6 && (p[1] - 0.16).abs() < 1e-6);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn pinv(&self) -> Array<f32> {
        pinv_of(self, default_tolerance(self), "pinv", Location::caller())
    }

    /// [`Array::pinv`] with an explicit relative tolerance: singular values `<= tol * s_max` are
    /// treated as zero.
    #[track_caller]
    pub fn pinv_tol(&self, tol: f32) -> Array<f32> {
        pinv_of(self, tol, "pinv_tol", Location::caller())
    }

    /// The numerical rank of every matrix, `[1, 1]` (`[1, 1, batch]`) as `u32`: the number of
    /// singular values above `max(m, n) * eps * s_max`. Matrices that fit workgroup memory
    /// (up to 96 columns) are decided by their singular values, lazily, on the device; a bigger
    /// matrix is bounded from both sides by QR factorisations when that is possible (see the notes
    /// of the `rank` module of the source: a rank that is not within the rounding errors of the
    /// factorisations of the cut costs a QR factorisation, not a Jacobi iteration) and by the
    /// singular values otherwise, with the same decision; that path waits for the device.
    ///
    /// ```
    /// # use luma::*;
    /// # let ctx = Context::new()?;
    /// let a = Array::from_slice(&[1.0f32, 2.0, 2.0, 4.0], [2, 2])?;
    /// assert_eq!(a.rank().to_vec()?, vec![1]);
    /// # Ok::<(), luma::LumaError>(())
    /// ```
    #[track_caller]
    pub fn rank(&self) -> Array<u32> {
        rank_of(self, default_tolerance(self), "rank", Location::caller())
    }

    /// [`Array::rank`] with an explicit relative tolerance: the singular values `> tol * s_max`.
    #[track_caller]
    pub fn rank_tol(&self, tol: f32) -> Array<u32> {
        rank_of(self, tol, "rank_tol", Location::caller())
    }

    /// The 2-norm condition number `s_max / s_min` of every matrix, `[1, 1]` (`[1, 1, batch]`);
    /// `Inf` for a singular matrix.
    #[track_caller]
    pub fn cond(&self) -> Array<f32> {
        let s = decompose(self, Kind::Svd, false, "cond", Location::caller()).s;
        let k = s.shape().dims()[0];
        &s.slice(0, 0..1) / &s.slice(0, k - 1..k)
    }
}

fn pinv_of(a: &Array<f32>, tol: f32, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let Svd { u, s, vt } = {
        let o = decompose(a, Kind::Svd, true, op, loc);
        Svd { u: o.u.expect("u"), s: o.s, vt: o.vt.expect("vt") }
    };
    let [k, m, batch, _] = u.shape().dims();
    // V diag(1/s): the columns of V (rows of Vt) scaled.
    let vs = (&vt.transpose() * &reciprocal(&s, tol)).cache();
    let ut = u.transpose().cache();
    if batch > 1 {
        let _ = (k, m);
        vs.matmul_batched(&ut)
    } else {
        matmul_f32(&vs, &ut)
    }
}

fn rank_of(a: &Array<f32>, tol: f32, op: &'static str, loc: &'static Location<'static>) -> Array<u32> {
    if let Ok(d) = Dims::of(a) {
        let (rr, cc) = if d.rows >= d.cols { (d.rows, d.cols) } else { (d.cols, d.rows) };
        if plan(a.context(), Kind::Svd, false, rr, cc).is_none() {
            return big_rank(a, d, tol, op, loc);
        }
    }
    above_cut(&decompose_ranking(a, Kind::Svd, false, tol, op, loc).s, tol)
}

/// The number of singular values above `tol * s_max` (`s` descending, `[k, 1, batch]`).
fn above_cut(s: &Array<f32>, tol: f32) -> Array<u32> {
    let cut = &s.slice(0, 0..1) * tol;
    s.gt(&cut).sum_axis(0)
}

/// The rank of every matrix of a batch of big ones: [`rank::certified`] for each, the singular
/// values for those it cannot decide. Waits for the device (the decisions are made on the host).
fn big_rank(a: &Array<f32>, d: Dims, tol: f32, op: &'static str, loc: &'static Location<'static>) -> Array<u32> {
    let ctx = a.context();
    let one = |i: usize| -> Result<u32, LumaError> {
        let m = if d.batch == 1 { a.clone() } else { nth(a, i) };
        match rank::certified(&m, tol)? {
            Some(k) => Ok(k),
            None => above_cut(&decompose_ranking(&m, Kind::Svd, false, tol, op, loc).s, tol).item(),
        }
    };
    let shape = Shape::from([1, 1, d.batch]);
    match (0..d.batch).map(one).collect::<Result<Vec<_>, _>>().and_then(|ranks| Array::from_slice_on(ctx, &ranks, shape)) {
        Ok(ranks) => ranks,
        Err(e) => Array::resolve(ctx, shape, Err(Failure::New(e)), op, loc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    fn matrix(ctx: &Context, rows: usize, cols: usize, rank: usize, seed: u64) -> Array<f32> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / 16777216.0) * 2.0 - 1.0
        };
        let b: Vec<f32> = (0..rows * rank).map(|_| next()).collect();
        let c: Vec<f32> = (0..rank * cols).map(|_| next()).collect();
        Array::from_slice_on(ctx, &b, [rank, rows]).unwrap().matmul_f32(&Array::from_slice_on(ctx, &c, [cols, rank]).unwrap()).cache()
    }

    fn max_abs(x: &Array<f32>) -> f32 {
        x.abs().max().item().unwrap()
    }

    /// The decomposition through the QR factorisation (used from 2048 columns on) of tall, wide,
    /// square and rank-deficient matrices: backward error, orthogonality, the values of the
    /// decomposition without vectors.
    #[test]
    fn the_qr_preconditioned_decomposition_is_a_decomposition() {
        let ctx = Context::new().unwrap();
        for (rows, cols, rank) in [(150, 150, 150), (200, 130, 130), (130, 200, 130), (150, 150, 70)] {
            let a = matrix(&ctx, rows, cols, rank, 9 + rows as u64);
            let k = rows.min(cols);
            let o = preconditioned(&a, true).unwrap();
            let (u, s, vt) = (o.u.unwrap(), o.s.reshape([k, 1]), o.vt.unwrap());
            assert_eq!(u.shape().dims()[..2], [k, rows]);
            assert_eq!(vt.shape().dims()[..2], [cols, k]);
            // A = U diag(s) Vt.
            let recon = (&u * &s).matmul_f32(&vt);
            let scale = max_abs(&a);
            assert!(max_abs(&(&recon - &a)) <= 3e-5 * scale * (k as f32).sqrt(), "{rows}x{cols} rank {rank}: backward error {}", max_abs(&(&recon - &a)) / scale);
            // Orthonormal columns of U, orthonormal rows of Vt.
            let eye = |n: usize| {
                let shape = Shape::from([n, n]);
                select(&Array::<u32>::range_on(&ctx, shape, 0).unwrap().eq(Array::<u32>::range_on(&ctx, shape, 1).unwrap()), 1.0f32, 0.0f32)
            };
            assert!(max_abs(&(&u.matmul_tn(&u) - &eye(k))) < 1e-4, "{rows}x{cols}: U");
            assert!(max_abs(&(&vt.matmul_nt(&vt) - &eye(k))) < 1e-4, "{rows}x{cols}: Vt");
            // The values alone are the same, and sorted.
            let alone = preconditioned(&a, false).unwrap().s.to_vec().unwrap();
            let s = s.to_vec().unwrap();
            assert!(alone.windows(2).all(|w| w[0] >= w[1]));
            assert!(alone.iter().zip(&s).all(|(x, y)| (x - y).abs() <= 1e-4 * s[0]), "{rows}x{cols}: values");
        }
    }
}
