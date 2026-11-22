//! Block Jacobi for matrices that do not fit workgroup memory.
//!
//! The columns are cut in blocks of [`B`]; a round pairs the blocks (round-robin, disjoint pairs).
//! A round is three launches:
//!
//! 1. `svd_gram`: partial Gram matrices `G = Xᵀ X` of the `2B` columns `X` of every block pair,
//!    one workgroup per pair and [`TR`] rows (the rows stream through workgroup memory);
//! 2. `svd_eigen`: one workgroup per pair sums the partials, diagonalises `G` with cyclic
//!    two-sided Jacobi in workgroup memory (`G = Q Λ Qᵀ`, [`INNER_SWEEPS`] sweep) and writes `Q`;
//! 3. `svd_apply`: the `2B` columns of `W` and of `V` are multiplied by `Q`, an invocation per row.
//!
//! One block round does the work of up to `2B - 1` rounds of the plain algorithm (which has `B`
//! times more, `B` times shorter launches), and what streams over the rows is two reads and one
//! write of the columns instead of a read and a write per rotation. A pair whose Gram matrix is
//! diagonal enough to the tolerance is not rotated. Block Jacobi converges in about as many
//! sweeps as the plain method, also with one inner sweep: the outer sweeps do the rest.

use super::{per_matrix, tolerance, ROTATION};
use crate::array::Array;
use crate::context::Context;
use crate::dense::grid;
use crate::error::LumaError;
use crate::kernel::{uniform, KernelArg};
use crate::shape::Shape;

/// Columns per block; a block pair is `N2 = 2 B` columns.
pub(super) const B: usize = 8;
const N2: usize = 2 * B;
/// Invocations of the eigen workgroup: one per entry of the Gram matrix.
const T1: usize = N2 * N2;
/// Rows of a Gram segment (one tile of workgroup memory per workgroup).
const TR: usize = 128;
/// Rows a workgroup of the apply kernel takes (an invocation per row).
const CH: usize = 128;

/// Sweeps of the inner two-sided Jacobi: one is enough, the outer sweeps iterate.
const INNER_SWEEPS: usize = 1;

#[derive(encase::ShaderType)]
struct Params {
    r: u32,
    c: u32,
    nb: u32,
    round: u32,
    sweep: u32,
    segs: u32,
    tol: f32,
}

const PARAMS: &str = "struct Params { r: u32, c: u32, nb: u32, round: u32, sweep: u32, segs: u32, tol: f32 }";

/// The block pair `ba`, `bb` of pair `k` in round `round` over `nb` blocks, as WGSL.
const PAIR: &str = r#"
    let cp = p.nb + (p.nb & 1u);
    let m1 = cp - 1u;
    let jb = cp - 1u - min(k, cp - 1u);
    let ba = (k + p.round) % m1;
    let bb = select((jb + p.round) % m1, m1, jb == m1);
"#;

/// Partial Gram matrices: workgroup `(seg, pair)` takes rows `seg * TR ..` of the `2B` columns.
fn gram_source() -> String {
    let template = r#"
@PARAMS@
@group(1) @binding(0) var<storage, read> w: array<f32>;
@group(1) @binding(1) var<storage, read_write> part: array<f32>;
@group(1) @binding(2) var<storage, read> chg: array<u32>;
@group(1) @binding(3) var<uniform> p: Params;
var<workgroup> tile: array<f32, @TILE@>;

@compute @workgroup_size(@T@)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let seg = wid.x;
    let k = wid.y + wid.z * nwg.y;
    @PAIR@
    let skip = p.sweep > 0u && chg[max(p.sweep, 1u) - 1u] == 0u;
    let r0 = seg * @TR@u;
    let live = !skip && k < cp / 2u && r0 < p.r;
    let rows = select(0u, min(@TR@u, p.r - min(r0, p.r)), live);
    for (var e = t; e < @TILEELEMS@u; e = e + @T@u) {
        let col = e / @TR@u;
        let row = e - col * @TR@u;
        let cix = select(ba, bb, col >= @B@u) * @B@u + (col % @B@u);
        var v = 0.0;
        if (cix < p.c && row < rows) { v = w[cix * p.r + r0 + row]; }
        tile[col * @TP@u + row] = v;
    }
    workgroupBarrier();
    let i = t / @N2@u;
    let j = t - i * @N2@u;
    var acc = 0.0;
    for (var r = 0u; r < @TR@u; r = r + 1u) {
        acc = fma(tile[i * @TP@u + r], tile[j * @TP@u + r], acc);
    }
    part[(k * p.segs + seg) * @T@u + t] = acc;
}
"#;
    template
        .replace("@PARAMS@", PARAMS)
        .replace("@PAIR@", PAIR)
        .replace("@TILEELEMS@", &(N2 * TR).to_string())
        .replace("@TILE@", &(N2 * (TR + 1)).to_string())
        .replace("@TP@", &(TR + 1).to_string())
        .replace("@TR@", &TR.to_string())
        .replace("@N2@", &N2.to_string())
        .replace("@B@", &B.to_string())
        .replace("@T@", &T1.to_string())
}

/// The eigenvectors `Q` of the Gram matrix of every block pair: sums the partials, then cyclic
/// two-sided Jacobi on `G` (the columns of `Q` rotated alongside).
fn eigen_source() -> String {
    let template = r#"
@PARAMS@
@group(1) @binding(0) var<storage, read> part: array<f32>;
@group(1) @binding(1) var<storage, read_write> qm: array<f32>;
@group(1) @binding(2) var<storage, read_write> act: array<u32>;
@group(1) @binding(3) var<storage, read_write> chg: array<u32>;
@group(1) @binding(4) var<uniform> p: Params;
var<workgroup> g: array<f32, @GSZ@>;
var<workgroup> q: array<f32, @GSZ@>;
var<workgroup> cs: array<f32, @CSSZ@>;
var<workgroup> pq: array<u32, @CSSZ@>;
var<workgroup> rflag: array<u32, @B@>;
var<workgroup> need: array<u32, @T@>;
var<workgroup> ctl: array<u32, 4>;
@ROTATION@

@compute @workgroup_size(@T@)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let k = wid.x + wid.y * nwg.x;
    @PAIR@
    let skip = p.sweep > 0u && chg[max(p.sweep, 1u) - 1u] == 0u;
    if (t == 0u) {
        ctl[0] = select(1u, 0u, skip || k >= cp / 2u);
    }
    let live = workgroupUniformLoad(&ctl[0]) != 0u;
    let i0 = t / @N2@u;
    let j0 = t - i0 * @N2@u;
    var sum = 0.0;
    if (live) {
        for (var sg = 0u; sg < p.segs; sg = sg + 1u) { sum = sum + part[(k * p.segs + sg) * @T@u + t]; }
    }
    g[i0 * @GP@u + j0] = sum;
    q[i0 * @GP@u + j0] = select(0.0, 1.0, i0 == j0);
    workgroupBarrier();
    var nd = 0u;
    if (i0 < j0 && abs(g[i0 * @GP@u + j0]) > p.tol * (sqrt(g[i0 * @GP@u + i0]) * sqrt(g[j0 * @GP@u + j0]))) { nd = 1u; }
    need[t] = nd;
    workgroupBarrier();
    if (t == 0u) {
        var any = 0u;
        for (var l = 0u; l < @T@u; l = l + 1u) { any = any | need[l]; }
        ctl[1] = select(0u, any, live);
        act[k] = select(0u, any, live);
        if (any != 0u && live) { chg[p.sweep] = 1u; }
    }
    if (workgroupUniformLoad(&ctl[1]) != 0u) {
        let tin = 0.5 * p.tol;
        for (var s2 = 0u; s2 < @INNER@u; s2 = s2 + 1u) {
            if (t < @B@u) { rflag[t] = 0u; }
            for (var r = 0u; r < @N2M1@u; r = r + 1u) {
                if (t < @B@u) {
                    let jb2 = @N2M1@u - t;
                    let pp = (t + r) % @N2M1@u;
                    let qq = select((jb2 + r) % @N2M1@u, @N2M1@u, jb2 == @N2M1@u);
                    let gpp = g[pp * @GP@u + pp];
                    let gqq = g[qq * @GP@u + qq];
                    let gpq = g[pp * @GP@u + qq];
                    var c = 1.0;
                    var s = 0.0;
                    if (abs(gpq) > tin * (sqrt(gpp) * sqrt(gqq))) {
                        let cs2 = rotation(gpp, gqq, gpq);
                        c = cs2.x;
                        s = cs2.y;
                        rflag[t] = 1u;
                    }
                    cs[2u * t] = c;
                    cs[2u * t + 1u] = s;
                    pq[2u * t] = pp;
                    pq[2u * t + 1u] = qq;
                }
                workgroupBarrier();
                for (var task = t; task < @N2B@u; task = task + @T@u) {
                    let i = task / @B@u;
                    let kk = task - i * @B@u;
                    let c = cs[2u * kk];
                    let s = cs[2u * kk + 1u];
                    let pp = pq[2u * kk];
                    let qq = pq[2u * kk + 1u];
                    let x = g[i * @GP@u + pp];
                    let y = g[i * @GP@u + qq];
                    g[i * @GP@u + pp] = c * x - s * y;
                    g[i * @GP@u + qq] = s * x + c * y;
                    let xq = q[i * @GP@u + pp];
                    let yq = q[i * @GP@u + qq];
                    q[i * @GP@u + pp] = c * xq - s * yq;
                    q[i * @GP@u + qq] = s * xq + c * yq;
                }
                workgroupBarrier();
                for (var task = t; task < @N2B@u; task = task + @T@u) {
                    let j = task / @B@u;
                    let kk = task - j * @B@u;
                    let c = cs[2u * kk];
                    let s = cs[2u * kk + 1u];
                    let pp = pq[2u * kk];
                    let qq = pq[2u * kk + 1u];
                    let x = g[pp * @GP@u + j];
                    let y = g[qq * @GP@u + j];
                    g[pp * @GP@u + j] = c * x - s * y;
                    g[qq * @GP@u + j] = s * x + c * y;
                }
                workgroupBarrier();
            }
            if (t == 0u) {
                var any = 0u;
                for (var l = 0u; l < @B@u; l = l + 1u) { any = any | rflag[l]; }
                ctl[2] = any;
            }
            if (workgroupUniformLoad(&ctl[2]) == 0u) { break; }
        }
        qm[k * @T@u + t] = q[i0 * @GP@u + j0];
    }
}
"#;
    template
        .replace("@PARAMS@", PARAMS)
        .replace("@PAIR@", PAIR)
        .replace("@ROTATION@", ROTATION)
        .replace("@GSZ@", &(N2 * (N2 + 1)).to_string())
        .replace("@GP@", &(N2 + 1).to_string())
        .replace("@CSSZ@", &(2 * B).to_string())
        .replace("@N2M1@", &(N2 - 1).to_string())
        .replace("@N2B@", &(N2 * B).to_string())
        .replace("@N2@", &N2.to_string())
        .replace("@INNER@", &INNER_SWEEPS.to_string())
        .replace("@B@", &B.to_string())
        .replace("@T@", &T1.to_string())
}

/// `W <- W Q` and `V <- V Q` on the columns of every active block pair: an invocation per row.
fn apply_source(vectors: bool) -> String {
    let mut cols = String::new();
    let mut load = String::new();
    let mut load_v = String::new();
    for k in 0..N2 {
        let base = if k < B { "ca" } else { "cb" };
        cols += &format!("    let c{k} = {base} + {}u;\n", k % B);
        load += &format!("        var x{k} = 0.0;\n        if (c{k} < p.c) {{ x{k} = w[c{k} * p.r + row]; }}\n");
        load_v += &format!("        var y{k} = 0.0;\n        if (c{k} < p.c) {{ y{k} = v[c{k} * p.c + vr]; }}\n");
    }
    let product = |x: &str, buf: &str, row: &str, stride: &str| -> String {
        let mut s = String::new();
        for j in 0..N2 {
            s += &format!("        var o{j} = 0.0;\n");
        }
        for k in 0..N2 {
            for jj in 0..N2 / 4 {
                s += &format!("        let q{k}_{jj} = qs[{}u];\n", k * (N2 / 4) + jj);
                for l in 0..4 {
                    let j = 4 * jj + l;
                    let comp = ["x", "y", "z", "w"][l];
                    s += &format!("        o{j} = fma({x}{k}, q{k}_{jj}.{comp}, o{j});\n");
                }
            }
        }
        for j in 0..N2 {
            s += &format!("        if (c{j} < p.c) {{ {buf}[c{j} * {stride} + {row}] = o{j}; }}\n");
        }
        s
    };
    let outs = product("x", "w", "row", "p.r");
    let outs_v = product("y", "v", "vr", "p.c");
    let (vbind, vpart, pb) = if vectors {
        (
            "@group(1) @binding(1) var<storage, read_write> v: array<f32>;",
            format!("    if (on && row >= p.r && row - p.r < p.c) {{\n        let vr = row - p.r;\n{load_v}{outs_v}    }}"),
            4,
        )
    } else {
        ("", String::new(), 3)
    };
    let template = r#"
@PARAMS@
@group(1) @binding(0) var<storage, read_write> w: array<f32>;
@VBIND@
@group(1) @binding(@QB@) var<storage, read> qm: array<f32>;
@group(1) @binding(@AB@) var<storage, read> act: array<u32>;
@group(1) @binding(@PB@) var<uniform> p: Params;
var<workgroup> qs: array<vec4<f32>, @QS4@>;

@compute @workgroup_size(@CH@)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let k = wid.y + wid.z * nwg.y;
    @PAIR@
    let on = k < cp / 2u && act[k] != 0u;
    for (var e = t; e < @QS4@u; e = e + @CH@u) {
        if (on) {
            let b4 = k * @QS@u + 4u * e;
            qs[e] = vec4<f32>(qm[b4], qm[b4 + 1u], qm[b4 + 2u], qm[b4 + 3u]);
        }
    }
    workgroupBarrier();
    let row = wid.x * @CH@u + t;
    let ca = ba * @B@u;
    let cb = bb * @B@u;
@COLS@
    if (on && row < p.r) {
@LOAD@
@OUTS@
    }
@VPART@
}
"#;
    template
        .replace("@PARAMS@", PARAMS)
        .replace("@VBIND@", vbind)
        .replace("@QB@", &(pb - 2).to_string())
        .replace("@AB@", &(pb - 1).to_string())
        .replace("@PB@", &pb.to_string())
        .replace("@PAIR@", PAIR)
        .replace("@COLS@", &cols)
        .replace("@LOAD@", &load)
        .replace("@OUTS@", &outs)
        .replace("@VPART@", &vpart)
        .replace("@QS4@", &(N2 * N2 / 4).to_string())
        .replace("@QS@", &(N2 * N2).to_string())
        .replace("@CH@", &CH.to_string())
        .replace("@B@", &B.to_string())
}

/// Records sweeps of block Jacobi on `w` (columns of `rr` rows, `cc` of them) and `v` until one
/// rotates nothing or `cap` sweeps are done: the sweeps are recorded in chunks, with a look at the
/// device's "something was rotated" flags between them.
#[allow(clippy::too_many_arguments)]
pub(super) fn sweeps(
    ctx: &Context,
    w: &mut Array<f32>,
    mut v: Option<&mut Array<f32>>,
    chg: &mut Array<u32>,
    rr: usize,
    cc: usize,
    cap: u32,
    first: u32,
    next: u32,
) -> Result<(), LumaError> {
    let nb = cc.div_ceil(B);
    let padded = nb + (nb & 1);
    let pairs = padded / 2;
    let segs = rr.div_ceil(TR);
    let tol = tolerance(rr);
    let gram = ctx.kernel(&gram_source()).label("svd_gram").build()?;
    let eigen = ctx.kernel(&eigen_source()).label("svd_eigen").build()?;
    let apply = ctx.kernel(&apply_source(v.is_some())).label("svd_apply").build()?;
    let mut part = Array::<f32>::alloc(ctx, Shape::from([pairs * segs * T1]))?;
    let mut qm = Array::<f32>::alloc(ctx, Shape::from([pairs * T1]))?;
    let mut act = Array::<u32>::alloc(ctx, Shape::from([pairs]))?;
    let gram_groups = grid(ctx, segs, pairs);
    let eigen_groups = per_matrix(ctx, pairs);
    let rows = rr + if v.is_some() { cc } else { 0 };
    let apply_groups = grid(ctx, rows.div_ceil(CH), pairs);
    let mut done = 0;
    while done < cap {
        let upto = (done + if done == 0 { first } else { next }).min(cap);
        for sweep in done..upto {
            for round in 0..padded - 1 {
                let params = Params { r: rr as u32, c: cc as u32, nb: nb as u32, round: round as u32, sweep, segs: segs as u32, tol };
                let args: [KernelArg<'_>; 4] = [(&*w).into(), (&mut part).into(), (&*chg).into(), uniform(&params)];
                gram.launch_raw(gram_groups, &args)?;
                let args: [KernelArg<'_>; 5] = [(&part).into(), (&mut qm).into(), (&mut act).into(), (&mut *chg).into(), uniform(&params)];
                eigen.launch_raw(eigen_groups, &args)?;
                match v.as_mut() {
                    Some(v) => {
                        let args: [KernelArg<'_>; 5] = [(&mut *w).into(), (&mut **v).into(), (&qm).into(), (&act).into(), uniform(&params)];
                        apply.launch_raw(apply_groups, &args)?;
                    }
                    None => {
                        let args: [KernelArg<'_>; 4] = [(&mut *w).into(), (&qm).into(), (&act).into(), uniform(&params)];
                        apply.launch_raw(apply_groups, &args)?;
                    }
                }
            }
        }
        done = upto;
        let flags = chg.to_vec()?;
        if flags[done as usize - 1] == 0 {
            break;
        }
    }
    Ok(())
}
