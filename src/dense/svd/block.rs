//! Block Jacobi for matrices that do not fit workgroup memory.
//!
//! The columns are cut in blocks of [`B`]; a round pairs the blocks (round-robin, disjoint pairs).
//! A round is three launches:
//!
//! 1. `svd_gram`: partial Gram matrices `G = Xᵀ X` of the `2B` columns `X` of every block pair,
//!    one workgroup per pair and run of rows (the rows stream through workgroup memory [`TR`] at a
//!    time, every invocation owns a 4 x 4 tile of the Gram matrix in registers);
//! 2. `svd_eigen`: one workgroup per pair sums the partials, diagonalises `G` with cyclic
//!    two-sided Jacobi in workgroup memory (`G = Q Λ Qᵀ`, up to [`INNER_SWEEPS`] sweeps), makes `Q`
//!    orthogonal to the last bit with a Newton-Schulz step (the product of the rotations is
//!    orthogonal to a few 1e-7 only, and `V` is multiplied by thousands of them: without it the
//!    backward error of a 4096 x 4096 decomposition was 5e-5, with it 7e-6) and writes `Q`;
//! 3. `svd_apply`: the `2B` columns of `W` and of `V` are multiplied by `Q`, an invocation per row.
//!
//! One block round does the work of up to `2B - 1` rounds of the plain algorithm (which has `B`
//! times more, `B` times shorter launches), and what streams over the rows is two reads and one
//! write of the columns instead of a read and a write per rotation. A pair whose Gram matrix is
//! diagonal enough to the tolerance is not rotated, nor is a pair with a column of a squared norm
//! of at most `floor2` (when the singular values are only to be counted above a cut, see
//! `decompose_ranking`).

use super::{per_matrix, tolerance, ROTATION};
use crate::array::Array;
use crate::context::Context;
use crate::dense::grid;
use crate::error::LumaError;
use crate::kernel::{uniform, KernelArg};
use crate::shape::Shape;

/// Columns per block; a block pair is `N2 = 2 B` columns. Blocks of 32 columns halve the rounds
/// (and the traffic of the columns through memory) but their eigenproblem and products are too
/// big for the workgroup memory and the registers: 1.3 times slower at 4096 columns on an RTX 3080
/// laptop GPU.
pub(super) const B: usize = 16;
const N2: usize = 2 * B;
/// Entries of the Gram matrix of a block pair.
const ENTRIES: usize = N2 * N2;
/// Invocations of the Gram workgroup: one 4 x 4 tile of the Gram matrix each.
const TG: usize = ENTRIES / 16;
/// Invocations of the eigen workgroup.
const ET: usize = 256;
/// Rows of a tile of workgroup memory in the Gram kernel.
const TR: usize = 64;
/// Rows a workgroup of the apply kernel takes (an invocation per row).
const CH: usize = 128;

/// Sweeps of the inner two-sided Jacobi (it stops earlier when no rotation is left): one is
/// enough, the outer sweeps iterate (3 inner sweeps took 18 outer ones instead of 16 at 4096
/// columns, and cost more per round).
const INNER_SWEEPS: usize = 1;

#[derive(encase::ShaderType)]
struct Params {
    r: u32,
    c: u32,
    nb: u32,
    round: u32,
    sweep: u32,
    segs: u32,
    rw: u32,
    tol: f32,
    floor2: f32,
}

const PARAMS: &str = "struct Params { r: u32, c: u32, nb: u32, round: u32, sweep: u32, segs: u32, rw: u32, tol: f32, floor2: f32 }";

/// The block pair `ba`, `bb` of pair `k` in round `round` over `nb` blocks, as WGSL.
const PAIR: &str = r#"
    let cp = p.nb + (p.nb & 1u);
    let m1 = cp - 1u;
    let jb = cp - 1u - min(k, cp - 1u);
    let ba = (k + p.round) % m1;
    let bb = select((jb + p.round) % m1, m1, jb == m1);
"#;

/// Partial Gram matrices: workgroup `(seg, pair)` takes the `rw` rows from `seg * rw` of the `2B`
/// columns, [`TR`] of them at a time. The tile in workgroup memory is row-major in vec4s (a row
/// of the `2B` columns is `2B / 4` of them, one more as padding so that the stores of a quarter
/// warp, rows apart, hit different banks); an invocation reads its 4 + 4 columns of a row with two
/// vec4 loads and does 16 multiply-adds on them.
fn gram_source() -> String {
    let (b, n2, entries, tg) = (B, N2, ENTRIES, TG);
    let stride = n2 / 4 + 1;
    let mut acc = String::new();
    let mut step = String::new();
    let mut store = String::new();
    for u in 0..4 {
        for v in 0..4 {
            acc += &format!("    var c{u}{v} = 0.0;\n");
            let (x, y) = (["x", "y", "z", "w"][u], ["x", "y", "z", "w"][v]);
            step += &format!("            c{u}{v} = fma(a.{x}, b.{y}, c{u}{v});\n");
            store += &format!("    part[base + (ti * 4u + {u}u) * {n2}u + tj * 4u + {v}u] = c{u}{v};\n");
        }
    }
    let template = r#"
@PARAMS@
@group(1) @binding(0) var<storage, read> w: array<f32>;
@group(1) @binding(1) var<storage, read_write> part: array<f32>;
@group(1) @binding(2) var<storage, read> chg: array<u32>;
@group(1) @binding(3) var<uniform> p: Params;
var<workgroup> tile: array<vec4<f32>, @TILE@>;
var<workgroup> ctl: u32;

@compute @workgroup_size(@T@)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let seg = wid.x;
    let k = wid.y + wid.z * nwg.y;
    @PAIR@
    if (t == 0u) {
        ctl = select(1u, 0u, (p.sweep > 0u && chg[max(p.sweep, 1u) - 1u] == 0u) || k >= cp / 2u);
    }
    if (workgroupUniformLoad(&ctl) == 0u) { return; }
    let ti = t / @NT@u;
    let tj = t % @NT@u;
@ACC@
    for (var c0 = 0u; c0 < p.rw; c0 = c0 + @TR@u) {
        let r0 = seg * p.rw + c0;
        if (r0 >= p.r) { break; }
        for (var i = 0u; i < @LOADS@u; i = i + 1u) {
            let e = t + i * @T@u;
            let c4 = e / @TR@u;
            let row = e - c4 * @TR@u;
            let first = select(ba, bb, c4 >= @NT2@u) * @B@u + (c4 % @NT2@u) * 4u;
            var v = vec4<f32>(0.0);
            if (r0 + row < p.r) {
                if (first < p.c) { v.x = w[first * p.r + r0 + row]; }
                if (first + 1u < p.c) { v.y = w[(first + 1u) * p.r + r0 + row]; }
                if (first + 2u < p.c) { v.z = w[(first + 2u) * p.r + r0 + row]; }
                if (first + 3u < p.c) { v.w = w[(first + 3u) * p.r + r0 + row]; }
            }
            tile[row * @STRIDE@u + c4] = v;
        }
        workgroupBarrier();
        for (var r = 0u; r < @TR@u; r = r + 1u) {
            let a = tile[r * @STRIDE@u + ti];
            let b = tile[r * @STRIDE@u + tj];
@STEP@        }
        workgroupBarrier();
    }
    let base = (k * p.segs + seg) * @ENTRIES@u;
@STORE@}
"#;
    template
        .replace("@PARAMS@", PARAMS)
        .replace("@PAIR@", PAIR)
        .replace("@TILE@", &(TR * stride).to_string())
        .replace("@STRIDE@", &stride.to_string())
        .replace("@TR@", &TR.to_string())
        .replace("@NT@", &(n2 / 4).to_string())
        .replace("@NT2@", &(b / 4).to_string())
        .replace("@B@", &b.to_string())
        .replace("@T@", &tg.to_string())
        .replace("@LOADS@", &(n2 / 4 * TR / tg).to_string())
        .replace("@ENTRIES@", &entries.to_string())
        .replace("@ACC@", &acc)
        .replace("@STEP@", &step)
        .replace("@STORE@", &store)
}

/// The eigenvectors `Q` of the Gram matrix of every block pair: sums the partials, then cyclic
/// two-sided Jacobi on `G` (the columns of `Q` rotated alongside).
fn eigen_source() -> String {
    let (b, n2, entries) = (B, N2, ENTRIES);
    let template = r#"
@PARAMS@
@group(1) @binding(0) var<storage, read> part: array<f32>;
@group(1) @binding(1) var<storage, read_write> qm: array<f32>;
@group(1) @binding(2) var<storage, read_write> act: array<u32>;
@group(1) @binding(3) var<storage, read_write> chg: array<atomic<u32>>;
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
    let skip = p.sweep > 0u && atomicLoad(&chg[max(p.sweep, 1u) - 1u]) == 0u;
    if (t == 0u) {
        ctl[0] = select(1u, 0u, skip || k >= cp / 2u);
        if (skip) { act[k] = 0u; }
    }
    if (workgroupUniformLoad(&ctl[0]) == 0u) { return; }
    var nd = 0u;
    for (var e = t; e < @ENTRIES@u; e = e + @T@u) {
        var sum = 0.0;
        for (var sg = 0u; sg < p.segs; sg = sg + 1u) { sum = sum + part[(k * p.segs + sg) * @ENTRIES@u + e]; }
        let i0 = e / @N2@u;
        let j0 = e - i0 * @N2@u;
        g[i0 * @GP@u + j0] = sum;
        q[i0 * @GP@u + j0] = select(0.0, 1.0, i0 == j0);
    }
    workgroupBarrier();
    for (var e = t; e < @ENTRIES@u; e = e + @T@u) {
        let i0 = e / @N2@u;
        let j0 = e - i0 * @N2@u;
        let gii = g[i0 * @GP@u + i0];
        let gjj = g[j0 * @GP@u + j0];
        if (i0 < j0 && gii > p.floor2 && gjj > p.floor2 && abs(g[i0 * @GP@u + j0]) > p.tol * (sqrt(gii) * sqrt(gjj))) { nd = 1u; }
    }
    need[t] = nd;
    for (var s = @T@u / 2u; s > 0u; s = s / 2u) {
        workgroupBarrier();
        if (t < s) { need[t] = need[t] | need[t + s]; }
    }
    workgroupBarrier();
    if (t == 0u) {
        ctl[1] = need[0];
        act[k] = need[0];
        if (need[0] != 0u) { atomicAdd(&chg[p.sweep], 1u); }
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
                    if (gpp > p.floor2 && gqq > p.floor2 && abs(gpq) > tin * (sqrt(gpp) * sqrt(gqq))) {
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
        // One Newton-Schulz step, Q <- Q (3 I - Qᵀ Q) / 2 = Q - Q E / 2 with E = Qᵀ Q - I: the product
        // of rotations is orthogonal to a few 1e-7 only, and what every round multiplies V by adds
        // up over the rounds. E is small: its entries are summed with their rounding errors kept
        // (TwoSum), the 1 of the diagonal taken off before the errors are added.
        workgroupBarrier();
        for (var e = t; e < @ENTRIES@u; e = e + @T@u) {
            let i0 = e / @N2@u;
            let j0 = e - i0 * @N2@u;
            var sm = 0.0;
            var comp = 0.0;
            for (var l = 0u; l < @N2@u; l = l + 1u) {
                let a = q[l * @GP@u + i0];
                let b = q[l * @GP@u + j0];
                let pr = a * b;
                let pe = fma(a, b, -pr);
                let t2 = sm + pr;
                let bb = t2 - sm;
                comp = comp + (((sm - (t2 - bb)) + (pr - bb)) + pe);
                sm = t2;
            }
            g[i0 * @GP@u + j0] = (sm - select(0.0, 1.0, i0 == j0)) + comp;
        }
        workgroupBarrier();
        for (var e = t; e < @ENTRIES@u; e = e + @T@u) {
            let i0 = e / @N2@u;
            let j0 = e - i0 * @N2@u;
            var corr = 0.0;
            for (var l = 0u; l < @N2@u; l = l + 1u) { corr = fma(q[i0 * @GP@u + l], g[l * @GP@u + j0], corr); }
            qm[k * @ENTRIES@u + e] = q[i0 * @GP@u + j0] - 0.5 * corr;
        }
    }
}
"#;
    template
        .replace("@PARAMS@", PARAMS)
        .replace("@PAIR@", PAIR)
        .replace("@ROTATION@", ROTATION)
        .replace("@GSZ@", &(n2 * (n2 + 1)).to_string())
        .replace("@GP@", &(n2 + 1).to_string())
        .replace("@CSSZ@", &(2 * b).to_string())
        .replace("@N2M1@", &(n2 - 1).to_string())
        .replace("@N2B@", &(n2 * b).to_string())
        .replace("@N2@", &n2.to_string())
        .replace("@ENTRIES@", &entries.to_string())
        .replace("@INNER@", &INNER_SWEEPS.to_string())
        .replace("@B@", &b.to_string())
        .replace("@T@", &ET.to_string())
}

/// `W <- W Q` and `V <- V Q` on the columns of every active block pair: an invocation per row.
fn apply_source(vectors: bool) -> String {
    let (b, n2) = (B, N2);
    let mut cols = String::new();
    let mut load = String::new();
    let mut load_v = String::new();
    for k in 0..n2 {
        let base = if k < b { "ca" } else { "cb" };
        cols += &format!("    let c{k} = {base} + {}u;\n", k % b);
        load += &format!("        var x{k} = 0.0;\n        if (c{k} < p.c) {{ x{k} = w[c{k} * p.r + row]; }}\n");
        load_v += &format!("        var y{k} = 0.0;\n        if (c{k} < p.c) {{ y{k} = v[c{k} * p.c + vr]; }}\n");
    }
    let product = |x: &str, buf: &str, row: &str, stride: &str| -> String {
        let mut s = String::new();
        for j in 0..n2 {
            s += &format!("        var o{j} = 0.0;\n");
        }
        for k in 0..n2 {
            for jj in 0..n2 / 4 {
                s += &format!("        let q{k}_{jj} = qs[{}u];\n", k * (n2 / 4) + jj);
                for l in 0..4 {
                    let j = 4 * jj + l;
                    let comp = ["x", "y", "z", "w"][l];
                    s += &format!("        o{j} = fma({x}{k}, q{k}_{jj}.{comp}, o{j});\n");
                }
            }
        }
        for j in 0..n2 {
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
        .replace("@QS4@", &(n2 * n2 / 4).to_string())
        .replace("@QS@", &(n2 * n2).to_string())
        .replace("@CH@", &CH.to_string())
        .replace("@B@", &b.to_string())
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
    floor2: f32,
    cap: u32,
    first: u32,
    next: u32,
) -> Result<(), LumaError> {
    let (b, entries) = (B, ENTRIES);
    let nb = cc.div_ceil(b);
    let padded = nb + (nb & 1);
    let pairs = padded / 2;
    // Runs of rows per Gram workgroup: about a thousand workgroups a round, whole tiles.
    let tiles = rr.div_ceil(TR);
    let want = 1024usize.div_ceil(pairs).clamp(1, tiles);
    let rw = TR * tiles.div_ceil(want);
    let segs = rr.div_ceil(rw);
    let tol = tolerance(rr);
    let gram = ctx.kernel(&gram_source()).label("svd_gram").build()?;
    let eigen = ctx.kernel(&eigen_source()).label("svd_eigen").build()?;
    let apply = ctx.kernel(&apply_source(v.is_some())).label("svd_apply").build()?;
    let mut part = Array::<f32>::alloc(ctx, Shape::from([pairs * segs * entries]))?;
    let mut qm = Array::<f32>::alloc(ctx, Shape::from([pairs * entries]))?;
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
                let params = Params { r: rr as u32, c: cc as u32, nb: nb as u32, round: round as u32, sweep, segs: segs as u32, rw: rw as u32, tol, floor2 };
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
