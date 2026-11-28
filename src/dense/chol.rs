//! Cholesky factorisation and SPD solves.
//!
//! A big matrix is factored left-looking in blocks of 128 columns (64 where a workgroup has less
//! than 33 KiB of memory): per block one product of everything finished so far (a matrix product, the
//! only big flops), one workgroup that factors the diagonal block in workgroup memory and one
//! launch of warps that solve the rows below it, all of them writing into the one output, so a
//! matrix takes about four launches per 128 columns. Up to 64 rows, and for every matrix of a batch,
//! one workgroup factors a matrix in workgroup memory.

use std::panic::Location;

use super::tri::{block_size, lower_left_factor, rhs_matrix};
use super::{block, fail, launch, launch_unchecked, nth, product, per_matrix, stack_batch, Dims, NAN};
use crate::array::Array;
use crate::context::Context;
use crate::jit::comprehension::Props;
use crate::jit::structured::boundary::FactorKind;
use crate::jit::structured::families::factor::{boundary_of, chol_stage};
use crate::jit::structured::strategy::{first_candidate_in, note_in, StrategyId};
use crate::kernel::{uniform, view, KernelArg};
use crate::shape::Shape;

/// Rows of the matrix a workgroup factors in workgroup memory, at most: 64 where the device has the
/// 16.5 KiB that takes, else 48.
pub(crate) fn leaf(ctx: &Context) -> usize {
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
    has_c: u32,
    t_row0: u32,
    t_col0: u32,
    t_stride: u32,
}

/// The kernel that factors the diagonal block of columns `k..k + nbw`: `A_kk - T[..nbw]` (`T` the
/// product of the finished columns, see [`blocked`]) in workgroup memory as a packed lower
/// triangle (row `i` at `i (i + 1) / 2`, so that the 32 rows of a warp read one column without bank
/// conflicts). Right-looking in panels of 8 columns: the first invocation factors the panel's 8 x 8
/// block in registers (a chain of 8 columns with no barrier between them), every row below solves
/// its 8 entries against it, then a rank-8 update of the rest of the triangle, the four invocations
/// of a row taking every fourth column each. The warps are placed so that each of the four
/// schedulers of a multiprocessor gets one warp of every group of 32 rows (the rows at the end of
/// the triangle have the longest updates). The block's loads and stores go eight rows at a step
/// per invocation, all the loads of a step issued together (a load that waits for its store before
/// the next one is issued costs a full memory latency per row). The factor goes to `dk` (the panel
/// kernel reads it) and into its place in the output.
fn diag_source(nb: usize) -> String {
    let size = nb * (nb + 1) / 2;
    let threads = 4 * nb;
    let groups = nb / 32;
    let rstep = 8 * (threads / nb);
    let mut loads = String::new();
    let mut stores = String::new();
    let mut outs = String::new();
    for u in 0..8 {
        loads += &format!(
            "        let live{u} = cid < nbw && cid <= i0 + {u}u && i0 + {u}u < nbw;
        let rr{u} = min(i0 + {u}u, nbw - 1u);
        let cc{u} = min(cid, nbw - 1u);
        let g{u} = a[luma_view_idx3(av, k + cc{u}, k + rr{u}, 0u)];
        let t{u} = tm[select(0u, (rr{u} + p.t_row0) * p.t_stride + p.t_col0 + cc{u}, p.has_t != 0u)];
        let c{u} = tc[select(0u, rr{u} * nbw + cc{u}, p.has_c != 0u)];
        let w{u} = select(g{u}, g{u} - t{u}, p.has_t != 0u);
        let v{u} = select(w{u}, w{u} - c{u}, p.has_c != 0u);
"
        );
        stores += &format!("        if (live{u}) {{ s[off(i0 + {u}u) + cid] = v{u}; }}\n");
        outs += &format!(
            "        if (cid < nbw && i0 + {u}u < nbw) {{
            let w{u} = select(0.0, s[off(i0 + {u}u) + cid], cid <= i0 + {u}u);
            dk[(i0 + {u}u) * {nb}u + cid] = w{u};
            o[(k + i0 + {u}u) * n + k + cid] = w{u};
        }}
"
        );
    }
    let lds = (0..8).map(|r| format!("            let l{r} = s[ri + c0 + {r}u];\n")).collect::<String>();
    // The trailing columns four at a step (four apart: the other invocations of the row take the ones
    // between): all their loads are issued before the first multiply-add.
    let mut quad_loads = String::new();
    let mut quad_fmas = String::new();
    let mut quad_stores = String::new();
    for w in 0..4 {
        quad_loads += &format!("                let rq{w} = off(q + {}u);\n                var acc{w} = s[ri + q + {}u];\n", 4 * w, 4 * w);
        for r in 0..8 {
            quad_loads += &format!("                let m{w}_{r} = s[rq{w} + c0 + {r}u];\n");
            quad_fmas += &format!("                acc{w} = fma(-l{r}, m{w}_{r}, acc{w});\n");
        }
        quad_stores += &format!("                s[ri + q + {}u] = acc{w};\n", 4 * w);
    }
    let fmas = (0..8).map(|r| format!("                acc = fma(-l{r}, s[rq + c0 + {r}u], acc);\n")).collect::<String>();
    // The panel's block, factored by one invocation: `a{p}_{q}` its entries, `l{p}_{q}` the factor,
    // `iv{q}` the reciprocals of its diagonal (kept in `rv` for the rows below); the rows past the
    // end of a short last panel are those of an identity.
    let mut potf = String::new();
    let mut solve = String::new();
    for p in 0..8 {
        for q in 0..=p {
            let dflt = if p == q { "1.0" } else { "0.0" };
            potf += &format!("            let a{p}_{q} = select({dflt}, s[off(c0 + {p}u) + c0 + {q}u], c0 + {p}u < nbw);\n");
        }
    }
    for q in 0..8 {
        let mut cur = format!("a{q}_{q}");
        for k in 0..q {
            potf += &format!("            let d{q}_{k} = fma(-l{q}_{k}, l{q}_{k}, {cur});\n");
            cur = format!("d{q}_{k}");
        }
        potf += &format!("            let iv{q} = select({NAN}, inverseSqrt({cur}), {cur} > 0.0);\n            let r{q} = {cur} * iv{q};\n");
        for p in q + 1..8 {
            let mut cur = format!("a{p}_{q}");
            for k in 0..q {
                potf += &format!("            let x{p}_{q}_{k} = fma(-l{p}_{k}, l{q}_{k}, {cur});\n");
                cur = format!("x{p}_{q}_{k}");
            }
            potf += &format!("            let l{p}_{q} = {cur} * iv{q};\n");
        }
    }
    for p in 0..8 {
        potf += &format!("            if (c0 + {p}u < nbw) {{\n");
        for q in 0..p {
            potf += &format!("                s[off(c0 + {p}u) + c0 + {q}u] = l{p}_{q};\n");
        }
        potf += &format!("                s[off(c0 + {p}u) + c0 + {p}u] = r{p};\n            }}\n            rv[{p}] = iv{p};\n");
    }
    for p in 0..8 {
        solve += &format!("            let b{p} = s[ri + c0 + {p}u];\n");
    }
    for p in 0..8 {
        let mut cur = format!("b{p}");
        for q in 0..p {
            solve += &format!("            let y{p}_{q} = fma(-x{q}, s[off(c0 + {p}u) + c0 + {q}u], {cur});\n");
            cur = format!("y{p}_{q}");
        }
        solve += &format!("            let x{p} = {cur} * rv[{p}];\n            s[ri + c0 + {p}u] = x{p};\n");
    }
    // The rank-32 update of what is left of the triangle after a panel of 32 columns: tiles of 4 x 4
    // entries, an invocation each, 8 loads of the panel's entries for 16 multiply-adds.
    let mut tile_setup = String::new();
    let mut tile_body = String::new();
    let mut tile_store = String::new();
    for r in 0..4 {
        tile_setup += &format!("                let ir{r} = min(i0 + {r}u, nbw - 1u);\n                let rb{r} = off(ir{r}) + c32;\n");
        tile_setup += &format!("                let qr{r} = min(q0 + {r}u, nbw - 1u);\n                let qb{r} = off(qr{r}) + c32;\n");
    }
    for r in 0..4 {
        for c in 0..4 {
            tile_setup += &format!("                var t{r}{c} = s[off(ir{r}) + qr{c}];\n");
        }
    }
    for kk in 0..4 {
        for r in 0..4 {
            tile_body += &format!("                    let u{r}_{kk} = s[rb{r} + kk + {kk}u];\n                    let w{r}_{kk} = s[qb{r} + kk + {kk}u];\n");
        }
        for r in 0..4 {
            for c in 0..4 {
                tile_body += &format!("                    t{r}{c} = fma(-u{r}_{kk}, w{c}_{kk}, t{r}{c});\n");
            }
        }
    }
    for r in 0..4 {
        for c in 0..4 {
            tile_store += &format!("                if (i0 + {r}u < nbw && q0 + {c}u <= i0 + {r}u) {{ s[off(i0 + {r}u) + q0 + {c}u] = t{r}{c}; }}\n");
        }
    }
    format!(
        "struct Params {{ n: u32, k: u32, nbw: u32, has_t: u32, has_c: u32, t_row0: u32, t_col0: u32, t_stride: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read> tm: array<f32>;
@group(1) @binding(3) var<storage, read> tc: array<f32>;
@group(1) @binding(4) var<storage, read_write> dk: array<f32>;
@group(1) @binding(5) var<storage, read_write> o: array<f32>;
@group(1) @binding(6) var<uniform> p: Params;

var<workgroup> rv: array<f32, 8>;
var<workgroup> s: array<f32, {size}>;

fn off(i: u32) -> u32 {{ return i * (i + 1u) / 2u; }}

@compute @workgroup_size({threads})
fn main(@builtin(local_invocation_index) t: u32) {{
    let nbw = p.nbw;
    let k = p.k;
    let n = p.n;
    let cid = t % {nb}u;
    let hh = t / {nb}u;
    for (var r0 = 0u; r0 < nbw; r0 = r0 + {rstep}u) {{
        let i0 = r0 + 8u * hh;
{loads}{stores}    }}
    workgroupBarrier();
    let wp = t / 32u;
    let h = wp / {groups}u;
    let rg = (wp % {groups}u + h) % {groups}u;
    let i = rg * 32u + t % 32u;
    let ri = off(i);
    for (var c0 = 0u; c0 < nbw; c0 = c0 + 8u) {{
        if (t == 0u) {{
{potf}        }}
        workgroupBarrier();
        if (h == 0u && i >= c0 + 8u && i < nbw) {{
{solve}        }}
        workgroupBarrier();
        if (i < nbw && i >= c0 + 8u) {{
{lds}            let qmax = min(i, c0 - c0 % 32u + 31u);
            var q = c0 + 8u + h;
            for (; q + 12u <= qmax; q = q + 16u) {{
{quad_loads}{quad_fmas}{quad_stores}            }}
            for (; q <= qmax; q = q + 4u) {{
                let rq = off(q);
                var acc = s[ri + q];
{fmas}                s[ri + q] = acc;
            }}
        }}
        workgroupBarrier();
        if ((c0 + 8u) % 32u == 0u && c0 + 8u < nbw) {{
            let c32 = c0 - 24u;
            let span = nbw - c32 - 32u;
            let side = (span + 3u) / 4u;
            let tiles = side * (side + 1u) / 2u;
            for (var id = t; id < tiles; id = id + {threads}u) {{
                var tr = u32(floor((sqrt(f32(8u * id + 1u)) - 1.0) * 0.5));
                if (tr * (tr + 1u) / 2u > id) {{ tr = tr - 1u; }}
                if ((tr + 1u) * (tr + 2u) / 2u <= id) {{ tr = tr + 1u; }}
                let tc = id - tr * (tr + 1u) / 2u;
                let i0 = c32 + 32u + 4u * tr;
                let q0 = c32 + 32u + 4u * tc;
{tile_setup}                for (var kk = 0u; kk < 32u; kk = kk + 4u) {{
{tile_body}                }}
{tile_store}            }}
            workgroupBarrier();
        }}
    }}
    for (var r0 = 0u; r0 < nbw; r0 = r0 + {rstep}u) {{
        let i0 = r0 + 8u * hh;
{outs}    }}
}}
"
    )
}

/// The kernel that solves the rows below a diagonal block: `X = (A_panel - T) L_kkᵀ⁻¹` by forward
/// substitution along each row, a warp (workgroup) taking 32 rows, a lane a row. The rows live in
/// workgroup memory column by column (pitch 33, no bank conflicts). `L_kk` is staged through workgroup
/// memory (as `vec4`s: four entries of a row of it per load) in squares of at most 64 columns: for
/// each block of 64 columns, first the products with the solved blocks before it (the square of
/// `L_kk` below-left), then its own triangle. Four columns at a step share the loads of the solved
/// ones: 16 multiply-adds for 4 loads of the row and 4 of `L_kk`.
fn panel_source(nb: usize) -> String {
    let half = nb.min(64);
    let h4 = half / 4;
    let comp = ["x", "y", "z", "w"];
    // One step of the product loops over four columns of the block and four of the solved ones; `t`
    // tells the copies apart when two steps are issued together.
    let step = |t: &str, col0: &str, rect: bool| -> String {
        let mut out = String::new();
        let base = if rect { "pc" } else { "c0" };
        for v in 0..4 {
            out += &format!("                    let x{t}{v} = tile[({base} + {col0} * 4u + {v}u) * 33u + lane];\n");
        }
        for u in 0..4 {
            out += &format!("                    let l{t}{u} = lb[(j + {u}u) * {h4}u + {col0}];\n");
        }
        for u in 0..4 {
            for (v, c) in comp.iter().enumerate() {
                out += &format!("                    s{u} = fma(-l{t}{u}.{c}, x{t}{v}, s{u});\n");
            }
        }
        out
    };
    let mut acc = String::new();
    let mut put = String::new();
    for u in 0..4 {
        acc += &format!("                var s{u} = tile[(c0 + j + {u}u) * 33u + lane];\n");
        put += &format!("                if (j + {u}u < cw) {{ tile[(c0 + j + {u}u) * 33u + lane] = s{u}; }}\n");
    }
    let dld = (0..4).map(|u| format!("                let d{u} = lb[(j + {u}u) * {h4}u + j / 4u];\n")).collect::<String>();
    let mut diag = String::new();
    for u in 0..4 {
        for (v, c) in comp.iter().enumerate().take(u) {
            diag += &format!("                s{u} = fma(-d{u}.{c}, s{v}, s{u});\n");
        }
        diag += &format!("                s{u} = s{u} / d{u}.{c};\n", c = comp[u]);
    }
    let rect_body = format!("{}{}", step("a", "q4", true), step("b", "(q4 + 1u)", true));
    let diag_pair = format!("{}{}", step("a", "q4", false), step("b", "(q4 + 1u)", false));
    let diag_one = step("a", "q4", false);
    let colblocks = nb / 32;
    let mut loads = String::new();
    let mut stores = String::new();
    for u in 0..4 {
        for cb in 0..colblocks {
            loads += &format!(
                "        let ro{u}_{cb} = min(rr + {u}u, rows - 1u);
        let co{u}_{cb} = min({cb}u * 32u + lane, nbw - 1u);
        let g{u}_{cb} = a[luma_view_idx3(av, k + co{u}_{cb}, r0 + ro{u}_{cb}, 0u)];
        let t{u}_{cb} = tm[select(0u, (r0 + ro{u}_{cb} - k + p.t_row0) * p.t_stride + p.t_col0 + co{u}_{cb}, p.has_t != 0u)];
        let c{u}_{cb} = tc[select(0u, (r0 + ro{u}_{cb} - k) * nbw + co{u}_{cb}, p.has_c != 0u)];
        let w{u}_{cb} = select(g{u}_{cb}, g{u}_{cb} - t{u}_{cb}, p.has_t != 0u);
        let v{u}_{cb} = select(w{u}_{cb}, w{u}_{cb} - c{u}_{cb}, p.has_c != 0u);
"
            );
            stores += &format!("        if (rr + {u}u < rows && {cb}u * 32u + lane < nbw) {{ tile[({cb}u * 32u + lane) * 33u + rr + {u}u] = v{u}_{cb}; }}\n");
        }
    }
    // Staging: four vec4s of a row of `L_kk` per lane and step, the loads issued together.
    let mut sl = String::new();
    let mut ss = String::new();
    for w in 0..4 {
        sl += &format!(
            "        let i{w} = idx + {}u;
        let b{w} = (c0 + i{w} / {h4}u) * {nb}u + pc + (i{w} % {h4}u) * 4u;
        let e{w} = min(b{w}, {last}u);
        let w{w} = vec4<f32>(dk[e{w}], dk[e{w} + 1u], dk[e{w} + 2u], dk[e{w} + 3u]);
",
            32 * w,
            last = nb * nb - 4
        );
        ss += &format!("        if (i{w} < total) {{ lb[i{w}] = w{w}; }}\n");
    }
    format!(
        "struct Params {{ n: u32, k: u32, nbw: u32, has_t: u32, has_c: u32, t_row0: u32, t_col0: u32, t_stride: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read> tm: array<f32>;
@group(1) @binding(3) var<storage, read> tc: array<f32>;
@group(1) @binding(4) var<storage, read> dk: array<f32>;
@group(1) @binding(5) var<storage, read_write> o: array<f32>;
@group(1) @binding(6) var<uniform> p: Params;

var<workgroup> tile: array<f32, {tsize}>;
var<workgroup> lb: array<vec4<f32>, {lsize}>;

fn stage(lane: u32, c0: u32, cw: u32, pc: u32) {{
    let total = cw * {h4}u;
    for (var idx = lane; idx < total; idx = idx + 128u) {{
{sl}{ss}    }}
}}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {{
    let nbw = p.nbw;
    let k = p.k;
    let n = p.n;
    let r0 = k + nbw + wid.x * 32u;
    let rows = min(32u, n - r0);
    for (var rr = 0u; rr < rows; rr = rr + 4u) {{
{loads}{stores}    }}
    workgroupBarrier();
    let blocks = (nbw + {half}u - 1u) / {half}u;
    for (var b = 0u; b < blocks; b = b + 1u) {{
        let c0 = b * {half}u;
        let cw = min({half}u, nbw - c0);
        for (var pb = 0u; pb < b; pb = pb + 1u) {{
            let pc = pb * {half}u;
            stage(lane, c0, cw, pc);
            workgroupBarrier();
            if (lane < rows) {{
                for (var j = 0u; j < cw; j = j + 4u) {{
{acc}                    for (var q4 = 0u; q4 < {h4}u; q4 = q4 + 2u) {{
{rect_body}                    }}
{put}                }}
            }}
            workgroupBarrier();
        }}
        stage(lane, c0, cw, c0);
        workgroupBarrier();
        if (lane < rows) {{
            for (var j = 0u; j < cw; j = j + 4u) {{
{acc}                var q4 = 0u;
                for (; q4 + 2u <= j / 4u; q4 = q4 + 2u) {{
{diag_pair}                }}
                if (q4 < j / 4u) {{
{diag_one}                }}
{dld}{diag}{put}            }}
        }}
        workgroupBarrier();
    }}
    for (var rr = 0u; rr < rows; rr = rr + 1u) {{
        for (var c = lane; c < nbw; c = c + 32u) {{
            o[(r0 + rr) * n + k + c] = tile[c * 33u + rr];
        }}
    }}
}}
",
        tsize = nb * 33,
        lsize = half * h4,
    )
}

/// The columns an outer step of [`blocked`] brings up to date together (`LUMA_CHOL_WIDE`: a multiple of `nb`; default `4 * nb`).
pub(crate) fn wide(nb: usize) -> usize {
    std::env::var("LUMA_CHOL_WIDE").ok().and_then(|v| v.parse().ok()).filter(|w: &usize| w.is_multiple_of(nb) && *w >= nb).unwrap_or(4 * nb)
}

/// The lower Cholesky factor of one `[n, n]` matrix, left-looking in blocks of `nb` columns. Per
/// block: `T = L[k.., ..k] L[k..k+nb, ..k]ᵀ` (one `matmul_f32` over everything finished so far, the
/// only big flops: `n³/3` in all, in `n / nb` products), the diagonal block `A_kk - T` factored by
/// one workgroup ([`diag_source`]) and the rows below it by a launch of one warp per 32 rows
/// ([`panel_source`]): three launches a block, every one of them writing into the one output.
fn blocked(a: &Array<f32>, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = a.context();
    let [n, ..] = a.shape().dims();
    let nb = block_size(ctx);
    // The columns an outer step brings up to date together with one product: a product of 128 columns
    // only fills a fifth of the multiprocessors, one of 512 runs about twice as fast.
    let wide = wide(nb);
    let shape = Shape::from([n, n]);
    let setup = (Array::<f32>::zeros_on(ctx, shape), Array::<f32>::alloc(ctx, Shape::from([nb, nb])), Array::<f32>::alloc(ctx, Shape::from(1)));
    let (mut out, mut dk, dummy) = match setup {
        (Ok(o), Ok(d), Ok(m)) => (o, d, m),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return fail(ctx, a.shape(), e, "cholesky", loc),
    };
    let mut ko = 0;
    while ko < n {
        let wo = wide.min(n - ko);
        // The rows of the outer step are the first rows of the operand: the finished columns are read where they are (a window of `out`).
        let big = (ko > 0).then(|| {
            let done = block(&out, ko..n, 0..ko);
            product(&done, &done.slice(1, 0..wo).transpose())
        });
        let mut k = ko;
        while k < ko + wo {
            let nbw = nb.min(ko + wo - k);
            // The columns of this outer step before the block, `K = k - ko` of them.
            let corr = (k > ko).then(|| {
                let done = block(&out, k..n, ko..k);
                product(&done, &done.slice(1, 0..nbw).transpose())
            });
            let params = BlockParams {
                n: n as u32,
                k: k as u32,
                nbw: nbw as u32,
                has_t: u32::from(big.is_some()),
                has_c: u32::from(corr.is_some()),
                t_row0: (k - ko) as u32,
                t_col0: (k - ko) as u32,
                t_stride: wo as u32,
            };
            let (tref, cref) = (big.as_ref().unwrap_or(&dummy), corr.as_ref().unwrap_or(&dummy));
            let diag: [KernelArg<'_>; 6] = [view(a), tref.into(), cref.into(), (&mut dk).into(), (&mut out).into(), uniform(&params)];
            let key = format!("chol_diag/{nb}");
            if let Err(e) = launch_unchecked(ctx, &key, "chol_diag", || diag_source(nb), [1, 1, 1], &diag) {
                return fail(ctx, a.shape(), e, "cholesky", loc);
            }
            let rows = n - k - nbw;
            if rows > 0 {
                let panel: [KernelArg<'_>; 6] = [view(a), tref.into(), cref.into(), (&dk).into(), (&mut out).into(), uniform(&params)];
                let key = format!("chol_panel/{nb}");
                if let Err(e) = launch_unchecked(ctx, &key, "chol_panel", || panel_source(nb), [rows.div_ceil(32) as u32, 1, 1], &panel) {
                    return fail(ctx, a.shape(), e, "cholesky", loc);
                }
            }
            k += nbw;
        }
        ko += wo;
    }
    out
}

/// The lower Cholesky factor of the lower triangle of every matrix of `a`.
pub(crate) fn lower(a: &Array<f32>, loc: &'static Location<'static>) -> Array<f32> {
    let d = match Dims::square(a) {
        Ok(d) => d,
        Err(e) => return fail(a.context(), a.shape(), e, "cholesky", loc),
    };
    let ctx = a.context();
    let boundary = boundary_of(FactorKind::Cholesky, "cholesky", loc, d, 0);
    let (id, params) = first_candidate_in(chol_stage(), ctx, &boundary).expect("a Cholesky always fits");
    note_in(ctx, &boundary, chol_stage(), id, &params, &[]);
    if id == StrategyId::CHOL_SMALL {
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
    /// workgroup each in one launch; bigger ones in blocks of 128 columns with `matmul_f32` products.
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
        let n = self.shape().dims()[0];
        lower(self, Location::caller()).with_props(Props::triangle(true, false, n, n))
    }

    /// The upper Cholesky factor `U`, `A = Uᵀ U`, reading the upper triangle of `A`: the
    /// transpose of the lower factor of `Aᵀ`.
    ///
    /// # Errors
    ///
    /// As [`Array::cholesky`].
    #[track_caller]
    pub fn cholesky_upper(&self) -> Array<f32> {
        let n = self.shape().dims()[0];
        lower(&self.transpose(), Location::caller()).transpose().with_props(Props::triangle(false, false, n, n))
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

pub(crate) fn cholesky_solve(l: &Array<f32>, b: &Array<f32>, loc: &'static Location<'static>) -> Array<f32> {
    let (rhs, vector) = rhs_matrix(l, b);
    let y = lower_left_factor(l, &rhs, false, "cholesky_solve", loc);
    // Lᵀ X = Y: the transpose is upper, solved on its flip.
    let lt = l.transpose().flip(0).flip(1);
    let x = lower_left_factor(&lt, &y.flip(1), false, "cholesky_solve", loc).flip(1);
    if vector { x.reshape(b.shape()) } else { x }
}


