//! Householder QR of one tall panel (up to 32 columns) by a two level tall-skinny QR (TSQR).
//!
//! The panel's rows are cut into blocks of up to [`Plan::bs`] rows. One launch factors every
//! block by Householder reflections with the block held in registers of one workgroup (a thread
//! owns 32 rows of one column; per column two barriers: the norm and the reflector, then the
//! products of the reflector with the other columns). A second launch factors the stack of the
//! blocks' `R`s the same way. The compact WY form of the whole panel is then rebuilt from the
//! explicit `Q` (Ballard et al., *Reconstructing Householder vectors from TSQR*): with
//! `Q1 = Q_blocks [Q_stack I; 0]` the thin factor, `Q1 - S = V U` is a no-pivot LU for a sign
//! matrix `S` chosen on the fly, `V` is the unit lower trapezoid of the panel's reflectors,
//! `R = S R_tsqr` and `T = -U S V1⁻ᵀ`. The top `w x w` block of the LU is a small kernel and every
//! other row of `V` is a `w x w` product of the row of its block's `V`, so the last launch is
//! parallel over all rows of the panel. A panel of one block is the first launch alone.
//!
//! In contrast with the global-memory kernels of [`super::qr`] (one workgroup streams the whole panel
//! through one multiprocessor for every column) the panel is spread over the multiprocessors
//! and every barrier-to-barrier step works on registers and workgroup memory.

use crate::array::Array;
use crate::error::LumaError;
use crate::kernel::{uniform, view, KernelArg};
use crate::shape::Shape;

use super::launch;

/// Rows a thread holds of its column.
const ROWS: usize = 32;
/// Rows of a block the last launch handles per workgroup.
const CHUNK: usize = 128;

#[derive(encase::ShaderType)]
struct Params {
    m: u32,
    w: u32,
    k: u32,
    bs: u32,
    /// Whether the factor launch also writes the unit lower `V`.
    wv: u32,
}

/// How a panel of `m` rows is cut.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Plan {
    /// Row groups (of 32 threads) of a block factorisation launch.
    pub groups: usize,
    /// Rows of a block.
    pub bs: usize,
    /// Blocks.
    pub k: usize,
}

/// Whether and how a `m x w` single panel can be factored by [`panel`] on `ctx`.
pub(crate) fn plan(ctx: &crate::Context, m: usize, w: usize) -> Option<Plan> {
    if matches!(std::env::var("LUMA_QR_TSQR").as_deref(), Ok("0")) || w == 0 || w > 32 || m < w {
        return None;
    }
    let limits = &ctx.info().limits;
    let cap = limits.max_workgroup_invocations.min(limits.max_workgroup_size[0]) as usize / 32;
    let gmax = cap.min(16);
    if gmax < 2 || limits.max_storage_buffers < 12 {
        return None;
    }
    if m <= gmax * 32 {
        return Some(Plan { groups: m.div_ceil(32), bs: gmax * 32, k: 1 });
    }
    // Blocks of `groups` x 32 rows and a stack of `k` `R`s: the two factorisations cost about their rows.
    let (groups, _) = (2..=gmax)
        .rev()
        .filter(|&g| m.div_ceil(g * 32) * w <= gmax * 32)
        .map(|g| (g, g + (m.div_ceil(g * 32) * w).div_ceil(32)))
        .min_by_key(|&(_, cost)| cost)?;
    let bs = groups * 32;
    // The matrices of the other launches live in workgroup memory.
    let smem = 4 * 1024 * 5 + 1024;
    (limits.max_workgroup_storage_bytes as usize >= smem).then_some(Plan { groups, bs, k: m.div_ceil(bs) })
}

/// The loop over the columns of a block factorisation: `rows` are the unrolled thread rows.
fn each(f: impl Fn(usize) -> String) -> String {
    (0..ROWS).map(f).collect()
}

/// A block (or the stack of `R`s, `merge`) factored in registers: see the module docs. A thread
/// `(g, c)` holds rows `g + G r` of column `c`: the diagonal of the column being reduced only
/// reaches the first `32 / G` registers, every other row needs no masks. The Householder vectors
/// stay unscaled (`raw`) in their column until the end, when every lane multiplies its column by
/// the `1 / (alpha - beta)` it kept.
fn factor_source(groups: usize, merge: bool) -> String {
    let bs = groups * 32;
    let r0 = 32usize.div_ceil(groups);
    let ri = |r: usize| format!("ri{r}");
    let prologue = if merge {
        let rows = each(|r| {
            format!(
                "    {{ let l = {r}u * {groups}u + g; let ai = l / w; let rr = l - ai * w; if (c < w && ai < p.k && c >= rr) {{ let gi = ai * p.bs + rr; if (gi < p.m) {{ x{r} = ld(gi, c); }} }} }}\n"
            )
        });
        format!("    let nrows = p.k * w;\n    let out0 = 0u;\n{rows}")
    } else {
        let rows = each(|r| format!("        if ({r}u * {groups}u + g < nrows) {{ x{r} = ld(row0 + {r}u * {groups}u + g, c); }}\n"));
        format!("    let row0 = blk * p.bs;\n    let nrows = min(p.bs, p.m - row0);\n    let out0 = row0;\n    if (c < w) {{\n{rows}    }}\n")
    };
    let decl = each(|r| format!("    var x{r} = 0.0;\n"));
    let rows = (0..r0).map(|r| format!("    let ri{r} = {}u + g;\n", r * groups)).collect::<String>();
    let owner_masked = (0..r0)
        .map(|r| format!("            let y{r} = select(0.0, x{r}, {} > j);\n            sq{} = fma(y{r}, y{r}, sq{});\n            ap = ap + select(0.0, x{r}, {} == j);\n", ri(r), r % 2, r % 2, ri(r)))
        .collect::<String>();
    let owner_plain = (r0..ROWS).map(|r| format!("            sq{} = fma(x{r}, x{r}, sq{});\n", r % 2, r % 2)).collect::<String>();
    let publish = (0..ROWS / 4)
        .map(|q| format!("            vs[g * 8u + {q}u] = vec4<f32>(x{}, x{}, x{}, x{});\n", 4 * q, 4 * q + 1, 4 * q + 2, 4 * q + 3))
        .collect::<String>();
    let sums = {
        let mut acc = String::new();
        for a in 0..4 {
            acc += &format!("        var xs{a} = 0.0;\n");
        }
        for q in 0..groups {
            acc += &format!("        xs{} = xs{} + redq[{q}];\n", q % 4, q % 4);
        }
        acc + "        let xn2 = (xs0 + xs1) + (xs2 + xs3);\n"
    };
    let raws = (0..ROWS / 4).map(|q| format!("        let raw{q} = vs[g * 8u + {q}u];\n")).collect::<String>();
    let u = |r: usize| format!("select(select(0.0, 1.0, {} == j), raw{}[{}] * inv, {} > j)", ri(r), r / 4, r % 4, ri(r));
    let dots_masked = (0..r0).map(|r| format!("        dm = fma({}, x{r}, dm);\n", u(r))).collect::<String>();
    let dots_plain = (r0..ROWS).map(|r| format!("        d{} = fma(raw{}[{}], x{r}, d{});\n", r % 4, r / 4, r % 4, r % 4)).collect::<String>();
    let gsum = {
        let mut acc = String::new();
        for a in 0..4 {
            acc += &format!("        var ss{a} = 0.0;\n");
        }
        for q in 0..groups {
            acc += &format!("        ss{} = ss{} + part[{}u + c];\n", q % 4, q % 4, q * 32);
        }
        acc + "        let s = (ss0 + ss1) + (ss2 + ss3);\n"
    };
    let apply_masked = (0..r0).map(|r| format!("            x{r} = fma(-wc, {}, x{r});\n", u(r))).collect::<String>();
    let apply_plain = (r0..ROWS).map(|r| format!("            x{r} = fma(-wi, raw{}[{}], x{r});\n", r / 4, r % 4)).collect::<String>();
    let pivot = (0..r0).map(|r| format!("            x{r} = select(x{r}, beta, {} == j);\n", ri(r))).collect::<String>();
    let scale = (0..ROWS)
        .map(|r| if r < r0 { format!("    x{r} = select(x{r}, x{r} * myinv, {} > c);\n", ri(r)) } else { format!("    x{r} = x{r} * myinv;\n") })
        .collect::<String>();
    let tdecl = each(|r| format!("        var tr{r} = 0.0;\n"));
    let tcols = (0..ROWS)
        .map(|j| {
            let sum = (0..j).map(|q| format!("            sj = fma(tr{q}, dds[{}u], sj);\n", j * 32 + q)).collect::<String>();
            format!(
                "        if ({j}u < w) {{\n            var sj = 0.0;\n{sum}            let tj = tauv[{j}];\n            tr{j} = select(select(0.0, tj, ta == {j}u), -tj * sj, ta < {j}u);\n            tf[blk * w * w + ta * w + {j}u] = tr{j};\n            if (ta == {j}u) {{ tau[blk * w + ta] = tr{j}; }}\n        }}\n"
            )
        })
        .collect::<String>();
    let stores = each(|r| format!("    if ({r}u * {groups}u + g < nrows && c < w) {{ f[(out0 + {r}u * {groups}u + g) * w + c] = x{r}; }}\n"));
    let vstores = (0..ROWS)
        .map(|r| {
            let v = if r < r0 { format!("select(select(0.0, 1.0, {} == c), x{r}, {} > c)", ri(r), ri(r)) } else { format!("x{r}") };
            format!("        if ({r}u * {groups}u + g < nrows && c < w) {{ vo[(out0 + {r}u * {groups}u + g) * w + c] = {v}; }}\n")
        })
        .collect::<String>();
    let src = "struct Params { m: u32, w: u32, k: u32, bs: u32, wv: u32 }
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read_write> f: array<f32>;
@group(1) @binding(3) var<storage, read_write> tau: array<f32>;
@group(1) @binding(4) var<storage, read_write> tf: array<f32>;
@group(1) @binding(5) var<storage, read_write> vo: array<f32>;
@group(1) @binding(6) var<uniform> p: Params;
fn ld(i: u32, c: u32) -> f32 { return a[luma_view_idx3(av, c, i, 0u)]; }

var<workgroup> vs: array<vec4<f32>, @B4@>;
var<workgroup> redq: array<f32, @G@>;
var<workgroup> alphas: f32;
var<workgroup> part: array<f32, @GP@>;
var<workgroup> dds: array<f32, 1024>;
var<workgroup> tauv: array<f32, 32>;

@compute @workgroup_size(@B@)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let w = p.w;
    let blk = wid.x;
    let g = t / 32u;
    let c = t % 32u;
    var myinv = 0.0;
@ROWS@@DECL@@PROLOGUE@
    for (var j = 0u; j < w; j = j + 1u) {
        if (c == j) {
            var sq0 = 0.0;
            var sq1 = 0.0;
            var ap = 0.0;
@OWNERM@@OWNERP@            if (g == j % @G@u) { alphas = ap; }
@PUBLISH@            redq[g] = sq0 + sq1;
        }
        workgroupBarrier();
@SUMS@        let alpha = alphas;
        var beta = alpha;
        var tj = 0.0;
        var inv = 0.0;
        if (xn2 != 0.0) {
            let nrm = sqrt(fma(alpha, alpha, xn2));
            beta = select(nrm, -nrm, alpha >= 0.0);
            tj = (beta - alpha) / beta;
            inv = 1.0 / (alpha - beta);
        }
@RAWS@        var d0 = 0.0;
        var d1 = 0.0;
        var d2 = 0.0;
        var d3 = 0.0;
        var dm = 0.0;
@DOTSP@@DOTSM@        part[g * 32u + c] = fma((d0 + d1) + (d2 + d3), inv, dm);
        workgroupBarrier();
@GSUM@        if (c > j) {
            let wc = tj * s;
            let wi = wc * inv;
@APPLYM@@APPLYP@        }
        if (c == j) {
            myinv = inv;
@PIVOT@        }
        if (g == 0u) { dds[j * 32u + c] = s * myinv; }
        if (t == 0u) { tauv[j] = tj; }
    }
    workgroupBarrier();
@SCALE@    if (t < 32u && t < w) {
        let ta = t;
@TDECL@@TCOLS@    }
@STORES@    if (p.wv != 0u) {
@VSTORES@    }
}
";
    src.replace("@B4@", &(bs / 4).to_string())
        .replace("@B@", &bs.to_string())
        .replace("@GP@", &(groups * 32).to_string())
        .replace("@G@", &groups.to_string())
        .replace("@ROWS@", &rows)
        .replace("@DECL@", &decl)
        .replace("@PROLOGUE@", &prologue)
        .replace("@OWNERM@", &owner_masked)
        .replace("@OWNERP@", &owner_plain)
        .replace("@PUBLISH@", &publish)
        .replace("@SUMS@", &sums)
        .replace("@RAWS@", &raws)
        .replace("@DOTSP@", &dots_plain)
        .replace("@DOTSM@", &dots_masked)
        .replace("@GSUM@", &gsum)
        .replace("@APPLYM@", &apply_masked)
        .replace("@APPLYP@", &apply_plain)
        .replace("@PIVOT@", &pivot)
        .replace("@SCALE@", &scale)
        .replace("@TDECL@", &tdecl)
        .replace("@TCOLS@", &tcols)
        .replace("@STORES@", &stores)
        .replace("@VSTORES@", &vstores)
}

/// `dst[a][c] = fin(Σ_q ea(a, q) eb(q, c))` for 32 x 32 matrices in workgroup memory by `threads`
/// threads (the templates take `@A@`, `@Q@` and `@C@`; `fin` also `@ACC@`).
fn mm(threads: usize, dst: &str, ea: &str, eb: &str, fin: &str) -> String {
    let nr = threads / 32;
    let per = 32 / nr;
    let at = |s: &str, a: String, q: &str, c: &str| s.replace("@A@", &a).replace("@Q@", q).replace("@C@", c);
    let accs = (0..per).map(|e| format!("        var acc{e} = 0.0;\n")).collect::<String>();
    let body = (0..per).map(|e| format!("            acc{e} = fma({}, bq, acc{e});\n", at(ea, format!("(r0 + {}u)", e * nr), "q", "cc"))).collect::<String>();
    let stores = (0..per)
        .map(|e| {
            let a = format!("(r0 + {}u)", e * nr);
            format!("        {dst}[{a} * 32u + cc] = {};\n", at(fin, a.clone(), "q", "cc").replace("@ACC@", &format!("acc{e}")))
        })
        .collect::<String>();
    format!(
        "    {{
        let cc = t % 32u;
        let r0 = t / 32u;
{accs}        for (var q = 0u; q < 32u; q = q + 1u) {{
            let bq = {};
{body}        }}
        workgroupBarrier();
{stores}        workgroupBarrier();
    }}
",
        at(eb, "0u".to_string(), "q", "cc")
    )
}

/// Issues the loads of the `w x w` matrix `expr(a, c)` (where `cond`) for the elements `idx = t + THREADS e` of a
/// 32 x 32 matrix into registers `name0..`; [`store`] puts them in workgroup memory later.
fn fetch(threads: usize, name: &str, cond: &str, expr: &str) -> String {
    (0..1024 / threads)
        .map(|e| {
            format!(
                "    var {name}{e} = 0.0;\n    {{\n        let idx = t + {}u;\n        let a = idx / 32u;\n        let c = idx % 32u;\n        if (a < w && c < w && {cond}) {{ {name}{e} = {expr}; }}\n    }}\n",
                e * threads
            )
        })
        .collect()
}

/// Stores the registers of [`fetch`] in `dst` and waits.
fn store(threads: usize, dst: &str, name: &str) -> String {
    let mut s: String = (0..1024 / threads).map(|e| format!("    {dst}[t + {}u] = {name}{e};\n", e * threads)).collect();
    s += "    workgroupBarrier();\n";
    s
}

/// The entry `(row, c)` of the unit lower trapezoid of a packed factor `src` (rows of stride `w`), stored at row `at`.
fn unit_lower(src: &str, at: &str, row: &str) -> String {
    format!("select(select(0.0, 1.0, {row} == c), {src}[({at}) * w + c], {row} > c)")
}

/// One workgroup: from the stack's factor `f1`, `t1` and the first block's factor `f0`, `t0` the
/// top block of the panel's factor (written to `f`), its `T` and `tau`, and for the launch that follows
/// `aux`: `Y = T' V1'ᵀ` (at 0) and the inverse of `U` (at 1024), both 32 x 32 with a row stride of 32.
fn recon_source() -> String {
    const THREADS: usize = 256;
    let mut s = String::from(
        "struct Params { m: u32, w: u32, k: u32, bs: u32, wv: u32 }
@group(1) @binding(0) var<storage, read_write> f1: array<f32>;
@group(1) @binding(1) var<storage, read_write> t1: array<f32>;
@group(1) @binding(2) var<storage, read_write> f0: array<f32>;
@group(1) @binding(3) var<storage, read_write> t0: array<f32>;
@group(1) @binding(4) var<storage, read_write> f: array<f32>;
@group(1) @binding(5) var<storage, read_write> tau: array<f32>;
@group(1) @binding(6) var<storage, read_write> tf: array<f32>;
@group(1) @binding(7) var<storage, read_write> aux: array<f32>;
@group(1) @binding(8) var<storage, read_write> vo: array<f32>;
@group(1) @binding(9) var<uniform> p: Params;
var<workgroup> m0: array<f32, 1024>;
var<workgroup> m1: array<f32, 1024>;
var<workgroup> m2: array<f32, 1024>;
var<workgroup> m3: array<f32, 1024>;
var<workgroup> sg: array<f32, 32>;
var<workgroup> pr: array<f32, 64>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    let w = p.w;
",
    );
    s += &fetch(THREADS, "ft", "true", "t1[a * w + c]");
    s += &fetch(THREADS, "fv", "true", &unit_lower("f1", "a", "a"));
    s += &fetch(THREADS, "gv", "true", &unit_lower("f0", "a", "a"));
    s += &fetch(THREADS, "gt", "true", "t0[a * w + c]");
    // m0 = T', m1 = V1' (unit lower).
    s += &store(THREADS, "m0", "ft");
    s += &store(THREADS, "m1", "fv");
    // Y = T' V1'ᵀ (kept in m2 and in aux).
    s += &mm(THREADS, "m2", "m0[@A@ * 32u + @Q@]", "m1[@C@ * 32u + @Q@]", "@ACC@");
    s += "    for (var idx = t; idx < 1024u; idx = idx + 256u) { aux[idx] = m2[idx]; }\n";
    // X0 = I - V1' Y (m3).
    s += &mm(THREADS, "m3", "m1[@A@ * 32u + @Q@]", "m2[@Q@ * 32u + @C@]", "select(0.0, 1.0, @A@ == @C@) - @ACC@");
    // The first block's top: m0 = V01 (unit lower), m1 = T0.
    s += &store(THREADS, "m0", "gv");
    s += &store(THREADS, "m1", "gt");
    // P = T0 V01ᵀ (m2), M = I - V01 P (m1), Q1top = M X0 (m0).
    s += &mm(THREADS, "m2", "m1[@A@ * 32u + @Q@]", "m0[@C@ * 32u + @Q@]", "@ACC@");
    s += &mm(THREADS, "m1", "m0[@A@ * 32u + @Q@]", "m2[@Q@ * 32u + @C@]", "select(0.0, 1.0, @A@ == @C@) - @ACC@");
    s += &mm(THREADS, "m0", "m1[@A@ * 32u + @Q@]", "m3[@Q@ * 32u + @C@]", "@ACC@");
    // No pivot LU of Q1top - S, S chosen on the fly: a thread per row, the pivot row through `pr`.
    s += "    var r0 = 0.0;\n";
    s += &(1..32).map(|k| format!("    var r{k} = 0.0;\n")).collect::<String>();
    s += "    if (t < 32u) {\n";
    s += &(0..32).map(|k| format!("        r{k} = m0[t * 32u + {k}u];\n")).collect::<String>();
    s += "    }\n";
    for j in 0..32 {
        let par = (j % 2) * 32;
        let publish = (j + 1..32).map(|k| format!("            pr[{}u] = r{k};\n", par + k)).collect::<String>();
        let update = (j + 1..32).map(|k| format!("            r{k} = r{k} - l * pr[{}u];\n", par + k)).collect::<String>();
        s += &format!(
            "    if ({j}u < w) {{
        if (t == {j}u) {{
            let sj = select(1.0, -1.0, r{j} >= 0.0);
            let pv = r{j} - sj;
            pr[{par}u + {j}u] = pv;
            sg[{j}u] = sj;
            r{j} = pv;
{publish}        }}
        workgroupBarrier();
        if (t > {j}u && t < w) {{
            let l = r{j} / pr[{par}u + {j}u];
            r{j} = l;
{update}        }}
    }}
"
        );
    }
    s += "    if (t < w) {\n";
    s += &(0..32).map(|k| format!("        m0[t * 32u + {k}u] = r{k};\n")).collect::<String>();
    s += "    }\n    workgroupBarrier();\n";
    // T = -U S L⁻ᵀ by rows (a warp), U⁻¹ by columns (another).
    let trow = (0..32)
        .map(|k| {
            let sub = (0..k).map(|q| format!("            acc = acc - m0[{}u] * tw{q};\n", k * 32 + q)).collect::<String>();
            format!(
                "        if ({k}u < w) {{\n            var acc = -select(0.0, m0[a * 32u + {k}u], {k}u >= a) * sg[{k}];\n{sub}            tw{k} = acc;\n            tf[a * w + {k}u] = acc;\n            if (a == {k}u) {{ tau[a] = acc; }}\n        }}\n"
            )
        })
        .collect::<String>();
    let tdecl = (0..32).map(|k| format!("        var tw{k} = 0.0;\n")).collect::<String>();
    s += &format!("    if (t < 32u && t < w) {{\n        let a = t;\n{tdecl}{trow}    }}\n");
    let zrow = (0..32)
        .rev()
        .map(|k| {
            let sub = (k + 1..32).map(|q| format!("            acc = acc - m0[{}u] * z{q};\n", k * 32 + q)).collect::<String>();
            format!(
                "        if ({k}u < w) {{\n            var acc = select(0.0, 1.0, cj == {k}u);\n{sub}            z{k} = acc / m0[{}u];\n            aux[1024u + {}u * 32u + cj] = z{k};\n        }}\n",
                k * 33,
                k
            )
        })
        .collect::<String>();
    let zdecl = (0..32).map(|k| format!("        var z{k} = 0.0;\n")).collect::<String>();
    s += &format!("    if (t >= 32u && t < 64u && t - 32u < w) {{\n        let cj = t - 32u;\n{zdecl}{zrow}    }}\n");
    // The top block of the panel's factor: R = S R_tsqr above and on the diagonal, L below.
    s += "    for (var idx = t; idx < w * w; idx = idx + 256u) {
        let a = idx / w;
        let c = idx - a * w;
        f[idx] = select(m0[a * 32u + c], sg[a] * f1[idx], a <= c);
        vo[idx] = select(select(0.0, 1.0, a == c), m0[a * 32u + c], a > c);
    }
}
";
    s
}

/// The rows of the panel's `V` below the top block: row `l` of block `i` is
/// `V_i[l] N_i` (`+ X_i[l] U⁻¹` for the first `w` rows of a block but the first), with
/// `X_i = [I; 0]_i - V'_i Y` the block's piece of the stack's `Q`, `W_i = T_i V_{i,1}ᵀ X_i`,
/// `N_i = -W_i U⁻¹`.
fn down_source() -> String {
    const THREADS: usize = CHUNK;
    let mut s = String::from(
        "struct Params { m: u32, w: u32, k: u32, bs: u32, wv: u32 }
@group(1) @binding(0) var<storage, read_write> f0: array<f32>;
@group(1) @binding(1) var<storage, read_write> t0: array<f32>;
@group(1) @binding(2) var<storage, read_write> f1: array<f32>;
@group(1) @binding(3) var<storage, read_write> aux: array<f32>;
@group(1) @binding(4) var<storage, read_write> f: array<f32>;
@group(1) @binding(5) var<storage, read_write> vo: array<f32>;
@group(1) @binding(6) var<uniform> p: Params;
var<workgroup> m0: array<f32, 1024>;
var<workgroup> m1: array<f32, 1024>;
var<workgroup> m2: array<f32, 1024>;
var<workgroup> m3: array<f32, 1024>;
var<workgroup> nv: array<vec4<f32>, 256>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let w = p.w;
    let cpb = (p.bs + 127u) / 128u;
    let blk = wid.x / cpb;
    let l0 = (wid.x - blk * cpb) * 128u;
    let l = l0 + t;
    let gr = blk * p.bs + l;
    let mine = l < p.bs && gr < p.m && gr >= w;
    let top = l < w;
",
    );
    s += &fetch(THREADS, "fv", "true", &unit_lower("f1", "(blk * w + a)", "(blk * w + a)"));
    s += &fetch(THREADS, "fy", "true", "aux[idx]");
    s += &fetch(THREADS, "gv", "blk * p.bs + a < p.m", &unit_lower("f0", "(blk * p.bs + a)", "a"));
    s += &fetch(THREADS, "gt", "true", "t0[blk * w * w + a * w + c]");
    s += &fetch(THREADS, "gu", "true", "aux[1024u + idx]");
    // The thread's row of the block's `V`.
    s += &(0..32).map(|q| format!("    var v{q} = 0.0;\n")).collect::<String>();
    s += "    if (mine) {\n";
    s += &(0..32)
        .map(|q| {
            let masked = format!("if (top) {{ v{q} = select(select(0.0, 1.0, {q}u == l), v{q}, {q}u < l); }}");
            format!("        if ({q}u < w) {{ v{q} = f0[gr * w + {q}u]; {masked} }}\n")
        })
        .collect::<String>();
    s += "    }\n";
    // X_i = [I; 0]_i - V'_i Y.
    s += &store(THREADS, "m1", "fv");
    s += &store(THREADS, "m2", "fy");
    s += &mm(THREADS, "m0", "m1[@A@ * 32u + @Q@]", "m2[@Q@ * 32u + @C@]", "select(0.0, 1.0, @A@ == @C@ && blk == 0u) - @ACC@");
    // Z = V_{i,1}ᵀ X_i, W = T_i Z.
    s += &store(THREADS, "m1", "gv");
    s += &mm(THREADS, "m2", "m1[@Q@ * 32u + @A@]", "m0[@Q@ * 32u + @C@]", "@ACC@");
    s += &store(THREADS, "m1", "gt");
    s += &mm(THREADS, "m3", "m1[@A@ * 32u + @Q@]", "m2[@Q@ * 32u + @C@]", "@ACC@");
    // N = -W U⁻¹ (m2), X_i U⁻¹ (m3) where the rows of the block's top are rewritten.
    s += &store(THREADS, "m1", "gu");
    s += &mm(THREADS, "m2", "m3[@A@ * 32u + @Q@]", "m1[@Q@ * 32u + @C@]", "-@ACC@");
    s += "    if (blk > 0u && l0 == 0u) {\n";
    s += &mm(THREADS, "m3", "m0[@A@ * 32u + @Q@]", "m1[@Q@ * 32u + @C@]", "@ACC@");
    s += "    }\n";
    s += "    for (var q4 = t; q4 < 256u; q4 = q4 + 128u) {
        nv[q4] = vec4<f32>(m2[4u * q4], m2[4u * q4 + 1u], m2[4u * q4 + 2u], m2[4u * q4 + 3u]);
    }
    workgroupBarrier();
    if (mine) {
";
    s += &(0..32).map(|c| format!("        var o{c} = select(0.0, m3[min(l, 31u) * 32u + {c}u], top);\n")).collect::<String>();
    for q in 0..32 {
        s += "        {\n";
        for c4 in 0..8 {
            s += &format!("            let n{c4} = nv[{}u];\n", q * 8 + c4);
            for e in 0..4 {
                let c = 4 * c4 + e;
                s += &format!("            o{c} = fma(v{q}, n{c4}[{e}], o{c});\n");
            }
        }
        s += "        }\n";
    }
    s += &(0..32).map(|c| format!("        if ({c}u < w) {{ f[gr * w + {c}u] = o{c}; vo[gr * w + {c}u] = o{c}; }}\n")).collect::<String>();
    s += "    }\n}\n";
    s
}

/// `a` (`[w, m]`, `w <= 32`) factored into packed `qr` (`[w, m]`), `tau` (`[w, 1]`), `t` (`[w, w]`)
/// and the unit lower trapezoid `v` of the reflectors (`[w, m]`, zeros above the diagonal).
pub(crate) fn panel(a: &Array<f32>, plan: Plan) -> Result<[Array<f32>; 4], LumaError> {
    let ctx = a.context();
    let [w, m, ..] = a.shape().dims();
    let Plan { groups, bs, k } = plan;
    let mut params = Params { m: m as u32, w: w as u32, k: k as u32, bs: bs as u32, wv: 0 };
    let mut f = Array::<f32>::alloc(ctx, a.shape())?;
    let mut v = Array::<f32>::alloc(ctx, a.shape())?;
    let mut tau = Array::<f32>::alloc(ctx, Shape::from([w, 1, 1]))?;
    let mut t = Array::<f32>::alloc(ctx, Shape::from([w, w, 1]))?;
    if k == 1 {
        params.wv = 1;
        let args: [KernelArg<'_>; 6] = [view(a), (&mut f).into(), (&mut tau).into(), (&mut t).into(), (&mut v).into(), uniform(&params)];
        launch(ctx, &format!("tsqr_factor/0/{groups}"), "tsqr_leaf", || factor_source(groups, false), [1, 1, 1], &args)?;
        return Ok([f, v, tau, t]);
    }
    let mut nov = Array::<f32>::alloc(ctx, Shape::from([1, 1, 1]))?;
    let mut f0 = Array::<f32>::alloc(ctx, Shape::from([w, m, 1]))?;
    let mut tau0 = Array::<f32>::alloc(ctx, Shape::from([w, k, 1]))?;
    let mut t0 = Array::<f32>::alloc(ctx, Shape::from([w, w * k, 1]))?;
    let args: [KernelArg<'_>; 6] = [view(a), (&mut f0).into(), (&mut tau0).into(), (&mut t0).into(), (&mut nov).into(), uniform(&params)];
    launch(ctx, &format!("tsqr_factor/0/{groups}"), "tsqr_leaf", || factor_source(groups, false), [k as u32, 1, 1], &args)?;
    let g2 = (k * w).div_ceil(32);
    let mut f1 = Array::<f32>::alloc(ctx, Shape::from([w, k * w, 1]))?;
    let mut tau1 = Array::<f32>::alloc(ctx, Shape::from([w, 1, 1]))?;
    let mut t1 = Array::<f32>::alloc(ctx, Shape::from([w, w, 1]))?;
    let args: [KernelArg<'_>; 6] = [view(&f0), (&mut f1).into(), (&mut tau1).into(), (&mut t1).into(), (&mut nov).into(), uniform(&params)];
    launch(ctx, &format!("tsqr_factor/1/{g2}"), "tsqr_merge", || factor_source(g2, true), [1, 1, 1], &args)?;
    let mut aux = Array::<f32>::alloc(ctx, Shape::from([32, 64, 1]))?;
    let args: [KernelArg<'_>; 10] = [
        (&mut f1).into(),
        (&mut t1).into(),
        (&mut f0).into(),
        (&mut t0).into(),
        (&mut f).into(),
        (&mut tau).into(),
        (&mut t).into(),
        (&mut aux).into(),
        (&mut v).into(),
        uniform(&params),
    ];
    launch(ctx, "tsqr_recon", "tsqr_recon", recon_source, [1, 1, 1], &args)?;
    let args: [KernelArg<'_>; 7] = [(&mut f0).into(), (&mut t0).into(), (&mut f1).into(), (&mut aux).into(), (&mut f).into(), (&mut v).into(), uniform(&params)];
    launch(ctx, "tsqr_down", "tsqr_down", down_source, [(k * bs.div_ceil(CHUNK)) as u32, 1, 1], &args)?;
    Ok([f, v, tau, t])
}
