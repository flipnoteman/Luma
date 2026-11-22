//! LU factorisation with partial pivoting, general solves, inverse and determinant.
//!
//! `P A = L U` with `L` unit lower and `U` upper triangular, packed in one matrix; `P` is the
//! permutation vector `perm` (row `i` of `P A` is row `perm[i]` of `A`).
//!
//! Matrices up to 64 rows (a batch of them in one launch) are factored by a workgroup each in
//! workgroup memory; a batch of tall panels of up to 32 columns by one workgroup each in place;
//! anything bigger by recursion on the columns (Toledo): factor the left half, apply its row
//! swaps to the right half, `U12 = L11⁻¹ A12` (a triangular solve), update `A22 - L21 U12` with
//! one [`matmul_f32`] and factor that.
//!
//! The leaves of that recursion, tall panels of up to 32 columns of one matrix, are factored by a
//! tournament (CALU), which puts every multiprocessor to work where the panel kernel streams the
//! whole panel through one: workgroups pick `w` pivot rows in chunks of up to 512 rows (a row in
//! the registers of an invocation), one workgroup plays the winners of the chunks off against one
//! another until `w` are left, and every row of the panel gets its multipliers `A21 U11⁻¹` in
//! parallel. A panel of one chunk (up to 512 rows) is pivoted exactly as partial pivoting does;
//! of several, by the tournament, whose multipliers can exceed 1 (mildly: a growth bound a little
//! worse than partial pivoting's, and the same results on every backend).

use std::panic::Location;

use super::tri::{lower_left, rhs_matrix};
use super::{block, fail, fence, gather_rows, launch, nth, per_matrix, stack_batch, Dims};
use crate::array::Array;
use crate::error::LumaError;
use crate::kernel::{uniform, view, KernelArg};
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
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read_write> f: array<f32>;
@group(1) @binding(3) var<storage, read_write> perm: array<u32>;
@group(1) @binding(4) var<uniform> p: Params;
fn ld(mi: u32, i: u32, c: u32) -> f32 {{ return a[luma_view_idx3(av, c, i, mi)]; }}

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
        s[i * {pitch}u + idx - i * w] = ld(mi, i, idx - i * w);
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

/// Rows a panel kernel's loops take per step, their loads issued together.
pub(crate) const UNROLL: usize = 8;

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
/// one), [`UNROLL`] rows per step with all their loads issued first, in which the lanes of the next column
/// also keep the largest entry they have updated: the next pivot comes from a reduction over the
/// row groups, no separate scan of the column. (An RTX 3090 spent 60 us per column on this loop
/// with one row per step and 256 invocations: 8 loads in flight per row group.)
fn panel_source(fence: &str, threads: usize) -> String {
    let t = threads;
    let g = threads / 32;
    format!(
        "struct Params {{ m: u32, w: u32, batch: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read_write> f: array<f32>;
@group(1) @binding(3) var<storage, read_write> perm: array<u32>;
@group(1) @binding(4) var<uniform> p: Params;
fn ld(mi: u32, i: u32, c: u32) -> f32 {{ return a[luma_view_idx3(av, c, i, mi)]; }}

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
        for (; i + {last}u < m; i = i + {step}u) {{
            let r0 = base + i * w + c;
{copy_loads}{copy_stores}            if (c == 0u) {{
{copy_cands}            }}
        }}
        for (; i < m; i = i + {g1}u) {{
            let v = ld(mi, i, c);
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
        var prow = 0.0;
        if (c < w) {{ prow = f[base + j * w + c]; }}
        let act = c > j && c < w;
        best = -1.0;
        bi = j + 1u;
        var i = j + 1u + g;
        for (; i + {last}u < m; i = i + {step}u) {{
            let r0 = base + i * w;
{upd_l}            if (act) {{
{upd_x}{upd_store}                if (c == j + 1u) {{
{upd_cands}                }}
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
        last = (UNROLL - 1) * g,
        step = UNROLL * g,
        copy_loads = (0..UNROLL).map(|k| format!("            let v{k} = ld(mi, i + {}u, c);\n", k * g)).collect::<String>(),
        copy_stores = (0..UNROLL).map(|k| format!("            f[r0 + {}u * w] = v{k};\n", k * g)).collect::<String>(),
        copy_cands = (0..UNROLL).map(|k| format!("                if (abs(v{k}) > best) {{ best = abs(v{k}); bi = i + {}u; }}\n", k * g)).collect::<String>(),
        upd_l = (0..UNROLL).map(|k| format!("            let l{k} = f[r0 + {}u * w + j] * rpiv;\n", k * g)).collect::<String>(),
        upd_x = (0..UNROLL).map(|k| format!("                let x{k} = fma(-l{k}, prow, f[r0 + {}u * w + c]);\n", k * g)).collect::<String>(),
        upd_store = (0..UNROLL).map(|k| format!("                f[r0 + {}u * w + c] = x{k};\n", k * g)).collect::<String>(),
        upd_cands = (0..UNROLL).map(|k| format!("                    if (abs(x{k}) > best) {{ best = abs(x{k}); bi = i + {}u; }}\n", k * g)).collect::<String>(),
    )
}

/// Columns a tournament panel has at most (the pitch of its workgroup-memory rows is one more).
const CALU_W: usize = 32;
/// Pitch of a row in workgroup memory: odd, so that rows at a stride are in different banks.
const CALU_P: usize = CALU_W | 1;

/// Rows of the chunks of the first round of the tournament and of the groups of the later ones:
/// an invocation holds a row (up to 32 values in registers) of the chunk it factors, so a chunk is as
/// many rows as the device has invocations in a workgroup, at most 512 and 1024. 0 where the device
/// has fewer than 64 (the old panel kernel then).
fn calu_rows(ctx: &crate::Context) -> (usize, usize) {
    let limits = &ctx.info().limits;
    let cap = (limits.max_workgroup_invocations.min(limits.max_workgroup_size[0]) as usize).checked_ilog2().map_or(0, |l| 1 << l);
    if cap < 64 { (0, 0) } else { (512.min(cap), 1024.min(cap)) }
}

#[derive(encase::ShaderType)]
struct CaluParams {
    m: u32,
    w: u32,
    /// Chunks of the first round.
    g: u32,
    /// Rows of a chunk (the last may have fewer).
    rc: u32,
}

/// Parts a chunk of `rows` keys is reduced in first.
fn calu_parts(rows: usize) -> usize {
    if rows <= 256 { 16 } else { 32 }
}

/// The largest of `n` keys and its index as statements declaring `bv` and `bi`: `key` and `index`
/// are the expressions of entry `{k}`. The loads come first and the comparisons are a tree of
/// branch-free steps (pairs, then pairs of pairs), so that the chain of dependent instructions is
/// short. Of equal keys the lowest index wins (`by_index`) or the left one, which has a lower
/// index than the right one.
fn reduce_text(key: &str, index: &str, n: usize, bv: &str, bi: &str, by_index: bool, indent: &str) -> String {
    let mut text = String::new();
    for k in 0..n {
        text += &format!("{indent}let v0_{k} = {};\n", key.replace("{k}", &k.to_string()));
        text += &format!("{indent}let j0_{k} = {};\n", index.replace("{k}", &k.to_string()));
    }
    let (mut level, mut count) = (0, n);
    while count > 1 {
        let next = count.div_ceil(2);
        for k in 0..next {
            let (l, r) = (2 * k, 2 * k + 1);
            let (lv, nl) = (level, level + 1);
            if r < count {
                let better = if by_index { format!("(v{lv}_{r} > v{lv}_{l} || (v{lv}_{r} == v{lv}_{l} && j{lv}_{r} < j{lv}_{l}))") } else { format!("v{lv}_{r} > v{lv}_{l}") };
                text += &format!("{indent}let j{nl}_{k} = select(j{lv}_{l}, j{lv}_{r}, {better});\n{indent}let v{nl}_{k} = max(v{lv}_{l}, v{lv}_{r});\n");
            } else {
                text += &format!("{indent}let j{nl}_{k} = j{lv}_{l};\n{indent}let v{nl}_{k} = v{lv}_{l};\n");
            }
        }
        level += 1;
        count = next;
    }
    text += &format!("{indent}let {bv} = v{level}_0;\n{indent}let {bi} = j{level}_0;\n");
    text
}

/// What the two tournament kernels share: partial-pivoting LU of `n <= rows` rows of `w <= wp`
/// columns, a row in the registers `x0 ..` of an invocation, the rows of the invocations `>= n`
/// absent. A pivot row is not swapped to the top: the invocation that holds it is done and copies
/// its row to `lu` (row `j` of the packed `L` and `U` of the winners) and its place in the panel
/// to `piv[j]`; the others eliminate with the row from `lu`. The next pivot comes from the keys
/// the invocations have just updated: they are in `cv`, one invocation per part reduces its part
/// and everybody reduces the parts (three barriers per column, no loop over the row's columns).
fn calu_declarations(rows: usize, wp: usize, sg: bool) -> String {
    let parts = if sg { rows / 32 } else { calu_parts(rows) };
    let cv = if sg { String::new() } else { format!("var<workgroup> cv: array<u32, {rows}>;\n") };
    format!(
        "const R: u32 = {rows}u;
const NP: u32 = {parts}u;
const PER: u32 = {per}u;
{cv}var<workgroup> s1v: array<u32, {parts}>;
var<workgroup> s1i: array<u32, {parts}>;
var<workgroup> lu: array<f32, {size}>;
var<workgroup> piv: array<u32, {wp}>;
var<workgroup> pvs: u32;
var<workgroup> wins: array<u32, {wp}>;
var<workgroup> lost: array<u32, {wp}>;
var<workgroup> rank: array<u32, {wp}>;
var<workgroup> slots: array<u32, {wp}>;
var<workgroup> lbs: array<u32, {wp}>;

// The key of a magnitude: larger is bigger (NaN above all), 0 is for a row that is done.
fn key(x: f32) -> u32 {{ return (bitcast<u32>(x) & 0x7fffffffu) + 1u; }}

{stage1}",
        per = rows / parts,
        size = wp * wp,
        stage1 = if sg {
            String::new()
        } else {
            format!(
                "fn stage1(t: u32) {{
    if (t < NP) {{
{}        s1v[t] = bv;
        s1i[t] = bi;
    }}
}}
",
                reduce_text(&format!("cv[t + {{k}}u * {parts}u]"), &format!("t + {{k}}u * {parts}u"), rows / parts, "bv", "bi", false, "        ")
            )
        }
    )
}

/// The LU of the rows in `x0 ..` as statements for the body of a kernel, not a function (the CPU
/// backend splits its phases at the barriers of the entry function only), unrolled over the
/// columns: `row` is the number of the invocation's row in the panel, `n` and `w` are in scope.
///
/// Where the device has subgroups of 32 (`sg`; `lane` is the invocation's place in its subgroup)
/// every subgroup reduces the keys of its rows with subgroup operations, writes its best to
/// `s1v` and `s1i`, and after a barrier every subgroup reduces those the same way, so everybody
/// knows the pivot: two barriers a column. Without, the keys go to `cv`, 16 or 32 invocations
/// reduce a part each, the first subgroup alone reduces the parts: four.
fn calu_lu(wp: usize, parts: usize, row: &str, sg: bool) -> String {
    let mut text = String::new();
    if sg {
        text += "var done = t >= n;
{
    let k0 = select(key(x0), 0u, done);
    let m = subgroupMax(k0);
    if (lane == 0u) { s1v[t / 32u] = m; }
    if (lane == firstTrailingBit(subgroupBallot(k0 == m).x)) { s1i[t / 32u] = t; }
}
workgroupBarrier();
";
    } else {
        text += "var done = t >= n;\ncv[t] = select(key(x0), 0u, done);\nworkgroupBarrier();\nstage1(t);\nworkgroupBarrier();\n";
    }
    for j in 0..wp {
        if sg {
            text += &format!(
                "if (w > {j}u) {{
    var kv = 0u;
    var ki = 0xffffffffu;
    if (lane < NP) {{ kv = s1v[lane]; ki = s1i[lane]; }}
    let m2 = subgroupMax(kv);
    let pv = subgroupShuffle(ki, firstTrailingBit(subgroupBallot(kv == m2).x));
    if (t == pv) {{
        done = true;
        piv[{j}] = {row};
"
            );
        } else {
            // The first subgroup alone picks the pivot from the parts (every warp doing it is most
            // of the instructions of a column), the invocation that holds the row copies it, the
            // others eliminate, the parts are reduced again.
            text += &format!(
                "if (w > {j}u) {{
    if (t < 32u) {{
{reduce}        if (t == 0u) {{ pvs = bi; }}
    }}
    workgroupBarrier();
    if (t == pvs) {{
        done = true;
        piv[{j}] = {row};
",
                reduce = reduce_text("s1v[{k}]", "s1i[{k}]", parts, "bv", "bi", true, "        ")
            );
        }
        for c in 0..wp {
            text += &format!("        lu[{}] = x{c};\n", j * wp + c);
        }
        text += &format!(
            "    }}
    workgroupBarrier();
    if (!done) {{
        let pr = lu[{d}];
        let l = select(0.0, x{j} / pr, pr != 0.0);
        x{j} = l;
",
            d = j * wp + j
        );
        for c in j + 1..wp {
            text += &format!("        x{c} = fma(-l, lu[{}], x{c});\n", j * wp + c);
        }
        let next = if j + 1 < wp { format!("x{}", j + 1) } else { "0.0".to_string() };
        if sg {
            text += &format!(
                "    }}
    {{
        let kk = select(key({next}), 0u, done);
        let m = subgroupMax(kk);
        if (lane == 0u) {{ s1v[t / 32u] = m; }}
        if (lane == firstTrailingBit(subgroupBallot(kk == m).x)) {{ s1i[t / 32u] = t; }}
    }}
    workgroupBarrier();
}}
"
            );
        } else {
            text += &format!(
                "    }}
    cv[t] = select(key({next}), 0u, done);
    workgroupBarrier();
    stage1(t);
    workgroupBarrier();
}}
"
            );
        }
    }
    text
}

/// Statements that declare the registers of a row and load row `row` of the panel into them.
fn calu_load(wp: usize, row: &str) -> String {
    let mut text = String::new();
    for c in 0..wp {
        text += &format!("var x{c} = 0.0;\n");
    }
    text += &format!("if (t < n) {{\n    let row = {row};\n");
    for c in 0..wp {
        text += &format!("    if ({c}u < w) {{ x{c} = ld(row, {c}u); }}\n");
    }
    text += "}\n";
    text
}

/// First round of the tournament: workgroup `g` factors the rows `g * rc ..` of the panel (at least
/// `w` of them) and writes the panel rows that won, in pivot order.
fn calu_chunks_source(rows: usize, wp: usize, fence: &str, sg: bool, single: bool) -> String {
    // A panel of one chunk is done: the first round writes the factors and the permutation.
    let (bindings, end) = if single {
        (
            "@group(1) @binding(2) var<storage, read_write> f: array<f32>;
@group(1) @binding(3) var<storage, read_write> perm: array<u32>;
@group(1) @binding(4) var<uniform> p: Params;",
            calu_epilogue(wp, fence, single),
        )
    } else {
        (
            "@group(1) @binding(2) var<storage, read_write> win: array<u32>;
@group(1) @binding(3) var<uniform> p: Params;",
            "    if (t < w) { win[g * w + t] = piv[t]; }\n".to_string(),
        )
    };
    format!(
        "struct Params {{ m: u32, w: u32, g: u32, rc: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
{bindings}
fn ld(i: u32, c: u32) -> f32 {{ return a[luma_view_idx3(av, c, i, 0u)]; }}

{decl}
@compute @workgroup_size({rows})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32{lane_arg}) {{
    let g = wid.x;
    let w = p.w;
    let first = g * p.rc;
    let n = min(p.rc, p.m - first);
{load}{lu}{end}}}
",
        decl = calu_declarations(rows, wp, sg),
        lane_arg = if sg { ", @builtin(subgroup_invocation_id) lane: u32" } else { "" },
        load = calu_load(wp, "first + t"),
        lu = calu_lu(wp, calu_parts(rows), "first + t", sg)
    )
}

/// The end of a tournament panel, after its last LU: the factors of the winners are the top of `f`
/// and the permutation is the winners on top, in order; the winners that were below the top take
/// the places of the top rows that did not win, in order of both; the others stay. With `rows`
/// (the panel is one chunk, all its rows are in the registers of the workgroup) the multipliers
/// of the other rows, which the elimination has left in their registers, are written too, each
/// to its place in the permutation: the panel is done.
fn calu_epilogue(wp: usize, fence: &str, rows: bool) -> String {
    let multipliers = if rows {
        format!(
            "    if (t < n && !done) {{
        var dest = t;
        if (t < w) {{ dest = slots[lbs[t]]; }}
{}    }}
",
            (0..wp).map(|c| format!("        if ({c}u < w) {{ f[dest * w + {c}u] = x{c}; }}\n")).collect::<String>()
        )
    } else {
        String::new()
    };
    format!(
        "    for (var e = t; e < w * w; e = e + R) {{
        let i = e / w;
        f[e] = lu[i * {wp}u + e - i * w];
    }}
    for (var i = w + t; i < p.m; i = i + R) {{ perm[i] = i; }}
    if (t < w) {{
        perm[t] = piv[t];
        var won = 0u;
        for (var k = 0u; k < w; k = k + 1u) {{ if (piv[k] == t) {{ won = 1u; }} }}
        wins[t] = won;
    }}
    workgroupBarrier();
    if (t < w) {{
        var lost_before = 0u;
        var outside_before = 0u;
        for (var k = 0u; k < t; k = k + 1u) {{
            lost_before = lost_before + 1u - wins[k];
            outside_before = outside_before + select(0u, 1u, piv[k] >= w);
        }}
        if (wins[t] == 0u) {{ lost[lost_before] = t; lbs[t] = lost_before; }}
        rank[t] = outside_before;
        if (piv[t] >= w) {{ slots[outside_before] = piv[t]; }}
    }}
    {fence}
    workgroupBarrier();
    if (t < w && piv[t] >= w) {{ perm[piv[t]] = lost[rank[t]]; }}
{multipliers}",
    )
}

/// The other rounds, one workgroup: groups of `R / w` winner sets (at most `R` rows) are factored
/// again from the original rows until one group is left, whose factors of the winners are the
/// top of `f` and whose order is the first `w` entries of `perm`. The winners of a group replace
/// the sets of the group in `win` (the sets of a group lie at or after the place of its winners).
fn calu_final_source(rows: usize, wp: usize, fence: &str, sg: bool) -> String {
    format!(
        "struct Params {{ m: u32, w: u32, g: u32, rc: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read_write> win: array<u32>;
@group(1) @binding(3) var<storage, read_write> f: array<f32>;
@group(1) @binding(4) var<storage, read_write> perm: array<u32>;
@group(1) @binding(5) var<uniform> p: Params;
fn ld(i: u32, c: u32) -> f32 {{ return a[luma_view_idx3(av, c, i, 0u)]; }}

{decl}
@compute @workgroup_size({rows})
fn main(@builtin(local_invocation_index) t: u32{lane_arg}) {{
    let w = p.w;
    let per_group = R / w;
    var count = p.g;
    loop {{
        let groups = (count + per_group - 1u) / per_group;
        for (var grp = 0u; grp < groups; grp = grp + 1u) {{
            let n = min(per_group, count - grp * per_group) * w;
            let base = grp * per_group * w;
            var panel_row = 0u;
            if (t < n) {{ panel_row = win[base + t]; }}
{load}{lu}            if (t < w) {{ win[grp * w + t] = piv[t]; }}
            {fence}
            workgroupBarrier();
        }}
        if (groups == 1u) {{ break; }}
        count = groups;
    }}
{epilogue}}}
",
        decl = calu_declarations(rows, wp, sg),
        lane_arg = if sg { ", @builtin(subgroup_invocation_id) lane: u32" } else { "" },
        epilogue = calu_epilogue(wp, fence, false),
        load = calu_load(wp, "panel_row"),
        lu = calu_lu(wp, calu_parts(rows), "panel_row", sg)
    )
}

/// Invocations of the kernel that finishes the panel.
const CALU_TAIL: usize = 64;

/// The last kernel of a tournament panel, over the rows below the top `w`: row `i` is row
/// `perm[i]` of the panel and its multipliers are `a[perm[i]] U11⁻¹` (a row of `w` unknowns by
/// substitution, on registers; `wp` columns, the padding is zero).
fn calu_tail_source(wp: usize) -> String {
    let q = CALU_P;
    let mut load = String::new();
    for e in 0..wp * wp / CALU_TAIL {
        load += &format!(
            "    {{
        let idx = t + {}u;
        let i = idx / {wp}u;
        let c = idx - i * {wp}u;
        let ok = i < w && c < w;
        let v = f[select(0u, i * w + c, ok)];
        u[i * {q}u + c] = select(0.0, v, ok);
    }}
",
            e * CALU_TAIL
        );
    }
    let mut body = String::new();
    let mut xload = String::new();
    for j in 0..wp {
        xload += &format!("    var acc{j} = 0.0;\n    if ({j}u < w) {{ acc{j} = ld(orig, {j}u); }}\n");
    }
    for k in 0..wp {
        body += &format!("    let y{k} = acc{k} * ru[{k}];\n");
        for j in k + 1..wp {
            body += &format!("    acc{j} = fma(-y{k}, u[{}], acc{j});\n", k * q + j);
        }
    }
    for j in 0..wp {
        body += &format!("    if ({j}u < w) {{ f[i * w + {j}u] = y{j}; }}\n");
    }
    format!(
        "struct Params {{ m: u32, w: u32, g: u32, rc: u32 }}
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(1) @binding(1) var<uniform> av: LumaView;
@group(1) @binding(2) var<storage, read_write> f: array<f32>;
@group(1) @binding(3) var<storage, read_write> perm: array<u32>;
@group(1) @binding(4) var<uniform> p: Params;
fn ld(i: u32, c: u32) -> f32 {{ return a[luma_view_idx3(av, c, i, 0u)]; }}

var<workgroup> u: array<f32, {usize}>;
var<workgroup> ru: array<f32, {wp}>;

@compute @workgroup_size({CALU_TAIL})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let w = p.w;
    // The loads that need nothing from workgroup memory are issued before the barrier.
    let i = w + wid.x * {CALU_TAIL}u + t;
    let orig = perm[min(i, p.m - 1u)];
{xload}{load}    if (t < {wp}u) {{
        let d = f[select(0u, t * w + t, t < w)];
        ru[t] = select(0.0, 1.0 / d, t < w && d != 0.0);
    }}
    workgroupBarrier();
    if (i >= p.m) {{ return; }}
{body}}}
",
        usize = wp * q,
        xload = xload
    )
}

/// Factors one `[w, m]` panel (`w <= 32`, `m >= w`) by a tournament: [`calu_chunks_source`]
/// picks `w` pivot rows in every chunk of rows, [`calu_final_source`] plays them off against one
/// another from the original rows and [`calu_tail_source`] permutes and computes the multipliers
/// of every row in parallel. Three launches whose workgroups all work, where the panel kernel has
/// one workgroup stream the whole panel for every column. `None` where the device cannot hold a
/// chunk.
fn calu_leaf(a: &Array<f32>, d: Dims, op: &'static str, loc: &'static Location<'static>) -> Option<Lu> {
    let ctx = a.context();
    let (rows, rows2) = calu_rows(ctx);
    if d.cols > CALU_W || rows < 64 || rows2 < 2 * d.cols || d.rows < d.cols {
        return None;
    }
    let perm_shape = Shape::from([d.rows, 1, 1]);
    let poisoned = |e: LumaError| Lu { lu: fail(ctx, a.shape(), e, op, loc), perm: Array::resolve(ctx, perm_shape, Err(crate::array::Failure::New(LumaError::InvalidShape("lu failed".into()))), op, loc) };
    // A panel that fits a workgroup is one chunk; a taller one is split in chunks of 256 rows
    // (a pivot takes less time in a workgroup of 8 subgroups than in one of 16).
    let rows = if d.rows <= rows { rows } else { rows.min(256) };
    let chunks = d.rows.div_ceil(rows);
    let chunk_rows = d.rows.div_ceil(chunks);
    // Every chunk has to hold the `w` rows it picks.
    if d.rows - (chunks - 1) * chunk_rows < d.cols {
        return None;
    }
    // Workgroups as small as the rows they factor: a pivot costs a pass of the invocations.
    let (rows, rows2) = (chunk_rows.next_power_of_two().clamp(64, rows), (chunks * d.cols).next_power_of_two().clamp(64, rows2));
    let params = CaluParams { m: d.rows as u32, w: d.cols as u32, g: chunks as u32, rc: chunk_rows as u32 };
    let (mut win, mut f, mut perm) = match (Array::<u32>::alloc(ctx, Shape::from([chunks * d.cols])), Array::<f32>::alloc(ctx, a.shape()), Array::<u32>::alloc(ctx, perm_shape)) {
        (Ok(w), Ok(f), Ok(p)) => (w, f, p),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return Some(poisoned(e)),
    };
    let fence = fence(ctx);
    let wp = d.cols.next_multiple_of(8);
    let sg = ctx.tuning().subgroup32;
    let tail_groups = (d.rows - d.cols).div_ceil(CALU_TAIL) as u32;
    let result = if chunks == 1 {
        let args: [KernelArg<'_>; 4] = [view(a), (&mut f).into(), (&mut perm).into(), uniform(&params)];
        launch(ctx, &format!("lu_calu_single/{rows}/{wp}/{}/{}", fence.len(), u8::from(sg)), "lu_calu_single", || calu_chunks_source(rows, wp, fence, sg, true), [1, 1, 1], &args)
    } else {
        launch(ctx, &format!("lu_calu_chunks/{rows}/{wp}/{}", u8::from(sg)), "lu_calu_chunks", || calu_chunks_source(rows, wp, fence, sg, false), [chunks as u32, 1, 1], &[view(a), (&mut win).into(), uniform(&params)])
            .and_then(|()| {
                let args: [KernelArg<'_>; 5] = [view(a), (&mut win).into(), (&mut f).into(), (&mut perm).into(), uniform(&params)];
                launch(ctx, &format!("lu_calu_final/{rows2}/{wp}/{}/{}", fence.len(), u8::from(sg)), "lu_calu_final", || calu_final_source(rows2, wp, fence, sg), [1, 1, 1], &args)
            })
            .and_then(|()| launch(ctx, &format!("lu_calu_tail/{wp}"), "lu_calu_tail", || calu_tail_source(wp), [tail_groups, 1, 1], &[view(a), (&mut f).into(), (&mut perm).into(), uniform(&params)]))
    };
    Some(match result {
        Ok(()) => Lu { lu: f, perm },
        Err(e) => poisoned(e),
    })
}

/// Rows of the triangle a substitution leaf solves, and the most rows of a leaf's `L`.
const LEAF: usize = 64;
/// The most right-hand sides for which a subgroup per column solves a leaf (see [`leaf_vec_source`]).
const VEC_LEAF_COLUMNS: usize = 16;

#[derive(encase::ShaderType)]
struct LeafParams {
    n: u32,
    k: u32,
}

/// The substitution leaf: `X = L⁻¹ B` for a lower triangular `L` of `n <= np` rows (`np` a multiple
/// of 8: the rows past `n` are absent) and `k` columns of `B`, an invocation per column with its
/// column in registers, the triangle in workgroup memory (every invocation reads the same entry),
/// unrolled: no loads or stores on the dependent chain of the substitution.
fn leaf_solve_source(np: usize, unit: bool) -> String {
    let mut body = String::new();
    for i in 0..np {
        body += &format!("    var x{i} = 0.0;\n    if ({i}u < n) {{ x{i} = ldb({i}u, c); }}\n");
    }
    for j in 0..np {
        if !unit {
            body += &format!("    x{j} = x{j} / ls[{}];\n", j * np + j);
        }
        for i in j + 1..np {
            body += &format!("    x{i} = fma(-ls[{}], x{j}, x{i});\n", i * np + j);
        }
    }
    for i in 0..np {
        body += &format!("    if ({i}u < n) {{ x[c + k * {i}u] = x{i}; }}\n");
    }
    let mut load = String::new();
    for e in 0..np * np / 64 {
        load += &format!(
            "    {{
        let idx = t + {}u;
        let i = idx / {np}u;
        let j = idx - i * {np}u;
        let ok = i < n && j < n && j <= i;
        let v = l[select(0u, luma_view_idx3(lv, j, i, 0u), ok)];
        ls[idx] = select(select(0.0, 1.0, i == j), v, ok);
    }}
",
            e * 64
        );
    }
    format!(
        "struct Params {{ n: u32, k: u32 }}
@group(1) @binding(0) var<storage, read> l: array<f32>;
@group(1) @binding(1) var<uniform> lv: LumaView;
@group(1) @binding(2) var<storage, read> b: array<f32>;
@group(1) @binding(3) var<uniform> bv: LumaView;
@group(1) @binding(4) var<storage, read_write> x: array<f32>;
@group(1) @binding(5) var<uniform> p: Params;
fn ldb(i: u32, c: u32) -> f32 {{ return b[luma_view_idx3(bv, c, i, 0u)]; }}

var<workgroup> ls: array<f32, {size}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p.n;
    let k = p.k;
{load}    workgroupBarrier();
    let c = wid.x * 64u + t;
    if (c >= k) {{ return; }}
{body}}}
",
        size = np * np
    )
}

/// The substitution leaf for a few right-hand sides on devices with subgroups of 32: a subgroup
/// per column of `B`, the triangle (up to 64 rows) in workgroup memory, a row to a lane (two to
/// the lanes of a subgroup), and the value of each row broadcast by a shuffle as it is known: a
/// step of the substitution is a shuffle and a fused multiply-add, where one invocation per column
/// (a single invocation for one right-hand side) takes a chain of 2000 of them.
fn leaf_vec_source(unit: bool) -> String {
    let mut body = String::new();
    for half in 0..2 {
        let (reg, rd) = if half == 0 { ("b0", "rd0") } else { ("b1", "rd1") };
        for j in 0..32 {
            let row = half * 32 + j;
            let scale = if unit { String::new() } else { format!(" * subgroupShuffle({rd}, {j}u)") };
            body += &format!("    {{\n        let xj = subgroupShuffle({reg}, {j}u){scale};\n        if (lane == {j}u && {row}u < n) {{ x[c + k * {row}u] = xj; }}\n");
            if half == 0 {
                body += &format!("        b0 = select(b0, fma(-ls[lane * 65u + {j}u], xj, b0), lane > {j}u);\n        b1 = fma(-ls[(lane + 32u) * 65u + {j}u], xj, b1);\n");
            } else {
                body += &format!("        b1 = select(b1, fma(-ls[(lane + 32u) * 65u + {row}u], xj, b1), lane > {j}u);\n");
            }
            body += "    }\n";
        }
    }
    let mut load = String::new();
    for e in 0..128 {
        load += &format!(
            "    {{
        let idx = lane + {}u;
        let i = idx / 64u;
        let j = idx - i * 64u;
        let ok = i < n && j < n && j <= i;
        let v = l[select(0u, luma_view_idx3(lv, j, i, 0u), ok)];
        ls[i * 65u + j] = select(0.0, v, ok);
    }}
",
            e * 32
        );
    }
    format!(
        "struct Params {{ n: u32, k: u32 }}
@group(1) @binding(0) var<storage, read> l: array<f32>;
@group(1) @binding(1) var<uniform> lv: LumaView;
@group(1) @binding(2) var<storage, read> b: array<f32>;
@group(1) @binding(3) var<uniform> bv: LumaView;
@group(1) @binding(4) var<storage, read_write> x: array<f32>;
@group(1) @binding(5) var<uniform> p: Params;
fn ldb(i: u32, c: u32) -> f32 {{ return b[luma_view_idx3(bv, c, i, 0u)]; }}

var<workgroup> ls: array<f32, 4160>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {{
    let n = p.n;
    let k = p.k;
    let c = wid.x;
    var b0 = 0.0;
    var b1 = 0.0;
    if (lane < n) {{ b0 = ldb(lane, c); }}
    if (lane + 32u < n) {{ b1 = ldb(lane + 32u, c); }}
    var rd0 = 1.0;
    var rd1 = 1.0;
    if ({unit} == 0u) {{
        if (lane < n) {{ rd0 = 1.0 / l[luma_view_idx3(lv, lane, lane, 0u)]; }}
        if (lane + 32u < n) {{ rd1 = 1.0 / l[luma_view_idx3(lv, lane + 32u, lane + 32u, 0u)]; }}
    }}
{load}    workgroupBarrier();
{body}}}
",
        unit = u8::from(unit)
    )
}

/// `L X = B` for the `[n, n]` lower triangle `l` (`unit`: with ones on the diagonal, not read) and
/// `[k, n]` right-hand sides `b`: leaves of [`LEAF`] rows by [`leaf_solve_source`], bigger triangles
/// split about in half with a [`matmul_f32`] product, as in [`super::tri`].
fn solve_lower(l: &Array<f32>, b: &Array<f32>, unit: bool, op: &'static str, loc: &'static Location<'static>) -> Array<f32> {
    let ctx = l.context();
    let [_, n, ..] = l.shape().dims();
    let [k, ..] = b.shape().dims();
    if n > LEAF {
        let h = super::tri::split(n).min(n - 1);
        let x1 = solve_lower(&block(l, 0..h, 0..h), &block(b, 0..h, 0..k), unit, op, loc).cache();
        let update = &block(b, h..n, 0..k) - &matmul_f32(&block(l, h..n, 0..h), &x1);
        let x2 = solve_lower(&block(l, h..n, h..n), &update, unit, op, loc);
        return concat(1, [&x1, &x2]);
    }
    let mut x = match Array::<f32>::alloc(ctx, b.shape()) {
        Ok(x) => x,
        Err(e) => return fail(ctx, b.shape(), e, op, loc),
    };
    let np = n.next_multiple_of(8);
    let params = LeafParams { n: n as u32, k: k as u32 };
    let args: [KernelArg<'_>; 4] = [view(l), view(b), (&mut x).into(), uniform(&params)];
    let result = if k <= VEC_LEAF_COLUMNS && ctx.tuning().subgroup32 {
        launch(ctx, &format!("lu_leaf_vec/{}", u8::from(unit)), "lu_leaf_vec", || leaf_vec_source(unit), [k as u32, 1, 1], &args)
    } else {
        launch(ctx, &format!("lu_leaf_solve/{np}/{}", u8::from(unit)), "lu_leaf_solve", || leaf_solve_source(np, unit), [k.div_ceil(64) as u32, 1, 1], &args)
    };
    match result {
        Ok(()) => x,
        Err(e) => fail(ctx, b.shape(), e, op, loc),
    }
}

/// The widest matrix (in columns) a leaf of `m` rows factors in one launch.
fn leaf_width(m: usize, batch: usize) -> usize {
    if m <= SMALL {
        SMALL
    } else if batch > 1 {
        PANEL
    } else {
        super::panel_width()
    }
}

/// Factors every `[w, m, batch]` matrix of `a` (`m >= w`) with one of the two kernels.
fn leaf(a: &Array<f32>, d: Dims, op: &'static str, loc: &'static Location<'static>) -> Lu {
    let ctx = a.context();
    if d.batch == 1
        && d.rows > SMALL
        && let Some(f) = calu_leaf(a, d, op, loc)
    {
        return f;
    }
    let perm_shape = Shape::from([d.rows, 1, d.batch]);
    let poisoned = |e: LumaError| Lu { lu: fail(ctx, a.shape(), e, op, loc), perm: Array::resolve(ctx, perm_shape, Err(crate::array::Failure::New(LumaError::InvalidShape("lu failed".into()))), op, loc) };
    let (mut f, mut perm) = match (Array::<f32>::alloc(ctx, a.shape()), Array::<u32>::alloc(ctx, perm_shape)) {
        (Ok(f), Ok(p)) => (f, p),
        (Err(e), _) | (_, Err(e)) => return poisoned(e),
    };
    let params = Params { m: d.rows as u32, w: d.cols as u32, batch: d.batch as u32 };
    let args: [KernelArg<'_>; 4] = [view(a), (&mut f).into(), (&mut perm).into(), uniform(&params)];
    let result = if d.rows <= SMALL {
        launch(ctx, "lu_small", "lu_small", small_source, per_matrix(ctx, d.batch), &args)
    } else {
        let threads = panel_threads(ctx, d.rows);
        launch(ctx, &format!("lu_panel/{threads}/{}", fence(ctx).len()), "lu_panel", || panel_source(fence(ctx), threads), per_matrix(ctx, d.batch), &args)
    };
    match result {
        Ok(()) => Lu { lu: f, perm },
        Err(e) => poisoned(e),
    }
}

/// The recursion over one `[w, m]` matrix, `m >= w`.
fn recurse(a: &Array<f32>, op: &'static str, loc: &'static Location<'static>) -> Lu {
    let [w, m, ..] = a.shape().dims();
    if w <= leaf_width(m, 1) {
        return leaf(a, Dims { rows: m, cols: w, batch: 1 }, op, loc);
    }
    let n1 = super::split_cols(w);
    let n2 = w - n1;
    let Lu { lu: f1, perm: p1 } = recurse(&block(a, 0..m, 0..n1), op, loc);
    let f1 = f1.cache();
    // The right columns in the order of the left panel's pivots.
    let right = gather_rows(&block(a, 0..m, n1..w), &p1, loc);
    let a12 = block(&right, 0..n1, 0..n2);
    let a22 = block(&right, n1..m, 0..n2);
    let u12 = solve_lower(&block(&f1, 0..n1, 0..n1), &a12, true, op, loc).cache();
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
        Ok(d) => {
            // Wide: factor the square left block, then U12 = L11⁻¹ (P A12).
            let left = factor(&a.slice(0, 0..d.rows), op, loc);
            let right = gather_rows(&a.slice(0, d.rows..d.cols), &left.perm, loc);
            let u12 = solve_lower(&left.lu, &right, true, op, loc);
            return Lu { lu: concat(0, [&left.lu, &u12]), perm: left.perm };
        }
        Err(e) => return Lu { lu: fail(ctx, a.shape(), e, op, loc), perm: Array::resolve(ctx, Shape::from(1), Err(crate::array::Failure::New(LumaError::InvalidShape("lu failed".into()))), op, loc) },
    };
    if d.cols <= leaf_width(d.rows, d.batch) {
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
    /// LU factorisation with partial pivoting of a matrix (`[n, m]`, tall, square or wide, or a batch
    /// `[n, m, batch]`): `P A = L U`, see [`Lu`]. Rows are swapped to bring the entry of largest
    /// magnitude of the column to the diagonal; a column with no nonzero entry (a singular
    /// matrix) leaves a zero on the diagonal of `U`, which [`Array::solve`] turns into `Inf` /
    /// `NaN` and [`Array::det`] into 0; nothing is checked. In a big single matrix the panels of
    /// more than 512 rows pick their pivot rows by a tournament between chunks of rows
    /// (communication-avoiding LU): the pivots are then not always the largest entries of their
    /// columns, and multipliers in `L` can exceed 1 (by a little; the factorisation is as
    /// accurate in practice). Matrices of up to 512 rows are pivoted exactly as above.
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
    /// The factors are poisoned if the array has a fourth axis.
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

    /// The upper triangular (trapezoidal, for a wide matrix) factor `U`: the first `min(m, n)` rows.
    pub fn upper(&self) -> Array<f32> {
        let [n, m, ..] = self.lu.shape().dims();
        let rows = self.lu.index(1);
        let cols = self.lu.index(0);
        select(&cols.ge(&rows), self.lu.clone(), 0.0).slice(1, 0..n.min(m))
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
    let k = rhs.shape().dims()[0];
    if f.lu.shape().dims()[2] == 1 && !super::trsv::applies(f.lu.context(), f.lu.shape().dims()[1], k, 1) {
        let y = solve_lower(&f.lu, &pb, true, op, loc);
        return solve_lower(&f.lu.flip(0).flip(1), &y.flip(1), false, op, loc).flip(1);
    }
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
